"""Energy calibration: does the NVML energy counter carry a fixed offset per block?

Phases, all with the 7B model loaded and nothing else on the GPU:
  idle         sleep-only EnergyMeter blocks of 0.5, 2, 10 and 30 s, 10 repeats
               each, in seeded shuffled order
  after_reset  the same blocks, 5 repeats each, each one right after
               reset_model_state and the same steps run.py takes before a run
  real_run     ReAct on REF-E-01, seed 0, through run.run_once (reset included)

Idle power is measured exactly as run.py does: load the model, settle 5 s, then
30 s of EnergyMeter sampling with no request. The counter delta over that idle
block is recorded too, for comparison.

Raw rows go to results/calibration/energy_blocks.csv; the summary is printed.

With --settle only one phase runs after the idle measurement:
  settle       reset_model_state (unload, load, warm-up), then a wait of 0, 3, 5,
               8, 12 or 15 s, then a 10 s sleep-only EnergyMeter block, 8 repeats
               per wait in seeded shuffled order. A second sampler records power
               and the energy counter every 100 ms through wait and block, for the
               power at the end of the wait and the aftermath curve.
Output: energy_settle_blocks.csv, energy_settle_trace.csv, energy_settle_meta.json.

Usage (PowerShell):
    .venv\\Scripts\\python.exe scripts\\calibrate_energy.py
    .venv\\Scripts\\python.exe scripts\\calibrate_energy.py --settle
"""

import argparse
import csv
import json
import random
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run
from agents.harness import harness_sha256
from agents.llm import LLMClient, load_config, model_residency, reset_model_state
from agents.prompts import load_policy, policy_sha256, prompt_sha256
from agents.registry import build_agent
from env.shop import Shop
from eval import scorer
from telemetry.energy import EnergyMeter

MODEL = "qwen2.5-7b-8k"
TASK_ID = "REF-E-01"
BLOCK_SECONDS = (0.5, 2.0, 10.0, 30.0)
IDLE_REPEATS = 10
RESET_REPEATS = 5
REAL_RUNS = 5
GAP_S = 1.0
SHUFFLE_SEED = 0
OUT_DIR = ROOT / "results" / "calibration"
CSV_PATH = OUT_DIR / "energy_blocks.csv"
FIELDS = (
    "phase", "order", "block_s", "repeat", "task_id", "seed", "duration_s",
    "counter_wh", "sampled_wh", "idle_power_w", "counter_net_wh", "sampled_net_wh",
    "counter_minus_sampled_j", "samples", "reset_s", "success",
)

SETTLE_WAITS_S = (0.0, 3.0, 5.0, 8.0, 12.0, 15.0)
SETTLE_REPEATS = 8
SETTLE_BLOCK_S = 10.0
SETTLE_TAIL_S = 2.0
SETTLE_TARGET_J = 10.0
AFTERMATH_LENGTHS_S = (0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0)
TRACE_INTERVAL_S = 0.1
SETTLE_CSV_PATH = OUT_DIR / "energy_settle_blocks.csv"
SETTLE_TRACE_PATH = OUT_DIR / "energy_settle_trace.csv"
SETTLE_META_PATH = OUT_DIR / "energy_settle_meta.json"
SETTLE_FIELDS = (
    "phase", "order", "wait_s", "block_s", "repeat", "duration_s", "counter_wh", "sampled_wh",
    "idle_power_w", "counter_net_wh", "sampled_net_wh", "counter_minus_sampled_j", "samples",
    "reset_s", "wait_tail_counter_w", "wait_tail_reported_w",
)
TRACE_FIELDS = ("order", "wait_s", "repeat", "segment", "t_s", "reported_w", "counter_mj")


def block_row(phase, order, block_s, repeat, meter, idle_w, reset_s=None):
    counter_net = meter.net_counter_energy_wh(idle_w)
    sampled_net = meter.net_energy_wh(idle_w)
    diff = None
    if meter.counter_energy_wh is not None and meter.energy_wh is not None:
        diff = (meter.counter_energy_wh - meter.energy_wh) * 3600.0
    return {
        "phase": phase, "order": order, "block_s": block_s, "repeat": repeat,
        "task_id": None, "seed": None, "duration_s": meter.duration_s,
        "counter_wh": meter.counter_energy_wh, "sampled_wh": meter.energy_wh,
        "idle_power_w": idle_w, "counter_net_wh": counter_net, "sampled_net_wh": sampled_net,
        "counter_minus_sampled_j": diff, "samples": len(meter.samples),
        "reset_s": reset_s, "success": None,
    }


def sleep_block(seconds):
    with EnergyMeter() as meter:
        time.sleep(seconds)
    if not meter.available or meter.counter_energy_wh is None:
        raise RuntimeError("NVML or its energy counter is unavailable; calibration needs both")
    return meter


def shuffled(repeats):
    plan = [(seconds, repeat) for seconds in BLOCK_SECONDS for repeat in range(1, repeats + 1)]
    random.Random(SHUFFLE_SEED).shuffle(plan)
    return plan


class RowWriter:
    """Appends each row to the CSV as soon as it exists, so a crash keeps the data."""

    def __init__(self, path, fields=FIELDS):
        self.handle = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=fields)
        self.writer.writeheader()
        self.rows = []

    def add(self, row):
        self.rows.append(row)
        self.writer.writerow(row)
        self.handle.flush()

    def close(self):
        self.handle.close()


def gpu_processes():
    """(pid, used MiB) of every compute process on GPU 0, or None without NVML."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return [
                (proc.pid, (proc.usedGpuMemory or 0) / 1024**2)
                for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            ]
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


class TraceSampler:
    """Reported power and energy counter every 100 ms on its own NVML session."""

    def __init__(self):
        import pynvml

        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.rows = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self):
        self.rows.append((
            time.perf_counter(),
            self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0,
            self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle),
        ))

    def _loop(self):
        self._sample()
        while not self._stop.wait(TRACE_INTERVAL_S):
            self._sample()

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        self._thread.join()
        self.nvml.nvmlShutdown()
        return False

    def counter_j_at(self, t):
        """Counter in joules at time t, linearly interpolated between samples."""
        rows = self.rows
        for (t0, _, e0), (t1, _, e1) in zip(rows, rows[1:]):
            if t0 <= t <= t1:
                return (e0 + (e1 - e0) * (t - t0) / (t1 - t0)) / 1000.0
        return None

    def tail_powers(self, t_start, t_end):
        """Counter-derived and mean reported power between two times."""
        e0, e1 = self.counter_j_at(t_start), self.counter_j_at(t_end)
        counter_w = (e1 - e0) / (t_end - t_start) if e0 is not None and e1 is not None else None
        reported = [w for t, w, _ in self.rows if t_start <= t <= t_end]
        return counter_w, statistics.mean(reported) if reported else None


def settle_plan():
    plan = [(wait, repeat) for wait in SETTLE_WAITS_S for repeat in range(1, SETTLE_REPEATS + 1)]
    random.Random(SHUFFLE_SEED).shuffle(plan)
    return plan


def settle_phase(config, idle_w, counter_idle_w):
    out = RowWriter(SETTLE_CSV_PATH, SETTLE_FIELDS)
    trace_out = RowWriter(SETTLE_TRACE_PATH, TRACE_FIELDS)
    aftermath = {}
    plan = settle_plan()
    print(f"settle phase: {len(plan)} blocks of {SETTLE_BLOCK_S:.0f} s after reset and a wait")
    for order, (wait, repeat) in enumerate(plan, start=1):
        reset_s = reset_model_state(MODEL, config)
        model_residency(MODEL, config)
        with TraceSampler() as sampler:
            t_wait = time.perf_counter()
            time.sleep(wait)
            with Shop():
                t_block = time.perf_counter()
                meter = sleep_block(SETTLE_BLOCK_S)
            time.sleep(TRACE_INTERVAL_S * 2)
        tail_counter_w, tail_reported_w = (None, None)
        if wait >= SETTLE_TAIL_S:
            tail_counter_w, tail_reported_w = sampler.tail_powers(t_block - SETTLE_TAIL_S, t_block)
        row = block_row("settle", order, SETTLE_BLOCK_S, repeat, meter, idle_w, reset_s)
        del row["task_id"], row["seed"], row["success"]
        row.update(wait_s=wait, wait_tail_counter_w=tail_counter_w, wait_tail_reported_w=tail_reported_w)
        out.add(row)
        for t, reported_w, counter_mj in sampler.rows:
            trace_out.add({
                "order": order, "wait_s": wait, "repeat": repeat,
                "segment": "wait" if t < t_block else "block",
                "t_s": round(t - t_block, 4), "reported_w": reported_w, "counter_mj": counter_mj,
            })
        start_j = sampler.counter_j_at(t_block)
        aftermath.setdefault(wait, []).append({
            length: (sampler.counter_j_at(t_block + length) - start_j) - idle_w * length
            for length in AFTERMATH_LENGTHS_S
            if start_j is not None and sampler.counter_j_at(t_block + length) is not None
        })
        print(f"  {order:>2}/{len(plan)} wait {wait:>4.0f} s: counter net"
              f" {joules(row['counter_net_wh']):6.1f} J, waited {t_block - t_wait:5.2f} s")
    out.close()
    trace_out.close()
    rows = out.rows

    SETTLE_META_PATH.write_text(json.dumps({
        "model": MODEL, "idle_power_w": idle_w, "idle_counter_power_w": counter_idle_w,
        "idle_seconds": run.IDLE_SECONDS, "waits_s": SETTLE_WAITS_S, "repeats": SETTLE_REPEATS,
        "block_s": SETTLE_BLOCK_S, "tail_s": SETTLE_TAIL_S, "target_j": SETTLE_TARGET_J,
        "shuffle_seed": SHUFFLE_SEED, "trace_interval_s": TRACE_INTERVAL_S,
        "aftermath_net_counter_j": {
            str(wait): {
                str(length): [run_curve.get(length) for run_curve in curves]
                for length in AFTERMATH_LENGTHS_S
            }
            for wait, curves in sorted(aftermath.items())
        },
    }, indent=2), encoding="utf-8")
    print(f"raw data written to {SETTLE_CSV_PATH.relative_to(ROOT)} and {SETTLE_TRACE_PATH.relative_to(ROOT)}")
    print()

    print(f"10 s sleep-only block after reset and wait; idle {idle_w:.3f} W; joules, mean +/- sd")
    print("wait_s  n   counter net          sampled net          tail counter W   tail reported W")
    recommended = None
    for wait in SETTLE_WAITS_S:
        mine = [r for r in rows if r["wait_s"] == wait]
        counter_net = [joules(r["counter_net_wh"]) for r in mine]
        print(
            f"{wait:<7.0f} {len(mine):<3}"
            f" {stats(counter_net):<20}"
            f" {stats(joules(r['sampled_net_wh']) for r in mine):<20}"
            f" {stats(r['wait_tail_counter_w'] for r in mine):<16}"
            f" {stats(r['wait_tail_reported_w'] for r in mine)}"
        )
        if recommended is None and statistics.mean(counter_net) < SETTLE_TARGET_J:
            recommended = wait
    print()
    if recommended is not None:
        print(f"recommended wait: {recommended:.0f} s (first wait with mean counter net under {SETTLE_TARGET_J:.0f} J)")
    else:
        print(f"no wait up to {max(SETTLE_WAITS_S):.0f} s brings the mean counter net under {SETTLE_TARGET_J:.0f} J")
    print()
    print("aftermath curve: counter net energy (J) from block start, by wait")
    print("wait_s  " + "  ".join(f"{length:>12}" for length in AFTERMATH_LENGTHS_S))
    for wait, curves in sorted(aftermath.items()):
        cells = [stats(curve.get(length) for curve in curves) for length in AFTERMATH_LENGTHS_S]
        print(f"{wait:<7.0f} " + "  ".join(f"{cell:>12}" for cell in cells))
    return 0


def joules(value_wh):
    return None if value_wh is None else value_wh * 3600.0


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return "n/a"
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.mean(values):8.1f} +/- {sd:5.1f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--settle", action="store_true", help="run only the settle phase")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    config = load_config()

    outputs = (SETTLE_CSV_PATH, SETTLE_TRACE_PATH, SETTLE_META_PATH) if args.settle else (CSV_PATH,)
    for path in outputs:
        if path.exists():
            print(f"STOP: {path.relative_to(ROOT)} already exists; move it before rerunning.")
            return 1
    problem = run.check_ollama([MODEL], config)
    if problem:
        print(f"STOP: {problem}")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    load_s = run.prepare_model(MODEL, config)
    time.sleep(run.IDLE_SETTLE_S)
    print(f"loaded {MODEL} in {load_s:.1f} s; measuring idle power for {run.IDLE_SECONDS} s as run.py does...")
    # Same as telemetry.energy.measure_idle_power, keeping the meter to read the counter too.
    with EnergyMeter() as idle_meter:
        time.sleep(run.IDLE_SECONDS)
    idle_w = idle_meter.mean_power_w
    counter_idle_w = (
        idle_meter.counter_energy_wh * 3600.0 / idle_meter.duration_s
        if idle_meter.counter_energy_wh is not None else None
    )
    print(f"idle power (sampled mean, used for all net values): {idle_w:.3f} W;"
          f" counter over the same 30 s: {counter_idle_w if counter_idle_w is None else round(counter_idle_w, 3)} W")
    print(f"GPU compute processes (pid, MiB): {gpu_processes()}")
    if args.settle:
        return settle_phase(config, idle_w, counter_idle_w)

    out = RowWriter(CSV_PATH)
    rows = out.rows
    order = 0
    print(f"idle phase: {len(BLOCK_SECONDS) * IDLE_REPEATS} sleep-only blocks")
    for seconds, repeat in shuffled(IDLE_REPEATS):
        time.sleep(GAP_S)
        order += 1
        out.add(block_row("idle", order, seconds, repeat, sleep_block(seconds), idle_w))

    print(f"after_reset phase: {len(BLOCK_SECONDS) * RESET_REPEATS} blocks, each right after a model reset")
    for seconds, repeat in shuffled(RESET_REPEATS):
        order += 1
        # Same steps as run.run_once before its measured block.
        reset_s = reset_model_state(MODEL, config)
        model_residency(MODEL, config)
        with Shop():
            meter = sleep_block(seconds)
        out.add(block_row("after_reset", order, seconds, repeat, meter, idle_w, reset_s))

    print(f"real_run phase: {REAL_RUNS} ReAct runs on {TASK_ID}")
    task = next(t for t in scorer.load_tasks() if t["task_id"] == TASK_ID)
    hashes = {
        "policy_sha256": policy_sha256(load_policy()),
        "prompt_sha256": prompt_sha256(load_policy()),
        "tasks_sha256": run.text_sha256(scorer.TASKS_PATH),
        "harness_sha256": harness_sha256(config),
    }
    traces_dir = OUT_DIR / "traces"
    traces_dir.mkdir(exist_ok=True)
    agent = build_agent("react")
    llm = LLMClient(MODEL, config=config)
    for repeat in range(1, REAL_RUNS + 1):
        order += 1
        row = run.run_once(agent, "react", task, 0, llm, config, idle_w, hashes, traces_dir)
        trace_path = traces_dir / row["trace_file"].split("/", 1)[1]
        renamed = trace_path.with_name(f"{trace_path.stem}__r{repeat}.json")
        trace_path.replace(renamed)
        diff = None
        if row["energy_counter_wh"] is not None and row["energy_wh"] is not None:
            diff = (row["energy_counter_wh"] - row["energy_wh"]) * 3600.0
        out.add({
            "phase": "real_run", "order": order, "block_s": None, "repeat": repeat,
            "task_id": TASK_ID, "seed": 0, "duration_s": row["wall_time_s"],
            "counter_wh": row["energy_counter_wh"], "sampled_wh": row["energy_wh"],
            "idle_power_w": idle_w, "counter_net_wh": row["net_energy_counter_wh"],
            "sampled_net_wh": row["net_energy_wh"], "counter_minus_sampled_j": diff,
            "samples": None, "reset_s": row["reset_s"], "success": row["success"],
        })

    out.close()
    (OUT_DIR / "energy_calibration_meta.json").write_text(json.dumps({
        "model": MODEL, "idle_power_w": idle_w, "idle_counter_power_w": counter_idle_w,
        "idle_seconds": run.IDLE_SECONDS, "block_seconds": BLOCK_SECONDS,
        "idle_repeats": IDLE_REPEATS, "reset_repeats": RESET_REPEATS, "real_runs": REAL_RUNS,
        "gap_s": GAP_S, "shuffle_seed": SHUFFLE_SEED, "hashes": hashes,
    }, indent=2), encoding="utf-8")
    print(f"raw data written to {CSV_PATH.relative_to(ROOT)}")
    print()

    print("all values in joules, mean +/- standard deviation")
    print("phase         block_s  n   counter net          sampled net          counter minus sampled")
    for phase in ("idle", "after_reset"):
        for seconds in BLOCK_SECONDS:
            mine = [r for r in rows if r["phase"] == phase and r["block_s"] == seconds]
            print(
                f"{phase:<13} {seconds:<8} {len(mine):<3}"
                f" {stats(joules(r['counter_net_wh']) for r in mine):<20}"
                f" {stats(joules(r['sampled_net_wh']) for r in mine):<20}"
                f" {stats(r['counter_minus_sampled_j'] for r in mine)}"
            )
    print()
    print("real runs, ReAct 7B on REF-E-01, seed 0 (joules)")
    print("repeat  duration_s  counter net  sampled net  difference")
    for r in [r for r in rows if r["phase"] == "real_run"]:
        print(
            f"{r['repeat']:<7} {r['duration_s']:<11.2f} {joules(r['counter_net_wh']):<12.1f}"
            f" {joules(r['sampled_net_wh']):<12.1f} {r['counter_minus_sampled_j']:.1f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
