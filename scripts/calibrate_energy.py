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

Usage (PowerShell):
    .venv\\Scripts\\python.exe scripts\\calibrate_energy.py
"""

import argparse
import csv
import json
import random
import statistics
import sys
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

    def __init__(self, path):
        self.handle = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=FIELDS)
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
    parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    config = load_config()

    if CSV_PATH.exists():
        print(f"STOP: {CSV_PATH.relative_to(ROOT)} already exists; move it before rerunning.")
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
