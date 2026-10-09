"""Lag between NVML reported power and power derived from the energy counter.

With the 7B model loaded, each repeat samples GPU 0 every 100 ms for 3 s of
idle, one long generation (300 tokens, seeded), and 4 s of idle after it. Each
sample holds the reported power (nvmlDeviceGetPowerUsage), the instant power
field (NVML_FI_DEV_POWER_INSTANT) and the cumulative energy counter. Counter
power is the counter difference between consecutive samples over their time
difference.

Analysis per repeat, on a 10 ms grid:
  lag      cross-correlation peak of reported power against counter power
           (positive means reported power trails the counter)
  window   width of the trailing moving average of counter power that best
           matches reported power (least squares)
  gap      counter energy minus trapezoid energy of reported power over the
           request window, i.e. what EnergyMeter would see for this block

Raw samples go to results/calibration/power_lag_samples.csv and the per-repeat
results to results/calibration/power_lag_summary.json.

With --real the same sampler runs through real ReAct runs on REF-E-01, seed 0,
through run.run_once (reset included), and records the EnergyMeter window. For
each run it reports the EnergyMeter gap (counter minus sampled) and the gap left
when the reported-power window is shifted later by 0 to 1.5 s: if the lag causes
the gap, a shift near the lag removes it. Output goes to power_lag_real_*.

Usage (PowerShell):
    .venv\\Scripts\\python.exe scripts\\measure_power_lag.py
    .venv\\Scripts\\python.exe scripts\\measure_power_lag.py --real
"""

import argparse
import csv
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run
from agents.harness import harness_sha256
from agents.llm import LLMClient, load_config, native_base, ollama_post
from agents.prompts import load_policy, policy_sha256, prompt_sha256
from agents.registry import build_agent
from eval import scorer
from telemetry.energy import EnergyMeter

MODEL = "qwen2.5-7b-8k"
REPEATS = 3
PRE_S = 3.0
POST_S = 4.0
INTERVAL_S = 0.1
GRID_S = 0.01
MAX_LAG_S = 2.0
NUM_PREDICT = 300
PROMPT = (
    "Write a long and detailed essay about the history of the printing press, "
    "its inventors, and its effect on science, religion and politics in Europe."
)
OUT_DIR = ROOT / "results" / "calibration"
CSV_PATH = OUT_DIR / "power_lag_samples.csv"
JSON_PATH = OUT_DIR / "power_lag_summary.json"
REAL_CSV_PATH = OUT_DIR / "power_lag_real_samples.csv"
REAL_JSON_PATH = OUT_DIR / "power_lag_real_summary.json"
REAL_TRACES_DIR = OUT_DIR / "power_lag_real_traces"
REAL_TASK_ID = "REF-E-01"
REAL_RUNS = 3
SHIFTS_S = tuple(round(0.05 * k, 2) for k in range(31))


class Sampler:
    def __init__(self, nvml, handle):
        self.nvml = nvml
        self.handle = handle
        self.rows = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self):
        nvml, handle = self.nvml, self.handle
        t = time.perf_counter()
        reported = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        field = nvml.nvmlDeviceGetFieldValues(handle, [nvml.NVML_FI_DEV_POWER_INSTANT])[0]
        instant = field.value.uiVal / 1000.0 if field.nvmlReturn == 0 else None
        counter_mj = nvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        self.rows.append((t, reported, instant, counter_mj))

    def _loop(self):
        self._sample()
        while not self._stop.wait(INTERVAL_S):
            self._sample()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()


def counter_power(rows):
    """(midpoint time, watts) from consecutive counter readings."""
    out = []
    for (t0, _, _, e0), (t1, _, _, e1) in zip(rows, rows[1:]):
        out.append(((t0 + t1) / 2.0, (e1 - e0) / 1000.0 / (t1 - t0)))
    return out


def on_grid(times, values, grid):
    return np.interp(grid, np.asarray(times), np.asarray(values))


def xcorr_lag(reported, derived):
    """Lag in seconds that maximises correlation of reported(t) with derived(t - lag)."""
    a = reported - reported.mean()
    b = derived - derived.mean()
    max_k = int(round(MAX_LAG_S / GRID_S))
    best_k, best_r = 0, -2.0
    for k in range(-max_k, max_k + 1):
        if k >= 0:
            x, y = a[k:], b[: len(b) - k]
        else:
            x, y = a[: len(a) + k], b[-k:]
        r = float(np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y)))
        if r > best_r:
            best_k, best_r = k, r
    return best_k * GRID_S, best_r


def best_window(reported, derived):
    """Trailing moving-average width (s) of derived power that best fits reported power."""
    results = []
    for width_ms in range(0, 2001, 50):
        n = max(1, int(round(width_ms / 1000.0 / GRID_S)))
        kernel = np.ones(n) / n
        smoothed = np.convolve(derived, kernel)[: len(derived)]
        start = int(round(MAX_LAG_S / GRID_S))
        rmse = float(np.sqrt(np.mean((smoothed[start:] - reported[start:]) ** 2)))
        results.append((width_ms / 1000.0, rmse))
    return min(results, key=lambda item: item[1])


def window_energy(rows, t_start, t_end):
    """Counter joules and reported-power trapezoid joules between two times."""
    times = np.array([r[0] for r in rows])
    counter_j = np.array([r[3] for r in rows]) / 1000.0
    counter = float(np.interp(t_end, times, counter_j) - np.interp(t_start, times, counter_j))
    inside = [(r[0], r[1]) for r in rows if t_start <= r[0] <= t_end]
    sampled = sum((w0 + w1) / 2.0 * (t1 - t0) for (t0, w0), (t1, w1) in zip(inside, inside[1:]))
    return counter, sampled


class WindowMeter(EnergyMeter):
    """EnergyMeter that also keeps its perf_counter window, for run.run_once."""

    windows = []

    def __enter__(self):
        result = super().__enter__()
        self.window_start = time.perf_counter()
        return result

    def __exit__(self, exc_type, exc_value, traceback):
        window_end = time.perf_counter()
        result = super().__exit__(exc_type, exc_value, traceback)
        WindowMeter.windows.append((self.window_start, window_end, self))
        return result


def shifted_sampled_j(rows, t_start, t_end, shift):
    """Joules of held reported power over the window moved later by shift seconds."""
    times = np.array([r[0] for r in rows])
    watts = np.array([r[1] for r in rows])
    grid = np.arange(t_start + shift, t_end + shift, GRID_S)
    held = watts[np.searchsorted(times, grid, side="right") - 1]
    return float(held.sum() * GRID_S)


def real_runs(config, pynvml, handle):
    for path in (REAL_CSV_PATH, REAL_JSON_PATH):
        if path.exists():
            print(f"STOP: {path.relative_to(ROOT)} already exists; move it before rerunning.")
            return 1
    REAL_TRACES_DIR.mkdir(parents=True, exist_ok=True)
    task = next(t for t in scorer.load_tasks() if t["task_id"] == REAL_TASK_ID)
    hashes = {
        "policy_sha256": policy_sha256(load_policy()),
        "prompt_sha256": prompt_sha256(load_policy()),
        "tasks_sha256": run.text_sha256(scorer.TASKS_PATH),
        "harness_sha256": harness_sha256(config),
    }
    agent = build_agent("react")
    llm = LLMClient(MODEL, config=config)
    run.EnergyMeter = WindowMeter
    run.prepare_model(MODEL, config)
    time.sleep(run.IDLE_SETTLE_S)
    results = []
    with REAL_CSV_PATH.open("w", newline="", encoding="utf-8") as handle_csv:
        writer = csv.writer(handle_csv)
        writer.writerow(["repeat", "t_s", "reported_w", "instant_w", "counter_mj", "in_window"])
        for repeat in range(1, REAL_RUNS + 1):
            sampler = Sampler(pynvml, handle)
            sampler.start()
            time.sleep(1.0)
            row = run.run_once(agent, "react", task, 0, llm, config, None, hashes, REAL_TRACES_DIR)
            time.sleep(POST_S)
            sampler.stop()
            trace_path = REAL_TRACES_DIR / row["trace_file"].split("/", 1)[1]
            trace_path.replace(trace_path.with_name(f"{trace_path.stem}__r{repeat}.json"))
            t_start, t_end, meter = WindowMeter.windows[-1]
            rows = sampler.rows
            t0 = rows[0][0]
            for t, reported, instant, counter_mj in rows:
                writer.writerow([repeat, round(t - t0, 4), reported, instant, counter_mj,
                                 int(t_start <= t <= t_end)])

            derived = counter_power(rows)
            grid = np.arange(derived[0][0], derived[-1][0], GRID_S)
            derived_g = on_grid([d[0] for d in derived], [d[1] for d in derived], grid)
            reported_g = on_grid([r[0] for r in rows], [r[1] for r in rows], grid)
            lag_s, lag_r = xcorr_lag(reported_g, derived_g)
            times = np.array([r[0] for r in rows])
            counter_j = np.array([r[3] for r in rows]) / 1000.0
            window_counter_j = float(np.interp(t_end, times, counter_j) - np.interp(t_start, times, counter_j))
            shifted = {
                shift: window_counter_j - shifted_sampled_j(rows, t_start, t_end, shift)
                for shift in SHIFTS_S
            }
            zero_shift = min(shifted, key=lambda k: abs(shifted[k]))
            before = [d[1] for d in derived if t_start - 1.0 <= d[0] < t_start]
            last = [d[1] for d in derived if t_end - 1.0 <= d[0] <= t_end]
            result = {
                "repeat": repeat,
                "window_s": t_end - t_start,
                "meter_counter_j": meter.counter_energy_wh * 3600.0,
                "meter_sampled_j": meter.energy_wh * 3600.0,
                "meter_gap_j": (meter.counter_energy_wh - meter.energy_wh) * 3600.0,
                "lag_reported_vs_counter_ms": 1000.0 * lag_s,
                "xcorr_peak_r": lag_r,
                "counter_power_1s_before_window_w": statistics.mean(before) if before else None,
                "counter_power_last_1s_of_window_w": statistics.mean(last) if last else None,
                "gap_by_shift_j": {str(k): v for k, v in shifted.items()},
                "shift_with_smallest_gap_s": zero_shift,
                "whole_trace_counter_j": float(counter_j[-1] - counter_j[0]),
                "whole_trace_reported_j": float(np.trapezoid([r[1] for r in rows], times)),
            }
            results.append(result)
            print(
                f"run {repeat}: window {result['window_s']:.2f} s; EnergyMeter gap {result['meter_gap_j']:.1f} J;"
                f" lag {result['lag_reported_vs_counter_ms']:.0f} ms;"
                f" gap at shift 0 / 0.5 / 0.75 / 1.0 s: {shifted[0.0]:.1f} / {shifted[0.5]:.1f} /"
                f" {shifted[0.75]:.1f} / {shifted[1.0]:.1f} J; smallest gap at {zero_shift:.2f} s;"
                f" power 1 s before / last 1 s of window {result['counter_power_1s_before_window_w']:.1f} /"
                f" {result['counter_power_last_1s_of_window_w']:.1f} W"
            )
    REAL_JSON_PATH.write_text(json.dumps({
        "model": MODEL, "task_id": REAL_TASK_ID, "seed": 0, "interval_s": INTERVAL_S,
        "grid_s": GRID_S, "post_s": POST_S, "hashes": hashes, "runs": results,
    }, indent=2), encoding="utf-8")
    print(f"written {REAL_CSV_PATH.relative_to(ROOT)} and {REAL_JSON_PATH.relative_to(ROOT)}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--real", action="store_true", help="trace real ReAct runs instead")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    config = load_config()
    if args.real:
        problem = run.check_ollama([MODEL], config)
        if problem:
            print(f"STOP: {problem}")
            return 1
        import pynvml

        pynvml.nvmlInit()
        try:
            return real_runs(config, pynvml, pynvml.nvmlDeviceGetHandleByIndex(0))
        finally:
            pynvml.nvmlShutdown()
    for path in (CSV_PATH, JSON_PATH):
        if path.exists():
            print(f"STOP: {path.relative_to(ROOT)} already exists; move it before rerunning.")
            return 1
    problem = run.check_ollama([MODEL], config)
    if problem:
        print(f"STOP: {problem}")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    url = f"{native_base(config['ollama_base_url'])}/api/generate"
    run.prepare_model(MODEL, config)
    time.sleep(run.IDLE_SETTLE_S)

    results = []
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle_csv:
        writer = csv.writer(handle_csv)
        writer.writerow(["repeat", "t_s", "reported_w", "instant_w", "counter_mj", "in_request"])
        for repeat in range(1, REPEATS + 1):
            sampler = Sampler(pynvml, handle)
            sampler.start()
            time.sleep(PRE_S)
            t_req0 = time.perf_counter()
            reply = ollama_post(url, {
                "model": MODEL, "prompt": PROMPT, "stream": False,
                "options": {"num_predict": NUM_PREDICT, "temperature": 0, "seed": repeat},
            }, config["request_timeout_s"])
            t_req1 = time.perf_counter()
            time.sleep(POST_S)
            sampler.stop()
            rows = sampler.rows
            t0 = rows[0][0]
            for t, reported, instant, counter_mj in rows:
                writer.writerow([repeat, round(t - t0, 4), reported, instant, counter_mj,
                                 int(t_req0 <= t <= t_req1)])

            derived = counter_power(rows)
            grid = np.arange(derived[0][0], derived[-1][0], GRID_S)
            derived_g = on_grid([d[0] for d in derived], [d[1] for d in derived], grid)
            reported_g = on_grid([r[0] for r in rows], [r[1] for r in rows], grid)
            instant_g = on_grid([r[0] for r in rows], [r[2] for r in rows], grid)
            lag_s, lag_r = xcorr_lag(reported_g, derived_g)
            inst_lag_s, inst_lag_r = xcorr_lag(instant_g, derived_g)
            width_s, width_rmse = best_window(reported_g, derived_g)
            counter_j, sampled_j = window_energy(rows, t_req0, t_req1)
            zero_steps = sum(1 for a, b in zip(rows, rows[1:]) if b[3] == a[3])
            in_req = [d[1] for d in derived if t_req0 + 1.0 <= d[0] <= t_req1]
            before = [d[1] for d in derived if t_req0 - 1.0 <= d[0] < t_req0]
            result = {
                "repeat": repeat,
                "request_s": t_req1 - t_req0,
                "eval_count": reply.get("eval_count"),
                "samples": len(rows),
                "mean_interval_ms": 1000.0 * (rows[-1][0] - rows[0][0]) / (len(rows) - 1),
                "counter_unchanged_steps": zero_steps,
                "lag_reported_vs_counter_ms": 1000.0 * lag_s,
                "xcorr_peak_r": lag_r,
                "lag_instant_vs_counter_ms": 1000.0 * inst_lag_s,
                "xcorr_instant_peak_r": inst_lag_r,
                "best_trailing_window_ms": 1000.0 * width_s,
                "best_window_rmse_w": width_rmse,
                "counter_power_during_request_w": statistics.mean(in_req) if in_req else None,
                "counter_power_1s_before_request_w": statistics.mean(before) if before else None,
                "request_counter_j": counter_j,
                "request_sampled_j": sampled_j,
                "request_counter_minus_sampled_j": counter_j - sampled_j,
            }
            results.append(result)
            print(
                f"repeat {repeat}: {result['eval_count']} tokens in {result['request_s']:.2f} s;"
                f" lag {result['lag_reported_vs_counter_ms']:.0f} ms (r {lag_r:.3f});"
                f" instant lag {result['lag_instant_vs_counter_ms']:.0f} ms;"
                f" best trailing window {result['best_trailing_window_ms']:.0f} ms;"
                f" request gap {result['request_counter_minus_sampled_j']:.1f} J"
            )
            time.sleep(2.0)
    pynvml.nvmlShutdown()

    JSON_PATH.write_text(json.dumps({
        "model": MODEL, "interval_s": INTERVAL_S, "grid_s": GRID_S, "num_predict": NUM_PREDICT,
        "pre_s": PRE_S, "post_s": POST_S, "repeats": results,
    }, indent=2), encoding="utf-8")
    print(f"written {CSV_PATH.relative_to(ROOT)} and {JSON_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
