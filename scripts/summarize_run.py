"""Summarize one run directory and project the time of a full run.

Reads results/<tag>/runs.jsonl and meta.json and prints success, policy
violations, timing and wrong-script replies (eval/script_check.py; read from
the wrong_script field, or from the trace for older runs) per model
and architecture, energy per model and architecture (net sampled energy, lag
corrected, as the main figure, net counter energy as the cross-check), reset
time and malformed tool call rate per model, GPU placement per model, and an
estimate for a full run from the measured per-run totals (reset included),
with the settle wait and the idle re-measurements of the current config added
where the measured runs predate them.

Usage (PowerShell):
    .venv\\Scripts\\python.exe scripts\\summarize_run.py --tag pilot
    .venv\\Scripts\\python.exe scripts\\summarize_run.py --tag pilot --full-tasks 80 --full-seeds 3
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run
from agents.llm import load_config
from eval.script_check import is_wrong_script, non_latin_scripts
from telemetry.energy import LAG_TAIL_S

RESULTS_DIR = ROOT / "results"
GPU_HOURS_LIMIT = 30


def attach_reply_scripts(out_dir, rows):
    """Set row['reply_scripts'] from the trace (None when missing) and fill
    row['wrong_script'] for runs recorded before the field existed."""
    for row in rows:
        path = out_dir / row["trace_file"]
        if not path.exists():
            row["reply_scripts"] = None
            row.setdefault("wrong_script", None)
            continue
        reply = json.loads(path.read_text(encoding="utf-8")).get("final_reply") or ""
        row["reply_scripts"] = sorted(non_latin_scripts(reply))
        if row.get("wrong_script") is None:
            row["wrong_script"] = is_wrong_script(row["language"], reply)


def wrong_script(row):
    return bool(row.get("wrong_script"))


def counter_energy(row, idle_w):
    """(energy_counter_wh, net_energy_counter_wh, derived) for one run.

    Runs recorded before the net counter field existed store the counter as
    counter_energy_wh; their net value is derived with wall_time_s as the
    duration, which is marked so it is not mistaken for a measured value.
    """
    if "net_energy_counter_wh" in row:
        return row.get("energy_counter_wh"), row["net_energy_counter_wh"], False
    counter = row.get("counter_energy_wh")
    if counter is None or idle_w is None:
        return counter, None, False
    return counter, counter - idle_w * row["wall_time_s"] / 3600.0, True


def print_energy(rows, idle_w):
    print("energy per run, mean Wh. main: net_sampled (NVML power, lag corrected, minus idle);"
          " cross-check: net_counter (NVML counter minus idle)")
    print("model            arch          net_sampled  net_counter  sampled  raw_sampled  counter  counter/sampled  no_counter")
    derived = 0
    uncorrected = sum("energy_sampled_raw_wh" not in row for row in rows)
    for (model, arch), mine in group(rows, "model", "arch").items():
        values = [counter_energy(row, idle_w) for row in mine]
        derived += sum(flag for _, _, flag in values)
        counters = [c for c, _, _ in values]
        sampled = [row["energy_wh"] for row in mine]
        ratios = [c / s for c, s in zip(counters, sampled) if c is not None and s]
        print(
            f"{model:<16} {arch:<13}"
            f" {fmt(mean(row['net_energy_wh'] for row in mine), '.4f'):<12}"
            f" {fmt(mean(n for _, n, _ in values), '.4f'):<12}"
            f" {fmt(mean(sampled), '.4f'):<8}"
            f" {fmt(mean(row.get('energy_sampled_raw_wh') for row in mine), '.4f'):<12}"
            f" {fmt(mean(counters), '.4f'):<8}"
            f" {fmt(mean(ratios), '.3f'):<16}"
            f" {sum(c is None for c in counters)}"
        )
    if uncorrected:
        print(f"  {uncorrected} run(s) predate the lag correction; their sampled values are uncorrected")
    if derived:
        print(f"  {derived} run(s) predate net_energy_counter_wh; their net counter value uses wall_time_s as duration")
    print()


def full_run_estimate(rows, models, archs, per_combo, settle_s, tail_s=LAG_TAIL_S,
                      idle_s=run.IDLE_SETTLE_S + run.IDLE_SECONDS, idle_every=run.IDLE_EVERY_RUNS):
    """Projected seconds for a full run, from the measured mean run_total_s per model and arch.

    Runs recorded before the settle wait existed get settle_s plus the meter tail
    added. Idle is measured at the start of every model and arch block and then
    every idle_every runs, each measurement taking idle_s.
    """
    per_arch, added = {}, {}
    for model in models:
        for arch in archs:
            mine = [row for row in rows if row["model"] == model and row["arch"] == arch]
            extra = mean(0.0 if "settle_seconds" in row else settle_s + tail_s for row in mine) or 0.0
            per_arch[(model, arch)] = ((mean(row["run_total_s"] for row in mine) or 0.0) + extra) * per_combo
            added[(model, arch)] = extra * per_combo
    idle_count = math.ceil(per_combo / idle_every) * len(models) * len(archs)
    return {
        "per_arch_s": per_arch,
        "added_settle_s": added,
        "idle_count": idle_count,
        "idle_total_s": idle_count * idle_s,
        "total_s": sum(per_arch.values()) + idle_count * idle_s,
    }


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def fmt(value, spec=".2f"):
    return "n/a" if value is None else format(value, spec)


def count(rows, key):
    """Sum of a counter or flag; None when the run directory predates the field."""
    if any(key not in row for row in rows):
        return None
    return sum(int(row[key] or 0) for row in rows)


def load_rows(tag):
    path = RESULTS_DIR / tag / "runs.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compare(old_tag, new_tag, new_rows):
    old_rows = load_rows(old_tag)
    rescored = 0
    for row in old_rows:
        # Older runs predate the empty-reply rule; apply it from the trace so
        # both sides are scored the same way.
        if "empty_reply" not in row:
            trace = json.loads((RESULTS_DIR / old_tag / row["trace_file"]).read_text(encoding="utf-8"))
            row["empty_reply"] = not (trace.get("final_reply") or "").strip()
            if row["success"] and row["empty_reply"]:
                row["success"] = False
                rescored += 1
    old_groups = group(old_rows, "model", "arch")
    print(f"comparison: {old_tag} -> {new_tag}"
          + (f" ({rescored} {old_tag} run(s) rescored as failed under the empty-reply rule)" if rescored else ""))
    print("model            arch          success        violations   mean wall_s      tasks newly passed / newly failed")
    for (model, arch), mine in group(new_rows, "model", "arch").items():
        before = old_groups.get((model, arch), [])
        old_pass = {row["task_id"] for row in before if row["success"]}
        new_pass = {row["task_id"] for row in mine if row["success"]}
        print(
            f"{model:<16} {arch:<13} {len(old_pass):>2} -> {len(new_pass):<9}"
            f" {sum(bool(r['policy_violation']) for r in before):>2} -> {sum(bool(r['policy_violation']) for r in mine):<7}"
            f" {fmt(mean(r['wall_time_s'] for r in before), '.1f'):>5} -> {fmt(mean(r['wall_time_s'] for r in mine), '.1f'):<7}"
            f" +{sorted(new_pass - old_pass)} -{sorted(old_pass - new_pass)}"
        )
    print(
        f"total success {sum(r['success'] for r in old_rows)}/{len(old_rows)} ->"
        f" {sum(r['success'] for r in new_rows)}/{len(new_rows)}"
    )
    print()


def group(rows, *keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--full-tasks", type=int, default=80)
    parser.add_argument("--full-seeds", type=int, default=3)
    parser.add_argument("--compare", help="another tag to compare success, violations and time with")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    out_dir = RESULTS_DIR / args.tag
    rows = [
        json.loads(line)
        for line in (out_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    models = list(dict.fromkeys(row["model"] for row in rows))
    archs = list(dict.fromkeys(row["arch"] for row in rows))
    print(f"results/{args.tag}: {len(rows)} runs, models {models}, archs {archs}")
    print(f"idle power {fmt(meta.get('idle_power_w'), '.2f')} W, ollama {meta.get('ollama_version')},"
          f" gpu {(meta.get('gpu') or {}).get('name')}")
    print(f"harness_sha256 {meta.get('harness_sha256') or 'not recorded'},"
          f" prompt_sha256 {meta.get('prompt_sha256')}")
    print()

    attach_reply_scripts(out_dir, rows)
    print("model            arch          runs  success  violations  dropped  empty  wall_s  run_total_s  timeouts  budget  errors  wrong_script")
    for (model, arch), mine in group(rows, "model", "arch").items():
        successes = sum(row["success"] for row in mine)
        missing = sum(row["wrong_script"] is None for row in mine)
        print(
            f"{model:<16} {arch:<13} {len(mine):<5} {successes:>2}/{len(mine):<5}"
            f" {sum(bool(row['policy_violation']) for row in mine):<11}"
            f" {fmt(count(mine, 'dropped_tool_calls'), 'd'):<8}"
            f" {fmt(count(mine, 'empty_reply'), 'd'):<6}"
            f" {fmt(mean(row['wall_time_s'] for row in mine), '.1f'):<7}"
            f" {fmt(mean(row['run_total_s'] for row in mine), '.1f'):<12}"
            f" {sum(row['llm_timeout'] for row in mine):<9}"
            f" {sum(row['budget_exceeded'] for row in mine):<7}"
            f" {sum(row['stop_reason'] == 'exception' for row in mine):<7}"
            f" {sum(wrong_script(row) for row in mine)}"
            + (f" ({missing} trace(s) missing)" if missing else "")
        )
    offenders = [row for row in rows if wrong_script(row)]
    if offenders:
        print("wrong-script replies:")
        for row in offenders:
            print(
                f"  {row['model']:<16} {row['arch']:<13} {row['task_id']:<10}"
                f" {row['language']:<9} {', '.join(row['reply_scripts'] or ['trace missing'])}"
            )
    print()
    print_energy(rows, meta.get("idle_power_w"))
    if args.compare:
        compare(args.compare, args.tag, rows)

    print("model            reset_s  malformed/attempted  rate    runs_with_malformed  text_calls  peak_vram_mib  min_gpu_share  placement")
    for (model,), mine in group(rows, "model").items():
        malformed = sum(row["malformed_tool_calls"] for row in mine)
        attempted = sum(
            row["tool_calls"] + row["malformed_tool_calls"] + (row["delegations"] or 0) for row in mine
        )
        shares = [row["gpu_fraction"] for row in mine if row["gpu_fraction"] is not None]
        residency = (meta.get("models", {}).get(model) or {}).get("residency") or {}
        print(
            f"{model:<16} {fmt(mean(row['reset_s'] for row in mine), '.2f'):<8}"
            f" {malformed:>4}/{attempted:<14} {fmt(malformed / attempted if attempted else None, '.3f'):<7}"
            f" {sum(row['malformed_tool_calls'] > 0 for row in mine):>3}/{len(mine):<16}"
            f" {sum(row['text_tool_calls'] for row in mine):<11}"
            f" {fmt(max((row['peak_vram_mib'] or 0) for row in mine), '.0f'):<14}"
            f" {fmt(min(shares) if shares else None, '.3f'):<14} {residency.get('processor', 'unknown')}"
        )
        if shares and min(shares) < 1:
            print(f"  WARNING: {model} was partly on CPU in {sum(s < 1 for s in shares)} of {len(mine)} runs")
    print()

    per_combo = args.full_tasks * args.full_seeds
    total_runs = per_combo * len(models) * len(archs)
    settle_s = load_config()["settle_seconds"]
    estimate = full_run_estimate(rows, models, archs, per_combo, settle_s)
    print(f"estimate for {args.full_tasks} tasks x {args.full_seeds} seeds x {len(models)} models x"
          f" {len(archs)} archs = {total_runs} runs (from mean run_total_s, reset included;"
          f" settle {settle_s} s and meter tail {LAG_TAIL_S} s added for runs recorded without them)")
    for model in models:
        model_s = sum(estimate["per_arch_s"][(model, arch)] for arch in archs)
        for arch in archs:
            print(f"  {model:<16} {arch:<13} {estimate['per_arch_s'][(model, arch)] / 3600:6.1f} h")
        mine = [row for row in rows if row["model"] == model]
        reset_share = (mean(row["reset_s"] for row in mine) or 0) * per_combo * len(archs)
        added = sum(estimate["added_settle_s"][(model, arch)] for arch in archs)
        print(f"  {model:<16} {'all':<13} {model_s / 3600:6.1f} h  (of which reset {reset_share / 3600:.1f} h,"
              f" added settle and tail {added / 3600:.1f} h)")
    print(f"  idle measurements: {estimate['idle_count']} x {run.IDLE_SETTLE_S + run.IDLE_SECONDS} s ="
          f" {estimate['idle_total_s'] / 3600:.2f} h")
    print(f"  total {estimate['total_s'] / 3600:.1f} h of GPU time")
    if estimate["total_s"] / 3600 > GPU_HOURS_LIMIT:
        print(f"  over the {GPU_HOURS_LIMIT} h target")
    return 0


if __name__ == "__main__":
    sys.exit(main())
