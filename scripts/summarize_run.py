"""Summarize one run directory and project the time of a full run.

Reads results/<tag>/runs.jsonl and meta.json and prints success, policy
violations and timing per model and architecture, reset time and malformed tool
call rate per model, GPU placement per model, and an estimate for a full run
from the measured per-run totals (reset included).

Usage (PowerShell):
    .venv\\Scripts\\python.exe scripts\\summarize_run.py --tag pilot
    .venv\\Scripts\\python.exe scripts\\summarize_run.py --tag pilot --full-tasks 80 --full-seeds 3
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
GPU_HOURS_LIMIT = 30


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
    print()

    print("model            arch          runs  success  violations  dropped  empty  wall_s  run_total_s  timeouts  budget  errors")
    for (model, arch), mine in group(rows, "model", "arch").items():
        successes = sum(row["success"] for row in mine)
        print(
            f"{model:<16} {arch:<13} {len(mine):<5} {successes:>2}/{len(mine):<5}"
            f" {sum(bool(row['policy_violation']) for row in mine):<11}"
            f" {fmt(count(mine, 'dropped_tool_calls'), 'd'):<8}"
            f" {fmt(count(mine, 'empty_reply'), 'd'):<6}"
            f" {fmt(mean(row['wall_time_s'] for row in mine), '.1f'):<7}"
            f" {fmt(mean(row['run_total_s'] for row in mine), '.1f'):<12}"
            f" {sum(row['llm_timeout'] for row in mine):<9}"
            f" {sum(row['budget_exceeded'] for row in mine):<7}"
            f" {sum(row['stop_reason'] == 'exception' for row in mine)}"
        )
    print()
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
    print(f"estimate for {args.full_tasks} tasks x {args.full_seeds} seeds x {len(models)} models x"
          f" {len(archs)} archs = {total_runs} runs (from mean run_total_s, reset included)")
    grand = 0.0
    for (model,), mine in group(rows, "model").items():
        model_s = 0.0
        for arch in archs:
            arch_rows = [row for row in mine if row["arch"] == arch]
            seconds = (mean(row["run_total_s"] for row in arch_rows) or 0) * per_combo
            model_s += seconds
            print(f"  {model:<16} {arch:<13} {seconds / 3600:6.1f} h")
        reset_share = (mean(row["reset_s"] for row in mine) or 0) * per_combo * len(archs)
        print(f"  {model:<16} {'all':<13} {model_s / 3600:6.1f} h  (of which reset {reset_share / 3600:.1f} h)")
        grand += model_s
    print(f"  total {grand / 3600:.1f} h of GPU time")
    if grand / 3600 > GPU_HOURS_LIMIT:
        print(f"  over the {GPU_HOURS_LIMIT} h target")
    return 0


if __name__ == "__main__":
    sys.exit(main())
