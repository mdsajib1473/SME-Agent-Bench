"""Load one run folder into a DataFrame, filling fields that older runs lack.

Every fallback used is recorded as a plain sentence in RunData.notes, so the
summary can list it under the limits of the run.
"""

import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.script_check import is_wrong_script

RESULTS_DIR = ROOT / "results"
DEFAULT_TASKS_PATH = ROOT / "tasks" / "tasks.jsonl"

ARCH_ORDER = ("react", "plan_execute", "supervisor")
ARCH_LABELS = {"react": "ReAct", "plan_execute": "Plan and Execute", "supervisor": "Supervisor"}
CATEGORY_ORDER = ("refund", "order_change", "complaint_routing", "quotation", "inquiry")
DIFFICULTY_ORDER = ("easy", "medium", "hard")
LANGUAGE_ORDER = ("en", "banglish")
LANGUAGE_LABELS = {"en": "English", "banglish": "Banglish"}
COUNT_FIELDS = ("llm_calls", "tool_calls", "malformed_tool_calls", "text_tool_calls", "prompt_tokens",
                "completion_tokens")


class EmptyRunError(Exception):
    """The run folder has no readable runs."""


@dataclass
class RunData:
    tag: str
    run_dir: Path
    rows: pd.DataFrame
    meta: dict
    tasks: dict
    traces: dict
    models: list
    archs: list
    seeds: list
    notes: list = field(default_factory=list)


def arch_label(arch):
    return ARCH_LABELS.get(arch, str(arch).replace("_", " ").title())


def category_label(category):
    return str(category).replace("_", " ").capitalize()


def language_label(language):
    return LANGUAGE_LABELS.get(language, str(language).capitalize())


def model_size_b(model):
    match = re.search(r"(\d+(?:\.\d+)?)b\b", str(model).lower().replace("-", " "))
    return float(match.group(1)) if match else None


def model_label(model):
    """'qwen2.5-7b-8k' becomes 'Qwen2.5 7B'; unknown names are returned unchanged."""
    size = model_size_b(model)
    if size is None:
        return str(model)
    family = str(model).split("-")[0]
    family = family[:1].upper() + family[1:]
    return f"{family} {size:g}B"


def run_id(row):
    return f"{row['model']}__{row['arch']}__{row['task_id']}__s{row['seed']}"


def text_sha256(path):
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def read_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def ordered(values, preferred):
    values = list(dict.fromkeys(values))
    known = [v for v in preferred if v in values]
    return known + sorted(v for v in values if v not in preferred)


def load_tasks(meta, notes):
    candidates = []
    if meta.get("tasks_path"):
        candidates.append(Path(meta["tasks_path"]))
    candidates.append(DEFAULT_TASKS_PATH)
    for path in candidates:
        if not path.exists():
            continue
        if path != candidates[0]:
            notes.append(f"The task file recorded in meta.json was not found; tasks were read from {path.name} instead.")
        if meta.get("tasks_sha256") and text_sha256(path) != meta["tasks_sha256"]:
            notes.append(
                f"The task file {path.name} differs from the one the run used (tasks_sha256 mismatch);"
                " instructions and gold actions are taken from traces where possible."
            )
        return {task["task_id"]: task for task in read_jsonl(path)}
    notes.append("No task file was found; instructions come from traces only and supervisor handoff checks are skipped.")
    return {}


def load_trace(run_dir, row):
    trace_file = row.get("trace_file")
    if not trace_file:
        return None
    path = run_dir / trace_file
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def final_reply(trace):
    return (trace or {}).get("final_reply") or ""


def dropped_from_trace(trace):
    """Responses with output tokens but no content and no tool call, as llm.py counts them."""
    count = 0
    for event in trace.get("trace", []):
        if event.get("type") != "llm_call":
            continue
        if "dropped" in event:
            count += bool(event["dropped"])
        elif (
            event.get("finish_reason") in ("stop", "length")
            and not (event.get("content") or "").strip()
            and not event.get("tool_calls")
            and (event.get("completion_tokens") or 0) > 0
        ):
            count += 1
    return count


def fill_from_traces(records, traces, column, derive, notes, how):
    missing = [r for r in records if column not in r]
    if not missing:
        return
    derived = 0
    for record in missing:
        trace = traces.get(record["run_id"])
        record[column] = derive(record, trace) if trace is not None else None
        derived += trace is not None
    notes.append(
        f"{column} is missing in {len(missing)} run(s); {how} for {derived} of them"
        + (f", left empty for {len(missing) - derived} without a trace." if derived < len(missing) else ".")
    )


def load_run(run_dir, tag=None):
    run_dir = Path(run_dir)
    tag = tag or run_dir.name
    runs_path = run_dir / "runs.jsonl"
    if not runs_path.exists():
        raise EmptyRunError(f"{runs_path} does not exist")
    records = read_jsonl(runs_path)
    if not records:
        raise EmptyRunError(f"{runs_path} has no readable runs")

    notes = []
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {}
        notes.append("meta.json is missing; idle power and hashes are unknown.")

    for record in records:
        record["run_id"] = run_id(record)
    unique = {}
    for record in records:
        unique[record["run_id"]] = record
    if len(unique) < len(records):
        notes.append(f"{len(records) - len(unique)} duplicate run line(s) found; the last line of each was kept.")
    records = list(unique.values())

    tasks = load_tasks(meta, notes)
    traces = {record["run_id"]: load_trace(run_dir, record) for record in records}
    without_trace = sum(trace is None for trace in traces.values())
    if without_trace:
        notes.append(f"{without_trace} of {len(records)} run(s) have no readable trace; trace based signals are empty for them.")

    for column in ("category", "difficulty", "language", "is_trap"):
        absent = [r for r in records if r.get(column) is None]
        for record in absent:
            record[column] = tasks.get(record["task_id"], {}).get(column)
        if absent:
            notes.append(f"{column} is missing in {len(absent)} run(s); taken from the task file.")

    fill_from_traces(records, traces, "empty_reply",
                     lambda r, t: not final_reply(t).strip(), notes, "derived from the final reply in the trace")
    stale = sum(bool(r.get("success")) and r.get("empty_reply") is True for r in records)
    if stale:
        notes.append(
            f"{stale} run(s) count as successes although the reply is empty; they predate the empty reply rule"
            " and keep their recorded score."
        )
    fill_from_traces(records, traces, "dropped_tool_calls",
                     lambda r, t: dropped_from_trace(t), notes,
                     "counted from responses with output tokens but no content and no tool call")
    fill_from_traces(records, traces, "wrong_script",
                     lambda r, t: is_wrong_script(r.get("language"), final_reply(t)), notes,
                     "derived from the final reply in the trace with eval/script_check.py")

    renamed = [r for r in records if "energy_counter_wh" not in r and "counter_energy_wh" in r]
    for record in renamed:
        record["energy_counter_wh"] = record["counter_energy_wh"]
    idle_w = meta.get("idle_power_w")
    derived = [r for r in records if "net_energy_counter_wh" not in r]
    for record in derived:
        counter = record.get("energy_counter_wh")
        record["net_energy_counter_wh"] = (
            counter - idle_w * record["wall_time_s"] / 3600.0
            if counter is not None and idle_w is not None and record.get("wall_time_s") is not None else None
        )
    if derived:
        notes.append(
            f"net_energy_counter_wh is missing in {len(derived)} run(s); derived from the counter energy minus"
            f" the single idle measurement in meta.json ({idle_w if idle_w is None else round(idle_w, 2)} W)"
            " times wall_time_s."
        )
    uncorrected = sum("energy_sampled_raw_wh" not in r for r in records)
    if uncorrected:
        notes.append(
            f"{uncorrected} run(s) predate the power lag correction; their net_energy_wh is the uncorrected"
            " sampled value, which reads low by a roughly constant amount per run (about 50 to 80 J per run in the calibration)."
        )
    no_settle = sum("settle_seconds" not in r for r in records)
    if no_settle:
        notes.append(
            f"{no_settle} run(s) were recorded without the settle wait after the model reset; their energy"
            " includes part of the reset aftermath (up to about 80 J per run)."
        )
    no_idle = sum("idle_w_used" not in r for r in records)
    if no_idle:
        notes.append(f"{no_idle} run(s) use a single idle measurement per run folder instead of the periodic one.")
    if not meta.get("harness_sha256") and not any(r.get("harness_sha256") for r in records):
        notes.append("harness_sha256 is not recorded for this run folder.")

    for column in COUNT_FIELDS + ("wall_time_s", "net_energy_wh", "delegations", "replans", "budget_exceeded",
                                  "policy_violation", "state_match", "output_match", "llm_timeout"):
        absent = sum(column not in r for r in records)
        if absent:
            for record in records:
                record.setdefault(column, None)
            notes.append(f"{column} is missing in {absent} run(s) and is left empty.")

    rows = pd.DataFrame(records)
    for column in ("success", "policy_violation", "budget_exceeded", "state_match", "output_match", "llm_timeout"):
        rows[column] = rows[column].map(lambda v: bool(v) if v is not None and v == v else False)
    for column in ("empty_reply", "wrong_script", "is_trap"):
        rows[column] = rows[column].astype("object")
    for column in COUNT_FIELDS + ("dropped_tool_calls", "wall_time_s", "net_energy_wh", "net_energy_counter_wh",
                                  "delegations", "replans"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    if rows["net_energy_wh"].isna().any():
        notes.append(f"net_energy_wh is empty in {int(rows['net_energy_wh'].isna().sum())} run(s) (no GPU reading).")

    models = list(dict.fromkeys(rows["model"]))
    sizes = [model_size_b(m) for m in models]
    if all(size is not None for size in sizes):
        models = [m for _, m in sorted(zip(sizes, models))]
    archs = ordered(rows["arch"], ARCH_ORDER)
    seeds = sorted(int(s) for s in rows["seed"].unique())
    return RunData(tag=tag, run_dir=run_dir, rows=rows, meta=meta, tasks=tasks, traces=traces,
                   models=models, archs=archs, seeds=seeds, notes=notes)


def bool_or_none(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return bool(value)
