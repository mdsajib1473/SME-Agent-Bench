"""Automatic failure signals for failed runs, and the stratified labeling sheet."""

import json
import re
from collections import Counter

import numpy as np
import pandas as pd

from agents.supervisor import SPECIALIST_TOOLS

LOOP_THRESHOLD = 3
SAMPLE_PER_CELL = 25
SAMPLE_SEED = 0
ORDER_ID = re.compile(r"\bORD-\d+\b", re.IGNORECASE)
PHONE = re.compile(r"(?<!\d)01\d{9}(?!\d)")
ERROR_STOPS = ("exception", "llm_error")
SIGNALS = (
    "budget_exceeded",
    "dropped_tool_calls",
    "malformed_tool_calls",
    "text_tool_calls",
    "loop",
    "policy_violation",
    "wrong_final_state",
    "missing_required_output",
    "empty_reply",
    "wrong_script",
    "run_error",
    "handoff_missing_tool",
    "handoff_missing_identifier",
)
TRACE_SIGNALS = ("loop", "handoff_missing_tool", "handoff_missing_identifier")
LABEL_COLUMNS = ("label_1", "label_2")
SHEET_COLUMNS = ("run_id", "arch", "model", "task_id", "instruction", "final_reply", "trace_summary",
                 "auto_signals", "label_1", "label_2", "notes")


def events(trace, kind):
    return [e for e in (trace or {}).get("trace", []) if e.get("type") == kind]


def customer_message(trace, task):
    for event in events(trace, "message"):
        if event.get("role") == "user":
            return event.get("content") or ""
    return (task or {}).get("instruction") or ""


def delegations(trace):
    return [e for e in events(trace, "delegation") if not e.get("refused")]


def identifiers(text):
    found = [m.group(0).upper() for m in ORDER_ID.finditer(text or "")]
    found += [m.group(0) for m in PHONE.finditer(text or "")]
    return list(dict.fromkeys(found))


def loop_detail(trace):
    calls = Counter(
        (e.get("name"), json.dumps(e.get("arguments"), sort_keys=True, ensure_ascii=False))
        for e in events(trace, "tool_call")
    )
    calls.update(
        (f"delegate to {e.get('specialist')}", e.get("instruction") or "") for e in delegations(trace)
    )
    repeated = [f"{name} x{n}" for (name, _), n in calls.items() if n >= LOOP_THRESHOLD]
    return ", ".join(repeated)


def handoff_tool_detail(trace, task):
    sent = [e.get("specialist") for e in delegations(trace)]
    if not sent or not task:
        return ""
    needed = list(dict.fromkeys(a["name"] for a in task.get("gold_actions", [])))
    missing = [
        tool for tool in needed
        if not any(tool in SPECIALIST_TOOLS.get(specialist, ()) for specialist in sent)
    ]
    if not missing:
        return ""
    return f"needs {', '.join(missing)}; delegated only to {', '.join(dict.fromkeys(sent))}"


def handoff_identifier_detail(trace, task):
    wanted = identifiers(customer_message(trace, task))
    problems = []
    for event in delegations(trace):
        text = (event.get("instruction") or "").upper()
        lacking = [i for i in wanted if i not in text]
        if lacking:
            problems.append(f"delegation {event.get('index')} to {event.get('specialist')} lacks {', '.join(lacking)}")
    return "; ".join(problems)


def score_detail(trace, key):
    value = ((trace or {}).get("score") or {}).get(key)
    if not value:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)[:200]


def run_signals(row, trace, task):
    """signal -> (fired, detail); fired is None when the signal needs a trace that is missing."""
    has_trace = trace is not None
    supervisor = row["arch"] == "supervisor"
    out = {
        "budget_exceeded": (bool(row["budget_exceeded"]), ""),
        "dropped_tool_calls": (count_flag(row["dropped_tool_calls"]), count_text(row["dropped_tool_calls"])),
        "malformed_tool_calls": (count_flag(row["malformed_tool_calls"]), count_text(row["malformed_tool_calls"])),
        "text_tool_calls": (count_flag(row["text_tool_calls"]), count_text(row["text_tool_calls"])),
        "policy_violation": (bool(row["policy_violation"]), score_detail(trace, "violation_details")
                             or score_detail(trace, "violations")),
        "wrong_final_state": (not bool(row["state_match"]), score_detail(trace, "diff")),
        "missing_required_output": (not bool(row["output_match"]), score_detail(trace, "missing_outputs")),
        "empty_reply": (flag_or_none(row["empty_reply"]), ""),
        "wrong_script": (flag_or_none(row["wrong_script"]), ""),
        "run_error": (row["stop_reason"] in ERROR_STOPS or bool(row["llm_timeout"]),
                      str(row.get("error") or row["stop_reason"])[:200]
                      if row["stop_reason"] in ERROR_STOPS or bool(row["llm_timeout"]) else ""),
    }
    loop = loop_detail(trace) if has_trace else None
    out["loop"] = (None, "") if loop is None else (bool(loop), loop)
    for name, detail_fn in (("handoff_missing_tool", handoff_tool_detail),
                            ("handoff_missing_identifier", handoff_identifier_detail)):
        if not supervisor:
            out[name] = (False, "")
        elif not has_trace:
            out[name] = (None, "")
        else:
            detail = detail_fn(trace, task)
            out[name] = (bool(detail), detail)
    return out


def count_flag(value):
    return None if value is None or value != value else value > 0


def count_text(value):
    return f"{int(value)} call(s)" if value is not None and value == value and value > 0 else ""


def flag_or_none(value):
    return None if value is None or value != value else bool(value)


def failure_signals(data):
    failed = data.rows[~data.rows["success"]]
    records = []
    for _, row in failed.iterrows():
        trace = data.traces.get(row["run_id"])
        task = data.tasks.get(row["task_id"])
        signals = run_signals(row, trace, task)
        fired = [name for name in SIGNALS if signals[name][0]]
        record = {
            "run_id": row["run_id"],
            "model": row["model"],
            "arch": row["arch"],
            "task_id": row["task_id"],
            "seed": int(row["seed"]),
            "category": row["category"],
            "difficulty": row["difficulty"],
            "language": row["language"],
            "is_trap": row["is_trap"],
            "stop_reason": row["stop_reason"],
            "trace_available": trace is not None,
            "n_signals": len(fired),
            "signals": "; ".join(fired),
        }
        for name in SIGNALS:
            value = signals[name][0]
            record[name] = "" if value is None else int(value)
        record["details"] = "; ".join(
            f"{name}: {signals[name][1]}" for name in SIGNALS if signals[name][0] and signals[name][1]
        )
        records.append(record)
    columns = ["run_id", "model", "arch", "task_id", "seed", "category", "difficulty", "language", "is_trap",
               "stop_reason", "trace_available", "n_signals", "signals", *SIGNALS, "details"]
    return pd.DataFrame(records, columns=columns)


def short(value, limit=60):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def trace_summary(trace, arch, limit=1500):
    """Tool calls in order with their arguments; supervisor handoffs appear where they were made."""
    if trace is None:
        return "trace not available"
    parts = []
    for event in trace.get("trace", []):
        kind = event.get("type")
        if kind == "llm_call" and event.get("agent") == "supervisor":
            for call in event.get("tool_calls") or []:
                if call.get("name") in SPECIALIST_TOOLS:
                    try:
                        instruction = json.loads(call.get("raw_arguments") or "{}").get("instruction", "")
                    except (json.JSONDecodeError, AttributeError):
                        instruction = call.get("raw_arguments") or ""
                    parts.append(f"delegate to {call['name']}: \"{short(instruction, 120)}\"")
        elif kind == "tool_call":
            arguments = event.get("arguments")
            if isinstance(arguments, dict):
                shown = ", ".join(f"{k}={short(v, 40)}" for k, v in arguments.items())
            else:
                shown = short(event.get("raw_arguments") or "", 60)
            result = event.get("result") or {}
            if event.get("malformed"):
                status = "malformed"
            elif result.get("ok") is False:
                status = f"error: {short(result.get('error') or '', 60)}"
            else:
                status = "ok"
            prefix = "" if arch == "react" else f"{event.get('agent')}: "
            parts.append(f"{prefix}{event.get('name')}({shown}) {status}")
    if not parts:
        return "no tool calls"
    text = "; ".join(f"{i}. {part}" for i, part in enumerate(parts, start=1))
    return text if len(text) <= limit else text[: limit - 3] + "..."


def labeling_sheet(data, signals_df, per_cell=SAMPLE_PER_CELL, seed=SAMPLE_SEED):
    """Up to per_cell failed runs per model and architecture, drawn at random, rows in random order."""
    rng = np.random.default_rng(seed)
    chosen = []
    for model in data.models:
        for arch in data.archs:
            cell = sorted(signals_df[(signals_df["model"] == model) & (signals_df["arch"] == arch)]["run_id"])
            if len(cell) > per_cell:
                cell = sorted(rng.choice(cell, size=per_cell, replace=False).tolist())
            chosen.extend(cell)
    chosen = [chosen[i] for i in rng.permutation(len(chosen))]
    by_id = signals_df.set_index("run_id")
    rows = data.rows.set_index("run_id")
    records = []
    for rid in chosen:
        row = rows.loc[rid]
        trace = data.traces.get(rid)
        task = data.tasks.get(row["task_id"])
        records.append({
            "run_id": rid,
            "arch": row["arch"],
            "model": row["model"],
            "task_id": row["task_id"],
            "instruction": customer_message(trace, task),
            "final_reply": (trace or {}).get("final_reply") or "",
            "trace_summary": trace_summary(trace, row["arch"]),
            "auto_signals": by_id.loc[rid, "signals"],
            "label_1": "",
            "label_2": "",
            "notes": "",
        })
    return pd.DataFrame(records, columns=list(SHEET_COLUMNS))


def reviewer_copy(sheet, reviewer):
    """The sheet with only this reviewer's label column, so neither sees the other's labels."""
    keep = [c for c in SHEET_COLUMNS if c not in LABEL_COLUMNS or c == f"label_{reviewer}"]
    return sheet[keep]


def has_labels(path):
    if not path.exists():
        return False
    sheet = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    return any(sheet[c].str.strip().ne("").any() for c in LABEL_COLUMNS if c in sheet.columns)
