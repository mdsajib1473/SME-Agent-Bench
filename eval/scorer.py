"""Task scorer.

Scoring follows the tau-bench shape: replay the gold write actions on a fresh
shop to get the expected database state, compare it against the state the agent
left behind, check the required strings appear in the final reply, and flag any
forbidden write tool that succeeded.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env import tools
from env.shop import Shop

TASKS_PATH = ROOT / "tasks" / "tasks.jsonl"

CATEGORIES = ("refund", "order_change", "complaint_routing", "quotation", "inquiry")
DIFFICULTIES = ("easy", "medium", "hard")
LANGUAGES = ("en", "banglish")

WRITE_TOOLS = (
    "cancel_order",
    "update_order",
    "issue_refund",
    "create_quote",
    "route_ticket",
    "apply_coupon",
)

# Free-text columns carry agent wording, which is not part of the graded state.
IGNORED_COLUMNS = {
    "refunds": ("reason",),
    "tickets": ("summary",),
}

MAX_DIFF_ITEMS = 4


def load_tasks(path=TASKS_PATH):
    tasks = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        text = line.strip()
        if not text:
            continue
        try:
            tasks.append(json.loads(text))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} is not valid JSON: {error}")
    return tasks


def comparable_state(snapshot):
    """Drop free-text columns so wording differences do not fail a task."""
    state = {}
    for table, rows in snapshot.items():
        ignored = IGNORED_COLUMNS.get(table, ())
        state[table] = [
            {key: value for key, value in row.items() if key not in ignored}
            for row in rows
        ]
    return state


def replay_gold_actions(shop, gold_actions):
    results = []
    for index, action in enumerate(gold_actions):
        name = action.get("name")
        arguments = action.get("arguments", {})
        result = tools.call_tool(shop, name, arguments)
        results.append({"index": index, "name": name, "result": result})
    return results


def expected_state(task, seed_db_path=None):
    """Comparable state after a correct agent performs exactly gold_actions."""
    with (Shop(seed_db_path) if seed_db_path else Shop()) as shop:
        results = replay_gold_actions(shop, task.get("gold_actions", []))
        failures = [r for r in results if not r["result"].get("ok")]
        if failures:
            raise ValueError(
                f"task {task.get('task_id')} gold_actions failed on replay: "
                f"{[(f['name'], f['result'].get('error')) for f in failures]}"
            )
        return comparable_state(shop.snapshot())


def _normalize_text(text):
    lowered = str(text).lower()
    return re.sub(r"(?<=\d),(?=\d)", "", lowered)


def _output_variants(value):
    variants = {_normalize_text(value)}
    if isinstance(value, bool):
        return variants
    if isinstance(value, (int, float)):
        variants.add(_normalize_text(f"{value:,}"))
        if isinstance(value, float) and value.is_integer():
            variants.add(_normalize_text(int(value)))
    else:
        text = str(value).strip()
        digits = text.replace(",", "")
        if digits.isdigit():
            variants.add(_normalize_text(digits))
            variants.add(_normalize_text(f"{int(digits):,}"))
    return variants


def check_outputs(required_outputs, final_reply):
    reply = _normalize_text(final_reply or "")
    missing = []
    for value in required_outputs or []:
        if not any(variant in reply for variant in _output_variants(value)):
            missing.append(value)
    return missing


CONDITION_OPS = ("gt", "lt", "eq", "ne")


def normalize_forbidden(forbidden_actions):
    """Turn both spellings into (tool, condition_or_None) pairs."""
    rules = []
    for entry in forbidden_actions or []:
        if isinstance(entry, str):
            rules.append((entry, None))
        elif isinstance(entry, dict):
            rules.append((entry.get("tool"), entry.get("when")))
    return rules


def _condition_holds(condition, arguments):
    """A missing argument never satisfies a condition."""
    if not isinstance(condition, dict):
        return False
    name = condition.get("arg")
    operator = condition.get("op")
    wanted = condition.get("value")
    if name not in (arguments or {}):
        return False
    actual = arguments[name]

    if operator in ("eq", "ne"):
        same = actual == wanted
        return same if operator == "eq" else not same

    numeric = (int, float)
    if isinstance(actual, bool) or isinstance(wanted, bool):
        return False
    if not isinstance(actual, numeric) or not isinstance(wanted, numeric):
        return False
    return actual > wanted if operator == "gt" else actual < wanted


def describe_condition(tool, condition):
    if not condition:
        return tool
    return (
        f"{tool}[{condition.get('arg')} {condition.get('op')}"
        f" {condition.get('value')}]"
    )


def find_violations(forbidden_actions, tool_calls):
    """Forbidden tools that succeeded, filtered by any argument condition."""
    rules = normalize_forbidden(forbidden_actions)
    violations = []
    details = []
    for call in tool_calls or []:
        name = call.get("name")
        result = call.get("result") or {}
        if result.get("ok") is not True:
            continue
        for tool, condition in rules:
            if tool != name:
                continue
            if condition is not None and not _condition_holds(
                condition, call.get("arguments")
            ):
                continue
            violations.append(name)
            details.append(describe_condition(tool, condition))
    return violations, details


def _row_key(table, row):
    keys = {
        "customers": ("customer_id",),
        "products": ("product_id",),
        "orders": ("order_id",),
        "order_items": ("order_id", "product_id"),
        "refunds": ("refund_id",),
        "tickets": ("ticket_id",),
        "quotes": ("quote_id",),
        "quote_items": ("quote_id", "product_id"),
        "coupons": ("code",),
        "coupon_uses": ("order_id", "code"),
    }[table]
    return tuple(str(row[key]) for key in keys)


def describe_diff(expected, actual):
    notes = []
    for table in expected:
        expected_rows = {_row_key(table, row): row for row in expected[table]}
        actual_rows = {_row_key(table, row): row for row in actual.get(table, [])}

        only_expected = sorted(set(expected_rows) - set(actual_rows))
        only_actual = sorted(set(actual_rows) - set(expected_rows))
        for key in only_expected[:MAX_DIFF_ITEMS]:
            notes.append(f"{table}: missing row {'/'.join(key)}")
        for key in only_actual[:MAX_DIFF_ITEMS]:
            notes.append(f"{table}: unexpected row {'/'.join(key)}")

        for key in sorted(set(expected_rows) & set(actual_rows)):
            for column, want in expected_rows[key].items():
                got = actual_rows[key].get(column)
                if want != got:
                    notes.append(
                        f"{table} {'/'.join(key)}: {column} expected {want!r},"
                        f" actual {got!r}"
                    )
            if len(notes) >= MAX_DIFF_ITEMS * 3:
                break
    if not notes:
        return ""
    shown = notes[: MAX_DIFF_ITEMS * 2]
    suffix = "" if len(notes) == len(shown) else f" (+{len(notes) - len(shown)} more)"
    return "; ".join(shown) + suffix


def score(task, actual_snapshot, final_reply, tool_calls, seed_db_path=None):
    expected = expected_state(task, seed_db_path)
    actual = comparable_state(actual_snapshot)

    state_match = expected == actual
    missing_outputs = check_outputs(task.get("required_outputs"), final_reply)
    violations, violation_details = find_violations(
        task.get("forbidden_actions"), tool_calls
    )

    output_match = not missing_outputs
    policy_violation = bool(violations)

    return {
        "task_id": task.get("task_id"),
        "state_match": state_match,
        "output_match": output_match,
        "policy_violation": policy_violation,
        "success": state_match and output_match and not policy_violation,
        "diff": "" if state_match else describe_diff(expected, actual),
        "missing_outputs": missing_outputs,
        "violations": violations,
        "violation_details": violation_details,
    }
