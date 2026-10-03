"""Task file validator.

Checks the schema, replays every gold action, confirms required outputs are
derivable from the seed database, then runs two reference agents: an oracle that
performs exactly the gold actions and quotes the required outputs, and a no-op
agent that does nothing. The oracle must pass everything; the no-op agent must
pass only the traps whose correct behaviour is to do nothing and say so.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.shop import Shop
from eval import scorer

NO_OP_REPLY = "Sorry, I cannot help"

REQUIRED_FIELDS = {
    "task_id": str,
    "category": str,
    "difficulty": str,
    "language": str,
    "is_trap": bool,
    "instruction": str,
    "gold_actions": list,
    "required_outputs": list,
    "forbidden_actions": list,
    "notes": str,
}


def check_schema(tasks):
    problems = []
    seen = set()
    for index, task in enumerate(tasks):
        label = task.get("task_id", f"line {index + 1}")

        for field, kind in REQUIRED_FIELDS.items():
            if field not in task:
                problems.append(f"{label}: missing field {field}")
            elif not isinstance(task[field], kind):
                problems.append(
                    f"{label}: field {field} should be {kind.__name__},"
                    f" found {type(task[field]).__name__}"
                )
        if problems and task.get("task_id") is None:
            continue

        if task.get("task_id") in seen:
            problems.append(f"{label}: duplicate task_id")
        seen.add(task.get("task_id"))

        if task.get("category") not in scorer.CATEGORIES:
            problems.append(f"{label}: category {task.get('category')!r} is unknown")
        if task.get("difficulty") not in scorer.DIFFICULTIES:
            problems.append(
                f"{label}: difficulty {task.get('difficulty')!r} is unknown"
            )
        if task.get("language") not in scorer.LANGUAGES:
            problems.append(f"{label}: language {task.get('language')!r} is unknown")
        if not str(task.get("instruction", "")).strip():
            problems.append(f"{label}: instruction is empty")
        if not str(task.get("notes", "")).strip():
            problems.append(f"{label}: notes is empty")

        problems.extend(check_forbidden_shape(label, task.get("forbidden_actions", [])))

        rules = scorer.normalize_forbidden(task.get("forbidden_actions", []))
        unconditional = {tool for tool, condition in rules if condition is None}

        for action in task.get("gold_actions", []):
            if not isinstance(action, dict):
                problems.append(f"{label}: gold action is not an object")
                continue
            name = action.get("name")
            if name not in scorer.WRITE_TOOLS:
                problems.append(
                    f"{label}: gold action {name!r} is not a write tool"
                )
            arguments = action.get("arguments", {})
            if not isinstance(arguments, dict):
                problems.append(f"{label}: gold action {name} arguments is not an object")
                continue
            if name in unconditional:
                problems.append(
                    f"{label}: {name} appears in gold_actions and forbidden_actions"
                )
            for tool, condition in rules:
                if condition is None or tool != name:
                    continue
                if scorer._condition_holds(condition, arguments):
                    problems.append(
                        f"{label}: gold action {name} satisfies its own forbidden"
                        f" condition {scorer.describe_condition(tool, condition)}"
                    )

    return problems


def check_forbidden_shape(label, forbidden_actions):
    """forbidden_actions holds tool names, or {tool, when} condition objects."""
    problems = []
    for entry in forbidden_actions:
        if isinstance(entry, str):
            if entry not in scorer.WRITE_TOOLS:
                problems.append(
                    f"{label}: forbidden action {entry!r} is not a write tool"
                )
            continue
        if not isinstance(entry, dict):
            problems.append(
                f"{label}: forbidden action must be a string or an object,"
                f" found {type(entry).__name__}"
            )
            continue

        extra = set(entry) - {"tool", "when"}
        if extra:
            problems.append(
                f"{label}: forbidden action has unknown keys {sorted(extra)}"
            )
        tool = entry.get("tool")
        if tool not in scorer.WRITE_TOOLS:
            problems.append(f"{label}: forbidden action {tool!r} is not a write tool")

        if "when" not in entry:
            continue
        condition = entry["when"]
        if not isinstance(condition, dict):
            problems.append(f"{label}: forbidden action {tool} when is not an object")
            continue
        condition_extra = set(condition) - {"arg", "op", "value"}
        if condition_extra:
            problems.append(
                f"{label}: forbidden action {tool} when has unknown keys"
                f" {sorted(condition_extra)}"
            )
        if not isinstance(condition.get("arg"), str) or not condition["arg"].strip():
            problems.append(
                f"{label}: forbidden action {tool} when needs a non-empty arg name"
            )
        if condition.get("op") not in scorer.CONDITION_OPS:
            problems.append(
                f"{label}: forbidden action {tool} when op"
                f" {condition.get('op')!r} is not one of"
                f" {', '.join(scorer.CONDITION_OPS)}"
            )
        if "value" not in condition:
            problems.append(f"{label}: forbidden action {tool} when needs a value")
        elif condition["op"] in ("gt", "lt") and (
            isinstance(condition["value"], bool)
            or not isinstance(condition["value"], (int, float))
        ):
            problems.append(
                f"{label}: forbidden action {tool} when op {condition['op']}"
                " needs a numeric value"
            )
    return problems


def replay_check(tasks):
    problems = []
    for task in tasks:
        with Shop() as shop:
            results = scorer.replay_gold_actions(shop, task.get("gold_actions", []))
        for entry in results:
            if not entry["result"].get("ok"):
                problems.append(
                    f"{task['task_id']}: gold action {entry['index']}"
                    f" ({entry['name']}) failed: {entry['result'].get('error')}"
                )
    return problems


def _seed_corpus():
    """Every value in the untouched seed, as one normalized searchable string."""
    with Shop() as shop:
        snapshot = shop.snapshot()
    pieces = []
    for rows in snapshot.values():
        for row in rows:
            pieces.extend(str(value) for value in row.values())
    return scorer._normalize_text(" ".join(pieces))


def outputs_derivable_check(tasks, seed_corpus):
    problems = []
    for task in tasks:
        if not task.get("required_outputs"):
            continue
        with Shop() as shop:
            results = scorer.replay_gold_actions(shop, task.get("gold_actions", []))
        replay_corpus = scorer._normalize_text(
            json.dumps([entry["result"] for entry in results])
        )
        corpus = f"{seed_corpus} {replay_corpus}"
        for value in task["required_outputs"]:
            variants = scorer._output_variants(value)
            if not any(variant in corpus for variant in variants):
                problems.append(
                    f"{task['task_id']}: required output {value!r} is not derivable"
                    " from the seed database or the gold action results"
                )
    return problems


ORDER_IN_TEXT = re.compile(r"ORD-\d{4}")
PHONE_IN_TEXT = re.compile(r"\b01\d{9}\b")

# Phrases a reviewer uses in notes to mark a deliberate identity mismatch.
IDENTITY_TRAP_MARKERS = (
    "does not match",
    "identity fails",
    "not the number registered",
)


def _seed_owners():
    with Shop() as shop:
        rows = shop.connection.execute(
            "SELECT o.order_id, c.phone, c.name FROM orders o"
            " JOIN customers c ON c.customer_id = o.customer_id"
        ).fetchall()
    return {row["order_id"]: (row["phone"], row["name"]) for row in rows}


def is_identity_trap(task):
    notes = str(task.get("notes", "")).lower()
    return bool(task.get("is_trap")) and any(
        marker in notes for marker in IDENTITY_TRAP_MARKERS
    )


def ownership_check(tasks):
    """Every order named in an instruction must carry a phone that owns it."""
    owners = _seed_owners()
    failures = []
    intended = []
    claims = defaultdict(dict)

    for task in tasks:
        label = task.get("task_id")
        instruction = str(task.get("instruction", ""))
        phones = set(PHONE_IN_TEXT.findall(instruction))
        trap = is_identity_trap(task)

        for order_id in sorted(set(ORDER_IN_TEXT.findall(instruction))):
            if order_id not in owners:
                failures.append(f"{label}: instruction names unknown order {order_id}")
                continue
            owner_phone, owner_name = owners[order_id]
            # An identity trap gives a wrong phone on purpose, so it cannot
            # take part in the cross-task consistency comparison.
            if not trap:
                claims[order_id][label] = frozenset(phones)

            if owner_phone in phones:
                continue
            detail = (
                f"{label}: {order_id} belongs to {owner_name} on {owner_phone},"
                f" but the instruction gives {sorted(phones) or ['no phone']}"
            )
            (intended if trap else failures).append(detail)

    for order_id, per_task in sorted(claims.items()):
        if len(per_task) < 2:
            continue
        labels = sorted(per_task)
        shared = frozenset.intersection(*per_task.values()) if per_task else frozenset()
        if not shared:
            spelled = ", ".join(
                f"{label} gives {sorted(per_task[label]) or ['no phone']}"
                for label in labels
            )
            failures.append(
                f"{order_id} is used with different phones: {spelled}"
            )

    return failures, intended


def run_oracle(task):
    """Perform exactly the gold actions and quote every required output."""
    with Shop() as shop:
        results = scorer.replay_gold_actions(shop, task.get("gold_actions", []))
        tool_calls = [
            {
                "name": entry["name"],
                "arguments": task["gold_actions"][entry["index"]].get("arguments", {}),
                "result": entry["result"],
            }
            for entry in results
        ]
        quoted = ", ".join(str(value) for value in task.get("required_outputs", []))
        reply = f"Handled as policy requires. {quoted}".strip()
        snapshot = shop.snapshot()
    return scorer.score(task, snapshot, reply, tool_calls)


def run_no_op(task):
    with Shop() as shop:
        snapshot = shop.snapshot()
    return scorer.score(task, snapshot, NO_OP_REPLY, [])


def is_do_nothing_trap(task):
    return (
        task.get("is_trap")
        and not task.get("gold_actions")
        and not task.get("required_outputs")
    )


def main(argv=None):
    paths = [Path(arg) for arg in (argv if argv is not None else sys.argv[1:])]
    if not paths:
        paths = [scorer.TASKS_PATH]

    tasks = []
    for path in paths:
        loaded = scorer.load_tasks(path)
        print(f"loaded {len(loaded)} tasks from {path}")
        tasks.extend(loaded)
    print(f"tasks loaded: {len(tasks)}")

    problems = check_schema(tasks)
    problems.extend(replay_check(tasks))
    problems.extend(outputs_derivable_check(tasks, _seed_corpus()))

    ownership_failures, intended_mismatches = ownership_check(tasks)
    problems.extend(ownership_failures)

    if problems:
        print("\nschema and replay problems:")
        for problem in problems:
            print(f"  {problem}")

    print(f"\nintended identity mismatches: {len(intended_mismatches)}")
    for note in intended_mismatches:
        print(f"  {note}")

    print("\noracle agent:")
    oracle_failures = []
    for task in tasks:
        result = run_oracle(task)
        flag = "pass" if result["success"] else "FAIL"
        print(f"  {task['task_id']:<10} {flag}")
        if not result["success"]:
            oracle_failures.append(result)
            print(
                f"    state_match={result['state_match']}"
                f" output_match={result['output_match']}"
                f" policy_violation={result['policy_violation']}"
            )
            if result["diff"]:
                print(f"    diff: {result['diff']}")
            if result["missing_outputs"]:
                print(f"    missing outputs: {result['missing_outputs']}")

    print("\nno-op agent:")
    no_op_successes = []
    for task in tasks:
        result = run_no_op(task)
        flag = "pass" if result["success"] else "fail"
        print(f"  {task['task_id']:<10} {flag}")
        if result["success"]:
            no_op_successes.append(task["task_id"])

    do_nothing_traps = [task["task_id"] for task in tasks if is_do_nothing_trap(task)]
    no_op_rate = len(no_op_successes) / len(tasks) if tasks else 0.0
    trap_share = len(do_nothing_traps) / len(tasks) if tasks else 0.0

    print("\nsummary")
    print(f"  schema and replay problems: {len(problems)}")
    print(f"  oracle success rate: {(len(tasks) - len(oracle_failures)) / len(tasks):.0%}")
    print(f"  no-op success rate:  {no_op_rate:.0%} {no_op_successes}")
    print(f"  do-nothing trap share: {trap_share:.0%} {do_nothing_traps}")

    unexpected = sorted(set(no_op_successes) - set(do_nothing_traps))
    if unexpected:
        print(f"  no-op passed tasks that are not do-nothing traps: {unexpected}")

    categories = sorted({task["category"] for task in tasks})
    difficulties = sorted({task["difficulty"] for task in tasks})
    print(f"  categories covered: {len(categories)} {categories}")
    print(f"  difficulties covered: {len(difficulties)} {difficulties}")
    print(f"  traps: {sum(1 for task in tasks if task['is_trap'])}")
    print(
        "  banglish tasks: "
        f"{sum(1 for task in tasks if task['language'] == 'banglish')}"
    )

    ok = not problems and not oracle_failures and not unexpected
    print(f"\nvalidation: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
