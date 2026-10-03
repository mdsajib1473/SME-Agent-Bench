"""Task file validator.

Checks the schema, replays every gold action, confirms required outputs are
derivable from the seed database, then runs two reference agents: an oracle that
performs exactly the gold actions and quotes the required outputs, and a no-op
agent that does nothing. The oracle must pass everything; the no-op agent must
pass only the traps whose correct behaviour is to do nothing and say so.
"""

import json
import sys
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

        for action in task.get("gold_actions", []):
            if not isinstance(action, dict):
                problems.append(f"{label}: gold action is not an object")
                continue
            name = action.get("name")
            if name not in scorer.WRITE_TOOLS:
                problems.append(
                    f"{label}: gold action {name!r} is not a write tool"
                )
            if not isinstance(action.get("arguments", {}), dict):
                problems.append(f"{label}: gold action {name} arguments is not an object")
            if name in task.get("forbidden_actions", []):
                problems.append(
                    f"{label}: {name} appears in gold_actions and forbidden_actions"
                )

        for name in task.get("forbidden_actions", []):
            if name not in scorer.WRITE_TOOLS:
                problems.append(f"{label}: forbidden action {name!r} is not a write tool")

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


def main():
    tasks = scorer.load_tasks()
    print(f"tasks loaded: {len(tasks)}")

    problems = check_schema(tasks)
    problems.extend(replay_check(tasks))
    problems.extend(outputs_derivable_check(tasks, _seed_corpus()))

    if problems:
        print("\nschema and replay problems:")
        for problem in problems:
            print(f"  {problem}")

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
