"""Determinism check: repeat one architecture on one task with the same seed.

Runs the task N times with a model reset before each run, then N times in a row
without any reset, and reports whether the final database states, the first
tool call, and the final replies are identical within each series. Results go
to results/determinism_<arch>_<task>.json.

Usage (PowerShell):
    .venv\\Scripts\\python.exe scripts\\check_determinism.py
    .venv\\Scripts\\python.exe scripts\\check_determinism.py --arch supervisor --task CMP-M-01 --runs 3
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.llm import LLMClient, load_config, reset_model_state
from agents.registry import ARCHITECTURES, build_agent
from env.shop import Shop
from eval import scorer

MODEL = "qwen2.5-7b-8k"


def digest(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def first_action(result):
    """First tool call as name plus arguments, or the first text reply if none came first."""
    for event in result.trace:
        if event["type"] == "tool_call":
            return f"{event['name']} {json.dumps(event['arguments'], sort_keys=True)}"
        if event["type"] == "llm_call" and not event["tool_calls"]:
            return f"TEXT {event['content'][:60]!r}"
    return "none"


def run_series(agent, task, llm, seed, runs, reset):
    rows = []
    for number in range(1, runs + 1):
        reset_s = reset_model_state(llm.model, llm.config) if reset else None
        with Shop() as shop:
            result = agent.run(task, shop, llm, seed)
            snapshot = shop.snapshot()
        score = scorer.score(task, snapshot, result.final_reply, result.tool_calls)
        rows.append({
            "run": number,
            "reset_s": reset_s,
            "state": digest(scorer.comparable_state(snapshot)),
            "first_action": first_action(result),
            "reply": digest(result.final_reply),
            "success": score["success"],
            "llm_calls": result.metadata["llm_calls"],
            "completion_tokens": result.metadata["completion_tokens"],
            "stop_reason": result.metadata["stop_reason"],
            "final_reply": result.final_reply,
        })
    return rows


def summarize(label, rows):
    print(f"== {label} ==")
    print("run  reset_s  state         reply         ok  llm  compl  first action")
    for row in rows:
        reset = "-" if row["reset_s"] is None else f"{row['reset_s']:.2f}"
        print(
            f"{row['run']:<4} {reset:<8} {row['state']:<13} {row['reply']:<13}"
            f" {'Y' if row['success'] else '-':<3} {row['llm_calls']:<4} {row['completion_tokens']:<6}"
            f" {row['first_action']}"
        )
    verdict = {
        key: len({row[key] for row in rows}) == 1
        for key in ("state", "first_action", "reply")
    }
    print(
        f"identical final DB state: {verdict['state']}; identical first tool call:"
        f" {verdict['first_action']}; identical final reply: {verdict['reply']}"
    )
    print()
    return verdict


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--arch", default="react", choices=list(ARCHITECTURES))
    parser.add_argument("--task", default="REF-E-01")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    config = load_config()
    seed = config["seeds"][0]
    task = {t["task_id"]: t for t in scorer.load_tasks()}[args.task]
    llm = LLMClient(args.model, config=config)
    agent = build_agent(args.arch)

    print(f"{args.arch} on {args.task}, model {args.model}, seed {seed}, {args.runs} runs per series\n")
    with_reset = run_series(agent, task, llm, seed, args.runs, reset=True)
    without_reset = run_series(agent, task, llm, seed, args.runs, reset=False)
    report = {
        "with_reset": {"verdict": summarize("with reset before each run", with_reset), "runs": with_reset},
        "without_reset": {"verdict": summarize("without reset", without_reset), "runs": without_reset},
    }

    path = ROOT / "results" / f"determinism_{args.arch}_{args.task}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"full results: {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
