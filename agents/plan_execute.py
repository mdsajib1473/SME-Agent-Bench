"""Plan-and-Execute agent.

Planner (no tools) writes a JSON plan; an executor runs each step as a short
tool-calling loop; a failed step triggers at most one replan; a responder (no
tools) writes the customer reply from the step results.

Budget: every role draws on the one per-task LLM budget. One call is held back
for the responder, so execution stops early rather than leaving no call for
the reply. Once the replan is used, a further failed step ends execution and
goes straight to the responder, so later steps never run on a broken premise.
"""

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.base import ALL_TOOL_NAMES, SCHEMA_BY_NAME, Agent, schemas_for

MAX_PLANNER_ATTEMPTS = 2
MAX_STEP_CALLS = 4
MAX_REPLANS = 1
RESPONDER_RESERVE = 1
RESULT_TEXT_LIMIT = 2000

FALLBACK_STEP = "Handle the customer's request as the policy requires."


def _tool_lines():
    lines = []
    for name in ALL_TOOL_NAMES:
        description = SCHEMA_BY_NAME[name]["function"]["description"]
        lines.append(f"- {name}: {description.split('. ')[0].rstrip('.')}.")
    return "\n".join(lines)


PLANNER_ROLE = (
    "You are the planner in a plan-and-execute team. You cannot use tools and"
    " you have not seen any shop data. Write a plan that an executor will carry"
    " out one step at a time with these tools:\n"
    f"{_tool_lines()}\n\n"
    "Each step is one short instruction. Include the lookups needed to check"
    " the policy before any action, and make an action conditional when the"
    " policy only allows it in some cases. Do not add a step for writing the"
    " reply; a responder writes it after the steps run. Use at most 6 steps.\n"
    'Reply with only JSON in this form: {"steps": ["first step", "second step"]}'
)

EXECUTOR_ROLE = (
    "You are the executor in a plan-and-execute team. Carry out only the"
    " current step of the plan, using the tools. Never take an action the"
    " policy forbids. When the step is done, reply without calling a tool,"
    " starting with DONE: and a short report of what you found or did, with"
    " exact IDs, amounts and statuses. If you cannot complete the step,"
    " including because the policy forbids it, reply starting with FAILED:"
    " and the reason."
)

RESPONDER_ROLE = (
    "You are the responder in a plan-and-execute team. Write the final reply"
    " to the customer from their request and the results of the steps that"
    " were run. Do not claim any action the results do not show. Reply with"
    " the message to the customer only."
)

FAILED_PATTERN = re.compile(r"^\W*failed\b", re.IGNORECASE)


def parse_plan(content):
    """Return (steps, error). Accepts an object with "steps" or a bare list."""
    text = (content or "").strip()
    payload = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char in "{[":
                try:
                    payload, _ = decoder.raw_decode(text, index)
                    break
                except json.JSONDecodeError:
                    continue
    if payload is None:
        return None, "no JSON found"
    steps = payload.get("steps") if isinstance(payload, dict) else payload
    if not isinstance(steps, list):
        return None, 'JSON must be an object with a "steps" list'

    normalized = []
    for entry in steps:
        if isinstance(entry, dict):
            entry = entry.get("instruction") or entry.get("step") or entry.get("description")
        if not isinstance(entry, str) or not entry.strip():
            return None, "every step must be a non-empty string"
        normalized.append(entry.strip())
    return normalized, None


@dataclass
class StepRecord:
    index: int
    instruction: str
    report: str
    failed: bool
    failure: str | None
    calls: list = field(default_factory=list)


def _clip(value):
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= RESULT_TEXT_LIMIT else text[:RESULT_TEXT_LIMIT] + "..."


def format_results(records):
    # Tool data is shown as JSON results only, never in call syntax: a 7B
    # executor shown "tool {args} -> result" lines starts writing fake calls as
    # prose instead of calling the tools.
    if not records:
        return "None yet."
    blocks = []
    for record in records:
        lines = [
            f"Step {record.index}: {record.instruction}",
            f"Outcome: {'failed' if record.failed else 'done'}",
            f"Executor report: {record.report or '(no report)'}",
        ]
        if record.calls:
            data = [{"tool": call.name, "result": result} for call, result in record.calls]
            lines.append(f"Data returned by tools: {_clip(data)}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def format_plan(steps):
    if not steps:
        return "(no steps)"
    return "\n".join(f"{number}. {step}" for number, step in enumerate(steps, start=1))


class PlanExecuteAgent(Agent):
    name = "plan_execute"
    tools_by_role = {"planner": (), "executor": ALL_TOOL_NAMES, "responder": ()}

    def _run(self, ctx):
        request = ctx.task["instruction"]
        ctx.extra.update(
            plan_valid=None, planner_calls=0, replans=0, steps_run=0, steps_failed=0
        )

        steps = self._plan(ctx, f"Customer request:\n{request}", kind="plan")
        ctx.extra["plan_valid"] = steps is not None
        if steps is None:
            steps = [FALLBACK_STEP]
        plan = list(steps)
        pending = list(steps)
        records = []

        while pending:
            if ctx.llm.calls_remaining - RESPONDER_RESERVE < 1:
                ctx.trace.add("budget_stop", reason="no calls left for the step", pending=pending)
                break
            instruction = pending.pop(0)
            record = self._execute_step(ctx, request, plan, records, instruction)
            records.append(record)
            if not record.failed:
                continue
            if ctx.extra["replans"] >= MAX_REPLANS:
                break
            if ctx.llm.calls_remaining - RESPONDER_RESERVE < 1:
                break
            ctx.extra["replans"] += 1
            new_steps = self._plan(ctx, self._replan_message(request, plan, records), kind="replan")
            if new_steps is None:
                break
            plan = [record.instruction for record in records] + new_steps
            pending = list(new_steps)

        return self._respond(ctx, request, records), "final_answer"

    def _plan(self, ctx, user_content, kind):
        messages = ctx.open_conversation("planner", PLANNER_ROLE, user_content)
        for attempt in range(1, MAX_PLANNER_ATTEMPTS + 1):
            if attempt > 1 and ctx.llm.calls_remaining - RESPONDER_RESERVE < 1:
                break
            result = ctx.chat(messages, agent="planner")
            ctx.extra["planner_calls"] += 1
            steps, error = parse_plan(result.content)
            ctx.trace.add("plan", kind=kind, attempt=attempt, valid=error is None, steps=steps, error=error)
            if error is None:
                return steps
            retry = (
                f"That reply was not a valid plan: {error}. Reply with only JSON"
                ' in this form: {"steps": ["first step", "second step"]}'
            )
            messages.append(result.message)
            messages.append({"role": "user", "content": retry})
            ctx.trace.message("user", retry, agent="planner")
        return None

    @staticmethod
    def _replan_message(request, plan, records):
        failed = records[-1]
        done = records[:-1]
        return (
            f"Customer request:\n{request}\n\n"
            f"Current plan:\n{format_plan(plan)}\n\n"
            f"Completed steps:\n{format_results(done)}\n\n"
            f"Step {failed.index} failed: {failed.instruction}\n"
            f"Failure: {failed.failure}\n\n"
            "Write a new plan for the remaining work only, taking the failure"
            ' into account. If no more tool work is needed, reply {"steps": []}.'
        )

    def _execute_step(self, ctx, request, plan, records, instruction):
        index = len(records) + 1
        user_content = (
            f"Customer request:\n{request}\n\n"
            f"Full plan:\n{format_plan(plan)}\n\n"
            f"What earlier steps found:\n{format_results(records)}\n\n"
            f"Your job now is step {index} only: {instruction}\n"
            "Call the tools this step needs. Then reply DONE: with a short report,"
            " or FAILED: with the reason if you cannot do it."
        )
        messages = ctx.open_conversation("executor", EXECUTOR_ROLE, user_content)
        max_calls = min(MAX_STEP_CALLS, ctx.llm.calls_remaining - RESPONDER_RESERVE)
        loop = ctx.tool_loop(messages, schemas_for(ALL_TOOL_NAMES), max_calls, agent="executor")

        report = loop.reply.strip()
        failure = None
        last = loop.last_result
        if not loop.finished:
            failure = f"executor stopped after {loop.llm_calls} LLM calls without a report"
        elif FAILED_PATTERN.match(report):
            failure = report
        elif last is not None and last.get("ok") is not True:
            failure = f"last tool call returned an error: {last.get('error')}"

        ctx.extra["steps_run"] += 1
        if failure:
            ctx.extra["steps_failed"] += 1
        ctx.trace.add(
            "step",
            index=index,
            instruction=instruction,
            status="failed" if failure else "done",
            report=report,
            failure=failure,
            llm_calls=loop.llm_calls,
        )
        return StepRecord(index, instruction, report, failure is not None, failure, loop.calls)

    def _respond(self, ctx, request, records):
        # The customer's message goes last, next to the generation, so the reply
        # follows its language style rather than the long English results block.
        user_content = (
            f"What the team found and did:\n{format_results(records)}\n\n"
            f"Customer's message:\n{request}\n\n"
            "Write the reply to this customer now, in the same language style as"
            " their message."
        )
        messages = ctx.open_conversation("responder", RESPONDER_ROLE, user_content)
        return ctx.chat(messages, agent="responder").content
