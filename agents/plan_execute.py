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


def _capability_lines():
    # Capabilities in words only: tool names and argument syntax in the plan
    # led the executor to copy them as malformed calls.
    lines = []
    for name in ALL_TOOL_NAMES:
        description = SCHEMA_BY_NAME[name]["function"]["description"]
        lines.append(f"- {description.split('. ')[0].rstrip('.')}.")
    return "\n".join(lines)


PLANNER_ROLE = (
    "You are the planner in a plan-and-execute team. You cannot use tools and"
    " you have not seen any shop data. Write a plan that an executor will carry"
    " out one step at a time. The executor can:\n"
    f"{_capability_lines()}\n\n"
    "Write each step as the goal of that step in plain words, for example"
    " 'Find the order and check that the phone number the customer gave is the"
    " one registered on it'. Never write tool names, tool arguments, IDs,"
    " numbers or other values in a step; the executor reads them from the"
    " customer's message. Include the lookups needed to check the policy"
    " before any action, and make an action conditional when the policy only"
    " allows it in some cases. Do not add a step for writing the reply; a"
    " responder writes it after the steps run. Use at most 6 steps.\n"
    'Reply with only JSON in this form: {"steps": ["first goal", "second goal"]}'
)

EXECUTOR_ROLE = (
    "You are the executor in a plan-and-execute team. Carry out only the"
    " current step of the plan, using the tools. Never take an action the"
    " policy forbids. When the step is done, reply without calling a tool,"
    " starting with DONE and then a short report in plain sentences of what you"
    " found or did, with exact IDs, amounts and statuses. If you cannot"
    " complete the step, including because the policy forbids it, reply"
    " starting with FAILED and the reason."
)

RESPONDER_ROLE = (
    "You are the responder in a plan-and-execute team. Write the final reply"
    " to the customer from the results of the steps that were run. Do not claim"
    " any action the results do not show."
)

# A lowercase "failed" mid-report is ordinary content ("failed payments"); only
# the uppercase marker or a report that opens with the word marks a failed step.
FAILED_MARKER = re.compile(r"\bFAILED\b")
FAILED_START = re.compile(r"^\W*failed\b", re.IGNORECASE)
REPORT_PREFIX = re.compile(r"^\W*(done|failed)\W*", re.IGNORECASE)


def report_failed(report):
    report = (report or "").strip()
    return bool(FAILED_MARKER.search(report) or FAILED_START.match(report))


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


def describe(value):
    """Flatten a tool result into plain words, with no brackets or quotes."""
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            if key == "ok":
                continue
            parts.append(f"{key.replace('_', ' ')} {describe(item)}")
        return ", ".join(parts)
    if isinstance(value, list):
        if not value:
            return "none"
        return "; ".join(describe(item) for item in value)
    if value is None:
        return "not set"
    return str(value)


def _sentence(text):
    text = " ".join(str(text).split())
    if len(text) > RESULT_TEXT_LIMIT:
        text = text[:RESULT_TEXT_LIMIT].rsplit(" ", 1)[0] + " and more"
    return text if text.endswith(".") else text + "."


def format_results(records):
    # Plain sentences only. A 7B executor shown labelled blocks or JSON started
    # writing the same blocks into its own reports instead of calling tools.
    if not records:
        return "No step has run yet."
    paragraphs = []
    for record in records:
        report = REPORT_PREFIX.sub("", record.report or "").strip()
        sentences = [
            f"Step {record.index} had the goal: {_sentence(record.instruction)}",
            "That step did not succeed." if record.failed else "That step succeeded.",
            f"The executor said: {_sentence(report)}" if report else "The executor gave no report.",
        ]
        for call, result in record.calls:
            if result.get("ok"):
                sentences.append(f"Looking this up with {call.name} gave {_sentence(describe(result))}")
            else:
                sentences.append(f"Calling {call.name} returned an error: {_sentence(result.get('error'))}")
        paragraphs.append(" ".join(sentences))
    return "\n\n".join(paragraphs)


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
                ' in this form: {"steps": ["first goal", "second goal"]}'
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
            f"Step {failed.index} did not succeed. Its goal was: {failed.instruction}\n"
            f"What went wrong: {failed.failure}\n\n"
            "Write a new plan for the remaining work only, taking this into"
            " account. Write each step as a goal in plain words, with no tool"
            ' names, arguments or IDs. If no more tool work is needed, reply {"steps": []}.'
        )

    def _execute_step(self, ctx, request, plan, records, instruction):
        index = len(records) + 1
        user_content = (
            f"Customer request:\n{request}\n\n"
            f"Full plan:\n{format_plan(plan)}\n\n"
            f"What earlier steps found:\n{format_results(records)}\n\n"
            f"Your job now is step {index} only: {instruction}\n"
            "Call the tools this step needs. Then reply DONE with a short report,"
            " or FAILED with the reason if you cannot do it."
        )
        messages = ctx.open_conversation("executor", EXECUTOR_ROLE, user_content)
        max_calls = min(MAX_STEP_CALLS, ctx.llm.calls_remaining - RESPONDER_RESERVE)
        loop = ctx.tool_loop(messages, schemas_for(ALL_TOOL_NAMES), max_calls, agent="executor")

        report = loop.reply.strip()
        failure = None
        last = loop.last_result
        if not loop.finished:
            failure = f"executor stopped after {loop.llm_calls} LLM calls without a report"
        elif report_failed(report):
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
        # The customer's message goes last, right before the generation, so the
        # reply follows its language style rather than the long results text.
        user_content = (
            f"What the team found and did:\n{format_results(records)}\n\n"
            "Write the reply to the customer now. Answer in the same language"
            " style as the customer's message below: if it is in English, reply in"
            " English; if it is in Banglish (Bangla written in Latin letters),"
            " reply in Banglish. Write plain text only, with no markdown, lists,"
            " headings or emoji.\n\n"
            f"Customer's message:\n{request}"
        )
        messages = ctx.open_conversation("responder", RESPONDER_ROLE, user_content)
        return ctx.chat(messages, agent="responder").content
