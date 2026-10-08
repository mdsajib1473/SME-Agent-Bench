"""Abstract agent and the run context every architecture shares.

Agent.run owns everything the arms must share: resetting the LLM budget, loading
the policy once, budget and endpoint failure handling, the trace, and the
metadata. Each architecture only implements _run(ctx), and talks to the model
and the shop only through ctx, so every arm uses the same client, the same
shared system prompt, the same tools, and the same per-task call budget.
"""

import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.llm import BudgetExceeded, LLMError
from agents.prompts import build_system_prompt, load_policy, policy_sha256
from env.tools import TOOL_SCHEMAS

ALL_TOOL_NAMES = tuple(entry["function"]["name"] for entry in TOOL_SCHEMAS)
SCHEMA_BY_NAME = {entry["function"]["name"]: entry for entry in TOOL_SCHEMAS}


def schemas_for(names):
    return [SCHEMA_BY_NAME[name] for name in names]


class Trace:
    """Ordered record of messages, LLM calls, tool calls and handoffs, timed from start."""

    def __init__(self):
        self.started = time.perf_counter()
        self.events = []

    def add(self, event_type, **payload):
        event = {"t": round(time.perf_counter() - self.started, 4), "type": event_type}
        event.update(payload)
        self.events.append(event)
        return event

    def message(self, role, content, agent=None):
        return self.add("message", agent=agent, role=role, content=content)

    def llm_call(self, result, agent=None):
        return self.add(
            "llm_call",
            agent=agent,
            call_index=result.call_index,
            latency_s=round(result.latency_s, 4),
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            finish_reason=result.finish_reason,
            content=result.content,
            tool_calls=[
                {
                    "id": call.id,
                    "name": call.name,
                    "raw_arguments": call.raw_arguments,
                    "source": call.source,
                    "error": call.error,
                }
                for call in result.tool_calls
            ],
        )

    def tool_call(self, call, result, latency_s, agent=None):
        return self.add(
            "tool_call",
            agent=agent,
            id=call.id,
            name=call.name,
            arguments=call.arguments,
            raw_arguments=call.raw_arguments,
            source=call.source,
            malformed=call.malformed,
            result=result,
            latency_s=round(latency_s, 4),
        )

    def elapsed(self):
        return time.perf_counter() - self.started


@dataclass
class LoopResult:
    reply: str
    finished: bool
    calls: list = field(default_factory=list)
    llm_calls: int = 0

    @property
    def last_result(self):
        return self.calls[-1][1] if self.calls else None


class RunContext:
    def __init__(self, task, shop, llm, seed, trace, policy_text):
        self.task = task
        self.shop = shop
        self.llm = llm
        self.seed = seed
        self.trace = trace
        self.policy_text = policy_text
        self.policy_sha256 = policy_sha256(policy_text)
        self.extra = {}
        self._logged_systems = set()

    def system_prompt(self, role_instructions):
        return build_system_prompt(role_instructions, policy_text=self.policy_text).text

    def open_conversation(self, agent, role_instructions, user_content):
        """Fresh message list for one agent; logs its system prompt the first time."""
        system = self.system_prompt(role_instructions)
        if agent not in self._logged_systems:
            self.trace.message("system", system, agent=agent)
            self._logged_systems.add(agent)
        self.trace.message("user", user_content, agent=agent)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    def chat(self, messages, tools=None, agent=None):
        result = self.llm.chat(messages, tools=tools, seed=self.seed)
        self.trace.llm_call(result, agent=agent)
        return result

    def execute(self, call, agent=None):
        started = time.perf_counter()
        result = self.llm.execute_tool_call(self.shop, call)
        self.trace.tool_call(call, result, time.perf_counter() - started, agent=agent)
        return result

    def tool_loop(self, messages, tools, max_calls, agent=None):
        """Call, execute tool calls, repeat until a reply without tool calls."""
        loop = LoopResult(reply="", finished=False)
        for _ in range(max_calls):
            result = self.chat(messages, tools=tools, agent=agent)
            loop.llm_calls += 1
            messages.append(result.message)
            if not result.tool_calls:
                loop.reply = result.content
                loop.finished = True
                return loop
            for call in result.tool_calls:
                outcome = self.execute(call, agent=agent)
                messages.append(self.llm.tool_message(call, outcome))
                loop.calls.append((call, outcome))
        return loop


@dataclass
class AgentResult:
    final_reply: str
    trace: list
    metadata: dict = field(default_factory=dict)

    @property
    def tool_calls(self):
        """Executed shop calls in the shape eval/scorer.py expects."""
        return [
            {"name": event["name"], "arguments": event["arguments"], "result": event["result"]}
            for event in self.trace
            if event["type"] == "tool_call" and not event["malformed"]
        ]


class Agent(ABC):
    name = "base"
    # Role name to the tool names that role may call; used by the fairness tests.
    tools_by_role = {}

    def run(self, task, shop, llm, seed):
        llm.reset()
        trace = Trace()
        ctx = RunContext(task, shop, llm, seed, trace, load_policy())
        error = None
        llm_timeout = False
        try:
            final_reply, stop_reason = self._run(ctx)
        except BudgetExceeded:
            final_reply, stop_reason = "", "budget_exceeded"
        except LLMError as exc:
            final_reply, stop_reason, error = "", "llm_error", str(exc)
            llm_timeout = exc.timeout

        metadata = {
            "architecture": self.name,
            "task_id": task.get("task_id"),
            "model": llm.model,
            "seed": seed,
            "policy_sha256": ctx.policy_sha256,
            "stop_reason": stop_reason,
            "error": error,
            "llm_timeout": llm_timeout,
            "wall_time_s": round(trace.elapsed(), 4),
            "max_llm_calls": llm.max_calls,
            **llm.snapshot_totals(),
            **ctx.extra,
        }
        return AgentResult(final_reply=final_reply, trace=trace.events, metadata=metadata)

    @abstractmethod
    def _run(self, ctx):
        """Return (final_reply, stop_reason)."""
