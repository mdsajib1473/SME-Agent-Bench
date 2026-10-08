"""Abstract agent.

Agent.run owns everything the arms must share: resetting the LLM budget, the
system prompt, budget and endpoint failure handling, the trace, and the
metadata. Each architecture only implements _run, the reasoning loop.
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
from agents.prompts import build_system_prompt


class Trace:
    """Ordered record of messages, LLM calls and tool calls, timed from start."""

    def __init__(self):
        self.started = time.perf_counter()
        self.events = []

    def _add(self, event_type, **payload):
        event = {"t": round(time.perf_counter() - self.started, 4), "type": event_type}
        event.update(payload)
        self.events.append(event)
        return event

    def message(self, role, content):
        return self._add("message", role=role, content=content)

    def llm_call(self, result):
        return self._add(
            "llm_call",
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

    def tool_call(self, call, result, latency_s):
        return self._add(
            "tool_call",
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
class AgentResult:
    final_reply: str
    trace: list
    metadata: dict = field(default_factory=dict)

    @property
    def tool_calls(self):
        """Executed calls in the shape eval/scorer.py expects."""
        return [
            {"name": event["name"], "arguments": event["arguments"], "result": event["result"]}
            for event in self.trace
            if event["type"] == "tool_call" and not event["malformed"]
        ]


class Agent(ABC):
    name = "base"
    role_instructions = None

    def run(self, task, shop, llm, seed):
        llm.reset()
        prompt = build_system_prompt(self.role_instructions)
        trace = Trace()
        error = None
        try:
            final_reply, stop_reason = self._run(task, shop, llm, seed, prompt, trace)
        except BudgetExceeded:
            final_reply, stop_reason = "", "budget_exceeded"
        except LLMError as exc:
            final_reply, stop_reason, error = "", "llm_error", str(exc)

        metadata = {
            "architecture": self.name,
            "task_id": task.get("task_id"),
            "model": llm.model,
            "seed": seed,
            "policy_sha256": prompt.policy_sha256,
            "stop_reason": stop_reason,
            "error": error,
            "wall_time_s": round(trace.elapsed(), 4),
            "max_llm_calls": llm.max_calls,
            **llm.snapshot_totals(),
        }
        return AgentResult(final_reply=final_reply, trace=trace.events, metadata=metadata)

    def execute_tool_calls(self, shop, llm, calls, trace):
        """Run calls in order; returns the tool messages to append to history."""
        messages = []
        for call in calls:
            started = time.perf_counter()
            result = llm.execute_tool_call(shop, call)
            trace.tool_call(call, result, time.perf_counter() - started)
            messages.append(llm.tool_message(call, result))
        return messages

    @abstractmethod
    def _run(self, task, shop, llm, seed, prompt, trace):
        """Return (final_reply, stop_reason)."""
