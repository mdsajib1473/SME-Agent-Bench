"""ReAct agent with native tool calling.

Call the model with all tools, execute any tool calls, feed the results back,
and repeat until the model answers without a tool call or a limit is reached.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.base import Agent
from env.tools import TOOL_SCHEMAS

MAX_ITERATIONS = 15

ROLE_INSTRUCTIONS = (
    "Work step by step. Use the tools to look up the facts you need and to"
    " take any action the policy allows. When you are finished, reply to the"
    " customer in plain text without calling a tool; that reply is your final"
    " answer."
)


class ReActAgent(Agent):
    name = "react"
    role_instructions = ROLE_INSTRUCTIONS

    def __init__(self, max_iterations=MAX_ITERATIONS):
        self.max_iterations = max_iterations

    def _run(self, task, shop, llm, seed, prompt, trace):
        messages = [
            {"role": "system", "content": prompt.text},
            {"role": "user", "content": task["instruction"]},
        ]
        trace.message("system", prompt.text)
        trace.message("user", task["instruction"])

        for _ in range(self.max_iterations):
            result = llm.chat(messages, tools=TOOL_SCHEMAS, seed=seed)
            trace.llm_call(result)
            messages.append(result.message)
            if not result.tool_calls:
                return result.content, "final_answer"
            messages.extend(
                self.execute_tool_calls(shop, llm, result.tool_calls, trace)
            )
        return "", "max_iterations"
