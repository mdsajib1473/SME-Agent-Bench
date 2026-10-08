"""ReAct agent with native tool calling.

Call the model with all tools, execute any tool calls, feed the results back,
and repeat until the model answers without a tool call or a limit is reached.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.base import ALL_TOOL_NAMES, Agent, schemas_for

MAX_ITERATIONS = 15

ROLE_INSTRUCTIONS = (
    "Work step by step. Use the tools to look up the facts you need and to"
    " take any action the policy allows. When you are finished, reply to the"
    " customer in plain text without calling a tool; that reply is your final"
    " answer."
)


class ReActAgent(Agent):
    name = "react"
    tools_by_role = {"react": ALL_TOOL_NAMES}

    def __init__(self, max_iterations=MAX_ITERATIONS):
        self.max_iterations = max_iterations

    def _run(self, ctx):
        messages = ctx.open_conversation(
            self.name, ROLE_INSTRUCTIONS, ctx.task["instruction"]
        )
        loop = ctx.tool_loop(
            messages, schemas_for(ALL_TOOL_NAMES), self.max_iterations, agent=self.name
        )
        if loop.finished:
            return loop.reply, "final_answer"
        return "", "max_iterations"
