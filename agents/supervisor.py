"""Supervisor multi-agent architecture.

The supervisor has four tools, one per specialist (OrderAgent, RefundAgent,
SalesAgent, SupportAgent), each taking a single instruction. Calling one runs
that specialist: it sees only the instruction, runs a short tool-calling loop
with its own tools, and returns a text report. All information between agents
passes through those instructions and reports.

Budget: every agent draws on the one per-task LLM budget. A delegation is only
started when at least one call remains for the supervisor afterwards, so a
specialist cannot spend the call the supervisor needs to read its report.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.base import Agent, schemas_for

MAX_DELEGATIONS = 6
MAX_SPECIALIST_CALLS = 4
MAX_SUPERVISOR_TURNS = 15
SUPERVISOR_RESERVE = 1

SPECIALIST_TOOLS = {
    "OrderAgent": (
        "get_customer",
        "get_order",
        "list_customer_orders",
        "cancel_order",
        "update_order",
        "apply_coupon",
    ),
    "RefundAgent": ("get_order", "issue_refund", "lookup_policy"),
    "SalesAgent": ("search_products", "check_stock", "create_quote"),
    "SupportAgent": ("get_order", "route_ticket", "lookup_policy"),
}

SPECIALIST_DUTIES = {
    "OrderAgent": "customer and order lookups, cancellations, address and quantity changes, and coupons",
    "RefundAgent": "refunds",
    "SalesAgent": "product search, stock checks, and bulk quotes",
    "SupportAgent": "complaint tickets routed to the right department",
}

SPECIALIST_ROLE_TEMPLATE = (
    "You are the {name}, a specialist in a team led by a supervisor. You handle"
    " {duties}. You can only use your own tools: {tools}. You cannot do anything"
    " that needs another tool. You receive one instruction from the supervisor"
    " and never see the customer's own message. Never take an action the policy"
    " forbids. When you are finished, reply without calling a tool with a short"
    " report for the supervisor. Report only actions you actually performed with"
    " your tools, with exact IDs, amounts and statuses, and what you checked. If"
    " you could not do something, because it needs a tool you do not have, the"
    " policy forbids it, or a lookup failed, say so plainly; never claim it was"
    " done."
)

SPECIALIST_ROLES = {
    name: SPECIALIST_ROLE_TEMPLATE.format(
        name=name, duties=SPECIALIST_DUTIES[name], tools=", ".join(tools)
    )
    for name, tools in SPECIALIST_TOOLS.items()
}

SUPERVISOR_ROLE = (
    "You are the supervisor of a team of specialists. You cannot use shop tools"
    " and you cannot see any order, customer or product data. Hand work to a"
    " specialist by calling the tool with its name; each takes one instruction:\n"
    + "\n".join(
        f"- {name}: {SPECIALIST_DUTIES[name]} (its tools: {', '.join(tools)})"
        for name, tools in SPECIALIST_TOOLS.items()
    )
    + "\n\nA specialist sees only your instruction, never the customer's message."
    " In every instruction, copy word for word every identifier the customer"
    " gave: order ID, phone number, product names, quantities, district,"
    " amounts, and the customer's reason. Say exactly what you want the"
    " specialist to do. A specialist can only use its own tools, so send each"
    " task to the specialist that has the tool for it."
    f" You may delegate at most {MAX_DELEGATIONS} times in total. Base your final"
    " reply only on what the specialists reported, and never state a fact or an"
    " action that no report contains. When you are done, reply to the customer"
    " without calling a tool; that reply is your final answer."
)


def _specialist_schema(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Ask the {name} to handle {SPECIALIST_DUTIES[name]}."
            " It sees only your instruction and returns a short report.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "What you want done, with every identifier the"
                        " customer gave, copied word for word.",
                    },
                },
                "required": ["instruction"],
            },
        },
    }


SPECIALIST_SCHEMAS = [_specialist_schema(name) for name in SPECIALIST_TOOLS]


class SupervisorAgent(Agent):
    name = "supervisor"
    tools_by_role = {"supervisor": tuple(SPECIALIST_TOOLS), **SPECIALIST_TOOLS}

    def _run(self, ctx):
        ctx.extra.update(delegations=0, delegations_refused=0)
        messages = ctx.open_conversation("supervisor", SUPERVISOR_ROLE, ctx.task["instruction"])
        for _ in range(MAX_SUPERVISOR_TURNS):
            result = ctx.chat(messages, tools=SPECIALIST_SCHEMAS, agent="supervisor")
            messages.append(result.message)
            if not result.tool_calls:
                return result.content, "final_answer"
            for call in result.tool_calls:
                outcome = self._handle(ctx, call)
                messages.append(ctx.llm.tool_message(call, outcome))
        return "", "max_turns"

    def _refuse(self, ctx, specialist, instruction, reason):
        ctx.extra["delegations_refused"] += 1
        ctx.trace.add(
            "delegation",
            specialist=specialist,
            instruction=instruction,
            report=None,
            refused=reason,
            llm_calls=0,
            tool_calls=[],
        )
        return {"ok": False, "error": reason}

    def _handle(self, ctx, call):
        error = ctx.llm.check_tool_call(call)
        if error is not None:
            ctx.trace.tool_call(call, error, 0.0, agent="supervisor")
            return error

        specialist = call.name
        instruction = call.arguments.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            return self._refuse(ctx, specialist, instruction, "instruction must be a non-empty string")
        if ctx.extra["delegations"] >= MAX_DELEGATIONS:
            return self._refuse(
                ctx, specialist, instruction,
                f"delegation limit of {MAX_DELEGATIONS} reached; reply to the customer now",
            )
        max_calls = min(MAX_SPECIALIST_CALLS, ctx.llm.calls_remaining - SUPERVISOR_RESERVE)
        if max_calls < 1:
            return self._refuse(
                ctx, specialist, instruction,
                "call budget too low to delegate; reply to the customer now",
            )

        ctx.extra["delegations"] += 1
        report, loop = self._run_specialist(ctx, specialist, instruction.strip(), max_calls)
        ctx.trace.add(
            "delegation",
            index=ctx.extra["delegations"],
            specialist=specialist,
            instruction=instruction.strip(),
            report=report,
            refused=None,
            finished=loop.finished,
            llm_calls=loop.llm_calls,
            tool_calls=[
                {"name": c.name, "ok": r.get("ok") is True} for c, r in loop.calls
            ],
        )
        return {"ok": True, "specialist": specialist, "report": report}

    def _run_specialist(self, ctx, specialist, instruction, max_calls):
        messages = ctx.open_conversation(specialist, SPECIALIST_ROLES[specialist], instruction)
        loop = ctx.tool_loop(
            messages, schemas_for(SPECIALIST_TOOLS[specialist]), max_calls, agent=specialist
        )
        report = loop.reply.strip()
        if loop.finished and report:
            return report, loop
        actions = ", ".join(
            f"{c.name} {'ok' if r.get('ok') is True else 'error'}" for c, r in loop.calls
        ) or "none"
        if loop.finished:
            return f"{specialist} returned an empty report. Tool calls made: {actions}.", loop
        return (
            f"{specialist} stopped after {loop.llm_calls} LLM calls without a report."
            f" Tool calls made: {actions}."
        ), loop
