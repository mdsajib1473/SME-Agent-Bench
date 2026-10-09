"""Fairness across architectures, plus plan-execute and supervisor limits.

Every arm must use the same LLMClient, the same shared policy text, the same
call budget, and between its roles the same 12 tools. All tests use a fake
endpoint, so they run without Ollama.
"""

import inspect
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import plan_execute, supervisor
from agents.base import Agent
from agents.llm import LLMClient, load_config
from agents.prompts import build_shared_prompt, load_policy, policy_sha256
from agents.registry import ARCHITECTURES, build_agent
from env.shop import Shop
from env.tools import TOOL_FUNCTIONS, TOOL_SCHEMAS
from tests.fakes import FakeClient, native_call, response, system_text, tool_names

TASK = {
    "task_id": "FAKE-01",
    "instruction": "My order ORD-1008, phone 01339537672, came in the wrong colour. Please refund.",
}
ONE_STEP_PLAN = json.dumps({"steps": ["Look up order ORD-1008."]})


@pytest.fixture
def shop():
    with Shop() as instance:
        yield instance


def make_llm(handler, max_calls=None):
    config = dict(load_config())
    if max_calls is not None:
        config["max_llm_calls_per_task"] = max_calls
    fake = FakeClient(handler)
    return LLMClient("fake-model", config=config, client=fake), fake


def is_role(request, marker):
    return marker in system_text(request)


def last_role(request):
    return request["messages"][-1]["role"]


def first_tool_call(request):
    """One plausible call for whatever tools this request offers."""
    names = tool_names(request)
    if "SupportAgent" in names:
        return native_call(
            "SupportAgent", {"instruction": "Look up order ORD-1008, phone 01339537672."}
        )
    if "get_order" in names:
        return native_call("get_order", {"order_id": "ORD-1008"})
    return native_call(names[0], {})


def polite_handler(request):
    """Each conversation makes one tool call, then answers."""
    if is_role(request, "You are the planner"):
        return response(ONE_STEP_PLAN)
    if not tool_names(request):
        return response("Your request is handled.")
    if last_role(request) == "user":
        return response(tool_calls=[first_tool_call(request)])
    return response("DONE: finished.")


def relentless_handler(request):
    """Every conversation with tools keeps calling tools and never answers."""
    if is_role(request, "You are the planner"):
        return response(json.dumps({"steps": [f"Step {n}" for n in range(1, 11)]}))
    if not tool_names(request):
        return response("Your request is handled.")
    return response(tool_calls=[first_tool_call(request)])


class TestRegistry:
    def test_three_architectures_by_name(self):
        assert set(ARCHITECTURES) == {"react", "plan_execute", "supervisor"}
        for name in ARCHITECTURES:
            assert build_agent(name).name == name

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError):
            build_agent("swarm")


class TestFairness:
    @pytest.mark.parametrize("name", list(ARCHITECTURES))
    def test_architecture_shares_run_and_never_builds_its_own_client(self, name):
        cls = ARCHITECTURES[name]
        assert issubclass(cls, Agent)
        assert cls.run is Agent.run
        source = Path(inspect.getsourcefile(cls)).read_text(encoding="utf-8")
        assert "openai" not in source.lower()
        assert "LLMClient(" not in source

    @pytest.mark.parametrize("name", list(ARCHITECTURES))
    def test_same_client_policy_and_budget(self, shop, name):
        llm, fake = make_llm(polite_handler)
        result = build_agent(name).run(TASK, shop, llm, seed=0)
        meta = result.metadata
        shared = build_shared_prompt(load_policy())

        assert type(llm) is LLMClient
        assert len(fake.requests) == meta["llm_calls"] > 0
        assert meta["max_llm_calls"] == load_config()["max_llm_calls_per_task"] == 20
        assert meta["policy_sha256"] == policy_sha256(load_policy())
        for request in fake.requests:
            assert request["messages"][0]["role"] == "system"
            assert system_text(request).startswith(shared)
            assert request["extra_body"] == {"seed": 0}
            assert request["max_tokens"] == load_config()["max_tokens_per_call"]
            assert request["temperature"] == load_config()["temperature"]
        for key in ("tool_calls", "malformed_tool_calls", "text_tool_calls",
                    "prompt_tokens", "completion_tokens", "llm_latency_s"):
            assert key in meta
        assert meta["stop_reason"] == "final_answer"

    @pytest.mark.parametrize("name", list(ARCHITECTURES))
    def test_tool_union_is_the_full_set(self, name):
        declared = set()
        for tools in build_agent(name).tools_by_role.values():
            declared.update(tools)
        assert declared - set(supervisor.SPECIALIST_TOOLS) == set(TOOL_FUNCTIONS)
        assert len(TOOL_FUNCTIONS) == 12

    @pytest.mark.parametrize("name", list(ARCHITECTURES))
    def test_tools_sent_are_the_shared_schemas(self, shop, name):
        llm, fake = make_llm(polite_handler)
        build_agent(name).run(TASK, shop, llm, seed=0)
        for request in fake.requests:
            for schema in request.get("tools") or []:
                assert schema in TOOL_SCHEMAS or schema in supervisor.SPECIALIST_SCHEMAS

    def test_specialist_tool_sets_match_the_design(self):
        assert supervisor.SPECIALIST_TOOLS == {
            "OrderAgent": ("get_customer", "get_order", "list_customer_orders",
                           "cancel_order", "update_order", "apply_coupon"),
            "RefundAgent": ("get_order", "issue_refund", "lookup_policy"),
            "SalesAgent": ("search_products", "check_stock", "create_quote"),
            "SupportAgent": ("get_order", "route_ticket", "lookup_policy"),
        }

    def test_specialists_only_get_their_own_tools(self, shop):
        llm, fake = make_llm(polite_handler)
        build_agent("supervisor").run(TASK, shop, llm, seed=0)
        for request in fake.requests:
            for name, tools in supervisor.SPECIALIST_TOOLS.items():
                if is_role(request, f"You are the {name}"):
                    assert tool_names(request) == list(tools)


class TestBudget:
    @pytest.mark.parametrize("name", list(ARCHITECTURES))
    def test_never_exceeds_the_shared_budget(self, shop, name):
        llm, fake = make_llm(relentless_handler)
        result = build_agent(name).run(TASK, shop, llm, seed=0)
        assert len(fake.requests) == result.metadata["llm_calls"] <= 20

    @pytest.mark.parametrize(
        "name, stop_reason",
        [("react", "budget_exceeded"), ("plan_execute", "final_answer"), ("supervisor", "budget_exceeded")],
    )
    def test_small_budget_stops_every_architecture(self, shop, name, stop_reason):
        llm, fake = make_llm(relentless_handler, max_calls=5)
        result = build_agent(name).run(TASK, shop, llm, seed=0)
        assert len(fake.requests) == 5
        assert result.metadata["llm_calls"] == 5
        assert result.metadata["stop_reason"] == stop_reason
        if stop_reason == "budget_exceeded":
            assert result.final_reply == ""

    def test_supervisor_spends_the_whole_budget_then_stops(self, shop):
        llm, fake = make_llm(relentless_handler)
        result = build_agent("supervisor").run(TASK, shop, llm, seed=0)
        assert len(fake.requests) == 20
        assert result.metadata["stop_reason"] == "budget_exceeded"


class TestPlanExecute:
    def test_invalid_plan_is_retried_once(self, shop):
        attempts = []

        def handler(request):
            if is_role(request, "You are the planner"):
                attempts.append(request)
                return response("Sure, here is my plan: first look it up." if len(attempts) == 1 else ONE_STEP_PLAN)
            if is_role(request, "You are the executor"):
                return response("DONE: order found.")
            return response("Reply.")

        llm, _ = make_llm(handler)
        result = build_agent("plan_execute").run(TASK, shop, llm, seed=0)
        assert result.metadata["planner_calls"] == 2
        assert result.metadata["plan_valid"] is True
        assert result.metadata["steps_run"] == 1
        assert "not a valid plan" in attempts[1]["messages"][-1]["content"]
        plans = [e for e in result.trace if e["type"] == "plan"]
        assert [p["valid"] for p in plans] == [False, True]

    def test_two_invalid_plans_fall_back_to_one_step(self, shop):
        executor_requests = []

        def handler(request):
            if is_role(request, "You are the planner"):
                return response("no plan")
            if is_role(request, "You are the executor"):
                executor_requests.append(request)
                return response("DONE: handled.")
            return response("Reply.")

        llm, _ = make_llm(handler)
        result = build_agent("plan_execute").run(TASK, shop, llm, seed=0)
        assert result.metadata["planner_calls"] == 2
        assert result.metadata["plan_valid"] is False
        assert result.metadata["steps_run"] == 1
        assert plan_execute.FALLBACK_STEP in executor_requests[0]["messages"][-1]["content"]

    def test_only_one_replan_per_task(self, shop):
        planner_requests = []

        def handler(request):
            if is_role(request, "You are the planner"):
                planner_requests.append(request)
                return response(json.dumps({"steps": ["Step A", "Step B"]}))
            if is_role(request, "You are the executor"):
                return response("FAILED: the order cannot be found.")
            return response("Reply.")

        llm, fake = make_llm(handler)
        result = build_agent("plan_execute").run(TASK, shop, llm, seed=0)
        meta = result.metadata
        assert meta["replans"] == 1
        assert meta["planner_calls"] == 2
        assert meta["steps_run"] == 2
        assert meta["steps_failed"] == 2
        assert len(fake.requests) == 5
        assert "FAILED: the order cannot be found." in planner_requests[1]["messages"][-1]["content"]
        assert result.final_reply == "Reply."

    def test_tool_error_marks_the_step_failed(self, shop):
        def handler(request):
            if is_role(request, "You are the planner"):
                return response(ONE_STEP_PLAN)
            if is_role(request, "You are the executor"):
                if last_role(request) == "user":
                    return response(tool_calls=[native_call("get_order", {"order_id": "ORD-9999"})])
                return response("DONE: looked it up.")
            return response("Reply.")

        llm, _ = make_llm(handler)
        result = build_agent("plan_execute").run(TASK, shop, llm, seed=0)
        steps = [e for e in result.trace if e["type"] == "step"]
        assert steps[0]["status"] == "failed"
        assert "error" in steps[0]["failure"]
        assert result.metadata["replans"] == 1

    def test_step_results_never_show_call_syntax(self):
        from agents.llm import ToolCall, find_text_tool_calls

        call = ToolCall("c1", "get_order", {"order_id": "ORD-1008"}, "{}", "native")
        record = plan_execute.StepRecord(1, "Look up the order.", "DONE: found.", False, None,
                                         [(call, {"ok": True, "order": {"order_id": "ORD-1008"}})])
        text = plan_execute.format_results([record])
        assert "ORD-1008" in text
        for marker in ("->", "{", "[", '"', "Outcome:", "Executor report:", "DONE", "FAILED", "failed"):
            assert marker not in text
        assert find_text_tool_calls(text)[0] == []

    def test_failed_anywhere_in_the_report_marks_the_step_failed(self, shop):
        def handler(request):
            if is_role(request, "You are the planner"):
                return response(ONE_STEP_PLAN)
            if is_role(request, "You are the executor"):
                return response("Step 1 summary. Outcome: this step FAILED because the order is shipped.")
            return response("Reply.")

        llm, _ = make_llm(handler)
        result = build_agent("plan_execute").run(TASK, shop, llm, seed=0)
        steps = [e for e in result.trace if e["type"] == "step"]
        assert steps[0]["status"] == "failed"
        assert result.metadata["replans"] == 1

    def test_responder_gets_the_customer_message_last(self, shop):
        responder = []

        def handler(request):
            if is_role(request, "You are the planner"):
                return response(json.dumps({"steps": []}))
            responder.append(request)
            return response("Reply.")

        llm, _ = make_llm(handler)
        build_agent("plan_execute").run(TASK, shop, llm, seed=0)
        content = responder[0]["messages"][-1]["content"]
        assert content.endswith(TASK["instruction"])
        assert "plain text only" in content
        for phrase in ("language style", "Banglish", "reply in English"):
            assert phrase not in content

    def test_planner_is_told_not_to_name_tools_or_arguments(self):
        role = plan_execute.PLANNER_ROLE
        assert "Never write tool names, tool arguments, IDs" in role
        for name in TOOL_FUNCTIONS:
            assert name not in role

    def test_parse_plan_accepts_fenced_json_and_bare_lists(self):
        assert plan_execute.parse_plan('```json\n{"steps": ["a", "b"]}\n```') == (["a", "b"], None)
        assert plan_execute.parse_plan('["a"]') == (["a"], None)
        assert plan_execute.parse_plan('{"steps": [{"instruction": "a"}]}') == (["a"], None)
        assert plan_execute.parse_plan('{"steps": []}') == ([], None)
        assert plan_execute.parse_plan('{"plan": "a"}')[0] is None
        assert plan_execute.parse_plan('{"steps": [1]}')[0] is None


class TestSupervisor:
    def test_delegation_limit(self, shop):
        specialist_requests = []

        def handler(request):
            if "SupportAgent" in tool_names(request):
                return response(tool_calls=[first_tool_call(request)])
            specialist_requests.append(request)
            return response("Order ORD-1008 is delivered.")

        llm, _ = make_llm(handler)
        result = build_agent("supervisor").run(TASK, shop, llm, seed=0)
        assert result.metadata["delegations"] == supervisor.MAX_DELEGATIONS == 6
        assert len(specialist_requests) == 6
        assert result.metadata["delegations_refused"] >= 1
        refused = [e for e in result.trace if e["type"] == "delegation" and e["refused"]]
        assert "delegation limit" in refused[0]["refused"]

    def test_specialist_sees_only_the_instruction_and_handoff_is_logged(self, shop):
        def handler(request):
            if is_role(request, "You are the supervisor"):
                if last_role(request) == "user":
                    return response(tool_calls=[native_call(
                        "RefundAgent", {"instruction": "Check ORD-1008."})])
                return response("Final reply to the customer.")
            return response("ORD-1008 is delivered, total 2920.")

        llm, fake = make_llm(handler)
        result = build_agent("supervisor").run(TASK, shop, llm, seed=0)
        specialist = fake.requests[1]
        assert [m["role"] for m in specialist["messages"]] == ["system", "user"]
        assert specialist["messages"][1]["content"] == "Check ORD-1008."
        assert all(TASK["instruction"] not in str(m["content"]) for m in specialist["messages"])
        tool_reply = json.loads(fake.requests[2]["messages"][-1]["content"])
        assert tool_reply["report"] == "ORD-1008 is delivered, total 2920."
        delegations = [e for e in result.trace if e["type"] == "delegation"]
        assert delegations == [
            {**delegations[0], "specialist": "RefundAgent", "instruction": "Check ORD-1008.",
             "report": "ORD-1008 is delivered, total 2920.", "refused": None}
        ]
        assert result.final_reply == "Final reply to the customer."

    def test_unknown_specialist_is_malformed_without_a_call(self, shop):
        def handler(request):
            if last_role(request) == "user":
                return response(tool_calls=[native_call("LawyerAgent", {"instruction": "Sue them."})])
            return response("Sorry.")

        llm, fake = make_llm(handler)
        result = build_agent("supervisor").run(TASK, shop, llm, seed=0)
        assert len(fake.requests) == 2
        assert result.metadata["delegations"] == 0
        assert result.metadata["malformed_tool_calls"] == 1

    def test_empty_instruction_is_refused(self, shop):
        def handler(request):
            if last_role(request) == "user":
                return response(tool_calls=[native_call("OrderAgent", {"instruction": "  "})])
            return response("Sorry.")

        llm, fake = make_llm(handler)
        result = build_agent("supervisor").run(TASK, shop, llm, seed=0)
        assert len(fake.requests) == 2
        assert result.metadata["delegations_refused"] == 1

    def test_supervisor_has_one_tool_per_specialist(self, shop):
        llm, fake = make_llm(polite_handler)
        build_agent("supervisor").run(TASK, shop, llm, seed=0)
        supervisor_requests = [r for r in fake.requests if is_role(r, "You are the supervisor")]
        assert supervisor_requests
        for request in supervisor_requests:
            assert tool_names(request) == ["OrderAgent", "RefundAgent", "SalesAgent", "SupportAgent"]
            for schema in request["tools"]:
                assert list(schema["function"]["parameters"]["properties"]) == ["instruction"]

    def test_role_prompts_carry_the_handoff_rules(self):
        role = supervisor.SUPERVISOR_ROLE
        assert "cannot see any order" in role
        assert "word for word" in role
        assert "only on what the specialists reported" in role
        for text in supervisor.SPECIALIST_ROLES.values():
            assert "can only use your own tools" in text
            assert "Report only actions you actually performed" in text
            assert "say so plainly" in text


class TestDroppedToolCalls:
    @pytest.mark.parametrize("name", list(ARCHITECTURES))
    def test_dropped_output_is_counted_and_marked_in_every_architecture(self, shop, name):
        def handler(request):
            if is_role(request, "You are the planner"):
                return response(ONE_STEP_PLAN)
            return response("", completion_tokens=57)

        llm, _ = make_llm(handler)
        result = build_agent(name).run(TASK, shop, llm, seed=0)
        dropped_events = [e for e in result.trace if e["type"] == "llm_call" and e["dropped"]]
        assert result.metadata["dropped_tool_calls"] == len(dropped_events) >= 1
        assert result.final_reply == ""

    def test_empty_response_without_tokens_is_not_dropped(self, shop):
        llm, _ = make_llm(lambda request: response("", completion_tokens=0))
        result = build_agent("react").run(TASK, shop, llm, seed=0)
        assert result.metadata["dropped_tool_calls"] == 0
