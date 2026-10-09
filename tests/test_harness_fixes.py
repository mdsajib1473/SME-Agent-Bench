"""Shared prompt rules, prompt hash, schema example IDs, validator, dropped calls,
tool state checks, the plan-execute failure marker, and the supervisor handoff rules."""

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import plan_execute, supervisor
from agents.llm import LLMClient
from agents.prompts import (
    CONVERSATION_RULES,
    build_shared_prompt,
    build_system_prompt,
    load_policy,
    prompt_sha256,
    text_sha256,
)
from agents.registry import ARCHITECTURES, build_agent
from env.shop import Shop
from env.tools import TOOL_SCHEMAS, cancel_order, issue_refund
from eval import scorer, validate_tasks
from tests.fakes import FakeClient, response, system_text

CONFIG = {
    "ollama_base_url": "http://fake/v1",
    "temperature": 0.3,
    "max_llm_calls_per_task": 20,
    "request_timeout_s": 5,
}
USER = [{"role": "user", "content": "hi"}]


class TestSharedPrompt:
    def test_conversation_rules_follow_the_policy(self):
        shared = build_shared_prompt(load_policy())
        assert shared.index("</policy>") < shared.index(CONVERSATION_RULES)
        assert shared.endswith(CONVERSATION_RULES)
        for phrase in (
            "single message conversation",
            "never ask the customer to confirm",
            "Never guess an order ID or a product ID",
            "Only state actions you have actually performed",
            "language style",
        ):
            assert phrase in CONVERSATION_RULES

    def test_prompt_hash_covers_the_whole_shared_text(self):
        policy = load_policy()
        prompt = build_system_prompt("Role.")
        assert prompt.prompt_sha256 == prompt_sha256(policy) == text_sha256(build_shared_prompt(policy))
        assert prompt.prompt_sha256 != prompt.policy_sha256

    @pytest.mark.parametrize("name", list(ARCHITECTURES))
    def test_prompt_hash_is_in_every_architecture_metadata(self, name):
        llm = LLMClient("fake", config=CONFIG, client=FakeClient(lambda r: response('{"steps": []}')))
        with Shop() as shop:
            result = build_agent(name).run({"task_id": "T", "instruction": "hi"}, shop, llm, seed=0)
        assert result.metadata["prompt_sha256"] == prompt_sha256(load_policy())


class TestSchemaExamples:
    def _seeded_values(self):
        with Shop() as shop:
            ids = {row[0] for row in shop.connection.execute("SELECT order_id FROM orders")}
            ids |= {row[0] for row in shop.connection.execute("SELECT product_id FROM products")}
            ids |= {row[0] for row in shop.connection.execute("SELECT code FROM coupons")}
        return ids

    def test_no_real_id_appears_in_any_tool_schema(self):
        text = json.dumps(TOOL_SCHEMAS)
        for value in self._seeded_values():
            assert value not in text, value

    def test_no_digit_id_pattern_appears_in_tool_schemas(self):
        text = json.dumps(TOOL_SCHEMAS)
        assert not re.search(r"\b(ORD|PRD|CUS|REF|TKT|QUO)-\d", text)
        assert "ORD-NNNN" in text and "PRD-NNN" in text


class TestValidator:
    def test_validator_passes_with_the_empty_reply_rule(self, capsys):
        assert validate_tasks.main([]) == 0
        out = capsys.readouterr().out
        assert "oracle success rate: 100%" in out
        assert "silent oracle success rate: 0%" in out

    def test_oracle_passes_and_silent_oracle_fails_every_task(self):
        from eval import scorer

        for task in scorer.load_tasks():
            assert validate_tasks.run_oracle(task)["success"], task["task_id"]
            silent = validate_tasks.run_oracle(task, silent=True)
            assert silent["empty_reply"] and not silent["success"], task["task_id"]


class TestDroppedAtClient:
    def _result(self, content, completion_tokens, finish_reason="stop", tool_calls=None):
        message = SimpleNamespace(content=content, tool_calls=tool_calls)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=completion_tokens),
        )

    @pytest.mark.parametrize("finish_reason", ["stop", "length"])
    def test_tokens_without_content_or_calls_count_as_dropped(self, finish_reason):
        llm = LLMClient("fake", config=CONFIG, client=FakeClient([self._result("", 60, finish_reason)]))
        result = llm.chat(USER, seed=0)
        assert result.dropped
        assert llm.totals["dropped_tool_calls"] == 1

    @pytest.mark.parametrize(
        "content, tokens, finish_reason",
        [("Hello.", 60, "stop"), ("", 0, "stop"), ("", 60, "tool_calls"), ("   ", 0, "stop")],
    )
    def test_other_responses_are_not_dropped(self, content, tokens, finish_reason):
        llm = LLMClient("fake", config=CONFIG, client=FakeClient([self._result(content, tokens, finish_reason)]))
        assert not llm.chat(USER, seed=0).dropped
        assert llm.totals["dropped_tool_calls"] == 0

    def test_whitespace_only_content_with_tokens_is_dropped(self):
        llm = LLMClient("fake", config=CONFIG, client=FakeClient([self._result(" \n", 30)]))
        assert llm.chat(USER, seed=0).dropped

    def test_dropped_is_not_retried(self):
        fake = FakeClient([self._result("", 60)])
        llm = LLMClient("fake", config=CONFIG, client=fake)
        llm.chat(USER, seed=0)
        assert len(fake.requests) == 1
        assert llm.totals["llm_calls"] == 1


DRAFT_TASKS_PATH = ROOT / "tasks" / "tasks_draft.jsonl"


class TestLanguageRule:
    def test_shared_prompt_states_the_script_rule(self):
        shared = build_shared_prompt(load_policy())
        for phrase in (
            "if the customer wrote in English, reply in plain English",
            "If the customer wrote in Banglish (Bangla written in Latin letters), reply in Banglish",
            "never use Bangla script, Chinese or any other script",
        ):
            assert phrase in shared

    @pytest.mark.parametrize("name", list(ARCHITECTURES))
    def test_every_role_gets_the_same_shared_prompt(self, name):
        fake = FakeClient(lambda r: response('{"steps": []}'))
        llm = LLMClient("fake", config=CONFIG, client=fake)
        with Shop() as shop:
            build_agent(name).run({"task_id": "T", "instruction": "hi"}, shop, llm, seed=0)
        shared = build_shared_prompt(load_policy())
        assert fake.requests
        for request in fake.requests:
            assert system_text(request).startswith(shared)


def _first_order(shop, status):
    return shop.connection.execute(
        "SELECT order_id FROM orders WHERE status = ? ORDER BY order_id LIMIT 1", (status,)
    ).fetchone()[0]


class TestToolStateChecks:
    def test_second_refund_on_an_order_is_an_error(self):
        with Shop() as shop:
            order_id = _first_order(shop, "delivered")
            first = issue_refund(shop, order_id, 100, "bkash", "one")
            before = shop.snapshot()
            second = issue_refund(shop, order_id, 50, "nagad", "two")
            assert first["ok"]
            assert not second["ok"]
            assert "already" in second["error"]
            assert second["existing_refund_id"] == first["refund_id"]
            assert shop.snapshot() == before

    def test_refunds_on_different_orders_still_succeed(self):
        with Shop() as shop:
            orders = [row[0] for row in shop.connection.execute(
                "SELECT order_id FROM orders WHERE status = 'delivered' ORDER BY order_id LIMIT 2"
            )]
            results = [issue_refund(shop, order_id, 100, "bkash", "late") for order_id in orders]
            assert [r["ok"] for r in results] == [True, True]

    def test_refund_check_runs_after_the_order_lookup(self):
        with Shop() as shop:
            result = issue_refund(shop, "ORD-9999", 100, "bkash", "late")
            assert not result["ok"]
            assert "no order found" in result["error"]

    def test_cancelling_an_already_cancelled_order_is_an_error(self):
        with Shop() as shop:
            order_id = _first_order(shop, "cancelled")
            before = shop.snapshot()
            result = cancel_order(shop, order_id, "customer asked again")
            assert not result["ok"]
            assert "already cancelled" in result["error"]
            assert shop.snapshot() == before

    def test_second_cancel_of_the_same_order_is_an_error(self):
        with Shop() as shop:
            order_id = _first_order(shop, "pending")
            first = cancel_order(shop, order_id, "changed mind")
            second = cancel_order(shop, order_id, "changed mind")
            assert first["ok"] and first["previous_status"] == "pending"
            assert not second["ok"]

    def test_policy_is_still_not_enforced(self):
        with Shop() as shop:
            assert cancel_order(shop, _first_order(shop, "shipped"), "late")["ok"]
            assert issue_refund(shop, _first_order(shop, "pending"), 100, "card", "x")["ok"]

    def test_gold_replays_oracle_and_validators_pass_on_every_task_file(self, capsys):
        tasks = scorer.load_tasks() + scorer.load_tasks(DRAFT_TASKS_PATH)
        assert validate_tasks.replay_check(tasks) == []
        assert validate_tasks.main([str(scorer.TASKS_PATH), str(DRAFT_TASKS_PATH)]) == 0
        assert "validation: PASS" in capsys.readouterr().out


class TestPlanExecuteFailureMarker:
    @pytest.mark.parametrize(
        "report, failed",
        [
            ("FAILED: the order is shipped.", True),
            ("Failed: no order with that ID.", True),
            ("failed to find the order", True),
            ("**FAILED** order not found", True),
            ("Step 1 summary. Outcome: this step FAILED because the order is shipped.", True),
            ("DONE. The customer reported failed payments; a billing ticket was opened.", False),
            ("DONE: Failed payments were reported, routed to billing.", False),
            ("DONE: payment failed twice.", False),
            ("", False),
        ],
    )
    def test_report_failed(self, report, failed):
        assert plan_execute.report_failed(report) is failed

    def test_billing_report_about_failed_payments_does_not_replan(self):
        report = (
            "DONE. The customer reported failed payments on order ORD-1008, so I"
            " opened a billing ticket with normal priority."
        )
        planner_requests = []

        def handler(request):
            text = system_text(request)
            if "You are the planner" in text:
                planner_requests.append(request)
                return response(json.dumps({"steps": ["Open a billing ticket for the payment problem."]}))
            if "You are the executor" in text:
                return response(report)
            return response("Reply.")

        llm = LLMClient("fake", config=CONFIG, client=FakeClient(handler))
        task = {"task_id": "T", "instruction": "My payment failed twice on ORD-1008, please check."}
        with Shop() as shop:
            result = build_agent("plan_execute").run(task, shop, llm, seed=0)
        meta = result.metadata
        assert meta["replans"] == 0
        assert meta["steps_failed"] == 0
        assert meta["planner_calls"] == 1 == len(planner_requests)
        steps = [e for e in result.trace if e["type"] == "step"]
        assert [s["status"] for s in steps] == ["done"]


class TestSupervisorHandoffRules:
    def test_role_carries_the_new_rules(self):
        role = supervisor.SUPERVISOR_ROLE
        assert "pass on the customer's request word for word" in role
        assert "copy word for word every identifier the customer gave" in role
        assert "Never paste policy text into an instruction" in role
        assert "Never tell a specialist which tool to call" in role

    def test_role_keeps_the_other_rules(self):
        role = supervisor.SUPERVISOR_ROLE
        for kept in (
            "cannot see any order",
            "never the customer's message",
            "Say exactly what you want the specialist to do",
            "send each task to the specialist that has the tool for it",
            f"at most {supervisor.MAX_DELEGATIONS} times",
            "only on what the specialists reported",
            "reply to the customer without calling a tool",
        ):
            assert kept in role


class TestWrongScriptCount:
    @pytest.mark.parametrize(
        "text, scripts",
        [
            ("Your refund of 500 BDT is done.", set()),
            ("Apnar order ta cancel kora hoyeche. Dhonnobad!", set()),
            ("Refund of ৳500 sent, café résumé µ", set()),
            ("আপনার order", {"BENGALI"}),
            ("Your order 已经 shipped.", {"CJK"}),
            ("Price ১২০ taka", {"BENGALI"}),
            ("Привет", {"CYRILLIC"}),
            ("", set()),
            (None, set()),
        ],
    )
    def test_non_latin_scripts(self, text, scripts):
        from scripts.summarize_run import non_latin_scripts

        assert non_latin_scripts(text) == scripts
