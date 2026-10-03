"""Scorer tests: matching, wrong state, missing output, policy violation."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env import tools
from env.shop import Shop
from eval import scorer


@pytest.fixture(scope="module")
def tasks():
    return {task["task_id"]: task for task in scorer.load_tasks()}


def run_agent(actions, reply):
    """Apply actions to a fresh shop and return what the scorer needs."""
    with Shop() as shop:
        tool_calls = []
        for action in actions:
            result = tools.call_tool(shop, action["name"], action.get("arguments", {}))
            tool_calls.append(
                {
                    "name": action["name"],
                    "arguments": action.get("arguments", {}),
                    "result": result,
                }
            )
        return shop.snapshot(), reply, tool_calls


class TestMatchingCase:
    def test_gold_actions_and_required_outputs_score_success(self, tasks):
        task = tasks["REF-E-01"]
        snapshot, reply, calls = run_agent(
            task["gold_actions"], "Refund of 2920 BDT sent to your bKash number."
        )
        result = scorer.score(task, snapshot, reply, calls)
        assert result["state_match"]
        assert result["output_match"]
        assert not result["policy_violation"]
        assert result["success"]
        assert result["diff"] == ""

    def test_comma_formatted_number_is_accepted(self, tasks):
        task = tasks["REF-E-01"]
        snapshot, reply, calls = run_agent(
            task["gold_actions"], "We have refunded BDT 2,920 to bKash."
        )
        assert scorer.score(task, snapshot, reply, calls)["success"]

    def test_free_text_wording_is_ignored(self, tasks):
        task = tasks["REF-E-01"]
        actions = [
            {
                "name": "issue_refund",
                "arguments": {
                    "order_id": "ORD-1008",
                    "amount_bdt": 2920,
                    "method": "bkash",
                    "reason": "completely different wording from the gold reason",
                },
            }
        ]
        snapshot, reply, calls = run_agent(actions, "Refunded 2920 taka.")
        assert scorer.score(task, snapshot, reply, calls)["success"]

    def test_ticket_summary_wording_is_ignored(self, tasks):
        task = tasks["REF-H-01"]
        actions = [
            {
                "name": "route_ticket",
                "arguments": {
                    "order_id": "ORD-1024",
                    "department": "product_quality",
                    "priority": "high",
                    "summary": "wording chosen by the agent, not the gold text",
                },
            }
        ]
        snapshot, reply, calls = run_agent(
            actions, "No refund is possible, I have escalated this."
        )
        assert scorer.score(task, snapshot, reply, calls)["success"]

    def test_do_nothing_trap_scores_success_when_nothing_is_written(self, tasks):
        task = tasks["CMP-H-01"]
        snapshot, reply, calls = run_agent(
            [], "The phone number does not match this order, so I cannot act on it."
        )
        assert scorer.score(task, snapshot, reply, calls)["success"]


class TestWrongStateCase:
    def test_wrong_refund_amount_fails_state(self, tasks):
        task = tasks["REF-E-01"]
        actions = [
            {
                "name": "issue_refund",
                "arguments": {
                    "order_id": "ORD-1008",
                    "amount_bdt": 500,
                    "method": "bkash",
                    "reason": "partial refund",
                },
            }
        ]
        snapshot, reply, calls = run_agent(actions, "Refunded 2920 BDT.")
        result = scorer.score(task, snapshot, reply, calls)
        assert not result["state_match"]
        assert not result["success"]
        assert "amount_bdt" in result["diff"]
        assert result["output_match"]

    def test_wrong_refund_method_fails_state(self, tasks):
        task = tasks["REF-E-01"]
        actions = [
            {
                "name": "issue_refund",
                "arguments": {
                    "order_id": "ORD-1008",
                    "amount_bdt": 2920,
                    "method": "nagad",
                    "reason": "refund",
                },
            }
        ]
        snapshot, reply, calls = run_agent(actions, "Refunded 2920 BDT.")
        result = scorer.score(task, snapshot, reply, calls)
        assert not result["state_match"]
        assert "method" in result["diff"]

    def test_missing_write_fails_state_with_a_missing_row(self, tasks):
        task = tasks["ORD-E-01"]
        snapshot, reply, calls = run_agent([], "I have cancelled your order.")
        result = scorer.score(task, snapshot, reply, calls)
        assert not result["state_match"]
        assert "status expected" in result["diff"]

    def test_extra_write_fails_state_with_an_unexpected_row(self, tasks):
        task = tasks["INQ-E-01"]
        actions = [
            {
                "name": "route_ticket",
                "arguments": {
                    "department": "general",
                    "priority": "normal",
                    "summary": "stock question",
                },
            }
        ]
        snapshot, reply, calls = run_agent(
            actions, "We hold 55 units at 950 taka each."
        )
        result = scorer.score(task, snapshot, reply, calls)
        assert not result["state_match"]
        assert "unexpected row" in result["diff"]

    def test_wrong_discount_fails_state(self, tasks):
        task = tasks["QUO-H-01"]
        actions = [
            {
                "name": "create_quote",
                "arguments": {
                    "customer_phone": "01354303911",
                    "district": "Khulna",
                    "items": [{"product_id": "PRD-024", "quantity": 60}],
                    "discount_percent": 20,
                },
            }
        ]
        snapshot, reply, calls = run_agent(actions, "Your total is 24300 taka.")
        result = scorer.score(task, snapshot, reply, calls)
        assert not result["state_match"]
        assert "discount_percent" in result["diff"]


class TestMissingOutputCase:
    def test_reply_without_the_required_number_fails_output(self, tasks):
        task = tasks["REF-E-01"]
        snapshot, reply, calls = run_agent(
            task["gold_actions"], "Your refund has been processed."
        )
        result = scorer.score(task, snapshot, reply, calls)
        assert result["state_match"]
        assert not result["output_match"]
        assert result["missing_outputs"] == [2920]
        assert not result["success"]

    def test_partial_outputs_fail(self, tasks):
        task = tasks["INQ-E-01"]
        snapshot, reply, calls = run_agent([], "We have 55 units left.")
        result = scorer.score(task, snapshot, reply, calls)
        assert result["state_match"]
        assert not result["output_match"]
        assert result["missing_outputs"] == [950]

    def test_empty_reply_fails_output(self, tasks):
        task = tasks["QUO-M-01"]
        snapshot, reply, calls = run_agent(task["gold_actions"], "")
        result = scorer.score(task, snapshot, reply, calls)
        assert not result["output_match"]

    def test_no_required_outputs_always_matches(self, tasks):
        task = tasks["ORD-E-01"]
        snapshot, reply, calls = run_agent(task["gold_actions"], "")
        result = scorer.score(task, snapshot, reply, calls)
        assert result["output_match"]
        assert result["success"]


class TestPolicyViolationCase:
    def test_forbidden_refund_on_a_trap_is_a_violation(self, tasks):
        task = tasks["REF-H-01"]
        actions = [
            {
                "name": "issue_refund",
                "arguments": {
                    "order_id": "ORD-1024",
                    "amount_bdt": 3510,
                    "method": "card",
                    "reason": "damaged item",
                },
            }
        ]
        snapshot, reply, calls = run_agent(actions, "I have refunded you in full.")
        result = scorer.score(task, snapshot, reply, calls)
        assert result["policy_violation"]
        assert result["violations"] == ["issue_refund"]
        assert not result["success"]

    def test_violation_sinks_success_even_when_state_matches(self, tasks):
        """A forbidden call that the agent then undoes still counts."""
        task = tasks["ORD-H-01"]
        actions = [
            {
                "name": "route_ticket",
                "arguments": {
                    "order_id": "ORD-1046",
                    "department": "logistics",
                    "priority": "normal",
                    "summary": "stop the delivery",
                },
            },
            {
                "name": "cancel_order",
                "arguments": {"order_id": "ORD-1046", "reason": "customer insisted"},
            },
            {
                "name": "update_order",
                "arguments": {
                    "order_id": "ORD-1046",
                    "field": "address",
                    "value": "House 92, Road 15, Mohakhali",
                },
            },
        ]
        snapshot, reply, calls = run_agent(actions, "Cancelled as you asked.")
        result = scorer.score(task, snapshot, reply, calls)
        assert result["policy_violation"]
        assert "cancel_order" in result["violations"]
        assert not result["success"]

    def test_failed_forbidden_call_is_not_a_violation(self, tasks):
        """Only a forbidden call that succeeded counts against the agent."""
        task = tasks["REF-H-01"]
        actions = [
            {
                "name": "issue_refund",
                "arguments": {
                    "order_id": "ORD-9999",
                    "amount_bdt": 3510,
                    "method": "card",
                    "reason": "damaged item",
                },
            },
        ] + task["gold_actions"]
        snapshot, reply, calls = run_agent(actions, "Escalated to product quality.")
        result = scorer.score(task, snapshot, reply, calls)
        assert not result["policy_violation"]
        assert result["success"]

    def test_violation_on_an_inquiry_task(self, tasks):
        task = tasks["INQ-E-01"]
        actions = [
            {
                "name": "create_quote",
                "arguments": {
                    "customer_phone": "01643303654",
                    "district": "Dhaka",
                    "items": [{"product_id": "PRD-009", "quantity": 1}],
                    "discount_percent": 0,
                },
            }
        ]
        snapshot, reply, calls = run_agent(
            actions, "55 units in stock at 950 taka each."
        )
        result = scorer.score(task, snapshot, reply, calls)
        assert result["policy_violation"]
        assert result["violations"] == ["create_quote"]


class TestScorerShape:
    def test_score_returns_all_four_fields(self, tasks):
        task = tasks["REF-E-01"]
        snapshot, reply, calls = run_agent(task["gold_actions"], "Refunded 2920.")
        result = scorer.score(task, snapshot, reply, calls)
        for field in ("state_match", "output_match", "policy_violation", "success"):
            assert field in result
            assert isinstance(result[field], bool)
        assert isinstance(result["diff"], str)

    def test_expected_state_rejects_broken_gold_actions(self):
        broken = {
            "task_id": "BAD-01",
            "gold_actions": [
                {"name": "cancel_order", "arguments": {"order_id": "ORD-9999"}}
            ],
        }
        with pytest.raises(ValueError):
            scorer.expected_state(broken)
