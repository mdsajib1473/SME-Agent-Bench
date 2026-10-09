"""Shared prompt rules, prompt hash, schema example IDs, validator, dropped calls."""

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from env.tools import TOOL_SCHEMAS
from eval import validate_tasks
from tests.fakes import FakeClient, response

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
