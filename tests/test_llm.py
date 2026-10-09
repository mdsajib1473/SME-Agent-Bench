"""LLM client tests with a fake endpoint: malformed calls, text calls, budget."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.llm import BudgetExceeded, LLMClient, find_text_tool_calls
from agents.prompts import build_system_prompt, load_policy, policy_sha256
from agents.react import ReActAgent
from env.shop import Shop
from env.tools import TOOL_SCHEMAS
from tests.fakes import FakeClient, native_call, response

CONFIG = {
    "ollama_base_url": "http://fake/v1",
    "temperature": 0.3,
    "max_llm_calls_per_task": 3,
    "request_timeout_s": 5,
}


def make_client(responses, max_calls=3):
    config = dict(CONFIG, max_llm_calls_per_task=max_calls)
    fake = FakeClient(responses)
    return LLMClient("fake-model", config=config, client=fake), fake


USER = [{"role": "user", "content": "hi"}]


@pytest.fixture
def shop():
    with Shop() as instance:
        yield instance


class TestMalformedToolCalls:
    def test_invalid_json_arguments_become_an_error_result(self, shop):
        llm, _ = make_client([response(tool_calls=[native_call("get_order", '{"order_id": ')])])
        before = shop.snapshot()
        result = llm.chat(USER, tools=TOOL_SCHEMAS, seed=0)
        call = result.tool_calls[0]
        assert call.malformed
        assert "not valid JSON" in call.error
        tool_result = llm.execute_tool_call(shop, call)
        assert tool_result["ok"] is False
        assert "malformed tool call" in tool_result["error"]
        assert llm.totals["malformed_tool_calls"] == 1
        assert llm.totals["tool_calls"] == 0
        assert shop.snapshot() == before

    def test_malformed_arguments_are_not_echoed_back(self):
        llm, _ = make_client([response(tool_calls=[native_call("get_order", "{oops")])])
        result = llm.chat(USER, tools=TOOL_SCHEMAS, seed=0)
        assert result.message["tool_calls"][0]["function"]["arguments"] == "{}"
        assert result.tool_calls[0].raw_arguments == "{oops"

    def test_unknown_tool_name_is_malformed(self, shop):
        llm, _ = make_client([response(tool_calls=[native_call("delete_everything", {})])])
        result = llm.chat(USER, tools=TOOL_SCHEMAS, seed=0)
        call = result.tool_calls[0]
        assert call.malformed
        assert "unknown tool" in call.error
        assert llm.execute_tool_call(shop, call)["ok"] is False
        assert llm.totals["malformed_tool_calls"] == 1

    def test_non_object_arguments_are_malformed(self):
        llm, _ = make_client([response(tool_calls=[native_call("get_order", "[1, 2]")])])
        call = llm.chat(USER, tools=TOOL_SCHEMAS, seed=0).tool_calls[0]
        assert call.malformed
        assert "JSON object" in call.error

    def test_well_formed_call_executes(self, shop):
        llm, _ = make_client(
            [response(tool_calls=[native_call("get_order", {"order_id": "ORD-1008"})])]
        )
        call = llm.chat(USER, tools=TOOL_SCHEMAS, seed=0).tool_calls[0]
        assert not call.malformed
        assert llm.execute_tool_call(shop, call)["ok"] is True
        assert llm.totals["tool_calls"] == 1
        assert llm.totals["malformed_tool_calls"] == 0
        assert llm.totals["text_tool_calls"] == 0


class TestTextToolCalls:
    @pytest.mark.parametrize(
        "content",
        [
            '{"name": "get_order", "arguments": {"order_id": "ORD-1008"}}',
            '<tool_call>\n{"name": "get_order", "arguments": {"order_id": "ORD-1008"}}\n</tool_call>',
            'Let me check.\n```json\n{"name": "get_order", "parameters": {"order_id": "ORD-1008"}}\n```',
            '{"type": "function", "function": {"name": "get_order", "arguments": "{\\"order_id\\": \\"ORD-1008\\"}"}}',
        ],
    )
    def test_json_in_content_is_parsed_and_executed(self, shop, content):
        llm, _ = make_client([response(content=content)])
        result = llm.chat(USER, tools=TOOL_SCHEMAS, seed=0)
        assert len(result.tool_calls) == 1
        call = result.tool_calls[0]
        assert call.source == "text"
        assert call.name == "get_order"
        assert call.arguments == {"order_id": "ORD-1008"}
        assert result.message["tool_calls"][0]["function"]["name"] == "get_order"
        assert "ORD-1008" not in result.message["content"]

        tool_result = llm.execute_tool_call(shop, call)
        assert tool_result["ok"] is True
        assert tool_result["order"]["order_id"] == "ORD-1008"
        assert llm.totals["text_tool_calls"] == 1
        assert llm.totals["tool_calls"] == 1

    def test_several_text_calls_keep_their_order(self):
        content = (
            '<tool_call>{"name": "get_order", "arguments": {"order_id": "ORD-1008"}}</tool_call>\n'
            '<tool_call>{"name": "check_stock", "arguments": {"product_id": "PRD-001"}}</tool_call>'
        )
        llm, _ = make_client([response(content=content)])
        result = llm.chat(USER, tools=TOOL_SCHEMAS, seed=0)
        assert [call.name for call in result.tool_calls] == ["get_order", "check_stock"]
        assert len({call.id for call in result.tool_calls}) == 2
        assert result.message["content"] == ""

    def test_unknown_name_in_text_is_malformed_text_call(self, shop):
        llm, _ = make_client([response(content='{"name": "refund_all", "arguments": {}}')])
        call = llm.chat(USER, tools=TOOL_SCHEMAS, seed=0).tool_calls[0]
        assert call.source == "text"
        assert call.malformed
        llm.execute_tool_call(shop, call)
        assert llm.totals["text_tool_calls"] == 1
        assert llm.totals["malformed_tool_calls"] == 1
        assert llm.totals["tool_calls"] == 0

    @pytest.mark.parametrize(
        "content",
        [
            "Your refund of 2920 taka has been sent to bKash.",
            'Here is the data: {"order_id": "ORD-1008", "total": 2920}',
            "Braces { without json } are fine.",
            "",
        ],
    )
    def test_plain_replies_are_not_tool_calls(self, content):
        llm, _ = make_client([response(content=content)])
        result = llm.chat(USER, tools=TOOL_SCHEMAS, seed=0)
        assert result.tool_calls == []
        assert "tool_calls" not in result.message
        assert result.content == content

    def test_native_calls_take_precedence_over_text(self):
        llm, _ = make_client(
            [
                response(
                    content='{"name": "check_stock", "arguments": {"product_id": "PRD-001"}}',
                    tool_calls=[native_call("get_order", {"order_id": "ORD-1008"})],
                )
            ]
        )
        result = llm.chat(USER, tools=TOOL_SCHEMAS, seed=0)
        assert [(c.name, c.source) for c in result.tool_calls] == [("get_order", "native")]

    def test_no_text_parsing_without_tools(self):
        llm, _ = make_client([response(content='{"name": "get_order", "arguments": {}}')])
        assert llm.chat(USER, seed=0).tool_calls == []

    def test_residual_prose_is_kept(self):
        calls, residual = find_text_tool_calls(
            'Checking now. <tool_call>{"name": "get_order", "arguments": {}}</tool_call>'
        )
        assert calls == [("get_order", {})]
        assert residual == "Checking now."


class TestBudget:
    def test_budget_raises_after_max_calls_without_sending(self):
        llm, fake = make_client([response("a"), response("b"), response("c"), response("d")])
        for _ in range(3):
            llm.chat(USER, seed=0)
        with pytest.raises(BudgetExceeded):
            llm.chat(USER, seed=0)
        assert len(fake.requests) == 3
        assert llm.totals["llm_calls"] == 3
        assert llm.calls_remaining == 0

    def test_reset_restores_budget_and_zeroes_totals(self):
        llm, _ = make_client([response("a"), response("b")], max_calls=1)
        llm.chat(USER, seed=0)
        with pytest.raises(BudgetExceeded):
            llm.chat(USER, seed=0)
        llm.reset()
        assert all(value == 0 for value in llm.totals.values())
        llm.chat(USER, seed=0)
        assert llm.totals["llm_calls"] == 1

    def test_failed_request_still_counts(self):
        class Boom(Exception):
            pass

        llm, fake = make_client([], max_calls=2)

        def explode(**request):
            raise Boom("endpoint down")

        fake.chat.completions.create = explode
        with pytest.raises(Boom):
            llm.chat(USER, seed=0)
        assert llm.totals["llm_calls"] == 1

    def test_totals_accumulate_tokens_and_latency(self):
        llm, _ = make_client(
            [response("a", prompt_tokens=120, completion_tokens=7), response("b", prompt_tokens=30, completion_tokens=5)]
        )
        llm.chat(USER, seed=0)
        llm.chat(USER, seed=0)
        assert llm.totals["prompt_tokens"] == 150
        assert llm.totals["completion_tokens"] == 12
        assert llm.totals["llm_latency_s"] >= 0

    def test_seed_and_settings_reach_the_endpoint(self):
        llm, fake = make_client([response("a")])
        llm.chat(USER, tools=TOOL_SCHEMAS, seed=7)
        request = fake.requests[0]
        assert request["extra_body"] == {"seed": 7}
        assert request["temperature"] == 0.3
        assert request["model"] == "fake-model"
        assert len(request["tools"]) == 12


class TestReActAgent:
    TASK = {"task_id": "T-1", "instruction": "Where is my order ORD-1008?"}

    def test_tool_then_final_answer(self, shop):
        llm, fake = make_client(
            [
                response(tool_calls=[native_call("get_order", {"order_id": "ORD-1008"})]),
                response(content="Your order ORD-1008 is delivered."),
            ]
        )
        result = ReActAgent().run(self.TASK, shop, llm, seed=0)
        assert result.final_reply == "Your order ORD-1008 is delivered."
        assert result.metadata["stop_reason"] == "final_answer"
        assert result.metadata["llm_calls"] == 2
        assert result.metadata["tool_calls"] == 1
        assert result.metadata["policy_sha256"] == policy_sha256(load_policy())
        assert [c["name"] for c in result.tool_calls] == ["get_order"]
        second = fake.requests[1]["messages"]
        assert second[-1]["role"] == "tool"
        assert second[-1]["tool_call_id"] == "call_x"
        types = [event["type"] for event in result.trace]
        assert types == ["message", "message", "llm_call", "tool_call", "llm_call"]

    def test_budget_exhaustion_ends_the_run(self, shop):
        looping = [
            response(tool_calls=[native_call("get_order", {"order_id": "ORD-1008"})])
            for _ in range(5)
        ]
        llm, fake = make_client(looping, max_calls=3)
        result = ReActAgent().run(self.TASK, shop, llm, seed=0)
        assert result.metadata["stop_reason"] == "budget_exceeded"
        assert result.final_reply == ""
        assert len(fake.requests) == 3

    def test_iteration_cap(self, shop):
        looping = [
            response(tool_calls=[native_call("get_order", {"order_id": "ORD-1008"})])
            for _ in range(20)
        ]
        llm, fake = make_client(looping, max_calls=20)
        result = ReActAgent().run(self.TASK, shop, llm, seed=0)
        assert result.metadata["stop_reason"] == "max_iterations"
        assert len(fake.requests) == 15

    def test_counters_reset_between_tasks(self, shop):
        llm, _ = make_client([response("one"), response("two")])
        ReActAgent().run(self.TASK, shop, llm, seed=0)
        result = ReActAgent().run(self.TASK, shop, llm, seed=0)
        assert result.metadata["llm_calls"] == 1


class TestPrompts:
    def test_shared_text_contains_policy_unchanged(self):
        policy = load_policy()
        prompt = build_system_prompt("Extra instructions.")
        assert policy.strip() in prompt.text
        assert prompt.text.startswith(prompt.shared_text)
        assert prompt.text.endswith("Extra instructions.")
        assert "2026-10-01 10:00" in prompt.shared_text
        assert "customer service assistant for Dokan" in prompt.shared_text
        assert "Banglish" in prompt.shared_text
        assert prompt.policy_sha256 == policy_sha256(policy)

    def test_shared_text_is_identical_across_architectures(self):
        assert build_system_prompt("A").shared_text == build_system_prompt("B").shared_text


def test_energy_meter_without_nvml(monkeypatch):
    from telemetry import energy

    monkeypatch.setitem(sys.modules, "pynvml", None)
    with energy.EnergyMeter() as meter:
        pass
    assert meter.available is False
    assert meter.energy_wh is None
    assert meter.energy_raw_wh is None
    assert meter.net_energy_wh(10.0) is None
    assert meter.counter_energy_wh is None
    assert meter.net_counter_energy_wh(10.0) is None
    assert energy.measure_idle_power(seconds=0) is None


def test_net_counter_energy_subtracts_idle_over_the_measured_duration():
    from telemetry import energy

    meter = energy.EnergyMeter()
    meter.counter_energy_wh = 0.5
    meter.duration_s = 36.0
    assert meter.net_counter_energy_wh(10.0) == 0.5 - 10.0 * 36.0 / 3600.0
    assert meter.net_counter_energy_wh(None) is None
    meter.counter_energy_wh = None
    assert meter.net_counter_energy_wh(10.0) is None


def test_integrate_wh_cuts_segments_at_the_window_edges():
    from telemetry import energy

    samples = [(0.1 * k, 100.0) for k in range(31)]
    assert energy.integrate_wh(samples, 1.05, 2.05) * 3600.0 == pytest.approx(100.0)
    ramp = [(0.0, 0.0), (1.0, 100.0)]
    assert energy.integrate_wh(ramp, 0.5, 1.0) * 3600.0 == pytest.approx(37.5)


def test_lag_correction_recovers_a_power_step():
    from telemetry import energy

    lag = 0.65
    # True power steps from 10 W to 110 W at t = 1.0; the reported series shows it lag seconds later.
    samples = [(0.1 * k, 10.0 if 0.1 * k < 1.0 + lag else 110.0) for k in range(41)]
    true_j = 10.0 * 0.5 + 110.0 * 1.5
    corrected_j = energy.integrate_wh(samples, 0.5, 2.5, lag) * 3600.0
    raw_j = energy.integrate_wh(samples, 0.5, 2.5) * 3600.0
    assert corrected_j == pytest.approx(true_j, abs=1.0)
    assert raw_j < true_j - 50.0


class FakeNvml:
    """Constant 50 W board: power in mW and a counter in mJ that follows perf_counter."""

    def __init__(self):
        import time

        self.time = time

    def nvmlInit(self):
        pass

    def nvmlShutdown(self):
        pass

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetPowerUsage(self, handle):
        return 50000

    def nvmlDeviceGetMemoryInfo(self, handle):
        from types import SimpleNamespace

        return SimpleNamespace(used=1024**2)

    def nvmlDeviceGetTotalEnergyConsumption(self, handle):
        return int(50000 * self.time.perf_counter())


def test_meter_samples_a_tail_but_reads_the_counter_at_the_block_end(monkeypatch):
    import time

    from telemetry import energy

    monkeypatch.setattr(energy, "_load_nvml", FakeNvml)
    monkeypatch.setattr(energy, "LAG_TAIL_S", 0.3)
    with energy.EnergyMeter(interval_s=0.02, lag_s=0.2) as meter:
        time.sleep(0.3)
    assert meter.samples[-1][0] - meter.ended >= 0.25
    assert meter.counter_energy_wh * 3600.0 == pytest.approx(50.0 * meter.duration_s, abs=1.0)
    assert meter.energy_raw_wh * 3600.0 == pytest.approx(50.0 * meter.duration_s, abs=1.0)
    assert meter.energy_wh == pytest.approx(meter.energy_raw_wh, rel=1e-6)
    assert meter.net_energy_wh(50.0) * 3600.0 == pytest.approx(0.0, abs=1.0)


def test_meter_without_lag_has_no_tail(monkeypatch):
    import time

    from telemetry import energy

    monkeypatch.setattr(energy, "_load_nvml", FakeNvml)
    with energy.EnergyMeter(interval_s=0.02) as meter:
        time.sleep(0.1)
    assert meter.samples[-1][0] - meter.ended < 0.05
    assert meter.energy_wh == meter.energy_raw_wh


class TestModelReset:
    def test_unloads_waits_then_warms_up(self, monkeypatch):
        from agents import llm as llm_module

        posts = []
        loaded = iter([True, True, False])
        monkeypatch.setattr(llm_module, "ollama_post", lambda url, payload, timeout: posts.append((url, payload)) or {})
        monkeypatch.setattr(llm_module, "_model_loaded", lambda native_url, model, timeout: next(loaded))
        monkeypatch.setattr(llm_module, "UNLOAD_POLL_S", 0)

        seconds = llm_module.reset_model_state("m", dict(CONFIG))
        assert seconds >= 0
        assert posts[0] == ("http://fake/api/generate", {"model": "m", "keep_alive": 0})
        assert posts[1][0] == "http://fake/api/generate"
        assert posts[1][1]["prompt"] == llm_module.RESET_WARMUP["prompt"]
        assert posts[1][1]["options"]["seed"] == 0
        assert len(posts) == 2

    def test_timeout_is_flagged_in_metadata(self, shop):
        import httpx2
        import openai

        llm, fake = make_client([], max_calls=2)

        def time_out(**request):
            raise openai.APITimeoutError(request=httpx2.Request("POST", "http://fake/v1"))

        fake.chat.completions.create = time_out
        result = ReActAgent().run({"task_id": "T", "instruction": "hi"}, shop, llm, seed=0)
        assert result.metadata["stop_reason"] == "llm_error"
        assert result.metadata["llm_timeout"] is True
