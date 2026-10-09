"""Runner tests with a fake endpoint and no Ollama: records, resume, failures."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run
from agents.llm import LLMClient
from tests.fakes import FakeClient, response

REQUIRED_FIELDS = (
    "task_id", "category", "difficulty", "language", "is_trap", "model", "arch", "seed",
    "success", "state_match", "output_match", "policy_violation", "llm_calls", "tool_calls",
    "malformed_tool_calls", "text_tool_calls", "prompt_tokens", "completion_tokens",
    "wall_time_s", "llm_latency_s", "energy_wh", "net_energy_wh", "budget_exceeded",
    "llm_timeout", "stop_reason", "delegations", "replans", "policy_sha256", "tasks_sha256",
    "error", "dropped_tool_calls", "empty_reply", "prompt_sha256",
)


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    """Point the runner at a temp results dir and replace every Ollama call."""
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run, "IDLE_SETTLE_S", 0)
    monkeypatch.setattr(run, "check_ollama", lambda models, config: None)
    monkeypatch.setattr(run, "prepare_model", lambda model, config: 0.0)
    monkeypatch.setattr(run, "reset_model_state", lambda model, config: 0.01)
    monkeypatch.setattr(run, "measure_idle_power", lambda seconds: 12.5)
    monkeypatch.setattr(run, "ollama_version", lambda config: "test")
    monkeypatch.setattr(
        run, "model_residency",
        lambda model, config: {"gpu_fraction": 1.0, "processor": "100% GPU"},
    )
    monkeypatch.setattr(
        run, "LLMClient",
        lambda model, config: LLMClient(model, config=config, client=FakeClient(lambda r: response("Done."))),
    )
    return tmp_path


def invoke(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["run.py", *argv])
    return run.main()


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


ARGS = ("--tag", "t", "--models", "fake-model", "--archs", "react", "--limit", "2", "--runs", "2")


def test_records_every_run_with_all_fields_and_traces(fake_env, monkeypatch):
    assert invoke(monkeypatch, *ARGS) == 0
    rows = read_rows(fake_env / "t" / "runs.jsonl")
    assert len(rows) == 4
    assert [(r["task_id"], r["seed"]) for r in rows][:2] == [(rows[0]["task_id"], 0), (rows[0]["task_id"], 1)]
    for row in rows:
        for field in REQUIRED_FIELDS:
            assert field in row
        assert row["delegations"] is None and row["replans"] is None
        assert (fake_env / "t" / row["trace_file"]).exists()
    meta = json.loads((fake_env / "t" / "meta.json").read_text(encoding="utf-8"))
    assert meta["idle_power_w"] == 12.5
    assert meta["policy_sha256"] == rows[0]["policy_sha256"]
    assert meta["tasks_sha256"] == rows[0]["tasks_sha256"]
    assert "max_llm_calls_per_task" in meta["config_yaml"]


def test_existing_tag_needs_resume(fake_env, monkeypatch):
    assert invoke(monkeypatch, *ARGS) == 0
    assert invoke(monkeypatch, *ARGS) == 1


def test_resume_skips_done_and_reruns_a_cut_off_line(fake_env, monkeypatch):
    assert invoke(monkeypatch, *ARGS) == 0
    runs_path = fake_env / "t" / "runs.jsonl"
    lines = runs_path.read_text(encoding="utf-8").splitlines()
    runs_path.write_text("\n".join(lines[:3]) + "\n" + lines[3][:40], encoding="utf-8")

    assert invoke(monkeypatch, *ARGS, "--resume") == 0
    rows = read_rows(runs_path)
    assert len(rows) == 4
    assert len({(r["task_id"], r["seed"]) for r in rows}) == 4
    assert (fake_env / "t" / "runs.jsonl.bak").exists()

    assert invoke(monkeypatch, *ARGS, "--resume") == 0
    assert len(read_rows(runs_path)) == 4


def test_resume_refuses_when_policy_or_tasks_changed(fake_env, monkeypatch, capsys):
    assert invoke(monkeypatch, *ARGS) == 0
    meta_path = fake_env / "t" / "meta.json"
    for key in ("policy_sha256", "prompt_sha256", "tasks_sha256"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        original = meta[key]
        meta[key] = "0" * 64
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        assert invoke(monkeypatch, *ARGS, "--resume") == 1
        assert key in capsys.readouterr().out
        meta[key] = original
        meta_path.write_text(json.dumps(meta), encoding="utf-8")


def test_agent_exception_is_recorded_and_the_runner_continues(fake_env, monkeypatch):
    from agents.react import ReActAgent

    calls = {"n": 0}
    original = ReActAgent._run

    def flaky(self, ctx):
        calls["n"] += 1
        if calls["n"] == 1:
            ctx.open_conversation("react", None, ctx.task["instruction"])
            raise ValueError("boom")
        return original(self, ctx)

    monkeypatch.setattr(ReActAgent, "_run", flaky)
    assert invoke(monkeypatch, *ARGS) == 0
    rows = read_rows(fake_env / "t" / "runs.jsonl")
    assert len(rows) == 4
    assert rows[0]["success"] is False
    assert rows[0]["stop_reason"] == "exception"
    assert rows[0]["error"] == "ValueError: boom"
    trace = json.loads((fake_env / "t" / rows[0]["trace_file"]).read_text(encoding="utf-8"))
    assert "ValueError" in trace["error_traceback"]
    assert trace["trace"], "partial trace should be kept"
    assert all(row["error"] is None for row in rows[1:])


def test_infrastructure_failure_stops_the_runner(fake_env, monkeypatch):
    def down(model, config):
        raise OSError("connection refused")

    monkeypatch.setattr(run, "reset_model_state", down)
    assert invoke(monkeypatch, *ARGS) == 1
    runs_path = fake_env / "t" / "runs.jsonl"
    assert not runs_path.exists() or read_rows(runs_path) == []
