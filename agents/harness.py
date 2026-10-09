"""harness_sha256: one hash over everything that shapes a run besides the model.

Covers the shared system prompt (policy included), every role prompt and fixed
per-role instruction of the three architectures, the specialist tool schemas,
the 12 shop tool schemas, the scorer source, and the limits: call budget, token
cap and temperature from config.yaml plus the iteration, step, delegation and
replan limits in the agents. prompt_sha256 stays as the hash of the shared
prompt alone.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import plan_execute, react, supervisor
from agents.llm import load_config
from agents.prompts import build_shared_prompt, load_policy, text_sha256
from env import tools

SCORER_PATH = ROOT / "eval" / "scorer.py"
CONFIG_LIMIT_KEYS = ("max_llm_calls_per_task", "max_tokens_per_call", "temperature")


def harness_components(config=None, policy_text=None):
    """Everything harness_sha256 covers, read from the modules at call time."""
    config = config if config is not None else load_config()
    policy_text = policy_text if policy_text is not None else load_policy()
    return {
        "shared_prompt": build_shared_prompt(policy_text),
        "react": {
            "role": react.ROLE_INSTRUCTIONS,
            "max_iterations": react.MAX_ITERATIONS,
        },
        "plan_execute": {
            "planner_role": plan_execute.PLANNER_ROLE,
            "executor_role": plan_execute.EXECUTOR_ROLE,
            "responder_role": plan_execute.RESPONDER_ROLE,
            "plan_retry_instruction": plan_execute.PLAN_RETRY_INSTRUCTION,
            "replan_instruction": plan_execute.REPLAN_INSTRUCTION,
            "executor_instruction": plan_execute.EXECUTOR_INSTRUCTION,
            "responder_instruction": plan_execute.RESPONDER_INSTRUCTION,
            "fallback_step": plan_execute.FALLBACK_STEP,
            "max_planner_attempts": plan_execute.MAX_PLANNER_ATTEMPTS,
            "max_step_calls": plan_execute.MAX_STEP_CALLS,
            "max_replans": plan_execute.MAX_REPLANS,
            "responder_reserve": plan_execute.RESPONDER_RESERVE,
            "result_text_limit": plan_execute.RESULT_TEXT_LIMIT,
        },
        "supervisor": {
            "supervisor_role": supervisor.SUPERVISOR_ROLE,
            "specialist_roles": dict(supervisor.SPECIALIST_ROLES),
            "specialist_tools": {k: list(v) for k, v in supervisor.SPECIALIST_TOOLS.items()},
            "specialist_schemas": supervisor.SPECIALIST_SCHEMAS,
            "max_delegations": supervisor.MAX_DELEGATIONS,
            "max_specialist_calls": supervisor.MAX_SPECIALIST_CALLS,
            "max_supervisor_turns": supervisor.MAX_SUPERVISOR_TURNS,
            "supervisor_reserve": supervisor.SUPERVISOR_RESERVE,
        },
        "tool_schemas": tools.TOOL_SCHEMAS,
        # Text mode reads CRLF as LF, so both checkouts hash the same.
        "scorer_source": SCORER_PATH.read_text(encoding="utf-8"),
        "config_limits": {key: config.get(key) for key in CONFIG_LIMIT_KEYS},
    }


def harness_sha256(config=None, policy_text=None):
    components = harness_components(config, policy_text)
    return text_sha256(json.dumps(components, sort_keys=True, ensure_ascii=False))
