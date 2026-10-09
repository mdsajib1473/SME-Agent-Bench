"""Shared system prompt.

Every architecture builds its system prompt here. The shared text (role, fixed
date, language rule, full policy, conversation rules) is identical for all arms;
an architecture may only append its own role instructions after it.
"""

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.shop import FIXED_CLOCK

POLICY_PATH = ROOT / "env" / "policy.md"


CONVERSATION_RULES = (
    "Rules for this conversation:\n"
    "- This is a single message conversation. The customer cannot reply, so"
    " never ask the customer to confirm anything or to provide more"
    " information before acting. If the policy allows the request, act on it"
    " now; otherwise refuse and explain why.\n"
    "- Never guess an order ID or a product ID. Find them with the lookup"
    " tools.\n"
    "- Only state actions you have actually performed.\n"
    "- Reply in the customer's language style."
)


@dataclass(frozen=True)
class SystemPrompt:
    text: str
    shared_text: str
    policy_sha256: str
    prompt_sha256: str


def load_policy(path=POLICY_PATH):
    return Path(path).read_text(encoding="utf-8")


def text_sha256(text):
    # Hashes the decoded text, so a CRLF checkout hashes the same as LF.
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def policy_sha256(policy_text):
    return text_sha256(policy_text)


def prompt_sha256(policy_text):
    """Hash of the full shared prompt every architecture receives."""
    return text_sha256(build_shared_prompt(policy_text))


def current_time_text():
    return FIXED_CLOCK.strftime("%Y-%m-%d %H:%M") + " Asia/Dhaka (UTC+06:00)"


def build_shared_prompt(policy_text):
    return (
        "You are a customer service assistant for Dokan, a small online shop"
        " in Bangladesh.\n\n"
        f"The current date and time is {current_time_text()}. Use this fixed"
        " time for every age and deadline, never the real clock.\n\n"
        "Reply to the customer in the same language style they used: if they"
        " wrote in English, reply in English; if they wrote in Banglish (Bangla"
        " written in Latin letters), reply in Banglish.\n\n"
        "You must follow the shop policy below.\n\n"
        "<policy>\n"
        f"{policy_text.strip()}\n"
        "</policy>\n\n"
        f"{CONVERSATION_RULES}"
    )


def build_system_prompt(role_instructions=None, policy_text=None, policy_path=POLICY_PATH):
    if policy_text is None:
        policy_text = load_policy(policy_path)
    shared = build_shared_prompt(policy_text)
    text = shared
    if role_instructions:
        text = f"{shared}\n\n{role_instructions.strip()}"
    return SystemPrompt(
        text=text,
        shared_text=shared,
        policy_sha256=policy_sha256(policy_text),
        prompt_sha256=text_sha256(shared),
    )
