"""Qwen + LoRA reply generator, task [REPLY]. Owner: Denise.

build_prompt() defines the training format; build the [REPLY] training file with it.
The pipeline guardrail still replaces any reply that names a place not in action.places.
"""

from __future__ import annotations

import json

from src.schemas import PolicyAction

from .templates import template_reply


def build_prompt(action: PolicyAction, style: str, user_text: str) -> str:
    return "\n".join([
        "[REPLY]",
        f"Style: {style}",
        f"Action: {json.dumps(action.to_dict(), ensure_ascii=False)}",
        f"User: {user_text}",
        "",
        "Write the bot's reply. Only mention places listed in the action.",
    ])


def generate(action: PolicyAction, style: str, user_text: str) -> str:
    from src.llm import complete

    return complete(build_prompt(action, style, user_text), "generator") or template_reply(action, style)
