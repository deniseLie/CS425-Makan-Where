"""Reply generator. Owner: Denise.

Interface (do not change):
    generate(action: PolicyAction, style: str, user_text: str) -> str
    template_reply(action: PolicyAction, style: str) -> str   # safe fallback used by the pipeline guardrail

Real version: Qwen2.5-1.5B + LoRA (same adapter as the tracker), task [REPLY],
conditioned on the policy action + style tag. It must only mention places in action.places.
"""
from .templates import template_reply
from .stub import generate

__all__ = ["generate", "template_reply"]
