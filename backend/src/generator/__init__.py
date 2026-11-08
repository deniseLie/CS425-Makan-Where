"""Reply generator. Owner: Denise.

Interface (do not change):
    generate(action: PolicyAction, style: str, user_text: str) -> str
    template_reply(action: PolicyAction, style: str) -> str   # safe fallback used by the pipeline guardrail

Qwen2.5-1.5B + its own LoRA adapter (not the tracker's), task [REPLY] (hf.py; adapter
MAKAN_GENERATOR_ADAPTER, see src/config.py), conditioned on the policy action + style tag.
It must only mention places in action.places.
"""
from .hf import generate
from .templates import template_reply

__all__ = ["generate", "template_reply"]
