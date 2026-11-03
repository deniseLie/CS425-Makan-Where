"""Reply generator. Owner: Denise.

Interface (do not change):
    generate(action: PolicyAction, style: str, user_text: str) -> str
    template_reply(action: PolicyAction, style: str) -> str   # safe fallback used by the pipeline guardrail

Real version: Qwen2.5-1.5B + its own LoRA adapter (not the tracker's), task [REPLY],
conditioned on the policy action + style tag. It must only mention places in action.places.
Pick stub.py or hf.py with MAKAN_GENERATOR=stub|hf (see src/config.py).
"""
from src.config import impl

from .templates import template_reply

if impl("generator") == "hf":
    from .hf import generate
else:
    from .stub import generate

__all__ = ["generate", "template_reply"]
