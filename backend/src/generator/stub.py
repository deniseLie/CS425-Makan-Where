from src.schemas import PolicyAction

from .templates import template_reply


def generate(action: PolicyAction, style: str, user_text: str) -> str:
    """Stub: template reply. Real version: Qwen + LoRA [REPLY]."""
    return template_reply(action, style)
