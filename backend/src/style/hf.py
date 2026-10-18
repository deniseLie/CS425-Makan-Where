"""SingBERT style classifier from the Hugging Face Hub. Owner: Varidhi.

Save the model with config.id2label = {0: "english", 1: "singlish"}.
"""

from functools import lru_cache

from src.config import model_id

_FALLBACK_LABELS = {"label_0": "english", "label_1": "singlish"}


@lru_cache(maxsize=1)
def _classifier():
    from transformers import pipeline

    return pipeline("text-classification", model=model_id("style"))


def detect_style(text: str) -> str:
    label = str(_classifier()(text, truncation=True)[0]["label"]).lower()
    label = _FALLBACK_LABELS.get(label, label)
    if label not in ("singlish", "english"):
        raise ValueError(f"style model returned unknown label {label!r}; expected singlish/english")
    return label
