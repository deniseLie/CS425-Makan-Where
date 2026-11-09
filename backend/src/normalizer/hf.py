"""Flan-T5 Singlish -> English normalizer with entity protection."""

from functools import lru_cache
import re

from src.config import model_id
from src.normalizer.entities import (
    protect_entities,
    restore_entities,
    all_entities_preserved,
)

PREFIX = "Normalize this Singlish into natural English: "
MAX_NEW_TOKENS = 128


@lru_cache(maxsize=1)
def _load():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    repo = model_id("normalizer")
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSeq2SeqLM.from_pretrained(repo).eval()
    return tokenizer, model


def _looks_repetitive(text: str) -> bool:
    words = text.lower().split()

    # Detect excessively elongated words, e.g. woooooooooooo.
    if any(re.search(r"(.)\1{5,}", word) for word in words):
        return True

    if len(words) >= 12:
        for n in range(3, 6):
            for i in range(len(words) - 2 * n + 1):
                if words[i:i+n] == words[i+n:i+2*n]:
                    return True

    return False


def normalize(text: str) -> str:
    if not text or not text.strip():
        return text

    protected, mapping = protect_entities(text)

    tokenizer, model = _load()
    inputs = tokenizer(
        PREFIX + protected,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    import torch

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    generated = tokenizer.decode(
        output[0],
        skip_special_tokens=True,
    ).strip()

    # Reject output if the model drops any protected entity.
    if not generated or not all_entities_preserved(generated, mapping):
        return text

    # Reject obvious repetition loops.
    if _looks_repetitive(generated):
        return text

    # Reject suspiciously long generations.
    if len(generated) > max(30, len(protected) * 4):
        return text

    return restore_entities(generated, mapping) or text
