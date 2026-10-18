"""Flan-T5 Singlish -> English normalizer from the Hugging Face Hub. Owner: Varidhi.

Train with the same PREFIX on every input.
"""

from functools import lru_cache

from src.config import model_id

PREFIX = "Translate Singlish to English: "
MAX_NEW_TOKENS = 128


@lru_cache(maxsize=1)
def _load():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    repo = model_id("normalizer")
    return AutoTokenizer.from_pretrained(repo), AutoModelForSeq2SeqLM.from_pretrained(repo).eval()


def normalize(text: str) -> str:
    import torch

    tok, model = _load()
    ids = tok(PREFIX + text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS, num_beams=4)
    return tok.decode(out[0], skip_special_tokens=True).strip() or text
