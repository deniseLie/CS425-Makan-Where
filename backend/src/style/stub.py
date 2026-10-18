import re

# Discourse particles and constructions that are strong Singlish signals.
_PATTERNS = [
    r"\b(lah|leh|lor|hor|meh|sia|liao|ah|mah|nia|bah)\b",
    r"\b(alamak|aiyo|aiyah|shiok|makan|dabao|tapau|ta pau|walao|wah)\b",
    r"\bgot\b.*\bor not\b",
    r"\b(can|want|got|is it|have) or not\b",
    r"\balso can\b",
    r"\bwhere got\b",
    r"\beat what\b",
    r"\b\w+ side\b",              # "bugis side"
]
_RX = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]


def detect_style(text: str) -> str:
    """Stub keyword heuristic. Real version: SingBERT classifier."""
    return "singlish" if any(rx.search(text) for rx in _RX) else "english"
