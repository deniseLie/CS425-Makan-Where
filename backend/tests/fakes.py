"""Offline stand-ins for the four models, so the tests run without torch, a GPU or the Hub.

conftest.py puts them in place of the real normalize / detect_style / track / generate for every
test. They only need to be good enough for the test conversations ("korean lah, bugis side",
"too far leh"); a test that cares what a trained model writes scripts its answers through
src.tracker.hf.parse instead.
"""

import re

from src.generator.templates import template_reply
from src.schemas import ANY, REQUIRED_SLOTS, SOFT_SLOTS, UNKNOWN, DialogueState, PolicyAction, TrackerOutput, Turn
from src.tracker.values import TABLES


def normalize(text: str) -> str:
    """The Flan-T5 normalizer's stand-in: the text unchanged."""
    return text


def generate(action: PolicyAction, style: str, user_text: str) -> str:
    """The [REPLY] generator's stand-in: the template reply."""
    return template_reply(action, style)


# SingBERT's stand-in: discourse particles and constructions that are strong Singlish signals.
_STYLE_RX = [re.compile(p, re.IGNORECASE) for p in [
    r"\b(lah|leh|lor|hor|meh|sia|liao|ah|mah|nia|bah)\b",
    r"\b(alamak|aiyo|aiyah|shiok|makan|dabao|tapau|ta pau|walao|wah)\b",
    r"\bgot\b.*\bor not\b",
    r"\b(can|want|got|is it|have) or not\b",
    r"\balso can\b",
    r"\bwhere got\b",
    r"\beat what\b",
    r"\b\w+ side\b",              # "bugis side"
]]


def detect_style(text: str) -> str:
    return "singlish" if any(rx.search(text) for rx in _STYLE_RX) else "english"


# The [TRACK] tracker's stand-in: phrases from the translator's tables, latest turn only, everything
# else carried over from `prev`. It writes place-list values directly.
PHRASES: dict[str, dict[str, str]] = {slot: dict(table) for slot, table in TABLES.items()}
PHRASES["budget"].update({"cheap": "<=8", "budget": "<=8", "affordable": "<=8"})
ANY_PHRASES = ["anything also can", "anything can", "any also can", "anything",
               "whatever", "dont mind", "don't mind", "no preference", "up to you"]
CRITIQUE_PHRASES = {
    "too_far": ["too far", "so far", "far leh", "far lah", "nearer", "closer"],
    "too_expensive": ["too expensive", "too ex", "so ex", "pricey", "cheaper", "too pricey"],
    "not_interested": ["dun want this one", "don't want this one", "dont want this one",
                       "not this one", "next one", "something else"],
}
_BUDGET_RX = re.compile(r"(?:under|below|less than|within|<=?|max)\s*\$?\s*(\d+)", re.IGNORECASE)


def _has(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None


def track(history: list[Turn], prev: DialogueState) -> TrackerOutput:
    turn = history[-1]
    text = (turn.normalized or turn.raw).lower()
    slots = dict(prev.slots)
    mentioned = False

    for slot, table in PHRASES.items():
        # longest phrase first so "jurong east" beats "jurong" and "east"
        for phrase in sorted(table, key=len, reverse=True):
            if _has(text, phrase):
                slots[slot] = table[phrase]
                mentioned = True
                break

    m = _BUDGET_RX.search(text)
    if m:
        slots["budget"] = f"<={m.group(1)}"
        mentioned = True

    critique = next((c for c, ps in CRITIQUE_PHRASES.items() if any(_has(text, p) for p in ps)), None)

    # "anything also can" -> ANY for the slot the policy would most likely have asked.
    if not mentioned and critique is None and any(_has(text, p) for p in ANY_PHRASES):
        for slot in REQUIRED_SLOTS + SOFT_SLOTS:
            if slots[slot] == UNKNOWN:
                slots[slot] = ANY
                break

    return TrackerOutput(slots=slots, reject=_find_reject(text), critique=critique)


def _find_reject(text: str) -> str | None:
    """'don't want <place name>' / 'not <place name>' -> that place_id."""
    if not re.search(r"\b(don'?t want|not|no|skip|bored of)\b", text):
        return None
    from src.recommender import load_places

    for row in load_places().itertuples():
        if row.name.lower() in text:
            return row.place_id
    return None
