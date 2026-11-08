"""Keyword/regex tracker stub. Only looks at the latest user turn and carries
over everything else from `prev`. Its phrases come from the translator's tables
(values.py), so it writes place-list values directly."""

import re

from src.schemas import ANY, REQUIRED_SLOTS, SOFT_SLOTS, UNKNOWN, DialogueState, TrackerOutput, Turn

from .values import TABLES, budget_update

PHRASES: dict[str, dict[str, str]] = {
    slot: table for slot, table in TABLES.items() if slot != "budget"
}

ANY_PHRASES = ["anything also can", "anything can", "any also can", "anything",
               "whatever", "dont mind", "don't mind", "no preference", "up to you"]
CRITIQUE_PHRASES = {
    "too_far": ["too far", "so far", "far leh", "far lah", "nearer", "closer"],
    "too_expensive": ["too expensive", "too ex", "so ex", "pricey", "cheaper", "too pricey"],
    "not_interested": ["dun want this one", "don't want this one", "dont want this one",
                       "not this one", "next one", "something else"],
}


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

    budget = budget_update(turn.raw)
    if budget is not None:
        slots["budget"] = budget
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
    from src.recommender import load_places  # lazy: stub-only coupling

    for row in load_places().itertuples():
        if row.name.lower() in text:
            return row.place_id
    return None
