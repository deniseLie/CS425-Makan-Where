"""Keyword/regex tracker stub. Only looks at the latest user turn and carries
over everything else from `prev`."""

import re

from src.schemas import ANY, REQUIRED_SLOTS, SOFT_SLOTS, UNKNOWN, DialogueState, TrackerOutput, Turn

ALIASES: dict[str, dict[str, str]] = {
    "cuisine": {
        "korean": "korean", "kbbq": "korean", "kimchi": "korean",
        "japanese": "japanese", "sushi": "japanese", "ramen": "japanese",
        "chinese": "chinese", "zi char": "chinese", "zichar": "chinese", "chicken rice": "chinese",
        "malay": "malay", "nasi lemak": "malay", "nasi padang": "malay",
        "indian": "indian", "prata": "indian", "biryani": "indian",
        "western": "western", "burger": "western", "pasta": "western",
        "thai": "thai",
    },
    "area": {
        "bugis": "bugis", "kampong glam": "bugis",
        "orchard": "orchard",
        "tampines": "tampines",
        "jurong east": "jurong east", "jurong": "jurong east", "je": "jurong east",
        "chinatown": "chinatown",
    },
    "dietary": {
        "halal": "halal", "muslim": "halal", "no pork": "halal",
        "vegetarian": "vegetarian", "veggie": "vegetarian", "vegan": "vegetarian",
    },
    "setting": {
        "hawker": "hawker", "kopitiam": "hawker", "coffeeshop": "hawker",
        "food court": "foodcourt", "foodcourt": "foodcourt",
        "cafe": "cafe", "restaurant": "restaurant", "aircon": "restaurant",
    },
    "time": {
        "breakfast": "breakfast", "lunch": "lunch", "dinner": "dinner",
        "supper": "late_night", "late night": "late_night",
    },
    "budget": {
        "cheap": "<=8", "budget": "<=8", "affordable": "<=8",
    },
}

ANY_PHRASES = ["anything also can", "anything can", "any also can", "anything",
               "whatever", "dont mind", "don't mind", "no preference", "up to you"]
CRITIQUE_PHRASES = {
    "too_far": ["too far", "so far", "far leh", "far lah", "nearer", "closer"],
    "too_expensive": ["too expensive", "too ex", "so ex", "pricey", "cheaper", "too pricey"],
}
_BUDGET_RX = re.compile(r"(?:under|below|less than|within|<=?|max)\s*\$?\s*(\d+)", re.IGNORECASE)


def _has(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None


def track(history: list[Turn], prev: DialogueState) -> TrackerOutput:
    turn = history[-1]
    text = (turn.normalized or turn.raw).lower()
    slots = dict(prev.slots)
    mentioned = False

    for slot, table in ALIASES.items():
        # longest alias first so "jurong east" beats "jurong"
        for alias in sorted(table, key=len, reverse=True):
            if _has(text, alias):
                slots[slot] = table[alias]
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
    from src.recommender import load_places  # lazy: stub-only coupling

    for row in load_places().itertuples():
        if row.name.lower() in text:
            return row.place_id
    return None
