"""Translate tracker slot values into place-list values. Owner: Denise, reviewed by Nichole.

The [TRACK] dataset, and so the trained tracker, writes what the user said: "Western",
"Ang Mo Kio", "Bugis", "below $10", "7pm". The recommender filters on place-list values:
"western", "ang mo kio", "downtown core", "<=10", "19:00". The pipeline calls translate()
right after track().

Two rules:
- Values already in place-list form map to themselves, so translating twice changes nothing.
- Anything not in the tables is lowercased and passed through. The recommender then finds no
  match, and the policy's fallback handles it.
"""

from __future__ import annotations

import re

from src.schemas import ANY, UNKNOWN, TrackerOutput

CUISINES = ("chinese", "malay", "indian", "korean", "japanese", "thai", "western", "vietnamese")

# The 30 URA planning areas in the NEA establishments data (and data/area_pool.py).
PLANNING_AREAS = (
    "downtown core", "bukit merah", "bishan", "toa payoh", "geylang", "kallang", "bukit timah",
    "tanglin", "outram", "river valley", "queenstown", "novena", "marine parade",
    "tampines", "bedok", "pasir ris",
    "clementi", "bukit batok", "jurong west", "jurong east", "choa chu kang", "bukit panjang",
    "sembawang", "yishun", "woodlands",
    "hougang", "ang mo kio", "serangoon", "sengkang", "punggol",
)
# A region stays a region; the recommender expands it to its planning areas.
REGIONS = ("central", "east", "west", "north", "north-east")

# Landmarks people say -> the planning area the NEA data files the places around them under
# (the most common area among NEA rows within 500 m). NEA uses 30 planning areas, so places in
# Orchard, Rochor, Singapore River and Changi are filed under a neighbouring area.
LANDMARKS = {
    "bugis": "downtown core", "clarke quay": "downtown core", "smu": "downtown core",
    "chinatown": "outram",
    "orchard": "river valley",
    "little india": "kallang", "kampong glam": "kallang",
    "holland village": "queenstown", "nus": "queenstown",
    "katong": "marine parade",
    "paya lebar": "geylang",
    "ntu": "jurong west", "jurong point": "jurong west",
    "changi": "pasir ris",  # no NEA rows at Changi itself; the nearest are in Pasir Ris
    "vivocity": "bukit merah",
    "jurong": "jurong east", "je": "jurong east",
}

DISHES = {
    "chicken rice": "chinese", "bak kut teh": "chinese", "char kway teow": "chinese",
    "hokkien mee": "chinese", "laksa": "chinese", "zi char": "chinese", "zichar": "chinese",
    "nasi lemak": "malay", "nasi padang": "malay",
    "roti prata": "indian", "prata": "indian", "biryani": "indian",
    "kbbq": "korean", "kimchi": "korean",
    "sushi": "japanese", "ramen": "japanese",
    "burger": "western", "pasta": "western",
}

MEALS = ("breakfast", "lunch", "dinner", "late night", "now")

# slot -> {value as the tracker writes it, lowercased: place-list value}
TABLES: dict[str, dict[str, str]] = {
    "cuisine": {**{c: c for c in CUISINES}, **DISHES},
    "area": {**{a: a for a in PLANNING_AREAS}, **{r: r for r in REGIONS},
             "northeast": "north-east", "north east": "north-east", **LANDMARKS},
    "budget": {"above $20": ANY},
    # vegan / no pork -> the closest value the place list can check; no beef has no filter yet.
    # non-vegetarian -> leave out the vegetarian-only places
    "dietary": {"halal": "halal", "muslim": "halal", "no pork": "halal",
                "vegetarian": "vegetarian", "veggie": "vegetarian", "vegan": "vegetarian",
                "no beef": "no beef",
                **{v: "non-vegetarian" for v in ("non-vegetarian", "non vegetarian", "non-veg",
                                                 "non veg", "nonveg", "not vegetarian")}},
    "setting": {"hawker": "hawker", "kopitiam": "hawker", "coffeeshop": "hawker",
                "aircon": "aircon", "cafe": "cafe", "food court": "food court", "foodcourt": "food court",
                "restaurant": "restaurant", "sit-down": "restaurant",
                "takeaway": "takeaway", "tapau": "takeaway", "dabao": "takeaway"},
    "time": {**{m: m for m in MEALS}, "supper": "late night", "late_night": "late night"},
}

# "19:30", "7pm", "7.30pm", "12.30", "8 am": minutes or am/pm (or both) make it a clock time.
_CLOCK_RX = re.compile(r"(\d{1,2})(?:[:.](\d{2}))?\s*([ap])?\.?m?\.?")

BUDGET_UNCLEAR = "budget amount needed"
_MONEY_UNIT = r"dollars?|bucks?"
_ONES = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen",
)
_TENS = ("twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_WORD_AMOUNTS = {word: number for number, word in enumerate(_ONES)}
_WORD_AMOUNTS.update({word: number * 10 for number, word in enumerate(_TENS, start=2)})
_SPOKEN_AMOUNT = re.compile(
    rf"\b(?:(?P<tens>{'|'.join(_TENS)})(?:[ -](?P<ones>{'|'.join(_ONES[1:10])}))?"
    rf"|(?P<simple>{'|'.join(_ONES)}))(?=\s+(?:{_MONEY_UNIT})\b)"
)
_AMOUNT = r"\d+(?:\.\d{1,2})?(?!\w|[.,]\d)"
_BUDGET_AMOUNT = (
    rf"(?:s?\$\s*)?(?P<first>{_AMOUNT})"
    rf"(?:\s*(?:[-–—]|\bto\b)\s*(?:s?\$\s*)?(?P<last>{_AMOUNT}))?"
    rf"(?:\s*(?:{_MONEY_UNIT})\b)?"
)
_BUDGET_PREFIX = (
    r"<=\s*|\b(?:max(?:imum)?(?:\s+of)?|at most|up to|under|below|"
    r"no more than|less than|within|don['’]t go over|budget(?:\s+(?:is|of|about|around|to))?)\s*"
)
_BUDGET_VALUE = re.compile(rf"(?:{_BUDGET_PREFIX})?{_BUDGET_AMOUNT}")
_BUDGET_MENTION = re.compile(
    rf"(?:{_BUDGET_PREFIX}|(?=s?\$)|(?=\d+(?:\.\d+)?\s*(?:{_MONEY_UNIT})\b)){_BUDGET_AMOUNT}"
)
_NO_CAP = re.compile(
    r"\b(?:any (?:budget|price)|no (?:budget|price) (?:limit|cap)|no limit on (?:the )?(?:budget|price)|"
    r"(?:budget|price) (?:does(?:n't| not) matter|dun matter|is flexible)|"
    r"(?:ignore|remove|drop) (?:the |my )?budget(?: cap)?|"
    r"(?:make|keep) (?:(?:my|our|the) )?budget flexible)\b"
)
_UNCLEAR_BUDGET = re.compile(r"\b(?:budget|cheap|cheaper|affordable|inexpensive|max(?:imum)? spend)\b")


def _budget_cap(match: re.Match) -> float:
    return float(match["last"] or match["first"])


def _spoken_amount(match: re.Match) -> str:
    if match["simple"]:
        return str(_WORD_AMOUNTS[match["simple"]])
    return str(_WORD_AMOUNTS[match["tens"]] + _WORD_AMOUNTS.get(match["ones"], 0))


def parse_budget(value: str) -> float | None:
    match = _BUDGET_VALUE.fullmatch(_SPOKEN_AMOUNT.sub(_spoken_amount, value.strip().lower()))
    return _budget_cap(match) if match else None


def budget_update(text: str, answering_budget: bool = False) -> str | None:
    """Recognize a budget instruction in the latest turn, independent of tracker state."""
    value = _SPOKEN_AMOUNT.sub(_spoken_amount, text.lower().strip().rstrip(".!?"))
    amounts = list(_BUDGET_MENTION.finditer(value))
    amounts = [
        match for match in amounts
        if not re.match(r"\s*(?:minutes?|mins?|hours?|hrs?|km|metres?|meters?|pax|people)\b",
                        value[match.end():])
    ]
    if len(amounts) == 1:
        match = amounts[0]
        # A minimum or an amount of extra spending needs clarification.
        if re.search(r"\b(?:at least|above|over|more than|another|extra)\s*$", value[:match.start()]) \
                or re.match(r"\s*(?:more|extra|or more)\b", value[match.end():]):
            return BUDGET_UNCLEAR
        return f"<={_budget_cap(match):g}"
    if not amounts and (_NO_CAP.search(value) or (
        answering_budget and value in {"any", "anything", "no limit", "no preference", "no need"}
    )):
        return ANY
    if answering_budget:
        cap = parse_budget(value)
        if cap is not None:
            return f"<={cap:g}"
    if amounts or "$" in value or answering_budget or _UNCLEAR_BUDGET.search(value):
        return BUDGET_UNCLEAR
    return None


_NO_DIETARY = re.compile(r"(?:no dietary (?:requirements|needs|restrictions)|no restrictions?)(?: lah)?")
_HALAL_OPTIONAL = re.compile(
    r"(?:non[- ]halal (?:is )?(?:fine|okay|ok|can|also can)|"
    r"(?:no need for|no need|don['’]t need|dun need) halal(?: also can)?)(?: lah)?"
)


def dietary_update(text: str, answering_dietary: bool, previous: str) -> str | None:
    value = text.lower().strip().rstrip(".!?")
    if _NO_DIETARY.fullmatch(value) or (
        answering_dietary and re.fullmatch(r"(?:no|nope|none|no need|no thanks)(?: lah)?", value)
    ):
        return ANY
    if _HALAL_OPTIONAL.fullmatch(value):
        return ANY if previous in {UNKNOWN, ANY, "halal", "non-halal"} else previous
    if re.fullmatch(r"(?:(?:i )?(?:only )?want |only )?non[- ]halal(?: (?:food|places?))?(?: only)?", value):
        return "non-halal"
    return None


def clock_time(value: str) -> str | None:
    """'7.30pm' -> '19:30'. None if the value isn't a clock time."""
    m = _CLOCK_RX.fullmatch(value.strip().lower())
    if not m or (m.group(2) is None and m.group(3) is None):
        return None
    hour, minute, half = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if half:
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if half == "p" else 0)
    if hour > 23 or minute > 59:
        return None
    return f"{hour:02d}:{minute:02d}"


def meal_of(time: str) -> str:
    """'19:30' -> 'dinner'. Breakfast 05:00-10:59, lunch 11:00-16:59, dinner 17:00-22:59,
    late night 23:00-04:59."""
    clock = clock_time(time)
    if clock is None:
        raise ValueError(f"not a clock time: {time!r}")
    hour = int(clock[:2])
    if 5 <= hour < 11:
        return "breakfast"
    if 11 <= hour < 17:
        return "lunch"
    if 17 <= hour < 23:
        return "dinner"
    return "late night"


def place_value(slot: str, value: str) -> str:
    """One tracker value -> its place-list value."""
    v = " ".join(str(value).split())
    if v.upper() in (UNKNOWN, ANY):
        return v.upper()
    low = v.lower()
    if slot == "budget":
        if _NO_CAP.fullmatch(low):
            return ANY
        cap = parse_budget(low)
        if cap is not None:
            return f"<={cap:g}"
    if slot == "dietary":
        return dietary_update(low, False, UNKNOWN) or TABLES[slot].get(low, low)
    if slot == "time":
        clock = clock_time(low)
        if clock:
            return clock
    return TABLES.get(slot, {}).get(low, low)


def translate(out: TrackerOutput) -> TrackerOutput:
    """TrackerOutput with dataset values -> the same output with place-list values."""
    slots = {s: place_value(s, v) for s, v in out.slots.items()}
    return TrackerOutput(slots=slots, reject=out.reject, critique=out.critique)
