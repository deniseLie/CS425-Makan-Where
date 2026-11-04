"""Translate tracker slot values into place-list values. Owner: Denise, reviewed by Nichole.

The [TRACK] dataset, and so the trained tracker, writes what the user said: "Western",
"Ang Mo Kio", "Bugis", "below $10", "7pm". The recommender filters on place-list values:
"western", "ang mo kio", "downtown core", "<=10", "19:00". The pipeline calls translate()
right after track().

Two rules:
- Values already in place-list form map to themselves, so translating twice changes nothing.
  The stub tracker writes place-list values directly and keeps working.
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
    "budget": {"below $5": "<=5", "below $10": "<=10", "$10-20": "<=20", "above $20": ANY},
    # vegan / no pork -> the closest value the place list can check; no beef has no filter yet
    "dietary": {"halal": "halal", "muslim": "halal", "no pork": "halal",
                "vegetarian": "vegetarian", "veggie": "vegetarian", "vegan": "vegetarian",
                "no beef": "no beef"},
    "setting": {"hawker": "hawker", "kopitiam": "hawker", "coffeeshop": "hawker",
                "aircon": "aircon", "cafe": "cafe", "food court": "food court", "foodcourt": "food court",
                "restaurant": "restaurant", "sit-down": "restaurant",
                "takeaway": "takeaway", "tapau": "takeaway", "dabao": "takeaway"},
    "time": {**{m: m for m in MEALS}, "supper": "late night", "late_night": "late night"},
}

# "19:30", "7pm", "7.30pm", "12.30", "8 am": minutes or am/pm (or both) make it a clock time.
_CLOCK_RX = re.compile(r"(\d{1,2})(?:[:.](\d{2}))?\s*([ap])?\.?m?\.?")


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
    if slot == "time":
        clock = clock_time(low)
        if clock:
            return clock
    return TABLES.get(slot, {}).get(low, low)


def translate(out: TrackerOutput) -> TrackerOutput:
    """TrackerOutput with dataset values -> the same output with place-list values."""
    slots = {s: place_value(s, v) for s, v in out.slots.items()}
    return TrackerOutput(slots=slots, reject=out.reject, critique=out.critique)
