"""Tracker value translator (src/tracker/values.py): one test per table row."""

import json

import pytest

from src.config import DATA_DIR
from src.schemas import ANY, SLOTS, UNKNOWN, TrackerOutput
from src.tracker.values import (CUISINES, LANDMARKS, PLANNING_AREAS, REGIONS, TABLES, meal_of,
                                place_value, translate)


@pytest.mark.parametrize("slot,value,expected", [
    # cuisine: lowercase, dishes -> cuisine
    ("cuisine", "Western", "western"),
    ("cuisine", "chicken rice", "chinese"),
    ("cuisine", "nasi lemak", "malay"),
    ("cuisine", "roti prata", "indian"),
    # area: planning area -> lowercase, landmark -> its planning area, region stays a region
    ("area", "Ang Mo Kio", "ang mo kio"),
    ("area", "Bugis", "downtown core"),
    ("area", "Orchard", "river valley"),
    ("area", "NUS", "queenstown"),
    ("area", "east", "east"),
    ("area", "north-east", "north-east"),
    # budget
    ("budget", "below $5", "<=5"),
    ("budget", "below $10", "<=10"),
    ("budget", "$10-20", "<=20"),
    ("budget", "above $20", ANY),
    # dietary
    ("dietary", "halal", "halal"),
    ("dietary", "vegetarian", "vegetarian"),
    ("dietary", "vegan", "vegetarian"),
    ("dietary", "no pork", "halal"),
    ("dietary", "no beef", "no beef"),
    # setting
    ("setting", "hawker", "hawker"),
    ("setting", "aircon", "aircon"),
    ("setting", "cafe", "cafe"),
    ("setting", "food court", "food court"),
    # time: meal words stay, exact times become HH:MM
    ("time", "supper", "late night"),
    ("time", "late night", "late night"),
    ("time", "dinner", "dinner"),
    ("time", "now", "now"),
    ("time", "7pm", "19:00"),
    ("time", "7.30pm", "19:30"),
    ("time", "19:30", "19:30"),
    ("time", "12.30", "12:30"),
    ("time", "8 am", "08:00"),
    ("time", "12am", "00:00"),
    ("time", "12pm", "12:00"),
])
def test_table_rows(slot, value, expected):
    assert place_value(slot, value) == expected


@pytest.mark.parametrize("slot", SLOTS)
def test_unknown_and_any_untouched(slot):
    assert place_value(slot, UNKNOWN) == UNKNOWN
    assert place_value(slot, ANY) == ANY


def test_not_in_tables_is_lowercased_and_passed_through():
    assert place_value("cuisine", "Italian") == "italian"
    assert place_value("area", "Sentosa") == "sentosa"
    assert place_value("time", "25:00") == "25:00"
    assert place_value("time", "13pm") == "13pm"


@pytest.mark.parametrize("slot", SLOTS)
def test_already_translated_values_unchanged(slot):
    outputs = {v for v in TABLES[slot].values()} | {"<=8", "<=10", "19:00"}
    for v in outputs:
        once = place_value(slot, v)
        assert place_value(slot, once) == once, (slot, v)
    for v in TABLES[slot].values():
        assert place_value(slot, v) == v, (slot, v)


def test_landmarks_land_in_planning_areas():
    assert set(LANDMARKS.values()) <= set(PLANNING_AREAS)
    assert len(PLANNING_AREAS) == 30


@pytest.mark.parametrize("time,meal", [
    ("05:00", "breakfast"), ("10:59", "breakfast"),
    ("11:00", "lunch"), ("16:59", "lunch"),
    ("17:00", "dinner"), ("19:30", "dinner"), ("22:59", "dinner"),
    ("23:00", "late night"), ("00:30", "late night"), ("04:59", "late night"),
    ("7pm", "dinner"),
])
def test_meal_of(time, meal):
    assert meal_of(time) == meal


def test_meal_of_rejects_non_times():
    with pytest.raises(ValueError):
        meal_of("dinner")


def test_translate_keeps_reject_and_critique():
    out = TrackerOutput(slots={"cuisine": "Western", "area": "Bugis", "budget": "below $10"},
                        reject="P001", critique="too_far")
    t = translate(out)
    assert t.slots["cuisine"] == "western" and t.slots["area"] == "downtown core"
    assert t.slots["budget"] == "<=10" and t.slots["time"] == UNKNOWN
    assert t.reject == "P001" and t.critique == "too_far"
    assert out.slots["area"] == "Bugis"  # input untouched


def test_every_synthetic_value_is_known():
    """Every value the synthetic data uses translates into place-list form."""
    paths = sorted((DATA_DIR / "synthetic_data").glob("*.jsonl"))
    if not paths:
        pytest.skip("no synthetic data")
    known = {
        "cuisine": set(CUISINES),
        "area": set(PLANNING_AREAS) | set(REGIONS),
        "budget": {"<=5", "<=10", "<=20"},
        "dietary": {"halal", "vegetarian", "no beef"},
        "setting": {"hawker", "aircon", "cafe", "food court"},
    }
    unknown = set()
    for path in paths:
        for line in path.open(encoding="utf-8"):
            gold = json.loads(json.loads(line)["output"])
            for slot, allowed in known.items():
                v = place_value(slot, gold[slot])
                if v not in allowed | {UNKNOWN, ANY}:
                    unknown.add((slot, gold[slot]))
            t = place_value("time", gold["time"])
            if t not in {UNKNOWN, ANY, "breakfast", "lunch", "dinner", "late night", "now"}:
                meal_of(t)  # must be a clock time
    assert not unknown, sorted(unknown)
