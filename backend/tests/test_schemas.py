import pytest

from src.schemas import (ANY, SLOTS, UNKNOWN, DialogueState, PolicyAction, RankedPlace, TrackerOutput,
                         Turn)


def test_defaults():
    s = DialogueState()
    assert list(s.slots) == SLOTS
    assert all(v == UNKNOWN for v in s.slots.values())
    assert s.radius_km == 2.0 and s.budget_max is None and s.excluded_place_ids == []


@pytest.mark.parametrize("obj", [
    Turn(raw="korean lah", normalized="korean", style="singlish", reply="ok"),
    DialogueState(slots={"cuisine": "korean", "area": ANY}, excluded_place_ids=["P001"], radius_km=1.0,
                  budget_max=8.0),
    TrackerOutput(slots={"cuisine": "korean"}, reject="P002", critique="too_far"),
    RankedPlace(place_id="P001", name="X", score=4.3, reasons=["halal", "~$5"]),
    PolicyAction(kind="recommend", places=[RankedPlace("P001", "X", 4.3, ["~$5"])]),
    PolicyAction(kind="ask", ask_slot="area", ask_options=["downtown core", "river valley"]),
    PolicyAction(kind="fallback_recommend", relaxed_slot="setting"),
])
def test_round_trip(obj):
    cls = type(obj)
    assert cls.from_dict(obj.to_dict()) == obj
    assert cls.from_json(obj.to_json()) == obj


def test_partial_slots_filled_with_unknown():
    s = DialogueState.from_dict({"slots": {"cuisine": "korean"}})
    assert s.slots["area"] == UNKNOWN and s.slots["cuisine"] == "korean"


def test_copy_is_deep():
    s = DialogueState()
    c = s.copy()
    c.slots["cuisine"] = "thai"
    c.excluded_place_ids.append("P1")
    assert s.slots["cuisine"] == UNKNOWN and s.excluded_place_ids == []


@pytest.mark.parametrize("bad", [
    lambda: DialogueState(slots={"colour": "red"}),
    lambda: TrackerOutput(critique="too_salty"),
    lambda: PolicyAction(kind="dance"),
    lambda: Turn(raw="x", style="klingon"),
])
def test_validation(bad):
    with pytest.raises((ValueError, TypeError)):
        bad()
