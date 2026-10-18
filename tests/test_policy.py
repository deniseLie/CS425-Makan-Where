from types import SimpleNamespace

from src import recommender
from src.policy import apply_tracker_output, decide
from src.schemas import ANY, DialogueState, RankedPlace, TrackerOutput


def state(**slots) -> DialogueState:
    return DialogueState(slots=slots)


def test_asks_cuisine_when_missing():
    a = decide(state(), 0, recommender)
    assert a.kind == "ask" and a.ask_slot == "cuisine"
    assert len(a.ask_options) == 3


def test_asks_area_when_only_cuisine_known():
    a = decide(state(cuisine="korean"), 1, recommender)
    assert a.kind == "ask" and a.ask_slot == "area"
    assert "bugis" in a.ask_options  # options come from korean candidates


def test_any_counts_as_answered():
    a = decide(state(cuisine="korean", area=ANY), 0, recommender)
    assert a.kind == "recommend"


def test_recommends_when_required_filled():
    a = decide(state(cuisine="korean", area="bugis"), 1, recommender)
    assert a.kind == "recommend"
    assert 1 <= len(a.places) <= 3
    assert all(isinstance(p, RankedPlace) for p in a.places)


def test_stops_asking_after_three():
    a = decide(state(), 3, recommender)
    assert a.kind == "recommend" and a.places


def test_fallback_when_no_candidates():
    # no korean hawker in bugis within 1 km -> relax setting
    s = DialogueState(slots={"cuisine": "korean", "area": "bugis", "setting": "hawker"}, radius_km=1.0)
    assert recommender.candidates(s) == []
    a = decide(s, 1, recommender)
    assert a.kind == "fallback_recommend"
    assert a.relaxed_slot == "setting"
    assert a.places


def test_fallback_relaxes_budget_last():
    s = DialogueState(slots={"cuisine": "western", "area": "orchard", "budget": "<=5"}, budget_max=5.0)
    a = decide(s, 1, recommender)
    assert a.kind == "fallback_recommend" and a.relaxed_slot == "budget" and a.places


def test_fallback_with_nothing_at_all():
    a = decide(state(cuisine="martian", area="bugis"), 1, recommender)
    assert a.kind == "fallback_recommend" and a.places == []


def test_entropy_ask_when_many_candidates():
    rows = [{"cuisine": "korean", "area": "bugis", "dietary": d, "setting": "restaurant"}
            for d in ["halal", "none", "vegetarian"] * 4]  # 12 rows; dietary varies, setting doesn't
    fake = SimpleNamespace(candidates=lambda st, user_latlon=None: rows,
                           rank=lambda *a, **k: [])
    a = decide(state(cuisine="korean", area="bugis"), 1, fake)
    assert a.kind == "ask" and a.ask_slot == "dietary"
    assert set(a.ask_options) == {"halal", "vegetarian"}


def test_critiques():
    prev = DialogueState(slots={"cuisine": "korean", "area": "bugis"})
    out = TrackerOutput(slots=prev.slots, critique="too_far", reject="P002")
    s = apply_tracker_output(prev, out)
    assert s.radius_km == 1.0 and s.excluded_place_ids == ["P002"]

    s2 = apply_tracker_output(s, TrackerOutput(slots={**s.slots, "budget": "<=20"}))
    assert s2.budget_max == 20.0
    s3 = apply_tracker_output(s2, TrackerOutput(slots=s2.slots, critique="too_expensive"))
    assert s3.budget_max == 14.0 and s3.slots["budget"] == "<=14"
    assert prev.radius_km == 2.0  # prev untouched
