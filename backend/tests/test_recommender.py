import itertools

import pytest

from src import recommender
from src.schemas import ANY, UNKNOWN, DialogueState

places = recommender.load_places()
AREAS = sorted(places["area"].unique()) + [UNKNOWN, ANY]
CUISINES = sorted(places["cuisine"].unique()) + [UNKNOWN, ANY]


@pytest.mark.parametrize("area,cuisine", list(itertools.product(AREAS, CUISINES)))
def test_halal_only_returns_halal(area, cuisine):
    s = DialogueState(slots={"area": area, "cuisine": cuisine, "dietary": "halal"}, radius_km=50)
    halal_ids = set(places.loc[places["halal"], "place_id"])
    assert all(c["place_id"] in halal_ids for c in recommender.candidates(s))
    assert all(p.place_id in halal_ids for p in recommender.rank(s, "", None, k=10))


def test_unknown_and_any_add_no_condition():
    assert len(recommender.candidates(DialogueState())) == len(places)
    assert len(recommender.candidates(DialogueState(slots={s: ANY for s in ["cuisine", "area"]}))) == len(places)


def test_excluded_ids_and_budget():
    s = DialogueState(slots={"cuisine": "korean", "area": "bugis"}, excluded_place_ids=["P004"], budget_max=13)
    ids = {c["place_id"] for c in recommender.candidates(s)}
    assert "P004" not in ids
    assert all(c["price_min"] <= 13 for c in recommender.candidates(s))


def test_radius_shrinks_results():
    wide = DialogueState(slots={"cuisine": "korean", "area": "bugis"})
    near = DialogueState(slots={"cuisine": "korean", "area": "bugis"}, radius_km=1.0)
    assert len(recommender.candidates(near)) < len(recommender.candidates(wide))


def test_rank_sorted_and_k():
    ranked = recommender.rank(DialogueState(), "", None, k=3)
    assert len(ranked) == 3
    assert [p.score for p in ranked] == sorted([p.score for p in ranked], reverse=True)
    assert all(p.reasons for p in ranked)
