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
    s = DialogueState(slots={"cuisine": "korean", "area": "downtown core"}, excluded_place_ids=["P004"], budget_max=13)
    ids = {c["place_id"] for c in recommender.candidates(s)}
    assert "P004" not in ids
    assert all(c["price_min"] <= 13 for c in recommender.candidates(s))


def test_radius_shrinks_results():
    wide = DialogueState(slots={"cuisine": "korean", "area": "downtown core"})
    near = DialogueState(slots={"cuisine": "korean", "area": "downtown core"}, radius_km=1.0)
    assert len(recommender.candidates(near)) < len(recommender.candidates(wide))


def test_rank_sorted_and_k():
    ranked = recommender.rank(DialogueState(), "", None, k=3)
    assert len(ranked) == 3
    assert [p.score for p in ranked] == sorted([p.score for p in ranked], reverse=True)
    assert all(p.reasons for p in ranked)


def test_region_keeps_its_planning_areas():
    east = recommender.candidates(DialogueState(slots={"area": "east"}, radius_km=0.1))
    assert {c["area"] for c in east} == {"tampines"}  # no radius filter: a region has no centre
    central = recommender.candidates(DialogueState(slots={"area": "central"}))
    assert {c["area"] for c in central} == {"downtown core", "river valley", "outram"}


def test_dataset_values_find_places():
    from src.schemas import TrackerOutput
    from src.tracker.values import translate

    out = translate(TrackerOutput(slots={"cuisine": "Western", "area": "Bugis",
                                         "budget": "below $10", "time": "19:30"}))
    s = DialogueState(slots=out.slots)
    assert [p.name for p in recommender.rank(s, "", None)] == ["Sunny Veggie Bowl"]
    malay_east = DialogueState(slots=translate(TrackerOutput(slots={"cuisine": "Malay", "area": "east"})).slots)
    assert [c["place_id"] for c in recommender.candidates(malay_east)] == ["P012"]


@pytest.fixture
def real_places(monkeypatch):
    """The recommender running on data/master/places.csv instead of the test sample."""
    from src.config import DATA_DIR
    from src.recommender import stub

    df = stub.load_places.__wrapped__(DATA_DIR / "master" / "places.csv")
    monkeypatch.setattr(stub, "load_places", lambda: df)
    stub.area_centroids.cache_clear()
    yield df
    stub.area_centroids.cache_clear()


def test_real_places_csv_reads_in_tracker_values(real_places):
    assert len(real_places["area"].unique()) == 30 and set(real_places["area"]) >= {"ang mo kio", "downtown core"}
    assert set(real_places["region"]) == {"central", "east", "west", "north", "north-east"}
    assert not real_places["halal"].any() and not real_places["vegetarian"].any()  # blank -> not halal


def test_real_places_csv_area_and_region(real_places):
    amk = recommender.candidates(DialogueState(slots={"area": "ang mo kio"}))
    assert amk and {c["area"] for c in amk} == {"ang mo kio"}
    assert all(c["dist_km"] <= 2.0 for c in amk)  # radius from the area's average position
    east = recommender.candidates(DialogueState(slots={"area": "east"}))
    assert len(east) == (real_places["region"] == "east").sum()
    ranked = recommender.rank(DialogueState(slots={"area": "ang mo kio"}), "", None, k=3)
    assert len(ranked) == 3 and all(p.score == 0.0 for p in ranked)  # no ratings yet
