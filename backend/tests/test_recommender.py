import itertools
import random

import pandas as pd
import pytest

from src import recommender
from src.config import DATA_DIR
from src.recommender import aliases, filters, hours, ranking
from src.schemas import ANY, UNKNOWN, DialogueState, TrackerOutput
from src.tracker.values import CUISINES, PLANNING_AREAS, REGIONS, TABLES, translate

places = recommender.load_places()
AREAS = sorted(places["area"].unique()) + [UNKNOWN, ANY]
SAMPLE_CUISINES = sorted(places["cuisine"].unique()) + [UNKNOWN, ANY]


# --------------------------------------------------------------------------------- hard filters

@pytest.mark.parametrize("area,cuisine", list(itertools.product(AREAS, SAMPLE_CUISINES)))
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


def test_user_location_is_the_radius_centre():
    bugis = (1.3000, 103.8558)
    near = recommender.candidates(DialogueState(radius_km=0.5), user_latlon=bugis)
    assert near and all(c["dist_km"] <= 0.5 for c in near)
    assert {c["area"] for c in near} == {"downtown core"}


def test_region_keeps_its_planning_areas():
    east = recommender.candidates(DialogueState(slots={"area": "east"}, radius_km=0.1))
    assert {c["area"] for c in east} == {"tampines"}  # no radius filter: a region has no centre
    central = recommender.candidates(DialogueState(slots={"area": "central"}))
    assert {c["area"] for c in central} == {"downtown core", "river valley", "outram"}


def test_no_beef_drops_beef_places():
    s = DialogueState(slots={"cuisine": "japanese", "area": "river valley", "dietary": "no beef"})
    assert recommender.candidates(DialogueState(slots={"cuisine": "japanese", "area": "river valley"}))
    assert all("Gyukatsu" not in c["name"] for c in recommender.candidates(s))


def test_aircon_keeps_unknown_and_drops_hawker():
    ids = {c["place_id"] for c in recommender.candidates(DialogueState(slots={"setting": "aircon"}))}
    assert "P001" in ids          # setting aircon
    assert "P008" in ids          # cafe: aircon unknown -> kept
    assert "P004" not in ids      # hawker


def test_setting_alias_and_value():
    a = {c["place_id"] for c in recommender.candidates(DialogueState(slots={"setting": "kopitiam"}))}
    b = {c["place_id"] for c in recommender.candidates(DialogueState(slots={"setting": "hawker"}))}
    assert a == b and a == set(places.loc[places["setting"] == "hawker", "place_id"])


def test_slot_values_go_through_the_alias_table():
    base = {c["place_id"] for c in recommender.candidates(DialogueState(slots={"cuisine": "korean", "area": "downtown core"}))}
    for cuisine, area in [("kbbq", "bugis"), ("Korean", "Bugis side"), ("korean", "near smu"), ("bibimbap", "town")]:
        got = {c["place_id"] for c in recommender.candidates(DialogueState(slots={"cuisine": cuisine, "area": area}))}
        assert got == base, (cuisine, area)


def test_sql_is_parameterized():
    evil = "korean' OR 1=1 --"
    sql, params = filters.build_query(DialogueState(slots={"cuisine": evil}))
    assert evil not in sql and evil.lower() in [str(p).lower() for p in params]
    assert recommender.candidates(DialogueState(slots={"cuisine": evil})) == []


def test_time_filter_uses_opening_hours(tmp_places):
    df = tmp_places([
        dict(place_id="A", name="Day Cafe", opening_hours="Mo-Su 08:00-17:00"),
        dict(place_id="B", name="Supper Prata", opening_hours="18:00-03:00"),
        dict(place_id="C", name="No Hours Known", opening_hours=""),
    ])
    late = {c["place_id"] for c in recommender.candidates(DialogueState(slots={"time": "late night"}))}
    assert late == {"B", "C"}
    lunch = {c["place_id"] for c in recommender.candidates(DialogueState(slots={"time": "12:30"}))}
    assert lunch == {"A", "C"}
    # known-open beats unknown in the soft score
    top = recommender.rank(DialogueState(slots={"time": "supper"}), "", None, k=2)
    assert [p.place_id for p in top] == ["B", "C"]


# --------------------------------------------------------------------------------------- ranking

def test_rank_sorted_and_k():
    ranked = recommender.rank(DialogueState(), "", None, k=3)
    assert len(ranked) == 3
    assert [p.score for p in ranked] == sorted([p.score for p in ranked], reverse=True)
    assert all(p.reasons for p in ranked)


def test_dataset_values_find_places():
    out = translate(TrackerOutput(slots={"cuisine": "Western", "area": "Bugis",
                                         "budget": "$10-20", "time": "19:30"}))
    s = DialogueState(slots=out.slots)
    assert [p.name for p in recommender.rank(s, "", None)] == ["Sunny Veggie Bowl"]
    malay_east = DialogueState(slots=translate(TrackerOutput(slots={"cuisine": "Malay", "area": "east"})).slots)
    assert [c["place_id"] for c in recommender.candidates(malay_east)] == ["P012"]


def test_content_similarity_follows_the_request():
    s = DialogueState(slots={"cuisine": "korean", "area": "downtown core"})
    assert recommender.rank(s, "I want kimchi jjigae", None, k=1)[0].name == "Bugis Kimchi House"
    assert recommender.rank(s, "korean bbq for a group celebration", None, k=1)[0].name == "Hanok BBQ Bugis"
    assert recommender.rank(s, "halal fried chicken", None, k=1)[0].name == "Daebak Halal Korean"


def test_bayesian_quality_beats_few_perfect_reviews(tmp_places):
    tmp_places([dict(place_id="FEW", name="Few Reviews", rating=5.0, n_reviews=3),
                dict(place_id="MANY", name="Many Reviews", rating=4.5, n_reviews=3000)]
               + [dict(place_id=f"AVG{i}", name=f"Average {i}", rating=3.9, n_reviews=200) for i in range(4)])
    assert recommender.rank(DialogueState(), "", None, k=1)[0].place_id == "MANY"


def test_cheap_prefers_lower_prices(tmp_places):
    tmp_places([dict(place_id="LOW", name="Cheap Eats", price_min=3, price_max=6),
                dict(place_id="MID", name="Mid Eats", price_min=10, price_max=18),
                dict(place_id="HIGH", name="Atas Eats", price_min=40, price_max=80)])
    assert [p.place_id for p in recommender.rank(DialogueState(slots={"budget": "cheap cheap"}), "", None)] \
        == ["LOW", "MID", "HIGH"]
    assert recommender.rank(DialogueState(slots={"budget": "atas"}), "", None, k=1)[0].place_id == "HIGH"


def test_budget_filters_places_over_cap_before_ranking(tmp_places):
    tmp_places([dict(place_id="FITS", name="Fits", price_min=5, price_max=9),
                dict(place_id="PART", name="Partly", price_min=6, price_max=13)])
    s = DialogueState(slots={"budget": "<=10"}, budget_max=10)
    df = ranking.score(filters.filtered(s), s).set_index("place_id")
    assert list(df.index) == ["FITS"] and df.loc["FITS", "s_soft"] == 1.0


def test_distance_decay_prefers_nearer(tmp_places):
    tmp_places([dict(place_id="NEAR", name="Near", lat=1.3000, lon=103.8558),
                dict(place_id="FAR", name="Far", lat=1.3100, lon=103.8558)])
    ranked = recommender.rank(DialogueState(radius_km=5), "", (1.3000, 103.8558))
    assert [p.place_id for p in ranked] == ["NEAR", "FAR"] and ranked[0].score > ranked[1].score


def test_top_k_spreads_over_names(tmp_places):
    tmp_places([dict(place_id=f"C{i}", name="Chain Coffeeshop", building=f"Blk {i}", rating=4.8, n_reviews=900)
                for i in range(3)] + [dict(place_id="OTHER", name="Other", rating=4.0, n_reviews=100)])
    names = [p.name for p in recommender.rank(DialogueState(), "", None, k=2)]
    assert names == ["Chain Coffeeshop", "Other"]


def test_top_k_spreads_over_buildings(tmp_places):
    tmp_places([dict(place_id=f"S{i}", name=f"Stall {i}", building="Maxwell Food Centre", rating=4.8, n_reviews=900)
                for i in range(3)] + [dict(place_id="OTHER", name="Other", building="Amoy Street", rating=4.0, n_reviews=100)])
    ids = [p.place_id for p in recommender.rank(DialogueState(), "", None, k=2)]
    assert ids[0].startswith("S") and ids[1] == "OTHER"
    assert len(recommender.rank(DialogueState(), "", None, k=4)) == 4  # still fills k when it must


def test_weights_can_be_changed_for_ablations(tmp_places):
    tmp_places([dict(place_id="RATED", name="Rated", rating=4.9, n_reviews=2000, lat=1.31, lon=103.8558),
                dict(place_id="CLOSE", name="Close", rating=3.5, n_reviews=2000, lat=1.30, lon=103.8558)])
    me = (1.3000, 103.8558)
    only_quality = {"content": 0, "soft": 0, "quality": 1, "distance": 0}
    only_distance = {"content": 0, "soft": 0, "quality": 0, "distance": 1}
    s = DialogueState(radius_km=5)
    assert ranking.rank(s, "", me, k=1, weights=only_quality)[0].place_id == "RATED"
    assert ranking.rank(s, "", me, k=1, weights=only_distance)[0].place_id == "CLOSE"


def test_estimated_prices_are_marked(tmp_places):
    tmp_places([dict(place_id="E", name="Estimate", price_min=3, price_max=8, price_source="estimate")])
    assert recommender.rank(DialogueState(), "", None)[0].reasons[0] == "~$6 (est.)"


# --------------------------------------------------------------------------------------- aliases

@pytest.mark.parametrize("slot,value,expected", [
    ("cuisine", "kbbq", "korean"), ("cuisine", "chiken rice", "chinese"), ("cuisine", "chicken rce", "chinese"),
    ("cuisine", "roti canai", "indian"), ("cuisine", "briyani", "indian"), ("cuisine", "nasi lemak", "malay"),
    ("cuisine", "mookata", "thai"), ("cuisine", "Italian", "western"), ("cuisine", "korean", "korean"),
    ("area", "amk", "ang mo kio"), ("area", "Bugis side", "downtown core"), ("area", "near SMU", "downtown core"),
    ("area", "tiong bahru", "bukit merah"), ("area", "east side", "east"), ("area", "jurong west", "jurong west"),
    ("setting", "tapau", "takeaway"), ("setting", "aircon one", "aircon"), ("setting", "kopitiam", "hawker"),
    ("time", "supper", "late night"), ("time", "19:30", "19:30"), ("dietary", "muslim", "halal"),
    ("dietary", "vegan", "vegetarian"), ("budget", "cheap cheap", "cheap"), ("budget", "<=10", "<=10"),
    ("cuisine", UNKNOWN, UNKNOWN), ("area", ANY, ANY), ("cuisine", "martian", "martian"),
])
def test_alias_canonical(slot, value, expected):
    assert aliases.canonical(slot, value) == expected


def test_alias_table_agrees_with_values_py():
    for slot, table in TABLES.items():
        if slot == "budget":
            continue
        for said, place_value in table.items():
            assert aliases.canonical(slot, said) == place_value, (slot, said)


def test_alias_canonicals_are_place_list_values():
    df = aliases.load_aliases()
    assert not df.duplicated(["slot", "alias"]).any()
    assert set(df.loc[df["slot"] == "cuisine", "canonical"]) <= set(CUISINES)
    assert set(df.loc[df["slot"] == "area", "canonical"]) <= set(PLANNING_AREAS) | set(REGIONS)
    assert set(df.loc[df["slot"] == "setting", "canonical"]) <= {"hawker", "aircon", "cafe", "food court",
                                                                  "restaurant", "takeaway"}


def test_expand_query():
    assert aliases.expand_query("got kbbq near bugis?").endswith("korean")
    assert aliases.expand_query("korean bbq") == "korean bbq"  # already there
    assert aliases.landmark_point("bugis side") == pytest.approx((1.3008, 103.8559))


# ----------------------------------------------------------------------------------------- hours

@pytest.mark.parametrize("h,t,expected", [
    ("Mo-Su 10:00-22:00", "12:00", True), ("Mo-Su 10:00-22:00", "22:00", False),
    ("11:30-14:30,17:30-22:00", "15:00", False), ("11:30-14:30,17:30-22:00", "18:00", True),
    ("18:00-02:00", "01:00", True), ("18:00-02:00", "03:00", False), ("24/7", "04:00", True),
    ("Mo-Fr 07:00-20:00; Sa,Su off", "08:00", True), ("", "08:00", None), ("by appointment", "08:00", None),
])
def test_is_open(h, t, expected):
    assert hours.is_open(h, t) is expected


# ------------------------------------------------------------------------------- real places.csv

@pytest.fixture
def real_places():
    """The recommender running on data/master/places.csv instead of the test sample."""
    recommender.use_places_csv(DATA_DIR / "master" / "places.csv")
    yield recommender.load_places()
    recommender.use_places_csv(None)


@pytest.fixture
def tmp_places(tmp_path):
    """Write a small catalogue (defaults filled in) and point the recommender at it."""
    def make(rows):
        base = dict(lat=1.3000, lon=103.8558, area="DOWNTOWN CORE", region="Central", cuisine="korean",
                    halal=0, vegetarian=0, setting="restaurant", price_min=10, price_max=20, rating=4.0,
                    n_reviews=100, description="", building="", opening_hours="")
        df = pd.DataFrame([{**base, **r} for r in rows])
        path = tmp_path / "places.csv"
        df.to_csv(path, index=False)
        recommender.use_places_csv(path)
        return df
    yield make
    recommender.use_places_csv(None)


def test_real_places_csv_reads_in_tracker_values(real_places):
    assert len(real_places["area"].unique()) == 30 and set(real_places["area"]) >= {"ang mo kio", "downtown core"}
    assert set(real_places["region"]) == {"central", "east", "west", "north", "north-east"}
    assert set(real_places["setting"]) <= {"hawker", "food court", "cafe", "restaurant", "takeaway"}
    assert real_places["place_id"].is_unique


def test_unnamed_places_are_excluded(tmp_places):
    tmp_places([dict(place_id="MISSING", name=None), dict(place_id="BLANK", name="  "),
                dict(place_id="NAMED", name="Named Place")])
    assert recommender.place_names() == ["Named Place"]
    assert [row["place_id"] for row in recommender.candidates(DialogueState())] == ["NAMED"]


def test_real_places_csv_is_cleaned(real_places):
    # the cleaning rules are for NEA licences; OSM / Google places are public eateries already
    nea = real_places[real_places["place_id"].str.startswith("P")]
    addr = nea["address"].str.upper()
    campus = (r"UNIVERSITY|POLYTECHNIC|BUSINESS SCHOOL|SCHOOL OF MEDICINE|\bSMU\b|SCHOOL OF THE ARTS"
              r"|\bSOTA\b|\bITE\b|INSTITUTE OF TECHNICAL EDUCATION")
    assert not (addr.str.contains("SCHOOL|JUNIOR COLLEGE") & ~addr.str.contains(campus)).any()
    assert not addr.str.contains(r"\b(?:KITCHEN|BANQUET|STAFF|CAMP)\b").any()
    assert real_places["licence_number"].dropna().is_unique
    assert real_places["postal_code"].dropna().str.fullmatch(r"\d{6}").all()
    # a stallholder's own name is never shown as the place name
    from_address = real_places["name_source"] == "address"
    assert (real_places.loc[from_address, "name"] != real_places.loc[from_address, "licensee"]).all()


def test_real_places_csv_area_and_region(real_places):
    amk = recommender.candidates(DialogueState(slots={"area": "ang mo kio"}))
    assert amk and {c["area"] for c in amk} == {"ang mo kio"}
    assert all(c["dist_km"] <= 2.0 for c in amk)  # radius from the area's average position
    east = recommender.candidates(DialogueState(slots={"area": "east"}))
    assert len(east) == (real_places["region"] == "east").sum()
    ranked = recommender.rank(DialogueState(slots={"area": "ang mo kio"}), "", None, k=3)
    assert len(ranked) == 3 and len({p.name for p in ranked}) == 3


def test_real_places_csv_cuisine_request(real_places):
    s = DialogueState(slots=translate(TrackerOutput(slots={"cuisine": "Japanese", "area": "Bugis"})).slots,
                      radius_km=3)
    ranked = recommender.rank(s, "Is there sushi near Bugis?", None, k=3)
    assert ranked
    by_id = real_places.set_index("place_id")
    assert all(by_id.loc[p.place_id, "cuisine"] == "japanese" or "japanese" in by_id.loc[p.place_id, "tags"]
               for p in ranked)


def test_real_places_hard_constraints_never_violated(real_places):
    """Random slot combinations: every returned place satisfies every hard constraint."""
    rng = random.Random(0)
    by_id = real_places.set_index("place_id")
    cuisines = [UNKNOWN, ANY, *CUISINES]
    areas = [UNKNOWN, ANY, *PLANNING_AREAS, *REGIONS]
    for _ in range(150):
        slots = {"cuisine": rng.choice(cuisines), "area": rng.choice(areas),
                 "dietary": rng.choice([UNKNOWN, "vegetarian", "halal"]),
                 "setting": rng.choice([UNKNOWN, "hawker", "cafe", "aircon"]),
                 "budget": rng.choice([UNKNOWN, "<=5", "<=10", "<=20"])}
        s = DialogueState(slots=slots, radius_km=rng.choice([1.0, 2.0, 4.0]))
        cap = filters.budget_cap(s)
        for p in recommender.rank(s, "", None, k=5):
            row = by_id.loc[p.place_id]
            if slots["dietary"] == "vegetarian":
                assert row["vegetarian"]
            if slots["dietary"] == "halal":
                assert row["halal"]
            if slots["cuisine"] in CUISINES:
                assert row["cuisine"] == slots["cuisine"] or slots["cuisine"] in row["tags"].split(";")
            if slots["area"] in PLANNING_AREAS:
                assert row["area"] == slots["area"]
            if slots["area"] in REGIONS:
                assert row["region"] == slots["area"]
            if slots["setting"] in ("hawker", "cafe"):
                assert row["setting"] == slots["setting"]
            if slots["setting"] == "aircon":
                assert row["aircon"] != 0
            if cap is not None:
                assert row["price_min"] <= cap


# --------------------------------------------------------------------------- budget: upper price

def test_budget_checks_the_upper_price(tmp_places):
    tmp_places([dict(place_id="WIDE", name="One To Ten", price_min=1, price_max=10),      # ~$6
                dict(place_id="CHEAP", name="Three To Six", price_min=3, price_max=6),    # ~$4
                dict(place_id="NOTOP", name="Only Min", price_min=5, price_max=None),     # $5
                dict(place_id="NOPRICE", name="No Price", price_min=None, price_max=None)])
    s = DialogueState(slots={"budget": "<=5"}, budget_max=5)
    ids = {c["place_id"] for c in recommender.candidates(s)}
    assert ids == {"NOTOP"}
    # the price the reply shows is never above the budget
    for p in recommender.rank(s, "", None, k=5):
        shown = [r for r in p.reasons if r.startswith("~$")]
        assert not shown or float(shown[0][2:].split()[0]) <= 5


def test_violations_use_the_upper_price():
    s = DialogueState(slots={"budget": "<=5"}, budget_max=5)
    assert "budget" in filters.violations(s, {"price_min": 1, "price_max": 10})
    assert "budget" in filters.violations(s, {"price_min": 3, "price_max": 6})
    assert "budget" not in filters.violations(s, {"price_min": 5, "price_max": float("nan")})
    assert "budget" in filters.violations(s, {"price_min": None, "price_max": None})


# ------------------------------------------------------------------ named spot ("near bras basah")

def test_landmark_helpers():
    assert aliases.landmark("near Bras Basah") == "bras basah"
    assert aliases.landmark("downtown core") is None and aliases.landmark("amk") is None
    assert aliases.landmark_in_text("ok can, near SMU pls", "downtown core") == "smu"
    assert aliases.landmark_in_text("bugis side", "tampines") is None
    tl = aliases.track_landmark
    assert tl(None, UNKNOWN, "downtown core", raw_area="Bras Basah") == "bras basah"   # tracker's words
    assert tl(None, UNKNOWN, "bras basah") == "bras basah"                            # kept in the slot
    assert tl(None, UNKNOWN, "downtown core", text="korean near bugis") == "bugis"    # user's message
    assert tl("bugis", "downtown core", "downtown core", text="too far leh") == "bugis"  # kept
    assert tl("bugis", "downtown core", "tampines", text="tampines then") is None     # moved on
    assert tl("bugis", "downtown core", ANY) is None


def test_named_spot_moves_the_search(tmp_places):
    tmp_places([
        dict(place_id="BRAS", name="Near Bras Basah", lat=1.2970, lon=103.8510, area="DOWNTOWN CORE"),
        dict(place_id="RAFF", name="Near Raffles Place", lat=1.2840, lon=103.8515, area="DOWNTOWN CORE"),
        dict(place_id="OTHER", name="Across The Line", lat=1.2985, lon=103.8495, area="MUSEUM"),
    ])
    near_bras = DialogueState(slots={"area": "downtown core"}, landmark="bras basah", radius_km=1.0)
    ids = {c["place_id"] for c in recommender.candidates(near_bras)}
    assert ids == {"BRAS", "OTHER"}  # radius around the spot, whichever planning area
    assert recommender.rank(near_bras, "", None, k=1)[0].place_id == "BRAS"
    near_raff = DialogueState(slots={"area": "downtown core"}, landmark="raffles place", radius_km=1.0)
    assert {c["place_id"] for c in recommender.candidates(near_raff)} == {"RAFF"}
    # a landmark the tracker wrote straight into the slot works without state.landmark
    assert {c["place_id"] for c in recommender.candidates(
        DialogueState(slots={"area": "bras basah"}, radius_km=1.0))} == {"BRAS", "OTHER"}
    # no spot: whole planning area, as before
    assert {c["place_id"] for c in recommender.candidates(
        DialogueState(slots={"area": "downtown core"}, radius_km=5))} == {"BRAS", "RAFF"}
    # nothing returned breaks a hard constraint
    for c in recommender.candidates(near_bras):
        assert filters.violations(near_bras, c) == []

