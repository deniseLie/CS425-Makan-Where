"""data/fetch_google_places.py with a fake Places API (no network, no key)."""

import importlib.util

import pandas as pd

from src.config import ROOT

spec = importlib.util.spec_from_file_location("fetch_google", ROOT / "data" / "fetch_google_places.py")
fg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fg)

PLACE = {
    "id": "ChIJabc-123", "displayName": {"text": "Hanjip Korean Grill"},
    "location": {"latitude": 1.2995, "longitude": 103.8550},
    "formattedAddress": "200 Victoria St, #01-10 Bugis Junction, Singapore 188021",
    "primaryType": "korean_barbecue_restaurant", "types": ["korean_barbecue_restaurant", "restaurant"],
    "rating": 4.6, "userRatingCount": 1834, "priceLevel": "PRICE_LEVEL_MODERATE",
    "priceRange": {"startPrice": {"currencyCode": "SGD", "units": "20"},
                   "endPrice": {"currencyCode": "SGD", "units": "40"}},
    "regularOpeningHours": {"periods": [
        {"open": {"day": 1, "hour": 11, "minute": 30}, "close": {"day": 1, "hour": 22, "minute": 0}},
        {"open": {"day": 5, "hour": 18, "minute": 0}, "close": {"day": 6, "hour": 2, "minute": 0}}]},
}


def test_to_row():
    r = fg.to_row(PLACE)
    assert r["name"] == "Hanjip Korean Grill" and r["cuisine"] == "korean"
    assert r["postcode"] == "188021" and r["unit"] == "01-10"
    assert r["rating"] == 4.6 and r["n_reviews"] == 1834 and r["price_level"] == 2
    assert (r["price_min"], r["price_max"]) == ("20", "40")
    assert r["opening_hours"] == "Mo 11:30-22:00; Fr 18:00-02:00"
    assert fg.opening_hours({"periods": [{"open": {"day": 0, "hour": 0, "minute": 0}}]}) == "24/7"


def test_run_pages_cache_and_limit(tmp_path):
    calls = []

    def fake_post(key, body):
        calls.append(body)
        if body.get("pageToken"):
            return {"places": [{**PLACE, "id": "second", "displayName": {"text": "Jangsu"}}]}
        return {"places": [PLACE], "nextPageToken": "next"}

    out, cache = tmp_path / "google_places.csv", tmp_path / "cache"
    rows = fg.run("k", ["bugis"], ["korean restaurant"], 3, 100, out, cache, post=fake_post)
    assert [r["name"] for r in rows] == ["Hanjip Korean Grill", "Jangsu"] and len(calls) == 2
    assert calls[0]["locationRestriction"]["rectangle"]["low"]["latitude"] < 1.3008 < \
        calls[0]["locationRestriction"]["rectangle"]["high"]["latitude"]
    fg.run("k", ["bugis"], ["korean restaurant"], 3, 100, out, cache, post=fake_post)
    assert len(calls) == 2  # second run served from the cache
    fg.run("k", ["tampines"], ["korean restaurant"], 3, 1, tmp_path / "x.csv", cache, post=fake_post)
    assert len(calls) == 3  # --max-requests 1 stops before the second page
    assert len(pd.read_csv(out)) == 2


def test_google_rows_merge_into_places(tmp_path):
    spec2 = importlib.util.spec_from_file_location("build_places", ROOT / "data" / "build_places.py")
    bp = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(bp)
    nea = pd.DataFrame([dict(place_id="P1", name="SEOUL GRILL PTE LTD", lat=1.2995, lon=103.8551,
                             area="DOWNTOWN CORE", region="Central", nea_type="Cafe/Restaurant (Mall)",
                             address="200 VICTORIA STREET #01-10 BUGIS JUNCTION SINGAPORE 188021",
                             postal_code=188021, licence_number="L1")])
    nea_path = tmp_path / "nea.csv"
    nea.to_csv(nea_path, index=False)
    raw = tmp_path / "raw"
    raw.mkdir()
    far = {**PLACE, "id": "far-1", "displayName": {"text": "Bugis Pho"}, "primaryType": "vietnamese_restaurant",
           "types": [], "formattedAddress": "1 Somewhere Rd, Singapore 199999", "priceRange": None,
           "location": {"latitude": 1.3040, "longitude": 103.8560}}
    pd.DataFrame([fg.to_row(PLACE), fg.to_row(far)]).to_csv(raw / "google_places.csv", index=False)
    out = bp.build(nea_path, raw, tmp_path / "places.csv").set_index("place_id")
    p1 = out.loc["P1"]
    assert p1["rating"] == 4.6 and p1["n_reviews"] == 1834 and p1["price_source"] == "google"
    assert (p1["price_min"], p1["price_max"]) == (20, 40) and p1["cuisine"] == "korean"
    assert p1["opening_hours"].startswith("Mo 11:30")
    new = out.loc["Gfar1"]
    assert new["name"] == "Bugis Pho" and new["cuisine"] == "vietnamese" and new["price_source"] == "google"
    assert new["area"] == "DOWNTOWN CORE"


def test_google_needs_a_name_match_for_named_places(tmp_path):
    """A Google Japanese restaurant 10 m from a named cafe must not label the cafe Japanese."""
    spec2 = importlib.util.spec_from_file_location("build_places", ROOT / "data" / "build_places.py")
    bp = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(bp)
    nea = pd.DataFrame([
        dict(place_id="P1", name="THE COFFEE BEAN & TEA LEAF PTE LTD", lat=1.3010, lon=103.8450,
             area="MUSEUM", region="Central", nea_type="Cafe/Restaurant (Mall)",
             address="68 ORCHARD ROAD #05-20 PLAZA SINGAPURA SINGAPORE 238839", postal_code=238839,
             licence_number="L1"),
        dict(place_id="P2", name="TAN AH KOW", lat=1.3100, lon=103.8500, area="MUSEUM", region="Central",
             nea_type="HC", address="ALBERT CENTRE Stall No 01-05", licence_number="L2"),
    ])
    nea_path = tmp_path / "nea.csv"
    nea.to_csv(nea_path, index=False)
    raw = tmp_path / "raw"
    raw.mkdir()
    sushi = {**PLACE, "id": "sushi-1", "displayName": {"text": "Genki Sushi"},
             "primaryType": "sushi_restaurant", "types": [], "formattedAddress": "68 Orchard Rd, Singapore 238839",
             "location": {"latitude": 1.30101, "longitude": 103.84501}}
    stall = {**PLACE, "id": "stall-1", "displayName": {"text": "Ah Kow Chicken Rice"},
             "primaryType": "chinese_restaurant", "types": [], "formattedAddress": "Albert Centre, Singapore",
             "location": {"latitude": 1.31001, "longitude": 103.85001}}
    pd.DataFrame([fg.to_row(sushi), fg.to_row(stall)]).to_csv(raw / "google_places.csv", index=False)
    out = bp.build(nea_path, raw, tmp_path / "places.csv").set_index("place_id")
    cafe = out.loc["P1"]
    assert cafe["cuisine"] != "japanese" and pd.isna(cafe["rating"])  # nothing copied from the sushi bar
    assert out.loc["Gsushi1", "name"] == "Genki Sushi"                # added as its own place
    # a stall named after its address has no real name, so location is enough
    assert out.loc["P2", "name"] == "Ah Kow Chicken Rice" and out.loc["P2", "cuisine"] == "chinese"

