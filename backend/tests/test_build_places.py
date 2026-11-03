"""data/build_places.py on a tiny made-up NEA file plus fake OSM/MUIS/Google files."""

import importlib.util
import json

import pandas as pd
import pytest

from src.config import ROOT

spec = importlib.util.spec_from_file_location("build_places", ROOT / "data" / "build_places.py")
bp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bp)

NEA_COLS = ["place_id", "name", "lat", "lon", "area", "cuisine", "halal", "vegetarian", "setting",
            "price_min", "price_max", "rating", "n_reviews", "description", "region", "address",
            "postal_code", "nea_type", "licence_number", "mrt_exits_500m", "bus_stops_500m"]


def nea_row(pid, name, lat, lon, address, nea_type, licence, postal=None):
    return dict(place_id=pid, name=name, lat=lat, lon=lon, area="DOWNTOWN CORE", region="Central",
                address=address, postal_code=postal, nea_type=nea_type, licence_number=licence)


@pytest.fixture
def nea(tmp_path):
    rows = [
        nea_row("P1", "SEOUL BITES PTE. LTD.", 1.30000, 103.85500,
                "200 VICTORIA STREET #01-10 BUGIS JUNCTION SINGAPORE 188021", "Cafe/Restaurant (Mall)", "L1", 188021),
        nea_row("P2", "TAN AH KOW", 1.28030, 103.84480, "MAXWELL FOOD CENTRE Stall No 01-05", "HC", "L2"),
        nea_row("P3", "LIM BEE HOON", 1.29000, 103.85000,
                "5 SOMEWHERE ROAD SINGAPORE PRIMARY SCHOOL SINGAPORE 111111", "Cafe/Bistro (Shophouse)", "L3", 111111),
        nea_row("P4", "GRAND HOTEL PTE LTD", 1.29100, 103.85100,
                "1 HOTEL ROAD (MAIN KITCHEN) SINGAPORE 222222", "Other F&B", "L4", 222222),
        nea_row("P5", "SEOUL BITES PTE. LTD.", 1.30000, 103.85500,
                "200 VICTORIA STREET #01-10 BUGIS JUNCTION SINGAPORE 188021", "Cafe/Restaurant (Mall)", "L1", 188021),
        nea_row("P6", "GREEN LEAF VEGETARIAN CAFE PTE LTD?", 1.31000, 103.86000,
                "10 JALAN BESAR #01-01 SINGAPORE 38982", "Cafe/Bistro (Shophouse)", "L6", 38982),
        nea_row("P7", "ONG MEI LING", 1.32000, 103.87000,
                "4 TAMPINES CENTRAL 5 #04-27 TAMPINES MALL SINGAPORE 529510", "Cafe/Restaurant (Mall)", "L7", 529510),
    ]
    path = tmp_path / "places_nea.csv"
    pd.DataFrame(rows).reindex(columns=NEA_COLS).to_csv(path, index=False)
    return path


def test_build_without_extra_sources(nea, tmp_path):
    out = bp.build(nea, tmp_path / "raw", tmp_path / "places.csv")
    ids = set(out["place_id"])
    assert ids == {"P1", "P2", "P6", "P7"}  # school, kitchen and duplicate rows gone
    p = out.set_index("place_id")
    assert p.loc["P1", "name"] == "Seoul Bites" and p.loc["P1", "cuisine"] == "korean"
    assert p.loc["P1", "setting"] == "restaurant" and p.loc["P1", "aircon"] == 1
    assert p.loc["P2", "name"] == "Stall 01-05 @ Maxwell Food Centre" and p.loc["P2", "setting"] == "hawker"
    assert p.loc["P2", "licensee"] == "TAN AH KOW"
    assert p.loc["P6", "vegetarian"] == 1 and p.loc["P6", "setting"] == "cafe"
    assert p.loc["P6", "postal_code"] == "038982" and not p.loc["P6", "licensee"].endswith("?")
    assert p.loc["P7", "name"] == "Eatery @ Tampines Mall #04-27"
    assert (out["halal"] == 0).all()  # never guessed
    assert (out["price_source"] == "estimate").all()
    assert "Korean restaurant at Bugis Junction" in p.loc["P1", "description"]


def test_build_with_osm_muis_google(nea, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "osm_food.json").write_text(json.dumps({"elements": [
        {"type": "node", "lat": 1.28032, "lon": 103.84482,
         "tags": {"amenity": "restaurant", "name": "Ah Kow Chicken Rice", "cuisine": "chinese;chicken",
                  "opening_hours": "Mo-Sa 10:00-20:00", "addr:unit": "01-05"}},
        {"type": "node", "lat": 1.5, "lon": 104.0, "tags": {"amenity": "cafe", "name": "Far Away"}},
        {"type": "node", "id": 42, "lat": 1.3203, "lon": 103.8703,
         "tags": {"amenity": "restaurant", "name": "Kimchi Corner Pte Ltd", "cuisine": "korean",
                  "opening_hours": "11:00-21:00", "diet:halal": "yes"}},
    ]}))
    pd.DataFrame([{"Name": "Seoul Bites", "Address": "200 Victoria Street #01-10 Bugis Junction",
                   "Postal Code": "188021"}]).to_csv(raw / "muis_halal.csv", index=False)
    pd.DataFrame([{"name": "Seoul Bites", "lat": 1.30001, "lng": 103.85501, "rating": 4.4,
                   "user_ratings_total": 321, "price_level": 2}]).to_csv(raw / "google_places.csv", index=False)

    out = bp.build(nea, raw, tmp_path / "places.csv").set_index("place_id")
    assert out.loc["P2", "name"] == "Ah Kow Chicken Rice" and out.loc["P2", "name_source"] == "osm"
    assert out.loc["P2", "cuisine"] == "chinese"
    assert out.loc["P2", "cuisine"] == "chinese" and out.loc["P2", "opening_hours"] == "Mo-Sa 10:00-20:00"
    assert out.loc["P1", "halal"] == 1 and out.loc["P1", "halal_source"] == "muis"
    assert out.loc["P1", "rating"] == 4.4 and out.loc["P1", "n_reviews"] == 321
    assert (out.loc["P1", "price_min"], out.loc["P1", "price_max"]) == (12, 30)
    assert out.loc["P1", "price_source"] == "google"
    assert out.loc["P7", "halal"] == 0
    # an OSM place with no NEA row becomes its own place; one 20 km out at sea doesn't
    new = out.loc["On42"]
    assert new["name"] == "Kimchi Corner" and new["cuisine"] == "korean" and new["area"] == "DOWNTOWN CORE"
    assert new["halal"] == 0 and "halal (osm, unverified)" in new["tags"]  # OSM halal is not certified
    assert "Far Away" not in set(out["name"])


def test_no_match_without_evidence(nea, tmp_path):
    """Two NEA rows 10 m apart and an OSM point with an unrelated name: no match."""
    df = bp.derive(bp.clean(pd.read_csv(nea, dtype={"licence_number": str}))[0])
    df.loc[len(df)] = df.iloc[0]
    df.loc[len(df) - 1, ["place_id", "lat"]] = ["PX", 1.30009]
    pts = pd.DataFrame([{"lat": 1.30004, "lon": 103.85500, "name": "Totally Different", "unit": ""}])
    assert bp.match_points(df.reset_index(drop=True), pts)[0] == -1


@pytest.mark.parametrize("address,expected", [
    ("4 TAMPINES CENTRAL 5 #04-27/29 TAMPINES MALL SINGAPORE 529510", ("4 Tampines Central 5", "04-27/29", "Tampines Mall")),
    ("TOA PAYOH LOR 8 BLK 210 Stall No 01-10", ("Blk 210 Toa Payoh Lor 8", "01-10", "Blk 210 Toa Payoh Lor 8")),
    ("233 EAST COAST ROAD SINGAPORE 428924", ("233 East Coast Road", "", "")),
])
def test_split_address(address, expected):
    assert bp.split_address(address) == expected


@pytest.mark.parametrize("name,cuisine", [
    ("SUSHI ZANMAI PTE LTD", "japanese"), ("RASA RASA WESTERN & THAI BBQ", ""), ("PIZZA HUT SINGAPORE", "western"),
    ("SERGEANT CHICKEN RICE", "chinese"), ("TAN AH KOW", ""), ("SPLIT PEA PRATA HOUSE", "indian"),
])
def test_cuisine_from_name(name, cuisine):
    assert bp.cuisine_of(name) == cuisine


def test_muis_export_format_and_coffeeshop_stalls(tmp_path):
    """halal-estab export: Name, Address (postal code at the end), Type, Scheme."""
    rows = [
        nea_row("P1", "SEOUL BITES PTE. LTD.", 1.30000, 103.85500,
                "200 VICTORIA STREET #01-10 BUGIS JUNCTION SINGAPORE 188021", "Cafe/Restaurant (Mall)", "L1", 188021),
        # a coffeeshop: three stall licences in one unit
        nea_row("P2", "TAN AH KOW", 1.37, 103.89, "205 BEDOK NORTH STREET 1 #01-357 SINGAPORE 460205",
                "Cafe/Bistro (Shophouse)", "L2", 460205),
        nea_row("P3", "LIM AH LEK", 1.37, 103.89, "205 BEDOK NORTH STREET 1 #01-357 SINGAPORE 460205",
                "Cafe/Bistro (Shophouse)", "L3", 460205),
        nea_row("P4", "NASI PADANG RIZQ PTE LTD", 1.37, 103.89, "205 BEDOK NORTH STREET 1 #01-357 SINGAPORE 460205",
                "Cafe/Bistro (Shophouse)", "L4", 460205),
        # hawker-centre stall: no postal code in NEA
        nea_row("P5", "SITI BINTE ALI", 1.316, 103.898, "GEYLANG SERAI MARKET Stall No 02-158", "HC", "L5"),
        nea_row("P6", "ONG MEI LING", 1.36, 103.85, "HOUGANG AVE 1 BLK 105 Stall No 02-49", "HC", "L6"),
    ]
    nea = tmp_path / "nea.csv"
    pd.DataFrame(rows).reindex(columns=NEA_COLS).to_csv(nea, index=False)
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame([
        {"Name": "Seoul Bites", "Address": "200 VICTORIA STREET #01-10 BUGIS JUNCTION 188021",
         "Type": "Restaurant", "Scheme": "Eating Establishment"},
        {"Name": "$5 WESTERN", "Address": "205 BEDOK NORTH STREET 1 #01-357, stall 1 460205",
         "Type": "Hawker", "Scheme": "Eating Establishment"},
        {"Name": "Nasi Padang Rizq", "Address": "205 BEDOK NORTH STREET 1 #01-357, stall 4 460205",
         "Type": "Hawker", "Scheme": "Eating Establishment"},
        {"Name": "Siti Nasi Lemak", "Address": "1 GEYLANG SERAI #02-158 GEYLANG SERAI MARKET 402001",
         "Type": "Hawker", "Scheme": "Eating Establishment"},
        {"Name": "Ong Mee Rebus", "Address": "105 HOUGANG AVENUE 1 #02-49 530105",
         "Type": "Hawker", "Scheme": "Eating Establishment"},
        {"Name": "Big Central Kitchen", "Address": "1 FOOD ROAD #01-01 999999",
         "Type": "Central Kitchen", "Scheme": "Food Preparation Area"},
    ]).to_csv(raw / "muis_halal.csv", index=False)

    out = bp.build(nea, raw, tmp_path / "places.csv").set_index("place_id")
    assert out.loc["P1", "halal"] == 1 and out.loc["P1", "halal_evidence"].startswith("Seoul Bites")
    # the named stall in the coffeeshop is halal; the unnamed stallholders are only tagged
    assert out.loc["P4", "halal"] == 1
    assert out.loc["P2", "halal"] == 0 and out.loc["P3", "halal"] == 0
    assert "halal stall" in out.loc["P2", "tags"] and "$5 Western" in out.loc["P2", "description"]
    # hawker-centre stalls matched by centre name / block + street + stall number
    assert out.loc["P5", "halal"] == 1 and out.loc["P5", "name"] == "Siti Nasi Lemak"
    assert out.loc["P6", "halal"] == 1
    assert not (tmp_path / "muis_unmatched.csv").exists()  # the central kitchen is skipped, not unmatched
