#!/usr/bin/env python3
"""
Convert NEA licensed eating establishments (NEA_Establishments.xlsx) into the
recommender's place database, data/master/places.csv.

Usage:
    pip install pandas openpyxl
    python data/convert_nea_places.py --xlsx data/raw/NEA_Establishments.xlsx --out data/master/places.csv

This is a plain 1:1 conversion: every NEA row becomes one place and every value is
copied as it is in the spreadsheet. Nothing is dropped, deduplicated or cleaned
(suggested deletions are listed in data/README.md). Columns NEA has no data for are
left blank.

Output columns follow the recommender's schema, then the remaining NEA columns:

    place_id,name,lat,lon,area,cuisine,halal,vegetarian,setting,
    price_min,price_max,rating,n_reviews,description,
    region,address,postal_code,nea_type,licence_number,mrt_exits_500m,bus_stops_500m
"""

import argparse
from pathlib import Path

import pandas as pd


COLUMNS = [
    "place_id", "name", "lat", "lon", "area", "cuisine", "halal", "vegetarian", "setting",
    "price_min", "price_max", "rating", "n_reviews", "description",
    "region", "address", "postal_code", "nea_type", "licence_number", "mrt_exits_500m", "bus_stops_500m",
]

# places.csv column <- NEA column. Schema columns not listed here stay blank.
FROM_NEA = {
    "name": "licensee_name",
    "lat": "latitude",
    "lon": "longitude",
    "area": "PLN_AREA_N",
    "region": "Region",
    "address": "premises_address",
    "postal_code": "postal_code",
    "nea_type": "Type",
    "licence_number": "licence_number",
    "mrt_exits_500m": "mrt_exits_500m",
    "bus_stops_500m": "bus_stops_500m",
}


def convert(xlsx: Path, out: Path):
    nea = pd.read_excel(xlsx, dtype={"licence_number": str})
    missing = set(FROM_NEA.values()) - set(nea.columns)
    assert not missing, f"NEA columns not found: {missing}"
    # Postal codes are numbers in the sheet with blanks, so pandas reads them as floats;
    # Int64 writes 38982 instead of 38982.0.
    nea["postal_code"] = nea["postal_code"].astype("Int64")

    places = nea[list(FROM_NEA.values())].rename(columns={v: k for k, v in FROM_NEA.items()})
    # Licence numbers repeat, so the id is the spreadsheet row order instead.
    places.insert(0, "place_id", [f"P{i:05d}" for i in range(1, len(places) + 1)])
    places = places.reindex(columns=COLUMNS)

    out.parent.mkdir(parents=True, exist_ok=True)
    places.to_csv(out, index=False, encoding="utf-8")

    empty = [c for c in COLUMNS if places[c].isna().all()]
    partial = {c: int(n) for c, n in places.isna().sum().items() if 0 < n < len(places)}
    print(f"NEA rows: {len(nea)} | places written: {len(places)}")
    print(f"empty columns: {', '.join(empty)}")
    print(f"partly empty: {partial}")
    print(f"Output: {out.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Convert NEA eating establishments into places.csv.")
    parser.add_argument(
        "--xlsx",
        type=Path,
        default=Path("data/raw/NEA_Establishments.xlsx"),
        help="NEA establishments spreadsheet",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/master/places.csv"),
        help="Output CSV",
    )
    args = parser.parse_args()

    convert(args.xlsx, args.out)


if __name__ == "__main__":
    main()
