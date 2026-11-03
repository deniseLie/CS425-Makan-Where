#!/usr/bin/env python3
"""
Fetch ratings, prices and opening hours from the Google Places API (New) into
data/raw/google_places.csv, which data/build_places.py then merges into places.csv.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
URL = "https://places.googleapis.com/v1/places:searchText"
FIELDS = ",".join([
    "places.id", "places.displayName", "places.location", "places.formattedAddress",
    "places.primaryType", "places.types", "places.rating", "places.userRatingCount",
    "places.priceLevel", "places.priceRange", "places.regularOpeningHours", "nextPageToken",
])

DEFAULT_AREAS = ["bugis", "smu", "chinatown", "little india", "tampines", "bedok", "jurong east",
                 "woodlands", "ang mo kio", "clementi", "serangoon", "punggol"]
DEFAULT_QUERIES = ["chinese restaurant", "malay food", "indian restaurant", "korean restaurant",
                   "japanese restaurant", "thai restaurant", "western food", "vietnamese restaurant",
                   "hawker centre", "cafe"]
BOX_KM = 1.5  # half-width of the search box around each area's point

PRICE_LEVEL = {"PRICE_LEVEL_INEXPENSIVE": 1, "PRICE_LEVEL_MODERATE": 2,
               "PRICE_LEVEL_EXPENSIVE": 3, "PRICE_LEVEL_VERY_EXPENSIVE": 4}

# Google place type -> the place list's cuisine
TYPE_CUISINE = {
    "chinese_restaurant": "chinese", "dim_sum_restaurant": "chinese", "cantonese_restaurant": "chinese",
    "hot_pot_restaurant": "chinese", "taiwanese_restaurant": "chinese",
    "malaysian_restaurant": "malay", "indonesian_restaurant": "malay", "malay_restaurant": "malay",
    "indian_restaurant": "indian", "south_indian_restaurant": "indian", "north_indian_restaurant": "indian",
    "korean_restaurant": "korean", "korean_barbecue_restaurant": "korean",
    "japanese_restaurant": "japanese", "sushi_restaurant": "japanese", "ramen_restaurant": "japanese",
    "japanese_curry_restaurant": "japanese", "yakiniku_restaurant": "japanese",
    "thai_restaurant": "thai", "vietnamese_restaurant": "vietnamese",
    "american_restaurant": "western", "italian_restaurant": "western", "pizza_restaurant": "western",
    "hamburger_restaurant": "western", "french_restaurant": "western", "steak_house": "western",
    "mexican_restaurant": "western", "spanish_restaurant": "western", "mediterranean_restaurant": "western",
    "british_restaurant": "western", "western_restaurant": "western", "brunch_restaurant": "western",
}
DAYS = ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"]


def load_area_points(path=HERE / "master" / "aliases.csv"):
    points = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["slot"] == "area" and r.get("lat"):
                points[r["alias"]] = (float(r["lat"]), float(r["lon"]))
    return points


def box(lat, lon, km=BOX_KM):
    dlat = km / 110.574
    dlon = km / 111.320  # cos(1.35 deg) is ~1
    return {"rectangle": {"low": {"latitude": lat - dlat, "longitude": lon - dlon},
                          "high": {"latitude": lat + dlat, "longitude": lon + dlon}}}


def opening_hours(regular):
    """regularOpeningHours -> OSM-style text the recommender reads ("Mo 11:00-21:00; Tu ...")."""
    if not regular or not regular.get("periods"):
        return ""
    periods = regular["periods"]
    if len(periods) == 1 and "close" not in periods[0]:
        return "24/7"
    rules = []
    for p in periods:
        o, c = p.get("open", {}), p.get("close")
        if not c:
            continue
        rules.append(f"{DAYS[o.get('day', 0)]} {o.get('hour', 0):02d}:{o.get('minute', 0):02d}"
                     f"-{c.get('hour', 0):02d}:{c.get('minute', 0):02d}")
    return "; ".join(rules)


def to_row(place):
    """One Places API result -> one google_places.csv row."""
    addr = place.get("formattedAddress", "")
    postcode = re.findall(r"\b(\d{6})\b", addr)
    unit = re.search(r"#\s*([0-9A-Z][0-9A-Z\-/]*)", addr, re.I)
    types = [place.get("primaryType", "")] + place.get("types", [])
    cuisine = next((TYPE_CUISINE[t] for t in types if t in TYPE_CUISINE), "")
    pr = place.get("priceRange") or {}
    lo = pr.get("startPrice", {}).get("units")
    hi = pr.get("endPrice", {}).get("units")
    return {
        "google_id": place.get("id", ""),
        "name": (place.get("displayName") or {}).get("text", ""),
        "lat": place.get("location", {}).get("latitude"),
        "lon": place.get("location", {}).get("longitude"),
        "address": addr,
        "postcode": postcode[-1] if postcode else "",
        "unit": unit.group(1) if unit else "",
        "category": (place.get("primaryType") or "").replace("_", " "),
        "cuisine": cuisine,
        "rating": place.get("rating", ""),
        "n_reviews": place.get("userRatingCount", ""),
        "price_level": PRICE_LEVEL.get(place.get("priceLevel", ""), ""),
        "price_min": lo if lo is not None else "",
        "price_max": hi if hi is not None else (lo if lo is not None else ""),
        "opening_hours": opening_hours(place.get("regularOpeningHours")),
    }


def search(key, query, lat, lon, page_token=None, cache_dir=None, post=None):
    """One Text Search page. Returns (json, from_cache)."""
    body = {"textQuery": query, "pageSize": 20, "regionCode": "SG", "locationRestriction": box(lat, lon)}
    if page_token:
        body["pageToken"] = page_token
    cache = None
    if cache_dir:
        h = hashlib.sha1(json.dumps([body, FIELDS], sort_keys=True).encode()).hexdigest()[:16]
        cache = Path(cache_dir) / f"{h}.json"
        if cache.exists():
            return json.loads(cache.read_text(encoding="utf-8")), True
    data = (post or _post)(key, body)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data), encoding="utf-8")
    return data, False


def _post(key, body):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST", headers={
        "Content-Type": "application/json", "X-Goog-Api-Key": key, "X-Goog-FieldMask": FIELDS})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Google API error {e.code}: {e.read().decode()[:500]}")


def run(key, areas, queries, pages, max_requests, out, cache_dir, dry_run=False, post=None):
    points = load_area_points()
    missing = [a for a in areas if a not in points]
    if missing:
        raise SystemExit(f"no coordinates for: {missing}. Use a landmark from data/master/aliases.csv "
                         f"(rows with lat/lon), e.g. {sorted(points)[:8]}")
    plan = [(q, a) for a in areas for q in queries]
    print(f"{len(plan)} searches x up to {pages} page(s) = up to {len(plan) * pages} requests")
    if dry_run:
        for q, a in plan[:10]:
            print(f"  {q} @ {a}")
        print("  ...")
        return []

    rows, seen, paid = [], set(), 0
    for q, a in plan:
        token = None
        for _ in range(pages):
            if paid >= max_requests:
                print(f"stopped at --max-requests {max_requests}")
                break
            data, cached = search(key, q, *points[a], page_token=token, cache_dir=cache_dir, post=post)
            paid += 0 if cached else 1
            for place in data.get("places", []):
                r = to_row(place)
                if r["google_id"] not in seen and r["lat"] is not None:
                    seen.add(r["google_id"])
                    rows.append(r)
            token = data.get("nextPageToken")
            if not token:
                break
            if not cached:
                time.sleep(0.2)
        print(f"{q} @ {a}: {len(rows)} places so far, {paid} paid requests")

    # add to what earlier runs saved (same place found again: the newer row wins)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged = {}
    if out.exists():
        with out.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                merged[r["google_id"]] = r
    n_before = len(merged)
    for r in rows:
        merged[r["google_id"]] = r
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(to_row({}).keys()))
        w.writeheader()
        w.writerows(merged.values())
    print(f"places this run: {len(rows)}  new: {len(merged) - n_before}  total in file: {len(merged)}")
    print(f"paid requests this run: {paid}")
    print(f"Output: {out.resolve()}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Fetch Google Places data into data/raw/google_places.csv.")
    parser.add_argument("--areas", default=",".join(DEFAULT_AREAS),
                        help="comma-separated landmarks from aliases.csv")
    parser.add_argument("--queries", default=",".join(DEFAULT_QUERIES), help="comma-separated search texts")
    parser.add_argument("--pages", type=int, default=1, help="pages of 20 per search (max 3)")
    parser.add_argument("--max-requests", type=int, default=300, help="stop after this many paid requests")
    parser.add_argument("--out", type=Path, default=HERE / "raw" / "google_places.csv")
    parser.add_argument("--cache", type=Path, default=HERE / "raw" / "google_cache")
    parser.add_argument("--key", default=os.getenv("GOOGLE_MAPS_API_KEY", ""))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.key and not args.dry_run:
        raise SystemExit("set GOOGLE_MAPS_API_KEY (or pass --key)")
    areas = [a.strip().lower() for a in args.areas.split(",") if a.strip()]
    queries = [q.strip() for q in args.queries.split(",") if q.strip()]
    run(args.key, areas, queries, min(args.pages, 3), args.max_requests, args.out, args.cache, args.dry_run)


if __name__ == "__main__":
    main()