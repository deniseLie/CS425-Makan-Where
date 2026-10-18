"""Pandas filter over data/master/places_sample.csv, scored by rating."""

from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from src.schemas import ANY, UNKNOWN, DialogueState, RankedPlace

PLACES_CSV = Path(__file__).resolve().parents[2] / "data" / "master" / "places_sample.csv"

# Reference point for the radius filter when we don't know where the user is.
AREA_CENTROIDS = {
    "bugis": (1.2993, 103.8555),
    "orchard": (1.3040, 103.8318),
    "tampines": (1.3526, 103.9447),
    "jurong east": (1.3331, 103.7423),
    "chinatown": (1.2838, 103.8443),
}


@lru_cache(maxsize=1)
def load_places(path: Path = PLACES_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["halal"] = df["halal"].astype(bool)
    df["vegetarian"] = df["vegetarian"].astype(bool)
    return df


def place_names() -> list[str]:
    return load_places()["name"].tolist()


def dietary_label(row) -> str:
    """Single dietary value per row, used for option lists / entropy."""
    if row["halal"]:
        return "halal"
    if row["vegetarian"]:
        return "vegetarian"
    return "none"


def budget_cap(state: DialogueState) -> Optional[float]:
    if state.budget_max is not None:
        return state.budget_max
    m = re.match(r"<=\s*(\d+(?:\.\d+)?)", state.slots["budget"])
    return float(m.group(1)) if m else None


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))


def _ref_point(state: DialogueState, user_latlon) -> Optional[tuple[float, float]]:
    if user_latlon is not None:
        return tuple(user_latlon)
    return AREA_CENTROIDS.get(state.slots["area"])


def _filtered(state: DialogueState, user_latlon=None) -> pd.DataFrame:
    df = load_places().copy()
    s = state.slots

    def has(slot: str) -> bool:  # UNKNOWN / ANY add no condition
        return s[slot] not in (UNKNOWN, ANY)

    if has("cuisine"):
        df = df[df["cuisine"] == s["cuisine"]]
    if has("area"):
        df = df[df["area"] == s["area"]]
    if has("dietary"):
        if s["dietary"] == "halal":
            df = df[df["halal"]]
        elif s["dietary"] == "vegetarian":
            df = df[df["vegetarian"]]
    if has("setting"):
        df = df[df["setting"] == s["setting"]]
    # time: no opening-hours column in the sample data yet -> no condition
    cap = budget_cap(state)
    if cap is not None:
        df = df[df["price_min"] <= cap]
    if state.excluded_place_ids:
        df = df[~df["place_id"].isin(state.excluded_place_ids)]

    ref = _ref_point(state, user_latlon)
    if ref is not None:
        df = df.assign(dist_km=[haversine_km(ref, (r.lat, r.lon)) for r in df.itertuples()])
        df = df[df["dist_km"] <= state.radius_km]
    else:
        df = df.assign(dist_km=float("nan"))
    return df


def candidates(state: DialogueState, user_latlon=None) -> list[dict]:
    df = _filtered(state, user_latlon)
    df = df.assign(dietary=[dietary_label(r) for _, r in df.iterrows()])
    return df.to_dict("records")


def _reasons(row) -> list[str]:
    reasons = []
    if row["halal"]:
        reasons.append("halal")
    if row["vegetarian"]:
        reasons.append("vegetarian")
    reasons.append(f"~${(row['price_min'] + row['price_max']) / 2:.0f}")
    if not math.isnan(row["dist_km"]):
        d = row["dist_km"]
        reasons.append(f"{max(100, round(d * 1000, -2)):.0f} m away" if d < 1 else f"{d:.1f} km away")
    reasons.append(f"{row['rating']:.1f} stars")
    return reasons


def rank(state: DialogueState, query_norm: str = "", user_latlon=None, k: int = 3) -> list[RankedPlace]:
    """Stub: score = rating (ties broken by review count). query_norm is ignored."""
    df = _filtered(state, user_latlon).sort_values(["rating", "n_reviews"], ascending=False).head(k)
    return [
        RankedPlace(place_id=r["place_id"], name=r["name"], score=float(r["rating"]), reasons=_reasons(r))
        for _, r in df.iterrows()
    ]
