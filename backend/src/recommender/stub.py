"""Pandas filter over data/master/places.csv, scored by rating."""

from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import PLACES_CSV
from src.schemas import ANY, UNKNOWN, DialogueState, RankedPlace


@lru_cache(maxsize=1)
def load_places(path: Path = PLACES_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    # places.csv copies NEA as is: upper-case planning areas and regions ("DOWNTOWN CORE",
    # "North-East"), and blanks where NEA has no data. The tracker's values are lower case,
    # and a blank halal / vegetarian means "not known to be", never True.
    for col in ("area", "region"):
        df[col] = df[col].str.lower()
    df["halal"] = df["halal"].fillna(0).astype(bool)
    df["vegetarian"] = df["vegetarian"].fillna(0).astype(bool)
    return df


@lru_cache(maxsize=1)
def area_centroids() -> dict[str, tuple[float, float]]:
    """Reference point per planning area for the radius filter when we don't know where the
    user is: the average position of the area's places."""
    means = load_places().groupby("area")[["lat", "lon"]].mean()
    return {area: (r.lat, r.lon) for area, r in means.iterrows()}


def is_region(value: str) -> bool:
    return value in set(load_places()["region"].dropna())


def place_names() -> list[str]:
    return load_places()["name"].tolist()


def dietary_labels(df: pd.DataFrame) -> np.ndarray:
    """Single dietary value per row, used for option lists / entropy."""
    return np.select([df["halal"], df["vegetarian"]], ["halal", "vegetarian"], default="none")


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
    return area_centroids().get(state.slots["area"])  # a region has no single centre -> None


def _filtered(state: DialogueState, user_latlon=None) -> pd.DataFrame:
    df = load_places().copy()
    s = state.slots

    def has(slot: str) -> bool:  # UNKNOWN / ANY add no condition
        return s[slot] not in (UNKNOWN, ANY)

    if has("cuisine"):
        df = df[df["cuisine"] == s["cuisine"]]
    if has("area"):  # a region ("east") keeps every place in its planning areas
        df = df[df["region" if is_region(s["area"]) else "area"] == s["area"]]
    if has("dietary"):
        if s["dietary"] == "halal":
            df = df[df["halal"]]
        elif s["dietary"] == "vegetarian":
            df = df[df["vegetarian"]]
    if has("setting"):
        df = df[df["setting"] == s["setting"]]
    # time: no opening-hours column yet -> no condition
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
    df = df.assign(dietary=dietary_labels(df))
    return df.to_dict("records")


def _reasons(row) -> list[str]:
    """Blank columns (no price or rating yet) are left out."""
    reasons = []
    if row["halal"]:
        reasons.append("halal")
    if row["vegetarian"]:
        reasons.append("vegetarian")
    if pd.notna(row["price_min"]) and pd.notna(row["price_max"]):
        reasons.append(f"~${(row['price_min'] + row['price_max']) / 2:.0f}")
    if not math.isnan(row["dist_km"]):
        d = row["dist_km"]
        reasons.append(f"{max(100, round(d * 1000, -2)):.0f} m away" if d < 1 else f"{d:.1f} km away")
    if pd.notna(row["rating"]):
        reasons.append(f"{row['rating']:.1f} stars")
    return reasons


def rank(state: DialogueState, query_norm: str = "", user_latlon=None, k: int = 3) -> list[RankedPlace]:
    """Stub: score = rating (ties broken by review count; no rating scores 0). query_norm is ignored."""
    df = _filtered(state, user_latlon).sort_values(["rating", "n_reviews"], ascending=False).head(k)
    return [
        RankedPlace(place_id=r["place_id"], name=r["name"],
                    score=float(r["rating"]) if pd.notna(r["rating"]) else 0.0, reasons=_reasons(r))
        for _, r in df.iterrows()
    ]
