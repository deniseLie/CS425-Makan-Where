"""Hybrid ranking over the hard-filtered candidates.

    score = w_content * content similarity
          + w_soft    * soft-slot match
          + w_quality * quality
          + w_dist    * distance

The top k is picked greedily, skipping a place with the same name or in the same building as one
already picked while other options remain.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from src.recommender import content, filters
from src.recommender.aliases import expand_query
from src.recommender.hours import check_time, is_open
from src.schemas import DialogueState, RankedPlace

WEIGHTS = {"content": 0.35, "soft": 0.20, "quality": 0.30, "distance": 0.15}
PRIOR_REVIEWS = 50
DIST_SCALE_KM = 1.0
CHEAP_RANGE = (3.0, 40.0)


def request_text(state: DialogueState, query_norm: str, slots: Optional[dict] = None) -> str:
    s = slots or filters.slots_of(state)
    words = [s[k] for k in ("cuisine", "dietary", "setting") if filters.filled(s[k])]
    said = state.slots.get("cuisine", "")
    if filters.filled(said) and said.lower() != s["cuisine"]:
        words.append(said)
    return expand_query(" ".join([query_norm or "", *words]).strip())


def content_part(df: pd.DataFrame, text: str) -> np.ndarray:
    sim = content.similarity(text, df.index.to_numpy())
    top = sim.max() if len(sim) else 0
    return sim / top if top > 0 else np.zeros(len(df))


def soft_part(df: pd.DataFrame, state: DialogueState, slots: dict) -> np.ndarray:
    parts = []
    lo = df["price_min"].astype(float).to_numpy()
    hi = df["price_max"].astype(float).to_numpy()
    mid = np.where(np.isnan(hi), lo, (lo + hi) / 2)

    cap = filters.budget_cap(state, slots)
    if cap is not None:
        width = np.where(hi > lo, hi - lo, 1.0)
        fit = np.where(hi <= cap, 1.0, np.clip((cap - lo) / width, 0, 1))
        parts.append(np.nan_to_num(fit, nan=0.5))
    elif slots["budget"] in ("cheap", "pricey"):
        a, b = CHEAP_RANGE
        cheap = np.clip(1 - (mid - a) / (b - a), 0, 1)
        parts.append(np.nan_to_num(cheap if slots["budget"] == "cheap" else 1 - cheap, nan=0.5))

    if slots["setting"] == "aircon":
        ac = df["aircon"].astype(float).to_numpy()
        parts.append(np.where(np.isnan(ac), 0.5, ac))

    if filters.filled(slots["time"]):
        t = check_time(slots["time"])
        if t:
            known = [is_open(h, t) for h in df["opening_hours"]]
            parts.append(np.array([0.5 if k is None else float(k) for k in known]))

    if not parts:
        return np.full(len(df), 0.5)
    return np.mean(parts, axis=0)


def quality_part(df: pd.DataFrame, catalogue_mean: Optional[float]) -> np.ndarray:
    r = df["rating"].astype(float).to_numpy()
    n = np.nan_to_num(df["n_reviews"].astype(float).to_numpy(), nan=0.0)
    n = np.where(np.isnan(r), 0.0, n)
    m = catalogue_mean if catalogue_mean is not None else 3.5
    bayes = (PRIOR_REVIEWS * m + n * np.nan_to_num(r, nan=m)) / (PRIOR_REVIEWS + n)
    return np.clip((bayes - 1) / 4, 0, 1)


def distance_part(df: pd.DataFrame) -> np.ndarray:
    d = df["dist_km"].astype(float).to_numpy()
    return np.where(np.isnan(d), 0.5, np.exp(-d / DIST_SCALE_KM))


def score(df: pd.DataFrame, state: DialogueState, query_norm: str = "",
          weights: Optional[dict] = None) -> pd.DataFrame:
    """Candidates with each part and the final score as columns (used by rank and for analysis)."""
    from src.recommender import db

    w = {**WEIGHTS, **(weights or {})}
    slots = filters.slots_of(state)
    rated = db.load_places()["rating"].dropna()
    parts = {
        "content": content_part(df, request_text(state, query_norm, slots)),
        "soft": soft_part(df, state, slots),
        "quality": quality_part(df, float(rated.mean()) if len(rated) else None),
        "distance": distance_part(df),
    }
    out = df.assign(**{f"s_{k}": v for k, v in parts.items()})
    return out.assign(score=sum(w[k] * parts[k] for k in parts))


def _pick(df: pd.DataFrame, k: int) -> pd.DataFrame:
    df = df.sort_values(["score", "n_reviews", "place_id"], ascending=[False, False, True])
    chosen, seen, rest = [], set(), []
    for idx, b, n in zip(df.index, df["building"].fillna("").to_numpy(), df["name"].to_numpy()):
        if len(chosen) == k:
            break
        keys = {("name", str(n).lower())} | ({("building", b)} if b else set())
        if keys & seen:
            rest.append(idx)
            continue
        chosen.append(idx)
        seen |= keys
    chosen += rest[:k - len(chosen)]
    return df.loc[chosen]


def reasons(row, capped: bool = False) -> list[str]:
    """Matched features for the reply ("halal", "~$5", "300 m away", "4.3 stars").
    Blank columns are left out; estimated prices say so."""
    out = []
    if row["halal"]:
        out.append("halal")
    if row["vegetarian"]:
        out.append("vegetarian")
    if capped and (upper := filters.price_upper(row)) is not None:
        out.append(f"~${upper:g} (upper estimate)")
    elif pd.notna(row["price_min"]) and pd.notna(row["price_max"]):
        price = f"~${(row['price_min'] + row['price_max']) / 2:.0f}"
        out.append(price + (" (est.)" if row.get("price_source") == "estimate" else ""))
    d = row["dist_km"]
    if not math.isnan(d):
        out.append(f"{max(100, round(d * 1000, -2)):.0f} m away" if d < 1 else f"{d:.1f} km away")
    if pd.notna(row["rating"]):
        out.append(f"{row['rating']:.1f} stars")
    return out


def rank(state: DialogueState, query_norm: str = "", user_latlon=None, k: int = 3,
         weights: Optional[dict] = None) -> list[RankedPlace]:
    df = filters.filtered(state, user_latlon)
    if df.empty:
        return []
    top = _pick(score(df, state, query_norm, weights), k)
    capped = filters.budget_cap(state) is not None
    return [RankedPlace(place_id=r["place_id"], name=r["name"], score=round(float(r["score"]), 4),
                        reasons=reasons(r, capped))
            for _, r in top.iterrows()]
