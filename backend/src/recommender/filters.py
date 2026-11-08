"""Candidate generation: one parameterized SQL query built only from filled slots.

UNKNOWN and ANY add no condition. Slot values are first mapped to canonical forms with the alias
table. Hard constraints:

    cuisine   cuisine = ?  or the cuisine appears in the place's tags
    area      area = ?  (or region = ? for "east", "north-east", ...). When the user named a spot
              ("near bras basah"), within state.radius_km of that spot instead
    radius    within state.radius_km of the named spot, else the user (if known), else the
              area's centre
    dietary   halal = 1 | vegetarian = 1 | "no beef": nothing beef-named
    setting   setting = ?;  "aircon": not known to be open-air (unknown is kept)
    time      open at that time (places with unknown hours are kept)
    budget    top listed estimate <= cap (cap from budget_max or the budget slot)
    excluded  place_id NOT IN (...)

Every value goes in as a ? parameter, never pasted into the SQL text.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

import pandas as pd

from src.recommender import db
from src.recommender.aliases import canonical, landmark, landmark_point
from src.recommender.hours import check_time
from src.schemas import ANY, UNKNOWN, DialogueState
from src.tracker.values import parse_budget

BEEF_WORDS = ("beef", "gyu", "wagyu", "steak", "yakiniku")
PRICE_UPPER_SQL = "COALESCE(price_max, price_min)"


def price_upper(place) -> Optional[float]:
    """Highest listed estimate, including catalogues with only one price field."""
    for price in (place.get("price_max"), place.get("price_min")):
        if pd.notna(price):
            return price
    return None


def price_mid(place) -> Optional[float]:
    """middle of the price range, or price_min when there is no top price; None if unknown."""
    lo, hi = place.get("price_min"), place.get("price_max")
    if lo is None or lo != lo:
        return None
    return (lo + hi) / 2 if hi is not None and hi == hi else lo


def slots_of(state: DialogueState) -> dict[str, str]:
    """State slots in canonical form (alias table applied)."""
    return {s: canonical(s, v) for s, v in state.slots.items()}


def filled(value: str) -> bool:
    return value not in (UNKNOWN, ANY) and value != ""


def budget_cap(state: DialogueState, slots: Optional[dict] = None) -> Optional[float]:
    if state.budget_max is not None:
        return state.budget_max
    return parse_budget((slots or state.slots)["budget"])


def landmark_of(state: DialogueState) -> Optional[str]:
    """the spot the user named: state.landmark, or a landmark kept in the area slot itself
    ("bras basah" isn't in values.py, so it reaches the state as is)."""
    return state.landmark or landmark(state.slots["area"])


def ref_point(state: DialogueState, user_latlon=None, slots: Optional[dict] = None) -> Optional[tuple[float, float]]:
    """Where the radius is measured from: the spot the user named, else the user if known, else
    the centre of the chosen planning area. A region has no single centre -> None (no radius)."""
    spot = landmark_of(state)
    if spot and landmark_point(spot):
        return landmark_point(spot)
    if user_latlon is not None:
        return tuple(user_latlon)
    area = (slots or slots_of(state))["area"]
    return db.area_centroids().get(area)


def build_query(state: DialogueState, user_latlon=None, now: Optional[datetime] = None) -> tuple[str, list]:
    """-> (sql, params) selecting the catalogue rows that satisfy every hard constraint."""
    s = slots_of(state)
    where, params = [], []

    near_spot = bool(landmark_of(state) and landmark_point(landmark_of(state)))
    if filled(s["area"]) and not near_spot:  # a named spot is searched by radius only
        where.append("region = ?" if s["area"] in db.regions() else "area = ?")
        params.append(s["area"])
    if filled(s["cuisine"]):
        where.append("(cuisine = ? OR (';' || IFNULL(tags, '') || ';') LIKE ?)")
        params += [s["cuisine"], f"%;{s['cuisine']};%"]
    if filled(s["dietary"]):
        if s["dietary"] == "halal":
            where.append("halal = 1")
        elif s["dietary"] == "vegetarian":
            where.append("vegetarian = 1")
        elif s["dietary"] == "no beef":
            text = "LOWER(name || ' ' || IFNULL(tags, '') || ' ' || IFNULL(description, ''))"
            where += [f"{text} NOT LIKE ?" for _ in BEEF_WORDS]
            params += [f"%{w}%" for w in BEEF_WORDS]
        # other dietary values: the place list has nothing to check them against
    if filled(s["setting"]):
        if s["setting"] == "aircon":
            where.append("(aircon IS NULL OR aircon != 0)")
        else:
            where.append("setting = ?")
            params.append(s["setting"])
    if filled(s["time"]):
        t = check_time(s["time"], now)
        if t:
            where.append("IFNULL(open_at(opening_hours, ?), 1) = 1")
            params.append(t)
    cap = budget_cap(state, s)
    if cap is not None:
        where.append(f"{PRICE_UPPER_SQL} <= ?")
        params.append(cap)
    if state.excluded_place_ids:
        where.append(f"place_id NOT IN ({', '.join('?' * len(state.excluded_place_ids))})")
        params += list(state.excluded_place_ids)

    ref = ref_point(state, user_latlon, s)
    if ref is not None:
        # cheap bounding box first, then the exact distance
        dlat = state.radius_km / 110.574
        dlon = state.radius_km / (111.320 * math.cos(math.radians(ref[0])))
        where.append("lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?")
        params += [ref[0] - dlat, ref[0] + dlat, ref[1] - dlon, ref[1] + dlon]
        where.append("dist_km(lat, lon, ?, ?) <= ?")
        params += [ref[0], ref[1], state.radius_km]

    sql = "SELECT row_idx FROM places"
    if where:
        sql += " WHERE " + " AND ".join(where)
    return sql + " ORDER BY row_idx", params


def filtered(state: DialogueState, user_latlon=None, now: Optional[datetime] = None) -> pd.DataFrame:
    """Catalogue rows passing the hard constraints, with a dist_km column (NaN when there is no
    reference point)."""
    sql, params = build_query(state, user_latlon, now)
    df = db.load_places().iloc[db.query(sql, params)]
    ref = ref_point(state, user_latlon)
    if ref is None:
        return df.assign(dist_km=float("nan"))
    return df.assign(dist_km=[db.haversine_km(ref[0], ref[1], la, lo) for la, lo in zip(df["lat"], df["lon"])])


def candidates(state: DialogueState, user_latlon=None) -> list[dict]:
    df = filtered(state, user_latlon)
    return df.assign(dietary=db.dietary_labels(df)).to_dict("records")


def violations(state: DialogueState, place: dict) -> list[str]:
    """Hard constraints a returned place breaks (should always be empty). Used by the evaluation
    to report the hard-constraint violation rate independently of the SQL above."""
    s = slots_of(state)
    bad = []
    if filled(s["dietary"]) and s["dietary"] in ("halal", "vegetarian") and not place[s["dietary"]]:
        bad.append(s["dietary"])
    if filled(s["cuisine"]) and place.get("cuisine") != s["cuisine"] \
            and s["cuisine"] not in str(place.get("tags", "")).split(";"):
        bad.append("cuisine")
    if filled(s["area"]) and not (landmark_of(state) and landmark_point(landmark_of(state))):
        col = "region" if s["area"] in db.regions() else "area"
        if place.get(col) != s["area"]:
            bad.append("area")
    if s["setting"] == "aircon" and place.get("aircon") == 0:
        bad.append("setting")
    elif filled(s["setting"]) and s["setting"] != "aircon" and place.get("setting") != s["setting"]:
        bad.append("setting")
    cap = budget_cap(state, s)
    upper = price_upper(place)
    if cap is not None and not (upper is not None and upper <= cap + 1e-9):
        bad.append("budget")
    if place.get("place_id") in state.excluded_place_ids:
        bad.append("excluded")
    d = place.get("dist_km")
    if d is not None and d == d and d > state.radius_km + 1e-9:
        bad.append("radius")
    return bad
