"""aliases for singlish / short / misspelt words -> the place list's canonical values."""

from __future__ import annotations

import difflib
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import ALIASES_CSV
from src.schemas import ANY, UNKNOWN

FUZZY_CUTOFF = 0.85 # difflib ratio needed to accept a misspelling
FUZZY_MIN_LEN = 5 # never fuzzy-match short words ("ex", "veg", "amk")

_FILLER = re.compile(r"^(?:near|nearby|around|at|in|somewhere)\s+|\s+(?:side|area|there|nearby|one|lah|leh|lor|ah)$")


@lru_cache(maxsize=4)
def load_aliases(path: Path = ALIASES_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"slot": str, "alias": str, "canonical": str})
    df["alias"] = df["alias"].str.lower().str.strip()
    df["canonical"] = df["canonical"].str.strip()
    return df


@lru_cache(maxsize=4)
def _tables(path: Path = ALIASES_CSV) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for r in load_aliases(path).itertuples():
        out.setdefault(r.slot, {})[r.alias] = r.canonical
    # canonical values always map to themselves
    for slot, table in out.items():
        for c in set(table.values()):
            table.setdefault(c.lower(), c)
    return out


def clean(value: str) -> str:
    v = " ".join(str(value).lower().replace("’", "'").split()).strip(" ?!.,")
    for _ in range(3):
        new = _FILLER.sub("", v).strip()
        if new == v:
            break
        v = new
    return v


def canonical(slot: str, value: str) -> str:
    """one slot value -> its canonical place-list value. UNKNOWN / ANY, budgets like "<=10" and
    clock times pass through; values not in the table are returned cleaned (lowercase)."""
    if value in (UNKNOWN, ANY) or value is None:
        return value
    raw = str(value)
    if raw.startswith("<=") or re.fullmatch(r"\d{2}:\d{2}", raw):
        return raw
    table = _tables().get(slot, {})
    v = clean(raw)
    if v in table:
        return table[v]
    if len(v) >= FUZZY_MIN_LEN:
        close = difflib.get_close_matches(v, [a for a in table if len(a) >= FUZZY_MIN_LEN], n=1, cutoff=FUZZY_CUTOFF)
        if close:
            return table[close[0]]
    return v


@lru_cache(maxsize=4)
def _cuisine_rx(path: Path = ALIASES_CSV) -> re.Pattern:
    words = sorted(_tables(path).get("cuisine", {}), key=len, reverse=True)
    return re.compile(r"(?<![a-z])(" + "|".join(re.escape(w) for w in words) + r")(?![a-z])")


def expand_query(text: str) -> str:
    """append the canonical cuisine of every dish / cuisine alias found in the text, so the
    content ranker sees "korean" when the user wrote "kbbq"."""
    low = str(text).lower()
    table = _tables().get("cuisine", {})
    found = []
    for m in _cuisine_rx().finditer(low):
        c = table[m.group(1)]
        if c not in found and c not in low:
            found.append(c)
    return (str(text) + " " + " ".join(found)).strip()


# short names for a whole town, not a spot in it
WHOLE_AREA_WORDS = {"amk", "tpy", "cck", "je", "jw", "bb", "bp", "bt batok", "bt panjang", "bt timah",
                    "bt merah", "town", "cbd", "city", "downtown", "jurong"}


@lru_cache(maxsize=4)
def _landmarks(path: Path = ALIASES_CSV) -> dict[str, tuple[float, float]]:
    """area aliases with coordinates that are a spot, not a whole planning area / region name."""
    df = load_aliases(path)
    rows = df[(df["slot"] == "area") & df["lat"].notna()]
    whole = set(df.loc[df["slot"] == "area", "canonical"].str.lower()) | WHOLE_AREA_WORDS
    return {r.alias: (float(r.lat), float(r.lon)) for r in rows.itertuples() if r.alias not in whole}


def landmark(value) -> Optional[str]:
    """the landmark alias in a slot value, or None ("near bras basah" -> "bras basah",
    "downtown core" -> None)."""
    if not isinstance(value, str) or value in (UNKNOWN, ANY):
        return None
    v = clean(value)
    return v if v in _landmarks() else None


@lru_cache(maxsize=4)
def _landmark_rx(path: Path = ALIASES_CSV) -> re.Pattern:
    words = sorted(_landmarks(path), key=len, reverse=True)
    return re.compile(r"(?<![a-z])(" + "|".join(re.escape(w) for w in words) + r")(?![a-z])")


def landmark_in_text(text: str, area: str) -> Optional[str]:
    """the longest landmark named in the text that lies in `area` (so "bugis" counts for
    downtown core, but not when the area slot says tampines)."""
    area = canonical("area", area)
    found = [m.group(1) for m in _landmark_rx().finditer(str(text).lower())]
    found = [w for w in found if canonical("area", w) == area]
    return max(found, key=len) if found else None


def landmark_point(alias: str) -> tuple[float, float] | None:
    """(lat, lon) of a landmark alias, if the table has one."""
    df = load_aliases()
    row = df[(df["slot"] == "area") & (df["alias"] == clean(alias))]
    if row.empty or pd.isna(row["lat"].iloc[0]):
        return None
    return float(row["lat"].iloc[0]), float(row["lon"].iloc[0])


def track_landmark(prev_landmark: Optional[str], prev_area: str, new_area: str,
                   raw_area: str = "", text: str = "") -> Optional[str]:
    """the spot to search around after this turn.
    1. the tracker's own words for the area ("Bras Basah", before values.py turns it into
       "downtown core"), or the area slot itself if it kept the landmark word
    2. a landmark named in the user's message that lies in the new area ("near SMU pls")
    3. the area didn't change -> keep the previous spot
    4. otherwise (new area, ANY, UNKNOWN) -> no spot"""
    if new_area in (UNKNOWN, ANY):
        return None
    for value in (raw_area, new_area):
        spot = landmark(value)
        if spot and canonical("area", spot) == canonical("area", new_area):
            return spot
    spot = landmark_in_text(text, new_area)
    if spot:
        return spot
    if canonical("area", new_area) == canonical("area", prev_area):
        return prev_landmark
    return None
