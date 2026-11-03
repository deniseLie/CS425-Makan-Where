"""aliases for singlish / short / misspelt words -> the place list's canonical values."""

from __future__ import annotations

import difflib
import re
from functools import lru_cache
from pathlib import Path

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


def landmark_point(alias: str) -> tuple[float, float] | None:
    """(lat, lon) of a landmark alias, if the table has one."""
    df = load_aliases()
    row = df[(df["slot"] == "area") & (df["alias"] == clean(alias))]
    if row.empty or pd.isna(row["lat"].iloc[0]):
        return None
    return float(row["lat"].iloc[0]), float(row["lon"].iloc[0])
