"""
places.csv loaded once into pandas and an in-memory SQLite table.
load_places() returns the DataFrame (the shared "catalogue" other modules read);
connection() returns the SQLite connection the SQL filter queries. Both are cached per CSV path.
use_places_csv(path) switches the catalogue (tests, evaluation on another file).
"""

from __future__ import annotations

import math
import sqlite3
import threading
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src import config
from src.recommender.hours import is_open

# columns the recommender relies on; anything missing from the CSV is added blank.
OPTIONAL = {"aircon": np.nan, "tags": "", "opening_hours": "", "description": "",
            "price_source": "", "building": "", "nea_type": ""}

_path: Optional[Path] = None
_lock = threading.Lock()


def places_csv() -> Path:
    return _path or config.PLACES_CSV


def use_places_csv(path: Optional[Path]) -> None:
    """Point the recommender at another places file. If None = back to config.PLACES_CSV."""
    global _path
    _path = Path(path) if path is not None else None
    _load.cache_clear()
    _connection.cache_clear()
    _centroids.cache_clear()
    from src.recommender import content  # noqa: avoid an import cycle at module load
    content.clear_cache()


@lru_cache(maxsize=2)
def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"postal_code": str, "licence_number": str, "nea_type": str,
                                  "building": str, "unit": str}, low_memory=False)
    # Unnamed rows cannot be displayed or checked against names in a reply.
    df = df.loc[df["name"].notna() & df["name"].str.strip().ne("")].copy()
    # Planning areas and regions are upper case in NEA ("DOWNTOWN CORE"); slot values are lower.
    for col in ("area", "region"):
        df[col] = df[col].str.lower()
    # A blank halal/vegetarian means "not known to be", never True.
    df["halal"] = df["halal"].fillna(0).astype(bool)
    df["vegetarian"] = df["vegetarian"].fillna(0).astype(bool)
    for col, default in OPTIONAL.items():
        if col not in df:
            df[col] = default
    for col in ("tags", "opening_hours", "description", "price_source", "building"):
        df[col] = df[col].fillna("").astype(str)
    # Older/hand-made files mark air-conditioned places with setting == "aircon".
    df.loc[df["aircon"].isna() & (df["setting"] == "aircon"), "aircon"] = 1.0
    df.loc[df["aircon"].isna() & (df["setting"] == "hawker"), "aircon"] = 0.0
    df["cuisine"] = df["cuisine"].where(df["cuisine"].isna(), df["cuisine"].astype(str).str.lower())
    return df.reset_index(drop=True)


def load_places() -> pd.DataFrame:
    return _load(places_csv())


def place_names() -> list[str]:
    return load_places()["name"].tolist()


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))


def _sql_open_at(hours, hhmm):
    r = is_open(hours or "", hhmm)
    return None if r is None else int(r)


SQL_COLUMNS = ["place_id", "name", "lat", "lon", "area", "region", "cuisine", "halal", "vegetarian",
               "setting", "aircon", "price_min", "price_max", "rating", "n_reviews", "tags",
               "opening_hours", "description"]


@lru_cache(maxsize=2)
def _connection(path: Path) -> sqlite3.Connection:
    df = _load(path)
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    table = df[SQL_COLUMNS].assign(row_idx=np.arange(len(df)),
                                   halal=df["halal"].astype(int), vegetarian=df["vegetarian"].astype(int))
    table.to_sql("places", conn, index=False)
    conn.execute("CREATE INDEX idx_area ON places(area)")
    conn.execute("CREATE INDEX idx_region ON places(region)")
    conn.execute("CREATE INDEX idx_cuisine ON places(cuisine)")
    conn.create_function("dist_km", 4, haversine_km, deterministic=True)
    conn.create_function("open_at", 2, _sql_open_at, deterministic=True)
    return conn


def query(sql: str, params: list) -> list[int]:
    """Run a SELECT row_idx ... query; returns catalogue row positions."""
    conn = _connection(places_csv())
    with _lock:
        return [r[0] for r in conn.execute(sql, params).fetchall()]


@lru_cache(maxsize=2)
def _centroids(path: Path) -> dict[str, tuple[float, float]]:
    means = _load(path).groupby("area")[["lat", "lon"]].mean()
    return {area: (r.lat, r.lon) for area, r in means.iterrows()}


def area_centroids() -> dict[str, tuple[float, float]]:
    """Reference point per planning area when we don't know where the user is: the average
    position of the area's places."""
    return _centroids(places_csv())


def regions() -> set[str]:
    return set(load_places()["region"].dropna())


def dietary_labels(df: pd.DataFrame) -> np.ndarray:
    """Single dietary value per row, used for the policy's option lists / entropy."""
    return np.select([df["halal"], df["vegetarian"]], ["halal", "vegetarian"], default="none")
