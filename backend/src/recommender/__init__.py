"""Recommender. Owner: Yuhanna.

Interface (do not change):
    candidates(state: DialogueState, user_latlon=None) -> list[dict]   # hard-filtered rows
    rank(state, query_norm: str, user_latlon, k=3) -> list[RankedPlace]
    load_places() -> pandas.DataFrame                                   # the catalogue
    place_names() -> list[str]                                          # used by the pipeline guardrail

Modules:
    db.py       places.csv -> pandas + in-memory SQLite; area centres
    aliases.py  alias table (data/master/aliases.csv): Singlish / short / misspelt -> canonical
    filters.py  candidates: one parameterized SQL query from the filled slots (hard constraints)
    ranking.py  hybrid score = content + soft-slot match + Bayesian quality + distance decay
    content.py  TF-IDF (default) or sentence-embedding similarity
    hours.py    opening hours for the time slot
    stub.py     the old rating-only baseline, kept for ablations
"""
from .content import warm_up
from .db import area_centroids, load_places, place_names, use_places_csv
from .filters import candidates
from .ranking import WEIGHTS, rank

__all__ = ["candidates", "rank", "load_places", "place_names", "area_centroids", "use_places_csv", "warm_up", "WEIGHTS"]
