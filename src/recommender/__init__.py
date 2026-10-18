"""Recommender. Owner: Yuhanna.

Interface (do not change):
    candidates(state: DialogueState, user_latlon=None) -> list[dict]   # hard-filtered rows
    rank(state, query_norm: str, user_latlon, k=3) -> list[RankedPlace]
    load_places() -> pandas.DataFrame                                   # the catalogue
    place_names() -> list[str]                                          # used by the pipeline guardrail

Real version: SQL filter (duckdb) over data/master/places.csv + hybrid ranking
(BM25/embeddings on description + rating + distance).
"""
from .stub import candidates, load_places, place_names, rank

__all__ = ["candidates", "rank", "load_places", "place_names"]
