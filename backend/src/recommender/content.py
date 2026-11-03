"""Content similarity between the user's request and each place's text profile."""

from __future__ import annotations

import hashlib
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.recommender import db


def profiles(df: pd.DataFrame) -> list[str]:
    cols = [df["name"], df["cuisine"].fillna(""), df["tags"].str.replace(";", " "),
            df["setting"].fillna(""), df["area"].fillna(""), df["description"]]
    return [" ".join(str(x) for x in parts if x) for parts in zip(*cols)]


class TfidfIndex:
    def __init__(self, texts: list[str]):
        from scipy.sparse import hstack
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)
        self.matrix = self._norm(hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr())

    @staticmethod
    def _norm(m):
        from sklearn.preprocessing import normalize
        return normalize(m)

    def scores(self, query: str, rows: np.ndarray) -> np.ndarray:
        from scipy.sparse import hstack
        q = self._norm(hstack([self.word.transform([query]), self.char.transform([query])]).tocsr())
        return np.asarray((self.matrix[rows] @ q.T).todense()).ravel()


class EmbedIndex:
    def __init__(self, texts: list[str], model_name: str, cache_dir: Path):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        key = hashlib.sha1(("\n".join(texts) + model_name).encode("utf-8")).hexdigest()[:16]
        cache = cache_dir / f"embed_{key}.npy"
        if cache.exists():
            self.matrix = np.load(cache)
        else:
            self.matrix = self.model.encode(texts, batch_size=256, normalize_embeddings=True,
                                            show_progress_bar=False).astype(np.float32)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, self.matrix)

    def scores(self, query: str, rows: np.ndarray) -> np.ndarray:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        return self.matrix[rows] @ q


def _cached_tfidf(texts: list[str], cache_dir: Path) -> TfidfIndex:
    key = hashlib.sha1(("tfidf-v1\n" + "\n".join(texts)).encode("utf-8")).hexdigest()[:16]
    cache = cache_dir / f"tfidf_{key}.pkl"
    if cache.exists():
        try:
            with cache.open("rb") as f:
                return pickle.load(f)
        except Exception:
            pass # stale or unreadable: rebuild
    idx = TfidfIndex(texts)
    if len(texts) >= 1000: # not worth caching tiny test catalogues
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with cache.open("wb") as f:
                pickle.dump(idx, f, protocol=pickle.HIGHEST_PROTOCOL)
        except OSError:
            pass
    return idx


@lru_cache(maxsize=1)
def index():
    df = db.load_places()
    texts = profiles(df)
    cache_dir = db.places_csv().parent / ".cache"
    if config.RECOMMENDER_CONTENT == "embed":
        return EmbedIndex(texts, config.EMBED_MODEL, cache_dir)
    return _cached_tfidf(texts, cache_dir)


def warm_up() -> None:
    """Load the catalogue, SQL table and content index now instead of on the first request."""
    db.query("SELECT row_idx FROM places LIMIT 1", [])
    index()


def clear_cache() -> None:
    index.cache_clear()


def similarity(query: str, rows) -> np.ndarray:
    """Cosine similarity (>= 0) of the query to each catalogue row in `rows`."""
    rows = np.asarray(rows, dtype=int)
    if not query.strip() or len(rows) == 0:
        return np.zeros(len(rows))
    return np.clip(index().scores(query, rows), 0, None)
