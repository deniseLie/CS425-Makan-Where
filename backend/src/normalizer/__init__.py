"""Singlish -> English normalizer. Owner: Varidhi.

Interface (do not change):  normalize(text: str) -> str
Implementations: stub.py (identity) and hf.py (Flan-T5 from the HF Hub).
Pick one with MAKAN_NORMALIZER=stub|hf (see src/config.py).
"""
from src.config import impl

if impl("normalizer") == "hf":
    from .hf import normalize
else:
    from .stub import normalize

__all__ = ["normalize"]
