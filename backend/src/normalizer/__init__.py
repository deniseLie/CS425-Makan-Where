"""Singlish -> English normalizer. Owner: Varidhi.

Interface (do not change):  normalize(text: str) -> str
Flan-T5 from the HF Hub (hf.py; model repo MAKAN_NORMALIZER_MODEL, see src/config.py).
"""
from .hf import normalize

__all__ = ["normalize"]
