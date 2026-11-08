"""Style detector ("singlish" | "english"). Owner: Varidhi.

Interface (do not change):  detect_style(text: str) -> str
SingBERT classifier from the HF Hub (hf.py; model repo MAKAN_STYLE_MODEL, see src/config.py).
"""
from .hf import detect_style

__all__ = ["detect_style"]
