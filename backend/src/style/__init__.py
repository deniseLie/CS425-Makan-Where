"""Style detector ("singlish" | "english"). Owner: Varidhi.

Interface (do not change):  detect_style(text: str) -> str
Implementations: stub.py (keyword heuristic) and hf.py (SingBERT from the HF Hub).
Pick one with MAKAN_STYLE=stub|hf (see src/config.py).
"""
from src.config import impl

if impl("style") == "hf":
    from .hf import detect_style
else:
    from .stub import detect_style

__all__ = ["detect_style"]
