"""Dialogue state tracker. Owner: Nichole.

Interface (do not change):  track(history: list[Turn], prev: DialogueState) -> TrackerOutput
Real version: Qwen2.5-1.5B + its own LoRA adapter, task [TRACK] - reads the whole dialogue and
emits the FULL slot state as JSON every turn (see TrackerOutput in src/schemas.py).
Pick stub.py or hf.py with MAKAN_TRACKER=stub|hf (see src/config.py).
"""
from src.config import impl

if impl("tracker") == "hf":
    from .hf import track
else:
    from .stub import track

__all__ = ["track"]
