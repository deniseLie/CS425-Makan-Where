"""Dialogue state tracker. Owner: Nichole.

Interface (do not change):  track(history: list[Turn], prev: DialogueState) -> TrackerOutput
Qwen2.5-1.5B + its own LoRA adapter, task [TRACK] (hf.py; adapter MAKAN_TRACKER_ADAPTER, see
src/config.py) - reads the whole dialogue and emits the FULL slot state as JSON every turn (see
TrackerOutput in src/schemas.py).
"""
from .hf import track

__all__ = ["track"]
