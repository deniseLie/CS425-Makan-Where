"""Dialogue state tracker. Owner: Nichole.

Interface (do not change):  track(history: list[Turn], prev: DialogueState) -> TrackerOutput
Real version: Qwen2.5-1.5B + LoRA, task [TRACK] - reads the whole dialogue and
emits the FULL slot state as JSON every turn (see TrackerOutput in src/schemas.py).
"""
from .stub import track

__all__ = ["track"]
