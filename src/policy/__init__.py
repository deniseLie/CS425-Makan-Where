"""Dialogue policy + critique handling. Owner: Nichole.

Interface (do not change):
    decide(state, n_asked, recommender, query_norm="", user_latlon=None) -> PolicyAction
    apply_tracker_output(prev: DialogueState, out: TrackerOutput) -> DialogueState
"""
from .critiques import apply_tracker_output
from .rules import MAX_ASKS, RELAX_ORDER, decide

__all__ = ["decide", "apply_tracker_output", "MAX_ASKS", "RELAX_ORDER"]
