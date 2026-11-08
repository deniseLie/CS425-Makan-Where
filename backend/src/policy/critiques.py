"""Merge a TrackerOutput into the DialogueState (deterministic code, no model)."""

from src.schemas import UNKNOWN, DialogueState, TrackerOutput
from src.tracker.values import parse_budget as _parse_budget

TOO_FAR_FACTOR = 0.5          # radius_km *= 0.5
TOO_EXPENSIVE_FACTOR = 0.7    # budget_max *= 0.7
DEFAULT_BUDGET_AFTER_CRITIQUE = 15.0  # "too expensive" with no stated budget yet
                                      # TODO: base this on the prices just shown


def apply_tracker_output(prev: DialogueState, out: TrackerOutput) -> DialogueState:
    state = prev.copy()
    state.slots = dict(out.slots)

    # The budget slot is the user-facing value; budget_max is what the filter uses.
    budget = state.slots["budget"]
    if budget != UNKNOWN or budget != prev.slots["budget"]:
        state.budget_max = _parse_budget(budget)

    if out.reject and out.reject not in state.excluded_place_ids:
        state.excluded_place_ids.append(out.reject)

    if out.critique == "too_far":
        state.radius_km = round(state.radius_km * TOO_FAR_FACTOR, 3)
    elif out.critique == "too_expensive":
        if state.budget_max is None:
            state.budget_max = DEFAULT_BUDGET_AFTER_CRITIQUE
        else:
            state.budget_max = round(state.budget_max * TOO_EXPENSIVE_FACTOR, 1)
        state.slots["budget"] = f"<={state.budget_max:g}"

    return state
