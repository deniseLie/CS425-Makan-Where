"""Shared data contract for the Makan Where? pipeline.

Every module talks to every other module ONLY through the types in this file.
If you need to change a field, raise it with the team first - the tracker's
training data, the gold eval set and the logs are all stored in this format.

Slot values are always strings:
    UNKNOWN  - the user has not mentioned this slot yet
    ANY      - the user said they have no preference ("anything also can")
    <value>  - a concrete value, e.g. "korean", "downtown core", "halal", "<=8"

Slots hold place-list values. The trained tracker writes what the user said ("Western",
"Bugis", "below $10"); src/tracker/values.py translates that before it reaches the state.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

UNKNOWN = "UNKNOWN"
ANY = "ANY"

SLOTS = ["cuisine", "area", "budget", "dietary", "setting", "time"]
REQUIRED_SLOTS = ["cuisine", "area"]          # policy asks for these first
SOFT_SLOTS = ["dietary", "setting", "time"]   # asked only to narrow a big candidate set
# budget: hard filter if the user states it, never asked

STYLES = ("singlish", "english")
CRITIQUES = ("too_far", "too_expensive", "not_interested")
ACTION_KINDS = ("ask", "recommend", "fallback_recommend")


def empty_slots() -> dict[str, str]:
    return {s: UNKNOWN for s in SLOTS}


def _check_slots(slots: dict[str, str]) -> None:
    extra = set(slots) - set(SLOTS)
    if extra:
        raise ValueError(f"unknown slot(s): {sorted(extra)}")
    for k, v in slots.items():
        if not isinstance(v, str):
            raise TypeError(f"slot {k!r} must be a string, got {type(v).__name__}")


class _JSONMixin:
    """to_dict / from_dict / to_json / from_json for logging and training data."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[call-overload]

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        return cls(**d)

    @classmethod
    def from_json(cls, s: str):
        return cls.from_dict(json.loads(s))


@dataclass
class Turn(_JSONMixin):
    """One user turn. `reply` is the system's answer to it (filled by the pipeline),
    so the tracker / generator can read the whole dialogue."""

    raw: str
    normalized: str = ""
    style: str = "english"
    reply: Optional[str] = None

    def __post_init__(self) -> None:
        if self.style not in STYLES:
            raise ValueError(f"style must be one of {STYLES}, got {self.style!r}")


@dataclass
class DialogueState(_JSONMixin):
    slots: dict[str, str] = field(default_factory=empty_slots)
    excluded_place_ids: list[str] = field(default_factory=list)
    radius_km: float = 2.0
    budget_max: Optional[float] = None
    # the exact spot the user named ("bras basah", "smu"), if any. The area slot only holds its
    # planning area ("downtown core"), so the recommender searches around this spot instead.
    landmark: Optional[str] = None

    def __post_init__(self) -> None:
        # Missing slots default to UNKNOWN so partial dicts (e.g. in tests) are fine.
        self.slots = {**empty_slots(), **self.slots}
        _check_slots(self.slots)

    def copy(self) -> "DialogueState":
        return DialogueState.from_dict(json.loads(self.to_json()))

    def filled(self, slot: str) -> bool:
        """True if the slot carries a concrete value (not UNKNOWN / ANY)."""
        return self.slots[slot] not in (UNKNOWN, ANY)


@dataclass
class TrackerOutput(_JSONMixin):
    """What the tracker emits every turn: the FULL slot state plus any critique."""

    slots: dict[str, str] = field(default_factory=empty_slots)
    reject: Optional[str] = None      # place_id the user rejected this turn
    critique: Optional[str] = None    # "too_far" | "too_expensive" | "not_interested" | None

    def __post_init__(self) -> None:
        self.slots = {**empty_slots(), **self.slots}
        _check_slots(self.slots)
        if self.critique is not None and self.critique not in CRITIQUES:
            raise ValueError(f"critique must be one of {CRITIQUES} or None, got {self.critique!r}")


@dataclass
class RankedPlace(_JSONMixin):
    place_id: str
    name: str
    score: float
    reasons: list[str] = field(default_factory=list)  # e.g. ["halal", "~$5", "300 m away"]


@dataclass
class PolicyAction(_JSONMixin):
    kind: str                                   # "ask" | "recommend" | "fallback_recommend"
    ask_slot: Optional[str] = None
    ask_options: list[str] = field(default_factory=list)
    places: list[RankedPlace] = field(default_factory=list)
    relaxed_slot: Optional[str] = None          # set for fallback_recommend

    def __post_init__(self) -> None:
        if self.kind not in ACTION_KINDS:
            raise ValueError(f"kind must be one of {ACTION_KINDS}, got {self.kind!r}")
        if self.ask_slot is not None and self.ask_slot not in SLOTS:
            raise ValueError(f"unknown ask_slot {self.ask_slot!r}")
        self.places = [p if isinstance(p, RankedPlace) else RankedPlace.from_dict(p) for p in self.places]
