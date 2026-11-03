"""Opening hours for the `time` slot."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Optional

SGT = timezone(timedelta(hours=8))

# one representative clock time per meal word.
MEAL_TIMES = {"breakfast": "08:30", "lunch": "12:30", "dinner": "19:00", "late night": "23:30"}

_RANGE = re.compile(r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\+?")


def check_time(time_slot: str, now: Optional[datetime] = None) -> Optional[str]:
    """Slot value -> 'HH:MM' to check, or None if the slot sets no time condition."""
    v = str(time_slot).strip().lower()
    if re.fullmatch(r"\d{2}:\d{2}", v):
        return v
    if v == "now":
        return (now or datetime.now(SGT)).strftime("%H:%M")
    return MEAL_TIMES.get(v)


@lru_cache(maxsize=65536)
def _ranges(hours: str) -> Optional[tuple]:
    h = (hours or "").strip()
    if not h:
        return None
    if h == "24/7":
        return ((0, 24 * 60),)
    out = []
    for rule in h.split(";"):
        if re.search(r"\b(?:off|closed)\b", rule, re.I):
            continue
        for m in _RANGE.finditer(rule):
            a = int(m.group(1)) * 60 + int(m.group(2))
            b = int(m.group(3)) * 60 + int(m.group(4))
            out.append((a, b))
    return tuple(out) if out else None


def is_open(hours: str, hhmm: str) -> Optional[bool]:
    """True/False, or None when the hours are unknown or can't be read."""
    rs = _ranges(hours if isinstance(hours, str) else "")
    if rs is None:
        return None
    t = int(hhmm[:2]) * 60 + int(hhmm[3:5])
    for a, b in rs:
        if b > a and a <= t < b:
            return True
        if b <= a and (t >= a or t < b): # past midnight, e.g. 18:00-02:00
            return True
    return False
