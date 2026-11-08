"""Rule-based policy: ask or recommend.

1. A required slot (cuisine, area) is UNKNOWN and n_asked < MAX_ASKS
   -> ask it, offering the 3 most common values among current candidates.
2. More than MANY_CANDIDATES candidates and n_asked < MAX_ASKS
   -> ask the UNKNOWN soft slot whose values are most spread out (highest entropy).
3. No candidates -> relax setting -> time until something matches, keeping the budget
   -> fallback_recommend (relaxed_slot says what was dropped).
4. Otherwise -> recommend the top 3.
"""

from __future__ import annotations

import math
from collections import Counter

from src.recommender.filters import budget_cap, price_upper
from src.schemas import ANY, REQUIRED_SLOTS, SOFT_SLOTS, UNKNOWN, DialogueState, PolicyAction
from src.tracker.values import parse_budget

MAX_ASKS = 3
MANY_CANDIDATES = 10
N_OPTIONS = 3
TOP_K = 3
RELAX_ORDER = ["setting", "time"]


def _values(cands: list[dict], slot: str) -> list[str]:
    return [str(c[slot]) for c in cands if c.get(slot) is not None and c.get(slot) == c.get(slot)]  # drop NaN


def top_options(cands: list[dict], slot: str, n: int = N_OPTIONS) -> list[str]:
    counts = Counter(v for v in _values(cands, slot) if v != "none")
    return [v for v, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


def entropy(cands: list[dict], slot: str) -> float:
    counts = Counter(_values(cands, slot))
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _relax(state: DialogueState, slot: str) -> DialogueState:
    relaxed = state.copy()
    relaxed.slots[slot] = ANY
    if slot == "budget":
        relaxed.budget_max = None
    return relaxed


def decide(state: DialogueState, n_asked: int, recommender, query_norm: str = "",
           user_latlon=None) -> PolicyAction:
    budget = state.slots["budget"]
    if budget not in (UNKNOWN, ANY) and parse_budget(budget) is None:
        return PolicyAction(kind="ask", ask_slot="budget")
    if state.slots["dietary"] == "non-halal":
        return PolicyAction(kind="fallback_recommend", relaxed_slot="dietary")

    cands = recommender.candidates(state, user_latlon=user_latlon)
    can_ask = n_asked < MAX_ASKS

    if can_ask:
        for slot in REQUIRED_SLOTS:
            if state.slots[slot] == UNKNOWN:
                return PolicyAction(kind="ask", ask_slot=slot, ask_options=top_options(cands, slot))

        if len(cands) > MANY_CANDIDATES:
            scored = [(entropy(cands, s), s) for s in SOFT_SLOTS if state.slots[s] == UNKNOWN]
            scored = [(h, s) for h, s in scored if h > 0]
            if scored:
                _, slot = max(scored, key=lambda hs: (hs[0], -SOFT_SLOTS.index(hs[1])))
                return PolicyAction(kind="ask", ask_slot=slot, ask_options=top_options(cands, slot))

    if not cands:
        relaxed = state
        for slot in RELAX_ORDER:
            if not relaxed.filled(slot):
                continue
            relaxed = _relax(relaxed, slot)
            places = recommender.rank(relaxed, query_norm, user_latlon, k=TOP_K)
            if places:
                return PolicyAction(kind="fallback_recommend", places=places, relaxed_slot=slot)
        return PolicyAction(kind="fallback_recommend",
                            relaxed_slot="budget" if budget_cap(state) is not None else None)

    return PolicyAction(kind="recommend", places=recommender.rank(state, query_norm, user_latlon, k=TOP_K))


def above_budget(state: DialogueState, recommender, query_norm: str, user_latlon=None) -> PolicyAction:
    """Search closest known prices on a copy, for one explicitly authorized turn."""
    cap = budget_cap(state)
    if cap is None:
        return PolicyAction(kind="recommend")
    search = _relax(state, "budget")
    prices = {}
    for row in recommender.candidates(search, user_latlon=user_latlon):
        upper = price_upper(row)
        if upper is not None and upper > cap:
            prices[row["place_id"]] = upper
        else:
            search.excluded_place_ids.append(row["place_id"])
    if not prices:
        return PolicyAction(kind="recommend")
    search.budget_max = sorted(prices.values())[:TOP_K][-1]
    places = recommender.rank(search, query_norm, user_latlon, k=TOP_K)
    places.sort(key=lambda place: prices[place.place_id])
    for place in places:
        upper = prices[place.place_id]
        place.reasons = [
            f"estimated up to ${upper:g} (${upper - cap:.2f} over budget)",
            *(reason for reason in place.reasons if not reason.startswith("~$")),
        ]
    return PolicyAction(kind="fallback_recommend", places=places, relaxed_slot="budget")
