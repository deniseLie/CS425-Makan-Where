"""
Dialogue policy for Makan Where?

Input : slot state (from the tracker) + dialogue context + a candidate DB
Output: an action dict for the reply generator:
    {"action": "ask", "slot": ..., "options": [...], "n_candidates": N}
    {"action": "recommend", "n_candidates": N}
    {"action": "no_results"}          # D's relaxation fallback should kick in

D needs to implement the CandidateDB interface below.
"""
from dataclasses import dataclass, field
from typing import Protocol

UNKNOWN, ANY = "UNKNOWN", "ANY"
REQUIRED = ["cuisine", "area"]            # asked first
SOFT = ["dietary", "setting", "time"]     # asked only if worth it
# budget is never asked (hard filter only if the user states it)


class CandidateDB(Protocol):
    def count(self, state: dict) -> int:
        """Number of restaurants passing all filled slots in `state`."""

    def value_counts(self, state: dict, slot: str) -> dict:
        """Among restaurants passing `state` with `slot` IGNORED,
        return {value: number_of_restaurants}. e.g. {"Malay": 12, "Chinese": 30}"""


@dataclass
class PolicyConfig:
    max_followups: int = 3         # hard cap on questions per conversation
    recommend_threshold: int = 5   # recommend once candidates <= this
    max_options: int = 4           # options listed in a question
    min_gain: float = 0.25         # soft slots must cut candidates by >= 25%


@dataclass
class DialogueContext:
    followups_asked: int = 0
    asked_slots: set = field(default_factory=set)   # never ask the same slot twice
    just_rejected: bool = False                     # set by the tracker on "too far lah" etc.


def split_gain(counts: dict) -> float:
    """Expected fraction of candidates removed if the user answers this slot.
    Expected remaining = sum(c^2)/total (user picks a value proportional to its size)."""
    total = sum(counts.values())
    if total == 0 or len(counts) < 2:
        return 0.0
    expected_remaining = sum(c * c for c in counts.values()) / total
    return 1 - expected_remaining / total


def decide(state: dict, ctx: DialogueContext, db: CandidateDB,
           cfg: PolicyConfig = PolicyConfig()) -> dict:
    n = db.count(state)

    if n == 0:
        return {"action": "no_results"}

    # After a rejection, show the next option instead of interrogating the user.
    if ctx.just_rejected:
        return {"action": "recommend", "n_candidates": n}

    if n <= cfg.recommend_threshold or ctx.followups_asked >= cfg.max_followups:
        return {"action": "recommend", "n_candidates": n}

    # Too many candidates: pick the best question.
    def best(slots, min_gain):
        pick, pick_gain, pick_counts = None, 0.0, None
        for s in slots:
            if state.get(s, UNKNOWN) != UNKNOWN or s in ctx.asked_slots:
                continue
            counts = db.value_counts(state, s)
            g = split_gain(counts)
            if g > min_gain and g > pick_gain:
                pick, pick_gain, pick_counts = s, g, counts
        return pick, pick_counts

    slot, counts = best(REQUIRED, 0.0)            # required: any useful split
    if slot is None:
        slot, counts = best(SOFT, cfg.min_gain)   # soft: must be worth asking
    if slot is None:
        return {"action": "recommend", "n_candidates": n}

    options = [v for v, _ in sorted(counts.items(), key=lambda kv: -kv[1])][: cfg.max_options]
    return {"action": "ask", "slot": slot, "options": options, "n_candidates": n}


def update_context(ctx: DialogueContext, action: dict) -> None:
    """Call after each decide() so the policy remembers what it asked."""
    if action["action"] == "ask":
        ctx.followups_asked += 1
        ctx.asked_slots.add(action["slot"])
    ctx.just_rejected = False   # reset; the tracker sets it again on the next rejection


# ---------------------------------------------------------------- stub DB for testing
class FakeDB:
    def __init__(self, rows):
        self.rows = rows

    def _match(self, state, ignore=None):
        for r in self.rows:
            ok = True
            for slot, val in state.items():
                if slot == ignore or val in (UNKNOWN, ANY) or slot not in r:
                    continue
                if r[slot] != val:
                    ok = False
                    break
            if ok:
                yield r

    def count(self, state):
        return sum(1 for _ in self._match(state))

    def value_counts(self, state, slot):
        out = {}
        for r in self._match(state, ignore=slot):
            if slot in r:
                out[r[slot]] = out.get(r[slot], 0) + 1
        return out


if __name__ == "__main__":
    rows = (
        [{"cuisine": "Chinese", "area": "Bugis", "setting": "hawker"}] * 8
        + [{"cuisine": "Malay", "area": "Bugis", "setting": "aircon"}] * 4
        + [{"cuisine": "Japanese", "area": "Bugis", "setting": "aircon"}] * 3
        + [{"cuisine": "Chinese", "area": "Chinatown", "setting": "hawker"}] * 10
    )
    db = FakeDB(rows)
    cfg = PolicyConfig()
    blank = {s: UNKNOWN for s in REQUIRED + SOFT + ["budget"]}

    ctx = DialogueContext()
    a = decide({**blank, "area": "Bugis"}, ctx, db, cfg)
    print("1.", a)                       # should ask cuisine
    update_context(ctx, a)

    a = decide({**blank, "area": "Bugis", "cuisine": "Malay"}, ctx, db, cfg)
    print("2.", a)                       # 4 candidates -> recommend

    ctx = DialogueContext(just_rejected=True)
    print("3.", decide({**blank, "area": "Chinatown"}, ctx, db, cfg))  # recommend after rejection

    print("4.", decide({**blank, "cuisine": "Korean"}, DialogueContext(), db, cfg))  # no_results