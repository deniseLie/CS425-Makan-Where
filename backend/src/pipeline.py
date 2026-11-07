"""End-to-end pipeline for one user turn. Owner: Denise.

raw -> normalize -> detect_style -> track -> translate values -> apply critiques
    -> policy.decide -> (recommender) -> generate -> guardrail -> reply
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

from src import recommender
from src.generator import generate, template_reply
from src.normalizer import normalize
from src.policy import apply_tracker_output, decide
from src.schemas import DialogueState, PolicyAction, Turn
from src.style import detect_style
from src.tracker import track
from src.tracker.values import TABLES, translate

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


def mentioned_places(reply: str, names: list[str]) -> set[str]:
    low = reply.lower()
    # substring check first: compiling a regex for each of ~31k place names takes seconds
    return {n for n in names
            if n.lower() in low and re.search(rf"(?<!\w){re.escape(n.lower())}(?!\w)", low)}


# Everyday words that aren't in the tracker's value tables. Places in places.csv are named just
# "Korean", "Halal Food", "Chicken Rice" or "EAT"; a reply using those words isn't naming the place.
COMMON_WORDS = {"rice", "noodles", "noodle", "dim", "sum", "food", "foods", "seafood", "coffee", "kopi",
                "bbq", "grill", "eat", "eats", "the", "and", "good", "nice", "best", "place", "shop",
                "house", "kitchen", "stall", "corner", "drinks", "drink", "snacks", "dessert", "desserts"}


@lru_cache(maxsize=1)
def generic_words() -> frozenset[str]:
    """Lowercased words the bot says on its own: every word of the slot values (cuisines, dishes,
    areas, settings...) plus COMMON_WORDS."""
    words = set(COMMON_WORDS)
    for table in TABLES.values():
        for value in (*table, *table.values()):
            words |= set(re.findall(r"\w+", value.lower()))
    return frozenset(words)


def is_generic(name: str) -> bool:
    """True if a place name is only everyday words ("Korean", "Halal Food", "Eat")."""
    words = re.findall(r"\w+", name.lower())
    return bool(words) and all(w in generic_words() for w in words)


def guardrail(reply: str, action: PolicyAction, style: str) -> tuple[str, bool]:
    """If the reply names a known place that the policy did not pick, fall back to the template.
    The picked names are taken out first ("Ha-Jun Korean" contains another place's name, "Korean"),
    and place names made only of everyday words don't count."""
    rest = reply
    for name in sorted({p.name for p in action.places}, key=len, reverse=True):  # longest first
        rest = re.sub(re.escape(name), " ", rest, flags=re.IGNORECASE)
    rogue = {n for n in mentioned_places(rest, recommender.place_names()) if not is_generic(n)}
    if rogue:
        return template_reply(action, style), True
    return reply, False


class MakanSession:
    def __init__(self, user_latlon: Optional[tuple[float, float]] = None,
                 log_dir: Optional[Path] = LOG_DIR, session_id: Optional[str] = None):
        self.user_latlon = user_latlon
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.session_id = session_id or datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.reset()

    def reset(self) -> None:
        self.history: list[Turn] = []
        self.state = DialogueState()
        self.n_asked = 0
        self.last_action: Optional[PolicyAction] = None

    @property
    def log_path(self) -> Optional[Path]:
        return self.log_dir / f"{self.session_id}.jsonl" if self.log_dir else None

    def step(self, raw_text: str) -> tuple[str, dict]:
        normalized = normalize(raw_text)
        style = detect_style(raw_text)
        turn = Turn(raw=raw_text, normalized=normalized, style=style)

        raw_out = track(self.history + [turn], self.state)
        tracker_out = translate(raw_out)
        # "dun want this one": the tracker doesn't name the place, but we know what we showed last.
        if tracker_out.critique == "not_interested" and tracker_out.reject is None \
                and self.last_action is not None and self.last_action.places:
            tracker_out.reject = self.last_action.places[0].place_id
        prev = self.state
        self.state = apply_tracker_output(self.state, tracker_out)
        # "near Bras Basah": the area slot becomes "downtown core", so keep the exact spot too
        self.state.landmark = recommender.track_landmark(
            prev.landmark, prev.slots["area"], self.state.slots["area"],
            raw_area=raw_out.slots["area"], text=normalized or raw_text)

        action = decide(self.state, self.n_asked, recommender,
                        query_norm=normalized, user_latlon=self.user_latlon)
        if action.kind == "ask":
            self.n_asked += 1
        self.last_action = action

        generated = generate(action, style, raw_text)
        reply, guard_triggered = guardrail(generated, action, style)

        turn.reply = reply
        self.history.append(turn)

        debug = {
            "session_id": self.session_id,
            "turn_idx": len(self.history) - 1,
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "raw": raw_text,
            "normalized": normalized,
            "style": style,
            "tracker_output": tracker_out.to_dict(),
            "state": self.state.to_dict(),
            "n_asked": self.n_asked,
            "action": action.to_dict(),
            "guardrail_triggered": guard_triggered,
            "replaced_reply": generated if guard_triggered else None,  # what the generator wrote
            "reply": reply,
        }
        self._log(debug)
        return reply, debug

    def _log(self, record: dict) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    s = MakanSession()
    print("Makan Where? (Ctrl+C to quit)")
    while True:
        print("bot>", s.step(input("you> "))[0])
