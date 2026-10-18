"""Qwen + LoRA tracker, task [TRACK]. Owner: Nichole.

build_prompt() / build_target() define the training format: build the training file with
them so serving sees exactly what the model was trained on.
"""

from __future__ import annotations

import json
import logging

from src.schemas import CRITIQUES, SLOTS, DialogueState, TrackerOutput, Turn

log = logging.getLogger(__name__)


def build_prompt(history: list[Turn]) -> str:
    lines = ["[TRACK]", f"Slots: {', '.join(SLOTS)}. Use UNKNOWN if not mentioned, ANY if no preference.", ""]
    for t in history:
        lines.append(f"User: {t.raw}")
        if t.normalized and t.normalized != t.raw:
            lines.append(f"User (English): {t.normalized}")
        if t.reply:
            lines.append(f"Bot: {t.reply}")
    lines += ["", 'Return JSON: {"slots": {...}, "reject": <place name or null>, '
                  f'"critique": {" | ".join(CRITIQUES)} | null}}']
    return "\n".join(lines)


def build_target(out: TrackerOutput, reject_name: str | None = None) -> str:
    """Gold completion for one training example."""
    return json.dumps({"slots": out.slots, "reject": reject_name, "critique": out.critique}, ensure_ascii=False)


def _reject_id(value) -> str | None:
    if not value:
        return None
    from src.recommender import load_places

    df = load_places()
    hit = df[(df["place_id"] == value) | (df["name"].str.lower() == str(value).lower())]
    return hit.iloc[0]["place_id"] if len(hit) else None


def parse(text: str, prev: DialogueState) -> TrackerOutput:
    """Model text -> TrackerOutput. Unparseable output keeps the previous state (and is logged)."""
    try:
        data = json.loads(text[text.index("{"):text.rindex("}") + 1])
        slots = {k: str(v) for k, v in (data.get("slots") or {}).items() if k in SLOTS}
        critique = data.get("critique") if data.get("critique") in CRITIQUES else None
        return TrackerOutput(slots={**prev.slots, **slots}, reject=_reject_id(data.get("reject")), critique=critique)
    except (ValueError, TypeError, AttributeError):
        log.warning("tracker output was not valid JSON; keeping previous state")
        return TrackerOutput(slots=dict(prev.slots))


def track(history: list[Turn], prev: DialogueState) -> TrackerOutput:
    from src.llm import complete

    return parse(complete(build_prompt(history)), prev)
