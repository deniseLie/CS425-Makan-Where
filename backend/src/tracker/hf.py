"""Qwen + LoRA tracker, task [TRACK]. Owner: Nichole.

build_prompt() / build_target() define the training format, the one the [TRACK] dataset uses:
build the training file with them so serving sees exactly what the model was trained on.

    [TRACK]
    User: craving western sia
    Normalized: I am craving Western.
    System: Ok can! Which side you want to makan?
    User: ang mo kio area can
    Normalized: Somewhere around Ang Mo Kio is fine.

    {"cuisine": "Western", "area": "Ang Mo Kio", "budget": "UNKNOWN", "dietary": "UNKNOWN",
     "setting": "UNKNOWN", "time": "UNKNOWN", "feedback": "none"}

The answer holds the full slot state after this turn, written the way the user said it, plus
this turn's feedback. The pipeline translates the values with src/tracker/values.py.
"""

from __future__ import annotations

import json
import logging

from src.schemas import CRITIQUES, SLOTS, DialogueState, TrackerOutput, Turn

log = logging.getLogger(__name__)

NO_FEEDBACK = "none"


def build_prompt(history: list[Turn]) -> str:
    lines = ["[TRACK]"]
    for t in history:
        lines.append(f"User: {t.raw}")
        lines.append(f"Normalized: {t.normalized or t.raw}")
        if t.reply:
            lines.append(f"System: {t.reply}")
    return "\n".join(lines)


def build_target(out: TrackerOutput) -> str:
    """Gold completion for one training example: the six slots, then this turn's feedback."""
    return json.dumps({**{s: out.slots[s] for s in SLOTS}, "feedback": out.critique or NO_FEEDBACK},
                      ensure_ascii=False)


def parse(text: str, prev: DialogueState) -> TrackerOutput:
    """Model text -> TrackerOutput. A missing slot keeps its previous value; unparseable output
    keeps the whole previous state (and is logged). `reject` is left for the pipeline to fill."""
    try:
        data = json.loads(text[text.index("{"):text.rindex("}") + 1])
        slots = {k: str(v) for k, v in data.items() if k in SLOTS and v is not None}
        critique = data.get("feedback") if data.get("feedback") in CRITIQUES else None
        return TrackerOutput(slots={**prev.slots, **slots}, critique=critique)
    except (ValueError, TypeError, AttributeError):
        log.warning("tracker output was not valid JSON; keeping previous state")
        return TrackerOutput(slots=dict(prev.slots))


def track(history: list[Turn], prev: DialogueState) -> TrackerOutput:
    from src.llm import complete

    return parse(complete(build_prompt(history), "tracker"), prev)
