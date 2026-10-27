"""Automatic checks on a bot reply, given the policy action it was written for. Owner: Denise.

Filters teacher-written [REPLY] training targets (see teacher.py) and scores the trained
generator. check_reply() returns a list of (code, detail); an empty list means the reply passes.

    length                empty, or more than MAX_WORDS words
    leak                  place ids, action field names, braces or markdown in the reply
    rogue_place           names a catalogue place the action did not pick
    missing_place         leaves out a place the action picked
    missing_option        an ask reply leaves out one of action.ask_options
    no_question           an ask reply has no "?"
    unexplained_fallback  a fallback_recommend reply never says what was relaxed
    invented_fact         a price, distance, rating, walking time or clock time that is not in
                          the places' reasons, the options or the user's text
    style_mismatch        detect_style(reply) disagrees with the requested style

Filter a JSONL of drafts (rows with action, style, user_text, reply), from backend/:
    python -m src.generator.checks drafts.jsonl    -> drafts.kept.jsonl + drafts.rejected.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Optional

from src import recommender
from src.schemas import PolicyAction
from src.style import detect_style

MAX_WORDS = 120
SILENT_OPTIONS = {"none"}  # "no dietary needs" can be said too many ways to require the word

_LEAK = re.compile(r"[{}]|\*\*|^\s*#|\b(place_id|ask_slot|ask_options|relaxed_slot)\b|\[(REPLY|TRACK)\]",
                   re.MULTILINE)
# group 1 is the number the reply claims
_FACTS = [
    r"\$\s?(\d+(?:\.\d+)?)",
    r"(\d+(?:\.\d+)?)\s?(?:m|km|metres?|meters?)\b",
    r"(\d(?:\.\d)?)\s?-?\s?stars?\b",
    r"(\d+)\s?-?\s?(?:min|mins|minutes?)\b",
    r"\b(\d{1,2})(?::\d{2})?\s?(?:am|pm)\b",
]
# words that show the reply explained which preference was relaxed (searched with place names removed)
_RELAX_WORDS = {
    "setting": r"setting|kinds? of places?|types? of places?|hawker|aircon|air-con|cafe|restaurant|atas|"
               r"food ?court|coffee ?shop|kopitiam",
    "time": r"\btim(e|ing)|\bopen|\bhours?\b|\blate\b|\bsupper\b|\bbreakfast\b|\blunch\b|\bdinner\b",
    "budget": r"budget|pric(e|ey|ier)|cheap|expensive|\bex\b|\bcost|afford|spend",
}


def mentioned_places(reply: str, names: list[str]) -> set[str]:
    low = reply.lower()
    return {n for n in names if re.search(rf"(?<!\w){re.escape(n.lower())}(?!\w)", low)}


def _squash(text: str) -> str:
    """Lowercase letters and digits only, accents dropped: 'Food Court', 'café' -> 'foodcourt', 'cafe'."""
    return "".join(c for c in unicodedata.normalize("NFKD", text).lower() if c.isalnum())


def _numbers(text: str) -> set[float]:
    return {float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)}


def check_reply(reply: str, action: PolicyAction, style: str, user_text: str = "",
                catalogue: Optional[list[str]] = None) -> list[tuple[str, str]]:
    """Every failed check for one reply. `catalogue` defaults to all place names in the database."""
    n_words = len(reply.split())
    if n_words == 0:
        return [("length", "empty")]
    failures = []
    if n_words > MAX_WORDS:
        failures.append(("length", f"{n_words} words"))

    leak = _LEAK.search(reply)
    ids = [p.place_id for p in action.places if p.place_id in reply]
    if leak or ids:
        failures.append(("leak", leak.group(0) if leak else ids[0]))

    picked = {p.name for p in action.places}
    rest = reply  # the reply without the picked names, so "Bugis Ramen Ya" doesn't also count as "Bugis Ramen"
    for name in picked:
        rest = re.sub(re.escape(name), " ", rest, flags=re.IGNORECASE)
    catalogue = recommender.place_names() if catalogue is None else catalogue
    failures += [("rogue_place", n) for n in sorted(mentioned_places(rest, catalogue) - picked)]
    failures += [("missing_place", n) for n in sorted(picked - mentioned_places(reply, sorted(picked)))]

    if action.kind == "ask":
        if "?" not in reply:
            failures.append(("no_question", ""))
        said = _squash(reply)
        failures += [("missing_option", o) for o in action.ask_options
                     if o not in SILENT_OPTIONS and _squash(o) not in said]

    if action.kind == "fallback_recommend" and action.places:
        rx = _RELAX_WORDS.get(action.relaxed_slot)
        if rx and not re.search(rx, rest, re.IGNORECASE):
            failures.append(("unexplained_fallback", action.relaxed_slot))

    known = _numbers(" ".join([user_text, *action.ask_options, *(r for p in action.places for r in p.reasons)]))
    for rx in _FACTS:
        failures += [("invented_fact", m.group(0).strip()) for m in re.finditer(rx, reply, re.IGNORECASE)
                     if float(m.group(1)) not in known]

    detected = detect_style(reply)
    if detected != style:
        failures.append(("style_mismatch", f"reads as {detected}"))
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description="Split [REPLY] drafts into kept / rejected with check_reply.")
    ap.add_argument("drafts", type=Path, help="JSONL rows with action, style, user_text, reply")
    args = ap.parse_args()

    catalogue = recommender.place_names()
    kept, rejected, codes = [], [], Counter()
    for line in args.drafts.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        failures = check_reply(row["reply"], PolicyAction.from_dict(row["action"]), row["style"],
                               row.get("user_text", ""), catalogue)
        codes.update(code for code, _ in failures)
        if failures:
            rejected.append({**row, "failures": failures})
        else:
            kept.append(row)

    for name, rows in (("kept", kept), ("rejected", rejected)):
        out = args.drafts.with_suffix(f".{name}.jsonl")
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"kept {len(kept)} / {len(kept) + len(rejected)}")
    for code, n in codes.most_common():
        print(f"  {code:22s} {n}")


if __name__ == "__main__":
    main()
