"""Teacher prompt for drafting [REPLY] training targets with a large LLM. Owner: Denise.

The teacher only writes the reply text. The student's input is still built by
generator.hf.build_prompt(), and generator.checks.check_reply() decides which drafts are kept.

    system, messages = teacher_messages(action, style, user_text, rng=random.Random(seed))
    # Anthropic SDK:           client.messages.create(system=system, messages=messages, ...)
    # OpenAI-style / HF chat:  [{"role": "system", "content": system}, *messages]

Variety comes from rotating the few-shot examples with `rng`, so it does not depend on the
teacher API offering sampling settings.

EXAMPLES are first drafts written against the sample catalogue. Replace or extend them with
hand-written replies (Singlish by native speakers), and keep them out of the test set.

Without an API (e.g. Claude Code subagents as the teacher, see data/reply_data/README.md), from backend/:
    python -m src.generator.teacher prepare JOB ../data/reply_data/drafts_train.jsonl ../data/reply_data/drafts_dev.jsonl
    # each worker writes JOB/replies_<NN>_*.jsonl lines {"id", "reply"}; later files override earlier ones
    python -m src.generator.teacher check JOB 03     # missing ids + check_reply failures for chunk 03
    python -m src.generator.teacher merge JOB        # drafts_X.jsonl -> replies_X.jsonl (drafts + reply)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import NamedTuple, Optional

from src.schemas import PolicyAction, RankedPlace

from .checks import check_reply
from .templates import SLOT_WORDS

SYSTEM = """\
You write replies for Makan Where?, a chatbot that helps people in Singapore decide where to eat. \
For each turn you get the bot's next action (already decided by code), the reply style, and the \
user's last message. Reply with the bot's message only.

Facts
- Name every listed place exactly as spelled, in the order given, and no other restaurant, stall or \
food centre. The places come from our database; any other name could be made up.
- Describe a place only with its listed details (for example halal, ~$18, 300 m away, 4.3 stars). \
Rephrase them freely, but add no prices, distances, walking times, ratings, opening hours or dishes \
that are not listed.

Actions
- ask: ask one question about the given preference, offer every listed option by name, and make it \
clear that anything is fine too. One or two sentences, ending with a question mark.
- recommend: present each place with its details, as a short numbered list or a sentence or two. \
You can end by inviting the user to ask for something else.
- fallback with places: say there was no exact match and which preference you loosened, in plain \
words, then present the places.
- fallback with no places: say nothing matched and suggest another area or cuisine. Name no places.
If the user's last message complains (too far, too expensive, not that one), acknowledge it briefly \
before the new suggestions.

Style
- singlish: a friendly Singaporean texting a friend. Use Singlish naturally, not in every sentence: \
particles (lah, leh, lor, ah, hor), forms like "got ... or not?", "can?", "which one you want?", and \
words like makan, shiok, atas. Keep it easy to read, not a caricature, and never vulgar.
- english: friendly, concise standard English with no Singlish particles or slang.

Format
Plain text under 80 words. No markdown (no ** or #), no quotation marks around the reply, no notes. \
Never mention place IDs, scores, JSON or field names. Many replies will be generated, so vary your \
openings and wording instead of copying the examples."""


class Example(NamedTuple):
    action: PolicyAction
    style: str
    user_text: str
    reply: str


def _p(place_id: str, name: str, *reasons: str) -> RankedPlace:
    return RankedPlace(place_id=place_id, name=name, score=0.0, reasons=list(reasons))


EXAMPLES = [
    Example(PolicyAction(kind="ask", ask_slot="cuisine", ask_options=["korean", "japanese", "malay"]),
            "singlish", "eh hungry leh, want to makan",
            "Wah hungry already ah? Want Korean, Japanese or Malay? Or anything also can lor."),
    Example(PolicyAction(kind="ask", ask_slot="area", ask_options=["bugis", "orchard", "jurong east"]),
            "english", "I'd like some Japanese food",
            "Japanese sounds good! Which area would you like to eat in - Bugis, Orchard or Jurong East? "
            "Anywhere is fine too."),
    Example(PolicyAction(kind="ask", ask_slot="dietary", ask_options=["halal", "vegetarian"]),
            "singlish", "chinatown can",
            "Chinatown got quite a lot of choices leh. Got any dietary needs or not - must be halal or "
            "vegetarian?"),
    Example(PolicyAction(kind="ask", ask_slot="setting", ask_options=["hawker", "restaurant", "cafe"]),
            "english", "Halal please",
            "Sure, halal it is. Do you prefer a hawker centre, a restaurant or a cafe?"),
    Example(PolicyAction(kind="recommend", places=[
                _p("P004", "Rochor Jjigae Corner", "~$8", "1.5 km away", "4.6 stars"),
                _p("P002", "Hanok BBQ Bugis", "~$35", "200 m away", "4.5 stars"),
                _p("P001", "Bugis Kimchi House", "~$18", "300 m away", "4.3 stars")]),
            "singlish", "korean lah, bugis side",
            "Bugis got some solid Korean leh:\n"
            "1. Rochor Jjigae Corner - ~$8, 1.5 km away, 4.6 stars\n"
            "2. Hanok BBQ Bugis - ~$35, 200 m away, 4.5 stars\n"
            "3. Bugis Kimchi House - ~$18, 300 m away, 4.3 stars\n"
            "Don't like these just tell me ah."),
    Example(PolicyAction(kind="recommend", places=[
                _p("P002", "Hanok BBQ Bugis", "~$35", "200 m away", "4.5 stars"),
                _p("P003", "Daebak Halal Korean", "halal", "~$21", "400 m away", "4.2 stars")]),
            "singlish", "too far leh",
            "Ok ok, nearer ones then. Hanok BBQ Bugis is just 200 m away, around $35, 4.5 stars. Or "
            "Daebak Halal Korean, 400 m away, halal some more, about $21. Both walking distance lah."),
    Example(PolicyAction(kind="recommend", places=[
                _p("P019", "Chinatown Complex Claypot Rice", "~$8", "200 m away", "4.6 stars"),
                _p("P021", "Smith Street Biryani", "halal", "~$8", "100 m away", "4.3 stars")]),
            "english", "Somewhere in Chinatown under $10 please",
            "Here are two good picks in Chinatown within your budget:\n"
            "1. Chinatown Complex Claypot Rice - about $8, 200 m away, rated 4.6 stars\n"
            "2. Smith Street Biryani - halal, about $8, 100 m away, 4.3 stars\n"
            "Let me know if you'd like something different."),
    Example(PolicyAction(kind="fallback_recommend", relaxed_slot="budget", places=[
                _p("P009", "Orchard Gyukatsu", "~$28", "300 m away", "4.4 stars")]),
            "english", "Japanese at Orchard, below $15 please",
            "I couldn't find Japanese food in Orchard under $15, so I've stretched the budget a bit. "
            "Orchard Gyukatsu is the closest match: about $28, 300 m away and rated 4.4 stars. Would you "
            "like me to look in another area instead?"),
    Example(PolicyAction(kind="fallback_recommend", relaxed_slot="setting", places=[
                _p("P015", "Tampines Green Leaf Cafe", "vegetarian", "~$16", "250 m away", "4.2 stars")]),
            "singlish", "want western hawker food, tampines side",
            "Aiyo, Tampines side no western hawker leh, so I look at cafes also. Tampines Green Leaf Cafe "
            "can - vegetarian, around $16, 250 m away, 4.2 stars. Try or not?"),
    Example(PolicyAction(kind="fallback_recommend", relaxed_slot="budget", places=[]),
            "english", "Halal Korean in Tampines please",
            "Sorry, I couldn't find any halal Korean places in Tampines, even after loosening the budget. "
            "Would you like to try another area, or a different cuisine?"),
]


def describe(action: PolicyAction, style: str, user_text: str) -> str:
    """The teacher's view of one turn: plain text, not the student's JSON prompt."""
    lines = [f"Style: {style}", f"User's last message: {user_text}"]
    if action.kind == "ask":
        lines.append(f"Action: ask about {action.ask_slot} ({SLOT_WORDS.get(action.ask_slot, action.ask_slot)})")
        lines.append("Options: " + (", ".join(action.ask_options) if action.ask_options else "none, just ask"))
        return "\n".join(lines)

    what = SLOT_WORDS.get(action.relaxed_slot, action.relaxed_slot)
    if action.kind == "recommend":
        lines.append("Action: recommend")
    elif action.places:
        lines.append(f"Action: fallback - no exact match, so the {what} preference was relaxed")
    else:
        lines.append(f"Action: fallback - nothing found, even after relaxing the {what} preference")
    lines.append("Places:" if action.places else "Places: none")
    lines += [f"{i}. {p.name} ({', '.join(p.reasons)})" for i, p in enumerate(action.places, 1)]
    return "\n".join(lines)


def pick_examples(action: PolicyAction, style: str, k: int = 3,
                  rng: Optional[random.Random] = None) -> list[Example]:
    """k examples in the same style, same action kind first; rng shuffles within those groups."""
    pool = [e for e in EXAMPLES if e.style == style]
    if rng is not None:
        rng.shuffle(pool)
    pool.sort(key=lambda e: e.action.kind != action.kind)  # stable sort keeps the shuffle
    return pool[:k]


def teacher_messages(action: PolicyAction, style: str, user_text: str, k: int = 3,
                     rng: Optional[random.Random] = None) -> tuple[str, list[dict]]:
    """(system prompt, chat messages) asking the teacher for one reply; few-shot examples go first."""
    messages = []
    for ex in pick_examples(action, style, k, rng):
        messages += [{"role": "user", "content": describe(ex.action, ex.style, ex.user_text)},
                     {"role": "assistant", "content": ex.reply}]
    messages.append({"role": "user", "content": describe(action, style, user_text)})
    return SYSTEM, messages


def _read(path: Path) -> list[dict]:
    rows = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError as e:
                raise ValueError(f"{path}:{i}: {e}") from None
    return rows


def _drafts(job: Path) -> dict[str, dict]:
    return {r["id"]: r for p in json.loads((job / "sources.json").read_text()) for r in _read(Path(p))}


def prepare(job: Path, drafts: list[Path], chunks: int = 8, seed: int = 0) -> list[Path]:
    """Shuffle the draft rows into JOB/chunk_NN.jsonl files of {"id", "prompt": describe(...)}."""
    job.mkdir(parents=True, exist_ok=True)
    (job / "sources.json").write_text(json.dumps([str(Path(p).resolve()) for p in drafts]))
    rows = [r for p in drafts for r in _read(p)]
    random.Random(seed).shuffle(rows)
    paths = []
    for k in range(chunks):
        lines = [json.dumps({"id": r["id"], "prompt": describe(PolicyAction.from_dict(r["action"]), r["style"],
                                                                r["user_text"])}, ensure_ascii=False)
                 for r in rows[k::chunks]]
        paths.append(job / f"chunk_{k:02d}.jsonl")
        paths[-1].write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return paths


def replies(job: Path, chunk: str = "*") -> dict[str, str]:
    """id -> reply from JOB/replies_<chunk>_*.jsonl; a later file (e.g. _fix1) overrides an earlier one."""
    return {r["id"]: r["reply"] for path in sorted(job.glob(f"replies_{chunk}_*.jsonl")) for r in _read(path)}


def check(job: Path, chunk: str, catalogue: Optional[list[str]] = None) -> tuple[list[str], dict[str, tuple]]:
    """(missing ids, {id: (check_reply failures, reply)}) for one chunk."""
    drafts, got = _drafts(job), replies(job, chunk)
    ids = [r["id"] for r in _read(job / f"chunk_{chunk}.jsonl")]
    failing = {}
    for i in ids:
        if i in got:
            r = drafts[i]
            failures = check_reply(got[i], PolicyAction.from_dict(r["action"]), r["style"], r["user_text"], catalogue)
            if failures:
                failing[i] = (failures, got[i])
    return [i for i in ids if i not in got], failing


def merge(job: Path) -> list[Path]:
    """Write each drafts file again with its `reply`, as replies_*.jsonl next to it."""
    got, written = replies(job), []
    for src in map(Path, json.loads((job / "sources.json").read_text())):
        name = src.name.replace("drafts", "replies", 1) if "drafts" in src.name else f"replies_{src.name}"
        rows = [{**r, "reply": got[r["id"]]} for r in _read(src) if r["id"] in got]
        written.append(src.with_name(name))
        written[-1].write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Hand [REPLY] drafts to teacher workers, check and merge their replies.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("job", type=Path)
    p.add_argument("drafts", type=Path, nargs="+")
    p.add_argument("--chunks", type=int, default=8)
    c = sub.add_parser("check")
    c.add_argument("job", type=Path)
    c.add_argument("chunk", help="e.g. 03")
    m = sub.add_parser("merge")
    m.add_argument("job", type=Path)
    args = ap.parse_args()

    if args.cmd == "prepare":
        for path in prepare(args.job, args.drafts, args.chunks):
            print(f"{path}  {len(_read(path))} rows")
    elif args.cmd == "check":
        missing, failing = check(args.job, args.chunk)
        for i, (failures, reply) in failing.items():
            print(f"--- {i}  {failures}\n    reply: {reply!r}")
        print(f"\nchunk {args.chunk}: {len(missing)} missing, {len(failing)} failing")
        if missing:
            print("missing ids:", " ".join(missing))
    else:
        for path in merge(args.job):
            print(f"{path}  {len(_read(path))} rows")


if __name__ == "__main__":
    main()
