"""Draft [REPLY] inputs by replaying the synthetic [TRACK] dialogues through the real policy. Owner: Denise.

For every user turn, the synthetic gold slot state stands in for the tracker. It is mapped to
the backend's values (tracker.stub.ALIASES + the place catalogue), merged with critiques by
apply_tracker_output, and policy.decide picks the action - so the actions come from the same
code that runs at serving time. Each row is one turn:

    {"id", "task": "[REPLY]", "source": "synthetic", "style", "user_text", "action", "state"}

Next steps: teacher.py writes a `reply` for each row, checks.py filters them, and `--kept` turns
kept rows into training examples with input = hf.build_prompt(action, style, user_text), output = reply.

Cuisine / area values the catalogue lacks (planning areas, regions, dishes it can't map) are
swapped for catalogue values in the state and in the user's text, picking areas that serve the
dialogue's cuisine. Dialogues with a dietary need the recommender can't filter ("no beef") are
dropped, and turns that change nothing ("ok can", "nasi lemak not malay") are skipped.

From backend/ (rerun when places.csv or the policy changes):
    python -m src.generator.drafts ../data/synthetic_data/dialogues_synth_train.jsonl \
        --out ../data/reply_data/drafts_train.jsonl --max-per-bucket 150
    python -m src.generator.drafts --kept ../data/reply_data/replies_train.kept.jsonl \
        --out ../data/reply_data/reply_train.jsonl          # rerun when build_prompt changes
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, NamedTuple, Optional

from src import recommender
from src.policy import apply_tracker_output, decide
from src.schemas import ANY, CRITIQUES, SLOTS, UNKNOWN, DialogueState, PolicyAction, TrackerOutput, empty_slots
from src.tracker.stub import ALIASES

from .hf import build_prompt


class Vocab(NamedTuple):
    cuisines: list[str]
    areas: list[str]
    pairs: set[tuple[str, str]]   # (cuisine, area) combinations that exist
    dietary: set[str]             # dietary values the recommender can filter on


def catalogue_vocab() -> Vocab:
    places = recommender.load_places()
    dietary = {c["dietary"] for c in recommender.candidates(DialogueState())} - {"none"}
    return Vocab(sorted(set(places["cuisine"])), sorted(set(places["area"])),
                 set(zip(places["cuisine"], places["area"])), dietary)


def dialogues(rows: Iterable[dict]) -> list[list[dict]]:
    """Synthetic rows (one per user turn, ids "SYN00001_3") -> dialogues of parsed turns, in order."""
    groups = defaultdict(list)
    for row in rows:
        did, turn = row["id"].rsplit("_", 1)
        last = row["input"].rsplit("User: ", 1)[1]  # the newest user turn closes the input
        raw, _, normalized = last.partition("\nNormalized: ")
        gold = json.loads(row["output"])
        feedback = gold.pop("feedback", "none")
        groups[did].append((int(turn), {"id": row["id"], "style": row["style"], "raw": raw.strip(),
                                        "normalized": normalized.strip(), "gold": gold, "feedback": feedback}))
    return [[t for _, t in sorted(turns, key=lambda x: x[0])] for _, turns in sorted(groups.items())]


def soft_value(slot: str, value: str) -> str:
    """Synthetic value -> backend value for budget / dietary / setting / time."""
    if value in (UNKNOWN, ANY):
        return value
    if slot == "budget":
        m = re.search(r"\$(\d+)(?:-(\d+))?", value)
        if value.startswith("above") or not m:
            return ANY  # "dun mind pay more": no cap
        return f"<={m.group(2) or m.group(1)}"
    return ALIASES.get(slot, {}).get(value.lower(), value.lower())


def relex_plan(dialogue: list[dict], vocab: Vocab, rng: random.Random) -> dict:
    """{(slot, synthetic value): (state value, text to write instead or None)} for cuisine and area."""
    plan = {}

    def plan_slot(slot, allowed) -> Optional[str]:
        values = dict.fromkeys(t["gold"][slot] for t in dialogue if t["gold"][slot] not in (UNKNOWN, ANY))
        # values kept as they are, so a swapped-in one never repeats them ("Korean, not Korean")
        used = {ALIASES[slot].get(v.lower()) for v in values} & set(allowed)
        for v in values:
            canon = ALIASES[slot].get(v.lower())
            if canon in allowed:
                plan[(slot, v)] = (canon, None)
            else:
                new = rng.choice([x for x in allowed if x not in used] or allowed)
                used.add(new)
                plan[(slot, v)] = (new, new)
        return plan[(slot, next(iter(values)))][0] if values else None

    first = plan_slot("cuisine", vocab.cuisines)
    plan_slot("area", [a for a in vocab.areas if first is None or (first, a) in vocab.pairs] or vocab.areas)
    return plan


def relex_text(text: str, plan: dict, style: str) -> str:
    """Write the plan's replacements into a user turn ("east side can" -> "tampines side can")."""
    for (slot, value), (_, new) in sorted(plan.items(), key=lambda kv: -len(kv[0][1])):
        if new is None:
            continue
        the = r"(?:\bthe\s+)?" if slot == "area" else ""  # "in the east" -> "in Tampines"
        text = re.sub(rf"{the}(?<!\w)({re.escape(value)})(?!\w)",
                      lambda m: new if style == "singlish" and m.group(1).islower() else new.title(),
                      text, flags=re.IGNORECASE)
    return text


def replay(dialogue: list[dict], vocab: Vocab, rng: random.Random) -> list[dict]:
    """One draft row per user turn the policy has to answer."""
    if any(soft_value("dietary", t["gold"]["dietary"]) not in (UNKNOWN, ANY, *vocab.dietary) for t in dialogue):
        return []
    plan = relex_plan(dialogue, vocab, rng)
    state, n_asked, prev_gold, prev_action, rows = DialogueState(), 0, empty_slots(), None, []
    for t in dialogue:
        changed = [s for s in SLOTS if t["gold"][s] != prev_gold[s]]
        prev_gold = t["gold"]

        slots = dict(state.slots)  # only overwrite what this turn changed, so critique effects stay
        for s in changed:
            v = t["gold"][s]
            slots[s] = plan[(s, v)][0] if (s, v) in plan else soft_value(s, v)
        reject = None
        if t["feedback"] == "not_interested" and prev_action is not None and prev_action.places:
            reject = prev_action.places[0].place_id  # the synthetic data rejects "this one": the top pick
        critique = t["feedback"] if t["feedback"] in CRITIQUES else None
        if slots == state.slots and reject is None and critique is None:
            continue  # "ok can", or a revision to the same value ("nasi lemak not malay"): nothing to answer

        raw = relex_text(t["raw"], plan, t["style"])
        normalized = relex_text(t["normalized"], plan, "english")
        state = apply_tracker_output(state, TrackerOutput(slots=slots, reject=reject, critique=critique))
        action = decide(state, n_asked, recommender, query_norm=normalized)
        n_asked += action.kind == "ask"
        prev_action = action

        # a turn whose swapped value didn't make it into the text would contradict its state
        swapped = [plan[(s, t["gold"][s])][1] for s in changed if (s, t["gold"][s]) in plan]
        if all(new is None or new.lower() in raw.lower() for new in swapped):
            rows.append({"id": t["id"], "task": "[REPLY]", "source": "synthetic", "style": t["style"],
                         "user_text": raw, "action": action.to_dict(), "state": state.to_dict()})
    return rows


def bucket(row: dict) -> tuple[str, str, str]:
    a = row["action"]
    if a["kind"] == "ask":
        detail = a["ask_slot"]
    elif not a["places"]:
        detail = "nothing found"
    else:
        detail = a["relaxed_slot"] or "-"
    return a["kind"], detail, row["style"]


def build_drafts(dialogue_list: list[list[dict]], seed: int = 0,
                 max_per_bucket: Optional[int] = None) -> list[dict]:
    """Replay every dialogue; drop exact duplicates and optionally cap rows per bucket()."""
    rng, vocab = random.Random(seed), catalogue_vocab()
    rows, seen = [], set()
    for d in dialogue_list:
        for row in replay(d, vocab, rng):
            key = json.dumps([row["style"], row["user_text"], row["action"]], sort_keys=True)
            if key not in seen:
                seen.add(key)
                rows.append(row)
    if max_per_bucket:
        rng.shuffle(rows)
        counts, kept = Counter(), []
        for row in rows:
            if counts[bucket(row)] < max_per_bucket:
                counts[bucket(row)] += 1
                kept.append(row)
        rows = sorted(kept, key=lambda r: r["id"])
    return rows


def training_row(row: dict) -> dict:
    """A kept draft with its teacher `reply` -> a [REPLY] example in the [TRACK] files' row format."""
    action = PolicyAction.from_dict(row["action"])
    return {"id": row["id"], "task": "[REPLY]", "source": row["source"], "style": row["style"],
            "input": build_prompt(action, row["style"], row["user_text"]), "output": row["reply"]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Draft [REPLY] rows by replaying synthetic [TRACK] dialogues.")
    ap.add_argument("synth", type=Path, nargs="?",
                    help="synthetic [TRACK] JSONL, e.g. data/synthetic_data/dialogues_synth_train.jsonl")
    ap.add_argument("--kept", type=Path, help="instead: kept drafts with replies -> [REPLY] training rows")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-per-bucket", type=int, default=None, help="cap rows per (action, slot, style)")
    args = ap.parse_args()
    if (args.synth is None) == (args.kept is None):
        ap.error("give either a synthetic file or --kept")

    lines = (args.kept or args.synth).read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    if args.kept:
        rows = [training_row(r) for r in records]
    else:
        rows = build_drafts(dialogues(records), args.seed, args.max_per_bucket)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} {'training rows' if args.kept else 'drafts'} -> {args.out}")
    if args.kept:
        return
    for b, n in sorted(Counter(bucket(r) for r in rows).items()):
        print(f"  {' / '.join(b):40s} {n}")


if __name__ == "__main__":
    main()
