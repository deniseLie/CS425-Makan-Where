"""
Relabel the [TRACK] budget slot with the amount the user actually said, as "<=N".

The tracker data used to keep only four budget buckets (below $5, below $10, $10-20, above $20), so
"budget about $15 lah" was labelled "$10-20" and the 15 was lost. The backend
(backend/src/tracker/values.py) wants "<=N", the most the user will pay. This script reads the
amount back from the user's own lines (User:, or its Normalized: line if the User line has none)
up to and including each row's turn, and rewrites the label:

  below $5 / below $10 / $10-20  ->  "<=N", N = the latest amount the user stated so far
                                     ("budget about $15 lah" -> "<=15", "ten to twenty dollar" -> "<=20",
                                     "$8-10" -> "<=10"). With no number in the user's lines ("cheap cheap
                                     pls", MultiWOZ's "cheap" / "moderate"): the bucket's top, "<=5" /
                                     "<=10" / "<=20".
  above $20                      ->  ANY (no cap; values.py already reads "above $20" as ANY)
  UNKNOWN, ANY, "<=N"            ->  unchanged, so running the script twice changes nothing
  OVERRIDES (below)              ->  rows whose old label missed a budget the user plainly stated

Rows with feedback too_expensive keep the user's last stated budget: the backend policy applies
the "too expensive" cut, not the tracker.

Usage (from the repo root):
    python data/relabel_budget.py              # rewrite the tracker files in place, print a report
    python data/relabel_budget.py --check      # report only; exit 1 if any row would change
    python data/relabel_budget.py --show 20    # also print 20 random changed rows to read by eye
    python data/relabel_budget.py path.jsonl   # specific files
"""
import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent
DEFAULT_FILES = (sorted(DATA.glob("woz_converted_data/dialogues_*.jsonl"))
                 + sorted(DATA.glob("synthetic_data/dialogues_synth_*.jsonl")))
UNKNOWN, ANY = "UNKNOWN", "ANY"

# old bucket -> (lowest, highest) amount that fits it, and the label when no number was said
BUCKETS = {"below $5": (0, 5), "below $10": (0, 10), "$10-20": (10, 20)}

# Rows the old labels got wrong, by dialogue id: {dialogue: (first user turn it applies to, label)}.
# The turn is the row id's suffix (synthetic) or turn_id (MultiWOZ). Applies to UNKNOWN labels only.
OVERRIDES = {
    "WOZ20375": (4, "<=25"),  # "Yes, under $25.00 per entree." was annotated as no price range
}

# ---------------------------------------------------------------------------------------------
# Reading an amount from one message
# ---------------------------------------------------------------------------------------------

_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
          "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
          "nineteen"]
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
_WORD = {w: i for i, w in enumerate(_UNITS)} | _TENS | {"hundred": 100}
_WORDS_RX = re.compile(
    rf"\b(?:({'|'.join(_TENS)})(?:[\s-]({'|'.join(_UNITS[1:10])}))?|({'|'.join(_UNITS)}|hundred))\b")
_MONEY_WORD = r"\s*(?:-|to)?\s*\w*\s*(?:dollars?|bucks?|sgd)\b"


def _digits(text: str) -> str:
    """Number words -> digits: "fifteen bucks" -> "15 bucks", "twenty-five" -> "25". A lone
    "one" stays a word unless money follows ("cheap one can", "this one"; but "one dollar")."""
    def sub(m):
        if m.group(3) == "one" and not re.match(_MONEY_WORD, text[m.end():]):
            return m.group(0)
        if m.group(3):
            return str(_WORD[m.group(3)])
        return str(_TENS[m.group(1)] + (_WORD[m.group(2)] if m.group(2) else 0))
    return _WORDS_RX.sub(sub, text)


_NUM = r"\d+(?:\.\d+)?"
_CUR = r"(?:s\$|\$|sgd\s?)"
_AMOUNT_RX = re.compile(
    rf"(?:(?P<cur>{_CUR})\s*|(?<![\w.:$£]))(?P<lo>{_NUM})(?![\d:]|\.\d)"
    rf"(?:\s*(?:-|–|~|to)\s*{_CUR}?\s*(?P<hi>{_NUM})(?![\d:]))?"
    r"(?:\s*(?P<unit>dollars?|bucks?|sgd|kuai)\b)?")
# after a bare number: not money ("2 pax", "7pm", "1 km", "10 mins", "4 stars", "2 nights")
_NOT_MONEY = {"pax", "people", "ppl", "person", "persons", "guests", "of", "am", "pm", "km", "kms",
              "m", "metres", "meters", "min", "mins", "minute", "minutes", "h", "hr", "hrs", "hour",
              "hours", "star", "stars", "night", "nights", "day", "days", "st", "nd", "rd", "th",
              "o", "oclock", "stop", "stops", "pound", "pounds", "mile", "miles", "adults",
              "kids", "children", "seats", "years", "weeks", "times", "floor"}
# a bare number is money when one of these comes right before it (up to two words between) ...
_CAP_BEFORE_RX = re.compile(
    r"(?:\b(?:under|below|less than|cheaper than|within|max|maximum|up to|upto|at most|no more than|"
    r"not more than|not exceeding|exceed|budget|spend|afford|only have|only got)\b(?:\W+\w+){0,2}"
    r"|<=?)\W*$")
# ... or right after it ("8 max", "7 per pax", "10 each")
_CAP_AFTER_RX = re.compile(r"\s*(?:max|maximum|tops|or less|and below|or below|and under|each|"
                           r"per (?:pax|person|head)|a (?:pax|person|head)|/pax|pp)\b")
# a floor, not a cap: "above $20", "more than $20", "$20 and above", "$20+" ...
_FLOOR_BEFORE_RX = re.compile(r"\b(?:above|over|more than|at least|minimum)\W*$")
_FLOOR_AFTER_RX = re.compile(r"\s*(?:\+|and above|or more|and more|onwards|and up)")
# ... unless negated: "no more than $10", "dun want more than $8", "cannot be more than $8"
_NEGATED_RX = re.compile(r"\b(?:no|not|dun|don'?t|do not|cannot|can'?t|never|won'?t)\b(?:\W+\w+){0,3}?"
                         r"\W+(?:above|over|more than)\W*$")
# no number, but no cap either
_NO_LIMIT_RX = re.compile(
    r"\b(?:price|budget|cost|money)\s*(?:also\s*|is\s*)?(?:dun|don'?t|doesn'?t|does not|do not|not|no)\s*"
    r"(?:matter|care|issue|problem|limit)|\b(?:any|no|whatever|forget(?: about)?(?: the)?|nvm|"
    r"never mind(?: the)?)\s*(?:budget|price)\b|\bno budget\b|\bdun mind pay(?:ing)? more|"
    r"\b(?:do not|don'?t) mind paying more|\b(?:dun|don'?t|do not) care (?:about )?(?:the )?"
    r"(?:price|budget)|\bnot (?:bothered|fussed) about (?:the )?price")


def stated_budget(text: str):
    """The budget this message states: "<=N" (a cap), ANY (a floor such as "above $20", or "price
    dun matter"), or None. "budget 8-10" -> "<=10"; if several amounts count, the last one wins.

    A number is money with a currency or unit ("$8", "8 dollars", "fifteen bucks", "8 sgd"), or a
    cap word before or after it ("max 8", "under 12", "budget 8-10", "7 per pax", "8 max"), and
    never when a unit follows it ("2 pax", "7pm", "1 km", "10 mins", "4 stars", "10:15")."""
    t = " ".join(_digits(text.lower()).split())
    t = re.sub(r"\bbetween\s+(\S+)\s+and\s+", r"\1 to ", t)
    found = None
    for m in _AMOUNT_RX.finditer(t):
        before, after = t[:m.start()], t[m.end():]
        money = bool(m.group("cur") or m.group("unit"))
        if not money:
            nxt = re.match(r"\s*([a-z]+)", after)
            if (nxt and nxt.group(1) in _NOT_MONEY) or re.match(r"\s*[:.]\d", after):
                continue
            if not (_CAP_BEFORE_RX.search(before) or _CAP_AFTER_RX.match(after)):
                continue
        if (_FLOOR_BEFORE_RX.search(before) and not _NEGATED_RX.search(before)) \
                or _FLOOR_AFTER_RX.match(after):
            found = ANY
            continue
        cap = max(float(m.group("lo")), float(m.group("hi") or 0))
        found = f"<={cap:g}"
    if found is None and _NO_LIMIT_RX.search(t):
        found = ANY
    return found


# ---------------------------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------------------------

def user_turns(prompt: str) -> list[tuple[str, str]]:
    """(User line, Normalized line) for each user turn in a [TRACK] prompt, oldest first."""
    turns = []
    for line in prompt.split("\n")[1:]:
        if line.startswith("User: "):
            turns.append([line[len("User: "):], ""])
        elif line.startswith("Normalized: ") and turns:
            turns[-1][1] = line[len("Normalized: "):]
    return [tuple(t) for t in turns]


def latest_stated(prompt: str, numeric: bool = False):
    """The latest budget the user stated up to this turn ("<=N" or ANY), or None. numeric=True
    skips ANY: an old bucket label already says there is a cap, and MultiWOZ's "price doesn't
    matter" is often about the hotel in the same dialogue."""
    said = None
    for user, normalized in user_turns(prompt):
        s = stated_budget(user) or stated_budget(normalized)
        if s and not (numeric and s == ANY):
            said = s
    return said


def row_id(row: dict) -> str:
    return row.get("id") or f"{row['dialogue_id']}_{row['turn_id']}"


def _override(row: dict):
    dialogue, turn = row_id(row).rsplit("_", 1)
    hit = OVERRIDES.get(dialogue)
    return hit[1] if hit and int(turn) >= hit[0] else None


def relabel(row: dict) -> tuple[str, str]:
    """(new budget label, why) for one row. why is one of: kept, said, no number, above,
    override, and disagree (the user's words don't fit the old bucket: its top is kept)."""
    old = json.loads(row["output"])["budget"]
    if old == "above $20":
        return ANY, "above"
    if old in BUCKETS:
        lo, hi = BUCKETS[old]
        said = latest_stated(row["input"], numeric=True)
        if said is None:
            return f"<={hi}", "no number"
        if lo <= float(said[2:]) <= hi:
            return said, "said"
        return f"<={hi}", "disagree"  # e.g. "under $25" then "cheap, please": the bucket is right
    if old == UNKNOWN and _override(row):
        return _override(row), "override"
    return old, "kept"


def disagreement(row: dict, label: str):
    """What the user said, if it doesn't match the label the row ends up with (else None).
    In MultiWOZ only amounts count: its dialogues also book hotels and trains, so "price doesn't
    matter" may not be about the restaurant."""
    said = latest_stated(row["input"], numeric=row.get("source") == "multiwoz")
    if said is None or said == label:
        return None
    if said == ANY and label == UNKNOWN:
        return None  # "price dun matter" before any budget was set
    return said


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--check", action="store_true", help="report only, don't write")
    ap.add_argument("--show", type=int, default=0, help="print this many random changed rows")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    changed_any, changed_rows, disagree = False, [], []
    for path in a.files or DEFAULT_FILES:
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        moves, whys = Counter(), Counter()
        for row in rows:
            out = json.loads(row["output"])
            new, why = relabel(row)
            whys[why] += 1
            if new != out["budget"]:
                moves[(out["budget"], new)] += 1
                changed_rows.append((path.name, row, out["budget"], new))
                out["budget"] = new
                row["output"] = json.dumps(out)
            said = disagreement(row, new)
            if said:
                disagree.append((path.name, row_id(row), new, said, user_turns(row["input"])[-1][0]))
        print(f"{path.relative_to(DATA) if path.is_relative_to(DATA) else path}: {len(rows)} rows, "
              f"{sum(moves.values())} relabelled")
        for (old, new), n in sorted(moves.items(), key=lambda kv: (kv[0][0], -kv[1])):
            print(f"  {n:5d}  {old:10s} -> {new}")
        if whys["no number"] or whys["disagree"]:
            print(f"  no number in the user's lines: {whys['no number']} rows (bucket top kept)"
                  + (f", words disagree with the bucket: {whys['disagree']}" if whys["disagree"] else ""))
        changed_any = changed_any or bool(moves)
        if not a.check and moves:
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.writelines(json.dumps(r) + "\n" for r in rows)

    if disagree:
        print(f"\n{len(disagree)} rows where the user's words and the label disagree:")
        for name, rid, label, said, last in disagree:
            print(f"  {name} {rid}: label {label}, user said {said}  (last user line: {last[:80]!r})")
    for name, row, old, new in rng.sample(changed_rows, min(a.show, len(changed_rows))):
        said = [u for u, _ in user_turns(row["input"]) if stated_budget(u)]
        print(f"\n{name} {row_id(row)}: {old} -> {new}")
        print("  user lines with an amount:", said or "-")
        print("  last user line:", user_turns(row["input"])[-1][0])
    if a.check:
        print("\nwould change rows" if changed_any else "\nnothing to change")
        sys.exit(1 if changed_any else 0)


if __name__ == "__main__":
    main()
