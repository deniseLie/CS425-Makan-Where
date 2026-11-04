"""
Check that every [TRACK] row follows the shared tracker format
(Part A, section 2: prompt layout, answer layout, allowed values, time rule).

Usage (from the repo root):
    python data/check_tracker_format.py                 # all tracker files
    python data/check_tracker_format.py path/to.jsonl   # specific files
Exits with status 1 if any row breaks the format.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

from area_pool import AREA_POOL
from build_synthetic import ALIASES, CUISINES, DISHES, REGIONS
from convert_final import CUISINE_MAP

DATA = Path(__file__).resolve().parent
DEFAULT_FILES = (sorted(DATA.glob("woz_converted_data/dialogues_*.jsonl"))
                 + sorted(DATA.glob("synthetic_data/dialogues_synth_*.jsonl")))

KEYS = ["cuisine", "area", "budget", "dietary", "setting", "time", "feedback"]
FEEDBACK = {"none", "too_far", "too_expensive", "not_interested"}
ALLOWED = {
    "cuisine": set(CUISINES) | set(DISHES) | {label for label, _ in CUISINE_MAP.values()},
    "area": {a for pool in AREA_POOL.values() for a in pool} | set(ALIASES) | set(REGIONS),
    "budget": {"below $5", "below $10", "$10-20", "above $20"},
    "dietary": {"halal", "vegetarian", "vegan", "no beef", "no pork"},
    "setting": {"hawker", "aircon", "cafe", "food court", "restaurant", "takeaway"},
}
TIME_RE = re.compile(r"breakfast|lunch|dinner|late night|now|([01]\d|2[0-3]):[0-5]\d")
# Leftover Cambridge contact details (deliberately looser than convert_final.py's).
CAMBRIDGE_RE = re.compile(r"\b0?1\s?223[\s-]?\d|\+44|\b0\d{9,11}\b|\bc\.?\s?b\.?\s?\d|\bpe\d",
                          re.IGNORECASE)


def check_input(text):
    """Yield (kind, detail) for each way the prompt breaks the layout."""
    lines = text.split("\n")
    if lines[0] != "[TRACK]" or len(lines) < 3 or not lines[1].startswith("User: "):
        yield "prompt start", "must be [TRACK] then a User line"
    prev = "System"
    for line in lines[1:]:
        speaker = line.split(": ", 1)[0]
        if speaker not in ("User", "Normalized", "System") or ": " not in line:
            yield "prompt line", repr(line[:70])
            return
        if (speaker == "Normalized") != (prev == "User"):
            yield "prompt order", f"each User line needs one Normalized line: {line[:60]!r}"
            return
        prev = speaker
    if prev != "Normalized":
        yield "prompt end", "must end with the current turn's Normalized line"
    match = CAMBRIDGE_RE.search(text)
    if match:
        yield "cambridge", repr(match.group())


def check_output(out):
    if list(out) != KEYS:
        yield "output keys", str(list(out))
        return
    if out["feedback"] not in FEEDBACK:
        yield "feedback", repr(out["feedback"])
    for slot in KEYS[:-1]:
        value = out[slot]
        if value in ("UNKNOWN", "ANY"):
            continue
        if not (TIME_RE.fullmatch(value) if slot == "time" else value in ALLOWED[slot]):
            yield slot, repr(value)


def main():
    paths = [Path(p) for p in sys.argv[1:]] or DEFAULT_FILES
    failed = False
    for path in paths:
        problems, example, rows, singlish_exact = Counter(), {}, 0, 0
        for n, line in enumerate(open(path, encoding="utf-8"), 1):
            row = json.loads(line)
            rows += 1
            try:
                out = json.loads(row["output"])
            except json.JSONDecodeError:
                out, errors = {}, [("output json", row["output"][:60])]
            else:
                errors = list(check_output(out))
            errors += check_input(row["input"])
            for kind, detail in errors:
                problems[kind] += 1
                example.setdefault(kind, f"line {n}: {detail}")
            if row.get("style") == "singlish" and re.fullmatch(r"\d\d:\d\d", str(out.get("time"))):
                singlish_exact += 1

        synthetic = row.get("source") == "synthetic" if rows else False
        if synthetic and not singlish_exact:
            problems["no singlish exact time"] += 1
            example["no singlish exact time"] = "no Singlish row has an HH:MM time"
        failed = failed or bool(problems)
        extra = f", {singlish_exact} Singlish rows with exact times" if synthetic else ""
        print(f"{'FAIL' if problems else 'ok  '} {path.name}: {rows} rows{extra}")
        for kind, count in problems.most_common():
            print(f"     {count:5d} x {kind:12s} e.g. {example[kind]}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
