#!/usr/bin/env python3
"""
Convert MultiWOZ 2.2 restaurant dialogues into Makan Where? [TRACK] data.

Outputs:
    data/dialogues_train.jsonl
    data/dialogues_dev.jsonl
    data/dialogues_test.jsonl

Each JSONL row represents one eligible USER turn:
    input  : [TRACK] + relevant conversation history
    output : JSON with exactly six slots

Slots:
    cuisine, area, budget, dietary, setting, time

Labels:
    UNKNOWN = not specified
    ANY     = explicit lack of preference
    value   = normalized preference

Official split directories are preserved. If a dialogue cannot be assigned
to an official split, processing stops instead of creating a new split.
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SLOTS = ("cuisine", "area", "budget", "dietary", "setting", "time")
SPLITS = ("train", "dev", "test")
UNKNOWN = "UNKNOWN"
ANY = "ANY"

RESTAURANT_SERVICES = {"restaurant", "restaurant_service"}
NON_RESTAURANT_BOOKING_INTENTS = {"book", "booking"}

# ---------------------------------------------------------------------
# Singapore localization
# Replace these pools with the application's canonical database values.
# ---------------------------------------------------------------------

AREA_POOL = {
    "central": [
        "Downtown Core", "Bukit Merah", "Bishan", "Toa Payoh",
        "Geylang", "Kallang", "Bukit Timah", "Tanglin", "Outram",
        "River Valley", "Queenstown", "Novena", "Marine Parade",
    ],
    "east": ["Tampines", "Bedok", "Pasir Ris"],
    "west": [
        "Clementi", "Bukit Batok", "Jurong West", "Jurong East",
        "Choa Chu Kang", "Bukit Panjang",
    ],
    "north": ["Sembawang", "Yishun", "Woodlands"],
    "north-east": [
        "Hougang", "Ang Mo Kio", "Serangoon", "Sengkang", "Punggol",
    ],
}

REGION = {
    "centre": "central",
    "center": "central",
    "south": "central",
    "north": "north",
    "east": "east",
    "west": "west",
}

# Confirm these ranges match the application's SQL filters.
BUDGET_MAP = {
    "cheap": "below $10",
    "moderate": "$10-20",
    "expensive": "above $20",
}

# MultiWOZ cuisine -> (canonical output label, utterance replacement).
# Unsupported non-ANY cuisines cause the dialogue to be dropped.
CUISINE_MAP = {
    "chinese": ("Chinese", None),
    "cantonese": ("Chinese", "chinese"),
    "indian": ("Indian", None),
    "korean": ("Korean", None),
    "japanese": ("Japanese", None),
    "thai": ("Thai", None),
    "vietnamese": ("Vietnamese", None),
    "italian": ("Italian", None),
    "french": ("French", None),
    "mexican": ("Mexican", None),
    "malaysian": ("Malay", "malay"),
}

# Optional aliases for recovering a cuisine from a named restaurant.
# Populate only with verified aliases from the restaurant database.
RESTAURANT_CUISINE_HINTS = {}

DIRECTION = r"(?:centre|center|north|south|east|west)"
DIR_RE = re.compile(
    rf"\b(?:the\s+)?(?:(?:city|town)\s+)?({DIRECTION})"
    rf"(?:\s+(?:part\s+of\s+town|part|side|area|of\s+town))?\b"
    rf"(?!\s+(?:american|indian|african|asian|korean|european))",
    re.IGNORECASE,
)

DONTCARE_VALUES = {
    "dontcare", "don't care", "do not care", "doesn't matter",
    "doesnt matter", "any", "no preference",
    "no preference either way",
}

UNKNOWN_VALUES = {
    "", "not given", "not mentioned", "unknown", "none", "null",
}

CUISINE_ANY_RE = re.compile(
    r"\b("
    r"open to (all|any)( (sorts?|types?|kinds?))?( of)?"
    r"( food| cuisines?)?"
    r"|(all|any) (sorts?|types?|kinds?) of (food|cuisines?)"
    r"|(don'?t|do not) (really )?(care|mind) (what|which)"
    r"( (sort|type|kind))?( of)? ?(food|cuisines?)?"
    r"|no (cuisine|food) preference"
    r"|(food|cuisine) (doesn'?t|does not) matter"
    r"|(food|cuisine) (is|would be) (fine|ok|okay|good)"
    r"|(any|whatever) (food|cuisine) (is|would be) (fine|ok|okay|good)"
    r")\b",
    re.IGNORECASE,
)

AFFIRM_RE = re.compile(
    r"\b("
    r"sounds (great|good|perfect|fine)"
    r"|that (works|is perfect|would be (great|good|perfect))"
    r"|perfect|yes|yeah|let'?s do (it|that)"
    r"|book (it|that)|i'?ll take (it|that)"
    r")\b",
    re.IGNORECASE,
)
AFFIRM_MAX_WORDS = 12

INTERNAL_KEY_MAP = {
    "food": "cuisine",
    "restaurant-food": "cuisine",
    "area": "area",
    "restaurant-area": "area",
    "pricerange": "budget",
    "price range": "budget",
    "restaurant-pricerange": "budget",
    "restaurant-price range": "budget",
    "restaurant-bookday": "book_day",
    "restaurant-book day": "book_day",
    "bookday": "book_day",
    "book day": "book_day",
    "restaurant-booktime": "book_time",
    "restaurant-book time": "book_time",
    "booktime": "book_time",
    "book time": "book_time",
}


# ---------------------------------------------------------------------
# Value normalization
# ---------------------------------------------------------------------

def normalize_key(key: Any) -> str:
    return str(key).strip().lower().replace("_", "-")


def clean_value(value: Any) -> str:
    if value is None:
        return UNKNOWN

    if isinstance(value, list):
        if not value:
            return UNKNOWN
        value = value[0]

    text = str(value).strip()
    lowered = " ".join(text.lower().replace("_", " ").split())

    if lowered in UNKNOWN_VALUES:
        return UNKNOWN
    if lowered in DONTCARE_VALUES:
        return ANY

    return lowered


def parse_restaurant_slot_values(
    slot_values: Dict[str, Any],
) -> Dict[str, str]:
    result = {}

    for raw_key, raw_value in slot_values.items():
        slot = INTERNAL_KEY_MAP.get(normalize_key(raw_key))
        if slot is None:
            continue

        value = clean_value(raw_value)
        if value == UNKNOWN:
            continue

        if value != ANY:
            if slot in ("cuisine", "area", "book_day"):
                value = value.title()
            elif slot == "budget":
                value = BUDGET_MAP.get(value.lower(), value)

        result[slot] = value

    return result


def combine_time(day: str, clock: str) -> str:
    known = [v for v in (day, clock) if v not in (UNKNOWN, ANY)]

    if known:
        return " ".join(known)
    if ANY in (day, clock):
        return ANY
    return UNKNOWN


def snapshot(
    belief: Dict[str, str],
    overrides: Dict[str, str],
) -> Dict[str, str]:
    slots = {name: UNKNOWN for name in SLOTS}

    for name in ("cuisine", "area", "budget"):
        value = belief.get(name, UNKNOWN)
        slots[name] = (
            value if value != UNKNOWN
            else overrides.get(name, UNKNOWN)
        )

    slots["time"] = combine_time(
        belief.get("book_day", UNKNOWN),
        belief.get("book_time", UNKNOWN),
    )

    # MultiWOZ does not directly annotate these two application slots.
    slots["dietary"] = UNKNOWN
    slots["setting"] = UNKNOWN

    return slots


# ---------------------------------------------------------------------
# Turn and frame helpers
# ---------------------------------------------------------------------

def get_speaker(turn: Dict[str, Any]) -> str:
    speaker = turn.get("speaker", "")

    if isinstance(speaker, int):
        return "user" if speaker == 0 else "assistant"

    speaker = str(speaker).strip().lower()

    if speaker in {"user", "human", "0"}:
        return "user"
    if speaker in {"system", "assistant", "bot", "1"}:
        return "assistant"

    return "unknown"


def get_utterance(turn: Dict[str, Any]) -> str:
    value = turn.get("utterance", turn.get("text", ""))
    return str(value).strip() if value is not None else ""


def frame_info(
    turn: Dict[str, Any],
) -> Tuple[Optional[Dict[str, str]], bool, bool, bool]:
    """Return restaurant state and active-intent flags."""

    restaurant_state = None
    restaurant_active = False
    other_active = False
    other_booking_active = False

    frames = turn.get("frames", [])
    if not isinstance(frames, list):
        return None, False, False, False

    for frame in frames:
        if not isinstance(frame, dict):
            continue

        service = str(frame.get("service", "")).strip().lower()
        state = frame.get("state", {})
        if not isinstance(state, dict):
            state = {}

        intent = str(state.get("active_intent", "NONE")).strip().lower()
        intent_on = intent not in ("none", "")

        if service in RESTAURANT_SERVICES:
            slot_values = state.get("slot_values", {})
            if isinstance(slot_values, dict):
                restaurant_state = parse_restaurant_slot_values(slot_values)
            restaurant_active = restaurant_active or intent_on
        elif intent_on:
            other_active = True
            if intent in NON_RESTAURANT_BOOKING_INTENTS:
                other_booking_active = True

    return (
        restaurant_state,
        restaurant_active,
        other_active,
        other_booking_active,
    )


def restaurant_started(turn: Dict[str, Any]) -> bool:
    state, active, _, _ = frame_info(turn)
    return active or bool(state)


# ---------------------------------------------------------------------
# Dialogue-wide Singapore area mapping and text rewriting
# ---------------------------------------------------------------------

def source_area(value: str) -> str:
    value = value.strip().lower()
    return "centre" if value == "center" else value


def choose_area(region: str, used: set, rng: random.Random) -> str:
    pool = AREA_POOL[REGION[region]]
    available = [area for area in pool if area not in used]
    chosen = rng.choice(available or pool)
    used.add(chosen)
    return chosen


def plan_dialogue(
    dialogue: Dict[str, Any],
    rng: random.Random,
) -> Optional[Dict[str, str]]:
    """Map source regions consistently within a dialogue."""

    area_map = {}
    used = set()

    for turn in dialogue.get("turns", []):
        if get_speaker(turn) != "user":
            continue

        state, _, _, _ = frame_info(turn)
        if not state:
            continue

        area = state.get("area", UNKNOWN)
        if area not in (UNKNOWN, ANY):
            region = source_area(area)
            if region in REGION and region not in area_map:
                area_map[region] = choose_area(region, used, rng)

        cuisine = state.get("cuisine", UNKNOWN)
        if cuisine not in (UNKNOWN, ANY):
            if cuisine.lower() not in CUISINE_MAP:
                return None

    # Also localize explicit regions mentioned in the dialogue text.
    for turn in dialogue.get("turns", []):
        for match in DIR_RE.finditer(get_utterance(turn)):
            region = source_area(match.group(1))
            if region in REGION and region not in area_map:
                area_map[region] = choose_area(region, used, rng)

    return area_map


def rewrite(
    text: str,
    area_map: Dict[str, str],
    cuisine_words: Dict[str, str],
) -> str:
    def replace_direction(match):
        region = source_area(match.group(1))
        return area_map.get(region, match.group(0))

    text = DIR_RE.sub(replace_direction, text)

    for old, new in cuisine_words.items():
        text = re.sub(rf"\b{re.escape(old)}\b", new, text, flags=re.I)

    return text


def localized_slots(
    belief: Dict[str, str],
    area_map: Dict[str, str],
) -> Dict[str, str]:
    slots = snapshot(belief, {})

    cuisine = slots["cuisine"]
    if cuisine not in (UNKNOWN, ANY):
        mapped = CUISINE_MAP.get(cuisine.lower())
        slots["cuisine"] = mapped[0] if mapped else cuisine

    area = slots["area"]
    if area not in (UNKNOWN, ANY):
        region = source_area(area)
        slots["area"] = area_map.get(region, area)

    return slots


# ---------------------------------------------------------------------
# Annotation repair
# ---------------------------------------------------------------------

def repair_acceptance_lag(records: List[Dict[str, Any]]) -> int:
    repaired = 0

    for index in range(len(records) - 1):
        current = records[index]
        following = records[index + 1]

        user_text = current["_user"]
        if len(user_text.split()) > AFFIRM_MAX_WORDS:
            continue
        if not AFFIRM_RE.search(user_text):
            continue

        offer = current["_prev_system"].lower()

        # Repair a missing value only when the next label contains it
        # and that value was mentioned in the preceding system offer.
        for slot in ("cuisine", "area", "budget"):
            now = current["slots"][slot]
            later = following["slots"][slot]

            if (
                now == UNKNOWN
                and later not in (UNKNOWN, ANY)
                and later.lower() in offer
            ):
                current["slots"][slot] = later
                repaired += 1

        # Optional, conservative cuisine recovery from a named restaurant.
        if current["slots"]["cuisine"] == UNKNOWN:
            for name, cuisine in RESTAURANT_CUISINE_HINTS.items():
                if name.lower() in offer:
                    current["slots"]["cuisine"] = cuisine
                    repaired += 1
                    break

    return repaired


def build_dialogue_records(
    dialogue: Dict[str, Any],
    area_map: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], int]:
    turns = [
        turn for turn in dialogue.get("turns", [])
        if isinstance(turn, dict)
    ]

    dialogue_id = str(dialogue.get("dialogue_id", "unknown"))
    if dialogue_id.lower().endswith(".json"):
        dialogue_id = dialogue_id[:-5]

    start = next(
        (i for i, turn in enumerate(turns) if restaurant_started(turn)),
        None,
    )
    if start is None:
        return [], 0

    cuisine_words = {
        old: mapping[1]
        for old, mapping in CUISINE_MAP.items()
        if mapping[1]
    }

    history = []
    belief = {}
    overrides = {}
    records = []
    last_system = ""

    for index, turn in enumerate(turns[start:], start=start):
        original_text = get_utterance(turn)
        if not original_text:
            continue

        text = rewrite(original_text, area_map, cuisine_words)
        speaker = get_speaker(turn)

        if speaker != "user":
            if speaker == "assistant":
                history.append({"role": "assistant", "content": text})
                last_system = text
            continue

        history.append({"role": "user", "content": text})

        state, _, other_active, other_booking_active = frame_info(turn)
        if state is not None:
            belief = state

        if CUISINE_ANY_RE.search(original_text):
            overrides["cuisine"] = ANY

        # Exclude user turns whose active intent is another domain.
        if other_active or other_booking_active:
            continue

        slots = localized_slots(belief, area_map)

        # Explicit no-preference statements override missing annotations.
        if overrides.get("cuisine") == ANY and slots["cuisine"] == UNKNOWN:
            slots["cuisine"] = ANY

        records.append({
            "task": "[TRACK]",
            "source": "multiwoz",
            "dialogue_id": dialogue_id,
            "turn_id": turn.get("turn_id", index),
            "history": [dict(message) for message in history],
            "slots": slots,
            "_user": original_text,
            "_prev_system": last_system,
        })

    return records, repair_acceptance_lag(records)


def finalize(record: Dict[str, Any]) -> Dict[str, Any]:
    record = dict(record)
    record.pop("_user", None)
    record.pop("_prev_system", None)

    history = record.pop("history")
    slots = record.pop("slots")

    record["input"] = (
        "[TRACK]\n"
        "Extract the current restaurant preferences from the conversation. "
        "Return valid JSON with exactly these keys: "
        "cuisine, area, budget, dietary, setting, time. "
        "Use UNKNOWN when unmentioned and ANY only for an explicit "
        "no-preference statement. Format known booking times as "
        '"Tuesday 12:00" when both day and time are available.\n'
        + "\n".join(
            f'{message["role"].upper()}: {message["content"]}'
            for message in history
        )
    )
    record["output"] = json.dumps(slots, ensure_ascii=False)

    return record


# ---------------------------------------------------------------------
# Official splits and JSONL output
# ---------------------------------------------------------------------

def assign_split(file_path: Path, root: Path) -> str:
    """Require an official split directory; never re-split dialogues."""

    try:
        relative_parts = file_path.relative_to(root).parts
    except ValueError:
        relative_parts = file_path.parts

    matches = {
        part.lower()
        for part in relative_parts
        if part.lower() in SPLITS
    }

    if len(matches) != 1:
        raise ValueError(
            f"Cannot determine a unique official split for {file_path}. "
            f"Expected a train/dev/test directory beneath {root}."
        )

    return matches.pop()


def iter_dialogues(root: Path) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    if not root.exists():
        raise FileNotFoundError(f"MultiWOZ directory not found: {root.resolve()}")

    for file_path in sorted(root.rglob("*.json")):
        # Ignore metadata files that are not part of an official split.
        try:
            assign_split(file_path, root)
        except ValueError:
            continue

        try:
            with file_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Skipping unreadable JSON file {file_path}: {exc}")
            continue

        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict) and isinstance(payload.get("dialogues"), list):
            items = payload["dialogues"]
        elif isinstance(payload, dict) and "turns" in payload:
            items = [payload]
        else:
            continue

        for dialogue in items:
            if isinstance(dialogue, dict) and isinstance(dialogue.get("turns"), list):
                yield file_path, dialogue


def process_multiwoz(
    input_dir: Path,
    output_dir: Path,
    seed: int = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    slot_counts = {slot: Counter() for slot in SLOTS}
    dialogue_ids = {split: set() for split in SPLITS}

    handles = {
        split: (output_dir / f"dialogues_{split}.jsonl").open(
            "w", encoding="utf-8"
        )
        for split in SPLITS
    }

    try:
        for file_path, dialogue in iter_dialogues(input_dir):
            stats["scanned"] += 1

            services = dialogue.get("services")
            if services:
                normalized_services = {
                    str(service).strip().lower() for service in services
                }
                if not normalized_services.intersection(RESTAURANT_SERVICES):
                    continue

            split = assign_split(file_path, input_dir)
            rng = random.Random(f"{seed}-{split}-{dialogue.get('dialogue_id')}")

            area_map = plan_dialogue(dialogue, rng)
            if area_map is None:
                stats["unsupported_cuisine"] += 1
                continue

            records, repaired = build_dialogue_records(dialogue, area_map)
            if not records:
                continue

            dialogue_id = str(dialogue.get("dialogue_id", "unknown"))
            if dialogue_id in dialogue_ids[split]:
                stats["duplicate_dialogues"] += 1
                continue

            dialogue_ids[split].add(dialogue_id)
            stats["dialogues_kept"] += 1
            stats["repairs"] += repaired

            for record in records:
                for slot in SLOTS:
                    value = record["slots"][slot]
                    if value == UNKNOWN:
                        slot_counts[slot]["UNKNOWN"] += 1
                    elif value == ANY:
                        slot_counts[slot]["ANY"] += 1
                    else:
                        slot_counts[slot]["labelled"] += 1

                handles[split].write(
                    json.dumps(finalize(record), ensure_ascii=False) + "\n"
                )
                stats[f"rows_{split}"] += 1

    finally:
        for handle in handles.values():
            handle.close()

    for i, split_a in enumerate(SPLITS):
        for split_b in SPLITS[i + 1:]:
            overlap = dialogue_ids[split_a] & dialogue_ids[split_b]
            assert not overlap, f"Dialogue leakage: {split_a} / {split_b}: {overlap}"

    print(
        f"Scanned: {stats['scanned']} | "
        f"Kept dialogues: {stats['dialogues_kept']} | "
        f"Unsupported-cuisine drops: {stats['unsupported_cuisine']} | "
        f"Acceptance repairs: {stats['repairs']}"
    )

    for split in SPLITS:
        print(
            f"{split}: {len(dialogue_ids[split])} dialogues, "
            f"{stats[f'rows_{split}']} rows"
        )

    print("Slot label statistics:")
    for slot in SLOTS:
        counts = slot_counts[slot]
        total = sum(counts.values()) or 1
        print(
            f"  {slot}: labelled={counts['labelled']} "
            f"({counts['labelled'] / total:.1%}), "
            f"ANY={counts['ANY']}, UNKNOWN={counts['UNKNOWN']}"
        )

    print(f"Output directory: {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Build Makan Where? restaurant tracker training data."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("MultiWOZ_2.2"),
        help="MultiWOZ 2.2 root containing train/dev/test directories",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data"),
        help="Output directory for JSONL files",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    process_multiwoz(args.root, args.out, args.seed)


if __name__ == "__main__":
    main()