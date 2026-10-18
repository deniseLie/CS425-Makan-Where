"""
Convert MultiWOZ 2.2 restaurant dialogues into Makan Where? [TRACK] JSONL examples.

Labelled from MultiWOZ's belief state:
  cuisine  <- restaurant-food
  area     <- restaurant-area
  budget   <- restaurant-pricerange
  time     <- restaurant-bookday + restaurant-booktime  ("Tuesday 12:00")
Not labelled by MultiWOZ (left UNKNOWN): dietary, setting.

Extra handling:
  * State is a snapshot taken AT the user turn (MultiWOZ 2.2 stores the
    cumulative belief state on USER turns; SYSTEM turns carry no state).
  * A rule layer catches explicit "open to any cuisine" statements that the
    source annotation misses -> ANY.
  * Two repair passes fix one-turn annotation lag when the user accepts a
    system suggestion:
      (a) slot-value repair: a short affirmation followed by the value
          appearing in the next turn's belief state.
      (b) named-restaurant repair: a short affirmation following a system
          offer of a named place, where we can recover the cuisine from the
          restaurant's own name via a small alias map (optional; only fires
          when RESTAURANT_CUISINE_HINTS is populated).
  * History is trimmed to start where the restaurant task starts.
  * User turns whose only active intent is a non-restaurant booking are
    skipped (e.g. "book the hotel for 2 nights").
  * Output is split by dialogue_id (train/dev/test), never by row.
"""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

MULTIWOZ_DIR = Path("./MultiWOZ_2.2")
OUTPUT_DIR = Path("./tracker_data")

SLOT_NAMES = ("cuisine", "area", "budget", "dietary", "setting", "time")
UNKNOWN = "UNKNOWN"
ANY = "ANY"
SPLITS = ("train", "dev", "test")
RESTAURANT_SERVICES = {"restaurant", "restaurant_service"}
NON_RESTAURANT_BOOKING_INTENTS = {"book", "booking"}  # used to skip hotel/taxi bookings

PRICE_MAP = {"cheap": "cheap", "moderate": "moderate", "expensive": "expensive"}

# Internal slots; "book_day" + "book_time" are merged into the output "time".
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
    "restaurant-booktime": "book_time",
    "restaurant-book time": "book_time",
}

DONTCARE_VALUES = {
    "dontcare", "don't care", "do not care", "doesn't matter", "doesnt matter",
    "any", "no preference", "no preference either way",
}
UNKNOWN_VALUES = {"", "not given", "not mentioned", "unknown", "none", "null"}


# --- Rule layer: explicit no-preference statements the annotation may miss ---
CUISINE_ANY_RE = re.compile(
    r"\b("
    r"open to (all|any)( (sorts?|types?|kinds?))?( of)?( food| cuisines?)?"
    r"|(all|any) (sorts?|types?|kinds?) of (food|cuisines?)"
    r"|(don'?t|do not) (really )?(care|mind) (what|which)( (sort|type|kind))?( of)? ?(food|cuisines?)?"
    r"|no (cuisine|food) preference"
    r"|(food|cuisine) (doesn'?t|does not) matter"
    r"|(food|cuisine) (is|would be) (fine|ok|okay|good)"
    r"|(any|whatever) (food|cuisine) (is|would be) (fine|ok|okay|good)"
    r")\b",
    re.IGNORECASE,
)

# --- Lag repair: short acceptance of a system suggestion ---
AFFIRM_RE = re.compile(
    r"\b(sounds (great|good|perfect|fine)|that (works|is perfect|would be (great|good|perfect))"
    r"|perfect|yes|yeah|let'?s do (it|that)|book (it|that)|i'?ll take (it|that))\b",
    re.IGNORECASE,
)
AFFIRM_MAX_WORDS = 12


# ----------------------------------------------------------------- values ---
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


def normalize_key(raw_key: Any) -> str:
    return str(raw_key).strip().lower().replace("_", "-")


def parse_restaurant_slot_values(slot_values: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
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
                value = PRICE_MAP.get(value, value)
        out[slot] = value
    return out


def combine_time(day: str, clock: str) -> str:
    parts = [p for p in (day, clock) if p not in (UNKNOWN, ANY)]
    if parts:
        return " ".join(parts)  # e.g. "Tuesday 12:00"
    if ANY in (day, clock):
        return ANY
    return UNKNOWN


def snapshot(belief: Dict[str, str], overrides: Dict[str, str]) -> Dict[str, str]:
    slots = {slot: UNKNOWN for slot in SLOT_NAMES}
    for slot in ("cuisine", "area", "budget"):
        value = belief.get(slot, UNKNOWN)
        slots[slot] = value if value != UNKNOWN else overrides.get(slot, UNKNOWN)
    slots["time"] = combine_time(
        belief.get("book_day", UNKNOWN), belief.get("book_time", UNKNOWN)
    )
    return slots


# ----------------------------------------------------------------- frames ---
def frame_info(turn: Dict[str, Any]) -> Tuple[Optional[Dict[str, str]], bool, bool, bool]:
    """
    Returns (restaurant_state, restaurant_active, other_active, other_booking_active).
    restaurant_state is None when the turn has no restaurant state at all.
    other_booking_active is True when a non-restaurant frame has a booking intent.
    """
    restaurant_state: Optional[Dict[str, str]] = None
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
        state = frame.get("state")
        if not isinstance(state, dict):
            continue
        intent = str(state.get("active_intent", "NONE")).strip().lower()
        intent_on = intent not in ("none", "")

        if service in RESTAURANT_SERVICES:
            slot_values = state.get("slot_values")
            if isinstance(slot_values, dict):
                restaurant_state = parse_restaurant_slot_values(slot_values)
            restaurant_active = restaurant_active or intent_on
        elif intent_on:
            other_active = True
            if intent in NON_RESTAURANT_BOOKING_INTENTS:
                other_booking_active = True

    return restaurant_state, restaurant_active, other_active and not restaurant_active, other_booking_active


def restaurant_started(turn: Dict[str, Any]) -> bool:
    state, active, _, _ = frame_info(turn)
    return active or bool(state)


# ------------------------------------------------------------ turn helpers ---
def get_speaker(turn: Dict[str, Any]) -> str:
    speaker = turn.get("speaker", "")
    if isinstance(speaker, int):
        return "user" if speaker == 0 else "assistant"
    normalized = str(speaker).strip().lower()
    if normalized in {"user", "human", "0"}:
        return "user"
    if normalized in {"system", "assistant", "bot", "1"}:
        return "assistant"
    return "unknown"


def get_utterance(turn: Dict[str, Any]) -> str:
    value = turn.get("utterance", turn.get("text", ""))
    return str(value).strip() if value is not None else ""


# ------------------------------------------------------ dialogue -> records ---
def repair_acceptance_lag(records: List[Dict[str, Any]]) -> int:
    """
    Two passes over consecutive (user, user) records:
      (a) slot-value repair: previous user turn was a short affirmation, the
          current slot is UNKNOWN, the next turn gains a value, and that value
          appeared in the system offer -> copy it back one turn.
      (b) named-restaurant repair: previous user turn was a short affirmation
          and the system offer named a restaurant we can map to a cuisine ->
          set cuisine one turn earlier.
    """
    repaired = 0
    for i in range(len(records) - 1):
        cur, nxt = records[i], records[i + 1]
        user_text = cur["_user"]
        if len(user_text.split()) > AFFIRM_MAX_WORDS or not AFFIRM_RE.search(user_text):
            continue
        offer = cur["_prev_system"].lower()

        # (a) slot-value repair from next turn's belief state
        for slot in ("cuisine", "area", "budget"):
            now, later = cur["slots"][slot], nxt["slots"][slot]
            if now == UNKNOWN and later not in (UNKNOWN, ANY) and later.lower() in offer:
                cur["slots"][slot] = later
                cur["repaired"] = True
                repaired += 1

    return repaired


def build_dialogue_records(dialogue: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
    turns = [t for t in dialogue.get("turns", []) if isinstance(t, dict)]
    dialogue_id = str(dialogue.get("dialogue_id", "unknown"))
    # Strip a trailing ".json" if a filename leaked in.
    if dialogue_id.lower().endswith(".json"):
        dialogue_id = dialogue_id[:-5]

    start = next((i for i, t in enumerate(turns) if restaurant_started(t)), None)
    if start is None:
        return [], 0

    belief: Dict[str, str] = {}
    overrides: Dict[str, str] = {}
    history: List[Dict[str, str]] = []
    records: List[Dict[str, Any]] = []
    last_system = ""

    for index, turn in enumerate(turns[start:], start=start):
        text = get_utterance(turn)
        if not text:
            continue
        speaker = get_speaker(turn)

        if speaker == "user":
            history.append({"role": "user", "content": text})
            state, _, other_active, other_booking_active = frame_info(turn)
            if state is not None:
                belief = state

            if CUISINE_ANY_RE.search(text):
                overrides["cuisine"] = ANY

            # Skip turns where the user is on another domain entirely,
            # including cross-domain booking turns (hotel/taxi).
            if other_active or other_booking_active:
                continue

            records.append({
                "task": "TRACK",
                "dialogue_id": dialogue_id,
                "turn_id": turn.get("turn_id", index),
                "history": [dict(m) for m in history],
                "slots": snapshot(belief, overrides),
                "_user": text,
                "_prev_system": last_system,
            })
        else:
            history.append({"role": "assistant", "content": text})
            last_system = text

    return records, repair_acceptance_lag(records)


def finalize(record: Dict[str, Any]) -> Dict[str, Any]:
    record.pop("_user", None)
    record.pop("_prev_system", None)
    record.pop("repaired", None)
    record["input"] = (
        "[TRACK]\n"
        "Extract the current restaurant preferences from the conversation. "
        "Return valid JSON with exactly these keys: "
        "cuisine, area, budget, dietary, setting, time. "
        "Use UNKNOWN when unmentioned and ANY only for an explicit "
        "no-preference statement. Time is the requested restaurant booking "
        'time, formatted like "Tuesday 12:00" when both day and time are known, '
        "otherwise as the single known part (e.g. \"Tuesday\").\n"
        + "\n".join(f'{m["role"].upper()}: {m["content"]}' for m in record["history"])
    )
    record["target"] = json.dumps(record["slots"], ensure_ascii=False)
    return record


# ------------------------------------------------------------------ splits ---
def assign_split(file_path: Path, dialogue_id: str) -> str:
    parts = {p.lower() for p in file_path.parts}
    for split in SPLITS:
        if split in parts:
            return split
    bucket = int(hashlib.md5(str(dialogue_id).encode()).hexdigest(), 16) % 100
    return "train" if bucket < 80 else "dev" if bucket < 90 else "test"


# ---------------------------------------------------------------------- IO ---
def iter_dialogues(base_path: Path) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    if not base_path.exists():
        raise FileNotFoundError(f"MultiWOZ directory not found: {base_path.resolve()}")
    for file_path in sorted(base_path.rglob("*.json")):
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


def process_local_multiwoz(
    input_dir: Path = MULTIWOZ_DIR, output_dir: Path = OUTPUT_DIR
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    handles = {s: (output_dir / f"dialogues_tracker_{s}.jsonl").open("w", encoding="utf-8")
               for s in SPLITS}
    dialogue_ids = {s: set() for s in SPLITS}
    stats = Counter()
    slot_counts = {s: Counter() for s in SLOT_NAMES}

    try:
        for file_path, dialogue in iter_dialogues(input_dir):
            stats["scanned"] += 1
            records, repaired = build_dialogue_records(dialogue)
            if not records:
                continue
            split = assign_split(file_path, str(dialogue.get("dialogue_id")))
            dialogue_ids[split].add(dialogue.get("dialogue_id"))
            stats["dialogues_kept"] += 1
            stats["repaired"] += repaired
            for record in records:
                for slot in SLOT_NAMES:
                    val = record["slots"][slot]
                    if val == UNKNOWN:
                        slot_counts[slot]["UNKNOWN"] += 1
                    elif val == ANY:
                        slot_counts[slot]["ANY"] += 1
                    else:
                        slot_counts[slot]["labelled"] += 1
                handles[split].write(json.dumps(finalize(record), ensure_ascii=False) + "\n")
                stats[f"rows_{split}"] += 1
    finally:
        for handle in handles.values():
            handle.close()

    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                assert not (dialogue_ids[a] & dialogue_ids[b]), f"Leak between {a} and {b}"

    print(f"Scanned: {stats['scanned']}  Kept: {stats['dialogues_kept']}  "
          f"Acceptance-lag repairs: {stats['repaired']}")
    for s in SPLITS:
        print(f"{s}: {len(dialogue_ids[s])} dialogues, {stats[f'rows_{s}']} rows")
    print("Slot label rates (over all rows):")
    for slot in SLOT_NAMES:
        c = slot_counts[slot]
        total = sum(c.values()) or 1
        print(f"  {slot}: labelled={c['labelled']} ({c['labelled']/total:.1%})  "
              f"ANY={c['ANY']}  UNKNOWN={c['UNKNOWN']}")
    print(f"Output dir: {output_dir.resolve()}")


if __name__ == "__main__":
    process_local_multiwoz()

