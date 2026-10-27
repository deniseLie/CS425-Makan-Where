"""[TRACK] format: build_prompt / build_target / parse against the tracker dataset files."""

import json

import pytest

from src.config import DATA_DIR
from src.schemas import SLOTS, DialogueState, TrackerOutput, Turn
from src.tracker.hf import build_prompt, build_target, parse

DATASET_FILES = sorted((DATA_DIR / "synthetic_data").glob("*.jsonl")) + \
    sorted((DATA_DIR / "woz_converted_data").glob("*.jsonl"))


def turns_from_prompt(prompt: str) -> list[Turn]:
    """Read the turns back out of a row's input, one tagged line at a time."""
    lines = prompt.split("\n")
    assert lines[0] == "[TRACK]"
    turns: list[Turn] = []
    for line in lines[1:]:
        tag, _, text = line.partition(": ")
        if tag == "User":
            turns.append(Turn(raw=text))
        elif tag == "Normalized":
            turns[-1].normalized = text
        elif tag == "System":
            turns[-1].reply = text
        else:
            raise AssertionError(f"unexpected line {line!r}")
    return turns


@pytest.mark.parametrize("path", DATASET_FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_round_trip_dataset_rows(path):
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    if not rows[0]["input"].startswith("[TRACK]\nUser:"):
        pytest.skip("not in the shared [TRACK] format yet (Part A rewrites this file)")

    bad = []
    for row in rows:
        try:
            assert build_prompt(turns_from_prompt(row["input"])) == row["input"]
            gold = json.loads(row["output"])
            out = parse(row["output"], DialogueState())
            assert out.slots == {s: gold[s] for s in SLOTS}
            assert out.critique == (None if gold["feedback"] == "none" else gold["feedback"])
            assert build_target(out) == row["output"]
        except AssertionError as e:
            bad.append(f"{row.get('id', row.get('dialogue_id'))}: {e}")
    assert not bad, f"{len(bad)} of {len(rows)} rows differ, e.g. {bad[:3]}"


def test_build_prompt_layout():
    history = [Turn(raw="must got aircon sia", normalized="It must be air-conditioned.",
                    reply="There is Western, Japanese, Indian nearby, which would you like?"),
               Turn(raw="Western please")]  # normalizer returned nothing -> repeat the raw text
    assert build_prompt(history) == (
        "[TRACK]\n"
        "User: must got aircon sia\n"
        "Normalized: It must be air-conditioned.\n"
        "System: There is Western, Japanese, Indian nearby, which would you like?\n"
        "User: Western please\n"
        "Normalized: Western please"
    )


def test_parse_reads_slots_and_feedback():
    prev = DialogueState(slots={"cuisine": "korean", "dietary": "halal"})
    out = parse('Sure: {"cuisine": "Western", "area": "Ang Mo Kio", "budget": "below $10", '
                '"setting": "UNKNOWN", "time": "19:30", "feedback": "too_far"}', prev)
    assert out.slots == {"cuisine": "Western", "area": "Ang Mo Kio", "budget": "below $10",
                         "dietary": "halal",  # missing key -> previous value
                         "setting": "UNKNOWN", "time": "19:30"}
    assert out.critique == "too_far" and out.reject is None


@pytest.mark.parametrize("feedback,critique", [
    ("none", None), ("too_far", "too_far"), ("too_expensive", "too_expensive"),
    ("not_interested", "not_interested"), ("too_salty", None),
])
def test_feedback_becomes_critique(feedback, critique):
    assert parse(json.dumps({"feedback": feedback}), DialogueState()).critique == critique


@pytest.mark.parametrize("text", ["not json", "[1, 2]", '{"cuisine": "Western"'])
def test_unparseable_keeps_previous_state(text):
    prev = DialogueState(slots={"cuisine": "korean", "area": "downtown core"})
    out = parse(text, prev)
    assert out.slots == prev.slots and out.critique is None


def test_build_target_order_and_feedback():
    out = TrackerOutput(slots={"cuisine": "Western"}, critique="not_interested")
    assert build_target(out) == ('{"cuisine": "Western", "area": "UNKNOWN", "budget": "UNKNOWN", '
                                 '"dietary": "UNKNOWN", "setting": "UNKNOWN", "time": "UNKNOWN", '
                                 '"feedback": "not_interested"}')
    assert json.loads(build_target(TrackerOutput()))["feedback"] == "none"
