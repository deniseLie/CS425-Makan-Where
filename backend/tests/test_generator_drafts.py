import json
import random

from src.config import ROOT
from src.generator.drafts import build_drafts, catalogue_vocab, dialogues, replay, training_row
from src.generator.hf import build_prompt
from src.schemas import SLOTS, UNKNOWN, PolicyAction


def synth(did, turns, style="singlish"):
    """Rows in the synthetic [TRACK] format; each turn is (user text, cumulative gold slots, feedback)."""
    rows, hist = [], []
    for i, (text, gold, feedback) in enumerate(turns):
        hist.append(f"User: {text}\nNormalized: {text}")
        rows.append({"id": f"{did}_{2 * i + 1}", "style": style, "input": "[TRACK]\n" + "\n".join(hist),
                     "output": json.dumps({**{s: UNKNOWN for s in SLOTS}, **gold, "feedback": feedback})})
    return rows


def run(rows, seed=0):
    [dialogue] = dialogues(rows)
    return replay(dialogue, catalogue_vocab(), random.Random(seed))


def test_dialogue_follows_the_real_policy():
    rows = run(synth("SYN1", [
        ("must got aircon sia", {"setting": "aircon"}, "none"),
        ("korean lor", {"setting": "aircon", "cuisine": "Korean"}, "none"),
        ("Downtown Core side lah", {"setting": "aircon", "cuisine": "Korean", "area": "Downtown Core"}, "none"),
        ("too far lah", {"setting": "aircon", "cuisine": "Korean", "area": "Downtown Core"}, "too_far"),
        ("ok can", {"setting": "aircon", "cuisine": "Korean", "area": "Downtown Core"}, "none"),
    ]))
    ask_cuisine, ask_area, rec, after = rows  # "ok can" is skipped
    assert after["user_text"] == "too far lah"
    assert (ask_cuisine["action"]["kind"], ask_cuisine["action"]["ask_slot"]) == ("ask", "cuisine")
    assert (ask_area["action"]["kind"], ask_area["action"]["ask_slot"]) == ("ask", "area")
    assert rec["state"]["slots"]["setting"] == "aircon"
    area = rec["state"]["slots"]["area"]
    assert area in {"bugis", "tampines"}  # an area that serves korean replaces "Downtown Core"
    assert rec["user_text"] == f"{area.title()} side lah"
    assert rec["action"]["kind"] in ("recommend", "fallback_recommend") and rec["action"]["places"]
    assert after["state"]["radius_km"] == rec["state"]["radius_km"] / 2
    assert all(r["task"] == "[REPLY]" and r["style"] == "singlish" for r in rows)
    for r in rows:
        PolicyAction.from_dict(r["action"])


def test_region_is_swapped_in_english_text():
    [row] = run(synth("SYN2", [("Somewhere in the east is fine.", {"area": "east"}, "none")], style="english"))
    area = row["state"]["slots"]["area"]
    assert row["user_text"] == f"Somewhere in {area.title()} is fine."
    assert row["action"]["ask_slot"] == "cuisine"


def test_dish_keeps_its_text_and_maps_to_a_cuisine():
    [row] = run(synth("SYN3", [("want eat chicken rice", {"cuisine": "chicken rice"}, "none")]))
    assert row["user_text"] == "want eat chicken rice"
    assert row["state"]["slots"]["cuisine"] == "chinese"


def test_swapped_value_never_repeats_a_kept_one():
    for seed in range(20):
        first, revised = run(synth("SYN8", [
            ("I would like Vietnamese.", {"cuisine": "Vietnamese"}, "none"),
            ("Actually, I want Korean, not Vietnamese.", {"cuisine": "Korean"}, "none"),
        ], style="english"), seed=seed)
        assert first["state"]["slots"]["cuisine"] != "korean"
        assert revised["state"]["slots"]["cuisine"] == "korean" and "not Korean" not in revised["user_text"]


def test_revision_to_the_same_value_is_skipped():
    rows = run(synth("SYN9", [
        ("malay lah", {"cuisine": "Malay"}, "none"),
        ("actually nasi lemak not malay lah", {"cuisine": "nasi lemak"}, "none"),
    ]))
    assert [r["user_text"] for r in rows] == ["malay lah"]


def test_not_interested_excludes_the_top_pick():
    first, second = run(synth("SYN4", [
        ("korean lah, bugis side", {"cuisine": "Korean", "area": "Bugis"}, "none"),
        ("dun want this one", {"cuisine": "Korean", "area": "Bugis"}, "not_interested"),
    ]))
    top = first["action"]["places"][0]["place_id"]
    assert second["state"]["excluded_place_ids"] == [top]
    assert top not in [p["place_id"] for p in second["action"]["places"]]


def test_budget_is_mapped_to_a_cap():
    [row] = run(synth("SYN5", [("under $5 can?", {"budget": "below $5"}, "none")]))
    assert row["state"]["slots"]["budget"] == "<=5" and row["state"]["budget_max"] == 5


def test_training_row_uses_the_serving_prompt():
    [draft] = run(synth("SYN7", [("korean lah, bugis side", {"cuisine": "Korean", "area": "Bugis"}, "none")]))
    row = training_row({**draft, "reply": "Ok try these lah"})
    action = PolicyAction.from_dict(draft["action"])
    assert row["input"] == build_prompt(action, "singlish", "korean lah, bugis side")
    assert (row["task"], row["output"], row["style"]) == ("[REPLY]", "Ok try these lah", "singlish")


def test_unfilterable_dietary_drops_the_dialogue():
    assert run(synth("SYN6", [("cannot eat beef", {"dietary": "no beef"}, "none")])) == []


def test_synthetic_dev_split_covers_every_action_kind_in_both_styles():
    path = ROOT / "data" / "synthetic_data" / "dialogues_synth_dev.jsonl"
    ds = dialogues(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    rows = build_drafts(ds)
    kinds = {(r["action"]["kind"], r["style"]) for r in rows}
    assert kinds == {(k, s) for k in ("ask", "recommend", "fallback_recommend") for s in ("singlish", "english")}
    keys = [json.dumps([r["style"], r["user_text"], r["action"]], sort_keys=True) for r in rows]
    assert len(keys) == len(set(keys))
    assert len(build_drafts(ds[:30], max_per_bucket=1)) < len(build_drafts(ds[:30]))
