import json
import random

import pytest

from src.generator import teacher
from src.generator.checks import check_reply
from src.generator.teacher import EXAMPLES, SYSTEM, describe, teacher_messages
from src.generator.templates import template_reply
from src.schemas import PolicyAction, RankedPlace

CATALOGUE = ["Bugis Kimchi House", "Hanok BBQ Bugis", "Orchard Sky Grill", "Bugis Ramen", "Bugis Ramen Ya"]
PLACES = [
    RankedPlace("P001", "Bugis Kimchi House", 4.3, ["~$18", "300 m away", "4.3 stars"]),
    RankedPlace("P002", "Hanok BBQ Bugis", 4.5, ["halal", "~$35", "1.2 km away", "4.5 stars"]),
]
REC = PolicyAction(kind="recommend", places=PLACES)
GOOD = "Try Bugis Kimchi House (~$18, 300 m away) or Hanok BBQ Bugis (halal, ~$35, 1.2 km away)."

ACTIONS = [
    PolicyAction(kind="ask", ask_slot="cuisine", ask_options=["korean", "japanese", "malay"]),
    PolicyAction(kind="ask", ask_slot="dietary", ask_options=["halal", "vegetarian", "none"]),
    PolicyAction(kind="ask", ask_slot="area"),
    REC,
    PolicyAction(kind="fallback_recommend", places=PLACES, relaxed_slot="setting"),
    PolicyAction(kind="fallback_recommend", places=PLACES, relaxed_slot="budget"),
    PolicyAction(kind="fallback_recommend", places=[], relaxed_slot="budget"),
]


def codes(reply, action, style="english", user_text=""):
    return [code for code, _ in check_reply(reply, action, style, user_text, catalogue=CATALOGUE)]


@pytest.mark.parametrize("style", ["singlish", "english"])
@pytest.mark.parametrize("action", ACTIONS, ids=lambda a: f"{a.kind}-{a.ask_slot or a.relaxed_slot}")
def test_template_replies_pass(action, style):
    # the guardrail falls back to these, so they must pass the same filter as teacher drafts
    assert check_reply(template_reply(action, style), action, style, catalogue=CATALOGUE) == []


def test_recommend_failures_are_caught():
    assert codes(GOOD, REC) == []
    assert codes(GOOD + " Orchard Sky Grill also nice.", REC) == ["rogue_place"]
    assert codes("Try Bugis Kimchi House, 300 m away.", REC) == ["missing_place"]
    assert codes(GOOD.replace("$18", "$12"), REC) == ["invented_fact"]
    assert codes(GOOD + " Open till 11pm, 5 min walk.", REC) == ["invented_fact", "invented_fact"]
    assert codes(GOOD + " Shiok lah!", REC) == ["style_mismatch"]
    assert codes(GOOD.replace("Try", "P001: try"), REC) == ["leak"]
    assert codes("**" + GOOD + "**", REC) == ["leak"]
    assert codes("", REC) == ["length"]
    assert codes(GOOD + " ok" * 150, REC) == ["length"]


def test_numbers_from_the_user_may_be_repeated():
    assert codes(GOOD + " Both under your $40 budget.", REC, user_text="below $40 please") == []


def test_a_picked_name_does_not_count_as_a_shorter_catalogue_name():
    action = PolicyAction(kind="recommend", places=[RankedPlace("P9", "Bugis Ramen Ya", 4.0, ["~$15"])])
    assert codes("Bugis Ramen Ya, about $15.", action) == []


def test_ask_failures_are_caught():
    ask = PolicyAction(kind="ask", ask_slot="dietary", ask_options=["halal", "vegetarian", "none"])
    assert codes("Any dietary needs, like halal or vegetarian?", ask) == []  # "none" need not be said
    assert codes("Any dietary needs, like halal or vegetarian.", ask) == ["no_question"]
    assert codes("Must it be halal?", ask) == ["missing_option"]
    assert codes("Halal or vegetarian? Bugis Kimchi House is nice.", ask) == ["rogue_place"]
    setting = PolicyAction(kind="ask", ask_slot="setting", ask_options=["foodcourt", "cafe"])
    assert codes("Food court or café?", setting) == []


def test_fallback_must_say_what_was_relaxed():
    fb = PolicyAction(kind="fallback_recommend", places=PLACES, relaxed_slot="budget")
    assert codes("No exact match, so I went over your budget. " + GOOD, fb) == []
    assert codes(GOOD, fb) == ["unexplained_fallback"]  # prices alone don't explain it
    cafe = PolicyAction(kind="fallback_recommend", relaxed_slot="setting",
                        places=[RankedPlace("P015", "Tampines Green Leaf Cafe", 4.2, ["~$16"])])
    assert codes("Try Tampines Green Leaf Cafe, about $16.", cafe) == ["unexplained_fallback"]
    assert codes("I looked at other kinds of places: Tampines Green Leaf Cafe, about $16.", cafe) == []
    assert codes("No food court matched, so try Tampines Green Leaf Cafe, about $16.", cafe) == []


def test_teacher_examples_pass_the_checks():
    names = sorted({p.name for ex in EXAMPLES for p in ex.action.places})
    for ex in EXAMPLES:
        assert check_reply(ex.reply, ex.action, ex.style, ex.user_text, catalogue=names) == [], ex.reply


def test_teacher_examples_cover_every_action_kind_in_both_styles():
    seen = {(ex.action.kind, ex.style) for ex in EXAMPLES}
    assert seen == {(k, s) for k in ("ask", "recommend", "fallback_recommend") for s in ("singlish", "english")}


def test_teacher_messages():
    system, messages = teacher_messages(REC, "singlish", "korean lah", rng=random.Random(0))
    assert system == SYSTEM
    assert [m["role"] for m in messages] == ["user", "assistant"] * 3 + ["user"]
    last = messages[-1]["content"]
    assert last == describe(REC, "singlish", "korean lah")
    assert all(p.name in last and all(r in last for r in p.reasons) for p in PLACES)
    assert {m["content"] for m in messages[1:-1:2]} <= {ex.reply for ex in EXAMPLES if ex.style == "singlish"}
    assert "Action: recommend" in messages[0]["content"]  # same action kind first


def test_teacher_job_round_trip(tmp_path):
    drafts = tmp_path / "drafts_dev.jsonl"
    rows = [{"id": f"D{i}", "style": "english", "user_text": "hi", "action": REC.to_dict()} for i in range(3)]
    drafts.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    job = tmp_path / "job"
    [chunk] = teacher.prepare(job, [drafts], chunks=1)
    assert [json.loads(line)["prompt"] for line in chunk.read_text().splitlines()] == [describe(REC, "english", "hi")] * 3

    (job / "replies_00_1.jsonl").write_text(json.dumps({"id": "D0", "reply": GOOD}) + "\n"
                                            + json.dumps({"id": "D1", "reply": "Try Orchard Sky Grill."}) + "\n")
    missing, failing = teacher.check(job, "00", catalogue=CATALOGUE)
    assert missing == ["D2"] and list(failing) == ["D1"]

    (job / "replies_00_fix1.jsonl").write_text(json.dumps({"id": "D1", "reply": GOOD}) + "\n")
    [out] = teacher.merge(job)
    assert out.name == "replies_dev.jsonl"
    assert [(r["id"], r["reply"]) for r in map(json.loads, out.read_text().splitlines())] == [("D0", GOOD), ("D1", GOOD)]


def test_teacher_examples_rotate_with_rng():
    picks = {tuple(m["content"] for m in teacher_messages(REC, "english", "hi", rng=random.Random(s))[1])
             for s in range(10)}
    assert len(picks) > 1
