import json

import pytest

import src.pipeline as pipeline
import src.tracker.hf as hf
from src.pipeline import MakanSession
from src.schemas import ANY, UNKNOWN, DialogueState, PolicyAction, RankedPlace


@pytest.fixture
def session(tmp_path):
    return MakanSession(log_dir=tmp_path, session_id="test")


def test_three_turn_conversation(session):
    r1, d1 = session.step("I want eat something leh")
    assert d1["style"] == "singlish"
    assert d1["action"]["kind"] == "ask" and d1["action"]["ask_slot"] == "cuisine"
    assert d1["state"]["slots"]["cuisine"] == UNKNOWN

    r2, d2 = session.step("korean lah, bugis side")
    assert d2["state"]["slots"]["cuisine"] == "korean"
    assert d2["state"]["slots"]["area"] == "downtown core"
    assert d2["action"]["kind"] == "recommend"
    first_ids = [p["place_id"] for p in d2["action"]["places"]]
    for p in d2["action"]["places"]:
        assert p["name"] in r2

    r3, d3 = session.step("too far leh")
    assert d3["tracker_output"]["critique"] == "too_far"
    assert d3["state"]["radius_km"] == 1.0
    assert d3["state"]["slots"]["cuisine"] == "korean"  # carried over
    third_ids = [p["place_id"] for p in d3["action"]["places"]]
    assert third_ids
    # the 1.2 km place drops out of the search; the hybrid ranker already preferred the
    # nearer ones, so the top 3 can stay the same
    from src import recommender
    before = {c["place_id"] for c in recommender.candidates(DialogueState.from_dict(d2["state"]))}
    after = {c["place_id"] for c in recommender.candidates(DialogueState.from_dict(d3["state"]))}
    assert "P004" in before and "P004" not in after and "P004" not in third_ids
    assert set(third_ids) <= after

    lines = session.log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    rec = json.loads(lines[-1])
    for key in ["raw", "normalized", "style", "tracker_output", "state", "action", "reply"]:
        assert key in rec


def test_guardrail_replaces_hallucinated_place(session, monkeypatch):
    monkeypatch.setattr(pipeline, "generate", lambda action, style, text: "Go Orchard Sky Grill lah!")
    session.step("korean food in bugis please")
    reply, debug = session.step("ok")
    assert debug["guardrail_triggered"] is True
    assert "Orchard Sky Grill" not in reply


def test_guardrail_logs_the_replaced_reply(session, monkeypatch):
    monkeypatch.setattr(pipeline, "generate", lambda action, style, text: "Go Orchard Sky Grill lah!")
    _, debug = session.step("korean food in bugis please")
    assert debug["replaced_reply"] == "Go Orchard Sky Grill lah!"
    rec = json.loads(session.log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["replaced_reply"] == "Go Orchard Sky Grill lah!"


def test_guardrail_ignores_generic_place_names_and_picked_names(monkeypatch):
    # places.csv has places named just "Korean", "Cafe", "Chicken Rice", "Dim Sum"...
    catalogue = ["Korean", "Cafe", "Chicken Rice", "Dim Sum", "Ha-Jun Korean", "Orchard Sky Grill"]
    monkeypatch.setattr(pipeline.recommender, "place_names", lambda: catalogue)
    ask = PolicyAction(kind="ask", ask_slot="setting", ask_options=["hawker", "cafe"])
    for reply in ["Want Korean or not? Cafe also can.", "Got chicken rice and dim sum lah."]:
        assert pipeline.guardrail(reply, ask, "singlish") == (reply, False)

    rec = PolicyAction(kind="recommend", places=[RankedPlace(place_id="P1", name="Ha-Jun Korean", score=1.0)])
    reply = "Try Ha-Jun Korean lah, the korean food there shiok."
    assert pipeline.guardrail(reply, rec, "singlish") == (reply, False)
    _, fired = pipeline.guardrail("Try Ha-Jun Korean or Orchard Sky Grill.", rec, "singlish")
    assert fired


def test_not_interested_excludes_last_top_pick(session):
    _, d1 = session.step("korean lah, bugis side")
    top = d1["action"]["places"][0]["place_id"]

    _, d2 = session.step("dun want this one")
    assert d2["tracker_output"]["critique"] == "not_interested"
    assert d2["tracker_output"]["reject"] == top
    assert d2["state"]["excluded_place_ids"] == [top]
    ids = [p["place_id"] for p in d2["action"]["places"]]
    assert ids and top not in ids


def test_not_interested_without_a_recommendation_does_nothing(session):
    _, d = session.step("dun want this one")
    assert d["tracker_output"]["reject"] is None and d["state"]["excluded_place_ids"] == []


def test_trained_tracker_answers_change_the_state(session, monkeypatch):
    """Answers in the [TRACK] dataset layout (as a trained tracker writes them) drive the pipeline."""
    def answer(feedback="none", **slots):
        return json.dumps({"cuisine": "Korean", "area": "Bugis", "budget": "UNKNOWN", "dietary": "UNKNOWN",
                           "setting": "UNKNOWN", "time": "UNKNOWN", **slots, "feedback": feedback})

    answers = iter([answer(), answer("too_far"), answer("not_interested"), answer(budget="below $10")])
    prompts = []

    def fake_track(history, prev):
        prompts.append(hf.build_prompt(history))
        return hf.parse(next(answers), prev)

    monkeypatch.setattr(pipeline, "track", fake_track)

    _, d1 = session.step("korean lah, bugis side")
    assert d1["state"]["slots"]["cuisine"] == "korean" and d1["state"]["slots"]["area"] == "downtown core"
    assert d1["action"]["kind"] == "recommend"
    ids1 = [p["place_id"] for p in d1["action"]["places"]]

    _, d2 = session.step("too far lah")
    ids2 = [p["place_id"] for p in d2["action"]["places"]]
    assert d2["state"]["radius_km"] == 1.0 and ids2 and ids2 != ids1

    _, d3 = session.step("dun want this one")
    ids3 = [p["place_id"] for p in d3["action"]["places"]]
    assert d3["state"]["excluded_place_ids"] == [ids2[0]] and ids3 and ids2[0] not in ids3

    _, d4 = session.step("cheap cheap pls")
    assert d4["state"]["budget_max"] is None and d4["action"]["ask_slot"] == "budget"

    assert prompts[1].startswith("[TRACK]\nUser: korean lah, bugis side\nNormalized: korean lah, bugis side\nSystem: ")
    assert prompts[-1].endswith("\nUser: cheap cheap pls\nNormalized: cheap cheap pls")


def test_reset(session):
    session.step("korean lah, bugis side")
    session.reset()
    assert session.history == [] and session.n_asked == 0 and session.state == DialogueState()
    assert session.last_action is None


def test_named_spot_is_kept_across_turns():
    from src.pipeline import MakanSession

    s = MakanSession(log_dir=None)
    _, d1 = s.step("korean near bugis")
    assert d1["state"]["slots"]["area"] == "downtown core" and d1["state"]["landmark"] == "bugis"
    _, d2 = s.step("too far leh")
    assert d2["state"]["landmark"] == "bugis" and d2["state"]["radius_km"] == 1.0
    _, d3 = s.step("tampines then")
    assert d3["state"]["slots"]["area"] == "tampines" and d3["state"]["landmark"] is None


def test_exact_budget_wins_over_tracker_and_is_kept_until_updated(session, monkeypatch):
    def fake_track(history, prev):
        budget = "$8-10" if "max" in history[-1].raw else "$10-20"
        return hf.parse(json.dumps({"cuisine": "Western", "area": "river valley", "budget": budget}), prev)

    monkeypatch.setattr(pipeline, "track", fake_track)
    for text, expected in [
        ("$15", 15), ("max $8", 8), ("restaurant", 8), ("actually twelve bucks", 12),
        ("cheap please", None), ("8.50", 8.5), ("any price is fine", None),
        ("any price under $8", 8), ("make my budget flexible", None),
        ("max $8", 8), ("no budget limit", None),
    ]:
        _, debug = session.step(text)
        assert debug["state"]["budget_max"] == expected
        if text == "cheap please":
            assert debug["action"]["ask_slot"] == "budget"
        elif expected is None:
            assert debug["state"]["slots"]["budget"] == ANY
    assert session.state.slots["budget"] == ANY


@pytest.mark.parametrize("answer", ["cheap cheap lah", "too ex leh", "anything also can, just recommend lah"])
def test_budget_is_asked_only_once(session, answer):
    _, d1 = session.step("cheap western food at jurong west")
    assert d1["action"]["ask_slot"] == "budget"
    _, d2 = session.step(answer)
    assert d2["action"]["ask_slot"] != "budget"
    assert d2["state"]["slots"]["budget"] == ANY and d2["state"]["budget_max"] is None


def test_above_budget_suggestions_require_opt_in_and_keep_cap(session):
    reply, debug = session.step("western food in river valley, max $5")
    assert not debug["action"]["places"] and "within your budget" in reply
    reply, debug = session.step("yes")
    assert debug["action"]["places"] and "over budget" in reply
    assert session.state.budget_max == 5
    _, debug = session.step("same area")
    assert not debug["action"]["places"]


@pytest.mark.parametrize("answer", ["no", "no need", "none", "non-halal is fine"])
def test_dietary_question_accepts_no_requirement(session, answer):
    session.state = DialogueState(slots={"cuisine": "korean", "area": "downtown core", "dietary": "halal"})
    session.last_action = PolicyAction(kind="ask", ask_slot="dietary")
    session.step(answer)
    assert session.state.slots["dietary"] == ANY



def test_english_turns_skip_the_normalizer(session, monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "normalize", lambda text: calls.append(text) or "REWRITTEN")
    stages = {}
    _, d1 = session.step("korean food in bugis please", on_stage=lambda s, ms, out: stages.update({s: out}))
    assert d1["style"] == "english" and d1["normalized"] == "korean food in bugis please" and not calls
    assert stages["normalize"]["skipped"] is True
    _, d2 = session.step("korean lah, bugis side", on_stage=lambda s, ms, out: stages.update({s: out}))
    assert d2["style"] == "singlish" and d2["normalized"] == "REWRITTEN" and calls == ["korean lah, bugis side"]
    assert stages["normalize"]["skipped"] is False


def test_debug_reports_stage_timings(session):
    _, debug = session.step("korean lah, bugis side")
    timings = debug["timings_ms"]
    assert list(timings) == ["style", "normalize", "track", "policy", "generate"]
    assert all(isinstance(ms, int) and ms >= 0 for ms in timings.values())


def test_debug_says_who_wrote_the_reply(session, monkeypatch):
    monkeypatch.setattr(pipeline, "generate", lambda action, style, text: "Go Orchard Sky Grill lah!")
    _, debug = session.step("korean food in bugis please")
    assert debug["reply_source"] == "guardrail"
    monkeypatch.setattr(pipeline, "generate", lambda action, style, text: "Can, here got some.")
    _, debug = session.step("ok")
    assert debug["reply_source"] == "llm"


def test_debug_keeps_the_raw_tracker_values(session):
    _, debug = session.step("korean lah, bugis side")
    # the state holds the planning area; the raw tracker output keeps what was said
    assert debug["state"]["slots"]["area"] == "downtown core"
    assert set(debug["tracker_raw"]["slots"]) == set(debug["state"]["slots"])
