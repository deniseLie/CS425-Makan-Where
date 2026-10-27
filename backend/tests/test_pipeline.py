import json

import pytest

import src.pipeline as pipeline
from src.pipeline import MakanSession
from src.schemas import UNKNOWN, DialogueState, Turn
from src.style import detect_style
from src.tracker import track


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
    assert third_ids and third_ids != first_ids
    assert "P004" in first_ids and "P004" not in third_ids  # the 1.5 km place drops out

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


def test_reset(session):
    session.step("korean lah, bugis side")
    session.reset()
    assert session.history == [] and session.n_asked == 0 and session.state == DialogueState()


@pytest.mark.parametrize("text,style", [
    ("can or not", "singlish"), ("got halal or not", "singlish"), ("eat where lor", "singlish"),
    ("I would like some Korean food in Bugis", "english"),
])
def test_style_stub(text, style):
    assert detect_style(text) == style


def test_tracker_stub():
    out = track([Turn(raw="cheap halal prata at jurong east")], DialogueState())
    assert out.slots["cuisine"] == "indian"
    assert out.slots["area"] == "jurong east"
    assert out.slots["dietary"] == "halal"
    assert out.slots["budget"] == "<=8"
    out = track([Turn(raw="anything also can")], DialogueState(slots={"cuisine": "korean"}))
    assert out.slots["area"] == "ANY"
    assert track([Turn(raw="don't want Hanok BBQ Bugis")], DialogueState()).reject == "P002"
