import pytest
from fastapi.testclient import TestClient

from src import config
from src.api import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.LOG_DIR", tmp_path)
    return TestClient(app)


def test_health_reports_stubs(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert all(body["components"][c]["impl"] == "stub" for c in config.COMPONENTS)


def test_conversation_and_reset(client):
    sid = client.post("/api/sessions").json()["session_id"]
    r = client.post(f"/api/sessions/{sid}/messages", json={"message": "I want eat something leh"}).json()
    assert r["debug"]["action"]["kind"] == "ask" and r["reply"]
    r = client.post(f"/api/sessions/{sid}/messages", json={"message": "korean lah, bugis side"}).json()
    assert r["debug"]["action"]["kind"] == "recommend"
    assert client.post(f"/api/sessions/{sid}/reset").json() == {"ok": True}
    r = client.post(f"/api/sessions/{sid}/messages", json={"message": "hello"}).json()
    assert r["debug"]["turn_idx"] == 0


def test_bad_requests(client):
    assert client.post("/api/sessions/nope/messages", json={"message": "hi"}).status_code == 404
    sid = client.post("/api/sessions").json()["session_id"]
    assert client.post(f"/api/sessions/{sid}/messages", json={"message": "   "}).status_code == 422
    assert client.post(f"/api/sessions/{sid}/messages", json={"message": "x" * 1001}).status_code == 422


def test_hf_without_model_id_fails_clearly(monkeypatch):
    monkeypatch.setenv("MAKAN_NORMALIZER", "hf")
    monkeypatch.delenv("MAKAN_NORMALIZER_MODEL", raising=False)
    assert config.impl("normalizer") == "hf"
    with pytest.raises(RuntimeError, match="MAKAN_NORMALIZER_MODEL"):
        config.model_id("normalizer")
    monkeypatch.setenv("MAKAN_STYLE", "bert")
    with pytest.raises(ValueError):
        config.impl("style")


def test_tracker_hf_parsing_without_model():
    from src.schemas import DialogueState, Turn
    from src.tracker.hf import build_prompt, parse

    prev = DialogueState(slots={"cuisine": "korean"})
    out = parse('Sure: {"slots": {"area": "bugis"}, "reject": "Hanok BBQ Bugis", "critique": "too_far"}', prev)
    assert out.slots["cuisine"] == "korean" and out.slots["area"] == "bugis"
    assert out.reject == "P002" and out.critique == "too_far"
    assert parse("not json", prev).slots == prev.slots
    assert build_prompt([Turn(raw="korean lah", reply="ok")]).startswith("[TRACK]")
