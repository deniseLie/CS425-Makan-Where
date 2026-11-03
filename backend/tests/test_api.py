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
    monkeypatch.setenv("MAKAN_STYLE", "hf")
    monkeypatch.delenv("MAKAN_STYLE_MODEL", raising=False)
    assert config.impl("style") == "hf"
    with pytest.raises(RuntimeError, match="MAKAN_STYLE_MODEL"):
        config.model_id("style")
    monkeypatch.setenv("MAKAN_STYLE", "bert")
    with pytest.raises(ValueError):
        config.impl("style")


def test_tracker_and_generator_have_their_own_adapters(monkeypatch):
    monkeypatch.delenv("MAKAN_TRACKER_ADAPTER", raising=False)
    monkeypatch.delenv("MAKAN_GENERATOR_ADAPTER", raising=False)
    assert config.model_id("tracker") != config.model_id("generator")
    monkeypatch.setenv("MAKAN_TRACKER_ADAPTER", "team/track")
    monkeypatch.setenv("MAKAN_GENERATOR_ADAPTER", "team/reply")
    assert (config.model_id("tracker"), config.model_id("generator")) == ("team/track", "team/reply")


def test_normalizer_defaults_to_the_team_model(monkeypatch):
    monkeypatch.delenv("MAKAN_NORMALIZER_MODEL", raising=False)
    assert config.model_id("normalizer") == "makan-where-cs425/makan-flan-normalizer"
