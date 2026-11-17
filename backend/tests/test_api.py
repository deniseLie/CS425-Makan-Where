import pytest
from fastapi.testclient import TestClient

from src import config
from src.api import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.LOG_DIR", tmp_path)
    return TestClient(app)


def test_health_reports_the_models(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert all(body["components"][c]["model"] == config.model_id(c) for c in config.COMPONENTS)
    assert body["components"]["base_llm"] == config.BASE_LLM


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


def test_no_model_id_fails_clearly(monkeypatch):
    monkeypatch.delenv("MAKAN_STYLE_MODEL", raising=False)
    monkeypatch.delitem(config.DEFAULT_MODELS, "style")
    with pytest.raises(RuntimeError, match="MAKAN_STYLE_MODEL"):
        config.model_id("style")


def test_tracker_and_generator_have_their_own_adapters(monkeypatch):
    monkeypatch.delenv("MAKAN_TRACKER_ADAPTER", raising=False)
    monkeypatch.delenv("MAKAN_GENERATOR_ADAPTER", raising=False)
    assert config.model_id("tracker") != config.model_id("generator")
    monkeypatch.setenv("MAKAN_TRACKER_ADAPTER", "team/track")
    monkeypatch.setenv("MAKAN_GENERATOR_ADAPTER", "team/reply")
    assert (config.model_id("tracker"), config.model_id("generator")) == ("team/track", "team/reply")


def test_normalizer_defaults_to_the_team_model(monkeypatch):
    monkeypatch.delenv("MAKAN_NORMALIZER_MODEL", raising=False)
    assert config.model_id("normalizer") == "makan-where-cs425/makan-flan-normalizer-masked"


def test_style_defaults_to_the_team_model(monkeypatch):
    monkeypatch.delenv("MAKAN_STYLE_MODEL", raising=False)
    assert config.model_id("style") == "makan-where-cs425/makan-singbert-style-classifier"
    monkeypatch.setenv("MAKAN_STYLE_MODEL", "team/style")
    assert config.model_id("style") == "team/style"


def test_recommendations_carry_place_details(client):
    sid = client.post("/api/sessions").json()["session_id"]
    client.post(f"/api/sessions/{sid}/messages", json={"message": "I want eat something leh"})
    r = client.post(f"/api/sessions/{sid}/messages", json={"message": "korean lah, bugis side"}).json()
    assert r["search"]["center"] and r["search"]["radius_km"] == r["debug"]["state"]["radius_km"]
    assert [p["place_id"] for p in r["places"]] == [p["place_id"] for p in r["debug"]["action"]["places"]]
    for p in r["places"]:
        assert isinstance(p["lat"], float) and isinstance(p["lon"], float)
        assert 0 <= p["distance_km"] <= r["search"]["radius_km"] + 0.5
        assert isinstance(p["halal"], bool)
    # asking a question returns no places, but still says where it is searching
    r = client.post(f"/api/sessions/{sid}/messages", json={"message": "hello"}).json()
    assert "search" in r and isinstance(r["places"], list)


def _events(response) -> list[dict]:
    import json
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def test_stream_reports_each_stage_then_the_reply(client):
    sid = client.post("/api/sessions").json()["session_id"]
    r = client.post(f"/api/sessions/{sid}/messages/stream", json={"message": "korean lah, bugis side"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = _events(r)
    assert events[0] == {"type": "start"} and events[-1]["type"] == "done"
    stages = [e for e in events if e["type"] == "stage"]
    assert [e["stage"] for e in stages] == ["normalize", "style", "track", "policy", "generate"]
    assert "normalized" in stages[0] and stages[1]["style"] in ("singlish", "english")
    assert stages[2]["state"]["slots"]["area"] == "downtown core" and "kind" in stages[3]["action"]
    done = events[-1]
    assert done["reply"] and done["debug"]["turn_idx"] == 0 and "places" in done and "search" in done


def test_stream_reports_errors_instead_of_hanging(client, monkeypatch):
    def boom(self, text, on_stage=None):
        raise RuntimeError("model fell over")
    monkeypatch.setattr("src.pipeline.MakanSession.step", boom)
    sid = client.post("/api/sessions").json()["session_id"]
    events = _events(client.post(f"/api/sessions/{sid}/messages/stream", json={"message": "hi"}))
    assert events[-1] == {"type": "error", "message": "RuntimeError: model fell over"}
    assert client.post("/api/sessions/nope/messages/stream", json={"message": "hi"}).status_code == 404
