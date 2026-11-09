"""HTTP API for the chat frontend.  Run from backend/:  uvicorn src.api:app --reload

    GET  /api/health                      which model each component runs
    POST /api/sessions                    -> {session_id}
    POST /api/sessions/{id}/messages      {message} -> {reply, debug, places, search}
    POST /api/sessions/{id}/messages/stream   same, as server-sent events: one "stage" event per
                                          pipeline stage as it finishes, then "done" with the above
    POST /api/sessions/{id}/reset         clear the dialogue

Sessions live in memory (lost on restart). If frontend/dist exists it is served at /,
so one process can host the whole app.
"""

from __future__ import annotations

import json
import math
import queue
from collections import OrderedDict
from threading import Lock, Thread
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import config, pipeline
from src.recommender import db
from src.recommender.filters import ref_point
from src.schemas import DialogueState

MAX_SESSIONS = 500

app = FastAPI(title="Makan Where?")
if config.CORS_ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])

_sessions: OrderedDict[str, pipeline.MakanSession] = OrderedDict()
_lock = Lock()


class Message(BaseModel):
    message: str = Field(min_length=1, max_length=1000)


def _get(session_id: str) -> pipeline.MakanSession:
    with _lock:
        session = _sessions.get(session_id)
        if session is None:
            raise HTTPException(404, "unknown session")
        _sessions.move_to_end(session_id)
        return session


# catalogue columns the frontend's place cards and map need
PLACE_FIELDS = ("lat", "lon", "cuisine", "area", "address", "price_min", "price_max", "price_source",
                "rating", "n_reviews", "halal", "vegetarian", "aircon")


def _plain(value):
    """numpy / pandas scalars -> JSON values; NaN -> None."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def search_area(debug: dict, user_latlon=None) -> dict:
    """Where this turn searched: the centre the radius is measured from (None for a whole region)."""
    state = DialogueState.from_dict(debug["state"])
    centre = ref_point(state, user_latlon)
    return {"center": list(centre) if centre else None, "radius_km": state.radius_km,
            "label": state.landmark or (state.slots["area"] if state.filled("area") else None)}


def place_details(places: list[dict], center: Optional[list]) -> list[dict]:
    """The picked places in rank order, with their catalogue columns and distance from the centre."""
    if not places:
        return []
    df = db.load_places()
    rows = df.loc[df["place_id"].isin([p["place_id"] for p in places])].set_index("place_id")
    out = []
    for p in places:
        info = {**p}
        if p["place_id"] in rows.index:
            row = rows.loc[p["place_id"]]
            info.update({f: _plain(row.get(f)) for f in PLACE_FIELDS})
            if center and info["lat"] is not None and info["lon"] is not None:
                info["distance_km"] = round(db.haversine_km(center[0], center[1], info["lat"], info["lon"]), 2)
        out.append(info)
    return out


@app.get("/api/health")
def health():
    return {"status": "ok", "components": config.summary()}


@app.post("/api/sessions")
def create_session():
    session = pipeline.MakanSession(log_dir=pipeline.LOG_DIR)
    with _lock:
        _sessions[session.session_id] = session
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)
    return {"session_id": session.session_id}


@app.post("/api/sessions/{session_id}/messages")
def send_message(session_id: str, body: Message):
    if not body.message.strip():
        raise HTTPException(422, "message is empty")
    session = _get(session_id)
    reply, debug = session.step(body.message.strip())
    return _turn_response(session, reply, debug)


def _turn_response(session: pipeline.MakanSession, reply: str, debug: dict) -> dict:
    search = search_area(debug, session.user_latlon)
    return {"reply": reply, "debug": debug, "search": search,
            "places": place_details(debug["action"]["places"], search["center"])}


@app.post("/api/sessions/{session_id}/messages/stream")
def stream_message(session_id: str, body: Message):
    """The same turn as send_message, streamed so the frontend can show each stage live."""
    if not body.message.strip():
        raise HTTPException(422, "message is empty")
    session = _get(session_id)
    events: queue.Queue = queue.Queue()

    def run() -> None:
        try:
            reply, debug = session.step(body.message.strip(), on_stage=lambda stage, ms, output: events.put(
                {"type": "stage", "stage": stage, "ms": ms, **output}))
            events.put({"type": "done", **_turn_response(session, reply, debug)})
        except Exception as exc:  # noqa: BLE001 - report it to the client instead of hanging the stream
            events.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            events.put(None)

    Thread(target=run, daemon=True).start()

    def sse():
        yield f"data: {json.dumps({'type': 'start'})}\n\n"
        while (event := events.get()) is not None:
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/sessions/{session_id}/reset")
def reset_session(session_id: str):
    _get(session_id).reset()
    return {"ok": True}


if config.FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=config.FRONTEND_DIST, html=True), name="frontend")
