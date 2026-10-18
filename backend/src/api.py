"""HTTP API for the chat frontend.  Run from backend/:  uvicorn src.api:app --reload

    GET  /api/health                      which implementation each component runs
    POST /api/sessions                    -> {session_id}
    POST /api/sessions/{id}/messages      {message} -> {reply, debug}
    POST /api/sessions/{id}/reset         clear the dialogue

Sessions live in memory (lost on restart). If frontend/dist exists it is served at /,
so one process can host the whole app.
"""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import config, pipeline

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
    reply, debug = _get(session_id).step(body.message.strip())
    return {"reply": reply, "debug": debug}


@app.post("/api/sessions/{session_id}/reset")
def reset_session(session_id: str):
    _get(session_id).reset()
    return {"ok": True}


if config.FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=config.FRONTEND_DIST, html=True), name="frontend")
