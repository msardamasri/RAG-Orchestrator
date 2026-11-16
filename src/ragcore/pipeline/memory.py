"""Where conversation state lives between messages."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict

from ragcore.types import ConversationState, Turn


def state_to_json(state: ConversationState) -> str:
    return json.dumps(asdict(state), ensure_ascii=False)


def state_from_json(raw: str) -> ConversationState:
    data = json.loads(raw)
    data["turns"] = [Turn(**t) for t in data.get("turns", [])]
    return ConversationState(**data)


class InMemoryStore:
    def __init__(self):
        self.sessions: dict[str, ConversationState] = {}

    def load(self, session_id: str) -> ConversationState:
        return self.sessions.setdefault(session_id, ConversationState())

    def save(self, session_id: str, state: ConversationState) -> None:
        self.sessions[session_id] = state


class SQLiteMemoryStore:
    """Sessions as JSON rows in a SQLite file, so a conversation survives a restart."""

    def __init__(self, path: str = "ragcore-memory.db"):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        with self.lock, self.db:
            self.db.execute("CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, state TEXT NOT NULL, "
                            "updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")

    def load(self, session_id: str) -> ConversationState:
        with self.lock:
            row = self.db.execute("SELECT state FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return state_from_json(row[0]) if row else ConversationState()

    def save(self, session_id: str, state: ConversationState) -> None:
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO sessions(id, state) VALUES(?, ?) ON CONFLICT(id) DO UPDATE SET "
                "state = excluded.state, updated_at = CURRENT_TIMESTAMP", (session_id, state_to_json(state)))
