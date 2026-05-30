import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.utils.path_handler import get_absolute_path
from agent.utils.logger_handler import get_logger

logger = get_logger()

DB_PATH = str(get_absolute_path("data/chat.db"))


class ConversationStore:
    def __init__(self, db_path: str = DB_PATH):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conv_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                thinking TEXT,
                seq INTEGER NOT NULL,
                FOREIGN KEY (conv_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conv_id, seq);
        """)
        # Migrate: add summary column if missing
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(conversations)").fetchall()}
        if "summary" not in cols:
            self.conn.execute("ALTER TABLE conversations ADD COLUMN summary TEXT")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.commit()

    # --- Conversations ---
    def list_conversations(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_conversation(self, conv_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
        if not row:
            return None
        conv = dict(row)
        conv["messages"] = self.get_messages(conv_id)
        return conv

    def create_conversation(self, conv_id: str, title: str) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        self.conn.execute(
            "INSERT OR IGNORE INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conv_id, title, now, now),
        )
        self.conn.commit()

    def update_title(self, conv_id: str, title: str) -> None:
        self.conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, datetime.now().isoformat(timespec="seconds"), conv_id),
        )
        self.conn.commit()

    def touch_conversation(self, conv_id: str) -> None:
        self.conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (datetime.now().isoformat(timespec="seconds"), conv_id),
        )
        self.conn.commit()

    def delete_conversation(self, conv_id: str) -> bool:
        cur = self.conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        self.conn.execute("DELETE FROM messages WHERE conv_id = ?", (conv_id,))
        self.conn.commit()
        return cur.rowcount > 0

    # --- Summary ---
    def update_summary(self, conv_id: str, summary: str) -> None:
        self.conn.execute(
            "UPDATE conversations SET summary = ?, updated_at = ? WHERE id = ?",
            (summary, datetime.now().isoformat(timespec="seconds"), conv_id),
        )
        self.conn.commit()

    def get_summary(self, conv_id: str) -> str | None:
        row = self.conn.execute(
            "SELECT summary FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
        return row["summary"] if row and row["summary"] else None

    # --- Messages ---
    def add_message(self, conv_id: str, role: str, content: str, thinking: list | None = None) -> int:
        max_seq = self.conn.execute(
            "SELECT COALESCE(MAX(seq), -1) FROM messages WHERE conv_id = ?", (conv_id,)
        ).fetchone()[0]
        seq = max_seq + 1
        thinking_json = json.dumps(thinking, ensure_ascii=False) if thinking else None
        cur = self.conn.execute(
            "INSERT INTO messages (conv_id, role, content, thinking, seq) VALUES (?, ?, ?, ?, ?)",
            (conv_id, role, content, thinking_json, seq),
        )
        self.touch_conversation(conv_id)
        self.conn.commit()
        return cur.lastrowid

    def update_message(self, msg_id: int, content: str, thinking: list | None = None) -> None:
        thinking_json = json.dumps(thinking, ensure_ascii=False) if thinking else None
        self.conn.execute(
            "UPDATE messages SET content = ?, thinking = ? WHERE id = ?",
            (content, thinking_json, msg_id),
        )
        self.conn.commit()

    def get_messages(self, conv_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT role, content, thinking, seq FROM messages WHERE conv_id = ? ORDER BY seq",
            (conv_id,),
        ).fetchall()
        result = []
        for r in rows:
            msg = {"role": r["role"], "content": r["content"], "seq": r["seq"]}
            if r["thinking"]:
                try:
                    msg["thinking"] = json.loads(r["thinking"])
                except json.JSONDecodeError:
                    msg["thinking"] = []
            else:
                msg["thinking"] = []
            result.append(msg)
        return result

    def delete_last_messages(self, conv_id: str, count: int = 2) -> int:
        """Delete the last N messages from a conversation. Returns number deleted."""
        rows = self.conn.execute(
            "SELECT id FROM messages WHERE conv_id = ? ORDER BY seq DESC LIMIT ?",
            (conv_id, count),
        ).fetchall()
        if not rows:
            return 0
        ids = [r["id"] for r in rows]
        placeholders = ",".join("?" * len(ids))
        cur = self.conn.execute(f"DELETE FROM messages WHERE id IN ({placeholders})", ids)
        self.touch_conversation(conv_id)
        self.conn.commit()
        return cur.rowcount


# Singleton
_store: ConversationStore | None = None


def get_store() -> ConversationStore:
    global _store
    if _store is None:
        _store = ConversationStore()
    return _store
