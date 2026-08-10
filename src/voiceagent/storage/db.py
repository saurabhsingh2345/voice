"""Encrypted conversation history and long-term memory.

DEVIATION, stated plainly: the brief specifies SQLCipher. `sqlcipher3-binary`
publishes no arm64 macOS wheel (manylinux x86_64 only), so using it means
`brew install sqlcipher` plus a source build -- another system library to bundle
for the Tauri app, which is exactly why pyaudio was rejected in Phase 4.

Instead this encrypts every message body with Fernet (AES-128-CBC + HMAC) using
a key held in the macOS Keychain, over plain SQLite. What that buys and what it
costs, honestly:

  * Same as SQLCipher: message content is unreadable without the Keychain key,
    and destroying the key makes the data unrecoverable.
  * Weaker than SQLCipher: the *schema*, row counts, and timestamps are visible
    to anyone who opens the file. SQLCipher encrypts the whole page store.
  * Consequence: content cannot be searched with SQL, so retrieval decrypts and
    scores in memory. Fine at personal-history scale; not fine at millions of rows.

Switch to SQLCipher by replacing the connection factory here if the system
dependency ever becomes acceptable.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parents[3] / "data" / "history.db"

KEYRING_SERVICE = "voiceagent.storage"
KEYRING_USER = "history-encryption-key"

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    title_enc   BLOB
);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content_enc     BLOB NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,
    content_enc BLOB NOT NULL,
    created_at  TEXT NOT NULL,
    last_used   TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
"""

#: Words too common to be evidence of relevance.
STOPWORDS = frozenset(
    """a an and are as at be but by for from had has have he her his i if in is it its
    me my of on or our she that the their them then there these they this to was we were
    what when where which who will with you your do does did can could should would""".split()
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class StoredMessage:
    id: int
    conversation_id: int
    role: str
    content: str
    created_at: str


@dataclass(frozen=True)
class Memory:
    id: int
    kind: str
    content: str
    created_at: str


class EncryptedStore:
    def __init__(self, path: Path = DEFAULT_DB) -> None:
        self.path = path
        self._conn: sqlite3.Connection | None = None
        self._fernet = None

    # --- key management ---------------------------------------------------

    def _key(self) -> bytes:
        import keyring
        from cryptography.fernet import Fernet

        existing = keyring.get_password(KEYRING_SERVICE, KEYRING_USER)
        if existing:
            return existing.encode()
        key = Fernet.generate_key()
        keyring.set_password(KEYRING_SERVICE, KEYRING_USER, key.decode())
        return key

    def _cipher(self):
        if self._fernet is None:
            from cryptography.fernet import Fernet

            self._fernet = Fernet(self._key())
        return self._fernet

    def _encrypt(self, text: str) -> bytes:
        return self._cipher().encrypt(text.encode("utf-8"))

    def _decrypt(self, blob: bytes) -> str:
        from cryptography.fernet import InvalidToken

        try:
            return self._cipher().decrypt(blob).decode("utf-8")
        except InvalidToken:
            # Key was rotated or destroyed; the row is unreadable by design.
            return "[unreadable: encrypted with a destroyed key]"

    # --- lifecycle --------------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.executescript(SCHEMA)
            self._conn.commit()
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # --- conversations ----------------------------------------------------

    def start_conversation(self, title: str | None = None) -> int:
        conn = self.connect()
        cursor = conn.execute(
            "INSERT INTO conversations (started_at, title_enc) VALUES (?, ?)",
            (_now(), self._encrypt(title) if title else None),
        )
        conn.commit()
        return cursor.lastrowid

    def add_message(self, conversation_id: int, role: str, content: str) -> int:
        conn = self.connect()
        cursor = conn.execute(
            "INSERT INTO messages (conversation_id, role, content_enc, created_at)"
            " VALUES (?, ?, ?, ?)",
            (conversation_id, role, self._encrypt(content), _now()),
        )
        conn.commit()
        return cursor.lastrowid

    def messages(self, conversation_id: int) -> list[StoredMessage]:
        conn = self.connect()
        rows = conn.execute(
            "SELECT id, conversation_id, role, content_enc, created_at FROM messages"
            " WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
        return [
            StoredMessage(r[0], r[1], r[2], self._decrypt(r[3]), r[4]) for r in rows
        ]

    def conversations(self) -> list[tuple[int, str]]:
        conn = self.connect()
        return conn.execute(
            "SELECT id, started_at FROM conversations ORDER BY id DESC"
        ).fetchall()

    # --- long-term memory -------------------------------------------------

    def remember(self, content: str, kind: str = "fact") -> int:
        conn = self.connect()
        cursor = conn.execute(
            "INSERT INTO memories (kind, content_enc, created_at) VALUES (?, ?, ?)",
            (kind, self._encrypt(content), _now()),
        )
        conn.commit()
        return cursor.lastrowid

    def all_memories(self) -> list[Memory]:
        conn = self.connect()
        rows = conn.execute(
            "SELECT id, kind, content_enc, created_at FROM memories ORDER BY id"
        ).fetchall()
        return [Memory(r[0], r[1], self._decrypt(r[2]), r[3]) for r in rows]

    def recall(self, query: str, limit: int = 3) -> list[Memory]:
        """Return memories relevant to `query`.

        Scoring happens in Python because the content is encrypted -- SQL cannot
        see inside it. Overlap of meaningful words, normalised by memory length
        so a long rambling note does not outrank a precise one.
        """
        terms = {w for w in _tokenise(query) if w not in STOPWORDS}
        if not terms:
            return []

        scored: list[tuple[float, Memory]] = []
        for memory in self.all_memories():
            words = {w for w in _tokenise(memory.content) if w not in STOPWORDS}
            if not words:
                continue
            overlap = terms & words
            if not overlap:
                continue
            score = len(overlap) / (len(words) ** 0.5)
            scored.append((score, memory))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [memory for _, memory in scored[:limit]]

    # --- deletion ---------------------------------------------------------

    def delete_all(self) -> dict[str, int]:
        """Wipe everything and destroy the key, so leftovers are unrecoverable."""
        conn = self.connect()
        counts = {
            "messages": conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
            "conversations": conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0],
            "memories": conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0],
        }
        conn.executescript(
            "DELETE FROM messages; DELETE FROM conversations; DELETE FROM memories;"
        )
        conn.commit()
        conn.execute("VACUUM")  # actually release the pages, not just mark free
        conn.commit()
        self.close()

        import keyring

        try:
            keyring.delete_password(KEYRING_SERVICE, KEYRING_USER)
        except keyring.errors.PasswordDeleteError:
            pass
        self._fernet = None

        if self.path.exists():
            self.path.unlink()
        return counts


def _tokenise(text: str) -> list[str]:
    return [w.strip(".,!?;:'\"()").lower() for w in text.split() if w.strip(".,!?;:'\"()")]
