"""Every generation, counted, from the first request.

Usage is the one thing in a paid product that cannot be reconstructed later. A
missing feature can be added next week and a wrong price can be changed; a month
of ungrecorded generations is simply gone, and with it the answer to "what does
this cost us" and "what do we charge them". So this lands before billing does,
and it records **whether or not anyone is being charged yet**.

Characters are the unit, because it is the unit this market already prices in:
Sarvam publishes per 10,000 characters, and a customer comparing us to them
should not have to convert. Audio seconds are recorded alongside for the ratio,
which is what turns a character price into a per-minute one when a customer asks
for it in minutes --- as broadcast and dubbing buyers will.

**Failures are recorded too.** A generation that died still occupied the one
machine for its duration, and usage that only counts successes will
systematically under-report load exactly when the system is unhealthy. What is
billable and what was spent are different questions and this table answers both;
`status` is how they are told apart.

Deliberately *not* in the encrypted store. That database holds conversation
memory, where the threat model is a stolen laptop. This is operational
accounting: it wants to be queryable, exportable and boring, and encrypting it
would buy nothing except a key to lose.

**Text is never stored.** Only its length. A metering table that accumulates
what customers typed is a breach waiting for an occasion, and the character
count is the entire billable fact.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from voiceagent import paths

DEFAULT_DB = paths.data_dir() / "usage.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    at                TEXT    NOT NULL,
    account           TEXT    NOT NULL,
    key_id            TEXT,
    characters        INTEGER NOT NULL,
    language          TEXT,
    voice             TEXT,
    audio_seconds     REAL,
    synthesis_seconds REAL,
    queued_seconds    REAL,
    status            TEXT    NOT NULL,
    detail            TEXT
);
CREATE INDEX IF NOT EXISTS idx_usage_account ON usage(account, at);
CREATE INDEX IF NOT EXISTS idx_usage_at ON usage(at);
"""

#: The states a request can end in. `rejected` covers a full queue or a refused
#: request --- it consumed no machine time and must never be billed, but it is
#: the number that says the capacity is too small, so it is not thrown away.
OK = "ok"
FAILED = "failed"
REJECTED = "rejected"


@dataclass(frozen=True)
class Usage:
    account: str
    characters: int
    status: str = OK
    key_id: str | None = None
    language: str | None = None
    voice: str | None = None
    audio_seconds: float | None = None
    synthesis_seconds: float | None = None
    queued_seconds: float | None = None
    detail: str | None = None


class Meter:
    def __init__(self, path: Path | str = DEFAULT_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    # --- writing ----------------------------------------------------------

    def record(self, usage: Usage) -> int:
        """Append one row. Returns its id.

        Never raises on a full disk or a locked database in the caller's path ---
        see `server.py`, which records outside the response path. Losing a
        metering row is bad; failing a generation the customer already waited
        for, because the accounting could not be written, is worse.
        """
        with closing(self._connect()) as conn:
            cur = conn.execute(
                """
                INSERT INTO usage (at, account, key_id, characters, language, voice,
                                   audio_seconds, synthesis_seconds, queued_seconds,
                                   status, detail)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    usage.account,
                    usage.key_id,
                    int(usage.characters),
                    usage.language,
                    usage.voice,
                    usage.audio_seconds,
                    usage.synthesis_seconds,
                    usage.queued_seconds,
                    usage.status,
                    usage.detail,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    # --- reading ----------------------------------------------------------

    def totals(self, account: str, since: str | None = None) -> dict:
        """What an account has used. `since` is an ISO timestamp.

        Billable characters count `ok` rows only. Everything else is reported
        beside it rather than folded in, because a customer disputing an invoice
        is owed the difference between "you generated this" and "this machine
        was busy for you".
        """
        where = "WHERE account = ?"
        args: list = [account]
        if since:
            where += " AND at >= ?"
            args.append(since)

        with closing(self._connect()) as conn:
            row = conn.execute(
                f"""
                SELECT
                    COALESCE(SUM(CASE WHEN status = '{OK}' THEN characters END), 0) AS billable_characters,
                    COALESCE(SUM(CASE WHEN status = '{OK}' THEN audio_seconds END), 0) AS audio_seconds,
                    COALESCE(SUM(CASE WHEN status = '{OK}' THEN synthesis_seconds END), 0) AS machine_seconds,
                    COALESCE(SUM(CASE WHEN status = '{OK}' THEN 1 END), 0) AS generations,
                    COALESCE(SUM(CASE WHEN status = '{FAILED}' THEN 1 END), 0) AS failed,
                    COALESCE(SUM(CASE WHEN status = '{REJECTED}' THEN 1 END), 0) AS rejected
                FROM usage {where}
                """,
                args,
            ).fetchone()

        return {
            "account": account,
            "since": since,
            "billable_characters": int(row["billable_characters"]),
            "audio_seconds": round(float(row["audio_seconds"]), 2),
            "machine_seconds": round(float(row["machine_seconds"]), 2),
            "generations": int(row["generations"]),
            "failed": int(row["failed"]),
            "rejected": int(row["rejected"]),
        }

    def recent(self, account: str, limit: int = 50) -> list[dict]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT at, characters, language, voice, audio_seconds, "
                "synthesis_seconds, status FROM usage WHERE account = ? "
                "ORDER BY id DESC LIMIT ?",
                (account, limit),
            ).fetchall()
        return [dict(r) for r in rows]


def characters_of(text: str) -> int:
    """The billable length of a request.

    Counts the text as the customer submitted it, before normalisation ---
    number expansion and the loanword table can multiply a string severalfold,
    and billing someone for our own preprocessing would be indefensible on the
    first invoice anyone reads closely. Whitespace at the ends is not charged.
    """
    return len(text.strip())
