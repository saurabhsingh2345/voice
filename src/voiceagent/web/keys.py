"""API keys: minting, storing, and checking them.

A key is the whole of a caller's identity here --- there are no accounts, no
passwords and no sessions --- so the rules are the few that actually matter and
each is a decision rather than a habit.

**Only a hash is stored.** The plaintext is returned once, at creation, and then
cannot be recovered by us. A leaked database is then not a leaked set of
credentials, and "please send me my key again" is answered by issuing a new one.

**SHA-256, not bcrypt.** Deliberate, and the opposite of the advice for
passwords. Password hashing is slow on purpose because passwords are low-entropy
and guessable. A key here is 32 bytes from `secrets` --- 256 bits of entropy ---
so there is nothing to guess, and a slow hash would only add latency to every
single request. The property that matters is the constant-time comparison, which
is used.

**Every key carries a visible id.** `swar_live_<id>_<secret>`: the id is stored
in the clear and is what appears in usage rows and support conversations, so a
customer can say which key without ever sending it. It is also what makes the
lookup a single indexed row rather than a scan of every hash.

**The prefix is a favour to everyone.** `swar_live_` is greppable, so it is
matched by secret scanners in CI and by our own logs, and a key pasted into a
public repository has a chance of being caught before it is used.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from voiceagent import paths

DEFAULT_DB = paths.data_dir() / "keys.db"

#: `live` leaves room for a `test` class later without reissuing anything.
PREFIX = "swar_live"

#: Bytes of randomness in the secret half. 32 is 256 bits; the id is separate
#: and is not a secret.
SECRET_BYTES = 32

#: Characters of the public id. Eight base32-ish characters is ~40 bits, which
#: is far more than enough to be unique across any number of keys this will ever
#: issue, and short enough to read down a phone line.
ID_CHARS = 8

SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    key_id     TEXT PRIMARY KEY,
    hash       TEXT NOT NULL,
    account    TEXT NOT NULL,
    label      TEXT,
    created_at TEXT NOT NULL,
    last_used  TEXT,
    revoked_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_keys_account ON api_keys(account);
"""


@dataclass(frozen=True)
class ApiKey:
    """A stored key. Never carries the secret."""

    key_id: str
    account: str
    label: str | None
    created_at: str
    last_used: str | None = None
    revoked_at: str | None = None

    @property
    def active(self) -> bool:
        return self.revoked_at is None

    @property
    def masked(self) -> str:
        """How a key is shown back to its owner and written in logs."""
        return f"{PREFIX}_{self.key_id}_{'.' * 8}"


def _hash(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def parse(token: str) -> tuple[str, str] | None:
    """Split a presented token into `(key_id, secret)`, or None if malformed.

    Returning None rather than raising keeps the caller's authentication path
    free of exception handling, where a missed `except` becomes an open door.
    """
    token = (token or "").strip()
    parts = token.split("_")
    #: swar, live, id, secret
    if len(parts) != 4:
        return None
    if f"{parts[0]}_{parts[1]}" != PREFIX:
        return None
    if not parts[2] or not parts[3]:
        return None
    return parts[2], parts[3]


class KeyStore:
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

    # --- minting ----------------------------------------------------------

    def create(self, account: str, label: str | None = None) -> tuple[ApiKey, str]:
        """Mint a key. Returns `(record, plaintext)`.

        The plaintext is the only copy that will ever exist. Callers must show
        it and then let it go --- storing it anywhere on our side would undo the
        reason for hashing it.
        """
        key_id = secrets.token_hex(ID_CHARS // 2)
        secret = secrets.token_urlsafe(SECRET_BYTES).replace("_", "").replace("-", "")
        token = f"{PREFIX}_{key_id}_{secret}"

        record = ApiKey(
            key_id=key_id,
            account=account,
            label=label,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO api_keys (key_id, hash, account, label, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (key_id, _hash(secret), account, label, record.created_at),
            )
            conn.commit()
        return record, token

    # --- checking ---------------------------------------------------------

    def verify(self, token: str, touch: bool = True) -> ApiKey | None:
        """Return the key if the token is valid and live, else None.

        One indexed lookup by id, then a constant-time comparison of the secret.
        Every failure --- malformed, unknown, revoked, wrong secret --- returns
        the same `None`, because a caller that can distinguish "no such key"
        from "wrong secret" has been handed an oracle.
        """
        parsed = parse(token)
        if parsed is None:
            return None
        key_id, secret = parsed

        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM api_keys WHERE key_id = ?", (key_id,)
            ).fetchone()
            if row is None:
                return None
            if not hmac.compare_digest(row["hash"], _hash(secret)):
                return None
            if row["revoked_at"] is not None:
                return None
            if touch:
                conn.execute(
                    "UPDATE api_keys SET last_used = ? WHERE key_id = ?",
                    (datetime.now(timezone.utc).isoformat(), key_id),
                )
                conn.commit()

        return ApiKey(
            key_id=row["key_id"],
            account=row["account"],
            label=row["label"],
            created_at=row["created_at"],
            last_used=row["last_used"],
            revoked_at=row["revoked_at"],
        )

    # --- management -------------------------------------------------------

    def revoke(self, key_id: str) -> bool:
        with closing(self._connect()) as conn:
            cur = conn.execute(
                "UPDATE api_keys SET revoked_at = ? WHERE key_id = ? AND revoked_at IS NULL",
                (datetime.now(timezone.utc).isoformat(), key_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def list(self, account: str | None = None) -> list[ApiKey]:
        query = "SELECT * FROM api_keys"
        args: tuple = ()
        if account:
            query += " WHERE account = ?"
            args = (account,)
        query += " ORDER BY created_at DESC"
        with closing(self._connect()) as conn:
            rows = conn.execute(query, args).fetchall()
        return [
            ApiKey(
                key_id=r["key_id"],
                account=r["account"],
                label=r["label"],
                created_at=r["created_at"],
                last_used=r["last_used"],
                revoked_at=r["revoked_at"],
            )
            for r in rows
        ]


def main(argv: list[str] | None = None) -> int:
    """`voice-keys` --- mint and manage keys from the machine that holds them.

    A CLI rather than an endpoint, deliberately: an HTTP route that mints keys
    is a route that must itself be authenticated, and the only thing available
    to authenticate it with is another key. That circle is broken by keeping
    minting on the box.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Manage Swar API keys")
    sub = parser.add_subparsers(dest="command", required=True)

    new = sub.add_parser("create", help="mint a key")
    new.add_argument("account", help="who it belongs to, e.g. an email or company")
    new.add_argument("--label", help="what it is for, e.g. 'staging'")

    show = sub.add_parser("list", help="list keys (never shows secrets)")
    show.add_argument("--account")

    kill = sub.add_parser("revoke", help="revoke a key by its id")
    kill.add_argument("key_id")

    args = parser.parse_args(argv)
    store = KeyStore()

    if args.command == "create":
        record, token = store.create(args.account, args.label)
        print(f"\n  account : {record.account}")
        print(f"  key id  : {record.key_id}")
        print(f"\n  {token}\n")
        print("  This is the only time the key is shown. It is stored hashed;")
        print("  if it is lost, revoke it and mint another.\n")
    elif args.command == "list":
        rows = store.list(args.account)
        if not rows:
            print("  no keys")
        for r in rows:
            state = "revoked" if not r.active else (r.last_used or "never used")
            print(f"  {r.key_id}  {r.account:24} {r.label or '-':16} {state}")
    else:
        print("  revoked" if store.revoke(args.key_id) else "  no such live key")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
