"""Accounts, plans and a credit ledger, denominated in rupees and characters.

This is the accounting half of charging money. The payment half — actually
taking a rupee off someone — is `web/razorpay.py`, and it stops at the point
where credentials are required.

WHY A LEDGER AND NOT A BALANCE COLUMN

A balance column is one `UPDATE` away from being wrong with no way to find out.
An append-only ledger cannot be wrong in that way: the balance is the sum, every
change says who made it and why, and a mistake is corrected by appending its
reverse rather than by editing history. Every double-entry system ever built has
this shape, and billing disputes are exactly the situation where "what did we
think last Tuesday" has to be answerable.

The cost of that choice is a `SUM` per balance check. At the volumes this will
see — one machine, ~2M characters a day at an unreachable ceiling — that is a
few thousand rows a month against an indexed column, and it is not close to
mattering. If it ever does, the fix is a periodic checkpoint row, not a mutable
column.

CHARACTERS ARE THE CREDIT; PAISE ARE THE MONEY

Two units, deliberately not merged. Credits are **characters**, because that is
what this market prices in — Sarvam publishes per 10,000 characters and a
customer comparing us should not have to convert. Money is **paise**, integer,
because money in a float is a bug with a delay on it. A purchase records both:
what was paid and what it bought, at the rate that applied on the day.

That last part matters more than it looks. Recording only "₹499" or only
"500,000 characters" makes a price change unauditable afterwards. Recording both
means an old row still explains itself when the pricing page has moved on.

WHAT THE FREE TIER IS ACTUALLY SPENDING

Not rupees. `eval_out/cogs/FINDINGS.md` measured the marginal cost of a character
at a fraction of a paisa — electricity is 0.14–0.28% of what Sarvam charges — so
a free tier does not have a meaningful *cost*. What it has is a claim on
**capacity**: one machine, concurrency 1, 86,400 machine-seconds a day, of which
a generation consumes about 40 seconds per 1000 characters.

So the free tier is capped on both, and the second cap is the real one. A free
account's monthly character allowance is small enough that the whole free
population cannot consume the day. Sizing it in rupees would have produced a
generous-looking number that quietly eats the product.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from voiceagent import paths

DEFAULT_DB = paths.data_dir() / "billing.db"

#: Seconds of the one machine per 1000 characters, measured in
#: `voiceagent.eval.cogs`. Five sweeps agreed within 37.7-41.9; the intercept did
#: not converge and is deliberately not modelled here. Used to express an
#: allowance in the unit that is actually scarce.
MACHINE_SECONDS_PER_1K_CHARS = 40.0

#: Seconds in a day. The entire capacity of the product, since concurrency is 1.
MACHINE_SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class Plan:
    """A tier. Prices in paise; allowances in characters per month."""

    name: str
    monthly_paise: int
    monthly_characters: int
    #: Whether an exhausted allowance blocks, or bills the overage. Free blocks;
    #: paid plans that block would fail a customer mid-narration for the sake of
    #: a few rupees, which is a worse outcome for both sides.
    blocks_when_empty: bool
    note: str = ""

    @property
    def monthly_inr(self) -> float:
        return self.monthly_paise / 100

    @property
    def machine_seconds_per_month(self) -> float:
        """What one account on this plan can claim of the machine."""
        return self.monthly_characters / 1000 * MACHINE_SECONDS_PER_1K_CHARS

    def share_of_daily_capacity(self, accounts: int = 1) -> float:
        """Fraction of a day's capacity `accounts` of these would consume.

        The number that decides whether a free tier is survivable. Sized in
        rupees the answer looks free; sized in seconds it is a queue.
        """
        monthly_capacity = MACHINE_SECONDS_PER_DAY * 30
        return self.machine_seconds_per_month * accounts / monthly_capacity

    def accounts_supported(self, utilisation: float = 0.30) -> int:
        """How many accounts on this plan one machine holds, fully spent.

        The number that decides when a second machine has to exist, and it is
        uncomfortably small. `utilisation` defaults to 0.30 rather than 1.0
        because the machine is also a workstation and because concurrency is 1:
        a queue that is busy 100% of the time is a queue nobody waits in.

        Assumes every account spends its whole allowance, which no real
        population does. Treat it as the floor of a planning range, not a
        forecast --- but plan against the floor, because the month everyone
        does use their allowance is the month the product falls over.
        """
        if self.monthly_characters <= 0:
            return 0
        usable = MACHINE_SECONDS_PER_DAY * 30 * utilisation
        return int(usable // self.machine_seconds_per_month)

    @property
    def inr_per_10k_characters(self) -> float:
        """The unit the market quotes, so the comparison is direct.

        Sarvam publishes ₹15 per 10k (Bulbul v2) and ₹30 (v3). A plan priced
        above that on this measure is not competing, whatever the monthly figure
        looks like.
        """
        if self.monthly_characters <= 0:
            return 0.0
        return self.monthly_inr / (self.monthly_characters / 10_000)


#: The tiers from plan §8, priced in rupees rather than converted from dollars.
#:
#: FREE is 5,000 characters a month: enough to hear the product on real text of
#: your own, not enough to ship with. In the unit that matters that is ~200
#: machine-seconds, so a thousand free accounts all spending their full
#: allowance in the same month come to ~2.3 days of machine time — survivable
#: precisely because the cap was set in seconds and then converted, rather than
#: the other way round.
#:
#: CREATOR at ₹499 undercuts ElevenLabs' $22 Starter on purchasing power rather
#: than on cost, which is the §8 rule. DEVELOPER is the same characters with the
#: overage door open, because an integration that stops mid-month is a support
#: ticket and a churned customer.
PLANS: dict[str, Plan] = {
    "free": Plan(
        name="free",
        monthly_paise=0,
        monthly_characters=5_000,
        blocks_when_empty=True,
        note="Enough to hear it on your own text; not enough to ship with.",
    ),
    "creator": Plan(
        name="creator",
        monthly_paise=49_900,
        monthly_characters=500_000,
        blocks_when_empty=False,
        note="₹499/month. About 10 hours of finished audio.",
    ),
    "developer": Plan(
        name="developer",
        monthly_paise=199_900,
        monthly_characters=2_500_000,
        blocks_when_empty=False,
        note="₹1,999/month, metered overage. The compounding channel.",
    ),
}

DEFAULT_PLAN = "free"

#: Overage price once a paid plan's monthly allowance is spent, in paise per
#: 10,000 characters. ₹25 sits under Sarvam's ₹30 for Bulbul v3 and above their
#: ₹15 for v2 --- deliberately not the cheapest, because winning on price against
#: a better-funded incumbent is the losing half of plan §1.3.
OVERAGE_PAISE_PER_10K = 2_500

#: Ledger entry kinds. Sign convention: `characters` is positive when it adds to
#: what the account may spend and negative when it consumes.
GRANT = "grant"          # monthly allowance, or goodwill
PURCHASE = "purchase"    # credits bought with money
DEBIT = "debit"          # a generation
REFUND = "refund"        # a reversal, always appended never edited
EXPIRY = "expiry"        # unused allowance removed at period end

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    account     TEXT PRIMARY KEY,
    plan        TEXT NOT NULL,
    email       TEXT,
    created_at  TEXT NOT NULL,
    period_start TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ledger (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    at          TEXT NOT NULL,
    account     TEXT NOT NULL,
    kind        TEXT NOT NULL,
    characters  INTEGER NOT NULL,
    paise       INTEGER NOT NULL DEFAULT 0,
    reference   TEXT,
    note        TEXT
);

CREATE INDEX IF NOT EXISTS idx_ledger_account ON ledger(account, at);

-- A payment may be notified more than once: Razorpay retries webhooks until it
-- gets a 2xx, and a retry after a timeout is normal rather than exceptional. The
-- unique index is what makes crediting an account idempotent, and it is enforced
-- by the database rather than by remembering to check.
CREATE UNIQUE INDEX IF NOT EXISTS idx_ledger_reference
    ON ledger(reference) WHERE reference IS NOT NULL;
"""


class InsufficientCredits(Exception):
    """Raised when a blocking plan has nothing left to spend."""

    def __init__(self, balance: int, needed: int) -> None:
        self.balance = balance
        self.needed = needed
        super().__init__(
            f"{needed} characters requested, {balance} remaining this period."
        )


@dataclass(frozen=True)
class Account:
    account: str
    plan: Plan
    email: str | None
    created_at: str
    period_start: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Billing:
    """Accounts and their ledger. One SQLite file, no ORM, no migrations yet."""

    def __init__(self, path: Path | str = DEFAULT_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    # --- accounts ---------------------------------------------------------

    def ensure_account(self, account: str, plan: str = DEFAULT_PLAN,
                       email: str | None = None) -> Account:
        """Fetch an account, creating it on the free plan if new.

        Creating on first sight rather than requiring a signup step: a key can
        already be minted without one, and an account that exists in `keys.db`
        but not here would be billable-but-unbilled, which is the failure this
        module exists to prevent.
        """
        with closing(self._connect()) as conn, conn:
            row = conn.execute(
                "SELECT * FROM accounts WHERE account = ?", (account,)
            ).fetchone()
            if row is None:
                now = _now()
                conn.execute(
                    "INSERT INTO accounts (account, plan, email, created_at, period_start)"
                    " VALUES (?,?,?,?,?)",
                    (account, plan, email, now, now),
                )
                conn.execute(
                    "INSERT INTO ledger (at, account, kind, characters, paise, note)"
                    " VALUES (?,?,?,?,?,?)",
                    (now, account, GRANT, PLANS[plan].monthly_characters, 0,
                     f"opening {plan} allowance"),
                )
                row = conn.execute(
                    "SELECT * FROM accounts WHERE account = ?", (account,)
                ).fetchone()
        return Account(
            account=row["account"],
            plan=PLANS.get(row["plan"], PLANS[DEFAULT_PLAN]),
            email=row["email"],
            created_at=row["created_at"],
            period_start=row["period_start"],
        )

    def set_plan(self, account: str, plan: str) -> Account:
        """Move an account to another plan and grant the difference.

        Grants the *difference* rather than the full new allowance, so upgrading
        mid-period neither double-grants nor silently removes what was already
        bought. Downgrading grants a negative number, which the ledger is happy
        to hold and which shows up in the history as what it was.
        """
        if plan not in PLANS:
            raise KeyError(f"unknown plan {plan!r}")
        current = self.ensure_account(account)
        delta = PLANS[plan].monthly_characters - current.plan.monthly_characters
        with closing(self._connect()) as conn, conn:
            conn.execute("UPDATE accounts SET plan = ? WHERE account = ?", (plan, account))
            if delta:
                conn.execute(
                    "INSERT INTO ledger (at, account, kind, characters, paise, note)"
                    " VALUES (?,?,?,?,?,?)",
                    (_now(), account, GRANT, delta, 0,
                     f"plan change {current.plan.name} -> {plan}"),
                )
        return self.ensure_account(account)

    # --- ledger -----------------------------------------------------------

    def balance(self, account: str) -> int:
        """Characters remaining. Negative means overage has been used."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(characters), 0) AS total FROM ledger WHERE account = ?",
                (account,),
            ).fetchone()
        return int(row["total"])

    def _append(self, account: str, kind: str, characters: int, paise: int = 0,
                reference: str | None = None, note: str | None = None) -> int | None:
        """Append one entry. Returns None if `reference` was already recorded.

        The duplicate case is not an error: it is a webhook retry, and the
        correct response to one is to do nothing and report success.
        """
        with closing(self._connect()) as conn, conn:
            try:
                cursor = conn.execute(
                    "INSERT INTO ledger (at, account, kind, characters, paise, reference, note)"
                    " VALUES (?,?,?,?,?,?,?)",
                    (_now(), account, kind, characters, paise, reference, note),
                )
            except sqlite3.IntegrityError:
                return None
            return int(cursor.lastrowid)

    def grant(self, account: str, characters: int, note: str = "") -> int | None:
        return self._append(account, GRANT, characters, note=note)

    def purchase(self, account: str, characters: int, paise: int,
                 reference: str, note: str = "") -> int | None:
        """Credits bought. `reference` is the payment id and must be unique."""
        return self._append(account, PURCHASE, characters, paise=paise,
                            reference=reference, note=note)

    def refund(self, account: str, characters: int, paise: int,
               reference: str, note: str = "") -> int | None:
        return self._append(account, REFUND, -abs(characters), paise=-abs(paise),
                            reference=reference, note=note)

    def debit(self, account: str, characters: int, reference: str | None = None,
              note: str = "") -> int | None:
        return self._append(account, DEBIT, -abs(characters), reference=reference,
                            note=note)

    # --- enforcement ------------------------------------------------------

    def check_affordable(self, account: str, characters: int) -> None:
        """Raise `InsufficientCredits` if this request must not proceed.

        Checked *before* synthesis, because the machine is the scarce thing and
        spending 40 seconds of it on a request that will be refused is the worst
        of both outcomes.

        Paid plans do not block. Their overage is billed, which is the honest
        trade: a narration that stops halfway through costs the customer more
        than the overage does, and costs us the customer.
        """
        acct = self.ensure_account(account)
        if not acct.plan.blocks_when_empty:
            return
        remaining = self.balance(account)
        if remaining < characters:
            raise InsufficientCredits(balance=remaining, needed=characters)

    def overage_paise(self, account: str) -> int:
        """What this period's overage would invoice at, in paise.

        Zero unless the balance is negative, which only a non-blocking plan can
        reach. Rounded up to a whole 10,000 the way the market quotes it.
        """
        deficit = -min(0, self.balance(account))
        if not deficit:
            return 0
        blocks = -(-deficit // 10_000)  # ceiling division
        return blocks * OVERAGE_PAISE_PER_10K

    # --- reporting --------------------------------------------------------

    def summary(self, account: str) -> dict:
        """Everything a customer or a support conversation needs at once."""
        acct = self.ensure_account(account)
        remaining = self.balance(account)
        return {
            "account": acct.account,
            "plan": acct.plan.name,
            "plan_inr_per_month": acct.plan.monthly_inr,
            "monthly_characters": acct.plan.monthly_characters,
            "characters_remaining": remaining,
            "blocks_when_empty": acct.plan.blocks_when_empty,
            "overage_inr": self.overage_paise(acct.account) / 100,
            # The unit that is actually scarce. Exposed because a developer
            # sizing a job should be able to see that a million characters is
            # eleven hours of a machine that has twenty-four.
            "machine_seconds_remaining": max(0, remaining)
            / 1000 * MACHINE_SECONDS_PER_1K_CHARS,
            "period_start": acct.period_start,
        }

    def entries(self, account: str, limit: int = 50) -> list[dict]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT at, kind, characters, paise, reference, note FROM ledger"
                " WHERE account = ? ORDER BY id DESC LIMIT ?",
                (account, limit),
            ).fetchall()
        return [dict(r) for r in rows]
