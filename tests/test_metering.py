"""Usage accounting, which is the one thing that cannot be reconstructed later.

A missing feature can be added next week and a wrong price can be changed. A
month of unrecorded generations is gone, and with it both "what does this cost
us" and "what do we charge them". So these tests mostly guard the boundaries an
invoice gets argued over: what is billable, what merely occupied the machine,
and what was never charged at all.
"""

from __future__ import annotations

import pytest

from voiceagent.web.metering import (
    FAILED,
    OK,
    REJECTED,
    Meter,
    Usage,
    characters_of,
)


@pytest.fixture()
def meter(tmp_path):
    return Meter(tmp_path / "usage.db")


# --- what gets billed ------------------------------------------------------


def test_a_successful_generation_is_billable(meter):
    meter.record(Usage(account="acme", characters=120, audio_seconds=8.0))
    assert meter.totals("acme")["billable_characters"] == 120


def test_a_failed_generation_is_recorded_but_not_billed(meter):
    """It occupied the one machine, so it is not discarded --- but the customer
    did not get audio and must not pay for it."""
    meter.record(Usage(account="acme", characters=120, status=FAILED))
    totals = meter.totals("acme")
    assert totals["billable_characters"] == 0
    assert totals["failed"] == 1


def test_a_rejected_request_is_recorded_and_not_billed(meter):
    """A full queue consumed no machine time, but the count of them is the
    number that says capacity is too small. Throwing it away hides the problem
    it exists to reveal."""
    meter.record(Usage(account="acme", characters=90, status=REJECTED))
    totals = meter.totals("acme")
    assert totals["billable_characters"] == 0
    assert totals["rejected"] == 1
    assert totals["generations"] == 0


def test_accounts_are_kept_apart(meter):
    meter.record(Usage(account="acme", characters=100))
    meter.record(Usage(account="globex", characters=250))
    assert meter.totals("acme")["billable_characters"] == 100
    assert meter.totals("globex")["billable_characters"] == 250


def test_an_account_with_no_usage_reports_zeroes_not_an_error(meter):
    """The first thing a new account's dashboard does is ask this question."""
    totals = meter.totals("nobody")
    assert totals["billable_characters"] == 0
    assert totals["generations"] == 0


# --- the ratio a customer will ask for -------------------------------------


def test_audio_and_machine_seconds_are_tracked_beside_characters(meter):
    """Characters are the billing unit, but broadcast and dubbing buyers ask in
    minutes, and machine seconds are what says whether the price covers cost."""
    meter.record(
        Usage(account="acme", characters=900, audio_seconds=60.0, synthesis_seconds=33.0)
    )
    totals = meter.totals("acme")
    assert totals["audio_seconds"] == 60.0
    assert totals["machine_seconds"] == 33.0


def test_recent_lists_newest_first(meter):
    for n in range(3):
        meter.record(Usage(account="acme", characters=n + 1))
    recent = meter.recent("acme")
    assert [r["characters"] for r in recent] == [3, 2, 1]


def test_since_filters_by_time(meter):
    meter.record(Usage(account="acme", characters=50))
    assert meter.totals("acme", since="2999-01-01T00:00:00+00:00")["billable_characters"] == 0


# --- the counting rule -----------------------------------------------------


def test_length_is_counted_before_normalisation():
    """Number expansion and the loanword table can multiply a string severalfold.
    Billing for our own preprocessing would not survive the first invoice anyone
    reads closely."""
    assert characters_of("मेरे पास 25 रुपये हैं") == len("मेरे पास 25 रुपये हैं")


def test_surrounding_whitespace_is_not_charged():
    assert characters_of("  नमस्ते  ") == len("नमस्ते")


# --- the privacy rule ------------------------------------------------------


def test_the_submitted_text_is_never_stored(meter):
    """A metering table that accumulates what customers typed is a breach
    waiting for an occasion. The length is the entire billable fact."""
    secret = "यह वाक्य कभी संग्रहीत नहीं होना चाहिए"
    meter.record(Usage(account="acme", characters=characters_of(secret)))
    blob = meter.path.read_bytes()
    assert secret.encode("utf-8") not in blob


def test_the_schema_has_no_text_column(meter):
    import sqlite3

    conn = sqlite3.connect(meter.path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(usage)")}
    conn.close()
    assert "text" not in columns and "prompt" not in columns
    assert "characters" in columns


def test_reopening_an_existing_database_is_safe(tmp_path):
    path = tmp_path / "usage.db"
    Meter(path).record(Usage(account="acme", characters=10))
    assert Meter(path).totals("acme")["billable_characters"] == 10
