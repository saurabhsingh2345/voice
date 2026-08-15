"""API keys, which are the whole of a caller's identity.

There are no accounts, no passwords and no sessions behind these, so a mistake
here is not a degraded experience --- it is unauthenticated access or a customer
locked out. Most of these tests are about the failure directions.
"""

from __future__ import annotations

import pytest

from voiceagent.web.keys import PREFIX, KeyStore, parse


@pytest.fixture()
def store(tmp_path):
    return KeyStore(tmp_path / "keys.db")


# --- the secret ------------------------------------------------------------


def test_a_minted_key_verifies(store):
    record, token = store.create("acme@example.com", "staging")
    checked = store.verify(token)
    assert checked is not None
    assert checked.account == "acme@example.com"
    assert checked.label == "staging"


def test_the_plaintext_is_never_stored(store):
    """A leaked database must not be a leaked set of credentials."""
    _record, token = store.create("acme")
    blob = store.path.read_bytes()
    secret = token.split("_")[-1]
    assert secret.encode() not in blob
    assert token.encode() not in blob


def test_two_keys_are_never_the_same(store):
    tokens = {store.create("acme")[1] for _ in range(20)}
    assert len(tokens) == 20


def test_the_key_carries_a_scannable_prefix():
    """Greppable by CI secret scanners and by our own logs, so a key pasted
    into a public repository has a chance of being caught before it is used."""
    store_prefix = PREFIX
    assert store_prefix.startswith("swar_")


def test_a_key_shown_back_is_masked(store):
    record, _token = store.create("acme")
    assert record.key_id in record.masked
    assert record.masked.endswith("........")


# --- every failure looks the same -----------------------------------------


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "nonsense", "swar_live_only_three", "bearer swar_live_a_b",
     "swar_test_abc_def", "swar_live__missing", "swar_live_abc_"],
)
def test_malformed_tokens_are_rejected(store, bad):
    assert store.verify(bad) is None


def test_an_unknown_key_and_a_wrong_secret_are_indistinguishable(store):
    """A caller that can tell "no such key" from "wrong secret" has been handed
    an oracle for enumerating valid ids."""
    _record, token = store.create("acme")
    key_id, _secret = parse(token)
    wrong_secret = f"{PREFIX}_{key_id}_definitelynotthesecret"
    unknown_id = f"{PREFIX}_ffffffff_whatever"
    assert store.verify(wrong_secret) is None
    assert store.verify(unknown_id) is None


def test_a_revoked_key_stops_working(store):
    record, token = store.create("acme")
    assert store.verify(token) is not None
    assert store.revoke(record.key_id)
    assert store.verify(token) is None


def test_revoking_twice_reports_the_second_as_a_miss(store):
    record, _token = store.create("acme")
    assert store.revoke(record.key_id)
    assert not store.revoke(record.key_id)


def test_revoking_an_unknown_key_is_not_an_error(store):
    assert not store.revoke("ffffffff")


# --- bookkeeping -----------------------------------------------------------


def test_last_used_is_recorded(store):
    record, token = store.create("acme")
    assert store.list()[0].last_used is None
    store.verify(token)
    assert store.list()[0].last_used is not None


def test_verification_can_skip_touching(store):
    """So an audit or a test can check a key without rewriting its history."""
    _record, token = store.create("acme")
    store.verify(token, touch=False)
    assert store.list()[0].last_used is None


def test_listing_never_exposes_a_hash_or_secret(store):
    store.create("acme")
    listed = store.list()[0]
    assert not hasattr(listed, "hash")
    assert "hash" not in listed.__dict__


def test_keys_can_be_listed_per_account(store):
    store.create("acme")
    store.create("acme")
    store.create("globex")
    assert len(store.list("acme")) == 2
    assert len(store.list("globex")) == 1
    assert len(store.list()) == 3


def test_reopening_an_existing_database_keeps_keys_working(tmp_path):
    path = tmp_path / "keys.db"
    _record, token = KeyStore(path).create("acme")
    assert KeyStore(path).verify(token) is not None
