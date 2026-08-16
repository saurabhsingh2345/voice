"""Tests for accounts, the credit ledger and the Razorpay boundary.

Money is the one part of this project where a silent error is unrecoverable: a
generation that sounds wrong gets regenerated, and a balance that drifts is
discovered a month later by a customer. So the cases here are the ones that
actually lose or invent money — double-crediting a retried webhook, charging for
a failed generation, verifying a signature against the wrong bytes, and sending
rupees where paise were meant.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from voiceagent.web import razorpay
from voiceagent.web.billing import (
    DEBIT,
    OVERAGE_PAISE_PER_10K,
    PLANS,
    Billing,
    InsufficientCredits,
)


@pytest.fixture()
def books(tmp_path):
    return Billing(tmp_path / "billing.db")


# --- accounts and allowances ---------------------------------------------


def test_a_new_account_starts_free_with_its_allowance_granted(books):
    """Created on first sight rather than at a signup step. An account that
    exists in keys.db but not here would be billable-but-unbilled."""
    account = books.ensure_account("acme")
    assert account.plan.name == "free"
    assert books.balance("acme") == PLANS["free"].monthly_characters


def test_ensuring_twice_does_not_grant_twice(books):
    books.ensure_account("acme")
    books.ensure_account("acme")
    assert books.balance("acme") == PLANS["free"].monthly_characters


def test_upgrading_grants_the_difference_not_the_whole_allowance(books):
    """Granting the full new allowance on upgrade would hand out the free tier
    twice; granting nothing would sell a plan and withhold it."""
    books.ensure_account("acme")
    books.debit("acme", 1_000)
    books.set_plan("acme", "creator")

    expected = PLANS["creator"].monthly_characters - 1_000
    assert books.balance("acme") == expected


def test_downgrading_removes_the_difference(books):
    books.ensure_account("acme")
    books.set_plan("acme", "creator")
    books.set_plan("acme", "free")
    assert books.balance("acme") == PLANS["free"].monthly_characters


# --- enforcement ----------------------------------------------------------


def test_the_free_plan_blocks_when_it_runs_out(books):
    books.ensure_account("acme")
    books.debit("acme", PLANS["free"].monthly_characters)

    with pytest.raises(InsufficientCredits) as caught:
        books.check_affordable("acme", 100)
    assert caught.value.balance == 0
    assert caught.value.needed == 100


def test_a_paid_plan_runs_into_overage_rather_than_stopping(books):
    """Deliberate. Stopping a narration halfway to protect a few rupees costs
    more in support and churn than the overage is worth."""
    books.ensure_account("acme")
    books.set_plan("acme", "creator")
    books.debit("acme", PLANS["creator"].monthly_characters)

    books.check_affordable("acme", 50_000)  # must not raise


def test_overage_is_charged_in_whole_blocks_of_ten_thousand(books):
    """Rounded up, the way the market quotes it. A single character past the
    allowance is a block, and pretending otherwise invents fractional paise."""
    books.ensure_account("acme")
    books.set_plan("acme", "creator")
    books.debit("acme", PLANS["creator"].monthly_characters + 1)

    assert books.overage_paise("acme") == OVERAGE_PAISE_PER_10K

    books.debit("acme", 10_000)
    assert books.overage_paise("acme") == 2 * OVERAGE_PAISE_PER_10K


def test_an_account_in_credit_has_no_overage(books):
    books.ensure_account("acme")
    assert books.overage_paise("acme") == 0


# --- the ledger is a ledger ----------------------------------------------


def test_the_balance_is_the_sum_of_history_not_a_stored_number(books):
    books.ensure_account("acme")
    books.grant("acme", 1_000, note="goodwill")
    books.debit("acme", 400)
    books.debit("acme", 100)

    assert books.balance("acme") == PLANS["free"].monthly_characters + 500
    kinds = [e["kind"] for e in books.entries("acme")]
    assert kinds.count(DEBIT) == 2


def test_a_repeated_payment_reference_is_credited_once(books):
    """The webhook case, and the one that costs real money. Razorpay retries
    until it gets a 2xx, so the same payment arrives more than once as a matter
    of course rather than as an error."""
    books.ensure_account("acme")
    before = books.balance("acme")

    first = books.purchase("acme", 500_000, 49_900, reference="razorpay:pay_1")
    second = books.purchase("acme", 500_000, 49_900, reference="razorpay:pay_1")

    assert first is not None
    assert second is None, "the retry must be a no-op, not a second credit"
    assert books.balance("acme") == before + 500_000


def test_distinct_payments_both_credit(books):
    books.ensure_account("acme")
    assert books.purchase("acme", 1_000, 100, reference="razorpay:pay_1") is not None
    assert books.purchase("acme", 1_000, 100, reference="razorpay:pay_2") is not None


def test_a_refund_is_appended_not_edited(books):
    """History is never rewritten. A mistake is corrected by its reverse, which
    is what makes 'what did we think last Tuesday' answerable."""
    books.ensure_account("acme")
    books.purchase("acme", 1_000, 100, reference="razorpay:pay_1")
    books.refund("acme", 1_000, 100, reference="razorpay:refund_1")

    assert books.balance("acme") == PLANS["free"].monthly_characters
    assert len(books.entries("acme")) == 3  # opening grant, purchase, refund


def test_debits_are_always_negative_however_they_are_called(books):
    """Guards a sign error that would silently top an account up every time it
    generated."""
    books.ensure_account("acme")
    books.debit("acme", 100)
    books.debit("acme", -100)
    assert books.balance("acme") == PLANS["free"].monthly_characters - 200


# --- the free tier is sized in the unit that is scarce --------------------


def test_the_free_tier_is_small_in_machine_seconds_not_just_rupees(books):
    """The sizing argument from `eval/cogs.py`. A thousand free accounts each
    spending their full monthly allowance must not exceed the machine's month."""
    free = PLANS["free"]
    assert free.share_of_daily_capacity(accounts=1_000) < 0.10


def test_a_paid_plan_claims_a_visible_slice_of_the_machine(books):
    """The other half: the price has to be worth the capacity it reserves. If a
    single creator account were a rounding error the plan would be underpriced,
    and if it were most of a month it would be unservable."""
    share = PLANS["creator"].share_of_daily_capacity(accounts=1)
    assert 0.0 < share < 0.02


def test_the_summary_reports_machine_seconds(books):
    books.ensure_account("acme")
    summary = books.summary("acme")
    assert summary["plan"] == "free"
    assert summary["machine_seconds_remaining"] > 0
    assert summary["characters_remaining"] == PLANS["free"].monthly_characters


# --- Razorpay: signatures -------------------------------------------------


WEBHOOK_SECRET = "whsec_test"
KEY_SECRET = "keysec_test"


def _sign(secret: str, message: bytes) -> str:
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


def test_a_correctly_signed_webhook_verifies():
    body = b'{"event":"payment.captured"}'
    razorpay.verify_webhook(body, _sign(WEBHOOK_SECRET, body), WEBHOOK_SECRET)


def test_a_tampered_body_is_refused():
    body = b'{"event":"payment.captured","amount":100}'
    signature = _sign(WEBHOOK_SECRET, body)
    with pytest.raises(razorpay.SignatureInvalid):
        razorpay.verify_webhook(body.replace(b"100", b"999"), signature, WEBHOOK_SECRET)


def test_the_wrong_secret_is_refused():
    body = b'{"event":"payment.captured"}'
    with pytest.raises(razorpay.SignatureInvalid):
        razorpay.verify_webhook(body, _sign("other", body), WEBHOOK_SECRET)


@pytest.mark.parametrize("signature", ["", "   ", "not-hex", None])
def test_a_missing_or_junk_signature_is_refused(signature):
    body = b'{"event":"payment.captured"}'
    with pytest.raises(razorpay.SignatureInvalid):
        razorpay.verify_webhook(body, signature, WEBHOOK_SECRET)


def test_reserialised_json_does_not_verify():
    """The failure that passes in testing and rejects every real webhook in
    production. `json.dumps` of a parsed payload differs in whitespace and key
    order, so the HMAC differs -- which is why `verify_webhook` takes bytes."""
    body = b'{"event":"payment.captured", "x":1}'
    signature = _sign(WEBHOOK_SECRET, body)
    reserialised = json.dumps(json.loads(body)).encode()

    assert reserialised != body
    with pytest.raises(razorpay.SignatureInvalid):
        razorpay.verify_webhook(reserialised, signature, WEBHOOK_SECRET)


def test_checkout_uses_a_different_message_and_a_different_secret():
    """Razorpay signs two things with two secrets and confusing them is easy and
    expensive. Checkout is order|payment keyed with the API secret; the webhook
    is the raw body keyed with the webhook secret."""
    order_id, payment_id = "order_abc", "pay_xyz"
    signature = _sign(KEY_SECRET, f"{order_id}|{payment_id}".encode())

    razorpay.verify_checkout(order_id, payment_id, signature, KEY_SECRET)

    with pytest.raises(razorpay.SignatureInvalid):
        razorpay.verify_checkout(order_id, payment_id, signature, WEBHOOK_SECRET)
    with pytest.raises(razorpay.SignatureInvalid):
        razorpay.verify_checkout(payment_id, order_id, signature, KEY_SECRET)


# --- Razorpay: the amount unit -------------------------------------------


def test_the_order_amount_is_paise_and_must_be_an_int():
    """Sending rupees charges a hundredth of the intended amount, and it fails
    silently in the direction that loses money."""
    payload = razorpay.order_payload(49_900, "acme", "creator", "r1")
    assert payload["amount"] == 49_900
    assert payload["currency"] == "INR"
    assert payload["notes"] == {"account": "acme", "plan": "creator"}

    with pytest.raises(TypeError):
        razorpay.order_payload(499.0, "acme", "creator", "r1")
    with pytest.raises(TypeError):
        razorpay.order_payload(True, "acme", "creator", "r1")
    with pytest.raises(ValueError):
        razorpay.order_payload(0, "acme", "creator", "r1")


# --- Razorpay: the boundary itself ---------------------------------------


def test_without_credentials_a_live_call_refuses_clearly(monkeypatch):
    """The boundary. It names the three environment variables rather than
    failing with a connection error, because the person hitting this is setting
    the integration up for the first time."""
    for name in (razorpay.ENV_KEY_ID, razorpay.ENV_KEY_SECRET,
                 razorpay.ENV_WEBHOOK_SECRET):
        monkeypatch.delenv(name, raising=False)

    assert razorpay.configured() is False
    with pytest.raises(razorpay.CredentialsMissing) as caught:
        razorpay.create_order(49_900, "acme", "creator", "r1")
    assert razorpay.ENV_KEY_ID in str(caught.value)


def test_partial_credentials_count_as_unconfigured(monkeypatch):
    """All-or-nothing. An integration that can create an order but cannot verify
    the webhook takes money it cannot confirm."""
    monkeypatch.setenv(razorpay.ENV_KEY_ID, "rzp_test_x")
    monkeypatch.setenv(razorpay.ENV_KEY_SECRET, "secret")
    monkeypatch.delenv(razorpay.ENV_WEBHOOK_SECRET, raising=False)

    assert razorpay.configured() is False


def test_full_credentials_are_read(monkeypatch):
    monkeypatch.setenv(razorpay.ENV_KEY_ID, "rzp_test_x")
    monkeypatch.setenv(razorpay.ENV_KEY_SECRET, "secret")
    monkeypatch.setenv(razorpay.ENV_WEBHOOK_SECRET, "whsec")

    assert razorpay.configured() is True
    assert razorpay.Credentials.from_env().key_id == "rzp_test_x"


# --- Razorpay: which events mean money -----------------------------------


def _event(name: str, payment_id: str = "pay_1", amount: int = 49_900,
           account: str | None = "acme", plan: str | None = "creator") -> bytes:
    notes = {}
    if account:
        notes["account"] = account
    if plan:
        notes["plan"] = plan
    return json.dumps({
        "event": name,
        "payload": {"payment": {"entity": {
            "id": payment_id, "amount": amount, "notes": notes,
        }}},
    }).encode()


def test_only_captured_payments_credit():
    """`authorized` means the bank agreed, not that we hold the funds. Crediting
    on it hands out credits for payments that can still fail."""
    captured = razorpay.parse_event(_event("payment.captured"))
    authorized = razorpay.parse_event(_event("payment.authorized"))

    assert captured.credits is True
    assert authorized.credits is False
    assert authorized.reverses is False


def test_refunds_and_failures_reverse():
    assert razorpay.parse_event(_event("refund.processed")).reverses is True
    assert razorpay.parse_event(_event("payment.failed")).reverses is True


def test_the_reference_is_stable_across_retries():
    """What makes the ledger's unique index able to swallow a retry."""
    first = razorpay.parse_event(_event("payment.captured", payment_id="pay_9"))
    again = razorpay.parse_event(_event("payment.captured", payment_id="pay_9"))
    other = razorpay.parse_event(_event("payment.captured", payment_id="pay_8"))

    assert first.reference == again.reference
    assert first.reference != other.reference


def test_an_event_without_an_account_is_readable_rather_than_crashing():
    """A payment made outside our checkout has no notes. It must not raise --
    the webhook has to answer 200 or Razorpay retries it forever."""
    event = razorpay.parse_event(_event("payment.captured", account=None, plan=None))
    assert event.account is None
    assert event.credits is True


# --- the pricing has to survive the comparison a buyer will make ---------


#: Sarvam's published rates, the anchor from plan §1.3. Checked 2026-08-14;
#: `outreach/CLAIMS.md` says to re-check before quoting these to anyone.
SARVAM_INR_PER_10K = {"bulbul_v2": 15.0, "bulbul_v3": 30.0}


@pytest.mark.parametrize("plan_name", ["creator", "developer"])
def test_paid_plans_undercut_sarvam_on_the_unit_the_market_quotes(plan_name):
    """A monthly price that looks cheap can still lose on ₹/10k, which is the
    number a buyer actually compares. Plan §8: undercut on purchasing power."""
    plan = PLANS[plan_name]
    assert 0 < plan.inr_per_10k_characters < SARVAM_INR_PER_10K["bulbul_v2"]


def test_overage_sits_between_sarvams_two_tiers():
    """Deliberately not the cheapest. Winning on price against a better-funded
    incumbent is the losing half of plan §1.3."""
    overage_inr = OVERAGE_PAISE_PER_10K / 100
    assert SARVAM_INR_PER_10K["bulbul_v2"] < overage_inr < SARVAM_INR_PER_10K["bulbul_v3"]


def test_one_machine_holds_a_small_and_known_number_of_paying_accounts():
    """The constraint that decides when a second machine has to exist. It is
    uncomfortably small and that is the point of asserting it: if a pricing
    change silently raises these, the machine cannot serve what was sold."""
    assert PLANS["creator"].accounts_supported(utilisation=0.30) == 38
    assert PLANS["developer"].accounts_supported(utilisation=0.30) == 7


def test_a_free_plan_supports_no_paid_capacity_planning():
    """Guards a division by zero, and states the obvious: a plan with no
    allowance has no capacity question."""
    from voiceagent.web.billing import Plan

    empty = Plan("empty", monthly_paise=0, monthly_characters=0, blocks_when_empty=True)
    assert empty.accounts_supported() == 0
    assert empty.inr_per_10k_characters == 0.0
