"""The Razorpay boundary: everything up to the credential, and nothing past it.

This module is deliberately the end of the road. It builds the order, verifies
the signatures and decides what a webhook means — all of which can be written and
tested now — and it refuses clearly at the one point that needs an account nobody
has created yet.

WHY RAZORPAY AND NOT STRIPE

Plan §8, and it does not bend: rupees and UPI. UPI is how India pays, Stripe's
India support is not the same product as its US one, and a card wall denominated
in dollars loses the market this is named after. Razorpay is the default for an
Indian company taking Indian money.

WHAT IS IMPLEMENTED AND WHAT IS NOT

Implemented, tested, and correct without any credential:

  * **Webhook signature verification.** HMAC-SHA256 over the raw body, compared
    in constant time. This is the security-critical half and it is pure
    computation, so there is no excuse for leaving it until credentials arrive.
  * **Order payload construction**, including the amount-in-paise convention that
    is the classic way to charge someone a hundred times too much.
  * **Event interpretation** — which webhook events credit an account, which are
    noise, and what reference makes the credit idempotent.

Not implemented, because it cannot be:

  * The HTTP call itself runs, but only once `RAZORPAY_KEY_ID` and
    `RAZORPAY_KEY_SECRET` exist in the environment. Without them
    `create_order` raises `CredentialsMissing` rather than pretending.

There is no sandbox-mode fake in here. A stub that returns a plausible order id
would let the whole flow appear to work, and the first time it met the real API
every untested assumption would surface at once, in production, holding
somebody's money.

THE TWO SIGNATURES ARE NOT THE SAME

Razorpay signs two different things with two different secrets and it is an easy
and expensive confusion:

  * **Checkout callback** — HMAC of `order_id|payment_id`, keyed with the **API
    key secret**. Proves the browser's success callback is genuine.
  * **Webhook** — HMAC of the **raw request body**, keyed with the separate
    **webhook secret** configured in the dashboard. Proves the server-to-server
    notification is genuine.

Both are here, named for which is which. Verify the webhook body *before*
parsing it as JSON, and against the bytes exactly as received: re-serialising
parsed JSON changes whitespace and key order, and the signature will not match.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import urllib.error
import urllib.request
from base64 import b64encode
from dataclasses import dataclass

#: Razorpay's REST base. Orders are created server-side; the client never holds
#: the key secret.
API_BASE = "https://api.razorpay.com/v1"

ENV_KEY_ID = "RAZORPAY_KEY_ID"
ENV_KEY_SECRET = "RAZORPAY_KEY_SECRET"
ENV_WEBHOOK_SECRET = "RAZORPAY_WEBHOOK_SECRET"

#: Events that mean money arrived. `payment.captured` is the one that matters:
#: `payment.authorized` means the customer's bank agreed, not that we hold the
#: funds, and crediting on it hands out credits for payments that can still fail.
CREDITING_EVENTS = frozenset({"payment.captured"})

#: Events that reverse it.
REVERSING_EVENTS = frozenset({"refund.processed", "payment.failed"})


class CredentialsMissing(RuntimeError):
    """Raised when a live call is attempted without configured credentials."""


class SignatureInvalid(Exception):
    """Raised when a payload's signature does not verify. Never log the body."""


@dataclass(frozen=True)
class Credentials:
    key_id: str
    key_secret: str
    webhook_secret: str

    @classmethod
    def from_env(cls) -> "Credentials | None":
        """Read credentials, or None if they are not all present.

        All-or-nothing on purpose. A half-configured integration that can create
        an order but cannot verify the webhook is worse than one that is plainly
        switched off, because it takes money and then cannot confirm it.
        """
        key_id = os.environ.get(ENV_KEY_ID, "").strip()
        key_secret = os.environ.get(ENV_KEY_SECRET, "").strip()
        webhook_secret = os.environ.get(ENV_WEBHOOK_SECRET, "").strip()
        if not (key_id and key_secret and webhook_secret):
            return None
        return cls(key_id, key_secret, webhook_secret)


def configured() -> bool:
    """Whether live calls are possible. The public form of the boundary."""
    return Credentials.from_env() is not None


# --- signatures -----------------------------------------------------------


def _hmac_hex(secret: str, message: bytes) -> str:
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


def verify_webhook(body: bytes, signature: str, webhook_secret: str) -> None:
    """Verify a webhook against the **raw** body. Raises `SignatureInvalid`.

    `body` must be the bytes as received. Passing a re-serialised dict is the
    usual way this fails in production and passes in testing: `json.dumps` of a
    parsed payload differs in whitespace and key order, so the HMAC differs and
    every real webhook is rejected.
    """
    expected = _hmac_hex(webhook_secret, body)
    if not hmac.compare_digest(expected, (signature or "").strip()):
        raise SignatureInvalid("webhook signature does not match the body")


def verify_checkout(order_id: str, payment_id: str, signature: str,
                    key_secret: str) -> None:
    """Verify a browser checkout callback. Raises `SignatureInvalid`.

    Note the different secret and the different message from `verify_webhook`:
    this one is `order_id|payment_id` keyed with the API key secret.
    """
    expected = _hmac_hex(key_secret, f"{order_id}|{payment_id}".encode())
    if not hmac.compare_digest(expected, (signature or "").strip()):
        raise SignatureInvalid("checkout signature does not match the order")


# --- orders ---------------------------------------------------------------


def order_payload(amount_paise: int, account: str, plan: str,
                  receipt: str) -> dict:
    """The body of a create-order call.

    `amount` is in **paise**, which is the single most consequential detail in
    this file. Razorpay takes the smallest currency unit; sending rupees charges
    a hundredth of the intended amount, and sending a float is rejected or
    silently truncated. It is an int here and the type is checked, because the
    failure is financial and silent in one direction.

    `notes` carries the account and plan so the webhook can credit the right
    ledger without a second lookup, and so a human reading the Razorpay
    dashboard can tell what a payment was for.
    """
    if not isinstance(amount_paise, int) or isinstance(amount_paise, bool):
        raise TypeError("amount_paise must be an int in paise, not rupees")
    if amount_paise <= 0:
        raise ValueError("amount_paise must be positive")
    return {
        "amount": amount_paise,
        "currency": "INR",
        "receipt": receipt,
        # Razorpay retries a create-order with the same receipt as a new order
        # unless told otherwise; capture is automatic so the money is ours as
        # soon as the customer completes, rather than sitting authorized.
        "payment_capture": 1,
        "notes": {"account": account, "plan": plan},
    }


def create_order(amount_paise: int, account: str, plan: str, receipt: str,
                 timeout: float = 15.0) -> dict:
    """Create a Razorpay order. Raises `CredentialsMissing` when unconfigured.

    Written with `urllib` rather than a client library so that reaching this
    boundary adds no dependency — and no dependency means no licence audit to
    re-run for a feature that cannot be switched on yet.
    """
    credentials = Credentials.from_env()
    if credentials is None:
        raise CredentialsMissing(
            "Razorpay is not configured. Set "
            f"{ENV_KEY_ID}, {ENV_KEY_SECRET} and {ENV_WEBHOOK_SECRET} "
            "from the Razorpay dashboard (Settings -> API Keys, and "
            "Settings -> Webhooks for the third). Until then the ledger works "
            "and payments do not."
        )

    payload = order_payload(amount_paise, account, plan, receipt)
    auth = b64encode(f"{credentials.key_id}:{credentials.key_secret}".encode()).decode()
    request = urllib.request.Request(
        f"{API_BASE}/orders",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        # Razorpay's errors are JSON and say what is wrong; passing the status
        # through without the body would turn a fixable "amount too small" into
        # an opaque 400.
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"Razorpay rejected the order ({exc.code}): {detail}") from exc


# --- webhook interpretation ----------------------------------------------


@dataclass(frozen=True)
class PaymentEvent:
    """A verified webhook, reduced to what the ledger needs."""

    event: str
    payment_id: str
    account: str | None
    plan: str | None
    amount_paise: int
    credits: bool
    reverses: bool

    @property
    def reference(self) -> str:
        """The idempotency key for the ledger.

        Razorpay retries a webhook until it gets a 2xx, so the same payment
        arrives more than once as a matter of course rather than as an error.
        The payment id is stable across those retries, and the ledger's unique
        index on `reference` is what makes a repeat a no-op.
        """
        return f"razorpay:{self.event}:{self.payment_id}"


def parse_event(body: bytes) -> PaymentEvent:
    """Interpret a webhook body that has **already been verified**.

    Kept separate from verification so that the order cannot be got wrong: this
    function parses, and parsing untrusted input before checking its signature is
    the mistake the split exists to prevent.
    """
    payload = json.loads(body)
    event = payload.get("event", "")
    entity = (
        payload.get("payload", {}).get("payment", {}).get("entity", {})
        or payload.get("payload", {}).get("refund", {}).get("entity", {})
    )
    notes = entity.get("notes") or {}
    return PaymentEvent(
        event=event,
        payment_id=entity.get("id", ""),
        account=notes.get("account"),
        plan=notes.get("plan"),
        amount_paise=int(entity.get("amount") or 0),
        credits=event in CREDITING_EVENTS,
        reverses=event in REVERSING_EVENTS,
    )
