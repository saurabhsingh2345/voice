"""The gate between a laptop and the open internet.

`voice-web` was written for `127.0.0.1`, where every endpoint is safe because
the only caller is the person who started it. A tunnel removes that assumption
in one step, and the surface includes `DELETE /api/data`. So these tests are
mostly about what must *not* be reachable.

Deny-by-default, not a blocklist: a blocklist is wrong the moment an endpoint is
added, and the new endpoint is exactly the one nobody remembers to add.
"""

from __future__ import annotations

import pytest

from voiceagent.web import public
from voiceagent.web.public import PUBLIC_ROUTES, RateLimiter, allowed_origins, client_ip


@pytest.fixture()
def public_mode(monkeypatch):
    monkeypatch.setenv(public.ENV_PUBLIC, "1")
    yield


# --- the allowlist ---------------------------------------------------------


def test_destructive_routes_are_not_public():
    """The specific reason this module exists: anyone with the link could
    otherwise delete the voice library."""
    paths = {path for _method, path in PUBLIC_ROUTES}
    methods = {method for method, _path in PUBLIC_ROUTES}
    assert "DELETE" not in methods
    assert "/api/data" not in paths
    assert not any("dataset" in p for p in paths)


def test_the_allowlist_is_only_what_the_studio_needs():
    assert PUBLIC_ROUTES == frozenset(
        {
            ("GET", "/api/config"),
            ("GET", "/api/queue"),
            ("GET", "/api/voices"),
            ("POST", "/api/speak"),
        }
    )


def test_public_mode_is_off_unless_asked_for(monkeypatch):
    """Running locally must be unchanged, or the gate becomes something people
    disable rather than something they rely on."""
    monkeypatch.delenv(public.ENV_PUBLIC, raising=False)
    assert not public.is_public()


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_public_mode_accepts_the_obvious_spellings(monkeypatch, value):
    monkeypatch.setenv(public.ENV_PUBLIC, value)
    assert public.is_public()


# --- CORS ------------------------------------------------------------------


def test_origins_default_to_localhost_only(monkeypatch):
    """A wildcard would let any page on the internet spend this machine's
    capacity through a visitor's browser."""
    monkeypatch.delenv(public.ENV_ORIGINS, raising=False)
    assert all("localhost" in o or "127.0.0.1" in o for o in allowed_origins())
    assert "*" not in allowed_origins()


def test_origins_are_read_from_the_environment(monkeypatch):
    monkeypatch.setenv(public.ENV_ORIGINS, "https://swar.vercel.app, https://swar.in")
    assert allowed_origins() == ["https://swar.vercel.app", "https://swar.in"]


# --- rate limiting ---------------------------------------------------------


def test_an_ip_is_cut_off_after_its_hourly_allowance():
    limiter = RateLimiter(per_hour=3, daily_chars=10**6)
    for _ in range(3):
        assert limiter.check("1.2.3.4", 10) is None
        limiter.record("1.2.3.4", 10)
    assert limiter.check("1.2.3.4", 10) is not None


def test_one_noisy_address_does_not_block_everyone_else():
    limiter = RateLimiter(per_hour=2, daily_chars=10**6)
    for _ in range(2):
        limiter.record("1.2.3.4", 10)
    assert limiter.check("1.2.3.4", 10) is not None
    assert limiter.check("5.6.7.8", 10) is None


def test_the_hourly_window_slides():
    limiter = RateLimiter(per_hour=1, daily_chars=10**6)
    limiter.record("1.2.3.4", 10, now=1000.0)
    assert limiter.check("1.2.3.4", 10, now=1000.0) is not None
    assert limiter.check("1.2.3.4", 10, now=1000.0 + 3601) is None


def test_a_global_daily_cap_backstops_the_per_ip_limit():
    """A hundred addresses each politely under their own limit still adds up to
    a machine that is busy all night."""
    limiter = RateLimiter(per_hour=1000, daily_chars=100)
    limiter.record("1.1.1.1", 60)
    limiter.record("2.2.2.2", 30)
    assert limiter.check("3.3.3.3", 30) is not None


def test_the_daily_cap_resets():
    limiter = RateLimiter(per_hour=1000, daily_chars=100)
    limiter.record("1.1.1.1", 100, now=1000.0)
    assert limiter.check("2.2.2.2", 10, now=1000.0) is not None
    assert limiter.check("2.2.2.2", 10, now=1000.0 + 86_401) is None


def test_a_refusal_says_when_to_come_back_not_just_no():
    limiter = RateLimiter(per_hour=1, daily_chars=10**6)
    limiter.record("1.2.3.4", 10)
    message = limiter.check("1.2.3.4", 10)
    assert "later" in message.lower()


def test_an_oversized_request_is_refused_before_it_runs():
    """Checked against the incoming size, not counted after the fact: the point
    is to not spend the machine on it."""
    limiter = RateLimiter(per_hour=100, daily_chars=100)
    assert limiter.check("1.2.3.4", 500) is not None


# --- identifying the caller through a tunnel -------------------------------


class _Req:
    def __init__(self, headers, host="10.0.0.1"):
        self.headers = headers
        self.client = type("C", (), {"host": host})()


def test_the_forwarded_address_is_preferred():
    """A tunnel is a reverse proxy, so the socket address is the tunnel itself
    and would rate-limit every visitor as one person."""
    assert client_ip(_Req({"x-forwarded-for": "203.0.113.9, 10.0.0.1"})) == "203.0.113.9"


def test_it_falls_back_to_the_socket_address():
    assert client_ip(_Req({}, host="198.51.100.7")) == "198.51.100.7"


# --- the invited surface ---------------------------------------------------


def test_artist_routes_write_and_are_therefore_separate():
    """Enrolment and contribution consume disk and create consent records.
    That is fine for people we invited and unacceptable from an open URL, and
    the difference is not expressible as a rate limit."""
    from voiceagent.web.public import ARTIST_ROUTES

    assert ("POST", "/api/voices") in ARTIST_ROUTES
    assert ("POST", "/api/contribute") in ARTIST_ROUTES
    assert not (ARTIST_ROUTES & PUBLIC_ROUTES)


def test_artist_routes_are_shut_when_no_code_is_configured(monkeypatch):
    """An unset variable is the likeliest mistake, and the failure it would
    cause is anonymous voice enrolment on a public URL. So it fails closed."""
    from voiceagent.web.public import invite_code, is_artist_route

    monkeypatch.delenv(public.ENV_INVITE, raising=False)
    assert invite_code() == ""
    assert is_artist_route("POST", "/api/voices")


def test_dataset_progress_is_matched_by_prefix():
    """The path carries a profile id, so exact matching would miss it and the
    artist would see their own progress 404."""
    from voiceagent.web.public import is_artist_route

    assert is_artist_route("GET", "/api/dataset/abc123")
    assert not is_artist_route("DELETE", "/api/dataset/abc123")


def test_a_destructive_dataset_route_is_still_not_reachable():
    from voiceagent.web.public import ARTIST_ROUTES, is_artist_route

    assert not is_artist_route("DELETE", "/api/dataset/abc/clips/x")
    assert not any(m == "DELETE" for m, _ in ARTIST_ROUTES)
