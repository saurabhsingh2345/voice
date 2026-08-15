"""What the internet is allowed to reach when this server is behind a tunnel.

`voice-web` grew as a local developer tool on `127.0.0.1`, where every endpoint
is safe because the only caller is the person who started it. Putting it behind
a tunnel changes that in one step, and the surface it exposes includes:

    DELETE /api/data                      -- wipes everything
    DELETE /api/voices/{profile_id}       -- destroys a consented voice profile
    DELETE /api/dataset/{profile_id}      -- destroys a recording set
    POST   /api/dataset/{id}/export       -- writes files
    POST   /api/voices/{id}/transcribe    -- loads Whisper on demand

Anyone with the URL could delete the voice library. So public mode is
**deny-by-default with a short allowlist**, rather than a list of things to
block: a blocklist is wrong the moment a new endpoint is added, and the new
endpoint is exactly the one nobody remembers to add to it.

Off unless `VOICEAGENT_PUBLIC` is set, so running locally is unchanged.

**Blocked routes answer 404, not 403.** A 403 confirms the endpoint exists,
which turns a scan into a map. Locally they behave normally.

This is a gate, not authentication. It assumes the tunnel URL is semi-public and
that the worst a caller can do is generate speech and use up the rate limit. The
moment there are accounts or money, this stops being sufficient --- but it is
what makes a shareable link safe today.
"""

from __future__ import annotations

import os
import time
from collections import deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

ENV_PUBLIC = "VOICEAGENT_PUBLIC"
ENV_ORIGINS = "VOICEAGENT_ALLOWED_ORIGINS"
ENV_RATE_PER_HOUR = "VOICEAGENT_RATE_PER_HOUR"
ENV_DAILY_CHARS = "VOICEAGENT_DAILY_CHARS"

#: Everything the studio needs and nothing else. Read-only apart from the one
#: endpoint that is the product.
PUBLIC_ROUTES: frozenset[tuple[str, str]] = frozenset(
    {
        ("GET", "/api/config"),
        ("GET", "/api/queue"),
        ("GET", "/api/voices"),
        ("POST", "/api/speak"),
    }
)

#: Generations per IP per hour. The machine serves one at a time and a tester
#: exploring for ten minutes will not approach this; a script will hit it in
#: seconds. Low enough to protect a single Mac, high enough not to interrupt
#: the people the link was sent to.
DEFAULT_RATE_PER_HOUR = 40

#: Total characters synthesized per day across everyone, as a backstop the
#: per-IP limit cannot give: a hundred IPs each politely under their own limit
#: still adds up to a machine that is busy all night. Roughly two hours of
#: audio, which is more than a day of honest testing.
DEFAULT_DAILY_CHARS = 120_000


def is_public() -> bool:
    return os.environ.get(ENV_PUBLIC, "").strip().lower() in {"1", "true", "yes", "on"}


def allowed_origins() -> list[str]:
    """Origins permitted to call this server from a browser.

    Defaults to localhost only. The deployed frontend's origin has to be named
    explicitly --- a wildcard would let any page on the internet spend this
    machine's capacity through a visitor's browser, which is the specific thing
    CORS exists to prevent.
    """
    raw = os.environ.get(ENV_ORIGINS, "").strip()
    if not raw:
        return ["http://localhost:3000", "http://127.0.0.1:3000"]
    return [o.strip() for o in raw.split(",") if o.strip()]


def _int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, "").strip())
    except ValueError:
        return default
    return value if value > 0 else default


class RateLimiter:
    """Per-IP generation counts and a global daily character cap.

    In memory and per process, which is the right size for one Mac behind one
    tunnel. It resets on restart; that is a real limitation and an acceptable
    one, because the thing it protects against is a script hammering for
    minutes, not a determined adversary with patience.
    """

    def __init__(
        self,
        per_hour: int | None = None,
        daily_chars: int | None = None,
    ) -> None:
        self.per_hour = per_hour or _int_env(ENV_RATE_PER_HOUR, DEFAULT_RATE_PER_HOUR)
        self.daily_chars = daily_chars or _int_env(ENV_DAILY_CHARS, DEFAULT_DAILY_CHARS)
        self._hits: dict[str, deque[float]] = {}
        self._chars_today = 0
        #: The day is a *bucket* of wall-clock time, not "24h since this object
        #: was built". Measuring elapsed-since-start drifts the reset away from
        #: the actual day on a server left running, so a cap called "daily"
        #: would quietly reset at 3am on the fourth day.
        self._day: int | None = None

    def _prune(self, ip: str, now: float) -> deque[float]:
        window = self._hits.setdefault(ip, deque())
        cutoff = now - 3600
        while window and window[0] < cutoff:
            window.popleft()
        return window

    def _roll_day(self, now: float) -> None:
        today = int(now // 86_400)
        if self._day != today:
            self._chars_today = 0
            self._day = today

    def check(self, ip: str, characters: int, now: float | None = None) -> str | None:
        """Return a refusal message, or None to allow.

        Checked *before* the work rather than counted after, because the point
        is to not spend the machine on it.
        """
        now = now if now is not None else time.time()
        self._roll_day(now)

        if self._chars_today + characters > self.daily_chars:
            return (
                "This preview has a daily limit on how much audio it generates, "
                "and today's is used up. It resets in a few hours."
            )

        window = self._prune(ip, now)
        if len(window) >= self.per_hour:
            return (
                f"That is {self.per_hour} generations in an hour from this address, "
                "which is this preview's limit. Try again a little later."
            )
        return None

    def record(self, ip: str, characters: int, now: float | None = None) -> None:
        now = now if now is not None else time.time()
        self._roll_day(now)
        self._prune(ip, now).append(now)
        self._chars_today += max(0, characters)

    def snapshot(self) -> dict:
        return {
            "per_hour": self.per_hour,
            "daily_chars": self.daily_chars,
            "chars_today": self._chars_today,
        }


class PublicSurface(BaseHTTPMiddleware):
    """Deny-by-default routing when `VOICEAGENT_PUBLIC` is set."""

    async def dispatch(self, request, call_next):
        if not is_public():
            return await call_next(request)

        #: Preflight must pass or every cross-origin call fails before the
        #: allowlist is ever consulted.
        if request.method == "OPTIONS":
            return await call_next(request)

        if (request.method, request.url.path) not in PUBLIC_ROUTES:
            return JSONResponse({"detail": "Not Found"}, status_code=404)

        return await call_next(request)


def client_ip(request) -> str:
    """The caller's address, trusting one proxy hop.

    A tunnel is a reverse proxy, so `request.client.host` is the tunnel itself
    and would rate-limit every visitor as one person. The first entry of
    `X-Forwarded-For` is the original client. Spoofable in general --- which is
    fine here, because this limit protects a machine from load rather than
    guarding anything of value.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
