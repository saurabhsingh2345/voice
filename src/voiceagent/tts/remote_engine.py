"""IndicF5 on another machine, behind the same TTSEngine interface.

This is arithmetic again, the same arithmetic that produced `TTSRouter`. The live
loop is ~5.5 GiB on an 18 GiB machine that usually has 1.6 GiB free. IndicF5 is
~1.4 GiB plus activations, and Chatterbox is another 0.68 GiB, so the router and
the web server both evict one to load the other. Meanwhile an 8 GiB M1 Air sits
next to it doing nothing.

Moving the *slow* engine onto the *other* machine removes the eviction rather
than scheduling around it: the Mac keeps the live loop resident and the Air holds
IndicF5 permanently loaded, which also deletes its 15-30 s reload every time the
language alternates.

WHY THIS IS SOUND HERE AND NOWHERE ELSE
    Indic synthesis is RTF ~3.4 -- three seconds of compute per second of audio.
    A LAN round trip is 1-5 ms and the reference clip is ~0.5 MB, so the network
    is roughly 0.1% of the request. The same trick applied to Kokoro (RTF ~0.1,
    280 ms to first audio) would be self-defeating, and applied to the LLM it
    would be worse: pipeline-sharding a model puts a network hop in the path of
    every generated token. Only move work that is already too slow to be
    interactive.

LICENCE CONSEQUENCE, STATED PLAINLY
    `models.audit_installed_packages()` walks the *current* venv. f5-tts is what
    drags in encodec (CC-BY-NC), Unidecode (GPL), frozendict and soxr, so
    installing the `indic` extra only on the service machine makes `voice-doctor`
    pass on this one. That is a real improvement -- the packaged desktop app
    stops depending on a non-commercial package -- and it is NOT a resolution:
    encodec is still non-commercial wherever it runs, so distributing the service
    commercially remains blocked. This isolates the violation to one optional
    component. Replacing IndicF5 with Indic Parler-TTS (Apache-2.0) is still the
    only actual fix.

PRIVACY CONSEQUENCE, ALSO STATED PLAINLY
    The reference clip is decrypted on this machine -- the Fernet key stays in
    this Keychain and the Air never sees it -- but the plaintext WAV then crosses
    the LAN. "Nothing leaves this machine" becomes "nothing leaves my hardware".
    That is a weaker claim and it is the product's central one, so the transport
    is not left open: every request carries a shared token, and `load()` refuses
    a non-loopback URL that is neither private nor https, so a misconfigured
    base URL cannot quietly ship someone's voice to the internet.
"""

from __future__ import annotations

import asyncio
import io
import ipaddress
import time
from collections.abc import AsyncIterator
from urllib.parse import urlparse

import numpy as np

from voiceagent.tts.base import AudioChunk, TTSEngine
from voiceagent.tts.chunker import SentenceChunker
from voiceagent.tts.indic_engine import IndicTTSEngine

#: Connect fast, read slow. These are deliberately lopsided: a sleeping or
#: unplugged Air should fail in seconds with a clear message, but a legitimate
#: narration at RTF 3.4 can genuinely take minutes and must not be cut off
#: mid-generation -- that would burn all the compute and return nothing.
CONNECT_TIMEOUT_S = 5.0
READ_TIMEOUT_S = 900.0

DEFAULT_URL_ENV = "VOICEAGENT_TTS_URL"
DEFAULT_TOKEN_ENV = "VOICEAGENT_TTS_TOKEN"


class RemoteTTSUnavailable(RuntimeError):
    """The TTS service could not be reached, or refused us."""


def _is_private(host: str) -> bool:
    """True for loopback and RFC1918/link-local addresses.

    Hostnames that are not literal IPs return False: we cannot resolve them
    without a DNS lookup at import-adjacent time, and guessing wrong in the
    permissive direction is what this check exists to prevent. Use the Air's IP
    or `.local` name over https if you need a hostname.
    """
    if host.endswith(".local"):
        return True
    try:
        return ipaddress.ip_address(host).is_private or ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class RemoteTTSEngine(TTSEngine):
    """Speaks by asking another machine to. Drop-in for IndicTTSEngine."""

    name = "indicf5-remote"

    #: Nothing to reclaim here, and evicting it would cost the service machine a
    #: reload. See TTSEngine.evictable and TTSRouter._activate.
    evictable = False

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        connect_timeout: float = CONNECT_TIMEOUT_S,
        read_timeout: float = READ_TIMEOUT_S,
    ) -> None:
        if not token:
            raise ValueError(
                "A shared token is required. The service refuses to start without "
                f"one, so an empty token here can only mean {DEFAULT_TOKEN_ENV} is "
                "unset on this machine."
            )
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self._cancelled = False
        self._ready = False
        self._remote: dict = {}
        self._raw_reference_text = ""

        #: Reference bookkeeping is delegated to a never-loaded IndicTTSEngine
        #: rather than reimplemented. That is deliberate: the 12-second rule
        #: (hand the audio over whole, trim the *transcript*) took a measured
        #: 88% -> 95% to find, and a second copy of it here would drift from the
        #: local path silently. Constructing the engine loads nothing -- its
        #: __init__ only assigns attributes, and f5_tts is imported inside
        #: load(), which is never called on this instance.
        self._ref = IndicTTSEngine()

    # --- lifecycle --------------------------------------------------------

    def load(self) -> None:
        """Check the service is up, and that the URL is one we may send voice to.

        Called by TTSRouter before first use, which is exactly where
        IndicTTSEngine would raise on a missing checkpoint -- so a
        misconfiguration surfaces at the same point in the same way, rather than
        as a confusing failure mid-synthesis.
        """
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"{self.base_url!r} is not an http(s) URL")
        if parsed.scheme == "http" and not _is_private(parsed.hostname or ""):
            raise ValueError(
                f"Refusing to send reference audio to {parsed.hostname!r} over plain "
                "HTTP: it is not a private or loopback address. A voice clip is "
                "biometric data. Use the machine's LAN IP (192.168.x.x), its "
                "`.local` name, or https."
            )

        import httpx

        try:
            response = httpx.get(
                f"{self.base_url}/health",
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=httpx.Timeout(self.connect_timeout, read=10.0),
            )
        except Exception as exc:  # noqa: BLE001
            raise RemoteTTSUnavailable(
                f"Cannot reach the TTS service at {self.base_url}: {exc}\n"
                "Is the other machine awake, on the same network, and running "
                "`uv run voice-tts-service --host 0.0.0.0`?"
            ) from exc

        if response.status_code == 401:
            raise RemoteTTSUnavailable(
                f"The TTS service at {self.base_url} rejected our token. "
                f"{DEFAULT_TOKEN_ENV} must be identical on both machines."
            )
        response.raise_for_status()
        self._remote = response.json()
        self._ready = True

    def unload(self) -> None:
        """Ask the service to drop its model, so router eviction still means something.

        Best-effort: the point of running remotely is that the far side has room
        to stay loaded, so failing to evict is not an error worth propagating. A
        dead service is discovered by load() on the next request anyway.
        """
        self._ready = False
        if not self.base_url:
            return
        try:
            import httpx

            httpx.post(
                f"{self.base_url}/tts/unload",
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=httpx.Timeout(self.connect_timeout, read=30.0),
            )
        except Exception:  # noqa: BLE001
            pass

    def cancel(self) -> None:
        """Stop yielding audio.

        Honest limit: a request already in flight on the other machine runs to
        completion -- HTTP gives us no way to interrupt f5-tts mid-generation.
        Cancellation therefore stops playback promptly but does not free the
        remote compute. This costs nothing in practice because Indic synthesis is
        type-and-listen; there is no live Hindi loop to barge in on.
        """
        self._cancelled = True

    # --- reference voice --------------------------------------------------

    def set_reference(self, audio: np.ndarray, text: str, sample_rate: int) -> None:
        """Keep the reference locally; send it untrimmed and let the service trim.

        The untrimmed transcript is what goes over the wire, and that is not an
        oversight. `IndicTTSEngine.set_reference` trims the transcript for a clip
        longer than 12 s, and the service calls it too -- so sending the already
        trimmed text would trim it twice, once against the original duration and
        again against the same duration, cutting roughly (12/N)^2 of the words.
        The output would come out long and slurred, which is precisely the failure
        the trimming exists to prevent, arrived at from the other direction.

        `_ref` is still fed the original so `reference_health()` reports the same
        warnings here as it would locally.
        """
        self._raw_reference_text = text.strip()
        self._ref.set_reference(audio, text, sample_rate)

    def reference_health(self) -> str | None:
        return self._ref.reference_health()

    @property
    def reference_text(self) -> str:
        """The transcript as supplied -- what is sent, not what the service uses."""
        return self._raw_reference_text

    def _reference_wav(self) -> bytes:
        import soundfile as sf

        if self._ref.reference_audio is None:
            raise RuntimeError(
                "IndicF5 is a voice-cloning model: call set_reference() with a "
                "consented clip and its transcript before synthesizing."
            )
        buffer = io.BytesIO()
        sf.write(
            buffer,
            self._ref.reference_audio,
            self._ref.reference_sample_rate,
            format="WAV",
            subtype="PCM_16",
        )
        return buffer.getvalue()

    # --- inference --------------------------------------------------------

    def _speak_blocking(self, text: str) -> tuple[np.ndarray, int]:
        import httpx
        import soundfile as sf

        payload = self._reference_wav()
        try:
            response = httpx.post(
                f"{self.base_url}/tts/speak",
                headers={"Authorization": f"Bearer {self.token}"},
                data={"text": text, "reference_text": self._raw_reference_text},
                files={"reference": ("reference.wav", payload, "audio/wav")},
                timeout=httpx.Timeout(self.connect_timeout, read=self.read_timeout),
            )
        except Exception as exc:  # noqa: BLE001
            raise RemoteTTSUnavailable(
                f"The TTS service at {self.base_url} stopped responding: {exc}"
            ) from exc

        if response.status_code >= 400:
            # Pass the far side's explanation through verbatim. Its guards say
            # things the user can act on ("only 1.2 GiB free on the service
            # machine"), and replacing that with a generic 502 would hide the one
            # useful sentence.
            detail = response.text
            try:
                detail = response.json().get("detail", detail)
            except Exception:  # noqa: BLE001
                pass
            raise RemoteTTSUnavailable(f"TTS service returned {response.status_code}: {detail}")

        samples, sample_rate = sf.read(io.BytesIO(response.content), dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        return np.asarray(samples, dtype=np.float32).reshape(-1), sample_rate

    async def synthesize(self, text: str, voice: str | None = None) -> AsyncIterator[AudioChunk]:
        """One request for the whole span; the service chunks it per sentence.

        Sentence chunking deliberately stays on the far side. It is not a
        formatting preference there -- it is the mitigation for the SIGSEGV in
        PyTorch's Metal shader library that killed the process on a five-batch
        narration. Splitting here as well would double the number of HTTP
        requests to no benefit, and would move that mitigation away from the
        machine it protects.
        """
        if not self._ready:
            self.load()

        self._cancelled = False
        started = time.perf_counter()
        samples, sample_rate = await asyncio.to_thread(self._speak_blocking, text)
        if self._cancelled or not samples.size:
            return
        yield AudioChunk(
            samples=samples,
            sample_rate=sample_rate,
            is_final=True,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    async def synthesize_stream(
        self, text_chunks: AsyncIterator[str], voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        """One request per sentence, so audio starts before the text is finished.

        Here the chunking *is* ours: the caller is feeding us a live token
        stream, and waiting for it to end before making a single request would
        throw away the entire point of streaming.
        """
        if not self._ready:
            self.load()

        self._cancelled = False
        chunker = SentenceChunker()
        started = time.perf_counter()
        first = True

        async def speak(sentence: str) -> AsyncIterator[AudioChunk]:
            nonlocal first
            samples, sample_rate = await asyncio.to_thread(self._speak_blocking, sentence)
            if self._cancelled or not samples.size:
                return
            latency = (time.perf_counter() - started) * 1000 if first else None
            first = False
            yield AudioChunk(samples=samples, sample_rate=sample_rate, latency_ms=latency)

        async for text in text_chunks:
            if self._cancelled:
                return
            for sentence in chunker.feed(text):
                async for chunk in speak(sentence):
                    yield chunk
        for sentence in chunker.flush():
            if self._cancelled:
                return
            async for chunk in speak(sentence):
                yield chunk

    @property
    def resident_bytes(self) -> int:
        """Zero, and that is the entire point of this class.

        The router's memory budget is a budget for *this* machine. Weights held
        on the Air do not compete with the live loop, so reporting anything else
        here would make the budget table lie.
        """
        return 0


# --- wiring ---------------------------------------------------------------


def from_env() -> RemoteTTSEngine | None:
    """Build the engine if this machine is configured to use a remote one.

    Returns None when unconfigured, so callers keep their local default and no
    behaviour changes for anyone who has not set this up.
    """
    import os

    url = os.environ.get(DEFAULT_URL_ENV, "").strip()
    if not url:
        return None
    token = os.environ.get(DEFAULT_TOKEN_ENV, "").strip()
    if not token:
        raise ValueError(
            f"{DEFAULT_URL_ENV} is set but {DEFAULT_TOKEN_ENV} is not. Both machines "
            "need the same token; the service will not start without it either."
        )
    return RemoteTTSEngine(url, token)
