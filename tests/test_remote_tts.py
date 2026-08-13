"""Properties of running Indic TTS on a second machine.

No model is loaded anywhere here. Every test either exercises a guard that
rejects before synthesis, or asserts a property of the client engine that holds
without a service being up.

The one that matters most is `test_the_transcript_is_trimmed_exactly_once`. Both
sides of this link call `IndicTTSEngine.set_reference`, which trims the reference
transcript for a clip longer than 12 s -- so the obvious implementation (client
trims, sends the trimmed text, service trims again) shortens it by (12/N)^2 and
produces exactly the rushed, syllable-swallowing output the trimming was written
to prevent. Nothing else in the system would report that as an error.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from voiceagent.tts.indic_engine import REFERENCE_CLIP_SECONDS, IndicTTSEngine
from voiceagent.tts.remote_engine import (
    DEFAULT_TOKEN_ENV,
    DEFAULT_URL_ENV,
    RemoteTTSEngine,
    from_env,
)
from voiceagent.web import tts_service as svc

TOKEN = "test-token-not-a-real-secret"


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setenv(svc.TOKEN_ENV, TOKEN)
    return TestClient(svc.app)


def wav_bytes(seconds: float = 6.0, sample_rate: int = 24_000) -> bytes:
    t = np.linspace(0, seconds, int(seconds * sample_rate), endpoint=False)
    tone = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, tone, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def tone(seconds: float, sample_rate: int = 24_000) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * sample_rate), endpoint=False)
    return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


# --- the double-trim hazard -----------------------------------------------


def test_the_transcript_is_trimmed_exactly_once():
    """The client must send the transcript it was given, not its trimmed copy.

    Simulates the full path for a 24 s clip: client sets the reference, the
    (untrimmed) text goes over the wire, the service applies the trim. The result
    has to match what a purely local engine would have used -- otherwise the two
    paths produce different audio from identical inputs.
    """
    seconds = REFERENCE_CLIP_SECONDS * 2
    audio = tone(seconds)
    words = [f"शब्द{i}" for i in range(80)]
    transcript = " ".join(words)

    local = IndicTTSEngine()
    local.set_reference(audio, transcript, 24_000)

    client = RemoteTTSEngine("http://127.0.0.1:8824", TOKEN)
    client.set_reference(audio, transcript, 24_000)

    # What actually crosses the wire.
    sent = client.reference_text
    assert sent == transcript, "the client must send the untrimmed transcript"

    # What the service then computes from it.
    service_side = IndicTTSEngine()
    service_side.set_reference(audio, sent, 24_000)

    assert service_side.reference_text == local.reference_text
    assert len(service_side.reference_text.split()) < len(words), "expected a trim to happen"


def test_reference_health_still_warns_on_the_client():
    """The user types the transcript on this machine, so the warning belongs here."""
    client = RemoteTTSEngine("http://127.0.0.1:8824", TOKEN)
    client.set_reference(tone(REFERENCE_CLIP_SECONDS * 2), "बहुत छोटा", 24_000)
    warning = client.reference_health()
    assert warning is not None and "12" in warning


# --- the client engine ----------------------------------------------------


def test_resident_bytes_is_zero():
    """The memory budget is a budget for this machine; remote weights are not on it."""
    assert RemoteTTSEngine("http://127.0.0.1:8824", TOKEN).resident_bytes == 0


def test_a_remote_engine_declares_itself_unevictable():
    """Declared, not inferred from resident_bytes -- a local engine may report zero
    before load or with no memory counter, and skipping *its* eviction would put
    the resident pipeline back over the ceiling."""
    from voiceagent.tts.indic_engine import IndicTTSEngine as Local

    assert RemoteTTSEngine("http://127.0.0.1:8824", TOKEN).evictable is False
    assert Local().evictable is True


def test_a_token_is_required_to_construct_one():
    with pytest.raises(ValueError):
        RemoteTTSEngine("http://127.0.0.1:8824", "")


def test_plain_http_to_a_public_address_is_refused():
    """A voice clip is biometric data; a mistyped host must not quietly ship it out."""
    engine = RemoteTTSEngine("http://example.com:8824", TOKEN)
    with pytest.raises(ValueError, match="private or loopback"):
        engine.load()


@pytest.mark.parametrize("url", ["http://127.0.0.1:8824", "http://192.168.1.42:8824",
                                "http://airbook.local:8824"])
def test_private_and_loopback_addresses_are_allowed(url):
    """load() must get past the address check and fail on the connection instead."""
    from voiceagent.tts.remote_engine import RemoteTTSUnavailable

    engine = RemoteTTSEngine(url, TOKEN)
    with pytest.raises(RemoteTTSUnavailable):
        engine.load()


def test_https_to_a_public_address_is_allowed():
    from voiceagent.tts.remote_engine import RemoteTTSUnavailable

    engine = RemoteTTSEngine("https://example.invalid", TOKEN)
    with pytest.raises(RemoteTTSUnavailable):
        engine.load()


# --- from_env -------------------------------------------------------------


def test_from_env_is_none_when_unconfigured(monkeypatch):
    """Nobody who has not set this up may see a behaviour change."""
    monkeypatch.delenv(DEFAULT_URL_ENV, raising=False)
    assert from_env() is None


def test_from_env_refuses_a_url_without_a_token(monkeypatch):
    monkeypatch.setenv(DEFAULT_URL_ENV, "http://192.168.1.42:8824")
    monkeypatch.delenv(DEFAULT_TOKEN_ENV, raising=False)
    with pytest.raises(ValueError, match=DEFAULT_TOKEN_ENV):
        from_env()


def test_from_env_builds_an_engine_when_both_are_set(monkeypatch):
    monkeypatch.setenv(DEFAULT_URL_ENV, "http://192.168.1.42:8824/")
    monkeypatch.setenv(DEFAULT_TOKEN_ENV, TOKEN)
    engine = from_env()
    assert isinstance(engine, RemoteTTSEngine)
    assert engine.base_url == "http://192.168.1.42:8824", "trailing slash should be stripped"


# --- the service: authentication ------------------------------------------


def test_health_without_a_token_is_401(service):
    assert service.get("/health").status_code == 401


def test_health_with_the_token_is_200(service):
    r = service.get("/health", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200
    assert r.json()["engine"] == "indicf5"


def test_a_wrong_token_is_401(service):
    r = service.get("/health", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_speak_without_a_token_is_refused_before_anything_else(service):
    """Authentication precedes validation: an unauthenticated caller must not be
    able to probe the service's limits, and must never reach a model load."""
    r = service.post(
        "/tts/speak",
        data={"text": "नमस्ते।", "reference_text": "नमस्ते"},
        files={"reference": ("r.wav", wav_bytes(), "audio/wav")},
    )
    assert r.status_code == 401


def test_the_service_refuses_to_serve_with_no_token_configured(monkeypatch):
    """Fails closed, like the tool confirmation gate: no token, no service."""
    monkeypatch.delenv(svc.TOKEN_ENV, raising=False)
    client = TestClient(svc.app)
    r = client.get("/health", headers={"Authorization": "Bearer anything"})
    assert r.status_code == 503


def test_main_exits_nonzero_without_a_token(monkeypatch, capsys):
    monkeypatch.delenv(svc.TOKEN_ENV, raising=False)
    monkeypatch.setattr("sys.argv", ["voice-tts-service"])
    assert svc.main() == 1
    assert svc.TOKEN_ENV in capsys.readouterr().out


# --- the service: guards --------------------------------------------------


def auth() -> dict:
    return {"Authorization": f"Bearer {TOKEN}"}


def test_oversized_text_is_refused(service):
    r = service.post(
        "/tts/speak",
        headers=auth(),
        data={"text": "यह एक परीक्षण वाक्य है। " * 200, "reference_text": "नमस्ते"},
        files={"reference": ("r.wav", wav_bytes(), "audio/wav")},
    )
    assert r.status_code == 413


def test_empty_text_is_refused(service):
    r = service.post(
        "/tts/speak",
        headers=auth(),
        data={"text": "   ", "reference_text": "नमस्ते"},
        files={"reference": ("r.wav", wav_bytes(), "audio/wav")},
    )
    assert r.status_code == 400


def test_a_missing_reference_transcript_is_refused(service):
    """IndicF5 estimates output length from the reference transcript, so an empty
    one is not a degraded clone -- it is wrong-length audio."""
    r = service.post(
        "/tts/speak",
        headers=auth(),
        data={"text": "नमस्ते।", "reference_text": "  "},
        files={"reference": ("r.wav", wav_bytes(), "audio/wav")},
    )
    assert r.status_code == 400
    assert "reference_text" in r.json()["detail"]


def test_undecodable_reference_audio_is_refused(service):
    r = service.post(
        "/tts/speak",
        headers=auth(),
        data={"text": "नमस्ते।", "reference_text": "नमस्ते"},
        files={"reference": ("r.wav", b"not audio", "audio/wav")},
    )
    assert r.status_code == 400


def test_a_second_request_is_refused_while_one_is_running(service):
    """Refused, not queued -- the same death spiral as the web UI, on a smaller
    machine where it bites sooner."""
    import asyncio

    async def hold():
        async with svc._synth_lock:
            return service.post(
                "/tts/speak",
                headers=auth(),
                data={"text": "नमस्ते।", "reference_text": "नमस्ते"},
                files={"reference": ("r.wav", wav_bytes(), "audio/wav")},
            )

    r = asyncio.run(hold())
    assert r.status_code == 429


def test_preload_requires_a_token(service):
    """It loads 1.4 GiB on an 8 GiB machine; an unauthenticated caller must not
    be able to do that, let alone repeatedly."""
    assert service.post("/tts/preload").status_code == 401


def test_preload_is_refused_while_synthesizing(service):
    import asyncio

    async def hold():
        async with svc._synth_lock:
            return service.post("/tts/preload", headers=auth())

    assert asyncio.run(hold()).status_code == 429


def test_unload_is_refused_while_synthesizing(service):
    """Pulling the model out from under a running generation is not a valid answer
    to the client's router evicting at an awkward moment."""
    import asyncio

    async def hold():
        async with svc._synth_lock:
            return service.post("/tts/unload", headers=auth())

    assert asyncio.run(hold()).status_code == 429


# --- the memory guard scales on the longest sentence ----------------------


def test_memory_requirement_scales_on_the_longest_sentence_not_the_total():
    """Because synthesis is now one f5_tts call per sentence, peak memory tracks
    the longest sentence. Scaling on total length would demand ~17 GiB for a
    3000-character narration and refuse every real request on an 8 GiB machine,
    while measuring an allocation no single call makes."""
    many_short = "नमस्ते। " * 300
    one_long = "क " * 700 + "।"

    assert svc._required_gib(many_short) == svc.MIN_FREE_GIB_FOR_INDIC
    assert svc._required_gib(one_long) > svc.MIN_FREE_GIB_FOR_INDIC


def test_the_floor_is_above_the_model_size():
    assert svc.MIN_FREE_GIB_FOR_INDIC >= 2.0


# --- router integration ---------------------------------------------------


def test_the_router_does_not_evict_a_remote_engine():
    """Eviction reclaims local memory. A remote engine holds none, so evicting it
    only costs the service machine a reload -- which is the cost this whole
    arrangement exists to remove."""
    from voiceagent.tts.router import Route, TTSRouter

    class Fake(RemoteTTSEngine):
        def __init__(self) -> None:
            super().__init__("http://127.0.0.1:8824", TOKEN)
            self.unloaded = False

        def load(self) -> None:
            self._ready = True

        def unload(self) -> None:
            self.unloaded = True

    remote = Fake()
    local_loads = []

    class Local(IndicTTSEngine):
        def load(self) -> None:
            local_loads.append(1)
            self._peak_bytes = 1_000_000

        def unload(self) -> None:
            pass

    hindi = Route(languages=frozenset({"hi"}), factory=lambda: remote, label="indic")
    english = Route(languages=frozenset({"en"}), factory=Local, label="kokoro")
    router = TTSRouter(routes=[hindi, english])

    router._activate(hindi)
    router._activate(english)  # switching away must not unload the remote
    assert not remote.unloaded
    assert hindi._engine is remote, "the remote engine should still be attached"

    router._activate(hindi)
    assert len(local_loads) == 1, "and it should not need reloading"


def test_an_engine_that_does_not_declare_the_flag_is_still_evicted():
    """The default must be the safe direction. Engines are duck-typed here as well
    as subclassed, and an undeclared backend silently surviving eviction would put
    two models resident on an 18 GiB machine."""
    from voiceagent.tts.router import Route, TTSRouter

    class Undeclared:
        def __init__(self) -> None:
            self.unloaded = False

        def load(self) -> None: ...

        def unload(self) -> None:
            self.unloaded = True

    first = Undeclared()
    hindi = Route(languages=frozenset({"hi"}), factory=lambda: first, label="indic")
    english = Route(languages=frozenset({"en"}), factory=Undeclared, label="kokoro")
    router = TTSRouter(routes=[hindi, english])

    router._activate(hindi)
    router._activate(english)
    assert first.unloaded, "an engine with no evictable flag must still be evicted"


def test_a_local_engine_is_still_evicted():
    """The rule above must not accidentally disable eviction for local engines --
    that is what keeps the resident pipeline under the memory ceiling."""
    from voiceagent.tts.router import Route, TTSRouter

    class Heavy(IndicTTSEngine):
        def __init__(self) -> None:
            super().__init__()
            self.unloaded = False

        def load(self) -> None:
            self._peak_bytes = 1_400_000_000

        def unload(self) -> None:
            self.unloaded = True

    heavy = Heavy()
    hindi = Route(languages=frozenset({"hi"}), factory=lambda: heavy, label="indic")
    english = Route(languages=frozenset({"en"}), factory=Heavy, label="kokoro")
    router = TTSRouter(routes=[hindi, english])

    router._activate(hindi)
    router._activate(english)
    assert heavy.unloaded
