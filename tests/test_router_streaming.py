"""Routing a live token stream, which is what made the loop bilingual.

`orchestration/loop.py` held a `KokoroEngine` directly, and that was the whole of
what kept the live loop English-only --- Devanagari sent to Kokoro produces
nothing usable, so Hindi had to be typed and listened to elsewhere. The loop now
holds the router, which needs a `synthesize_stream` and a `cancel` it did not
have.

Fakes throughout. The decisions here are about *which* engine gets *what text*,
and no audio is needed to pin that down.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from voiceagent.tts.base import AudioChunk, TTSEngine
from voiceagent.tts.router import Route, TTSRouter


class FakeEngine(TTSEngine):
    def __init__(self, name: str) -> None:
        self.name = name
        self.loaded = False
        self.cancelled = False
        self.received: list[str] = []

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def cancel(self) -> None:
        self.cancelled = True

    async def synthesize(self, text, voice=None):
        self.received.append(text)
        yield AudioChunk(samples=np.zeros(4, dtype=np.float32), sample_rate=24_000)

    async def synthesize_stream(self, text_chunks, voice=None):
        async for text in text_chunks:
            self.received.append(text)
            yield AudioChunk(samples=np.zeros(4, dtype=np.float32), sample_rate=24_000)

    @property
    def resident_bytes(self) -> int:
        return 0


@pytest.fixture
def router():
    engines = {"indic": FakeEngine("indic"), "kokoro": FakeEngine("kokoro")}
    r = TTSRouter(
        routes=[
            Route(
                languages=frozenset({"hi", "bn", "ta"}),
                factory=lambda: engines["indic"],
                normalizer=lambda t: f"<norm>{t}",
                label="indic",
            ),
            Route(languages=frozenset({"en"}), factory=lambda: engines["kokoro"], label="kokoro"),
        ]
    )
    r.engines = engines
    return r


def drain(router, tokens, chunk_size=None):
    async def source():
        for t in tokens:
            yield t

    async def go():
        return [c async for c in router.synthesize_stream(source())]

    return asyncio.run(go())


# --- which engine gets the stream ------------------------------------------


def test_a_hindi_stream_reaches_the_indic_engine(router):
    drain(router, ["आज ", "मौसम ", "बहुत ", "अच्छा है। ", "धूप निकली है।"])
    assert router.engines["indic"].received
    assert not router.engines["kokoro"].received


def test_an_english_stream_reaches_kokoro(router):
    drain(router, ["the ", "weather ", "is ", "lovely today. ", "Very sunny."])
    assert router.engines["kokoro"].received
    assert not router.engines["indic"].received


def test_the_language_is_decided_on_the_first_sentence_not_the_first_token(router):
    """A Hindi reply can easily open on an English loanword. Deciding per token
    would answer the whole reply in the wrong voice."""
    drain(router, ["Report ", "तैयार ", "है, ", "देख लीजिए। ", "बाकी कल।"])
    assert router.engines["indic"].received
    assert not router.engines["kokoro"].received


def test_a_reply_shorter_than_one_sentence_still_speaks(router):
    """No terminal punctuation anywhere; the flush has to decide."""
    drain(router, ["बिल्कुल"])
    assert router.engines["indic"].received


def test_an_empty_stream_yields_nothing_and_loads_nothing(router):
    assert drain(router, []) == []
    assert not router.engines["indic"].loaded
    assert not router.engines["kokoro"].loaded


def test_the_engine_is_not_reconsidered_mid_reply(router):
    """A reply drifting Hindi -> English must not evict and reload a model
    mid-sentence: the router holds one engine at a time, so that is seconds of
    silence in the middle of a spoken answer. The first sentence wins."""
    drain(router, ["आज मौसम अच्छा है। ", "The weather is lovely. ", "Very sunny indeed."])
    assert router.engines["indic"].received
    assert not router.engines["kokoro"].received


# --- what text the engine gets ---------------------------------------------


def test_the_normalizer_runs_per_sentence(router):
    """normalize_hi transliterates Latin to Devanagari and spells digits out;
    both need whole words. Applied per token it would corrupt the very sequences
    it exists to handle."""
    drain(router, ["आज ", "मौसम ", "अच्छा है। ", "कल बारिश होगी।"])
    got = router.engines["indic"].received
    assert all(t.startswith("<norm>") for t in got)
    assert all("<norm>" not in t[len("<norm>") :] for t in got), "normalized once, not per token"


def test_no_normalizer_means_text_passes_through(router):
    drain(router, ["the weather is lovely. ", "Very sunny."])
    assert all("<norm>" not in t for t in router.engines["kokoro"].received)


def test_nothing_is_dropped_between_detection_and_delegation(router):
    """The sentences buffered while deciding the language have to be replayed,
    not discarded -- otherwise the reply starts from its second sentence."""
    drain(router, ["एक। ", "दो। ", "तीन।"])
    joined = " ".join(router.engines["indic"].received)
    for part in ("एक", "दो", "तीन"):
        assert part in joined


# --- barge-in ---------------------------------------------------------------


def test_cancel_reaches_a_built_engine(router):
    drain(router, ["आज मौसम अच्छा है।"])
    router.cancel()
    assert router.engines["indic"].cancelled


def test_cancel_before_anything_is_built_does_not_raise(router):
    router.cancel()


def test_cancel_reaches_every_engine_that_is_still_built():
    """With keep_resident, both engines stay loaded and either could be the one
    speaking, so cancel has to reach both.

    Without it there is only ever one: switching languages evicts the previous
    engine, and an evicted engine has nothing to cancel. That asymmetry is why
    this iterates the routes rather than touching `_active`.
    """
    engines = {"indic": FakeEngine("indic"), "kokoro": FakeEngine("kokoro")}
    r = TTSRouter(
        routes=[
            Route(languages=frozenset({"hi"}), factory=lambda: engines["indic"], label="indic"),
            Route(languages=frozenset({"en"}), factory=lambda: engines["kokoro"], label="kokoro"),
        ],
        keep_resident=True,
    )
    r.engines = engines
    drain(r, ["आज मौसम अच्छा है।"])
    drain(r, ["the weather is lovely."])
    r.cancel()
    assert engines["indic"].cancelled and engines["kokoro"].cancelled


def test_switching_language_evicts_the_previous_engine(router):
    """One engine at a time is the memory contract this whole router exists for:
    Kokoro and Chatterbox Multilingual do not fit alongside the LLM together."""
    drain(router, ["आज मौसम अच्छा है।"])
    drain(router, ["the weather is lovely."])
    assert not router.engines["indic"].loaded
    assert router.engines["kokoro"].loaded


# --- the loop actually uses it ----------------------------------------------


def test_the_live_loop_holds_the_router_not_a_bare_engine():
    """This is the change. Asserted on the source because constructing the loop
    loads several GB of weights."""
    import inspect

    from voiceagent.orchestration import loop as loop_module

    source = inspect.getsource(loop_module.VoiceLoop.load)
    assert "build_default_router()" in source
    assert "KokoroEngine()" not in source
