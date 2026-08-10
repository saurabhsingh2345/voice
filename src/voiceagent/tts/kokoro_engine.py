"""Kokoro-82M text-to-speech on MLX.

LICENSE-CRITICAL: Kokoro's text frontend (misaki) can fall back to `phonemizer`
for out-of-dictionary words, and phonemizer is GPLv3 -- viral copyleft that
would infect a distributed desktop app. misaki itself makes that fallback
optional (`en.G2P(fallback=None)`); only mlx-audio's pipeline imports it
unconditionally. We therefore install a stub so the fallback is refused and the
GPL package need never be present. See `_disable_gpl_espeak_fallback`.

The cost of that choice is that words missing from misaki's dictionary are
skipped rather than guessed. `check_coverage()` exists to measure how often
that actually happens on the text this agent produces.
"""

from __future__ import annotations

import asyncio
import sys
import time
import types
from collections.abc import AsyncIterator

import numpy as np

from voiceagent.tts.base import AudioChunk, TTSEngine
from voiceagent.tts.chunker import SentenceChunker

DEFAULT_REPO = "mlx-community/Kokoro-82M-bf16"
DEFAULT_VOICE = "af_heart"

#: Kokoro synthesizes at 24 kHz.
SAMPLE_RATE = 24_000


class EspeakFallbackDisabled(RuntimeError):
    """Raised (and caught by mlx-audio) to refuse the GPL phonemizer path."""


def _disable_gpl_espeak_fallback() -> None:
    """Make `misaki.espeak` importable but non-functional.

    mlx-audio calls `_get_misaki_espeak()` outside the try/except that tolerates
    a missing fallback, so a plain ImportError aborts pipeline construction.
    Registering a stub lets the import succeed and pushes the failure into
    `EspeakFallback(...)`, which mlx-audio already handles by setting
    `fallback=None` -- exactly the configuration we want.
    """
    if "misaki.espeak" in sys.modules:
        return

    stub = types.ModuleType("misaki.espeak")

    def _refuse(*args, **kwargs):
        raise EspeakFallbackDisabled(
            "espeak/phonemizer fallback is disabled: phonemizer is GPLv3 and "
            "this project ships under permissive licenses only."
        )

    stub.EspeakFallback = _refuse
    stub.EspeakG2P = _refuse
    sys.modules["misaki.espeak"] = stub


class KokoroEngine(TTSEngine):
    name = "kokoro-82m"

    def __init__(
        self,
        repo: str = DEFAULT_REPO,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        lang_code: str = "a",
    ) -> None:
        self.repo = repo
        self.voice = voice
        self.speed = speed
        self.lang_code = lang_code
        self._model = None
        self._peak_bytes = 0
        self._cancelled = False

    # --- lifecycle --------------------------------------------------------

    def load(self) -> None:
        import mlx.core as mx

        _disable_gpl_espeak_fallback()
        from mlx_audio.tts.utils import load_model

        mx.reset_peak_memory()
        self._model = load_model(self.repo)
        # First synthesis compiles Metal kernels; pay that now, not on the
        # user's first spoken reply.
        for _ in self._model.generate(
            text="Ready.", voice=self.voice, speed=self.speed, lang_code=self.lang_code
        ):
            pass
        self._peak_bytes = mx.get_peak_memory()

    def unload(self) -> None:
        import gc

        import mlx.core as mx

        self._model = None
        gc.collect()
        mx.clear_cache()

    # --- inference --------------------------------------------------------

    def cancel(self) -> None:
        """Stop synthesis at the next chunk boundary (Phase 4 barge-in)."""
        self._cancelled = True

    def _generate_blocking(self, text: str, voice: str | None) -> list[np.ndarray]:
        segments = []
        for result in self._model.generate(
            text=text,
            voice=voice or self.voice,
            speed=self.speed,
            lang_code=self.lang_code,
        ):
            audio = result.audio
            segments.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        return segments

    async def synthesize(self, text: str, voice: str | None = None) -> AsyncIterator[AudioChunk]:
        if self._model is None:
            raise RuntimeError("load() must be called before synthesize()")

        self._cancelled = False
        started = time.perf_counter()
        first = True

        segments = await asyncio.to_thread(self._generate_blocking, text, voice)
        for index, samples in enumerate(segments):
            if self._cancelled:
                return
            latency = (time.perf_counter() - started) * 1000 if first else None
            first = False
            yield AudioChunk(
                samples=samples,
                sample_rate=SAMPLE_RATE,
                is_final=index == len(segments) - 1,
                latency_ms=latency,
            )

    async def synthesize_stream(
        self, text_chunks: AsyncIterator[str], voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        """Synthesize sentence-by-sentence as the LLM writes them.

        Waiting for a full response before speaking would add the entire
        generation time to perceived latency, so each sentence is synthesized
        the moment it is complete.
        """
        if self._model is None:
            raise RuntimeError("load() must be called before synthesize_stream()")

        self._cancelled = False
        chunker = SentenceChunker()
        started = time.perf_counter()
        first = True

        async def speak(sentence: str) -> AsyncIterator[AudioChunk]:
            nonlocal first
            segments = await asyncio.to_thread(self._generate_blocking, sentence, voice)
            for samples in segments:
                if self._cancelled:
                    return
                latency = (time.perf_counter() - started) * 1000 if first else None
                first = False
                yield AudioChunk(samples=samples, sample_rate=SAMPLE_RATE, latency_ms=latency)

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

        yield AudioChunk(
            samples=np.zeros(0, dtype=np.float32), sample_rate=SAMPLE_RATE, is_final=True
        )

    # --- diagnostics ------------------------------------------------------

    def check_coverage(self, text: str) -> list[str]:
        """Return words misaki cannot pronounce without the espeak fallback.

        These are silently dropped from the audio, so this is how we find out
        whether disabling the GPL fallback costs us anything in practice.
        """
        _disable_gpl_espeak_fallback()
        from misaki import en

        g2p = en.G2P(trf=False, british=False, fallback=None, unk="")
        phonemes, tokens = g2p(text)
        return [t.text for t in tokens if getattr(t, "phonemes", None) == "" and t.text.strip()]

    @property
    def resident_bytes(self) -> int:
        return self._peak_bytes
