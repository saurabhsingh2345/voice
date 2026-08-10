"""Zero-shot voice cloning with Chatterbox Turbo on MLX.

Chatterbox Turbo (350M, MIT) clones a voice from a short reference clip with no
per-speaker training. Fish Speech was ruled out for this role -- its weights are
CC-BY-NC-SA-4.0 -- and XTTS v2 / F5-TTS are non-commercial too.

Every synthesis path goes through :class:`VoiceProfileStore`, so audio can only
be produced from a reference clip that carries a recorded ConsentRecord. The
plaintext reference is decrypted into memory for the call and never written to
disk.
"""

from __future__ import annotations

import asyncio
import io
import time
from collections.abc import AsyncIterator

import numpy as np

from voiceagent.tts.base import AudioChunk, TTSEngine
from voiceagent.tts.chunker import SentenceChunker
from voiceagent.voice_clone.store import ConsentError, VoiceProfileStore

DEFAULT_REPO = "mlx-community/chatterbox-turbo-fp16"

#: Chatterbox synthesizes at 24 kHz.
SAMPLE_RATE = 24_000


class ChatterboxCloneEngine(TTSEngine):
    name = "chatterbox-turbo"

    def __init__(self, repo: str = DEFAULT_REPO, store: VoiceProfileStore | None = None) -> None:
        self.repo = repo
        self.store = store or VoiceProfileStore()
        self._model = None
        self._peak_bytes = 0
        self._cancelled = False
        #: Decoded reference clips, keyed by profile id. Memory only.
        self._reference_cache: dict[str, tuple[np.ndarray, int]] = {}
        #: Which profile's conditioning is currently loaded on the model.
        self._active_profile: str | None = None

    # --- lifecycle --------------------------------------------------------

    def load(self, warmup: bool = True) -> None:
        import mlx.core as mx
        from mlx_audio.tts.utils import load_model

        mx.reset_peak_memory()
        self._model = load_model(self.repo)

        if warmup:
            # The first generation compiles Metal kernels and dominates the
            # measurement; burn it here against a throwaway reference so the
            # user's first real request is representative.
            import numpy as _np

            tone = 0.05 * _np.sin(
                2 * _np.pi * 130 * _np.linspace(0, 3.0, 3 * 24_000, dtype=_np.float32)
            )
            try:
                for _ in self._model.generate(
                    text="Ready.", ref_audio=tone, sample_rate=24_000,
                    max_tokens=self.MIN_TOKENS, verbose=False,
                ):
                    pass
            except Exception:  # noqa: BLE001 -- warmup is best-effort
                pass

        self._peak_bytes = mx.get_peak_memory()

    def unload(self) -> None:
        import gc

        import mlx.core as mx

        self._model = None
        self._reference_cache.clear()
        self._active_profile = None
        gc.collect()
        mx.clear_cache()

    def cancel(self) -> None:
        self._cancelled = True

    # --- references -------------------------------------------------------

    def _reference(self, profile_id: str) -> tuple[np.ndarray, int]:
        """Decrypt and decode a consented reference clip into memory."""
        if profile_id in self._reference_cache:
            return self._reference_cache[profile_id]

        profile = self.store.get(profile_id)
        if profile is None:
            raise ConsentError(f"No consented voice profile named {profile_id!r}.")

        import soundfile as sf

        wav_bytes = self.store.reference_audio(profile_id)
        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        self._reference_cache[profile_id] = (audio, sr)
        return audio, sr

    def forget_reference(self, profile_id: str) -> None:
        """Drop a decrypted clip from memory (called on profile deletion)."""
        self._reference_cache.pop(profile_id, None)
        if self._active_profile == profile_id:
            self._active_profile = None

    # --- inference --------------------------------------------------------

    #: Chatterbox emits S3 speech tokens at 25 Hz, and English runs ~15
    #: characters per second, so a character needs roughly 1.7 tokens. The
    #: factor below carries ~2x headroom for slow or emphatic delivery.
    TOKENS_PER_CHAR = 3.5
    MIN_TOKENS = 64
    MAX_TOKENS = 800

    def _token_budget(self, text: str) -> int:
        """Bound generation by how much speech the text can plausibly need.

        Left at the default 800, short inputs are pathologically slow: the model
        often fails to emit EOS for a two-word phrase and grinds through the
        full budget before the vocoder trims the result. "Hello there." took
        40.8s to produce 0.88s of audio. Scaling the cap with the text fixes it
        without truncating longer passages.
        """
        estimate = int(len(text) * self.TOKENS_PER_CHAR)
        return max(self.MIN_TOKENS, min(self.MAX_TOKENS, estimate))

    def _ensure_conditioned(self, profile_id: str) -> None:
        """Encode the reference clip once, not on every request.

        Passing ``ref_audio`` to generate() re-runs the speaker encoder and S3
        tokenizer over the whole reference every call -- a fixed ~3s cost that
        dominates short utterances. prepare_conditionals() caches that work on
        the model, so it is paid once per voice.
        """
        if self._active_profile == profile_id:
            return
        reference, ref_sr = self._reference(profile_id)
        self._model.prepare_conditionals(reference, sample_rate=ref_sr, exaggeration=0.0)
        self._active_profile = profile_id

    def _generate_blocking(self, text: str, profile_id: str) -> list[np.ndarray]:
        self._ensure_conditioned(profile_id)
        segments = []
        for result in self._model.generate(
            text=text,
            max_tokens=self._token_budget(text),
            verbose=False,
        ):
            segments.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
        return segments

    async def synthesize(self, text: str, voice: str | None = None) -> AsyncIterator[AudioChunk]:
        """Speak `text` in the voice of profile `voice`."""
        if self._model is None:
            raise RuntimeError("load() must be called before synthesize()")
        if not voice:
            raise ConsentError("A consented voice profile id is required.")

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
        """Sentence-chunked synthesis, same contract as the Kokoro engine."""
        if self._model is None:
            raise RuntimeError("load() must be called before synthesize_stream()")
        if not voice:
            raise ConsentError("A consented voice profile id is required.")

        self._cancelled = False
        chunker = SentenceChunker()
        started = time.perf_counter()
        first = True

        async def speak(sentence: str):
            nonlocal first
            for samples in await asyncio.to_thread(self._generate_blocking, sentence, voice):
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

    @property
    def resident_bytes(self) -> int:
        return self._peak_bytes
