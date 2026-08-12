"""Whisper Large-v3-Turbo on MLX.

Phase 1 benchmark candidate B: higher accuracy, larger footprint. Whisper is a
batch model with a fixed 30-second receptive field, so "streaming" here means
re-decoding a growing buffer on a cadence. That is more expensive per interim
than a true streaming model, which is exactly what the benchmark should reveal.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import numpy as np

from voiceagent.stt.base import SAMPLE_RATE, STTEngine, Transcript

DEFAULT_REPO = "mlx-community/whisper-large-v3-turbo"


class MLXWhisperEngine(STTEngine):
    name = "mlx-whisper"

    def __init__(
        self,
        repo: str = DEFAULT_REPO,
        interim_interval_s: float = 0.6,
        language: str | None = "en",
    ) -> None:
        self.repo = repo
        self.interim_interval_s = interim_interval_s
        #: None means let Whisper detect the language per utterance, which is
        #: what a bilingual loop needs. Pinning it is still worth doing when the
        #: language is known: detection costs an extra pass and can flip on short
        #: or noisy audio, and a wrong flip sends the text to a TTS engine that
        #: cannot pronounce it.
        self.language = language
        self._loaded = False
        self._peak_bytes = 0

    # --- lifecycle --------------------------------------------------------

    def load(self) -> None:
        """Warm the model.

        mlx-whisper caches weights globally on first transcribe() rather than
        exposing a load call, so we force the download and JIT by decoding a
        short burst of silence.
        """
        import mlx.core as mx
        import mlx_whisper

        mx.reset_peak_memory()
        silence = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
        mlx_whisper.transcribe(
            silence,
            path_or_hf_repo=self.repo,
            language=self.language,
            fp16=True,
            verbose=None,
        )
        self._peak_bytes = mx.get_peak_memory()
        self._loaded = True

    def unload(self) -> None:
        import gc

        import mlx.core as mx

        # mlx-whisper keeps the model on a class-level ModelHolder, so it stays
        # resident until we clear it. Note `mlx_whisper.transcribe` the
        # attribute is the *function*; the submodule of the same name has to be
        # reached through the import system.
        from mlx_whisper.transcribe import ModelHolder

        ModelHolder.model = None
        ModelHolder.model_path = None
        gc.collect()
        mx.clear_cache()
        self._loaded = False

    # --- inference --------------------------------------------------------

    def transcribe(self, audio: np.ndarray) -> Transcript:
        import mlx.core as mx
        import mlx_whisper

        started = time.perf_counter()
        result = mlx_whisper.transcribe(
            audio.astype(np.float32),
            path_or_hf_repo=self.repo,
            language=self.language,
            fp16=True,
            verbose=None,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        self._peak_bytes = max(self._peak_bytes, mx.get_peak_memory())
        return Transcript(
            text=result["text"].strip(),
            is_final=True,
            latency_ms=elapsed_ms,
            # When the language is pinned, Whisper echoes it back; when it is
            # None it reports what it detected. Either way this is the language
            # the text is actually in.
            language=result.get("language", self.language),
        )

    async def stream(self, audio_chunks: AsyncIterator[np.ndarray]) -> AsyncIterator[Transcript]:
        """Re-decode a growing buffer on a cadence to fake streaming."""
        buffer: list[np.ndarray] = []
        samples_since_interim = 0
        interim_every = int(self.interim_interval_s * SAMPLE_RATE)

        async for chunk in audio_chunks:
            buffer.append(chunk)
            samples_since_interim += len(chunk)

            if samples_since_interim >= interim_every:
                samples_since_interim = 0
                audio = np.concatenate(buffer)
                # Decode off the event loop so mic capture keeps draining.
                partial = await asyncio.to_thread(self.transcribe, audio)
                yield Transcript(
                    text=partial.text,
                    is_final=False,
                    latency_ms=partial.latency_ms,
                    language=partial.language,
                )

        if buffer:
            audio = np.concatenate(buffer)
            final = await asyncio.to_thread(self.transcribe, audio)
            yield Transcript(
                text=final.text,
                is_final=True,
                latency_ms=final.latency_ms,
                language=final.language,
            )

    @property
    def resident_bytes(self) -> int:
        return self._peak_bytes
