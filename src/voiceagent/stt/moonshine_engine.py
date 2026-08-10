"""Moonshine STT.

Phase 1 benchmark candidate A: a natively streaming model built for low-latency
on-device use. Unlike Whisper it has no fixed 30-second window, so interim
results cost a fraction of a full re-decode.

License note: only the code and the ENGLISH models are MIT. Other languages
ship under the Moonshine Community License and are not cleared by this
project's allow-list -- see ``voiceagent.models``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import numpy as np

from voiceagent.stt.base import SAMPLE_RATE, STTEngine, Transcript


class MoonshineEngine(STTEngine):
    name = "moonshine"

    def __init__(
        self,
        arch: str = "BASE_STREAMING",
        language: str = "en",
        update_interval: float = 0.3,
    ) -> None:
        self.arch_name = arch
        self.language = language
        self.update_interval = update_interval
        self._transcriber = None
        self._rss_before = 0
        self._rss_after = 0

    # --- lifecycle --------------------------------------------------------

    def load(self) -> None:
        import psutil
        from moonshine_voice import get_model_for_language
        from moonshine_voice.moonshine_api import ModelArch
        from moonshine_voice.transcriber import Transcriber

        if self.language != "en":
            raise ValueError(
                f"Language {self.language!r} is not permitted: only Moonshine's "
                "English models are MIT-licensed."
            )

        proc = psutil.Process()
        self._rss_before = proc.memory_info().rss

        model_path, model_arch = get_model_for_language(
            self.language, getattr(ModelArch, self.arch_name)
        )
        self._transcriber = Transcriber(
            model_path=model_path,
            model_arch=model_arch,
            update_interval=self.update_interval,
        )
        # Force weights to actually page in before we measure.
        self._transcriber.transcribe_without_streaming(
            [0.0] * (SAMPLE_RATE // 2), SAMPLE_RATE
        )
        self._rss_after = proc.memory_info().rss

    def unload(self) -> None:
        import gc

        if self._transcriber is not None:
            self._transcriber.close()
            self._transcriber = None
        gc.collect()

    # --- inference --------------------------------------------------------

    @staticmethod
    def _join(transcript) -> str:
        return " ".join(line.text for line in transcript.lines).strip()

    def transcribe(self, audio: np.ndarray) -> Transcript:
        if self._transcriber is None:
            raise RuntimeError("load() must be called before transcribe()")

        samples = audio.astype(np.float32).tolist()
        started = time.perf_counter()
        result = self._transcriber.transcribe_without_streaming(samples, SAMPLE_RATE)
        elapsed_ms = (time.perf_counter() - started) * 1000

        return Transcript(text=self._join(result), is_final=True, latency_ms=elapsed_ms)

    async def stream(self, audio_chunks: AsyncIterator[np.ndarray]) -> AsyncIterator[Transcript]:
        """True streaming: Moonshine emits line events as speech is recognized."""
        if self._transcriber is None:
            raise RuntimeError("load() must be called before stream()")

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Transcript | None] = asyncio.Queue()

        def on_event(event) -> None:
            line = event.line
            # Called from Moonshine's own thread -- hop back to the event loop.
            loop.call_soon_threadsafe(
                queue.put_nowait,
                Transcript(
                    text=line.text.strip(),
                    is_final=line.is_complete,
                    latency_ms=float(line.last_transcription_latency_ms),
                ),
            )

        self._transcriber.add_listener(on_event)
        self._transcriber.start()

        async def pump() -> None:
            try:
                async for chunk in audio_chunks:
                    await asyncio.to_thread(
                        self._transcriber.add_audio,
                        chunk.astype(np.float32).tolist(),
                        SAMPLE_RATE,
                    )
            finally:
                await asyncio.to_thread(self._transcriber.stop)
                loop.call_soon_threadsafe(queue.put_nowait, None)

        pump_task = asyncio.create_task(pump())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            pump_task.cancel()
            self._transcriber.remove_all_listeners()

    @property
    def resident_bytes(self) -> int:
        return max(0, self._rss_after - self._rss_before)
