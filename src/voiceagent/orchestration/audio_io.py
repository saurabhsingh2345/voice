"""Microphone capture and cancellable speaker playback.

Barge-in is the reason playback is built this way. Interrupting the agent means
dropping audio that has been synthesized but not yet played, so the queue has to
be discardable mid-stream -- a plain blocking write of the whole utterance
cannot be taken back.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import AsyncIterator

import numpy as np

MIC_SAMPLE_RATE = 16_000
#: 32 ms blocks: one Silero frame, small enough to keep barge-in responsive.
MIC_BLOCK = 512


class Microphone:
    """Async iterator of float32 mono blocks from the default input device."""

    def __init__(self, sample_rate: int = MIC_SAMPLE_RATE, block: int = MIC_BLOCK) -> None:
        self.sample_rate = sample_rate
        self.block = block
        self._stream = None

    async def blocks(self, stop: "threading.Event") -> AsyncIterator[np.ndarray]:
        import asyncio

        import sounddevice as sd

        loop = asyncio.get_running_loop()
        q: asyncio.Queue[np.ndarray] = asyncio.Queue()

        def callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
            # Copy: sounddevice reuses this buffer after the callback returns.
            loop.call_soon_threadsafe(q.put_nowait, indata[:, 0].copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.block,
            callback=callback,
        )
        with self._stream:
            while not stop.is_set():
                try:
                    yield await asyncio.wait_for(q.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue


class Speaker:
    """Streaming playback that can be flushed instantly for barge-in."""

    def __init__(self, sample_rate: int = 24_000) -> None:
        self.sample_rate = sample_rate
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._playing = threading.Event()
        self._generation = 0
        self._lock = threading.Lock()

    @property
    def is_playing(self) -> bool:
        return self._playing.is_set()

    def start(self) -> None:
        import sounddevice as sd

        def run() -> None:
            with sd.OutputStream(
                samplerate=self.sample_rate, channels=1, dtype="float32", blocksize=1024
            ) as out:
                while True:
                    item = self._queue.get()
                    if item is None:
                        break
                    if item.size == 0:
                        self._playing.clear()
                        continue
                    self._playing.set()
                    # Write in small slices so a barge-in takes effect within a
                    # few tens of milliseconds rather than at the end of the clip.
                    generation = self._generation
                    for start in range(0, len(item), 1024):
                        if generation != self._generation:
                            break  # cancelled
                        out.write(item[start : start + 1024].reshape(-1, 1))
                    if self._queue.empty():
                        self._playing.clear()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def play(self, samples: np.ndarray) -> None:
        if samples.size:
            self._queue.put(samples.astype(np.float32))

    def flush(self) -> None:
        """Drop everything queued and stop the current clip. Used for barge-in."""
        with self._lock:
            self._generation += 1
        drained = 0
        while True:
            try:
                self._queue.get_nowait()
                drained += 1
            except queue.Empty:
                break
        self._playing.clear()

    def close(self) -> None:
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=2)
