"""Text-to-speech interface.

Phase 3 implements this over Kokoro-82M. Phase 7 may add a consent-gated
cloned-voice backend behind the same interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioChunk:
    """A synthesized span of speech."""

    samples: np.ndarray
    """float32 mono PCM in [-1, 1]."""
    sample_rate: int
    is_final: bool = False
    latency_ms: float | None = None
    """For the first chunk: time from request to first audio."""


class TTSEngine(ABC):
    """A speech synthesis backend."""

    name: str

    evictable: bool = True
    """Whether unloading this engine reclaims memory on *this* machine.

    `TTSRouter` evicts one engine to make room for another, which is only ever
    worth doing when there is local memory to reclaim. An engine that runs
    somewhere else sets this False: evicting it frees nothing here and costs the
    machine that does the work a full model reload.

    Deliberately a declared flag rather than `resident_bytes == 0`. A local engine
    can legitimately report zero -- before load, or because its backend does not
    expose a memory counter -- and silently not evicting one of those would
    reintroduce the over-budget condition the router exists to prevent.
    """

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    async def synthesize(self, text: str, voice: str | None = None) -> AsyncIterator[AudioChunk]:
        """Synthesize one span of text, yielding audio as it is produced."""

    @abstractmethod
    async def synthesize_stream(
        self, text_chunks: AsyncIterator[str], voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        """Sentence-chunked synthesis over a live token stream.

        Implementations must begin emitting audio at the first sentence
        boundary rather than waiting for the stream to finish -- this is what
        keeps perceived latency under a second.
        """

    @abstractmethod
    def cancel(self) -> None:
        """Stop synthesis immediately. Required for Phase 4 barge-in."""

    @property
    @abstractmethod
    def resident_bytes(self) -> int: ...
