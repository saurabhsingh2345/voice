"""Speech-to-text interface.

Any STT backend (Moonshine, MLX-Whisper, ...) implements this so Phase 1 can
benchmark rivals and Phase 4 can swap the winner in without touching the
pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass

import numpy as np

#: All audio crossing a module boundary is float32 mono PCM in [-1, 1].
SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class Transcript:
    text: str
    is_final: bool
    """False for interim hypotheses that may still be revised."""
    latency_ms: float | None = None
    """Time from end-of-audio to this result. Populated for benchmarking."""
    language: str | None = None
    """ISO code the engine detected, or None if it cannot detect one.

    A bilingual loop needs this, not as a nicety: the reply has to be routed to
    a TTS engine that can actually speak the language that was spoken. Kokoro
    cannot say Hindi at all in this build and IndicF5 cannot pronounce English
    orthography, so guessing wrong is not a degraded voice, it is unintelligible
    output. Engines that only ever handle one language report that language
    rather than None, so callers never have to special-case them.
    """


class STTEngine(ABC):
    """A speech-to-text backend."""

    name: str

    @abstractmethod
    def load(self) -> None:
        """Load weights into memory. Separate from __init__ so we can measure it."""

    @abstractmethod
    def unload(self) -> None:
        """Release weights so a rival engine can be benchmarked cleanly."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> Transcript:
        """Batch-transcribe a complete utterance."""

    @abstractmethod
    async def stream(self, audio_chunks: AsyncIterator[np.ndarray]) -> AsyncIterator[Transcript]:
        """Consume live audio, yielding interim and final transcripts."""

    @property
    @abstractmethod
    def resident_bytes(self) -> int:
        """Actual resident memory of the loaded model, for the budget table."""
