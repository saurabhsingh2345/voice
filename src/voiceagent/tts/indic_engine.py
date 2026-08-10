"""Indic-native TTS, behind the same TTSEngine interface as Kokoro.

Phase 9 established that Kokoro cannot speak Hindi in this build at all: its
Hindi route needs espeak/phonemizer, which is GPLv3 and deliberately disabled.
This engine is the replacement for every Indic language.

Two backends, both permissively licensed and both gated on Hugging Face
(`gated=auto`, so access is granted instantly on accepting the terms):

  * IndicF5 (MIT, 1.4 GB) -- zero-shot voice cloning from a reference clip, so
    it can speak Hindi *in the user's own voice*. This is the default because it
    collapses the "fine-tune a narrator voice" phase into an enrolment step we
    already have from Phase 7.
  * Indic Parler-TTS (Apache-2.0, 3.8 GB) -- description-prompted voices, no
    cloning, and it needs the separate `parler-tts` package which pins an older
    transformers than this project uses. Kept as the fallback because its
    licensing is unambiguous.

LICENSE NOTE on IndicF5: the card credits the F5-TTS authors, and plain F5-TTS
is non-commercial (trained on Emilia, CC-BY-NC) and is on this project's
denylist. IndicF5 is tagged MIT and lists Indic training corpora, but whether it
inherits F5-TTS weights is not stated. Fine for personal use; verify the lineage
before shipping commercially.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import numpy as np

from voiceagent.tts.base import AudioChunk, TTSEngine
from voiceagent.tts.chunker import SentenceChunker

INDICF5_REPO = "ai4bharat/IndicF5"
PARLER_REPO = "ai4bharat/indic-parler-tts"

#: IndicF5 synthesizes at 24 kHz, matching Kokoro and Chatterbox.
SAMPLE_RATE = 24_000


class IndicTTSAccessError(RuntimeError):
    """Raised when the model is gated and the machine is not authenticated."""


GATED_HELP = """
{repo} is a gated Hugging Face repo. Access is auto-approved, so this takes
about two minutes:

  1. Open https://huggingface.co/{repo} and click "Agree and access repository"
  2. Create a token at https://huggingface.co/settings/tokens (read scope)
  3. Run:  uv run hf auth login

Nothing is uploaded by this -- the token only authorises the download.
""".strip()


class IndicTTSEngine(TTSEngine):
    name = "indicf5"

    def __init__(
        self,
        repo: str = INDICF5_REPO,
        reference_audio: np.ndarray | None = None,
        reference_text: str = "",
        reference_sample_rate: int = 24_000,
    ) -> None:
        self.repo = repo
        #: IndicF5 is a cloning model: it needs a reference clip plus that
        #: clip's transcript to condition on.
        self.reference_audio = reference_audio
        self.reference_text = reference_text
        self.reference_sample_rate = reference_sample_rate
        self._model = None
        self._peak_bytes = 0
        self._cancelled = False

    # --- lifecycle --------------------------------------------------------

    def load(self) -> None:
        import torch
        from transformers import AutoModel

        try:
            self._model = AutoModel.from_pretrained(self.repo, trust_remote_code=True)
        except Exception as exc:  # noqa: BLE001
            if "gated" in str(exc).lower() or "401" in str(exc):
                raise IndicTTSAccessError(GATED_HELP.format(repo=self.repo)) from exc
            raise

        # MPS first, since it is an order of magnitude faster than CPU here, but
        # Parler/F5-style stacks have historically had gaps in the Metal
        # backend. Fall back rather than crash.
        self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        try:
            self._model = self._model.to(self._device)
        except Exception:  # noqa: BLE001
            self._device = "cpu"
            self._model = self._model.to("cpu")

        self._peak_bytes = sum(
            p.numel() * p.element_size() for p in self._model.parameters()
        )

    def unload(self) -> None:
        import gc

        self._model = None
        gc.collect()
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def cancel(self) -> None:
        self._cancelled = True

    # --- reference voice --------------------------------------------------

    def set_reference(self, audio: np.ndarray, text: str, sample_rate: int) -> None:
        """Point the engine at a consented reference clip and its transcript."""
        self.reference_audio = audio
        self.reference_text = text
        self.reference_sample_rate = sample_rate

    def _require_reference(self) -> None:
        if self.reference_audio is None:
            raise RuntimeError(
                "IndicF5 is a voice-cloning model: call set_reference() with a "
                "consented clip and its transcript before synthesizing."
            )

    # --- inference --------------------------------------------------------

    def _generate_blocking(self, text: str) -> np.ndarray:
        import soundfile as sf
        import tempfile
        from pathlib import Path

        self._require_reference()

        # The model's remote code takes a reference *path*, so the decrypted
        # clip is written to a temp file that is deleted immediately after.
        with tempfile.TemporaryDirectory() as tmp:
            ref_path = Path(tmp) / "ref.wav"
            sf.write(ref_path, self.reference_audio, self.reference_sample_rate)
            audio = self._model(
                text,
                ref_audio_path=str(ref_path),
                ref_text=self.reference_text,
            )

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        # IndicF5 returns int16-scaled float in some versions.
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if peak > 1.5:
            audio = audio / 32768.0
        return audio

    async def synthesize(self, text: str, voice: str | None = None) -> AsyncIterator[AudioChunk]:
        if self._model is None:
            raise RuntimeError("load() must be called before synthesize()")

        self._cancelled = False
        started = time.perf_counter()
        audio = await asyncio.to_thread(self._generate_blocking, text)
        if self._cancelled or not audio.size:
            return

        yield AudioChunk(
            samples=audio,
            sample_rate=SAMPLE_RATE,
            is_final=True,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    async def synthesize_stream(
        self, text_chunks: AsyncIterator[str], voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        if self._model is None:
            raise RuntimeError("load() must be called before synthesize_stream()")

        self._cancelled = False
        chunker = SentenceChunker()
        started = time.perf_counter()
        first = True

        async def speak(sentence: str):
            nonlocal first
            audio = await asyncio.to_thread(self._generate_blocking, sentence)
            if self._cancelled or not audio.size:
                return
            latency = (time.perf_counter() - started) * 1000 if first else None
            first = False
            yield AudioChunk(samples=audio, sample_rate=SAMPLE_RATE, latency_ms=latency)

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
