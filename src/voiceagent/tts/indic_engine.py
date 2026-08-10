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

#: f5_tts clips any reference longer than this internally. We truncate to the
#: same bound *before* handing it over, so the transcript the caller supplies
#: describes the audio the model actually hears. Getting this wrong is not a
#: subtle quality issue: F5 estimates how long the output should be from
#: (generated text length / reference text length) x reference duration, so a
#: transcript that under-describes its audio inflates the output. A 4-second
#: Hindi sentence came out as 25 seconds of audio that way.
REFERENCE_CLIP_SECONDS = 12.0


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
        speed: float = 1.0,
    ) -> None:
        self.repo = repo
        #: IndicF5 is a cloning model: it needs a reference clip plus that
        #: clip's transcript to condition on.
        self.reference_audio = reference_audio
        self.reference_text = reference_text
        self.reference_sample_rate = reference_sample_rate
        self.speed = speed
        self._model = None
        self._vocoder = None
        self._device = "cpu"
        self._output_sample_rate = SAMPLE_RATE
        self._peak_bytes = 0
        self._cancelled = False

    # --- lifecycle --------------------------------------------------------

    def load(self) -> None:
        """Construct the model directly rather than through from_pretrained.

        `AutoModel.from_pretrained` cannot load this model on transformers 5.x.
        v5 runs a model's `__init__` inside a meta-device context so weights can
        be materialized lazily, but IndicF5's `__init__` eagerly builds a
        torchaudio vocoder, which allocates real CPU tensors and blows up with
        "Tensor on device cpu is not on the expected device meta!".

        Downgrading transformers is not an option -- mlx-lm requires >=5.0.0 and
        it runs the LLM. So the model class is instantiated the same way the
        upstream module's own __main__ does it, outside any meta context, and
        the checkpoint is loaded afterwards.
        """
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        from f5_tts.infer import utils_infer as U
        from f5_tts.model import CFM, DiT
        from f5_tts.model.utils import get_tokenizer

        try:
            weights_path = hf_hub_download(self.repo, filename="model.safetensors")
            vocab_path = hf_hub_download(self.repo, filename="checkpoints/vocab.txt")
        except Exception as exc:  # noqa: BLE001
            if "gated" in str(exc).lower() or "401" in str(exc):
                raise IndicTTSAccessError(GATED_HELP.format(repo=self.repo)) from exc
            raise

        # f5_tts defaults to mps and works there; the upstream IndicF5 wrapper
        # hardcodes cuda-or-cpu, which is one more reason not to use it.
        self._device = "mps" if torch.backends.mps.is_available() else "cpu"

        vocab_char_map, vocab_size = get_tokenizer(vocab_path, "custom")

        # Architecture is fixed by the checkpoint; these are IndicF5's values.
        model = CFM(
            transformer=DiT(
                dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4,
                text_num_embeds=vocab_size, mel_dim=U.n_mel_channels,
            ),
            mel_spec_kwargs=dict(
                n_fft=U.n_fft,
                hop_length=U.hop_length,
                win_length=U.win_length,
                n_mel_channels=U.n_mel_channels,
                target_sample_rate=U.target_sample_rate,
                mel_spec_type="vocos",
            ),
            odeint_kwargs=dict(method="euler"),
            vocab_char_map=vocab_char_map,
        ).to(self._device)

        # The checkpoint was saved through an EMA wrapper around a compiled
        # module, so every key carries an "ema_model._orig_mod." prefix that
        # CFM does not have. Strip it rather than reconstructing their wrapper.
        raw = load_file(weights_path, device="cpu")
        state = {
            key.removeprefix("ema_model._orig_mod."): value
            for key, value in raw.items()
            if key.startswith("ema_model._orig_mod.")
        }
        if not state:
            raise RuntimeError(f"no ema_model weights found in {weights_path}")

        missing, _ = model.load_state_dict(state, strict=False)
        real_missing = [k for k in missing if not k.startswith("mel_spec.")]
        if real_missing:
            raise RuntimeError(
                f"checkpoint does not match the model: {len(real_missing)} missing keys, "
                f"e.g. {real_missing[:3]}"
            )

        model.eval()
        self._model = model
        self._vocoder = U.load_vocoder(
            vocoder_name="vocos", is_local=False, device=self._device
        )
        self._peak_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    def unload(self) -> None:
        import gc

        self._model = None
        self._vocoder = None
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
        limit = int(REFERENCE_CLIP_SECONDS * sample_rate)
        self.reference_audio = audio[:limit]
        self.reference_text = text.strip()
        self.reference_sample_rate = sample_rate

    def reference_health(self) -> str | None:
        """Warn when the transcript plainly does not describe the audio.

        Speech runs roughly 12-18 characters per second. Far outside that band
        means the transcript is wrong, and the symptom is wildly wrong output
        length rather than an error.
        """
        if self.reference_audio is None or not self.reference_text:
            return None
        seconds = len(self.reference_audio) / self.reference_sample_rate
        if seconds < 0.5:
            return None
        rate = len(self.reference_text) / seconds
        if rate < 5:
            return (
                f"reference transcript looks too short for {seconds:.1f}s of audio "
                f"({rate:.1f} chars/sec); output will be much longer than expected"
            )
        if rate > 40:
            return (
                f"reference transcript looks too long for {seconds:.1f}s of audio "
                f"({rate:.1f} chars/sec); output may be clipped"
            )
        return None

    def _require_reference(self) -> None:
        if self.reference_audio is None:
            raise RuntimeError(
                "IndicF5 is a voice-cloning model: call set_reference() with a "
                "consented clip and its transcript before synthesizing."
            )

    # --- inference --------------------------------------------------------

    def _generate_blocking(self, text: str) -> np.ndarray:
        import tempfile
        from pathlib import Path

        import soundfile as sf
        from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

        self._require_reference()

        # f5_tts wants a reference *path*, so the (decrypted) clip is written to
        # a temp file that is removed as soon as inference returns -- the
        # plaintext never lands anywhere persistent.
        with tempfile.TemporaryDirectory() as tmp:
            ref_path = Path(tmp) / "ref.wav"
            sf.write(ref_path, self.reference_audio, self.reference_sample_rate)
            ref_audio, ref_text = preprocess_ref_audio_text(
                str(ref_path), self.reference_text
            )
            audio, sample_rate, _ = infer_process(
                ref_audio,
                ref_text,
                text,
                self._model,
                self._vocoder,
                mel_spec_type="vocos",
                speed=self.speed,
                device=self._device,
            )

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if peak > 1.5:  # some paths return int16-scaled floats
            audio = audio / 32768.0
        self._output_sample_rate = sample_rate
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
