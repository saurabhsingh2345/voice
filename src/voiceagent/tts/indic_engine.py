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

#: Flow matching starts from Gaussian noise, and f5_tts's `infer_process` does
#: not expose a seed -- it draws from the global torch RNG. So the same sentence
#: is a different utterance every call, and occasionally a bad one: the round-trip
#: check saw one formal sentence score 100% on one run and 44% (transcribed as
#: English) on the next, with no code change between them.
#:
#: Seeding per call makes output reproducible, which is what makes the round-trip
#: eval usable as a regression gate and a bug report reproducible. Pass
#: `seed=None` for a fresh voice each call, e.g. to retry a degenerate one.
#:
#: This does not *prevent* degeneration -- it makes it deterministic. Some
#: sentences may need a different seed to come out cleanly.
DEFAULT_SEED = 0


#: Why this engine pins two innocuous-looking DiT flags.
#:
#: IndicF5 (March 2025) was built against an f5-tts whose attention applied the
#: rotary embedding to the *flat* [b, n, 1024] projection and only then split it
#: into 16 heads of 64. `apply_rotary_pos_emb` rotates `t[..., :freqs.shape[-1]]`,
#: i.e. channels 0:64 -- which after the reshape are exactly head 0. So the
#: model was trained with rope on ONE head.
#:
#: Current f5-tts (1.1.x) splits heads *first* and rotates all 16, and its text
#: embedding masks padding through the ConvNeXt blocks (`mask_padding=True`),
#: which did not exist in the old version.
#:
#: Every tensor still loads -- 364/364 keys, no missing, no unexpected -- because
#: the *shapes* did not change, only where the positional signal lands. The
#: symptom is therefore not a crash but confident babble in a random language:
#: the same Hindi sentence came back from Whisper as Arabic, and previously as
#: Indonesian and Welsh.
#:
#: `pe_attn_head=1` restores rope-on-head-0; `text_mask_padding=False` restores
#: the unmasked text path. Measured by round-trip transcription (synthesize,
#: then Whisper the result back) on the same Hindi sentence:
#:
#:   pe_attn_head=None, mask=True   ->   0% overlap, detected Arabic   (babble)
#:   pe_attn_head=1,    mask=True   ->  55% overlap, detected Hindi    (garbled)
#:   pe_attn_head=1,    mask=False  ->  92% overlap, detected Hindi    (correct)
#:
#: Both flags are needed; either alone is not enough. This is why an earlier
#: pass that flipped only `text_mask_padding` concluded it made no difference.
#:
#: Verify with: uv run python -m voiceagent.eval.roundtrip <wav> "<expected>" hi
OLD_SEMANTICS = "pe_attn_head=1, text_mask_padding=False"


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
        seed: int | None = DEFAULT_SEED,
    ) -> None:
        self.repo = repo
        self.seed = seed
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
        # pe_attn_head and text_mask_padding are NOT cosmetic -- see
        # OLD_SEMANTICS below. Without them this model emits fluent-sounding
        # babble in a random language.
        model = CFM(
            transformer=DiT(
                dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4,
                text_num_embeds=vocab_size, mel_dim=U.n_mel_channels,
                pe_attn_head=1,
                text_mask_padding=False,
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
        """Point the engine at a consented reference clip and its transcript.

        A clip longer than REFERENCE_CLIP_SECONDS gets its *transcript* trimmed
        rather than its audio truncated, which is the opposite of what this used
        to do and measurably better.

        f5_tts clips the reference to 12s internally, and it does so at a silence
        boundary when it can find one -- far better than a hard cut mid-word. So
        the audio is handed over whole and that logic is left alone. What f5_tts
        does *not* do is adjust the transcript, and the transcript is what sets
        the output length: duration is estimated as
        (generated text length / reference text length) x reference duration. A
        transcript describing 21s of speech attached to the 12s the model
        actually hears inflates the denominator, so the output comes out too
        short -- and syllables get swallowed to fit.

        Measured on a 21.1s clip whose transcript described all 21.1s,
        synthesizing the same sentence (round-trip overlap, Hindi pinned):

            hard-cut audio 12s, full transcript   -> 88%, 2.89s (rushed)
            whole audio,        full transcript   -> 82%, 2.76s (worse)
            whole audio,        trimmed transcript-> 95%, 5.00s (correct)

        Raising nfe_step from 32 to 48 changed nothing once the transcript
        matched, so this -- not the sampler -- was the binding constraint.
        """
        self.reference_audio = audio
        self.reference_sample_rate = sample_rate

        text = text.strip()
        seconds = len(audio) / sample_rate if sample_rate else 0.0
        if seconds > REFERENCE_CLIP_SECONDS and seconds > 0:
            # Trim on a word boundary, proportionally to the share of the clip
            # f5_tts will keep. Approximate -- it cuts at a silence, not exactly
            # at 12.0s -- but it keeps the chars-per-second ratio honest, which
            # is what the duration estimate depends on.
            words = text.split()
            keep = max(1, int(len(words) * REFERENCE_CLIP_SECONDS / seconds))
            self.reference_text = " ".join(words[:keep])
        else:
            self.reference_text = text

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
        if seconds > REFERENCE_CLIP_SECONDS:
            # Report against the portion f5_tts will actually keep, not the whole
            # clip; measuring the trimmed transcript against the full duration
            # gives a rate that looks fine while describing audio the model never
            # hears. Say so, because the user can fix it by re-recording shorter.
            return (
                f"reference clip is {seconds:.1f}s but only the first "
                f"{REFERENCE_CLIP_SECONDS:.0f}s is used; the transcript was "
                f"trimmed to match. A clip of {REFERENCE_CLIP_SECONDS:.0f}s or "
                f"less with an exact transcript clones more faithfully."
            )
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
        import torch
        from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

        self._require_reference()

        # See DEFAULT_SEED: the sampler reads the global RNG, so this is the only
        # place the seed can be set.
        if self.seed is not None:
            torch.manual_seed(self.seed)

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
