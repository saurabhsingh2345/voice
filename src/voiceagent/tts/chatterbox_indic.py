"""Hindi TTS on Chatterbox Multilingual, behind the same TTSEngine interface.

Replaces `tts/indic_engine.py` (IndicF5). The reason is licensing first and
quality second, and both were measured before the switch --- see
`eval_out/chatterbox_spike/FINDINGS.md`.

**Licensing.** IndicF5's *weights* are MIT, but running it needs the `f5-tts`
package, which drags in `encodec` (CC-BY-NC), `Unidecode` (GPL), `frozendict`
(LGPL-3) and `soxr` (LGPL-2.1). `voice-doctor` failed on that tree and was right
to: a non-commercial licence restricts *use*, not only redistribution, so "we
never ship the weights" does not clear it. Chatterbox Multilingual is MIT and
reaches this project through `mlx-audio`, which was already installed for Kokoro
and the cloning path. The whole subtree goes away.

**Quality.** On the 12 held-out sentences, the same reference clip, and the
project's own round-trip scorer:

    condition     mean overlap   code-mixed   aggregate RTF
    human            90.2 %         91.5 %        --          (metric ceiling)
    IndicF5          88.7 %         86.0 %        3.40
    Chatterbox       93.5 %         94.9 %        1.17   (fp32)
    Chatterbox       93.5 %         94.5 %        0.63   (8-bit, the default)

Better on 9 of 12 sentences, tied on 2, worse on none. The largest gains are
digits (75 -> 95 %) and mid-sentence code-switching (81 -> 94 %). Read the 94 %
as "at this metric's ceiling", not "sounds better": round trip measures
intelligibility, and it cannot see prosody. The listening harness is the
instrument for that, now as a regression test rather than a verdict.

WHAT THIS ENGINE DOES NOT DO, that IndicF5 did:

  * **Only Hindi.** IndicF5 covered 11 Indian languages; Chatterbox
    Multilingual's 23 include Hindi and no other Indic language. Every other
    Indic script now raises `UnsupportedLanguage` rather than being spoken
    badly. See `_require_hindi`.
  * **No speed control.** `generate()` accepts `speed` and ignores it. The
    0.75 correction derived for IndicF5 was an artifact of f5-tts estimating
    duration arithmetically from the enrolment clip, and it does not transfer:
    measured here, output runs 0.81x the speaker's duration rather than 0.75x.
    There is no knob to apply it with, and applying the old constant would have
    made output ~8 % too slow.
  * **No reference transcript.** f5-tts set output length from
    (generated text length / reference text length) x reference duration, which
    is why a wrong transcript produced wildly wrong durations. Chatterbox
    conditions on the audio alone. `set_reference` still accepts the transcript
    so callers need not change, and so the consent record stays intact, but it
    no longer affects synthesis. A whole class of bug is gone.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from voiceagent.tts.base import AudioChunk, TTSEngine
from voiceagent.tts.chunker import SentenceChunker

#: MIT. Converted to mlx-audio's layout (single model.safetensors with ve./t3./
#: s3gen. prefixes); the S3 tokenizer is fetched separately from
#: mlx-community/S3TokenizerV2 by `Model.from_pretrained`.
#:
#: The source of truth, but not what runs: see `resolve_checkpoint`.
CHATTERBOX_REPO = "mlx-community/chatterbox-multilingual-v3"

#: 8-bit, quantized locally from CHATTERBOX_REPO on first use.
#:
#: Measured on an idle machine, 12 held-out sentences, medians of 3 runs:
#:
#:                  aggregate RTF   resident   peak    mean overlap   code-mixed
#:     fp32              1.17       3.04 GiB  4.55 GiB     93.5 %       94.9 %
#:     8-bit             0.63       1.33 GiB  2.77 GiB     93.5 %       94.5 %
#:
#: 1.9x faster, 44 % of the memory, and **the same quality** --- identical mean,
#: and the code-mixed difference is one sentence moving 6 points, well inside the
#: ±5 of an unseeded sampler. Every sentence is under RTF 1.0, so Hindi synthesis
#: keeps up with playback for the first time in this project.
#:
#: Built locally rather than pulled: four community requants exist and every one
#: declares **no licence**. A quantization of an MIT model is presumably MIT, and
#: "presumably" is the word `voice-doctor` exists to remove. See `tts/quantize.py`.
#:
#: The `data/` tree is gitignored, so this is a cache, not an artifact. Deleting
#: it costs seven seconds on the next load.
LOCAL_QUANTIZED = Path("data/models/chatterbox-multilingual-v3-8bit")


def resolve_checkpoint(build: bool = True) -> str:
    """The checkpoint to load: the local 8-bit build, quantizing it if needed.

    Falls back to the fp32 repo if quantization fails for any reason. That path
    is slower and needs more headroom, but it always works --- a build step that
    can turn into "Hindi is broken" would be a bad trade for 1.9x.
    """
    if LOCAL_QUANTIZED.exists():
        return str(LOCAL_QUANTIZED)
    if not build:
        return CHATTERBOX_REPO
    try:
        from voiceagent.tts.quantize import quantize

        print(f"Quantizing {CHATTERBOX_REPO} to 8-bit (once, ~7s) ...")
        return str(quantize(bits=8))
    except Exception as exc:  # noqa: BLE001
        print(f"Quantization unavailable ({type(exc).__name__}: {exc}); using fp32.")
        return CHATTERBOX_REPO


#: Chatterbox synthesizes at 24 kHz, matching Kokoro and IndicF5 before it.
SAMPLE_RATE = 24_000

#: The only Indic language this checkpoint speaks. Deliberately a set so the
#: check reads the same if Resemble ever adds Bengali or Tamil.
SUPPORTED_INDIC = frozenset({"hi"})

#: Chatterbox conditions the decoder on at most 10 s of reference audio
#: (`Model.DEC_COND_LEN` at 24 kHz) and the T3 encoder on the first 6 s. It
#: truncates internally; we record the bound so the UI can tell a user that the
#: tail of a long clip is not being heard.
#:
#: Unlike f5-tts this is *only* a fidelity question. There is no duration
#: arithmetic downstream of it, so a clip longer than this makes the clone less
#: like the speaker and nothing else.
REFERENCE_CLIP_SECONDS = 10.0

#: Sampling defaults from Praxy Voice (arXiv 2604.25441), which tuned Chatterbox
#: specifically for Indic text and published the recipe. Used unchanged for the
#: measurements quoted above, so changing them invalidates those numbers.
#:
#: Note the paper's other finding: its LoRA improves Telugu and Tamil but
#: *regresses* Hindi, and its own model card says to use vanilla Chatterbox for
#: Hindi. That is what this does.
EXAGGERATION = 0.7
TEMPERATURE = 0.6
MIN_P = 0.1

#: Chatterbox's own default. Kept explicit because this is the knob that stops a
#: speech-token loop, which is the same failure class as the Qwen3 repetition
#: bug elsewhere in this project.
REPETITION_PENALTY = 1.2

#: Speech-token budget per call. 1000 tokens at the tokenizer's 25 Hz is ~40 s
#: of audio, far above any single span `group_sentences` will produce; it is a
#: runaway guard, not a length limit.
MAX_NEW_TOKENS = 1000

#: The T3 sampler draws from MLX's global RNG and `generate()` exposes no seed,
#: so the same sentence is a different utterance every call. Measured across
#: five repeats of one sentence: 88-96 % round-trip overlap with no code change.
#: That is enough noise to make an unseeded harness useless as a regression gate,
#: which is exactly the trap `indic_engine` fell into and fixed the same way.
#:
#: Seeding does not *prevent* a degenerate generation, it makes one reproducible.
#: Pass `seed=None` for a fresh voice each call, e.g. to retry a bad one.
#:
#: Verified: with the seed set, T3's speech tokens are bit-identical across runs.
#: The *audio* is not, by ~1.2e-07 peak absolute difference --- Metal reduction
#: order, not sampling. So compare seeded output with a tolerance, never with
#: `array_equal`; a strict check here looks like the seed is being ignored.
DEFAULT_SEED = 0

#: Memory this engine needs free before it is safe to synthesize.
#:
#: Sized on the default 8-bit checkpoint, measured with `mx.get_peak_memory()`
#: on an idle machine:
#:
#:     8-bit   1.33 GiB after load   2.77 GiB peak during generation
#:     fp32    3.04 GiB              4.55 GiB
#:
#: Those are MLX's own peaks and they *include its buffer cache*, so they are an
#: upper bound on pressure rather than a floor. 3.0 sits just above the 8-bit
#: peak.
#:
#: This briefly stood at 4.0, sized for fp32, which was higher than IndicF5's 2.5
#: and would have refused Hindi on a busy machine that used to manage it.
#: Quantizing took that cost back and then some.
#:
#: The fp32 fallback in `resolve_checkpoint` needs more than this floor allows.
#: Accepted: the fallback only fires if quantization fails, and refusing early is
#: better than the wedge this guard exists to prevent --- the model paging out
#: mid-inference, so the request neither finishes nor fails.
#:
#: HONESTY NOTE: GIB_PER_EXTRA_BATCH is inherited from the IndicF5 envelope and
#: has NOT been re-measured against Chatterbox's allocator. Treat it the way
#: `models.py` treats `measured=False`.
MIN_FREE_GIB = 3.0
CHARS_PER_BATCH = 100
GIB_PER_EXTRA_BATCH = 0.5

#: Seconds of overlap when joining two separately-generated spans.
#:
#: Carried over from the IndicF5 path unchanged. It was chosen to match f5-tts's
#: internal `cross_fade_duration` so our joins were indistinguishable from
#: theirs; that specific reason is gone, but 0.15 s is still the right length for
#: a seam between independent generations, and keeping the value means narrations
#: rendered before and after this switch splice together identically.
CROSS_FADE_SECONDS = 0.15

#: Bytes of text per synthesis call, when no reference-derived budget applies.
#:
#: Devanagari is 3 bytes per character in UTF-8, so this is ~200 characters.
#: Chosen to keep a span near the length the spike actually measured: the longest
#: held-out sentence (h3, three clauses, 108 characters / 300 bytes) rendered
#: cleanly at 6.9 s of audio, so one such sentence per call is a proven size and
#: two is not.
DEFAULT_SPAN_BYTES = 600


class UnsupportedLanguage(RuntimeError):
    """Raised for an Indic language this checkpoint cannot speak.

    Explicit rather than silent. The router sends every Indic script here
    because that is still where Indic text belongs, and the honest failure is a
    named error --- not Bengali text read aloud by a Hindi model, which is what
    falling through to the nearest available voice would produce.
    """


def required_free_gib(text: str) -> float:
    """Headroom needed for the largest single call `text` will produce.

    Sized on the longest *span* rather than the whole request, because
    `synthesize` groups sentences up to `DEFAULT_SPAN_BYTES` and a span never
    exceeds that --- unless one sentence does, in which case the span is that
    sentence. Scaling on total length instead would refuse a long narration of
    short sentences that never allocates more than the floor.
    """
    chunker = SentenceChunker()
    sentences = [*chunker.feed(text), *chunker.flush()]
    longest = max((len(s) for s in sentences), default=len(text))
    batches = max(1, -(-longest // CHARS_PER_BATCH))
    return MIN_FREE_GIB + GIB_PER_EXTRA_BATCH * (batches - 1)


def group_sentences(sentences: list[str], budget_bytes: int) -> list[str]:
    """Pack whole sentences into spans of at most `budget_bytes`.

    Carried over from the IndicF5 path, where per-sentence calls were measured
    at 2.6x the compute of grouped ones for identical audio, because the
    reference conditioning and a full generation are paid per call. That
    arithmetic still holds here --- less strongly, since `_conds` is now cached
    across calls, but the seam argument is unchanged: five sentences synthesized
    separately means four joins between independent generations.

    A sentence longer than the budget is left whole rather than split. A cut we
    make blind is worse than a long call.
    """
    spans: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if current and len(candidate.encode("utf-8")) > budget_bytes:
            spans.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        spans.append(current)
    return spans


def concat_with_crossfade(
    parts: list[np.ndarray], sample_rate: int, seconds: float = CROSS_FADE_SECONDS
) -> np.ndarray:
    """Join spans with a cross-fade instead of a butt join.

    Linear ramps, not equal-power. Equal-power (sqrt of the linear ramp) holds
    power constant for uncorrelated signals and is the textbook choice, but its
    gains sum to 1.414 where the two spans *are* correlated --- and this output
    already reaches full scale, so that is clipping rather than a smoother join.
    Linear gains always sum to 1.
    """
    parts = [p for p in parts if p is not None and p.size]
    if not parts:
        return np.zeros(0, dtype=np.float32)
    if len(parts) == 1:
        return _limit(parts[0])

    n = int(seconds * sample_rate)
    out = parts[0]
    for nxt in parts[1:]:
        overlap = min(n, len(out), len(nxt))
        if overlap <= 0:
            out = np.concatenate([out, nxt])
            continue
        fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
        blended = out[-overlap:] * fade_out + nxt[:overlap] * fade_in
        out = np.concatenate([out[:-overlap], blended, nxt[overlap:]])
    return _limit(out.astype(np.float32))


def _limit(audio: np.ndarray, ceiling: float = 0.99) -> np.ndarray:
    """Scale down if the model overshot full scale, otherwise leave it alone.

    Scaling the whole span by one factor rather than clamping per sample keeps
    the relative levels inside a narration intact. Deliberately not normalising
    *up*: raising a quiet passage to the ceiling would make level depend on
    whatever the loudest moment happened to be, so two paragraphs of the same
    text would come back at different volumes.
    """
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak <= ceiling:
        return audio
    return (audio * (ceiling / peak)).astype(np.float32)


class ChatterboxIndicEngine(TTSEngine):
    name = "chatterbox-multilingual"

    def __init__(
        self,
        repo: str | None = None,
        reference_audio: np.ndarray | None = None,
        reference_text: str = "",
        reference_sample_rate: int = SAMPLE_RATE,
        seed: int | None = DEFAULT_SEED,
        exaggeration: float = EXAGGERATION,
        temperature: float = TEMPERATURE,
        min_p: float = MIN_P,
    ) -> None:
        #: None means "resolve at load", which is what picks the 8-bit build.
        #: Pass an explicit path or repo id to pin one, e.g. to compare them.
        self.repo = repo
        #: Chatterbox is a cloning model: it needs a reference clip. Unlike
        #: IndicF5 it does *not* need that clip's transcript.
        self.reference_audio = reference_audio
        self.reference_text = reference_text
        self.reference_sample_rate = reference_sample_rate
        self.seed = seed
        self.exaggeration = exaggeration
        self.temperature = temperature
        self.min_p = min_p
        self._model = None
        self._conds = None
        self._peak_bytes = 0
        self._cancelled = False
        self._span_seed: int | None = None
        #: See `_run`. Every MLX call this engine makes goes through this one
        #: thread, including the load.
        self._executor: ThreadPoolExecutor | None = None

    # --- thread affinity --------------------------------------------------

    def _run(self, fn, *args):
        """Run `fn` on this engine's single dedicated thread, and wait.

        MLX arrays are bound to the thread that created them. Touching one from
        another thread raises "There is no Stream(gpu, 0) in current thread" ---
        including for a plain multiply, not just RNG ops. So loading the weights
        on the caller's thread and then generating inside `asyncio.to_thread`
        cannot work: the worker inherits arrays it is not allowed to evaluate.

        This is easy to get wrong because MLX is lazy. Building the graph
        succeeds on any thread; only `mx.eval` touches the stream. The first
        attempt here ran the mel spectrogram, the S3 tokenizer and 20-odd
        transformer layers in a worker before dying at the sampler's per-token
        `mx.eval` --- which looks like a bug in the sampler and is not.

        One worker, `max_workers=1`, created lazily and owning both load and
        generate, fixes it. Serialising GPU access is a second benefit: the web
        server shares one engine instance across requests, and two overlapping
        synthesis calls would otherwise interleave on the same weights.
        """
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="chatterbox-tts"
            )
        return self._executor.submit(fn, *args).result()

    async def _run_async(self, fn, *args):
        loop = asyncio.get_running_loop()
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="chatterbox-tts"
            )
        return await loop.run_in_executor(self._executor, fn, *args)

    # --- lifecycle --------------------------------------------------------

    def load(self) -> None:
        self._run(self._load_on_worker)

    def _load_on_worker(self) -> None:
        import mlx.core as mx
        from mlx_audio.tts.utils import load_model

        if self.repo is None:
            self.repo = resolve_checkpoint()
        mx.reset_peak_memory()
        self._model = load_model(self.repo)
        self._conds = None
        self._peak_bytes = mx.get_peak_memory()

    def unload(self) -> None:
        import gc

        if self._executor is not None:
            try:
                self._run(self._unload_on_worker)
            finally:
                self._executor.shutdown(wait=True)
                self._executor = None
        self._model = None
        self._conds = None
        gc.collect()

    def _unload_on_worker(self) -> None:
        import mlx.core as mx

        self._model = None
        self._conds = None
        mx.clear_cache()

    def cancel(self) -> None:
        self._cancelled = True

    # --- reference voice --------------------------------------------------

    def set_reference(self, audio: np.ndarray, text: str, sample_rate: int) -> None:
        """Point the engine at a consented reference clip.

        The audio is handed over whole; Chatterbox truncates to its own
        conditioning windows (10 s decoder, 6 s encoder). We do not pre-trim,
        because the model's truncation is the one it was trained against.

        `text` is stored but unused --- see the module docstring. It stays in the
        signature because callers pass a consent-linked transcript and dropping
        the parameter would quietly discard that association.
        """
        self.reference_audio = audio
        self.reference_sample_rate = sample_rate
        self.reference_text = text.strip()
        # Conditionals are derived from the clip; a new clip invalidates them.
        self._conds = None

    def reference_health(self) -> str | None:
        """Warn when the clip will not clone well.

        Much shorter than the IndicF5 version, and that is the point. Every
        chars-per-second check there existed because the transcript drove output
        duration. It no longer does, so a mismatched transcript is harmless and
        warning about it would be noise.
        """
        if self.reference_audio is None:
            return None
        seconds = len(self.reference_audio) / max(self.reference_sample_rate, 1)
        if seconds < 3.0:
            return (
                f"reference clip is only {seconds:.1f}s; Chatterbox conditions on "
                f"up to {REFERENCE_CLIP_SECONDS:.0f}s and clones more faithfully "
                "with more of it"
            )
        if seconds > REFERENCE_CLIP_SECONDS:
            return (
                f"reference clip is {seconds:.1f}s but only the first "
                f"{REFERENCE_CLIP_SECONDS:.0f}s conditions the voice; the rest is "
                "ignored"
            )
        return None

    def _require_reference(self) -> None:
        if self.reference_audio is None:
            raise RuntimeError(
                "Hindi needs an enrolled voice. Chatterbox Multilingual is a "
                "cloning model with no built-in speaker, so unlike Kokoro it "
                "cannot speak until it has a reference clip.\n\n"
                "  uv run voice-web   ->   http://127.0.0.1:8823   ->   enrol a voice\n\n"
                "No default speaker ships with this project on purpose: it would "
                "be a real person's voice with no consent record attached. "
                "Callers holding their own clip can pass it to set_reference()."
            )

    def _ensure_conds(self):
        """Derive and cache the conditionals once per reference clip.

        `generate()` recomputes these from `audio_prompt` on every call ---
        a mel spectrogram, an S3 tokenization and a speaker embedding --- which
        is pure waste across the several spans of one narration.
        """
        if self._conds is None:
            import mlx.core as mx

            self._conds = self._model.prepare_conditionals(
                mx.array(np.asarray(self.reference_audio, dtype=np.float32)),
                self.reference_sample_rate,
                self.exaggeration,
            )
        return self._conds

    # --- inference --------------------------------------------------------

    @staticmethod
    def _require_hindi(text: str) -> None:
        from voiceagent.text.detect import detect

        detection = detect(text)
        if detection.is_indic and detection.language not in SUPPORTED_INDIC:
            raise UnsupportedLanguage(
                f"{detection.language!r} ({detection.script}) is not supported: "
                "Chatterbox Multilingual speaks Hindi and no other Indic "
                "language. IndicF5 covered 11; that capability was traded for a "
                "clean licence tree. See tts/chatterbox_indic.py."
            )

    def _generate_blocking(self, text: str) -> np.ndarray:
        import mlx.core as mx

        self._require_reference()

        seed = self._span_seed if self._span_seed is not None else self.seed
        if seed is not None:
            mx.random.seed(seed)

        conds = self._ensure_conds()
        chunks: list[np.ndarray] = []
        for result in self._model.generate(
            text=text,
            conds=conds,
            lang_code="hi",
            exaggeration=self.exaggeration,
            temperature=self.temperature,
            min_p=self.min_p,
            repetition_penalty=REPETITION_PENALTY,
            max_new_tokens=MAX_NEW_TOKENS,
            verbose=False,
        ):
            chunks.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))

        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return _limit(np.concatenate(chunks))

    async def synthesize(self, text: str, voice: str | None = None) -> AsyncIterator[AudioChunk]:
        """Synthesize a span at a time, yielding one chunk per span."""
        if self._model is None:
            raise RuntimeError("load() must be called before synthesize()")
        self._require_hindi(text)

        self._cancelled = False
        started = time.perf_counter()

        chunker = SentenceChunker()
        sentences = [*chunker.feed(text), *chunker.flush()]
        if not sentences:
            return

        spans = group_sentences(sentences, self.batch_budget)

        first = True
        for index, span in enumerate(spans):
            if self._cancelled:
                return
            # A different seed per span. With one fixed seed every span starts
            # from identical noise, which makes a long passage sound like the
            # same intonation contour repeated. Derived from the base seed so
            # the whole narration stays reproducible.
            if self.seed is not None:
                self._span_seed = self.seed + index
            audio = await self._run_async(self._generate_blocking, span)
            if self._cancelled or not audio.size:
                continue
            yield AudioChunk(
                samples=audio,
                sample_rate=SAMPLE_RATE,
                is_final=index == len(spans) - 1,
                # Latency is time to *first* audio, which is what a listener
                # waits for; later chunks have already started playing.
                latency_ms=(time.perf_counter() - started) * 1000 if first else None,
            )
            first = False

    async def synthesize_stream(
        self, text_chunks: AsyncIterator[str], voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        if self._model is None:
            raise RuntimeError("load() must be called before synthesize_stream()")

        # Deliberately NOT grouped, unlike synthesize(). Grouping buys prosody
        # and compute by waiting for more text, and waiting is the one thing a
        # live stream cannot do --- the caller is feeding tokens so it can start
        # speaking at the first sentence boundary. Narration should use
        # synthesize().
        self._cancelled = False
        chunker = SentenceChunker()
        started = time.perf_counter()
        first = True
        spoken = 0

        async def speak(sentence: str):
            nonlocal first, spoken
            self._require_hindi(sentence)
            if self.seed is not None:
                self._span_seed = self.seed + spoken
            spoken += 1
            audio = await self._run_async(self._generate_blocking, sentence)
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
    def batch_budget(self) -> int:
        """Bytes of text to put in one synthesis call.

        A constant, unlike the IndicF5 version. That one derived the budget from
        the reference clip's speaking rate because f5-tts sized its internal
        batches that way; Chatterbox does not batch on text length, so there is
        nothing to mirror and a fixed span length is the honest implementation.
        """
        return DEFAULT_SPAN_BYTES

    @property
    def resident_bytes(self) -> int:
        return self._peak_bytes
