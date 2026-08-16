"""Local web UI: enrol a voice, type text, hear it back in that voice.

    uv run voice-web        then open http://127.0.0.1:8823

Binds to 127.0.0.1 only. Nothing is uploaded anywhere; the model runs on this
machine and the reference clip is encrypted at rest.

Tauri remains the eventual packaging target (Phase 8); this serves the same
local HTTP API a Tauri shell would call, so that port is a repackaging job
rather than a rewrite.
"""

from __future__ import annotations

import asyncio
import io
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

from voiceagent import version
from voiceagent.text.detect import detect, devanagari_language_note
from voiceagent.web import razorpay
from voiceagent.web.billing import (
    OVERAGE_PAISE_PER_10K,
    PLANS,
    Billing,
    InsufficientCredits,
)
from voiceagent.web.keys import ApiKey, KeyStore
from voiceagent.web.metering import FAILED, OK, REJECTED, Meter, Usage, characters_of
from voiceagent.web.public import (
    PublicSurface,
    RateLimiter,
    allowed_origins,
    client_ip,
    is_public,
)
from voiceagent.web.queue import Full as QueueFull
from voiceagent.web.queue import SynthesisQueue
from voiceagent.text.normalize_hi import normalize as normalize_hi
from voiceagent.tts.chatterbox_indic import (
    UnsupportedLanguage,
    CHARS_PER_BATCH,
    concat_with_crossfade,
    GIB_PER_EXTRA_BATCH,
    MIN_FREE_GIB,
    required_free_gib,
)
from voiceagent.voice_clone.dataset import (
    MAX_CLIP_SECONDS,
    MAX_SEGMENT_SECONDS,
    MEDIUM_CUT_SECONDS,
    MIN_CLIP_SECONDS,
    MINIMUM_USEFUL_SECONDS,
    DatasetError,
    VoiceDataset,
    plan_segments,
)
from voiceagent.voice_clone.engine import SAMPLE_RATE, ChatterboxCloneEngine
from voiceagent.voice_clone.store import (
    CONSENT_PHRASE,
    MAX_REFERENCE_SECONDS,
    MIN_REFERENCE_SECONDS,
    ConsentError,
    ConsentRecord,
    VoiceProfileStore,
)

app = FastAPI(title="Local Voice Agent")

#: Order matters. CORS is added last and therefore runs *outermost*, so a
#: request blocked by the public allowlist still comes back with the headers a
#: browser needs to read the response --- otherwise the page sees an opaque
#: network error instead of the 404, and the bug looks like the tunnel.
app.add_middleware(PublicSurface)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    #: The X-* headers carry the queue position and the billable count, and a
    #: browser cannot read them unless they are named here.
    expose_headers=[
        "X-Synthesis-Ms",
        "X-Audio-Seconds",
        "X-Realtime-Factor",
        "X-Audio-Format",
        "X-Language",
        "X-Language-Warning",
        "X-Queued-Seconds",
        "X-Queue-Ahead",
        "X-Billable-Characters",
        "Retry-After",
    ],
)

#: Protects the one machine when the link is public. Off-path locally: `check`
#: is only consulted in public mode.
rate_limiter = RateLimiter()

store = VoiceProfileStore()
engine = ChatterboxCloneEngine(store=store)

#: Captured at import, which is the instant Python binds every module this
#: process will ever run. `/api/config` compares it against the tree on each
#: poll so a server left running across a code change says so instead of failing
#: obscurely --- see voiceagent.version for why this is an mtime check and not a
#: SHA one.
SOURCE_AT_START = version.snapshot()

#: Training clips, stored inside each profile directory so the existing deletion
#: paths reach them. See voice_clone.dataset.
dataset = VoiceDataset(profiles=store)

#: Chatterbox *Turbo* -- the cloning engine below -- is English-only, so
#: Devanagari sent to it produces nothing usable. Hindi goes to Chatterbox
#: *Multilingual*, a separate checkpoint that clones from the same reference
#: clip, so one enrolment still gives the user their voice in both languages.
#: Only one of the two is held in memory at a time; together they would not fit
#: alongside the rest of the pipeline.
_indic_engine = None
_indic_lock = asyncio.Lock()

#: Serialization moved to `synth_queue` below, which owns the lock now. The
#: reason for it is unchanged and worth keeping in front of whoever tries to
#: raise concurrency: it is a **correctness** requirement before a performance
#: one. Both engines are single shared mutable objects and the Indic path calls
#: `set_reference()` on the shared instance, so two overlapping requests for
#: different voices would have the second overwrite the first's reference --- and
#: a request could be answered in someone else's voice.
#:
#: It also prevents the failure that actually bit: on an 18 GiB machine with
#: ~1.5 GiB free, two concurrent Indic requests thrash. Observed state was a
#: process in uninterruptible I/O wait at 0.3% CPU with RSS *shrinking* (94 ->
#: 63 MiB, the model paging out), 68 MiB free against 14.9 GiB of swap, and zero
#: progress. Neither request could finish.

#: A second request while one is running used to be refused rather than queued.
#: That has changed --- see `synth_queue` --- but the reasoning is why the queue
#: is capped rather than unbounded. Queueing
#: is what made this dangerous: synthesis is slow enough (RTF ~3.4 for Indic) to
#: look hung, so the natural response is to click again, and every extra click
#: made the machine slower rather than the answer sooner. Refusing turns that
#: into "still working" instead of a death spiral.
BUSY_MESSAGE = (
    "Still synthesizing the previous request. Only one runs at a time: the model "
    "is a single shared instance and this machine cannot fit two. Wait for the "
    "current one to finish -- retrying now makes it slower, not faster."
)

#: When the in-flight synthesis started, so the busy message can say how long it
#: has been going. "Busy" with no number is indistinguishable from "stuck", which
#: is the confusion that caused the retrying in the first place.
_synth_started_at: float | None = None

#: The line for the one machine. See `web.queue`: concurrency stays 1 and the
#: queue is hard-capped, so this changes *how* an overloaded server says no,
#: not whether it does.
synth_queue = SynthesisQueue()

#: Usage accounting. Opened once at import so a failure to create the database
#: is a startup error rather than a surprise on the first paid request.
meter = Meter()

#: API keys for `/v1`. Same reasoning: fail at startup, not on a customer's
#: first authenticated call.
key_store = KeyStore()

#: Accounts, plans and the credit ledger. Same reasoning again.
billing = Billing()

#: Until accounts exist, everything bills to one tenant. Named rather than
#: blank so the rows written today are still readable once real accounts land,
#: and so `totals()` has something to key on from the first request.
DEFAULT_ACCOUNT = "local"


def _meter_quietly(usage: Usage) -> None:
    """Record usage without ever failing the request that produced it.

    Losing a metering row is bad. Failing a generation the customer already
    waited for, because the accounting could not be written, is worse --- and a
    full disk would otherwise turn a working product into a broken one at
    exactly the moment there is most to lose.
    """
    try:
        meter.record(usage)
    except Exception:  # noqa: BLE001
        pass


def _debit_quietly(account: str, characters: int, note: str = "") -> None:
    """Spend credits without ever failing the request that spent them.

    Same trade as `_meter_quietly` and the same reasoning, with one difference
    worth being explicit about: a lost debit is revenue we do not collect, which
    is a cost to us, while a failed request is a cost to the customer who already
    waited. We take the former.

    The asymmetry is bounded because the *check* happens before synthesis. An
    account cannot run far past its allowance on lost debits: the next request
    reads a balance that is only wrong by however many writes failed, and a
    database sick enough to drop them is not going to serve many more requests.
    """
    try:
        billing.debit(account, characters, note=note)
    except Exception:  # noqa: BLE001
        pass

#: Backstop against a pathological paste, NOT a quality-of-service limit.
#:
#: This was briefly 800, which was wrong. Narrating a paragraph or a short essay
#: is a real use of this app and it worked before -- slowly, but it worked. The
#: hang it was meant to prevent came from *concurrency* (two generations in
#: flight, see BUSY_MESSAGE), not from length; length only makes one request
#: slow. Serializing synthesis fixed the hang, so capping at 800 just removed a
#: working feature. Roughly 3000 characters is about four minutes of speech and
#: perhaps fifteen of compute: long enough for any real narration, low enough
#: that a stray paste of a whole document still gets a clear answer.
MAX_SPEAK_CHARS = 3000

#: Re-exported under this module's historical name. The definition and the
#: reasoning live in `tts.chatterbox_indic`, beside the synthesis strategy they
#: depend on -- this module used to own them, a copy was made in a second
#: module, and the two diverged within a day.
MIN_FREE_GIB_FOR_INDIC = MIN_FREE_GIB

#: Deliberately NOT guarding on swap percentage, having tried it and been wrong.
#:
#: The reasoning was that a server killed mid-generation coincided with swap at
#: 15.33 of 16.00 GiB, so near-full swap meant no room for a large allocation.
#: But macOS sizes its swap file dynamically: with 113 GiB free on disk it read
#: "92% full" at 13.00 GiB total while macOS's own `memory_pressure` reported 74%
#: of memory free and the machine was healthy. Swap percentage is therefore
#: almost always high on this platform and says nothing about headroom -- it
#: describes how tightly the file is sized, not whether it can grow. Guarding on
#: it refused work that would have succeeded, which is a worse failure than the
#: one it was meant to prevent. Available memory, scaled by the size of the job,
#: is the signal that actually tracked the kills.

STATIC = Path(__file__).resolve().parent / "static"

#: The model is loaded on first use, not at import, so the page opens instantly.
_load_lock = asyncio.Lock()
_loaded = False


async def _ensure_loaded() -> None:
    global _loaded
    async with _load_lock:
        if not _loaded:
            await asyncio.to_thread(engine.load)
            _loaded = True


async def _ensure_indic(profile_id: str | None = None) -> "object":
    """Get the Hindi engine, evicting the English one to make room.

    Much smaller than it was, and the deletions are the interesting part.

    **No per-profile checkpoint.** IndicF5 was fine-tunable per voice, so this
    used to look up `data/f5tts_ckpts/<profile>/model_last.pt` and reload
    whenever the resident weights belonged to a different voice --- a ~15 s cost
    every time a caller alternated between a trained voice and a stock one.
    Chatterbox clones zero-shot from the reference clip, so there is one
    checkpoint for every voice and nothing to swap. `profile_id` is kept in the
    signature because callers pass it and the reference clip is still per-profile.

    **No remote branch.** The LAN service existed to keep `f5-tts` --- and its
    CC-BY-NC dependency --- off this machine. With that gone there is nothing to
    quarantine, and local synthesis is now RTF 1.24 rather than 3.40.
    """
    global _indic_engine, _loaded
    from voiceagent.tts.chatterbox_indic import ChatterboxIndicEngine

    async with _indic_lock:
        if _indic_engine is None:
            if _loaded:
                engine.unload()
                _loaded = False
            _indic_engine = ChatterboxIndicEngine()
            await asyncio.to_thread(_indic_engine.load)
    return _indic_engine


def _decode_upload(raw: bytes) -> tuple[np.ndarray, int, float]:
    """Decode an uploaded clip to mono float32, whatever container it arrived in."""
    try:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception:
        # Browsers hand back webm/ogg from MediaRecorder, which libsndfile
        # cannot read; fall back to ffmpeg-free decoding via miniaudio.
        try:
            import miniaudio

            decoded = miniaudio.decode(raw, output_format=miniaudio.SampleFormat.FLOAT32)
            audio = np.asarray(decoded.samples, dtype=np.float32)
            sr = decoded.sample_rate
            if decoded.nchannels > 1:
                audio = audio.reshape(-1, decoded.nchannels).mean(axis=1)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"Could not decode audio: {exc}") from exc

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr, len(audio) / sr


def _whisper(
    audio: np.ndarray, sample_rate: int, target: int = 16_000, words: bool = False
) -> dict:
    """Transcribe with Whisper, resampling to the 16 kHz it expects.

    Shared by the reference transcriber and the dataset transcriber. It was inline
    in the first of those; a second copy is how the memory-guard pair drifted, so
    this one is extracted the first time it is needed twice rather than the second.
    """
    import mlx_whisper

    if sample_rate != target:
        idx = (np.arange(int(len(audio) * target / sample_rate)) * sample_rate / target).astype(int)
        audio = audio[idx[idx < len(audio)]]
    return mlx_whisper.transcribe(
        audio,
        path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
        fp16=True,
        verbose=None,
        word_timestamps=words,
    )


def _to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


#: Output containers we can produce without shelling out to ffmpeg.
#: WebM is deliberately absent: it needs a Matroska muxer, and libsndfile has
#: none. Ogg/Opus carries the identical Opus codec, so the browser encodes a
#: real .webm client-side when that exact container is wanted.
OUTPUT_FORMATS: dict[str, tuple[str, str, str, str]] = {
    # key: (libsndfile format, subtype, media type, extension)
    "wav": ("WAV", "PCM_16", "audio/wav", "wav"),
    "mp3": ("MP3", "MPEG_LAYER_III", "audio/mpeg", "mp3"),
    "opus": ("OGG", "OPUS", "audio/ogg", "opus"),
    "flac": ("FLAC", "PCM_16", "audio/flac", "flac"),
}

#: MP3 is what people expect to receive, and it is ~25x smaller than the WAV ---
#: which matters over a home tunnel far more than it does in a datacentre.
#:
#: **The licence position, stated rather than assumed.** No new dependency is
#: added: `soundfile` is already here, declares BSD-3, and passes the audit,
#: which reads Python package metadata. The `libsndfile` bundled inside that
#: wheel is LGPL-2.1, and its MP3 *encoder* is LAME, also LGPL. So the audit's
#: green does not by itself clear this, and this project's own rule --- a
#: permissive model card is not a clean dependency tree --- says to look.
#:
#: Where that lands:
#:   * **Hosted API and website:** LGPL imposes nothing. The library is used to
#:     serve requests, not distributed.
#:   * **Desktop `.app`:** the library *is* distributed, and LGPL then requires
#:     that the user can replace it. libsndfile ships as a separate shared
#:     object inside the wheel, so relinking is possible and the obligation is
#:     satisfiable --- but it must be honoured in the bundle, not assumed.
#:
#: Only Layer III encodes; libsndfile advertises Layers I and II and raises
#: "unimplemented format" on both, so they are deliberately not offered.
MP3_NOTE = "libsndfile/LAME, LGPL-2.1 — see comment above before shipping in the bundle"


def _encode(audio: np.ndarray, sample_rate: int, fmt: str) -> tuple[bytes, str, str]:
    """Encode to the requested container. Returns (bytes, media_type, extension)."""
    try:
        sndfile_fmt, subtype, media_type, ext = OUTPUT_FORMATS[fmt]
    except KeyError:
        raise HTTPException(
            400, f"unsupported format {fmt!r}; choose one of {sorted(OUTPUT_FORMATS)}"
        ) from None

    buffer = io.BytesIO()
    if sndfile_fmt == "OGG" and subtype == "OPUS":
        # Opus is defined only at 48 kHz; libsndfile resamples for us, but it
        # refuses rates it cannot handle, so be explicit about the failure.
        try:
            sf.write(buffer, audio, sample_rate, format="OGG", subtype="OPUS")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"Opus encoding failed: {exc}") from exc
    else:
        sf.write(buffer, audio, sample_rate, format=sndfile_fmt, subtype=subtype)
    return buffer.getvalue(), media_type, ext


# --- pages ----------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (STATIC / "index.html").read_text()


@app.get("/api/config")
async def config() -> dict:
    # `with_git=False` keeps this endpoint free of subprocesses: the mtime scan
    # alone decides staleness, and `drift()` pays for git only once it has
    # something to report.
    stale = version.drift(SOURCE_AT_START, version.snapshot(with_git=False))
    return {
        "consent_phrase": CONSENT_PHRASE,
        "min_seconds": MIN_REFERENCE_SECONDS,
        "max_seconds": MAX_REFERENCE_SECONDS,
        "model_loaded": _loaded,
        "formats": sorted(OUTPUT_FORMATS),
        "source": SOURCE_AT_START.label(),
        "stale": stale,
    }


@app.get("/api/queue")
async def queue_state() -> dict:
    """How busy the one machine is.

    Polled by the studio while a generation is in flight, so a waiting person
    sees a position instead of a spinner. That is the whole reason queueing is
    safe here: someone who can see "2 ahead of you" does not click again, and
    clicking again was what turned a slow request into a stuck machine.

    Deliberately carries no identities --- only shape --- so it can be served to
    an unauthenticated page without leaking who else is generating.
    """
    return synth_queue.snapshot()


@app.get("/api/usage")
async def usage(account: str = DEFAULT_ACCOUNT, since: str | None = None) -> dict:
    """What an account has used. `since` is an ISO timestamp.

    Billable characters count successful generations only; failures and
    rejections are reported beside them rather than folded in, because a
    customer disputing an invoice is owed the difference between "you generated
    this" and "this machine was busy for you".
    """
    return meter.totals(account, since=since)


# --- profiles -------------------------------------------------------------


@app.get("/api/voices")
async def list_voices() -> list[dict]:
    """The voices available to generate with.

    Trimmed in public mode. The consent record --- the exact words the speaker
    read and when they granted it --- belongs to that person and to whoever
    audits us, not to every visitor with the link. A picker needs a name, a
    length and an id; the rest is disclosure with no purpose.
    """
    public = is_public()
    return [
        {
            "profile_id": p.profile_id,
            "speaker_name": p.speaker_name,
            "duration_seconds": round(p.duration_seconds, 1),
            **(
                {}
                if public
                else {
                    "created_at": p.created_at,
                    "consent_granted_at": p.consent.granted_at,
                    "reference_text": p.reference_text,
                }
            ),
        }
        for p in store.list()
    ]


@app.post("/api/voices")
async def enrol(
    speaker_name: str = Form(...),
    consent_phrase: str = Form(...),
    clip: UploadFile = Form(...),
    reference_text: str = Form(""),
) -> JSONResponse:
    """Enrol a voice. Rejected unless the consent phrase is typed exactly."""
    try:
        consent = ConsentRecord.create(speaker_name, consent_phrase)
    except ConsentError as exc:
        raise HTTPException(403, str(exc)) from exc

    audio, sr, duration = _decode_upload(await clip.read())

    if duration > MAX_REFERENCE_SECONDS:
        audio = audio[: int(MAX_REFERENCE_SECONDS * sr)]
        duration = MAX_REFERENCE_SECONDS

    try:
        profile = store.save(
            consent, _to_wav_bytes(audio, sr), duration, sr, reference_text=reference_text
        )
    except (ValueError, ConsentError) as exc:
        raise HTTPException(400, str(exc)) from exc

    return JSONResponse(
        {
            "profile_id": profile.profile_id,
            "speaker_name": profile.speaker_name,
            "duration_seconds": round(duration, 1),
        }
    )


@app.patch("/api/voices/{profile_id}")
async def update_voice(profile_id: str, reference_text: str = Form(...)) -> dict:
    """Attach or correct the reference transcript that Indic synthesis needs."""
    try:
        profile = store.set_reference_text(profile_id, reference_text)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    # No cache to invalidate here: unlike the cloning engine, the Indic engine
    # holds no per-profile reference cache, and /api/speak re-reads the clip and
    # transcript from the store on every request. This used to call
    # _indic_engine.forget_reference(), a method that only exists on the
    # Chatterbox engine -- so correcting a transcript raised AttributeError once
    # Hindi synthesis had loaded the Indic engine. That is the exact path the
    # output-length fix depends on.
    return {"profile_id": profile.profile_id, "reference_text": profile.reference_text}


@app.post("/api/voices/{profile_id}/transcribe")
async def transcribe_reference(profile_id: str) -> dict:
    """Transcribe the reference clip in whatever language it is spoken.

    This uses Whisper rather than Moonshine. Moonshine is English-only and does
    not fail on Hindi -- it invents English words that sound similar, and F5
    conditions on that text, so the synthesis came out as babble. Whisper
    detects the language and returns Devanagari for Hindi, which is what the
    Indic model actually needs.
    """
    import io as _io

    import soundfile as _sf

    profile = store.get(profile_id)
    if profile is None:
        raise HTTPException(404, "no such voice profile")

    audio, sr = _sf.read(_io.BytesIO(store.reference_audio(profile_id)), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    from voiceagent.tts.chatterbox_indic import REFERENCE_CLIP_SECONDS

    # Transcribe exactly the span the TTS will condition on, so the text and the
    # audio describe the same thing.
    audio = audio[: int(REFERENCE_CLIP_SECONDS * sr)]

    try:
        result = await asyncio.to_thread(_whisper, audio, sr)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"transcription failed: {exc}") from exc

    text = (result.get("text") or "").strip()
    if not text:
        raise HTTPException(422, "nothing intelligible was found in this clip")

    # See update_voice: nothing to invalidate, and calling forget_reference here
    # used to raise AttributeError on the Indic engine.
    profile = store.set_reference_text(profile_id, text)
    return {
        "profile_id": profile_id,
        "reference_text": profile.reference_text,
        "language": result.get("language", "?"),
    }


@app.delete("/api/voices/{profile_id}")
async def delete_voice(profile_id: str) -> dict:
    engine.forget_reference(profile_id)
    if not store.delete(profile_id):
        raise HTTPException(404, "no such voice profile")
    return {"deleted": profile_id}


@app.delete("/api/data")
async def delete_all() -> dict:
    """The brief's one-click 'delete all my data'."""
    engine._reference_cache.clear()
    return {"deleted_profiles": store.delete_all()}


@app.post("/api/encode")
async def encode(audio: UploadFile, format: str = Form("flac")) -> Response:
    """Re-encode already-synthesized audio, without generating it again.

    This exists to kill a real hazard. The Download button used to re-POST the
    text to /api/speak for any format other than WebM, so downloading a FLAC
    meant paying for a second full synthesis -- and because its button-disable
    was independent of the Speak button's, clicking Download while Speak was
    still running put two generations in flight at once. On an 18 GiB machine
    that is what wedged the server: both requests thrashed in swap and neither
    finished.

    The old comment justified it with "re-synthesizing would give different
    audio", which was true of an unseeded sampler. It no longer is -- the Indic
    engine seeds per call -- but regenerating identical audio for a container
    change was always wasted work. Encoding is pure CPU and takes milliseconds,
    so it does not need the synthesis lock.
    """
    fmt = format.lower()
    if fmt not in OUTPUT_FORMATS:
        raise HTTPException(
            400, f"unsupported format {fmt!r}; choose one of {sorted(OUTPUT_FORMATS)}"
        )

    raw = await audio.read()
    if not raw:
        raise HTTPException(400, "audio is required")
    try:
        samples, sample_rate = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"could not decode the uploaded audio: {exc}") from exc
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    payload, media_type, ext = _encode(samples, sample_rate, fmt)
    return Response(
        content=payload,
        media_type=media_type,
        headers={"X-Audio-Format": ext, "X-Audio-Seconds": f"{len(samples)/sample_rate:.2f}"},
    )


# --- synthesis ------------------------------------------------------------


@app.post("/api/speak")
async def speak(
    request: Request,
    text: str = Form(...),
    profile_id: str = Form(...),
    format: str = Form("wav"),
    account: str = Form(DEFAULT_ACCOUNT),
    key_id: str | None = None,
) -> Response:
    # Declared up front: this function both reads it (in the busy check) and
    # writes it (when synthesis starts), and Python requires the declaration
    # before the first read.
    global _synth_started_at

    if not text.strip():
        raise HTTPException(400, "text is required")

    # Refused here, at the same altitude as the other cheap validations and long
    # before the machine is held. Marathi is Devanagari, so it never reaches
    # `UnsupportedLanguage` the way Tamil does — it is detected as Hindi and
    # would synthesize into audio that is confidently not Marathi.
    #
    # This used to be a warning header. It is a 400 because the measurement is
    # not marginal: `ळ` came back 0 of 4 generations and the model substituted
    # Hindi words for Marathi ones every time (eval_out/devanagari/FINDINGS.md).
    # Serving that bills a customer for a language we cannot speak, and the
    # existing rule for Bengali applies unchanged — a named error beats a wrong
    # voice. Nepali stays a warning: its evidence is weaker (`र्` survived 2 of
    # 4) and so is its detector.
    note = devanagari_language_note(text)
    if note is not None and note.refuses:
        raise ApiError(400, "unsupported_language", note.message, language=note.language)

    # Validate before synthesizing: encoding is the last step, and rejecting an
    # unknown format after a 30s generation wastes all of it.
    fmt = format.lower()
    if fmt not in OUTPUT_FORMATS:
        raise HTTPException(
            400, f"unsupported format {fmt!r}; choose one of {sorted(OUTPUT_FORMATS)}"
        )

    if len(text.strip()) > MAX_SPEAK_CHARS:
        raise HTTPException(
            413,
            f"text is {len(text.strip())} characters; the limit is {MAX_SPEAK_CHARS}. "
            "Synthesis runs slower than real time, so this would take a very long "
            "while and produce nothing until it finished. Split it into sections.",
        )

    # Route on script. Devanagari (and other Indic scripts) cannot be spoken by
    # Chatterbox at all, so this is a correctness decision, not a preference.
    detection = detect(text)
    spoken = text.strip()
    parts: list[np.ndarray] = []
    started = time.perf_counter()
    billable = characters_of(text)

    # Checked before any work, because the point is to not spend the machine on
    # it. Only consulted in public mode: a local user rate-limiting themselves
    # would be a bug, not a protection.
    # `key_id` set means /v1 already authenticated and already charged the
    # limiter against that key. Checking again here would bill one request
    # twice and cut a paying customer off at half their allowance.
    caller = client_ip(request)
    if is_public() and key_id is None:
        refusal = rate_limiter.check(caller, billable)
        if refusal:
            _meter_quietly(
                Usage(
                    account=account,
                    characters=billable,
                    status=REJECTED,
                    voice=profile_id,
                    detail="rate limited",
                )
            )
            raise HTTPException(429, refusal)
        # Counted at admission rather than on success, so a request that fails
        # after occupying the machine still costs its caller a slot.
        rate_limiter.record(caller, billable)

    # Queue rather than refuse. This used to raise 429 the moment the lock was
    # held, and that was right for a developer tool -- see BUSY_MESSAGE for the
    # spiral it prevented. It is wrong for something someone paid for, which
    # reads a 429 as broken. `web.queue` keeps the property that made refusing
    # safe: the line is short, hard-capped, and every waiter is told its
    # position. Past capacity the honest answer is still no.
    #
    # `QueueFull` can only come out of the `async with` header, never the body,
    # so catching it around the whole block cannot swallow a synthesis error.
    ahead = 0
    queued_seconds = 0.0
    try:
        async with synth_queue.slot() as ticket:
            ahead = ticket.ahead
            queued_seconds = time.perf_counter() - started
            _synth_started_at = time.perf_counter()
            if detection.is_indic:
                profile = store.get(profile_id)
                if profile is None:
                    raise HTTPException(403, "no such consented voice profile")
                # Two guards used to stand here and both are gone with f5-tts.
                #
                # The first refused a voice with no reference transcript; the
                # second refused one whose transcript was in Latin script. Both
                # existed because f5-tts *conditioned on that text*: it set output
                # duration from (generated chars / reference chars) x reference
                # duration, so a missing transcript had no duration to work from
                # and a wrong-script one -- usually Hindi audio auto-transcribed
                # by the English-only STT into invented English words -- produced
                # babble rather than an error.
                #
                # Chatterbox conditions on the reference *audio* alone. A missing
                # or mismatched transcript can no longer affect synthesis, so
                # refusing on it would block work that now succeeds. The
                # transcript is still stored and still shown, because it belongs
                # to the consent record.

                # Refuse when the machine plainly cannot do it. Without this the
                # request does not fail, it wedges: the model pages out to swap
                # mid-inference and neither finishes nor errors. A message the user
                # can act on beats a ten-minute hang.
                import psutil as _psutil

                free_gib = _psutil.virtual_memory().available / 1024**3
                needed = required_free_gib(spoken)
                if free_gib < needed:
                    raise HTTPException(
                        507,
                        f"Only {free_gib:.1f} GiB of memory is free, and the longest "
                        f"sentence here needs roughly {needed:.1f} GiB. Close whatever is "
                        "holding memory (a running VM, extra editor or browser windows), "
                        "or break up the longest sentence -- total length is not the "
                        "constraint, since each sentence is synthesized separately. "
                        "Starting anyway risks the system killing this server mid-request "
                        "rather than returning an error.",
                    )

                indic = await _ensure_indic(profile_id)
                import io as _io
                import soundfile as _sf

                ref_audio, ref_sr = _sf.read(
                    _io.BytesIO(store.reference_audio(profile_id)), dtype="float32"
                )
                if ref_audio.ndim > 1:
                    ref_audio = ref_audio.mean(axis=1)
                indic.set_reference(ref_audio, profile.reference_text, ref_sr)

                # Numbers and dates must become Hindi words before synthesis, or the
                # model reads them in whatever language it defaults to.
                spoken = normalize_hi(spoken)
                started = time.perf_counter()
                async for chunk in indic.synthesize(spoken):
                    parts.append(chunk.samples)
            else:
                await _ensure_loaded()
                started = time.perf_counter()
                async for chunk in engine.synthesize(spoken, voice=profile_id):
                    parts.append(chunk.samples)
    except HTTPException:
        raise
    except ConsentError as exc:
        raise HTTPException(403, str(exc)) from exc
    except QueueFull as exc:
        # 503 with a wait, not 429. The line is full; the machine is fine.
        _meter_quietly(
            Usage(
                account=account,
                key_id=key_id,
                characters=billable,
                status=REJECTED,
                language=detection.language,
                voice=profile_id,
                detail="queue full",
            )
        )
        raise HTTPException(
            503, str(exc), headers={"Retry-After": str(max(1, int(exc.eta_seconds)))}
        ) from exc
    except UnsupportedLanguage as exc:
        # 400, not 500: the request is the problem, and the message names the
        # language. Chatterbox Multilingual speaks Hindi and no other Indic
        # language; IndicF5 covered eleven. See tts/chatterbox_indic.py.
        _meter_quietly(
            Usage(
                account=account,
                key_id=key_id,
                characters=billable,
                status=FAILED,
                language=detection.language,
                voice=profile_id,
                queued_seconds=queued_seconds,
                detail="unsupported language",
            )
        )
        raise HTTPException(400, str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        # Recorded before re-raising. A generation that died still occupied the
        # one machine for its duration, and usage that counts only successes
        # under-reports load exactly when the system is unhealthy.
        _meter_quietly(
            Usage(
                account=account,
                key_id=key_id,
                characters=billable,
                status=FAILED,
                language=detection.language,
                voice=profile_id,
                queued_seconds=queued_seconds,
                synthesis_seconds=time.perf_counter() - started,
                detail=type(exc).__name__,
            )
        )
        raise
    finally:
        # Cleared unconditionally: a stale timestamp would make the next busy
        # message report a nonsense duration.
        _synth_started_at = None

    if not parts:
        raise HTTPException(500, "no audio was produced")

    # Cross-fade rather than butt-join. Each part is an independent generation,
    # so a raw concatenate leaves an audible seam at every boundary -- which is
    # what f5-tts avoids internally with its own 0.15 s cross-fade, and what we
    # bypassed by splitting the text ourselves.
    audio = concat_with_crossfade(parts, SAMPLE_RATE)
    elapsed_ms = (time.perf_counter() - started) * 1000
    seconds = len(audio) / SAMPLE_RATE

    payload, media_type, ext = _encode(audio, SAMPLE_RATE, fmt)

    # Recorded after the audio exists and before it is handed back, so a
    # generation is billed if and only if the customer actually received it.
    _meter_quietly(
        Usage(
            account=account,
            key_id=key_id,
            characters=billable,
            status=OK,
            language=detection.language,
            voice=profile_id,
            audio_seconds=round(seconds, 3),
            synthesis_seconds=round(elapsed_ms / 1000, 3),
            queued_seconds=round(queued_seconds, 3),
        )
    )

    # Debited here and nowhere else: after the audio exists, in the same place
    # the usage row is written. A failed generation is recorded as spent
    # capacity (it held the machine) but is never charged, which is why the
    # debit sits on this side of the try and the metering of failures does not.
    _debit_quietly(account, billable, note=f"{detection.language} {profile_id}")

    # Nepali is Devanagari, so it is detected as Hindi and never reaches
    # `UnsupportedLanguage`. It synthesizes understandably in Hindi phonology,
    # which is worth serving and worth saying out loud, so it rides on a header.
    # Marathi is refused before synthesis instead — see `_refuse_unspeakable`.
    note = devanagari_language_note(text)
    language_warning = note.message if note else None

    return Response(
        content=payload,
        media_type=media_type,
        headers={
            "X-Synthesis-Ms": f"{elapsed_ms:.0f}",
            "X-Audio-Seconds": f"{seconds:.2f}",
            "X-Realtime-Factor": f"{(elapsed_ms / 1000) / seconds:.2f}" if seconds else "0",
            "X-Audio-Format": ext,
            "X-Language": detection.language,
            **({"X-Language-Warning": language_warning} if language_warning else {}),
            # What the caller waited behind, and what it will be charged. Both
            # in headers so a client can show a queue notice and a running
            # total without a second request.
            "X-Queued-Seconds": f"{queued_seconds:.2f}",
            "X-Queue-Ahead": str(ahead),
            "X-Billable-Characters": str(billable),
            # Always "stock": Chatterbox clones zero-shot, so there is no
            # per-voice fine-tune to distinguish. Kept so the header contract
            # does not change under existing clients.
            "X-Weights": "stock",
            "X-Engine": (
                "chatterbox-multilingual" if detection.is_indic else "chatterbox-turbo"
            ),
            "X-Audio-Bytes": str(len(payload)),
            "Content-Disposition": f'inline; filename="speech.{ext}"',
        },
    )


# --- training dataset -----------------------------------------------------
#
# Zero-shot cloning transfers timbre from one 12-second prompt and stops there.
# Everything below exists to get past that: many clips of one speaker, each with a
# transcript, accumulated until there is enough to fine-tune on.


@app.get("/api/prompts")
async def prompts() -> list[dict]:
    """Register-spanning sentences to read, from the project's own eval set.

    Volume alone does not capture how someone speaks. Thirty minutes of flat
    read-aloud teaches the model to sound flat, so the UI cycles prompts across
    registers -- formal, colloquial, code-mixed, numeric -- to push the dataset
    across the range instead of letting it settle into one tone.

    The whole set is about four minutes of speech, so it cannot get anyone to
    thirty. That is the intended division of labour and worth being explicit about:
    prompts buy *coverage* -- the retroflexes, aspirates, nuqta consonants and
    clusters a model cannot learn without hearing, plus question and exclamation
    contours it will never learn from declaratives. Volume has to come from free
    speech. Reading the same 94 sentences ten times would overfit to them.
    """
    from voiceagent.eval import sentences as S
    from voiceagent.train import prompts as P

    return [
        {"slug": p.slug, "text": p.text, "register": p.register, "note": p.focus}
        for p in P.ALL
    ] + [
        {"slug": s.slug, "text": s.text, "register": s.register, "note": s.note}
        for s in S.HINDI_ONLY
    ]


@app.get("/api/dataset/{profile_id}")
async def dataset_summary(profile_id: str) -> dict:
    summary = dataset.summary(profile_id)
    return {
        "profile_id": profile_id,
        "speaker_name": summary.speaker_name,
        "clip_count": summary.clip_count,
        "total_seconds": summary.total_seconds,
        "target_seconds": summary.target_seconds,
        "fraction_of_target": round(summary.fraction_of_target, 4),
        "usable": summary.usable,
        "minimum_useful_seconds": MINIMUM_USEFUL_SECONDS,
        "min_clip_seconds": MIN_CLIP_SECONDS,
        "max_clip_seconds": MAX_CLIP_SECONDS,
        "clips": [
            {
                "clip_id": c.clip_id,
                "text": c.text,
                "duration_seconds": c.duration_seconds,
                "language": c.language,
                "created_at": c.created_at,
            }
            for c in dataset.clips(profile_id)
        ],
    }


@app.post("/api/dataset/{profile_id}/transcribe")
async def transcribe_clip(profile_id: str, clip: UploadFile = Form(...)) -> dict:
    """Transcribe a clip before it is saved, so the text can be corrected first.

    Whisper rather than Moonshine, for the reason Phase 9 established: Moonshine is
    English-only and does not refuse Hindi, it invents English words that sound
    similar. Training on that text would teach the model a false mapping, which is
    worse than having no clip at all.
    """
    if dataset.profiles.get(profile_id) is None:
        raise HTTPException(404, "no such consented voice profile")

    audio, sr, duration = _decode_upload(await clip.read())
    try:
        result = await asyncio.to_thread(_whisper, audio, sr)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"transcription failed: {exc}") from exc
    return {
        "text": (result.get("text") or "").strip(),
        "language": result.get("language", "?"),
        "duration_seconds": round(duration, 2),
    }


@app.post("/api/dataset/{profile_id}/segment")
async def segment_take(
    profile_id: str, clip: UploadFile = Form(...), language: str = Form("hi")
) -> dict:
    """Split one long recording into many clips, in a single Whisper pass.

    This exists because 150 separate recordings is the actual obstacle to building a
    dataset, not the per-clip length cap. Raising that cap would not help: clips are
    batched by frames at 93.8 frames per second, so a clip longer than
    `batch_size_per_gpu` cannot be batched at all -- at the 800 frames this machine
    trains comfortably at, that is 8.5 seconds. Longer clips are not more data, they
    are skipped data.

    Whisper's word timestamps do the splitting, rather than a separate VAD pass, for
    a reason worth recording: VAD finds where speech stops, but it does not know what
    was said in each span, so a VAD split still needs one transcription call per
    segment. Word timestamps give boundaries *and* aligned text from one pass over
    the whole take -- and the boundaries land between words rather than wherever the
    energy happened to dip.

    Segments are accumulated up to `MAX_SEGMENT_SECONDS` and cut at the last word
    boundary that fits. Anything under `MIN_CLIP_SECONDS` is merged forward instead
    of being saved, because a fragment with a fragment of a transcript is exactly the
    false mapping this store refuses elsewhere.
    """
    if dataset.profiles.get(profile_id) is None:
        raise HTTPException(404, "no such consented voice profile")

    audio, sr, duration = _decode_upload(await clip.read())
    if duration < MIN_CLIP_SECONDS:
        raise HTTPException(400, f"Take is {duration:.1f}s; nothing to split.")

    try:
        result = await asyncio.to_thread(_whisper, audio, sr, 16_000, True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"transcription failed: {exc}") from exc

    words = [
        w
        for seg in result.get("segments", [])
        for w in seg.get("words", [])
        if (w.get("word") or "").strip()
    ]
    if not words:
        raise HTTPException(
            422, "Nothing intelligible was found in this take, so there is nothing to split."
        )

    saved, skipped, total = [], 0, 0.0
    for begin, finish, text in plan_segments(words):
        piece = audio[int(begin * sr) : int(finish * sr)]
        try:
            clip_saved = dataset.add_clip(
                profile_id, _to_wav_bytes(piece, sr), text, len(piece) / sr, sr, language=language
            )
        except DatasetError:
            skipped += 1
            continue
        saved.append({
            "clip_id": clip_saved.clip_id,
            "text": text,
            "duration_seconds": clip_saved.duration_seconds,
        })
        total += clip_saved.duration_seconds

    summary = dataset.summary(profile_id)
    return {
        "saved": len(saved),
        "skipped": skipped,
        "seconds_added": round(total, 2),
        "take_seconds": round(duration, 2),
        "clips": saved,
        "clip_count": summary.clip_count,
        "total_seconds": summary.total_seconds,
        "review_note": (
            "These transcripts come from Whisper, not from a prompt, so check them in "
            "the clip list. It reads Hindi at about 4.8% CER and gets real words wrong."
        ),
    }


@app.post("/api/dataset/{profile_id}/clips")
async def add_clip(
    profile_id: str,
    clip: UploadFile = Form(...),
    text: str = Form(...),
    language: str = Form("hi"),
) -> dict:
    audio, sr, duration = _decode_upload(await clip.read())
    try:
        saved = dataset.add_clip(
            profile_id, _to_wav_bytes(audio, sr), text, duration, sr, language=language
        )
    except ConsentError as exc:
        raise HTTPException(403, str(exc)) from exc
    except DatasetError as exc:
        raise HTTPException(400, str(exc)) from exc

    summary = dataset.summary(profile_id)
    return {
        "clip_id": saved.clip_id,
        "duration_seconds": saved.duration_seconds,
        "clip_count": summary.clip_count,
        "total_seconds": summary.total_seconds,
    }


@app.post("/api/contribute")
async def contribute(
    request: Request,
    profile_id: str = Form(...),
    text: str = Form(...),
    clip: UploadFile = Form(...),
    language: str = Form("hi"),
    synthetic_seconds: float = Form(0.0),
) -> dict:
    """A speaker reads back a line their own voice generated, and we keep it.

    This is the flywheel: every accepted contribution is `(text, synthetic,
    real)` for one speaker on one sentence, which is the corpus a fine-tune
    wants and is normally expensive to buy. Consent rides along for free ---
    the contributor is the speaker, and `dataset.add_clip` refuses any profile
    that lacks a `ConsentRecord`.

    **The recording is checked against the text before it is stored.** A clip
    filed under the wrong sentence teaches the voice something false and nothing
    downstream ever notices --- training loss falls exactly the same. Collected
    from people reading off a screen, mismatches are not an edge case: a misread
    word, a false start, a phone that captured a second of silence. See
    `voice_clone.contribution` for why round-trip overlap is the right
    instrument for this one question and the wrong one for ranking quality.

    Nothing here promises the voice improves today. It does not: this builds a
    dataset and a measurement, and training is a separate, later, GPU-bound
    step. The UI must not imply otherwise.
    """
    from voiceagent.voice_clone.contribution import verify_recording

    if is_public():
        refusal = rate_limiter.check(client_ip(request), characters_of(text))
        if refusal:
            raise HTTPException(429, refusal)
        rate_limiter.record(client_ip(request), characters_of(text))

    if store.get(profile_id) is None:
        raise HTTPException(403, "no such consented voice profile")

    audio, sr, duration = _decode_upload(await clip.read())
    wav = _to_wav_bytes(audio, sr)

    # Written to a temporary file because the scorer reads a path -- Whisper
    # wants a file, not an array, and reusing the project's one decode path
    # beats a second in-memory variant that could drift from it.
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as handle:
        handle.write(wav)
        handle.flush()
        verdict = verify_recording(
            handle.name,
            text,
            language=language,
            synthetic_seconds=synthetic_seconds or None,
            recorded_seconds=duration,
        )

    if not verdict.accepted:
        # 422, not 400: the request is well formed and the *content* is what
        # fails, and the caller can fix it by reading the line again.
        raise HTTPException(422, verdict.reason)

    try:
        saved = dataset.add_clip(profile_id, wav, text, duration, sr, language=language)
    except ConsentError as exc:
        raise HTTPException(403, str(exc)) from exc
    except DatasetError as exc:
        raise HTTPException(400, str(exc)) from exc

    summary = dataset.summary(profile_id)
    return {
        "clip_id": saved.clip_id,
        "duration_seconds": round(saved.duration_seconds, 2),
        "match": round(verdict.overlap, 3),
        "duration_ratio": round(verdict.duration_ratio, 2) if verdict.duration_ratio else None,
        "unusual_length": verdict.unusual_length,
        "clip_count": summary.clip_count,
        "total_seconds": round(summary.total_seconds, 1),
        "target_seconds": summary.target_seconds,
        "fraction_of_target": round(summary.fraction_of_target, 3),
        "usable": summary.usable,
    }


@app.patch("/api/dataset/{profile_id}/clips/{clip_id}")
async def edit_clip(profile_id: str, clip_id: str, text: str = Form(...)) -> dict:
    try:
        clip = dataset.set_text(profile_id, clip_id, text)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except DatasetError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"clip_id": clip.clip_id, "text": clip.text}


@app.delete("/api/dataset/{profile_id}/clips/{clip_id}")
async def remove_clip(profile_id: str, clip_id: str) -> dict:
    if not dataset.delete_clip(profile_id, clip_id):
        raise HTTPException(404, "no such clip")
    summary = dataset.summary(profile_id)
    return {"deleted": clip_id, "clip_count": summary.clip_count,
            "total_seconds": summary.total_seconds}


@app.post("/api/dataset/{profile_id}/merge-from")
async def merge_dataset(profile_id: str, source: str = Form(...)) -> dict:
    """Move every clip from `source` into this profile. Same speaker, two profiles.

    Consent records are not merged and not rewritten -- see
    VoiceDataset.move_clips. This moves training clips only.
    """
    try:
        moved = dataset.move_clips(source, profile_id)
    except ConsentError as exc:
        raise HTTPException(403, str(exc)) from exc
    except DatasetError as exc:
        raise HTTPException(400, str(exc)) from exc
    summary = dataset.summary(profile_id)
    return {
        "moved": moved,
        "clip_count": summary.clip_count,
        "total_seconds": summary.total_seconds,
        "usable": summary.usable,
    }


@app.post("/api/dataset/{profile_id}/export")
async def export_dataset(profile_id: str, language: str = Form("")) -> dict:
    """Write a plaintext training set and return the commands to train on it.

    This is the one operation here that puts decrypted audio on disk, because the
    trainer is a separate process that reads files. The destination is under
    `data/`, which is gitignored, and `purge` removes it again.
    """
    destination = Path(dataset.root).parent / "training" / profile_id
    try:
        metadata, count, seconds = dataset.export(
            profile_id, destination, language=language.strip() or None
        )
    except DatasetError as exc:
        raise HTTPException(400, str(exc)) from exc

    return {
        "metadata_csv": str(metadata),
        "clip_count": count,
        "total_seconds": seconds,
        "plaintext_warning": (
            "This directory holds DECRYPTED audio. Delete it when training finishes."
        ),
        # There is no training command any more, and saying so is more useful
        # than pointing at one that no longer exists.
        #
        # This used to print `voice-train-prep`, which fine-tuned IndicF5 through
        # f5-tts's trainer. Both went when the Hindi path moved to Chatterbox
        # Multilingual, which clones zero-shot from the reference clip: there is
        # one checkpoint for every voice and nothing to train.
        #
        # The export itself is kept. A clean, transcribed, per-clip dataset is
        # worth having regardless of what consumes it -- it is the asset a future
        # adapter would need, and it is the thing that takes weeks to collect.
        "next_steps": [
            "Fine-tuning is no longer part of this project: Chatterbox clones "
            "zero-shot from the enrolled reference clip, so a better clone comes "
            "from a better reference recording, not from training.",
            "This export is still worth keeping as a dataset in its own right.",
            f"rm -rf {destination}   # remove the decrypted copy when you are done",
        ],
    }


@app.delete("/api/dataset/{profile_id}")
async def clear_dataset(profile_id: str) -> dict:
    """Drop every training clip for a voice, leaving the enrolled voice itself."""
    return {"deleted_clips": dataset.delete_all(profile_id)}


# --- blind listening test -------------------------------------------------
#
# Serves the A/B harness. The listener endpoints deliberately expose item ids and
# audio only; the item-to-system mapping stays in the manifest, which nothing here
# returns. See eval.abtest for why the blinding is structural rather than a promise.


def _bench(benchmark_id: str):
    from voiceagent.eval.abtest import Benchmark

    resolved = benchmark_id if benchmark_id != "latest" else Benchmark.latest()
    if not resolved:
        raise HTTPException(404, "No benchmark has been built yet.")
    try:
        return Benchmark.load(resolved)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/listen", response_class=HTMLResponse)
async def listen_page() -> str:
    return (STATIC / "listen.html").read_text()


@app.get("/api/listen/{benchmark_id}/session")
async def listen_session(benchmark_id: str, listener: str, kind: str) -> dict:
    """The listener's queue: every item id in their order, and which they finished.

    Returns no system names and no transcripts. The sentence text is withheld on
    purpose for the identity test -- a listener who can read along is judging
    against the text rather than deciding whether a human said it.
    """
    from voiceagent.eval.abtest import IDENTITY, NATURALNESS

    if kind not in (IDENTITY, NATURALNESS):
        raise HTTPException(400, f"kind must be {IDENTITY!r} or {NATURALNESS!r}")
    bench = _bench(benchmark_id)
    order = bench.order_for(listener, kind)
    done = {r["item_id"] for r in bench.all_ratings() if r["listener"] == bench.ratings_path(listener).stem}
    return {
        "benchmark_id": bench.benchmark_id,
        "kind": kind,
        "items": order,
        "done": [i for i in order if i in done],
    }


@app.get("/api/listen/{benchmark_id}/audio/{item_id}")
async def listen_audio(benchmark_id: str, item_id: str) -> Response:
    bench = _bench(benchmark_id)
    try:
        bench.item(item_id)
    except KeyError:
        raise HTTPException(404, "no such item") from None
    path = bench.audio_path(item_id)
    if not path.exists():
        raise HTTPException(404, "audio missing for this item")
    return Response(
        content=path.read_bytes(),
        media_type="audio/wav",
        # No filename: a Content-Disposition carrying the system name would undo
        # the blinding that naming the file by item id exists to provide.
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/listen/{benchmark_id}/rate")
async def listen_rate(
    benchmark_id: str,
    listener: str = Form(...),
    item_id: str = Form(...),
    ms: int = Form(...),
    score: int | None = Form(None),
    called_real: bool | None = Form(None),
) -> dict:
    bench = _bench(benchmark_id)
    answer: dict = {"ms": ms}
    if score is not None:
        if not 1 <= score <= 5:
            raise HTTPException(400, "score must be 1-5")
        answer["score"] = score
    if called_real is not None:
        answer["called_real"] = called_real
    if len(answer) == 1:
        raise HTTPException(400, "a rating needs either a score or a real/synthetic answer")
    try:
        total = bench.record(listener, item_id, answer)
    except KeyError:
        raise HTTPException(404, "no such item") from None
    return {"recorded": item_id, "given": total}


@app.get("/api/listen/{benchmark_id}/results")
async def listen_results(benchmark_id: str, include_rushed: bool = False) -> dict:
    """Scores per system. Not linked from the listening page on purpose -- a listener
    who sees the running tally is no longer blind."""
    from voiceagent.eval.abtest import IDENTITY, NATURALNESS

    bench = _bench(benchmark_id)
    return {
        "benchmark_id": bench.benchmark_id,
        "created_at": bench.created_at,
        IDENTITY: bench.results(IDENTITY, include_rushed),
        NATURALNESS: bench.results(NATURALNESS, include_rushed),
    }


@app.get("/record-benchmark", response_class=HTMLResponse)
async def record_benchmark_page() -> str:
    return (STATIC / "record_benchmark.html").read_text()


@app.get("/api/benchmark/sentences")
async def benchmark_sentences() -> list[dict]:
    """The held-out sentences, with which already have a real recording."""
    from voiceagent.eval import heldout
    from voiceagent.eval.build_benchmark import REAL_HELDOUT

    return [
        {
            "slug": s.slug,
            "text": s.text,
            "targets": s.targets,
            "recorded": (REAL_HELDOUT / f"{s.slug}.wav").exists(),
        }
        for s in heldout.SENTENCES
    ]


@app.post("/api/benchmark/record")
async def benchmark_record(slug: str = Form(...), clip: UploadFile = Form(...)) -> dict:
    """Save the speaker reading one held-out sentence.

    This exists to remove a confound that made the first benchmark
    uninterpretable. The "real" condition used training clips -- spontaneous speech,
    different sentences from the synthetic condition. So a listener could sort the
    two by *content and speaking style* rather than by voice, and the numbers showed
    it from both directions: identity was easy to call, while naturalness scored the
    synthetic clips ABOVE the real ones, because clean read sentences sound tidier
    than real speech with disfluencies in it.

    With the speaker reading the same held-out sentences the model synthesises, real
    and synthetic differ in one thing only, which is the thing being measured.
    """
    from voiceagent.eval import heldout
    from voiceagent.eval.build_benchmark import REAL_HELDOUT

    try:
        heldout.by_slug(slug)
    except KeyError:
        raise HTTPException(404, f"no such held-out sentence: {slug}") from None

    audio, sr, duration = _decode_upload(await clip.read())
    if duration < 1.0:
        raise HTTPException(400, f"Recording is {duration:.1f}s -- too short to judge.")

    REAL_HELDOUT.mkdir(parents=True, exist_ok=True)
    path = REAL_HELDOUT / f"{slug}.wav"
    path.write_bytes(_to_wav_bytes(audio, sr))
    done = len(list(REAL_HELDOUT.glob("*.wav")))
    return {"slug": slug, "seconds": round(duration, 2), "recorded": done,
            "total": len(heldout.SENTENCES)}


@app.delete("/api/benchmark/record/{slug}")
async def benchmark_unrecord(slug: str) -> dict:
    from voiceagent.eval.build_benchmark import REAL_HELDOUT

    path = REAL_HELDOUT / f"{slug}.wav"
    if path.exists():
        path.unlink()
    return {"deleted": slug}


#: Default port, and the one the desktop shell looks for.
DEFAULT_PORT = 8823

#: Overridable, but only the port and only to loopback.
#:
#: Two reasons, and the second is the one that made this necessary. A second
#: instance -- the packaged app while a checkout is already serving, say -- would
#: otherwise die on "address already in use", and did: a bundle test appeared to
#: pass because the responses were coming from an unrelated server on the same
#: port, which is a worse failure than a crash because it looks like success.
#:
#: The host stays pinned to 127.0.0.1. Making it configurable would turn one
#: environment variable into the difference between a private assistant and one
#: answering the network, and nothing in this project needs that.
PORT_ENV = "VOICEAGENT_PORT"


def resolved_port() -> int:
    raw = os.environ.get(PORT_ENV, "").strip()
    if not raw:
        return DEFAULT_PORT
    try:
        port = int(raw)
    except ValueError as exc:
        raise SystemExit(f"{PORT_ENV}={raw!r} is not a port number") from exc
    if not (1 <= port <= 65535):
        raise SystemExit(f"{PORT_ENV}={port} is not in 1-65535")
    return port


def main() -> int:
    import uvicorn

    port = resolved_port()
    banner = f"\n  Local Voice Agent -> http://127.0.0.1:{port}"
    # Printed so that the answer to "what is this process actually running?" is
    # in the scrollback of the terminal that started it, months later.
    if SOURCE_AT_START.watchable:
        banner += f"\n  source: {SOURCE_AT_START.label()}  (restart after any code change)"
    # `flush` because stdout is block-buffered whenever it is not a terminal,
    # and uvicorn then blocks forever without flushing. Redirect the server to a
    # log --- which is how it gets left running for days, which is the situation
    # this line exists for --- and without this the banner never appears at all.
    print(banner + "\n", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# --- the developer API ----------------------------------------------------
#
# Versioned under /v1 and deliberately unlike the /api routes the studio uses.
# Those are an internal contract between this server and a page we ship
# together, and they change when the page changes. This one is a promise to
# someone else's codebase, so it is named separately, authenticated per request,
# and takes a *voice name* rather than an internal profile id -- a caller should
# write voice="saurabh", not a hex string they have to look up and that means
# nothing when it appears in their logs.


class ApiError(HTTPException):
    """An error shaped for a machine, and readable by the person debugging it."""

    def __init__(self, status: int, code: str, message: str, **extra) -> None:
        super().__init__(status, {"error": {"code": code, "message": message, **extra}})


def _authenticate(request: Request) -> ApiKey:
    """Resolve `Authorization: Bearer ...` to a key, or refuse.

    401 with a message that names the header, because the single most common
    integration failure is a key sent the wrong way, and "Unauthorized" alone
    sends a developer looking for a problem in their key rather than their
    header.
    """
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise ApiError(
            401,
            "missing_key",
            "Send your key as an Authorization header: "
            "`Authorization: Bearer swar_live_...`.",
        )
    key = key_store.verify(token.strip())
    if key is None:
        raise ApiError(
            401,
            "invalid_key",
            "That key is not valid, or it has been revoked. Keys are shown once "
            "when created; if it was lost, revoke it and mint another.",
        )
    return key


def _resolve_voice(name: str):
    """Find a consented profile by speaker name, case-insensitively.

    Names are what the API takes, so ambiguity is a real possibility once two
    speakers share one. Refusing beats guessing: silently picking the older
    `priya` would answer in the wrong person's voice, which is the one mistake
    a voice product must never make quietly.
    """
    wanted = (name or "").strip().lower()
    if not wanted:
        raise ApiError(400, "missing_voice", "A `voice` is required, e.g. \"saurabh\".")

    matches = [p for p in store.list() if p.speaker_name.strip().lower() == wanted]
    if not matches:
        available = sorted(p.speaker_name for p in store.list())
        raise ApiError(
            404,
            "unknown_voice",
            f"No voice called {name!r}.",
            available=available,
        )
    if len(matches) > 1:
        raise ApiError(
            409,
            "ambiguous_voice",
            f"More than one voice is called {name!r}. Ask us to rename one; "
            "answering in the wrong person's voice is not a risk we will take.",
            profile_ids=[p.profile_id for p in matches],
        )
    return matches[0]


@app.get("/v1/voices")
async def v1_voices(request: Request) -> dict:
    """The voices this key may generate with."""
    _authenticate(request)
    return {
        "voices": [
            {
                "name": p.speaker_name,
                "reference_seconds": round(p.duration_seconds, 1),
                "languages": ["hi", "en"],
            }
            for p in store.list()
        ]
    }


@app.get("/v1/usage")
async def v1_usage(request: Request, since: str | None = None) -> dict:
    """What this key has spent. Same numbers the invoice would be built from."""
    key = _authenticate(request)
    return meter.totals(key.account, since=since)


@app.get("/v1/balance")
async def v1_balance(request: Request) -> dict:
    """Plan, credits left, and what that is worth in machine time.

    `machine_seconds_remaining` is here because it is the honest unit. A million
    characters sounds unlimited and is eleven hours of a machine that has
    twenty-four, and a developer sizing a batch job should be able to see that
    before starting it rather than from a queue notice afterwards.
    """
    key = _authenticate(request)
    return billing.summary(key.account)


@app.get("/v1/plans")
async def v1_plans() -> dict:
    """The pricing page, as data. Deliberately unauthenticated.

    Someone deciding whether to sign up should not need a key to see the price,
    and a client that wants to show an upgrade prompt should not have to hardcode
    what we charge.
    """
    return {
        "currency": "INR",
        "plans": [
            {
                "name": plan.name,
                "inr_per_month": plan.monthly_inr,
                "characters_per_month": plan.monthly_characters,
                "blocks_when_empty": plan.blocks_when_empty,
                "note": plan.note,
            }
            for plan in PLANS.values()
        ],
        "overage_inr_per_10k_characters": OVERAGE_PAISE_PER_10K / 100,
        # Stated rather than implied. A customer who can see that payments are
        # not live yet will not spend an afternoon debugging their integration.
        "payments_enabled": razorpay.configured(),
    }


@app.post("/v1/checkout")
async def v1_checkout(request: Request) -> dict:
    """Start a payment for a plan. Requires Razorpay credentials.

    This is the boundary. Everything behind it — the ledger, the idempotent
    credit, the signature checks — is written and tested; this call is the one
    piece that cannot work until someone creates a Razorpay account and puts
    three values in the environment. It says so rather than failing obscurely.
    """
    key = _authenticate(request)
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        raise ApiError(400, "invalid_json", "The request body must be JSON.") from None

    plan_name = (body.get("plan") or "").lower()
    if plan_name not in PLANS:
        raise ApiError(400, "unknown_plan", f"No plan named {plan_name!r}.",
                       plans=sorted(PLANS))
    plan = PLANS[plan_name]
    if plan.monthly_paise <= 0:
        raise ApiError(400, "plan_is_free",
                       f"The {plan.name} plan costs nothing; there is nothing to pay.")

    try:
        order = razorpay.create_order(
            amount_paise=plan.monthly_paise,
            account=key.account,
            plan=plan.name,
            receipt=f"{key.account}:{plan.name}",
        )
    except razorpay.CredentialsMissing as exc:
        raise ApiError(503, "payments_unavailable", str(exc)) from exc
    except RuntimeError as exc:
        raise ApiError(502, "payment_provider_error", str(exc)) from exc

    return {
        "order": order,
        "key_id": razorpay.Credentials.from_env().key_id,
        "amount_inr": plan.monthly_inr,
        "plan": plan.name,
    }


@app.post("/v1/webhooks/razorpay")
async def v1_razorpay_webhook(request: Request) -> dict:
    """Credit an account when Razorpay says the money arrived.

    Unauthenticated by design — Razorpay has no key of ours to send — so the
    signature *is* the authentication, and it is checked against the raw body
    before anything is parsed. Verifying after parsing, or against a
    re-serialised dict, is the standard way this endpoint appears to work and
    is in fact wide open.

    Always returns 200 once the signature verifies, including for events we do
    not act on. Razorpay retries anything else until it gives up, and a retry
    storm caused by returning 400 for an event we simply ignore is self-inflicted.
    """
    credentials = razorpay.Credentials.from_env()
    if credentials is None:
        raise ApiError(
            503, "payments_unavailable",
            "Razorpay is not configured on this server, so webhooks cannot be "
            "verified and are refused rather than trusted.",
        )

    raw = await request.body()
    try:
        razorpay.verify_webhook(
            raw,
            request.headers.get("x-razorpay-signature", ""),
            credentials.webhook_secret,
        )
    except razorpay.SignatureInvalid as exc:
        # No body, no signature, nothing from the payload in the log. An
        # attacker probing this endpoint learns only that it refused.
        raise ApiError(401, "invalid_signature", "Signature verification failed.") from exc

    event = razorpay.parse_event(raw)
    if not event.account:
        return {"ok": True, "ignored": "no account in payment notes"}

    if event.credits:
        plan = PLANS.get(event.plan or "", None)
        characters = plan.monthly_characters if plan else 0
        written = billing.purchase(
            account=event.account,
            characters=characters,
            paise=event.amount_paise,
            reference=event.reference,
            note=f"razorpay {event.event}",
        )
        if plan:
            billing.set_plan(event.account, plan.name)
        # `written is None` means this payment was already credited. That is a
        # retry, which is normal, and the honest answer is success.
        return {"ok": True, "credited": written is not None}

    if event.reverses:
        plan = PLANS.get(event.plan or "", None)
        billing.refund(
            account=event.account,
            characters=plan.monthly_characters if plan else 0,
            paise=event.amount_paise,
            reference=event.reference,
            note=f"razorpay {event.event}",
        )
        return {"ok": True, "reversed": True}

    return {"ok": True, "ignored": event.event}


@app.post("/v1/speech")
async def v1_speech(request: Request) -> Response:
    """Text in, audio out.

    Accepts JSON, because that is what a developer expects to send and the
    studio's multipart form is an artefact of uploading files rather than a
    design. The response is the audio itself rather than a URL: there is no
    object store behind this, and a link that expires is a worse contract than
    bytes that do not.
    """
    key = _authenticate(request)

    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        raise ApiError(400, "invalid_json", "The request body must be JSON.") from None

    text = (body.get("text") or "").strip()
    voice_name = body.get("voice") or ""
    fmt = (body.get("format") or "mp3").lower()

    if not text:
        raise ApiError(400, "missing_text", "`text` is required.")
    if len(text) > MAX_SPEAK_CHARS:
        raise ApiError(
            413,
            "text_too_long",
            f"`text` is {len(text)} characters; the limit is {MAX_SPEAK_CHARS}.",
            limit=MAX_SPEAK_CHARS,
        )
    if fmt not in OUTPUT_FORMATS:
        raise ApiError(
            400,
            "unsupported_format",
            f"Unknown format {fmt!r}.",
            supported=sorted(OUTPUT_FORMATS),
        )

    profile = _resolve_voice(voice_name)
    billable = characters_of(text)

    # Rate limited by key, not by address: a customer behind one office NAT is
    # one caller, and two customers on the same cloud provider are not.
    refusal = rate_limiter.check(f"key:{key.key_id}", billable)
    if refusal:
        _meter_quietly(
            Usage(account=key.account, key_id=key.key_id, characters=billable,
                  status=REJECTED, voice=profile.profile_id, detail="rate limited")
        )
        raise ApiError(429, "rate_limited", refusal)
    rate_limiter.record(f"key:{key.key_id}", billable)

    # Checked before synthesis, never after. A generation holds the only machine
    # for ~40 s per 1000 characters, and spending that on a request we are going
    # to refuse is the worst of both outcomes -- the customer waits and is then
    # told no, and the capacity is gone either way.
    #
    # Only blocking plans raise here. A paid plan runs into overage instead,
    # because stopping a narration halfway to protect a few rupees costs more in
    # support and churn than the overage is worth.
    try:
        billing.check_affordable(key.account, billable)
    except InsufficientCredits as exc:
        _meter_quietly(
            Usage(account=key.account, key_id=key.key_id, characters=billable,
                  status=REJECTED, voice=profile.profile_id, detail="no credits")
        )
        raise ApiError(
            402,
            "insufficient_credits",
            f"{exc.needed} characters requested and {exc.balance} left this "
            "period. Upgrade at /v1/plans, or wait for the period to reset.",
            characters_remaining=exc.balance,
            characters_requested=exc.needed,
        ) from exc

    # Reuses the studio's path exactly -- same queue, same engine, same
    # normalisation. Two synthesis paths would drift, and the one that drifted
    # would be whichever had fewer eyes on it.
    return await speak(
        request=request,
        text=text,
        profile_id=profile.profile_id,
        format=fmt,
        account=key.account,
        key_id=key.key_id,
    )
