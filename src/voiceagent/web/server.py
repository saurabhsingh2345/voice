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
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

from voiceagent.text.detect import detect
from voiceagent.text.normalize_hi import normalize as normalize_hi
from voiceagent.tts.indic_engine import (
    CHARS_PER_BATCH,
    concat_with_crossfade,
    GIB_PER_EXTRA_BATCH,
    MIN_FREE_GIB,
    required_free_gib,
)
from voiceagent.tts.remote_engine import RemoteTTSUnavailable
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
store = VoiceProfileStore()
engine = ChatterboxCloneEngine(store=store)

#: Training clips, stored inside each profile directory so the existing deletion
#: paths reach them. See voice_clone.dataset.
dataset = VoiceDataset(profiles=store)

#: Chatterbox is English-only, so Devanagari sent to it produces nothing usable.
#: Indic text goes to IndicF5, which clones from the same reference clip -- so
#: one enrolment gives the user their voice in both languages. Only one of the
#: two is held in memory at a time; together they would not fit alongside the
#: rest of the pipeline.
_indic_engine = None
_indic_lock = asyncio.Lock()

#: Serializes synthesis. This is a correctness requirement before it is a
#: performance one: both engines are single shared mutable objects, and the Indic
#: path calls `set_reference()` on the shared instance. Two overlapping requests
#: for different voices would have the second overwrite the first's reference,
#: so a request could be answered in someone else's voice.
#:
#: It also prevents the failure that actually bit: on an 18 GiB machine with
#: ~1.5 GiB free, two concurrent Indic requests thrash. Observed state was a
#: process in uninterruptible I/O wait at 0.3% CPU with RSS *shrinking* (94 ->
#: 63 MiB, the model paging out), 68 MiB free against 14.9 GiB of swap, and zero
#: progress. Neither request could finish.
_synth_lock = asyncio.Lock()

#: A second request while one is running is refused rather than queued. Queueing
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

#: Re-exported under this module's historical names. The definitions and the
#: reasoning now live in `tts.indic_engine`, beside the synthesis strategy they
#: depend on -- this module used to own them, a copy was made in
#: `web.tts_service`, and the two diverged within a day.
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


#: True once _ensure_indic has resolved to a service on another machine. Read by
#: the memory guard in /api/speak, which would otherwise measure the wrong host.
_indic_is_remote = False

#: Which checkpoint the resident Indic engine was loaded from, so a request for a
#: differently-trained voice can be detected rather than silently answered wrong.
_indic_checkpoint: "Path | None" = None


def indic_checkpoint_for(profile_id: str) -> "Path | None":
    """The fine-tuned weights for this voice, if it has any.

    A fine-tune belongs to the voice it was trained on, so it is looked up by
    profile id rather than configured globally: enrol two voices, train one, and the
    trained one should use its own weights without anybody selecting anything.
    `model_last.pt` rather than the numbered checkpoints -- the trainer writes it
    every `last_per_updates`, so it is the most recent state.
    """
    candidate = Path(dataset.root).parent / "f5tts_ckpts" / profile_id / "model_last.pt"
    return candidate if candidate.exists() else None


async def _ensure_indic(profile_id: str | None = None) -> "object":
    """Get the Indic engine, local or remote.

    Reloads when the requested voice needs different weights from the ones already
    resident. That is the same eviction trade as everywhere else here -- one engine
    fits, two do not -- and it costs a ~15 s load when alternating between a
    fine-tuned voice and a stock one.

    With VOICEAGENT_TTS_URL set, synthesis happens on another machine and this
    one keeps Chatterbox resident -- the eviction below exists only because both
    models cannot fit here at once, and that stops being true when one of them
    is somewhere else. Removing it also removes Chatterbox's 30 s reload on the
    next English request, which is the eviction's real cost.
    """
    global _indic_engine, _loaded, _indic_is_remote, _indic_checkpoint
    from voiceagent.tts.indic_engine import IndicTTSEngine
    from voiceagent.tts.remote_engine import from_env

    wanted = indic_checkpoint_for(profile_id) if profile_id else None

    async with _indic_lock:
        # Wrong weights resident: unload before loading the right ones, rather than
        # answering in a voice the caller did not ask for.
        if _indic_engine is not None and not _indic_is_remote and wanted != _indic_checkpoint:
            await asyncio.to_thread(_indic_engine.unload)
            _indic_engine = None

        if _indic_engine is None:
            remote = from_env()
            if remote is not None:
                # load() here is a health check, not a model load -- it fails
                # fast and loudly if the other machine is asleep, rather than
                # halfway through a request.
                await asyncio.to_thread(remote.load)
                _indic_engine = remote
                _indic_is_remote = True
            else:
                if _loaded:
                    engine.unload()
                    _loaded = False
                _indic_engine = IndicTTSEngine(checkpoint=wanted)
                await asyncio.to_thread(_indic_engine.load)
                _indic_checkpoint = wanted
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
    "opus": ("OGG", "OPUS", "audio/ogg", "opus"),
    "flac": ("FLAC", "PCM_16", "audio/flac", "flac"),
}


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
    return {
        "consent_phrase": CONSENT_PHRASE,
        "min_seconds": MIN_REFERENCE_SECONDS,
        "max_seconds": MAX_REFERENCE_SECONDS,
        "model_loaded": _loaded,
        "formats": sorted(OUTPUT_FORMATS),
    }


# --- profiles -------------------------------------------------------------


@app.get("/api/voices")
async def list_voices() -> list[dict]:
    return [
        {
            "profile_id": p.profile_id,
            "speaker_name": p.speaker_name,
            "created_at": p.created_at,
            "duration_seconds": round(p.duration_seconds, 1),
            "consent_granted_at": p.consent.granted_at,
            "reference_text": p.reference_text,
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
    # No cache to invalidate here: unlike the Chatterbox engine, IndicTTSEngine
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

    from voiceagent.tts.indic_engine import REFERENCE_CLIP_SECONDS

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
    text: str = Form(...),
    profile_id: str = Form(...),
    format: str = Form("wav"),
) -> Response:
    # Declared up front: this function both reads it (in the busy check) and
    # writes it (when synthesis starts), and Python requires the declaration
    # before the first read.
    global _synth_started_at

    if not text.strip():
        raise HTTPException(400, "text is required")

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

    # Refuse rather than queue -- see BUSY_MESSAGE. Checking `locked()` before
    # acquiring is deliberate: it makes a concurrent request fail immediately
    # instead of blocking, which is the whole point.
    if _synth_lock.locked():
        elapsed = ""
        if _synth_started_at is not None:
            elapsed = f" It has been running {time.perf_counter() - _synth_started_at:.0f}s."
        raise HTTPException(429, BUSY_MESSAGE + elapsed)

    # Route on script. Devanagari (and other Indic scripts) cannot be spoken by
    # Chatterbox at all, so this is a correctness decision, not a preference.
    detection = detect(text)
    spoken = text.strip()
    parts: list[np.ndarray] = []
    started = time.perf_counter()

    try:
        async with _synth_lock:
            _synth_started_at = time.perf_counter()
            if detection.is_indic:
                profile = store.get(profile_id)
                if profile is None:
                    raise HTTPException(403, "no such consented voice profile")
                if not profile.reference_text:
                    raise HTTPException(
                        400,
                        "This voice has no reference transcript. IndicF5 needs to know "
                        "what the reference clip says. Add a transcript for this voice "
                        "and try again.",
                    )

                # A transcript in the wrong script means it does not describe the
                # audio -- the usual cause is auto-transcribing Hindi speech with
                # the English-only STT, which invents plausible English words. F5
                # conditions on that text, so the output becomes babble rather than
                # failing. Refuse instead of synthesizing nonsense.
                ref_detect = detect(profile.reference_text)
                if not ref_detect.is_indic:
                    raise HTTPException(
                        400,
                        "The reference transcript for this voice is in Latin script, but "
                        "you asked for Indic output. If the clip is spoken in Hindi, type "
                        "the Devanagari transcript by hand -- Auto-transcribe uses an "
                        "English-only model and will invent English words for Hindi audio, "
                        "which makes the synthesis unintelligible. If the clip really is "
                        "English, record a new one in Hindi for native results.",
                    )

                # Refuse when the machine plainly cannot do it. Without this the
                # request does not fail, it wedges: the model pages out to swap
                # mid-inference and neither finishes nor errors. A message the user
                # can act on beats a ten-minute hang.
                #
                # Skipped entirely when a remote service is configured, and the
                # env var is read rather than `_indic_is_remote` so the answer
                # does not depend on whether _ensure_indic has run yet. This
                # guard measures *this* machine's memory; when the work happens
                # elsewhere that number is not merely irrelevant, it is wrong in
                # the dangerous direction -- it would refuse a request the
                # service machine had ample room for. The far side runs the same
                # guard against its own memory, and its 507 is passed through.
                import os as _os

                from voiceagent.tts.remote_engine import DEFAULT_URL_ENV

                if not _os.environ.get(DEFAULT_URL_ENV, "").strip():
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
                            "rather than returning an error.\n\n"
                            "If you meant this to run on the TTS service machine: this guard "
                            "only runs when VOICEAGENT_TTS_URL is unset, so seeing this "
                            "message means the variable did not reach this process. Export it "
                            "and restart the server.",
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
    except RemoteTTSUnavailable as exc:
        # 502, not 500: the failure is the other machine's, and the message names
        # it. Reported as-is because "is the Air awake and on this network" is
        # something the user can check, and a generic 500 would send them looking
        # in this process instead.
        raise HTTPException(502, str(exc)) from exc
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

    return Response(
        content=payload,
        media_type=media_type,
        headers={
            "X-Synthesis-Ms": f"{elapsed_ms:.0f}",
            "X-Audio-Seconds": f"{seconds:.2f}",
            "X-Realtime-Factor": f"{(elapsed_ms / 1000) / seconds:.2f}" if seconds else "0",
            "X-Audio-Format": ext,
            "X-Language": detection.language,
            "X-Weights": ("fine-tuned" if _indic_checkpoint else "stock"),
            "X-Engine": (
                ("indicf5-remote" if _indic_is_remote else "indicf5")
                if detection.is_indic
                else "chatterbox"
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
        # One command, deliberately. This used to print prepare_csv_wavs and
        # f5-tts_finetune-cli directly, which was wrong in four ways at once:
        # the stock CLI builds the model with text_mask_padding=True and
        # pe_attn_head=null, IndicF5 needs False and 1; it defaults to the pinyin
        # tokenizer instead of IndicF5's 2545-entry vocab; it cannot load a
        # checkpoint whose keys read ema_model._orig_mod.*; and it writes both the
        # dataset and the checkpoints inside .venv. The flag mismatch is the
        # dangerous one -- every tensor still loads, so it does not error, it just
        # spends hours unlearning the pretrained weights. voice-train-prep fixes
        # all four and prints the two commands that follow it.
        "next_steps": [
            f"uv run voice-train-prep {profile_id}",
            "then run the two commands it prints",
            f"rm -rf {destination}   # remove the decrypted copy when training finishes",
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


def main() -> int:
    import uvicorn

    print("\n  Local Voice Agent -> http://127.0.0.1:8823\n")
    uvicorn.run(app, host="127.0.0.1", port=8823, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
