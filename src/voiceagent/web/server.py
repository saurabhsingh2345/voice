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
from voiceagent.tts.remote_engine import RemoteTTSUnavailable
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

#: Free memory below this makes Indic synthesis a coin flip between very slow and
#: wedged, so it is refused with an explanation instead. The model is ~1.4 GiB and
#: needs headroom for activations on top.
MIN_FREE_GIB_FOR_INDIC = 2.5

#: Extra memory to require per synthesis batch beyond the first.
#:
#: f5-tts splits text into batches of roughly this many characters and holds
#: activations for the batch it is working on, so the requirement grows with
#: length -- which is why a short sentence succeeded and a five-batch narration
#: was killed with the *same* memory free.
#:
#: The numbers are a rough envelope from observation, not a model of the
#: allocator: one batch completed at ~4.9 GiB available; five batches died there;
#: five batches completed at ~7.5 GiB.
CHARS_PER_BATCH = 100
GIB_PER_EXTRA_BATCH = 0.5

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


async def _ensure_indic() -> "object":
    """Get the Indic engine, local or remote.

    With VOICEAGENT_TTS_URL set, synthesis happens on another machine and this
    one keeps Chatterbox resident -- the eviction below exists only because both
    models cannot fit here at once, and that stops being true when one of them
    is somewhere else. Removing it also removes Chatterbox's 30 s reload on the
    next English request, which is the eviction's real cost.
    """
    global _indic_engine, _loaded, _indic_is_remote
    from voiceagent.tts.indic_engine import IndicTTSEngine
    from voiceagent.tts.remote_engine import from_env

    async with _indic_lock:
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
                _indic_engine = IndicTTSEngine()
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

    target = 16_000
    if sr != target:
        idx = (np.arange(int(len(audio) * target / sr)) * sr / target).astype(int)
        audio = audio[idx[idx < len(audio)]]

    def _run() -> dict:
        import mlx_whisper

        return mlx_whisper.transcribe(
            audio,
            path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
            fp16=True,
            verbose=None,
        )

    try:
        result = await asyncio.to_thread(_run)
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
                    batches = max(1, -(-len(spoken) // CHARS_PER_BATCH))
                    needed = MIN_FREE_GIB_FOR_INDIC + GIB_PER_EXTRA_BATCH * (batches - 1)
                    if free_gib < needed:
                        raise HTTPException(
                            507,
                            f"Only {free_gib:.1f} GiB of memory is free. This text is about "
                            f"{batches} synthesis batch{'es' if batches > 1 else ''} and needs "
                            f"roughly {needed:.1f} GiB. Close whatever is holding memory (a "
                            "running VM, extra editor or browser windows), or send a shorter "
                            "passage. Starting anyway risks the system killing this server "
                            "mid-request rather than returning an error.",
                        )

                indic = await _ensure_indic()
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

    audio = np.concatenate(parts)
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
            "X-Engine": (
                ("indicf5-remote" if _indic_is_remote else "indicf5")
                if detection.is_indic
                else "chatterbox"
            ),
            "X-Audio-Bytes": str(len(payload)),
            "Content-Disposition": f'inline; filename="speech.{ext}"',
        },
    )


def main() -> int:
    import uvicorn

    print("\n  Local Voice Agent -> http://127.0.0.1:8823\n")
    uvicorn.run(app, host="127.0.0.1", port=8823, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
