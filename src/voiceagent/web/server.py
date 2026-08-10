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


async def _ensure_indic() -> "object":
    """Load IndicF5 on first Indic request, evicting Chatterbox to make room."""
    global _indic_engine, _loaded
    from voiceagent.tts.indic_engine import IndicTTSEngine

    async with _indic_lock:
        if _indic_engine is None:
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
    # The engine caches the decoded clip and its transcript; drop it so the new
    # text is picked up on the next request.
    if _indic_engine is not None:
        _indic_engine.forget_reference(profile_id)
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

    profile = store.set_reference_text(profile_id, text)
    if _indic_engine is not None:
        _indic_engine.forget_reference(profile_id)
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


# --- synthesis ------------------------------------------------------------


@app.post("/api/speak")
async def speak(
    text: str = Form(...),
    profile_id: str = Form(...),
    format: str = Form("wav"),
) -> Response:
    if not text.strip():
        raise HTTPException(400, "text is required")

    # Validate before synthesizing: encoding is the last step, and rejecting an
    # unknown format after a 30s generation wastes all of it.
    fmt = format.lower()
    if fmt not in OUTPUT_FORMATS:
        raise HTTPException(
            400, f"unsupported format {fmt!r}; choose one of {sorted(OUTPUT_FORMATS)}"
        )

    # Route on script. Devanagari (and other Indic scripts) cannot be spoken by
    # Chatterbox at all, so this is a correctness decision, not a preference.
    detection = detect(text)
    spoken = text.strip()
    parts: list[np.ndarray] = []
    started = time.perf_counter()

    try:
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
            "X-Engine": "indicf5" if detection.is_indic else "chatterbox",
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
