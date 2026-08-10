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
        }
        for p in store.list()
    ]


@app.post("/api/voices")
async def enrol(
    speaker_name: str = Form(...),
    consent_phrase: str = Form(...),
    clip: UploadFile = Form(...),
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
        profile = store.save(consent, _to_wav_bytes(audio, sr), duration, sr)
    except (ValueError, ConsentError) as exc:
        raise HTTPException(400, str(exc)) from exc

    return JSONResponse(
        {
            "profile_id": profile.profile_id,
            "speaker_name": profile.speaker_name,
            "duration_seconds": round(duration, 1),
        }
    )


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
async def speak(text: str = Form(...), profile_id: str = Form(...)) -> Response:
    if not text.strip():
        raise HTTPException(400, "text is required")

    await _ensure_loaded()

    started = time.perf_counter()
    parts: list[np.ndarray] = []
    try:
        async for chunk in engine.synthesize(text.strip(), voice=profile_id):
            parts.append(chunk.samples)
    except ConsentError as exc:
        raise HTTPException(403, str(exc)) from exc

    if not parts:
        raise HTTPException(500, "no audio was produced")

    audio = np.concatenate(parts)
    elapsed_ms = (time.perf_counter() - started) * 1000
    seconds = len(audio) / SAMPLE_RATE

    return Response(
        content=_to_wav_bytes(audio, SAMPLE_RATE),
        media_type="audio/wav",
        headers={
            "X-Synthesis-Ms": f"{elapsed_ms:.0f}",
            "X-Audio-Seconds": f"{seconds:.2f}",
            "X-Realtime-Factor": f"{(elapsed_ms / 1000) / seconds:.2f}" if seconds else "0",
        },
    )


def main() -> int:
    import uvicorn

    print("\n  Local Voice Agent -> http://127.0.0.1:8823\n")
    uvicorn.run(app, host="127.0.0.1", port=8823, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
