"""Indic TTS as a service, to be run on a second machine.

    VOICEAGENT_TTS_TOKEN=... uv run voice-tts-service --host 0.0.0.0

Runs on the machine that is *not* running the voice loop -- here, an 8 GiB M1
Air. It holds IndicF5 loaded permanently and answers synthesis requests over the
LAN, so the 18 GiB Mac never has to evict Kokoro or Chatterbox to speak Hindi.
See `voiceagent.tts.remote_engine` for why this is sound for a RTF-3.4 engine and
would be a mistake for anything interactive.

Deliberately does NOT import `voiceagent.web.server` or `voiceagent.voice_clone`.
This process needs neither: the reference clip arrives with each request and the
Fernet key stays in the client machine's Keychain. Importing them would drag
mlx-audio and keyring onto a machine that has no use for either, and would give
this process the ability to read voice profiles it has no business reading.

WHAT THIS PROCESS PERSISTS: nothing. The reference clip lives in memory for the
duration of one request, apart from the temp file f5_tts requires for its own
loader, which `IndicTTSEngine._generate_blocking` already deletes on return.
"""

from __future__ import annotations

import asyncio
import hmac
import io
import os
import time

import numpy as np
from fastapi import FastAPI, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response

from voiceagent.tts.indic_engine import MIN_FREE_GIB, required_free_gib

TOKEN_ENV = "VOICEAGENT_TTS_TOKEN"
DEFAULT_PORT = 8824

#: Same cap as the web UI, for the same reason: a backstop against a pathological
#: paste, not a quality-of-service limit.
MAX_SPEAK_CHARS = 3000

#: Re-exported under this module's original name; the definition lives in
#: `tts.indic_engine` now. This module held the corrected copy and `web.server`
#: held the stale one, which is exactly the drift that made sharing it necessary.
MIN_FREE_GIB_FOR_INDIC = MIN_FREE_GIB

BUSY_MESSAGE = (
    "The TTS service is already synthesizing. Only one request runs at a time: the "
    "model is a single shared instance holding a per-request reference voice, so a "
    "second concurrent request would both overwrite that reference and thrash this "
    "machine's memory. Wait for the current one."
)

NO_TOKEN_HELP = f"""
{TOKEN_ENV} is not set, so this service will not start.

It listens for reference *voice clips* -- biometric data -- so an unauthenticated
port on your LAN is not an acceptable default. Generate a token once:

  python -c "import secrets; print(secrets.token_urlsafe(32))"

then set the same value on both machines:

  # on this machine (the service)
  export {TOKEN_ENV}='<token>'
  uv run voice-tts-service --host 0.0.0.0

  # on the machine running the voice loop
  export {TOKEN_ENV}='<token>'
  export VOICEAGENT_TTS_URL='http://<this-machine-ip>:{DEFAULT_PORT}'
""".strip()

app = FastAPI(title="Indic TTS service")

_engine = None
_load_lock = asyncio.Lock()

#: Serialized for the same two reasons as the web UI: the engine is a single
#: shared mutable object whose reference voice is set per request, and two
#: concurrent generations on a small machine thrash rather than finish.
_synth_lock = asyncio.Lock()
_synth_started_at: float | None = None


def _expected_token() -> str:
    return os.environ.get(TOKEN_ENV, "").strip()


def require_token(authorization: str = Header(default="")) -> None:
    """Reject anything without the shared token.

    Fails closed by construction: an unset token cannot match, because `main()`
    refuses to start without one and a missing header compares against a
    non-empty secret. `compare_digest` rather than `==` so the comparison does
    not leak the token's length or prefix through timing.
    """
    expected = _expected_token()
    if not expected:
        raise HTTPException(503, "This service has no token configured and cannot serve.")
    supplied = authorization.removeprefix("Bearer ").strip()
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(401, "Invalid or missing token.")


async def _ensure_engine():
    """Load IndicF5 once and keep it. Staying loaded is the point of this process."""
    global _engine
    from voiceagent.tts.indic_engine import IndicTTSEngine

    async with _load_lock:
        if _engine is None:
            engine = IndicTTSEngine()
            await asyncio.to_thread(engine.load)
            _engine = engine
    return _engine


def _free_gib() -> float:
    import psutil

    return psutil.virtual_memory().available / 1024**3


#: Kept as a module-level name so this file reads the same as before the shared
#: definition moved out.
_required_gib = required_free_gib


# --- endpoints ------------------------------------------------------------


@app.get("/health")
async def health(authorization: str = Header(default="")) -> dict:
    require_token(authorization)
    return {
        "engine": "indicf5",
        "loaded": _engine is not None,
        "busy": _synth_lock.locked(),
        "free_gib": round(_free_gib(), 2),
        "max_chars": MAX_SPEAK_CHARS,
    }


@app.post("/tts/preload")
async def preload(authorization: str = Header(default="")) -> dict:
    """Load the model now, so the first real request does not pay for it.

    Worth a call from a login script or after a reboot: a cold first request cost
    19.5 s against 8.4 s warm, and that difference lands on whoever happens to
    type the first sentence. Returns once the weights are resident.
    """
    require_token(authorization)
    if _synth_lock.locked():
        raise HTTPException(429, BUSY_MESSAGE)
    await _ensure_engine()
    return {"loaded": True, "free_gib": round(_free_gib(), 2)}


@app.post("/tts/unload")
async def unload(authorization: str = Header(default="")) -> dict:
    """Drop the model, so the client's router eviction still means something."""
    global _engine
    require_token(authorization)
    if _synth_lock.locked():
        raise HTTPException(429, BUSY_MESSAGE)
    if _engine is not None:
        await asyncio.to_thread(_engine.unload)
        _engine = None
    return {"loaded": False}


@app.post("/tts/speak")
async def speak(
    authorization: str = Header(default=""),
    text: str = Form(...),
    reference_text: str = Form(...),
    reference: UploadFile = Form(...),
) -> Response:
    """Synthesize `text` in the voice of `reference`, and return a 24 kHz WAV.

    The reference arrives with every request rather than being cached under an
    identifier. That is a privacy decision before it is a simplicity one: this
    process never holds a voice between requests, so there is nothing here to
    leak, enumerate, or forget to delete. It costs ~0.5 MB on a LAN against a
    request that runs for seconds.
    """
    global _synth_started_at
    require_token(authorization)

    import soundfile as sf

    spoken = text.strip()
    if not spoken:
        raise HTTPException(400, "text is required")
    if len(spoken) > MAX_SPEAK_CHARS:
        raise HTTPException(
            413,
            f"text is {len(spoken)} characters; the limit is {MAX_SPEAK_CHARS}. "
            "Synthesis runs slower than real time, so this would produce nothing "
            "until it finished. Split it into sections.",
        )
    if not reference_text.strip():
        raise HTTPException(
            400,
            "reference_text is required: IndicF5 conditions on the reference "
            "transcript as well as the audio, and uses its length to estimate how "
            "long the output should be.",
        )

    if _synth_lock.locked():
        elapsed = ""
        if _synth_started_at is not None:
            elapsed = f" It has been running {time.perf_counter() - _synth_started_at:.0f}s."
        raise HTTPException(429, BUSY_MESSAGE + elapsed)

    raw = await reference.read()
    if not raw:
        raise HTTPException(400, "reference audio is required")
    try:
        ref_audio, ref_sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"could not decode the reference audio: {exc}") from exc
    if ref_audio.ndim > 1:
        ref_audio = ref_audio.mean(axis=1)

    # Measured on THIS machine, which is the fix as much as the offload is: the
    # client's copy of this guard was reading the memory of a machine that was no
    # longer doing the work.
    free = _free_gib()
    needed = _required_gib(spoken)
    if free < needed:
        raise HTTPException(
            507,
            f"The TTS service machine has only {free:.1f} GiB free and this request "
            f"needs roughly {needed:.1f} GiB for its longest sentence. Close what is "
            "holding memory there, or send shorter sentences. Starting anyway risks "
            "the model paging out mid-inference, which neither finishes nor errors.",
        )

    parts: list[np.ndarray] = []
    sample_rate = 24_000
    started = time.perf_counter()
    try:
        async with _synth_lock:
            _synth_started_at = time.perf_counter()
            engine = await _ensure_engine()
            # The single trim happens here. The client deliberately sends the
            # untrimmed transcript so that this -- the same code that reads the
            # audio -- is the only place the 12-second rule is applied.
            engine.set_reference(ref_audio, reference_text, ref_sr)
            started = time.perf_counter()
            async for chunk in engine.synthesize(spoken):
                parts.append(chunk.samples)
                sample_rate = chunk.sample_rate
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"synthesis failed: {exc}") from exc
    finally:
        _synth_started_at = None

    if not parts:
        raise HTTPException(500, "no audio was produced")

    audio = np.concatenate(parts)
    elapsed_ms = (time.perf_counter() - started) * 1000
    seconds = len(audio) / sample_rate

    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")

    return Response(
        content=buffer.getvalue(),
        media_type="audio/wav",
        headers={
            "X-Synthesis-Ms": f"{elapsed_ms:.0f}",
            "X-Audio-Seconds": f"{seconds:.2f}",
            "X-Realtime-Factor": f"{(elapsed_ms / 1000) / seconds:.2f}" if seconds else "0",
            "X-Engine": "indicf5",
        },
    )


def main() -> int:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Indic TTS service (run me on the spare machine)")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Interface to bind. Defaults to loopback: exposing a port that accepts "
            "voice clips has to be an explicit choice. Pass 0.0.0.0 to serve the LAN."
        ),
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    if not _expected_token():
        print(f"\n{NO_TOKEN_HELP}\n")
        return 1

    where = "this machine only" if args.host.startswith("127.") else f"the LAN on {args.host}"
    print(f"\n  Indic TTS service -> http://{args.host}:{args.port}  (serving {where})")
    print("  Model loads on the first request and then stays resident.\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
