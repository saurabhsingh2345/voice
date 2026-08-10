"""Is the synthesized speech actually intelligible?

    uv run python -m voiceagent.eval.roundtrip out.wav "the text it should say"

Synthesize, then transcribe the result with Whisper and compare. Spectral
measures cannot answer this: unintelligible IndicF5 output measured 0.06-0.10
spectral flatness against 0.088 for real speech, i.e. indistinguishable from
genuine speech by that metric while being babble. Whisper transcribing it as
"Terima kasih." (Indonesian) settled the question in one step.

Use this before claiming any TTS change improved anything. It is the only check
here that does not need a human ear.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

WHISPER_REPO = "mlx-community/whisper-large-v3-turbo"


def transcribe(path: str | Path) -> tuple[str, str]:
    """Return (text, detected_language) for an audio file."""
    import mlx_whisper

    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16_000:
        idx = (np.arange(int(len(audio) * 16_000 / sr)) * sr / 16_000).astype(int)
        audio = audio[idx[idx < len(audio)]]

    result = mlx_whisper.transcribe(audio, path_or_hf_repo=WHISPER_REPO, fp16=True, verbose=None)
    return result.get("text", "").strip(), result.get("language", "?")


def character_overlap(expected: str, heard: str) -> float:
    """Crude similarity: share of expected characters present in the output.

    Deliberately not a WER. The failure being detected is total -- the output is
    a different language -- so a cheap measure separates it from a good result
    without pulling in an alignment library.
    """
    expected_chars = [c for c in expected if not c.isspace()]
    if not expected_chars:
        return 0.0
    pool = list(heard)
    hits = 0
    for char in expected_chars:
        if char in pool:
            pool.remove(char)
            hits += 1
    return hits / len(expected_chars)


def check(path: str | Path, expected: str, expect_language: str | None = None) -> bool:
    heard, language = transcribe(path)
    overlap = character_overlap(expected, heard)

    print(f"file     : {path}")
    print(f"expected : {expected}")
    print(f"heard    : {heard}")
    print(f"language : {language}" + (f" (expected {expect_language})" if expect_language else ""))
    print(f"overlap  : {overlap:.0%}")

    ok = overlap >= 0.5 and (expect_language is None or language == expect_language)
    print("verdict  : " + ("INTELLIGIBLE" if ok else "NOT INTELLIGIBLE"))
    return ok


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    expect_language = sys.argv[3] if len(sys.argv) > 3 else None
    return 0 if check(sys.argv[1], sys.argv[2], expect_language) else 1


if __name__ == "__main__":
    raise SystemExit(main())
