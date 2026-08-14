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

import soundfile as sf

WHISPER_REPO = "mlx-community/whisper-large-v3-turbo"


def transcribe(path: str | Path, language: str | None = None) -> tuple[str, str]:
    """Return (text, detected_language) for an audio file.

    `language` pins the decode. Left as None it auto-detects, which is what makes
    this a real check -- a wrong-language result is the signal. Pin it only to
    re-decode into a specific script (see EQUIVALENT_LANGUAGES).
    """
    import mlx_whisper

    from voiceagent.eval.audio import resample

    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Band-limited, not index-dropping. This is the input to the only automated
    # ground truth the project has, and dropping samples folded everything above
    # 8 kHz back on top of the speech it was about to be judged on.
    audio = resample(audio, sr, 16_000)

    result = mlx_whisper.transcribe(
        audio, path_or_hf_repo=WHISPER_REPO, fp16=True, verbose=None, language=language
    )
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


#: Languages Whisper may legitimately return for correct Hindi speech.
#:
#: Hindi and Urdu are the same spoken language (Hindustani) in two scripts, so
#: Whisper can transcribe perfectly good Hindi audio into Perso-Arabic and label
#: it "ur". This is not a hypothetical: a correct rendering of
#: "आज मौसम बहुत सुहावना है..." came back as
#: "آج موسم بہت سہاونا ہے اور آسمان بلکل صاف ہے" -- the same sentence, word for
#: word -- and scored 0% overlap purely because the scripts differ. Re-running it
#: with the language pinned to Hindi scored 95%.
#:
#: Treating "ur" as a failure would reject good audio, so it is accepted as an
#: alias and the transcript is re-decoded with Hindi pinned to score it.
EQUIVALENT_LANGUAGES: dict[str, frozenset[str]] = {
    "hi": frozenset({"hi", "ur"}),
}


def normalized(text: str, language: str | None) -> str:
    """Both sides of the comparison, in the script the model actually speaks.

    Without this the check scores its own text handling instead of the audio. Two
    ways it goes wrong, and both are silent:

      - Whisper writes loanwords in Devanagari. Expected text keeps them in Latin,
        so "एक documentary देखी" is compared against "एक डॉक्यूमेंटरी देखी" and loses
        every character of the English word. The engine transliterates before
        synthesis, so Devanagari is the correct side to compare on.
      - Whisper applies inverse text normalization, so a correct
        "एक हज़ार दो सौ निन्यानवे" comes back as "1299".

    Measured: a flawless human reading of the code-mixed held-out sentence scored
    54 % raw. A human recording is intelligible by construction, which is what
    identifies that number as the scorer's floor rather than the speaker's.

    `hindi_tts` has always done this; `check` never did, and `check` is the one the
    README tells you to reach for on a single file.
    """
    if language != "hi":
        return text
    from voiceagent.text.normalize_hi import normalize

    return normalize(text)


def check(path: str | Path, expected: str, expect_language: str | None = None) -> bool:
    heard, language = transcribe(path)

    accepted = EQUIVALENT_LANGUAGES.get(expect_language or "", frozenset())
    if expect_language and language in accepted and language != expect_language:
        # Same language, other script. Re-decode pinned so the comparison is
        # script-for-script rather than scoring Devanagari against Perso-Arabic.
        heard, _ = transcribe(path, language=expect_language)
        language = expect_language

    scored_expected = normalized(expected, expect_language)
    scored_heard = normalized(heard, expect_language)
    overlap = character_overlap(scored_expected, scored_heard)

    print(f"file     : {path}")
    print(f"expected : {expected}")
    print(f"heard    : {heard}")
    if scored_expected != expected or scored_heard != heard:
        print(f"compared : {scored_expected}")
        print(f"       vs : {scored_heard}")
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
