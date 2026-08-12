"""Reference handling for the Indic cloning engine.

The transcript sets the output length. F5 estimates duration as
(generated text length / reference text length) x reference duration, so a
transcript that describes more speech than the model actually hears makes the
output too short, and syllables get swallowed to fit. Measured on a 21.1s clip
whose transcript described all 21.1s, round-trip overlap with Hindi pinned:

    hard-cut audio to 12s, full transcript    -> 88%, 2.89s (rushed, सहुना)
    whole audio,           full transcript    -> 82%, 2.76s (worse)
    whole audio,           trimmed transcript -> 95%, 5.00s (verbatim)

No model is loaded here; this covers the arithmetic that feeds it.
"""

from __future__ import annotations

import numpy as np
import pytest

from voiceagent.tts.indic_engine import REFERENCE_CLIP_SECONDS, IndicTTSEngine

SR = 24_000


def clip(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def test_short_clip_is_passed_through_untouched():
    """Clips within the limit must behave exactly as before this change, or the
    whole existing 22-sentence gate moves for no reason."""
    engine = IndicTTSEngine()
    text = "नमस्ते, मेरा नाम लेखा है।"
    audio = clip(8.26)  # the tracked Hindi fixture's length
    engine.set_reference(audio, text, SR)

    assert engine.reference_text == text
    assert len(engine.reference_audio) == len(audio)


def test_long_clip_keeps_all_audio():
    """The audio is handed over whole so f5_tts can clip at a silence boundary,
    which is better than a hard cut mid-word."""
    engine = IndicTTSEngine()
    audio = clip(21.1)
    engine.set_reference(audio, "एक दो तीन चार पाँच छह सात आठ नौ दस", SR)
    assert len(engine.reference_audio) == len(audio)


def test_long_clip_trims_the_transcript_proportionally():
    engine = IndicTTSEngine()
    words = [f"शब्द{i}" for i in range(20)]
    engine.set_reference(clip(24.0), " ".join(words), SR)

    # 12 of 24 seconds is kept, so about half the words should remain.
    kept = engine.reference_text.split()
    assert len(kept) == pytest.approx(10, abs=1)
    # Trimmed from the front, in order -- not sampled or reordered.
    assert kept == words[: len(kept)]


def test_trimming_keeps_the_chars_per_second_ratio_honest():
    """This ratio is the thing F5 actually consumes."""
    engine = IndicTTSEngine()
    seconds, text = 21.0, " ".join(f"शब्द{i}" for i in range(42))
    engine.set_reference(clip(seconds), text, SR)

    original_rate = len(text) / seconds
    effective_rate = len(engine.reference_text) / REFERENCE_CLIP_SECONDS
    # Word-boundary trimming makes this approximate, not exact.
    assert effective_rate == pytest.approx(original_rate, rel=0.2)


def test_a_long_clip_is_reported_not_silently_accepted():
    engine = IndicTTSEngine()
    engine.set_reference(clip(21.1), "एक दो तीन चार पाँच छह सात आठ नौ दस", SR)
    warning = engine.reference_health()
    assert warning is not None
    assert "21.1s" in warning and "12s" in warning


def test_a_good_clip_produces_no_warning():
    engine = IndicTTSEngine()
    # ~13 chars/sec, comfortably inside the plausible speech band.
    engine.set_reference(clip(8.0), "आज मौसम बहुत सुहावना है और आसमान साफ़ है।" * 2, SR)
    assert engine.reference_health() is None


def test_transcript_shorter_than_its_audio_is_still_caught():
    """The original failure this check existed for: a transcript that
    under-describes its audio inflates the output length."""
    engine = IndicTTSEngine()
    engine.set_reference(clip(10.0), "हाँ।", SR)
    warning = engine.reference_health()
    assert warning is not None and "too short" in warning


def test_never_trims_to_nothing():
    engine = IndicTTSEngine()
    engine.set_reference(clip(300.0), "नमस्ते", SR)
    assert engine.reference_text.strip()
