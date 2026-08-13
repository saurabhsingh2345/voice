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


# --- narration: grouping and joins ----------------------------------------


def test_sentences_are_grouped_into_one_f5_batch():
    """One f5-tts call per sentence was wrong twice over: four raw joins in a
    five-sentence passage, and the reference conditioning plus a full diffusion run
    paid per call. Measured 45 s per-sentence against 17 s grouped for the same
    10.2 s of audio and the same 64% round-trip score."""
    from voiceagent.tts.indic_engine import group_sentences

    sentences = ["नमस्ते।", "आज मौसम सुहावना है।", "आसमान साफ़ है।", "मैं यहाँ हूँ।"]
    assert len(group_sentences(sentences, 465)) == 1


def test_grouping_respects_the_byte_budget():
    from voiceagent.tts.indic_engine import group_sentences

    sentences = ["आज मौसम बहुत सुहावना है।"] * 10
    spans = group_sentences(sentences, 100)
    assert len(spans) > 1
    # A span may exceed the budget only when a single sentence does.
    for span in spans:
        assert len(span.encode()) <= 100 or " " not in span.strip()


def test_an_oversized_sentence_is_not_split():
    """f5-tts batches it internally and cross-fades itself, which beats any cut we
    could make without knowing where the words are."""
    from voiceagent.tts.indic_engine import group_sentences

    long_one = "क " * 300 + "।"
    assert group_sentences([long_one], 100) == [long_one]


def test_the_budget_is_derived_from_the_reference_speaking_rate():
    from voiceagent.tts.indic_engine import batch_budget_bytes

    slow = batch_budget_bytes("नमस्ते।" * 5, 10.0)
    fast = batch_budget_bytes("नमस्ते।" * 20, 10.0)
    assert fast > slow, "a faster reference should allow more text per batch"


def test_crossfade_gains_sum_to_one():
    """Linear, matching f5-tts's own internal joins. Equal-power ramps sum to 1.414
    on correlated spans, and this model's output already peaks at 1.000 -- so the
    textbook choice would clip rather than smooth."""
    import numpy as np

    from voiceagent.tts.indic_engine import concat_with_crossfade

    ones = np.ones(24_000, dtype=np.float32)
    joined = concat_with_crossfade([ones, ones], 24_000)
    assert joined.max() <= 1.0 + 1e-6, "cross-fade must not push correlated spans over full scale"


def test_crossfade_shortens_by_the_overlap():
    import numpy as np

    from voiceagent.tts.indic_engine import CROSS_FADE_SECONDS, concat_with_crossfade

    a = np.ones(24_000, dtype=np.float32)
    joined = concat_with_crossfade([a, a], 24_000)
    assert len(joined) == 2 * 24_000 - int(CROSS_FADE_SECONDS * 24_000)


def test_crossfade_handles_a_single_span_and_none():
    import numpy as np

    from voiceagent.tts.indic_engine import concat_with_crossfade

    a = np.ones(10, dtype=np.float32)
    assert len(concat_with_crossfade([a], 24_000)) == 10
    assert len(concat_with_crossfade([], 24_000)) == 0


def test_overshoot_is_scaled_down_not_clipped():
    """IndicF5 returned a peak of 1.335 on a 10-sentence narration, and PCM_16
    hard-clips anything over 1.0 (0.04% of samples measured). Scale, don't clamp."""
    import numpy as np

    from voiceagent.tts.indic_engine import concat_with_crossfade

    hot = (np.linspace(-1.335, 1.335, 24_000)).astype(np.float32)
    out = concat_with_crossfade([hot], 24_000)
    assert np.abs(out).max() <= 0.99 + 1e-6
    # shape preserved -- scaled, not squashed
    assert abs(float(out[0] / out[-1]) + 1.0) < 1e-3


def test_quiet_audio_is_not_normalised_up():
    """Otherwise level would depend on whatever the loudest moment happened to be,
    and the same text would come back at different volumes run to run."""
    import numpy as np

    from voiceagent.tts.indic_engine import concat_with_crossfade

    quiet = (np.ones(1000, dtype=np.float32) * 0.05)
    assert float(np.abs(concat_with_crossfade([quiet], 24_000)).max()) == pytest.approx(0.05)
