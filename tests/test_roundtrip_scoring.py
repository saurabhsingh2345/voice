"""What the round-trip check compares, before any audio is involved.

The scoring half is pure text and cheap to pin down; the Whisper half is not tested
here. It is worth pinning because the failure mode is a *low* score on correct
audio, which reads as "the model is bad" rather than as a bug, and the project has
already changed its mind once on the strength of a number like that.
"""

from __future__ import annotations

from voiceagent.eval.roundtrip import character_overlap, normalized

# The code-mixed held-out sentence, and what Whisper returns for a flawless human
# reading of it. Verbatim from eval_out/benchmark_samples/real/h1.wav.
EXPECTED = "मैंने कल रात एक documentary देखी जो climate change के बारे में थी।"
HEARD = "मैंने कल रात एक डॉक्यूमेंटरी देखी जो क्लाइमेट चेंज के बारे में थी"


def test_latin_loanwords_are_compared_in_the_script_the_model_speaks():
    """The engine transliterates before synthesis, so the audio says डॉक्यूमेंटरी and
    can never say "documentary". Comparing against the Latin form scores the
    project's own text handling and calls it intelligibility."""
    raw = character_overlap(EXPECTED, HEARD)
    fixed = character_overlap(normalized(EXPECTED, "hi"), normalized(HEARD, "hi"))
    assert raw < 0.6, "the bug: a perfect human recording scored 54%"
    assert fixed > 0.85
    assert fixed > raw


def test_normalization_is_applied_to_both_sides_or_neither():
    """Whisper applies inverse text normalization, so it writes "43" where the
    speaker said "तैंतालीस". Normalizing only the expected side leaves the digits on
    the heard side and scores a correct reading as a miss — which is the shape of
    the bug that once made `nfe_step` look like it helped."""
    expected = "इस साल टीम ने तैंतालीस प्रोजेक्ट पूरे किए।"
    heard = "इस साल टीम ने 43 प्रोजेक्ट पूरे किए"

    one_sided = character_overlap(normalized(expected, "hi"), heard)
    both = character_overlap(normalized(expected, "hi"), normalized(heard, "hi"))
    assert both > one_sided
    assert both > 0.95


def test_a_non_hindi_check_is_left_alone():
    """`normalize_hi` is Hindi-specific; running it on English would corrupt the
    comparison it is supposed to protect."""
    english = "the quick brown fox"
    assert normalized(english, "en") == english
    assert normalized(english, None) == english


def test_identical_text_still_scores_one():
    assert character_overlap(normalized(HEARD, "hi"), normalized(HEARD, "hi")) == 1.0


def test_unrelated_text_still_scores_low():
    """The fix must not make everything pass. Babble in another script was the
    original thing this check existed to catch."""
    babble = "Terima kasih."
    assert character_overlap(normalized(EXPECTED, "hi"), normalized(babble, "hi")) < 0.2
