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


# --- when the language label cannot be trusted -----------------------------
#
# These use a fake transcriber. The point is the decision logic, not Whisper.

import soundfile as sf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from voiceagent.eval import roundtrip  # noqa: E402


def _wav(tmp_path, seconds, name="clip.wav", rate=24_000):
    path = tmp_path / name
    sf.write(path, np.zeros(int(seconds * rate), dtype="float32"), rate)
    return path


@pytest.fixture
def fake_transcribe(monkeypatch):
    """Records how it was called, so a test can assert the pin actually happened."""
    calls = []

    def factory(auto_result, pinned_result):
        def fake(path, language=None):
            calls.append(language)
            return (pinned_result, language) if language else auto_result

        monkeypatch.setattr(roundtrip, "transcribe", fake)
        return calls

    return factory


def test_a_short_clip_labelled_a_wrong_language_is_re_decoded(tmp_path, fake_transcribe):
    """The h8 regression, exactly.

    "बिल्कुल, हो जाएगा।" at 1.7s auto-detected as Korean and scored 0%. The
    transcript 밀쿨 호자에가 romanises to "milkul hojaega" -- the target sentence,
    correctly synthesized, written in the wrong script by a detector with too
    little audio to go on. Pinned to Hindi the same file scores 88%.
    """
    calls = fake_transcribe(("밀쿨 호자에가", "ko"), "बिल्कुल हो जाएगा")
    heard, language, note = roundtrip.decode_for_scoring(_wav(tmp_path, 1.7), "hi")

    assert calls == [None, "hi"], "must re-decode, pinned, exactly once"
    assert heard == "बिल्कुल हो जाएगा"
    assert language == "hi"
    assert note and "too short" in note


def test_a_long_clip_labelled_a_wrong_language_is_left_alone(tmp_path, fake_transcribe):
    """The check still has teeth where there is enough audio to trust it. This is
    the babble case the harness exists to catch -- a Hindi sentence coming back as
    Arabic is how the rope-on-16-heads bug announced itself."""
    calls = fake_transcribe(("مرحبا بالعالم", "ar"), "should not be used")
    heard, language, note = roundtrip.decode_for_scoring(_wav(tmp_path, 6.0), "hi")

    assert calls == [None], "no re-decode: 6s is plenty to identify a language"
    assert language == "ar" and note is None
    assert heard == "مرحبا بالعالم"


def test_urdu_is_re_decoded_at_any_length(tmp_path, fake_transcribe):
    """Hindi and Urdu are one spoken language in two scripts, so this alias does
    not depend on clip length -- it is a predictable script swap, not a detector
    running out of evidence."""
    calls = fake_transcribe(("آج موسم بہت سہاونا ہے", "ur"), "आज मौसम बहुत सुहावना है")
    heard, language, note = roundtrip.decode_for_scoring(_wav(tmp_path, 9.0), "hi")

    assert calls == [None, "hi"]
    assert language == "hi" and heard.startswith("आज")
    assert note and "same language" in note


def test_a_correct_label_is_never_second_guessed(tmp_path, fake_transcribe):
    calls = fake_transcribe(("आज मौसम अच्छा है", "hi"), "unused")
    heard, language, note = roundtrip.decode_for_scoring(_wav(tmp_path, 1.0), "hi")

    assert calls == [None], "already the expected language; nothing to re-decode"
    assert (heard, language, note) == ("आज मौसम अच्छा है", "hi", None)


def test_no_expected_language_means_no_re_decoding(tmp_path, fake_transcribe):
    """Auto-detect is the whole point when the caller has not said what to expect."""
    calls = fake_transcribe(("whatever this is", "ko"), "unused")
    _, language, note = roundtrip.decode_for_scoring(_wav(tmp_path, 1.0), None)

    assert calls == [None]
    assert language == "ko" and note is None


def test_the_threshold_sits_between_the_observed_failures_and_successes():
    """Measured: failures at 1.66-2.02s, and every clip from 2.74s up decoded
    correctly on all five repeats. A threshold outside that gap is either still
    broken or needlessly blind."""
    assert 2.02 < roundtrip.SHORT_CLIP_SECONDS < 2.74


def test_hindi_tts_does_not_keep_its_own_copy_of_this():
    """It used to, and the two had diverged: hindi_tts accepted ("ur", "pa")
    while EQUIVALENT_LANGUAGES listed only "ur", so the same file could pass one
    harness and fail the other."""
    from pathlib import Path

    from voiceagent.eval import hindi_tts

    source = Path(hindi_tts.__file__).read_text()
    assert "decode_for_scoring" in source
    assert 'language="hi")' not in source, "no hand-rolled pinned re-decode"
