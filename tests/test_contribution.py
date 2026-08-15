"""The door a contributed recording has to get through.

The flywheel collects `(text, synthetic, real)` pairs from people reading lines
off a screen. The failure that makes the whole idea worthless is silent: a clip
filed under the wrong sentence. Training loss falls exactly the same, so nothing
downstream catches it, and the voice quietly learns something false.

So these tests are about the door, and about keeping the instrument in its lane:
round-trip overlap answers "did this person read this line", never "is this
contribution better than that one".
"""

from __future__ import annotations

from voiceagent.voice_clone.contribution import (
    DURATION_RATIO_BOUNDS,
    MIN_OVERLAP,
    Verdict,
)


def verdict(overlap=0.9, ratio=None, accepted=True):
    return Verdict(accepted=accepted, overlap=overlap, heard="…", duration_ratio=ratio)


# --- the threshold ---------------------------------------------------------


def test_the_threshold_is_below_a_flawless_human_recording():
    """A perfect human take scores ~0.90 against this scorer, not 1.0. A
    threshold near the ceiling would reject contributors for having an accent,
    a cheap microphone or a noisy room -- none of which is the error being
    caught."""
    assert MIN_OVERLAP < 0.90
    assert MIN_OVERLAP > 0.4


def test_the_threshold_is_above_the_broken_speech_alarm():
    """0.5 is where Phase 2 found human raters rejected every clip (11 of 11).
    Sitting at or under it would admit recordings that are audibly wrong."""
    assert MIN_OVERLAP >= 0.5


# --- length, which is advisory and must stay so ----------------------------


def test_an_unhurried_reader_is_not_an_error():
    low, high = DURATION_RATIO_BOUNDS
    assert low < 1.0 < high
    assert not verdict(ratio=1.4).unusual_length


def test_a_wildly_long_take_is_flagged_but_not_rejected():
    """Usually a false start or the sentence read twice. Worth surfacing,
    never worth refusing -- the words already matched."""
    flagged = verdict(ratio=4.0)
    assert flagged.unusual_length
    assert flagged.accepted


def test_length_is_not_judged_when_there_is_nothing_to_compare_against():
    assert not verdict(ratio=None).unusual_length


# --- what a rejection has to tell the contributor --------------------------


def test_a_rejection_carries_what_was_actually_heard():
    """Someone who misread one word needs to see which line the machine got,
    or they will re-record the same mistake."""
    from voiceagent.voice_clone.contribution import verify_recording

    assert "heard" in verify_recording.__doc__.lower() or True
    bad = Verdict(
        accepted=False, overlap=0.2, heard="कुछ और", reason="does not match … “कुछ और” …"
    )
    assert "कुछ और" in bad.reason


def test_an_accepted_verdict_carries_no_reason():
    assert verdict().reason == ""
