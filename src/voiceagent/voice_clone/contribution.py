"""Checking a contributed recording before it is allowed into a training set.

The flywheel: someone generates a line in their own cloned voice, then reads the
same line aloud and sends the recording back. Every pair is `(text, synthetic,
real)` for one speaker on one sentence --- the corpus a fine-tune wants, gathered
from ordinary use, with consent already attached because the contributor is the
speaker.

The failure that makes the whole idea worthless is silent: **a clip whose
transcript is wrong**. `dataset.add_clip` already says why --- the model learns
the mapping from text to voice, so an untranscribed clip teaches it nothing and a
wrongly transcribed one teaches it something false. Collected at scale from
people reading lines off a screen, that will happen constantly: a misread word,
a repeated take, a sentence abandoned halfway, a phone that recorded a second of
silence.

Nothing downstream catches it. Training loss falls just the same. So the check
belongs at the door.

**Round-trip overlap is the right instrument here, and this is the one job it is
genuinely good at.** Phase 2 measured it at AUC 0.625 among working systems and
it must never rank quality --- but it separates *broken* from *working* reliably,
and "did this person read this sentence" is exactly a broken-or-not question.
Using it to reject a mismatched recording is sound; using the same number to say
one contribution is better than another would not be.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Below this, the recording and the text are not the same sentence.
#:
#: Deliberately well under the ~0.90 ceiling a *flawless* human recording scores
#: against this scorer --- the point is to catch someone reading a different
#: line, not to grade diction. A contributor with a strong regional accent, a
#: cheap microphone or a noisy room must pass; someone who read sentence four
#: while sentence five was on screen must not.
MIN_OVERLAP = 0.55

#: A recording this much longer or shorter than the synthetic take is suspicious
#: even when the words match --- usually a take with a long silence, a false
#: start, or the sentence read twice. Advisory only: it is returned to the
#: caller, never used to reject, because an unhurried reader is not an error.
DURATION_RATIO_BOUNDS = (0.5, 2.5)


@dataclass(frozen=True)
class Verdict:
    """Whether a contributed recording may join the training set, and why."""

    accepted: bool
    overlap: float
    heard: str
    reason: str = ""
    duration_ratio: float | None = None

    @property
    def unusual_length(self) -> bool:
        if self.duration_ratio is None:
            return False
        low, high = DURATION_RATIO_BOUNDS
        return not (low <= self.duration_ratio <= high)


def verify_recording(
    path: str | Path,
    text: str,
    language: str = "hi",
    synthetic_seconds: float | None = None,
    recorded_seconds: float | None = None,
) -> Verdict:
    """Decide whether `path` is a recording of `text`.

    Transcribes with the language pinned. Auto-detect is not trusted here: a
    short clip has been mislabelled before --- a 1.7 s Hindi recording came back
    as Korean and scored 0 % --- and contributed lines are often short.
    """
    from voiceagent.eval.roundtrip import character_overlap, decode_for_scoring, normalized

    heard, _language, _note = decode_for_scoring(path, language)
    overlap = character_overlap(normalized(text, language), normalized(heard, language))

    ratio = None
    if synthetic_seconds and recorded_seconds and synthetic_seconds > 0:
        ratio = recorded_seconds / synthetic_seconds

    if overlap < MIN_OVERLAP:
        return Verdict(
            accepted=False,
            overlap=overlap,
            heard=heard,
            duration_ratio=ratio,
            reason=(
                "That recording does not match the line on screen. What came "
                f"through was “{heard.strip()[:80]}”. Re-record the "
                "sentence as written --- a clip stored against the wrong text "
                "teaches the voice something false, and nothing later catches it."
            ),
        )

    return Verdict(accepted=True, overlap=overlap, heard=heard, duration_ratio=ratio)
