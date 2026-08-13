"""The test set must not leak into training.

A benchmark run on sentences the model was fine-tuned on measures memorisation and
reports it as quality — and it reports it *favourably*, which is the dangerous
direction. The disjointness is asserted here rather than trusted, because the pools
are edited by hand and a copy-paste between files would be invisible.
"""

from __future__ import annotations

import re

import pytest

from voiceagent.eval import heldout
from voiceagent.eval import sentences as S
from voiceagent.train import prompts as P


def normalize(text: str) -> str:
    """Compare on content, ignoring punctuation and spacing.

    A sentence differing from a training one only by a comma is still the same
    sentence for a model, so exact-string comparison would miss the leak it matters
    most to catch.
    """
    return re.sub(r"[\s।॥.,;:!?—-]+", "", text)


def training_texts() -> set[str]:
    """Transcripts actually used to fine-tune, if a dataset is present."""
    from voiceagent.voice_clone.dataset import VoiceDataset

    dataset = VoiceDataset()
    out: set[str] = set()
    for profile in dataset.profiles.list():
        for clip in dataset.clips(profile.profile_id):
            out.add(normalize(clip.text))
    return out


def test_held_out_sentences_do_not_appear_in_the_prompt_pools():
    pools = {normalize(s.text) for s in S.ALL} | {normalize(p.text) for p in P.ALL}
    leaked = [s.slug for s in heldout.SENTENCES if normalize(s.text) in pools]
    assert not leaked, f"these held-out sentences are also prompts: {leaked}"


def test_held_out_sentences_do_not_appear_in_the_training_data():
    """Skips cleanly when there is no local dataset, so CI does not depend on it."""
    trained = training_texts()
    if not trained:
        pytest.skip("no local training clips to check against")
    leaked = [s.slug for s in heldout.SENTENCES if normalize(s.text) in trained]
    assert not leaked, f"these held-out sentences were trained on: {leaked}"


def test_no_held_out_sentence_is_a_substring_of_a_training_transcript():
    """Catches the partial case: a long training clip containing a test sentence
    verbatim leaks it just as thoroughly as an exact match would."""
    trained = training_texts()
    if not trained:
        pytest.skip("no local training clips to check against")
    leaked = [
        s.slug
        for s in heldout.SENTENCES
        if any(normalize(s.text) in t for t in trained if len(t) > len(normalize(s.text)))
    ]
    assert not leaked, f"these appear inside a training transcript: {leaked}"


def test_the_set_is_internally_unique():
    slugs = [s.slug for s in heldout.SENTENCES]
    bodies = [normalize(s.text) for s in heldout.SENTENCES]
    assert len(set(slugs)) == len(slugs), "duplicate slugs"
    assert len(set(bodies)) == len(bodies), "duplicate sentences"


def test_every_sentence_states_what_it_targets():
    """A sentence with no stated failure mode is decoration, and a benchmark made of
    decoration cannot tell you what to fix."""
    for s in heldout.SENTENCES:
        assert len(s.targets) > 20, f"{s.slug} does not say what it exposes"


def test_the_documented_failure_modes_are_all_covered():
    """The reported weaknesses are code-mixing, numbers, long-form drift and
    region-specific phonemes. Missing one means the benchmark cannot detect it."""
    joined = " ".join(s.targets for s in heldout.SENTENCES).lower()
    for mode in ("code-mixing", "digits", "long form", "retroflex", "nuqta", "prosody"):
        assert mode in joined, f"nothing in the set targets {mode!r}"


def test_lengths_are_spread():
    """A set of uniformly short sentences cannot expose accent drift; a set of
    uniformly long ones hides onset and final-lengthening problems."""
    lengths = sorted(len(s.text) for s in heldout.SENTENCES)
    assert lengths[0] < 30, "no short utterance in the set"
    assert lengths[-1] > 80, "no long-form sentence in the set"


def test_code_mixed_sentences_actually_contain_latin():
    mixed = [s for s in heldout.SENTENCES if "code-mix" in s.targets]
    assert mixed, "no code-mixed sentences"
    for s in mixed:
        assert re.search(r"[A-Za-z]{3,}", s.text), f"{s.slug} claims code-mixing but has no Latin"


def test_digit_sentences_actually_contain_digits():
    numeric = [s for s in heldout.SENTENCES if "digit" in s.targets]
    assert numeric, "no digit sentences"
    for s in numeric:
        assert re.search(r"\d", s.text), f"{s.slug} claims digits but has none"
