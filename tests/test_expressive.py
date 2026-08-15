"""The expressiveness sweep, and the one thing it must never do.

The sweep exists because `EXAGGERATION = 0.7` was inherited and never compared
against anything. The risk in building it is not that it renders badly --- it is
that a number falls out of it and gets read as a ranking. Round-trip overlap is
blind to prosody, Phase 2 measured it at AUC 0.625 among working systems, and
the sweep computes it per clip. So most of these tests are about keeping that
number in its lane.
"""

from __future__ import annotations

from voiceagent.eval import expressive


def row(condition="shipped", slug="h1", overlap=0.9, rtf=0.6):
    return expressive.Rendered(
        condition=condition,
        slug=slug,
        path=f"audio/{condition}__{slug}.wav",
        seconds=4.0,
        synthesis_seconds=4.0 * rtf,
        rtf=rtf,
        overlap=overlap,
    )


# --- the alarm, and its limits -------------------------------------------


def test_a_mangled_clip_is_suspect():
    assert row(overlap=0.31).suspect


def test_a_clip_that_kept_its_words_is_not():
    assert not row(overlap=0.88).suspect


def test_an_unscored_clip_is_not_suspect():
    """A scorer that failed to run is not evidence that a clip is bad.

    `None` and `0.0` must not collapse: zero is a disqualifying value and would
    silently drop a clip that was never actually judged.
    """
    assert not row(overlap=None).suspect


def test_the_threshold_is_the_one_phase_2_validated():
    """0.5, because every arena clip below it was rejected by human raters too
    (11 of 11). Not a round number someone liked."""
    assert expressive.DISQUALIFYING_OVERLAP == 0.5


# --- the grid -------------------------------------------------------------


def test_the_shipped_setting_is_in_the_grid():
    """A sweep that cannot reproduce the current default cannot tell you whether
    moving off it is an improvement."""
    shipped = [c for c in expressive.CONDITIONS if c["name"] == "shipped"]
    assert len(shipped) == 1
    from voiceagent.tts.chatterbox_indic import EXAGGERATION, TEMPERATURE

    assert shipped[0]["exaggeration"] == EXAGGERATION
    assert shipped[0]["temperature"] == TEMPERATURE


def test_exaggeration_is_bracketed_in_both_directions():
    """Only testing upward would answer "is 0.9 worse" and never "is 0.5 better"."""
    shipped = next(c for c in expressive.CONDITIONS if c["name"] == "shipped")
    values = [c["exaggeration"] for c in expressive.CONDITIONS]
    assert any(v < shipped["exaggeration"] for v in values)
    assert any(v > shipped["exaggeration"] for v in values)


def test_one_condition_varies_temperature_alone():
    """Liveliness and variability are easy to confuse by ear and fail
    differently, so at least one point has to move temperature by itself."""
    shipped = next(c for c in expressive.CONDITIONS if c["name"] == "shipped")
    off_axis = [
        c
        for c in expressive.CONDITIONS
        if c["exaggeration"] == shipped["exaggeration"]
        and c["temperature"] != shipped["temperature"]
    ]
    assert off_axis


def test_condition_names_are_unique():
    """Names become `abtest` system ids; a collision would silently merge two
    settings into one score."""
    names = [c["name"] for c in expressive.CONDITIONS]
    assert len(names) == len(set(names))


def test_the_grid_stays_small_enough_to_listen_to():
    """Every condition multiplies listening time, and an inattentive listener
    produces worse data than one who was never asked."""
    assert len(expressive.CONDITIONS) <= 6
    assert len(expressive.DEFAULT_SLUGS) <= 8


def test_the_sentences_are_real_held_out_slugs():
    from voiceagent.eval import heldout

    for slug in expressive.DEFAULT_SLUGS:
        assert heldout.by_slug(slug).text
