"""Tests for the cost model.

The arithmetic here decides a price, so the parts worth testing are the ones a
pricing decision would be read off: that the fit separates fixed from marginal
cost, that batching arithmetic is right, and above all that the model knows when
it is extrapolating. The first run of this module produced a confident cost for a
5,000-character request from data that topped out at 142, and that is the failure
mode `in_range` exists to make visible.
"""

from __future__ import annotations

import pytest

from voiceagent.eval.cogs import CostModel, Sample, fit, sweep_text


def _model(fixed: float, per_char: float, lo: int = 0, hi: int = 10_000) -> CostModel:
    return CostModel(
        fixed_seconds=fixed,
        marginal_seconds_per_char=per_char,
        samples=10,
        min_chars=lo,
        max_chars=hi,
        chars_per_audio_second=14.8,
    )


def test_the_fit_recovers_a_known_line():
    """Synthetic data with no noise: 2 s fixed, 0.04 s per character."""
    samples = [
        Sample(characters=n, audio_seconds=n / 14.8, synthesis_seconds=2.0 + 0.04 * n)
        for n in (50, 200, 800, 2000)
    ]
    model = fit(samples)
    assert model.fixed_seconds == pytest.approx(2.0, abs=1e-6)
    assert model.marginal_seconds_per_char == pytest.approx(0.04, abs=1e-9)


def test_a_fit_over_one_length_is_refused_rather_than_guessed():
    """Every sample the same size cannot separate a fixed term from a slope, and
    silently returning one would be the exact error this module warns about."""
    samples = [Sample(100, 6.8, 4.0), Sample(100, 6.8, 4.3)]
    with pytest.raises(ValueError, match="same length"):
        fit(samples)


def test_batching_arithmetic_counts_the_overhead_once_per_request():
    """10,000 characters as one call pays the fixed cost once; as 250 calls it
    pays it 250 times. This is the whole pricing argument in one assertion."""
    model = _model(fixed=3.0, per_char=0.02)
    one_call = model.machine_seconds_per_10k(10_000)
    many_calls = model.machine_seconds_per_10k(40)

    assert one_call == pytest.approx(3.0 + 200.0)
    assert many_calls == pytest.approx(250 * (3.0 + 0.8))
    assert many_calls > one_call


def test_with_no_fixed_cost_batching_stops_mattering():
    """The corrective case. When the fixed term is ~0 the cost is proportional to
    characters and request size is irrelevant -- which is what the sweep actually
    measured, and why the narrow metering fit's batching penalty was an artefact."""
    model = _model(fixed=0.0, per_char=0.02)
    assert model.machine_seconds_per_10k(40) == pytest.approx(
        model.machine_seconds_per_10k(5_000)
    )


def test_the_model_knows_when_it_is_extrapolating():
    model = _model(fixed=1.0, per_char=0.02, lo=56, hi=2036)
    assert model.in_range(400)
    assert model.in_range(56)
    assert model.in_range(2036)
    assert not model.in_range(40)
    assert not model.in_range(5_000)


def test_cost_scales_with_the_assumed_tariff_and_capacity_does_not():
    """The asymmetry the module is built around: rupee figures rest on two
    unmeasured constants, capacity figures rest on none of them."""
    from voiceagent.eval import cogs

    model = _model(fixed=1.0, per_char=0.02)
    before_inr = model.inr_per_10k(1_000)
    before_chars = model.chars_per_hour(1_000)

    original = cogs.ASSUMED_TARIFF_INR_PER_KWH
    try:
        cogs.ASSUMED_TARIFF_INR_PER_KWH = original * 3
        assert model.inr_per_10k(1_000) == pytest.approx(before_inr * 3)
        assert model.chars_per_hour(1_000) == pytest.approx(before_chars)
    finally:
        cogs.ASSUMED_TARIFF_INR_PER_KWH = original


@pytest.mark.parametrize("target", [50, 150, 400, 1000, 2000])
def test_sweep_text_reaches_its_target_without_repeating_one_clause(target):
    """A model given the same sentence forty times is a different workload from
    one given prose, and the cheap version of this benchmark would flatter
    itself."""
    text = sweep_text(target)
    assert len(text) >= target
    # Whole sentences, so it overshoots -- but never by more than one of them.
    assert len(text) < target + 80
    assert text == text.strip()


def test_sweep_text_is_devanagari():
    """If any of this reaches the engine as Latin, something upstream romanized
    it and the measurement is of the wrong thing."""
    text = sweep_text(400)
    devanagari = sum(1 for c in text if "ऀ" <= c <= "ॿ")
    assert devanagari / sum(1 for c in text if c.isalpha()) > 0.95


def test_a_narrow_fit_knows_it_cannot_separate_fixed_from_marginal():
    """The guard that exists because this module got it wrong twice. A fit over
    a 5x span of request sizes reported 3.12 s of fixed overhead and a 4.6x
    batching penalty; a 36x sweep put the same term near zero and the penalty at
    1.1x. Range, not sample count, is what identifies the slope."""
    narrow = _model(fixed=3.12, per_char=0.0195, lo=29, hi=142)
    wide = _model(fixed=0.20, per_char=0.0415, lo=56, hi=2036)

    assert narrow.range_ratio == pytest.approx(142 / 29, rel=1e-3)
    assert not narrow.separates_fixed_from_marginal
    assert wide.separates_fixed_from_marginal


def test_the_narrow_fit_refuses_to_publish_a_batching_verdict():
    """A findings file is read by someone making a price. It must not state a
    conclusion the data cannot carry."""
    from voiceagent.eval.cogs import _capacity_lines

    narrow = "\n".join(_capacity_lines(_model(3.12, 0.0195, lo=29, hi=142)))
    wide = "\n".join(_capacity_lines(_model(0.20, 0.0415, lo=56, hi=2036)))

    assert "No verdict on batching" in narrow
    assert "--sweep" in narrow

    # The wide fit gets further, but not to a confident answer: the batching
    # penalty is reported as the band the unresolved intercept implies, never as
    # a single number. Three different confident answers came out of this fit
    # before it was made to say "unresolved".
    assert "No verdict on batching" not in wide
    assert "Unresolved, and bounded between" in wide
    assert "charge per character" in wide


def test_neither_fit_ever_states_a_single_batching_multiplier():
    """The regression that matters. Every earlier version of this module wrote a
    confident penalty into a findings file -- 4.6x, then 1.0x -- read off an
    intercept that moves with machine load."""
    from voiceagent.eval.cogs import _capacity_lines

    for model in (_model(3.12, 0.0195, lo=29, hi=142),
                  _model(0.20, 0.0415, lo=56, hi=2036),
                  _model(3.27, 0.0419, lo=56, hi=2036)):
        text = "\n".join(_capacity_lines(model))
        assert "is the point: identical revenue" not in text
        assert "batching is not a pricing variable" not in text


def test_a_load_spike_in_one_bucket_does_not_rewrite_the_fit():
    """The failure this module actually hit. One sweep's 430-char bucket ran at
    RTF 0.97 instead of 0.59 and moved the fitted intercept from 0.20 s to
    3.27 s by itself. Medians over repeats are what stop a busy machine
    rewriting a pricing conclusion."""
    from voiceagent.eval.cogs import by_bucket_median

    clean = [(56, 2.8), (166, 7.0), (430, 17.1), (1046, 44.0), (2036, 87.8)]
    samples = []
    for chars, seconds in clean:
        samples.append(Sample(chars, chars / 14.8, seconds))
        samples.append(Sample(chars, chars / 14.8, seconds))
        # A third, spiked, repeat for one bucket only.
        spike = seconds * 1.7 if chars == 430 else seconds
        samples.append(Sample(chars, chars / 14.8, spike))

    naive = fit(samples)
    damped = fit(by_bucket_median(samples))

    # The median throws the spike away entirely; the naive fit does not.
    assert damped.fixed_seconds < naive.fixed_seconds
    assert damped.fixed_seconds == pytest.approx(fit([
        Sample(c, c / 14.8, s) for c, s in clean
    ]).fixed_seconds, abs=1e-6)


def test_by_bucket_median_returns_one_sample_per_length():
    from voiceagent.eval.cogs import by_bucket_median

    samples = [Sample(100, 6.8, 4.0), Sample(100, 6.8, 4.4), Sample(500, 34.0, 20.0)]
    collapsed = by_bucket_median(samples)
    assert [s.characters for s in collapsed] == [100, 500]
    assert collapsed[0].synthesis_seconds == pytest.approx(4.2)
