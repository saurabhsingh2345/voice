"""Does the Bradley-Terry fit actually recover the strengths it is shown?

The arena verdict rests entirely on this solver. If it is wrong, every number
downstream is wrong in a way that looks completely plausible -- a ranking is
seven numbers in a sensible-looking order whether or not the fit converged, and
there is no eyeball check that catches a subtly broken one.

So the tests here are recovery tests: build comparisons from *known* strengths,
fit, and require the fit to return what was put in. That is the only check that
distinguishes a working solver from one that merely produces a plausible order.

No network and no dataset: these run against synthetic preferences.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from voiceagent.eval.arena_bt import CENTRE, SCALE, Comparison, fit, format_table


def synthetic(strengths: dict[str, float], *, per_pair: int, seed: int = 0) -> list[Comparison]:
    """Sample comparisons from a true Bradley-Terry model.

    P(i beats j) = p_i / (p_i + p_j), which is the model the solver inverts. If
    the solver is right it gets `strengths` back up to an additive constant on
    the log scale (Bradley-Terry is only identified up to that constant, which
    is exactly why the fit re-centres on 1000).
    """
    rng = np.random.default_rng(seed)
    names = sorted(strengths)
    out: list[Comparison] = []
    for a_index, a in enumerate(names):
        for b in names[a_index + 1 :]:
            pa = strengths[a] / (strengths[a] + strengths[b])
            for _ in range(per_pair):
                out.append(Comparison(a, b, "a" if rng.random() < pa else "b"))
    return out


def test_recovers_known_strengths():
    truth = {"strong": 8.0, "middle": 4.0, "weak": 1.0}
    scores = fit(synthetic(truth, per_pair=4000))

    assert [s.name for s in scores] == ["strong", "middle", "weak"]

    # Bradley-Terry is identified up to an additive shift in log-space, so
    # compare *differences*, which are the identified quantity. The expected gap
    # is the log-strength ratio on the Elo scale.
    by_name = {s.name: s.score for s in scores}
    expected = SCALE * math.log(truth["strong"] / truth["weak"])
    assert by_name["strong"] - by_name["weak"] == pytest.approx(expected, abs=25)

    expected_mid = SCALE * math.log(truth["middle"] / truth["weak"])
    assert by_name["middle"] - by_name["weak"] == pytest.approx(expected_mid, abs=25)


def test_scores_are_centred_on_1000():
    """The paper's scale. Without this our numbers cannot sit beside Table 4."""
    scores = fit(synthetic({"a": 3.0, "b": 2.0, "c": 1.0}, per_pair=500))
    assert np.mean([s.score for s in scores]) == pytest.approx(CENTRE, abs=1e-6)


def test_equal_systems_score_equally():
    scores = fit(synthetic({"x": 1.0, "y": 1.0}, per_pair=5000))
    assert scores[0].score - scores[1].score == pytest.approx(0.0, abs=20)


def test_ties_are_half_credit_and_do_not_move_a_symmetric_pair():
    """A pile of ties is evidence of equality, not evidence for either side."""
    comparisons = [Comparison("x", "y", "both_good") for _ in range(100)]
    comparisons += [Comparison("x", "y", "both_bad") for _ in range(100)]
    scores = fit(comparisons)
    assert scores[0].score == pytest.approx(scores[1].score, abs=1e-6)
    assert all(s.win_rate == pytest.approx(0.5) for s in scores)


def test_win_rate_and_model_disagree_when_schedules_differ():
    """Why we fit a model instead of counting wins.

    `easy` only ever plays `weak`; `hard` only ever plays `strong`. Counting
    wins ranks `easy` above `hard`. The model knows `hard`'s opponent was better
    and is supposed to correct for it -- that correction is the whole point of
    Bradley-Terry over a leaderboard of win percentages.
    """
    comparisons: list[Comparison] = []
    # easy beats weak 7 times out of 10.
    comparisons += [Comparison("easy", "weak", "a") for _ in range(70)]
    comparisons += [Comparison("easy", "weak", "b") for _ in range(30)]
    # hard loses to strong 6 times out of 10 -- a worse record against a better
    # opponent.
    comparisons += [Comparison("hard", "strong", "a") for _ in range(40)]
    comparisons += [Comparison("hard", "strong", "b") for _ in range(60)]
    # The two halves of the graph have to be connected or the fit is not
    # identified: strong and weak meet, and strong dominates.
    comparisons += [Comparison("strong", "weak", "a") for _ in range(90)]
    comparisons += [Comparison("strong", "weak", "b") for _ in range(10)]

    scores = {s.name: s for s in fit(comparisons)}
    assert scores["easy"].win_rate > scores["hard"].win_rate
    assert scores["hard"].score > scores["easy"].score


def test_bootstrap_interval_shrinks_with_evidence():
    """An interval that ignores sample size would let us publish noise."""
    few = fit(synthetic({"a": 3.0, "b": 1.0}, per_pair=30, seed=1), bootstrap=200)
    many = fit(synthetic({"a": 3.0, "b": 1.0}, per_pair=3000, seed=1), bootstrap=200)
    assert few[0].ci95 is not None and many[0].ci95 is not None
    assert many[0].ci95 < few[0].ci95


def test_bootstrap_is_deterministic_for_a_seed():
    kwargs = dict(bootstrap=100, seed=7)
    data = synthetic({"a": 2.0, "b": 1.0}, per_pair=100)
    assert [s.ci95 for s in fit(data, **kwargs)] == [s.ci95 for s in fit(data, **kwargs)]


def test_a_system_that_never_wins_is_ranked_last_and_not_a_nan():
    """The degenerate case the MM update divides by zero on.

    Bradley-Terry has no finite MLE for a system with zero wins -- its strength
    runs to zero and its score to negative infinity. The solver floors it, and
    what matters is that the output is still a usable number: a NaN here would
    propagate into the report silently.
    """
    comparisons = [Comparison("winner", "loser", "a") for _ in range(50)]
    comparisons += [Comparison("winner", "other", "a") for _ in range(20)]
    comparisons += [Comparison("other", "loser", "a") for _ in range(20)]
    scores = fit(comparisons)
    assert [s.name for s in scores] == ["winner", "other", "loser"]
    assert all(math.isfinite(s.score) for s in scores)
    assert scores[-1].win_rate == 0.0


def test_empty_input_is_not_an_exception():
    assert fit([]) == []
    assert "no comparisons" in format_table([])


def test_comparison_counts_are_reported():
    scores = {s.name: s for s in fit(synthetic({"a": 2.0, "b": 1.0}, per_pair=25))}
    assert scores["a"].comparisons == 25
    assert scores["b"].comparisons == 25
