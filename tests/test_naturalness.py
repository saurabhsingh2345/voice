"""Tests for the naturalness calibration.

The arithmetic here decides whether a metric gets trusted, and this project has
already shipped one metric it could not rank with. So the cases are the ones that
would silently flatter a predictor: ties, a single-class split, and the sign of a
rank correlation.
"""

from __future__ import annotations

import pytest

from voiceagent.eval.naturalness import auc, spearman


def test_a_perfect_separator_scores_one():
    assert auc([0.1, 0.2, 0.9, 1.0], [False, False, True, True]) == pytest.approx(1.0)


def test_a_perfectly_inverted_separator_scores_zero():
    """Below 0.5 is not "no signal", it is *backwards* signal, and the
    distinction matters: every system-level correlation in this study came out
    negative, which is a finding rather than a null."""
    assert auc([0.9, 1.0, 0.1, 0.2], [False, False, True, True]) == pytest.approx(0.0)


def test_all_ties_score_exactly_chance():
    """STOI saturates near 1.0 and produces many exact ties. Counting a tie as a
    win would flatter it toward 1.0; the rank correction makes ties score 0.5."""
    assert auc([0.99] * 6, [True, False, True, False, True, False]) == pytest.approx(0.5)


def test_partial_ties_are_split_not_awarded():
    scored = auc([1.0, 1.0, 0.0], [True, False, False])
    assert scored == pytest.approx(0.75)


def test_one_class_is_undefined_rather_than_chance():
    """Returning 0.5 here would read as "no signal" when the truth is "no test",
    and a table full of quiet 0.5s is how an axis with no negative examples gets
    mistaken for a metric that failed."""
    assert auc([0.1, 0.5, 0.9], [True, True, True]) is None
    assert auc([0.1, 0.5, 0.9], [False, False, False]) is None


def test_spearman_recovers_a_known_ordering():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_refuses_too_few_points():
    """Six or seven systems is already thin. Two is not a correlation."""
    assert spearman([1, 2], [2, 1]) is None


def test_spearman_handles_ties_without_dividing_by_zero():
    assert spearman([1, 1, 1], [1, 2, 3]) is None


def test_the_broken_system_is_held_out_of_the_working_band():
    """The whole discipline of this module in one assertion: the system native
    raters rejected 87% of the time must not be what makes a metric look good."""
    from voiceagent.eval.naturalness import BROKEN_SYSTEM, BT_RANKING

    assert BROKEN_SYSTEM in BT_RANKING
    assert BT_RANKING[BROKEN_SYSTEM] == min(BT_RANKING.values())
    others = [v for k, v in BT_RANKING.items() if k != BROKEN_SYSTEM]
    # It is not merely last, it is off the bottom of the pack by ~300 points,
    # which is why including it makes almost any metric look correlated.
    assert min(others) - BT_RANKING[BROKEN_SYSTEM] > 250
