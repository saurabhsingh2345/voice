"""Bradley-Terry ranking from pairwise preferences.

This is the instrument AI4Bharat used to rank seven TTS systems from 120K
pairwise comparisons in SpeechArenaBench (arXiv 2604.21481). It lives here so
this project can fit the *same* model on the Hindi slice of their released
preference data, which the paper never published as numbers: their Table 4
breaks scores down by input type across all ten languages pooled, and their
Figure 1 shows per-language rankings as a picture with no table behind it.
Hindi-only, code-mixed-only Bradley-Terry scores therefore do not exist in the
literature and have to be fitted.

Why fit it ourselves rather than quote the paper: quoting gives a ranking of
*their* seven systems, which is a fact about them. Fitting gives a ranking we
can add an eighth system to, and it is the only way this project's engine ever
appears on the same axis as 1,900 native raters.

    from voiceagent.eval.arena_bt import Comparison, fit

    scores = fit([Comparison("gemini", "indicf5", "a"), ...])

The scale is the Chatbot-Arena convention the paper follows: log-strengths
mapped through 400/ln(10) and centred so the mean system sits at 1000. That
convention is why their seven scores average 1000.6 rather than anything else,
and matching it is what makes our numbers comparable to their Table 4.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np

#: How a rater resolved one pair. `a`/`b` are wins; the two tie kinds are kept
#: distinct because they are *not* interchangeable evidence even though this
#: model scores them identically -- "Both Bad" on a pair of cloud systems means
#: something very different from "Both Good", and a caller filtering to
#: confident judgements only needs to be able to tell them apart.
Outcome = Literal["a", "b", "both_good", "both_bad"]

#: Elo-like scale constants. 400/ln(10) is the Elo convention: 400 points is a
#: 10:1 odds ratio.
SCALE = 400.0 / math.log(10.0)
CENTRE = 1000.0


@dataclass(frozen=True)
class Comparison:
    """One rater's verdict on one pair."""

    model_a: str
    model_b: str
    outcome: Outcome


@dataclass(frozen=True)
class SystemScore:
    name: str
    score: float
    #: Half-width of the 95% bootstrap interval, or None if not bootstrapped.
    ci95: float | None
    #: Comparisons this system took part in. Reported because a score fitted on
    #: 40 comparisons and one fitted on 40,000 print identically otherwise.
    comparisons: int
    #: Share of its comparisons the system won, ties counted as half. This is
    #: the descriptive number; `score` is the modelled one. They disagree when a
    #: system's opponents were unusually strong or weak, which is the entire
    #: reason to fit a model instead of counting wins.
    win_rate: float


def _tally(
    comparisons: Sequence[Comparison], names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (wins, pair_counts) with ties credited half to each side."""
    index = {name: i for i, name in enumerate(names)}
    n = len(names)
    wins = np.zeros(n, dtype=float)
    pairs = np.zeros((n, n), dtype=float)

    for c in comparisons:
        i, j = index[c.model_a], index[c.model_b]
        pairs[i, j] += 1.0
        pairs[j, i] += 1.0
        if c.outcome == "a":
            wins[i] += 1.0
        elif c.outcome == "b":
            wins[j] += 1.0
        else:
            # Both ties are half-credit. This is the Chatbot Arena convention
            # and it is a real modelling choice, not a neutral one: it treats
            # "Both Bad" as evidence the two systems are equal, which for the
            # bottom of a TTS ranking is arguably generous to the worse system.
            # Kept because deviating from it would make our numbers
            # incomparable with the paper's, which is the whole point.
            wins[i] += 0.5
            wins[j] += 0.5

    return wins, pairs


def _mm_fit(wins: np.ndarray, pairs: np.ndarray, iterations: int, tol: float) -> np.ndarray:
    """Hunter's MM algorithm for the Bradley-Terry MLE. Returns log-strengths.

    Chosen over gradient descent on a logistic regression because it needs no
    step size, no optimizer dependency, and is monotone -- every iteration
    increases the likelihood, so it cannot diverge on a sparse pair graph. That
    matters here: the arena's pairs are randomly sampled, so some system pairs
    meet far less often than others.
    """
    n = len(wins)
    strength = np.ones(n, dtype=float)

    for _ in range(iterations):
        previous = strength.copy()
        # denom[i] = sum_j n_ij / (p_i + p_j), the MM surrogate's denominator.
        totals = strength[:, None] + strength[None, :]
        np.fill_diagonal(totals, 1.0)  # unused; avoids a divide-by-zero warning
        denom = (pairs / totals).sum(axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            updated = np.where(denom > 0, wins / denom, strength)
        # A system that never won anything has strength 0, which is -inf on the
        # log scale and would poison the centring. Floor it at a value far below
        # the pack instead, and let the caller see its win rate.
        updated = np.where(updated > 0, updated, 1e-12)
        # Normalise to geometric mean 1 so the iteration cannot drift.
        strength = updated / np.exp(np.log(updated).mean())

        if np.max(np.abs(strength - previous)) < tol:
            break

    return np.log(strength)


def fit(
    comparisons: Iterable[Comparison],
    *,
    bootstrap: int = 0,
    iterations: int = 1000,
    tol: float = 1e-9,
    seed: int = 0,
) -> list[SystemScore]:
    """Fit Bradley-Terry scores, highest first.

    `bootstrap` resamples the comparisons with replacement that many times to
    get a 95% interval; the paper used 500. Zero skips it, which is what the
    unit tests want and what a quick look wants.

    The interval is the honest part of the output. With a few hundred
    comparisons the intervals overlap so heavily that the ranking is not a
    ranking, and a caller that prints scores without them will read noise as a
    result.
    """
    comparisons = list(comparisons)
    if not comparisons:
        return []

    names = sorted({c.model_a for c in comparisons} | {c.model_b for c in comparisons})
    wins, pairs = _tally(comparisons, names)
    log_strength = _mm_fit(wins, pairs, iterations, tol)
    scores = CENTRE + SCALE * (log_strength - log_strength.mean())

    spread: dict[str, float] | None = None
    if bootstrap > 0:
        rng = np.random.default_rng(seed)
        samples = np.empty((bootstrap, len(names)), dtype=float)
        indices = np.arange(len(comparisons))
        for b in range(bootstrap):
            picked = rng.choice(indices, size=len(comparisons), replace=True)
            resampled = [comparisons[i] for i in picked]
            w, p = _tally(resampled, names)
            ls = _mm_fit(w, p, iterations, tol)
            samples[b] = CENTRE + SCALE * (ls - ls.mean())
        lo = np.percentile(samples, 2.5, axis=0)
        hi = np.percentile(samples, 97.5, axis=0)
        spread = {name: (hi[i] - lo[i]) / 2.0 for i, name in enumerate(names)}

    appearances = pairs.sum(axis=1)
    out = [
        SystemScore(
            name=name,
            score=float(scores[i]),
            ci95=None if spread is None else float(spread[name]),
            comparisons=int(appearances[i]),
            win_rate=float(wins[i] / appearances[i]) if appearances[i] else 0.0,
        )
        for i, name in enumerate(names)
    ]
    out.sort(key=lambda s: s.score, reverse=True)
    return out


def format_table(scores: Sequence[SystemScore]) -> str:
    """Plain-text ranking, for the CLI and for pasting into a findings file."""
    if not scores:
        return "(no comparisons)"
    width = max(len(s.name) for s in scores)
    lines = [f"{'system'.ljust(width)}  {'BT':>9}  {'95% CI':>8}  {'win':>6}  {'n':>7}"]
    for s in scores:
        ci = f"±{s.ci95:.0f}" if s.ci95 is not None else "--"
        lines.append(
            f"{s.name.ljust(width)}  {s.score:9.2f}  {ci:>8}  {s.win_rate:6.1%}  {s.comparisons:7d}"
        )
    return "\n".join(lines)
