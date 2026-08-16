"""H0.1 — does any automatic predictor know what a Hindi listener thinks?

`plan-hindi.md` opens on the fact that naturalness has never been measured here.
Round trip cannot do it: it asks whether Whisper recovers the words, so a flat
monotone with every phoneme correct scores at ceiling, and Phase 2 measured it at
**AUC 0.625** against native judgement once the one broken system is removed.
This module looks for something better, and holds it to the same test.

WHY THIS CAN BE ANSWERED TODAY, WITHOUT RECRUITING ANYONE

SpeechArenaBench ships 654 Hindi clips *with the native rater's judgement
attached*, MIT-licensed and already on disk from Phase 2. Each clip carries six
axes, and three of them are the ones round trip is structurally blind to:

    expressiveness    voice_quality    liveliness

So a candidate metric can be scored against real human verdicts on real Hindi,
on identical audio, before it is trusted with a single decision. Published MOS
predictors are trained overwhelmingly on English, and the literature reports that
zero-shot MOS prediction *degrades on Indian languages* — which is exactly the
reason to measure rather than to adopt.

THE TEST THAT MATTERS IS THE SECOND ONE

Two numbers are reported for every predictor and every axis:

  * **all systems** — includes Indic F5, which native raters approved 13% of the
    time. Any metric that notices broken speech scores well here, and it means
    almost nothing.
  * **working band** — the six systems that work, Indic F5 removed. This is
    where round trip collapsed from 0.671 to 0.625, and where its ordering
    actually *inverted* against the raters.

A predictor is only useful to this project if it ranks inside the working band,
because that is the band we are in and the band we are trying to move within.
Judge on the second column. The first is a formality.

WHAT IS BEING SCORED, AND WHY THESE

`torchaudio` ships Meta's SQUIM, BSD-licensed and **already installed**, so this
costs no new dependency and does not touch `voice-doctor`. Two heads:

  * **SQUIM subjective** — a MOS estimate, the thing we actually want. It is
    non-intrusive but needs a *non-matching reference*: a clean utterance from a
    different speaker saying something different, used to anchor the scale.
  * **SQUIM objective** — STOI, PESQ and SI-SDR, designed for transmission
    quality rather than synthesis. Included as a control: if a codec metric
    predicts a Hindi listener's verdict about *expressiveness*, that says the
    labels are tracking recording conditions rather than performance.

`overlap` — the existing round-trip score, already computed per clip in
`scores.json` — is carried through as the incumbent baseline. A new metric that
cannot beat it is not worth adopting.

HONEST LIMITS OF WHAT THIS CAN CONCLUDE

The six axes are **binary** in this data: only 1 and 5 occur, which Phase 2 found
the hard way after the data card documented 1–5 scales. So AUC is the right
statistic and a correlation coefficient is not.

Each clip carries *one* rater's opinion, not a panel mean. Single-rater labels
are noisy, and that noise puts a ceiling on the AUC any predictor can reach —
a perfect metric would not score 1.0 here. Read the *ordering* of predictors and
the gap to 0.500, not the absolute value.

And a predictor that passes this test has been validated for **judging existing
systems**, not for being optimised against. Chasing a proxy is how you get a
model that scores well and sounds worse.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "eval_out" / "arena"
SCORES = CACHE / "scores.json"
OUT = ROOT / "eval_out" / "naturalness"

#: The system native raters rejected 87% of the time. Held out of the working
#: band, because a metric that only separates it from everything else has learned
#: nothing we can use.
BROKEN_SYSTEM = "Indic F5"

#: The three axes round trip is blind to by construction, and the reason this
#: module exists. `intelligibility` is scored too, as the control: it is the axis
#: round trip *is* meant to track, so a predictor beating it there is a bonus and
#: a predictor losing there is a sanity check on the pipeline.
NATURALNESS_AXES = ("expressiveness", "voice_quality", "liveliness")
CONTROL_AXES = ("intelligibility",)

#: A rating counts as approval at or above this. The axes are binary (1 or 5),
#: so any threshold in between gives the same split; 4 matches Phase 2.
GOOD_RATING = 4

#: The Hindi code-mixed Bradley-Terry ranking fitted in `eval/arena_bt.py` from
#: 10,268 native votes (`eval_out/arena/FINDINGS.md` §1). Used for the
#: system-level test: does a predictor's mean score order the systems the way
#: 1,900 raters did?
BT_RANKING = {
    "Gemini 2.5 Pro TTS": 1131.3,
    "Bulbul V3 Beta": 1056.4,
    "Sonic 3": 1034.8,
    "Eleven Labs v3": 1031.0,
    "GPT 4o Mini TTS": 1017.3,
    "Speech 2.8 HD": 1014.6,
    "Indic F5": 714.7,
}


@dataclass
class Scored:
    clip_id: str
    system: str
    ratings: dict[str, int]
    metrics: dict[str, float]


def auc(scores: list[float], labels: list[bool]) -> float | None:
    """Probability a randomly chosen approved clip outscores a rejected one.

    Rank-based with tie correction, which matters here: STOI saturates near 1.0
    and produces many exact ties, and counting a tie as a win would flatter it.
    Returns None when one class is absent, because an AUC over a single class is
    undefined rather than 0.5 — and silently returning 0.5 would read as "no
    signal" when the truth is "no test".
    """
    pairs = sorted(zip(scores, labels), key=lambda p: p[0])
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None

    # Average ranks within tied score groups.
    ranks: list[float] = [0.0] * len(pairs)
    index = 0
    while index < len(pairs):
        end = index
        while end + 1 < len(pairs) and pairs[end + 1][0] == pairs[index][0]:
            end += 1
        average = (index + end) / 2 + 1
        for position in range(index, end + 1):
            ranks[position] = average
        index = end + 1

    positive_rank_sum = sum(r for r, (_, label) in zip(ranks, pairs) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def spearman(a: list[float], b: list[float]) -> float | None:
    """Rank correlation. Used only at system level, where n is 6 or 7."""
    if len(a) < 3:
        return None

    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = average
            i = j + 1
        return out

    ra, rb = rank(a), rank(b)
    n = len(a)
    mean_a, mean_b = sum(ra) / n, sum(rb) / n
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    den = (sum((x - mean_a) ** 2 for x in ra) * sum((y - mean_b) ** 2 for y in rb)) ** 0.5
    return num / den if den else None


def _load_audio(path: str, target: int = 16_000):
    import soundfile as sf
    import torch
    import torchaudio

    audio, rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    wave = torch.from_numpy(audio).unsqueeze(0)
    if rate != target:
        wave = torchaudio.functional.resample(wave, rate, target)
    return wave


def score_clips(limit: int | None = None) -> list[Scored]:
    """Run every predictor over the rated clips."""
    import torch
    from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE

    clips = json.loads(SCORES.read_text())
    if limit:
        clips = clips[:limit]

    subjective = SQUIM_SUBJECTIVE.get_model()
    objective = SQUIM_OBJECTIVE.get_model()

    # SQUIM subjective is anchored on a non-matching reference: clean speech from
    # a different speaker saying something different. A clip from the top-rated
    # system is used, held constant so every score is on the same scale --- a
    # varying reference would add noise that looks like signal.
    reference_clip = next(
        (c for c in clips if c["system"] == "Gemini 2.5 Pro TTS"), clips[0]
    )
    reference = _load_audio(reference_clip["wav"])

    out: list[Scored] = []
    for index, clip in enumerate(clips, 1):
        path = ROOT / clip["wav"] if not Path(clip["wav"]).is_absolute() else Path(clip["wav"])
        if not path.exists():
            continue
        wave = _load_audio(str(path))
        metrics: dict[str, float] = {"overlap": float(clip.get("overlap") or 0.0)}
        try:
            with torch.no_grad():
                metrics["squim_mos"] = float(subjective(wave, reference).item())
                stoi, pesq, sisdr = (float(x.item()) for x in objective(wave))
            metrics.update(squim_stoi=stoi, squim_pesq=pesq, squim_sisdr=sisdr)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]skipped {clip['clip_id']}: {type(exc).__name__} {exc}[/]")
            continue

        out.append(Scored(clip["clip_id"], clip["system"], clip["ratings"], metrics))
        if index % 25 == 0:
            # Written to stderr and flushed, not through `rich`. A progress line
            # that buffers is worse than none: the first run of this looked
            # stalled at 350/654 for half an hour while it was working fine, and
            # the only way to tell was to check the process's CPU time.
            print(f"  scored {index}/{len(clips)}", file=sys.stderr, flush=True)
    return out


def report(scored: list[Scored]) -> list[str]:
    metric_names = ["overlap", "squim_mos", "squim_pesq", "squim_stoi", "squim_sisdr"]
    axes = NATURALNESS_AXES + CONTROL_AXES

    working = [s for s in scored if s.system != BROKEN_SYSTEM]

    table = Table(
        title="AUC against native judgement — Hindi, 654 arena clips",
        title_justify="left", header_style="bold",
    )
    table.add_column("Metric")
    table.add_column("Axis")
    table.add_column("All systems", justify="right")
    table.add_column("Working band", justify="right")

    lines = [
        "# H0.1 — can any automatic predictor judge Hindi naturalness?",
        "",
        "Generated by `uv run python -m voiceagent.eval.naturalness`.",
        "",
        f"{len(scored)} rated clips, {len(working)} once `{BROKEN_SYSTEM}` is removed.",
        "Ratings are binary (1 or 5), so AUC is the statistic and correlation is not.",
        "",
        "**Read the working-band column.** Every metric separates broken speech from",
        "working speech; the question is whether it can rank inside the band we are in.",
        "Round trip scored 0.671 all-systems and **0.625** in the band, and its ordering",
        "inverted against the raters there.",
        "",
        "| Metric | Axis | All systems | Working band |",
        "| --- | --- | --- | --- |",
    ]

    best: tuple[float, str, str] | None = None
    for metric in metric_names:
        for axis in axes:
            def split(rows: list[Scored]) -> tuple[list[float], list[bool]]:
                pairs = [
                    (r.metrics[metric], r.ratings[axis] >= GOOD_RATING)
                    for r in rows
                    if metric in r.metrics and axis in r.ratings
                ]
                return [p[0] for p in pairs], [p[1] for p in pairs]

            all_auc = auc(*split(scored))
            band_auc = auc(*split(working))
            fmt = lambda v: "-" if v is None else f"{v:.3f}"

            marker = ""
            if band_auc is not None and axis in NATURALNESS_AXES:
                if best is None or band_auc > best[0]:
                    best = (band_auc, metric, axis)
                if band_auc >= 0.70:
                    marker = " ✓"

            table.add_row(metric, axis, fmt(all_auc), f"{fmt(band_auc)}{marker}")
            lines.append(f"| `{metric}` | {axis} | {fmt(all_auc)} | **{fmt(band_auc)}**{marker} |")

    console.print(table)

    # System level: does the metric's mean order the systems the way the raters
    # did? This is the test round trip failed most visibly --- r=0.948 across all
    # seven looked excellent and was one broken outlier dragging a line through a
    # cluster.
    lines += ["", "## System-level ranking", "",
              "Spearman between each metric's mean and the Bradley-Terry ranking fitted",
              "from 10,268 native votes. The all-systems column is the trap: a single",
              "broken outlier makes almost anything look correlated.",
              "",
              "| Metric | All 7 | Working 6 |", "| --- | --- | --- |"]

    console.print()
    console.print("[bold]System-level Spearman against the Bradley-Terry ranking[/]")
    for metric in metric_names:
        def means(rows: list[Scored]) -> tuple[list[float], list[float]]:
            by_system: dict[str, list[float]] = {}
            for row in rows:
                if metric in row.metrics:
                    by_system.setdefault(row.system, []).append(row.metrics[metric])
            systems = [s for s in by_system if s in BT_RANKING]
            return (
                [statistics.mean(by_system[s]) for s in systems],
                [BT_RANKING[s] for s in systems],
            )

        rho_all = spearman(*means(scored))
        rho_band = spearman(*means(working))
        fmt = lambda v: "-" if v is None else f"{v:+.3f}"
        console.print(f"  {metric:14} all {fmt(rho_all):>7}   working band {fmt(rho_band):>7}")
        lines.append(f"| `{metric}` | {fmt(rho_all)} | **{fmt(rho_band)}** |")

    lines += _axis_entanglement(scored)

    verdict = _verdict(best)
    lines += ["", "## Verdict", "", verdict]
    console.print()
    console.print(f"[bold]{verdict}[/]")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "FINDINGS.md").write_text("\n".join(lines) + "\n")
    console.print(f"\nwritten: {OUT / 'FINDINGS.md'}")
    return lines


def _axis_entanglement(scored: list[Scored]) -> list[str]:
    """How independent the six axes actually are.

    Checked because the smoke run returned *identical* AUCs for expressiveness,
    voice_quality and liveliness, which would mean they are one axis wearing
    three labels. On the full set they are not identical — but they are close
    enough that no result here can cleanly separate "sounds natural" from
    "is intelligible", and a reader deserves to know that before believing a
    naturalness number derived from these labels.
    """
    import itertools

    axes = list(NATURALNESS_AXES) + list(CONTROL_AXES) + ["hallucinations", "noise"]
    present = [a for a in axes if scored and a in scored[0].ratings]

    lines = [
        "", "## How independent are the axes?", "",
        "Pairwise agreement between the raters' six binary axes. This is a property of",
        "the *labels*, not of any predictor.",
        "",
        "| Axis A | Axis B | Agree |", "| --- | --- | --- |",
    ]
    worst = 1.0
    for a, b in itertools.combinations(present, 2):
        agree = sum(1 for s in scored if s.ratings[a] == s.ratings[b]) / len(scored)
        worst = min(worst, agree)
        lines.append(f"| {a} | {b} | {agree * 100:.1f}% |")

    lines += [
        "",
        f"Nothing disagrees more than {(1 - worst) * 100:.0f}% of the time. Raters were",
        "largely giving one global verdict and distributing it across six boxes —",
        "`intelligibility` and `hallucinations` agree 93%.",
        "",
        "**So this dataset cannot cleanly isolate naturalness from intelligibility.**",
        "A predictor scoring well on `expressiveness` here may simply be tracking",
        "whether the clip is broken. That is a limit of the labels and no amount of",
        "modelling fixes it — it is an argument for the listening panel, where the",
        "question can be asked directly and one axis at a time.",
    ]
    return lines


def _verdict(best: tuple[float, str, str] | None) -> str:
    if best is None:
        return "No predictor could be scored. Nothing to conclude."
    score, metric, axis = best
    if score >= 0.70:
        return (
            f"`{metric}` reaches AUC {score:.3f} on **{axis}** inside the working band, "
            f"against round trip's 0.625 on intelligibility. That is a usable "
            f"instrument for judging systems — not a target to optimise against."
        )
    if score >= 0.60:
        return (
            f"Best is `{metric}` at AUC {score:.3f} on **{axis}** inside the working "
            f"band. Better than a coin flip and not by enough to trust a decision to. "
            f"Treat as weak evidence; the listening panel is still the ground truth."
        )
    return (
        f"**Nothing here works.** Best is `{metric}` at AUC {score:.3f} on {axis} "
        f"inside the working band, against 0.500 for chance. No off-the-shelf "
        f"predictor in this set can judge Hindi naturalness, which makes the human "
        f"panel in `eval/abtest.py` the only instrument available — and it is the "
        f"one nobody has run. That is a finding, and it is the answer to 'why not "
        f"just use a metric'."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="first N clips (smoke test)")
    args = parser.parse_args()

    if not SCORES.exists():
        console.print(f"[red]missing {SCORES}[/] — run `voiceagent.eval.arena score` first.")
        return 2

    scored = score_clips(args.limit)
    if not scored:
        console.print("[red]nothing scored[/]")
        return 2
    report(scored)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scores.json").write_text(json.dumps(
        [{"clip_id": s.clip_id, "system": s.system, "ratings": s.ratings,
          "metrics": s.metrics} for s in scored], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
