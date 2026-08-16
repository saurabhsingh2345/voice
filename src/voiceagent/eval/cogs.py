"""What a generation actually costs, on the machine we actually own.

Plan §12 item 4: know the cost per generation before pricing. Sarvam publishes
₹15 per 10,000 characters (Bulbul v2) and ₹30 (v3), and the margin has to live
under that. This module answers it from measurement rather than arithmetic.

THE ANSWER IS NOT A RUPEE FIGURE, AND THAT IS THE POINT

On owned hardware the marginal cost of a character is electricity, and
electricity is negligible — a fraction of a paisa per 10,000 characters against a
market price of ₹15-30. Anyone can do that division and conclude the margin is
99.8%, and they would be answering the wrong question.

The binding constraint is **capacity**. Synthesis holds a lock: shared mutable
engines and `set_reference` on a shared instance mean concurrency is 1 by design,
not by tuning (plan §7 Phase C). So the machine does not have a cost ceiling
denominated in rupees, it has one denominated in *seconds*, and every request
spends from the same 86,400 a day. What this module measures is how those
seconds are spent.

THE FINDING, WHICH IS THE OPPOSITE OF WHAT THIS MODULE WAS BUILT EXPECTING

The hypothesis was that cost is per **request** while billing is per
**character**, so a conversational customer sending short turns would cost far
more than a narration customer for the same invoice. The first fit, against
production metering, appeared to confirm it: 3.12 s of fixed overhead per call,
implying short requests cost ~5x per character.

**Measurement did not confirm it, and then refused to settle it.** Swept across
56-2036 characters, the marginal rate is solid at roughly **40 s per 1000
characters** — four independent sweeps landed within 37.7-41.9. The intercept did
not converge at all: the same four sweeps put it at -0.03, 0.20, 0.69 and 3.27
seconds, and the 3.27 traced to a single bucket that caught a load spike and ran
at RTF 0.97 instead of 0.59.

So the honest answer has two halves:

  * **The per-character rate is trustworthy.** Price on it. Capacity follows from
    it, and so does every figure in this module worth acting on.
  * **The per-request fixed cost is unresolved**, somewhere between zero and a
    few seconds, and no batching verdict should be built on it in either
    direction. The metering fit's confident 3.12 s was an artefact of a 5x span;
    the sweep's near-zero readings are an artefact of quiet buckets. Repeats are
    now collapsed to per-bucket medians to damp the spikes, which narrows the
    swing without earning the right to a claim.

That is a weaker conclusion than either of the two this module reached on its way
here, and it is the one the data supports. The first said short requests cost 5x;
the second said batching is irrelevant. Both were read off an intercept that
moves when the machine gets busy.

WHAT THE CONSTRAINT ACTUALLY IS

Capacity, and it is not close. Electricity is a fraction of a percent of what
Sarvam charges, so margin per character is not a question worth asking. The
machine serves one generation at a time and produces roughly 2.1M characters —
about 40 hours of audio — in a day of 100% utilisation it will never get, being
also a workstation. That is the number a free tier spends from.

WHY BOTH A FIT AND A SWEEP

`--from-metering` fits the model against real production rows in `usage.db` —
genuine traffic, no synthetic assumptions. Its weakness is range: real requests
so far are 29-142 characters, and extrapolating a fixed-plus-marginal model 35×
beyond its data is how you get a confident wrong number.

`--sweep` generates timed samples across length buckets up to a few thousand
characters, so the fit is interpolation rather than extrapolation. It writes
nothing to `usage.db`: that table is production accounting and synthetic load
does not belong in it.

Run both. If they disagree, the metering fit is the one describing reality and
the sweep is the one describing the machine.

CAUTIONS

Every number here is load-sensitive and the machine is often loaded. Run
`voiceagent.eval.specsheet --check` first; it exits non-zero and says why. A
benchmark taken at load average 537 once made a figure wrong by 10×.
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
import statistics as stats
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from rich.console import Console
from rich.table import Table

console = Console()

ROOT = Path(__file__).resolve().parents[3]
USAGE_DB = ROOT / "data" / "usage.db"
OUT_DIR = ROOT / "eval_out" / "cogs"
REF_WAV = ROOT / "fixtures" / "hi" / "reference_lekha.wav"

#: Sustained package power for an M3 Pro under a single-threaded Metal load, in
#: watts. NOT measured — this is a published envelope, and it is the weakest
#: number in this module. It is isolated here so the sensitivity is obvious:
#: every rupee figure below scales linearly with it, and none of the *capacity*
#: figures depend on it at all. That asymmetry is the argument for pricing on
#: capacity rather than on power.
ASSUMED_WATTS = 40.0

#: ₹ per kWh, Indian commercial tariff, order-of-magnitude. Same caveat.
ASSUMED_TARIFF_INR_PER_KWH = 9.0

#: What the market charges, for the comparison that matters. Checked 2026-08-14;
#: `outreach/CLAIMS.md` says to re-check before quoting these to anyone.
SARVAM_INR_PER_10K = {"Bulbul v2": 15.0, "Bulbul v3": 30.0}

#: Fitted intercepts observed across four independent sweeps of the same buckets
#: (seconds). The slope over the same four runs stayed within 37.7-41.9 s per
#: 1000 characters; this is what refused to converge, and the 3.27 traced to one
#: bucket that caught a load spike and ran at RTF 0.97 instead of 0.59.
#:
#: Recorded rather than averaged, because the honest statement about a
#: per-request fixed cost is a range, and a range is what the batching question
#: has to be answered with until a quieter machine narrows it.
OBSERVED_INTERCEPT_SECONDS = (-0.03, 3.27)

#: Length buckets for the sweep, in characters. The top of the range is set by
#: MAX_SPEAK_CHARS in the server rather than by what is interesting: pricing a
#: request size the API refuses would be theatre.
SWEEP_BUCKETS = (50, 150, 400, 1000, 2000)

#: Hindi source text for the sweep. Real sentences rather than a repeated string,
#: because a model given the same clause forty times is a different workload from
#: one given prose --- span grouping, the repetition guard and the KV cache all
#: behave differently, and the cheap version of this test would flatter itself.
_SWEEP_SOURCE = (
    "आज सुबह मैंने खिड़की खोली तो बाहर हल्की धूप फैली हुई थी। ",
    "पिछले हफ्ते हुई बारिश के बाद हवा में अब भी नमी बनी हुई है। ",
    "उसने चाय का प्याला मेज़ पर रखा और अख़बार उठा लिया। ",
    "इस साल फ़सल अच्छी रही, इसलिए गाँव में सब लोग संतुष्ट दिखाई दे रहे हैं। ",
    "रेलगाड़ी थोड़ी देर से चल रही थी, फिर भी वह समय पर पहुँच गया। ",
    "बच्चे मैदान में खेल रहे थे और उनकी आवाज़ें दूर तक सुनाई दे रही थीं। ",
    "उन्होंने कहा कि यह काम कठिन ज़रूर है, लेकिन असंभव बिल्कुल नहीं। ",
    "शाम होते ही बाज़ार में रौनक बढ़ जाती है और दुकानें भर जाती हैं। ",
)


def sweep_text(target_chars: int) -> str:
    """Prose of approximately `target_chars`, cut at a sentence boundary."""
    out: list[str] = []
    total = 0
    index = 0
    while total < target_chars:
        piece = _SWEEP_SOURCE[index % len(_SWEEP_SOURCE)]
        out.append(piece)
        total += len(piece)
        index += 1
    return "".join(out).strip()


@dataclass(frozen=True)
class Sample:
    characters: int
    audio_seconds: float
    synthesis_seconds: float


@dataclass(frozen=True)
class CostModel:
    """`synthesis_seconds = fixed + marginal_per_char * characters`."""

    fixed_seconds: float
    marginal_seconds_per_char: float
    samples: int
    min_chars: int
    max_chars: int
    chars_per_audio_second: float

    def seconds_for(self, characters: int) -> float:
        return self.fixed_seconds + self.marginal_seconds_per_char * characters

    @property
    def range_ratio(self) -> float:
        """Ratio of longest to shortest sample. How identifiable the slope is."""
        return self.max_chars / self.min_chars if self.min_chars else float("inf")

    @property
    def separates_fixed_from_marginal(self) -> bool:
        """Whether this fit can tell a per-call cost from a per-character one.

        It cannot, over a narrow span. The production-metering fit spans 5x and
        confidently reported 3.12 s of fixed overhead; a sweep over 36x put the
        same term at 0.20 s. The intercept of a short-range regression absorbs
        load and noise and then presents them as structure.

        10x is a judgement, not a theorem: it is comfortably above the span that
        misled this module and comfortably below the span that corrected it.
        """
        return self.range_ratio >= 10.0

    def in_range(self, characters: int) -> bool:
        """Whether `characters` is interpolation rather than extrapolation.

        Kept explicit because the whole failure mode this module guards against
        is quoting a confident cost for a request size never measured.
        """
        return self.min_chars <= characters <= self.max_chars

    def machine_seconds_per_10k(self, request_chars: int) -> float:
        """Machine time to bill 10,000 characters in requests of this size."""
        requests = 10_000 / request_chars
        return requests * self.seconds_for(request_chars)

    def inr_per_10k(self, request_chars: int) -> float:
        seconds = self.machine_seconds_per_10k(request_chars)
        kwh = (ASSUMED_WATTS / 1000.0) * (seconds / 3600.0)
        return kwh * ASSUMED_TARIFF_INR_PER_KWH

    def chars_per_hour(self, request_chars: int) -> float:
        return 3600.0 / self.seconds_for(request_chars) * request_chars


def by_bucket_median(samples: list[Sample]) -> list[Sample]:
    """Collapse repeats of the same length to their median.

    Load spikes are the dominant noise source on this machine and they hit one
    bucket at a time. Measured: a sweep where the 430-character bucket happened
    to run at RTF 0.97 instead of 0.59 moved the fitted intercept from 0.20 s to
    3.27 s on its own, while the slope barely moved. A median over repeats is
    what stops one perturbed bucket rewriting the conclusion, and it is what the
    Hindi RTF table in the README already does for the same reason.
    """
    grouped: dict[int, list[Sample]] = {}
    for sample in samples:
        grouped.setdefault(sample.characters, []).append(sample)
    return [
        Sample(
            characters=chars,
            audio_seconds=stats.median(s.audio_seconds for s in group),
            synthesis_seconds=stats.median(s.synthesis_seconds for s in group),
        )
        for chars, group in sorted(grouped.items())
    ]


def fit(samples: list[Sample]) -> CostModel:
    """Ordinary least squares of synthesis time against character count.

    A straight line because the mechanism is a straight line: a fixed
    per-call cost (model conditioning, tokenizer, encode, lock acquisition) plus
    a per-token generation cost. It is not chosen for convenience, and if the
    residuals ever stop looking flat that is a finding, not a fitting problem.
    """
    if len(samples) < 2:
        raise ValueError("need at least two samples to fit")

    xs = [float(s.characters) for s in samples]
    ys = [s.synthesis_seconds for s in samples]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    denominator = sum((x - mx) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError("every sample is the same length; cannot separate fixed from marginal")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denominator

    return CostModel(
        fixed_seconds=my - slope * mx,
        marginal_seconds_per_char=slope,
        samples=len(samples),
        min_chars=int(min(xs)),
        max_chars=int(max(xs)),
        chars_per_audio_second=stats.median(
            s.characters / s.audio_seconds for s in samples if s.audio_seconds > 0
        ),
    )


def from_metering(db: Path = USAGE_DB) -> list[Sample]:
    """Successful, timed generations from production accounting."""
    if not db.exists():
        return []
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "select characters, audio_seconds, synthesis_seconds from usage "
            "where status='ok' and audio_seconds>0 and synthesis_seconds>0"
        ).fetchall()
    return [Sample(int(c), float(a), float(s)) for c, a, s in rows]


async def run_sweep(repeats: int) -> list[Sample]:
    """Time real synthesis across length buckets.

    Deliberately goes through the engine rather than the HTTP layer: this is
    measuring the machine, and a loopback request would add a constant that
    varies with nothing.
    """
    from voiceagent.tts.chatterbox_indic import SAMPLE_RATE, ChatterboxIndicEngine
    from voiceagent.text.normalize_hi import normalize as normalize_hi

    if not REF_WAV.exists():
        console.print(f"[red]missing reference clip:[/] {REF_WAV}")
        return []

    audio, sr = sf.read(str(REF_WAV), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    engine = ChatterboxIndicEngine()
    engine.set_reference(audio, "", sr)
    console.print("loading engine ...")
    engine.load()

    # One throwaway generation first. The first call after load pays for lazy
    # graph construction and conditional caching, and folding that into the
    # fixed term would attribute a one-off startup cost to every request.
    await engine._run_async(engine._generate_blocking, "नमस्ते।")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples: list[Sample] = []
    for target in SWEEP_BUCKETS:
        text = normalize_hi(sweep_text(target))
        for repeat in range(repeats):
            started = time.perf_counter()
            chunks = [chunk async for chunk in engine.synthesize(text)]
            elapsed = time.perf_counter() - started
            if not chunks:
                console.print(f"[red]no audio at {target} chars[/]")
                continue
            pcm = np.concatenate([c.samples for c in chunks])
            seconds = len(pcm) / SAMPLE_RATE
            samples.append(Sample(len(text), seconds, elapsed))
            console.print(
                f"  {len(text):5d} chars  audio {seconds:6.1f}s  "
                f"synth {elapsed:6.1f}s  rtf {elapsed / seconds:.2f}"
            )
            if repeat == 0:
                sf.write(OUT_DIR / f"sweep_{target}.wav", pcm, SAMPLE_RATE)
    return samples


def report(model: CostModel, source: str) -> list[str]:
    """Console table plus the same content as markdown lines."""
    console.print()
    console.print(f"[bold]Cost model — {source}[/]")
    console.print(
        f"  synthesis_seconds = {model.fixed_seconds:.2f} "
        f"+ {model.marginal_seconds_per_char * 1000:.1f} per 1000 chars"
    )
    console.print(
        f"  fitted on {model.samples} samples spanning "
        f"{model.min_chars}-{model.max_chars} characters"
    )
    console.print(f"  {model.chars_per_audio_second:.1f} characters per second of audio")

    table = Table(title="The same 10,000 billable characters", title_justify="left",
                  header_style="bold")
    table.add_column("Request size", justify="right")
    table.add_column("Machine time", justify="right")
    table.add_column("vs best", justify="right")
    table.add_column("₹/10k", justify="right")
    table.add_column("Chars/hour", justify="right")
    table.add_column("", justify="left")

    sizes = [s for s in (40, 150, 400, 1000, 2000, 5000)]
    # Anchored on the cheapest batching across the whole table, not on the
    # cheapest *in-range* one. Anchoring in-range made the worst row read as
    # 1.0x and every other row as a fraction of it, which is exactly backwards
    # and is the number a pricing decision would be read off.
    best = min(model.machine_seconds_per_10k(s) for s in sizes)

    lines = [
        f"## Cost model — {source}",
        "",
        f"`synthesis_seconds = {model.fixed_seconds:.2f} + "
        f"{model.marginal_seconds_per_char * 1000:.1f} per 1000 chars`  ",
        f"Fitted on {model.samples} samples spanning {model.min_chars}–{model.max_chars} "
        f"characters. {model.chars_per_audio_second:.1f} characters per second of audio.",
        "",
        "| Request size | Machine time for 10k chars | vs best | ₹/10k | Chars/hour | |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for size in sizes:
        seconds = model.machine_seconds_per_10k(size)
        ratio = f"{seconds / best:.1f}×" if best else "-"
        note = "" if model.in_range(size) else "extrapolated"
        table.add_row(
            f"{size:,}",
            f"{seconds / 60:.1f} min",
            ratio,
            f"₹{model.inr_per_10k(size):.3f}",
            f"{model.chars_per_hour(size):,.0f}",
            f"[yellow]{note}[/]" if note else "",
        )
        lines.append(
            f"| {size:,} | {seconds / 60:.1f} min | {ratio} | "
            f"₹{model.inr_per_10k(size):.3f} | {model.chars_per_hour(size):,.0f} | "
            f"{'**extrapolated**' if note else ''} |"
        )

    console.print()
    console.print(table)

    # For the market comparison, use the cheapest batching we actually measured.
    cheapest = max((s for s in sizes if model.in_range(s)), default=min(sizes))
    lines += ["", "### Against the market", ""]
    console.print()
    console.print("[bold]Against the market[/]")
    for name, price in SARVAM_INR_PER_10K.items():
        if cheapest:
            ours = model.inr_per_10k(cheapest)
            share = ours / price * 100
            msg = f"  {name}: ₹{price:.0f}/10k — our electricity is {share:.2f}% of it"
            console.print(msg)
            lines.append(f"- **{name}**: ₹{price:.0f} per 10k. Our electricity is "
                         f"**{share:.2f}%** of that.")

    lines += [
        "",
        "Which is the wrong comparison to stop at. Electricity is not the",
        "constraint; the synthesis lock is. Concurrency is 1 by design, so the",
        "product's real budget is 86,400 machine-seconds a day, and the column",
        "that matters above is chars/hour, not ₹.",
        "",
        f"Every ₹ figure here scales linearly with two assumed numbers — "
        f"{ASSUMED_WATTS:.0f} W package power and ₹{ASSUMED_TARIFF_INR_PER_KWH:.0f}/kWh — "
        "neither of which is measured. No capacity figure depends on either. That "
        "asymmetry is the argument for pricing and rate-limiting on capacity.",
    ]
    console.print()
    console.print("[yellow]Electricity is not the constraint; the synthesis lock is. "
                  "Concurrency is 1, so the real budget is 86,400 machine-seconds a day.[/]")
    return lines


def _capacity_lines(model: CostModel) -> list[str]:
    """What a day of the one machine actually holds."""
    lines = ["", "### A day of the one machine", "",
             "| If every request is | Requests/day | Billable chars/day | Audio/day |",
             "| --- | --- | --- | --- |"]
    for size in (40, 400, 2000):
        per_request = model.seconds_for(size)
        requests = 86_400 / per_request
        chars = requests * size
        audio_hours = chars / model.chars_per_audio_second / 3600
        flag = "" if model.in_range(size) else " *(extrapolated)*"
        lines.append(
            f"| {size:,} chars{flag} | {requests:,.0f} | {chars:,.0f} | "
            f"{audio_hours:.1f} h |"
        )
    # Whether request size matters at all is an empirical question, and the
    # answer changed once the sweep replaced the narrow metering fit. Writing
    # either conclusion as a fixed sentence would leave the prose contradicting
    # the table above it the next time the machine is re-measured.
    spread = max(rows_chars := [
        model.seconds_for(size) / size for size in (40, 400, 2000)
    ]) / min(rows_chars)

    # The intercept is the fragile term and this is the honest statement of it.
    # Four independent sweeps over the same buckets produced intercepts of
    # -0.03, 0.20, 0.69 and 3.27 seconds while the slope stayed within 37.7-41.9
    # s per 1000 characters. Whatever the current run says about fixed cost, it
    # is one load spike away from saying something else.
    lines += [
        "",
        "**The slope is the trustworthy half of this model.** Across four independent",
        "sweeps of the same buckets the marginal rate stayed within 37.7–41.9 s per",
        "1000 characters, while the fitted intercept moved between −0.03 s and 3.27 s",
        "— driven by whichever bucket happened to catch a load spike. Repeats are now",
        "collapsed to per-bucket medians to damp that, but the conclusion stands:",
        "**price on the per-character rate, and treat any fixed-cost figure here as",
        "unresolved in either direction.**",
    ]

    if not model.separates_fixed_from_marginal:
        # The whole point of this guard. A fit over a 5x span reported a 4.6x
        # batching penalty that a 36x sweep showed did not exist, and the
        # default invocation of this module used to write that number into a
        # findings file as though it were settled.
        return lines + [
            "",
            f"**No verdict on batching from this fit.** It spans "
            f"{model.min_chars}–{model.max_chars} characters, a "
            f"{model.range_ratio:.0f}× range, which cannot separate a fixed "
            "per-request cost from a per-character one — the intercept absorbs "
            "load and noise and reports them as structure. The rows above differ "
            f"by {spread:.1f}×, and that number is an artefact until a wider fit "
            "confirms it.",
            "",
            "Run `--sweep` for a fit that spans enough range to answer this.",
        ]

    # The batching penalty is whatever the intercept is, and the intercept is a
    # range rather than a number. Quoting it from the current run's fit would be
    # the third confident answer this module has given to the same question.
    low, high = OBSERVED_INTERCEPT_SECONDS
    per_char_at = lambda fixed, size: (fixed + model.marginal_seconds_per_char * size) / size
    penalty_low = per_char_at(max(low, 0.0), 40) / per_char_at(max(low, 0.0), 2000)
    penalty_high = per_char_at(high, 40) / per_char_at(high, 2000)

    lines += [
        "",
        "At 100% utilisation of a machine that is also a workstation, so read these",
        "as a ceiling nobody reaches, not a plan.",
        "",
        "### Does request size change what a customer costs?",
        "",
        f"**Unresolved, and bounded between {penalty_low:.1f}× and {penalty_high:.1f}×.** "
        "That is the honest answer and it is the third one this module has produced. "
        "The penalty is entirely a function of the fixed per-request cost, which across "
        f"four sweeps sat anywhere from {low:.2f} s to {high:.2f} s — so a 40-character "
        f"request costs between {penalty_low:.1f}× and {penalty_high:.1f}× as much per "
        "character as a 2,000-character one, and the data does not say where in that "
        "band the truth is.",
        "",
        "What follows for pricing: **charge per character**, because the marginal rate "
        "is solid and dominates at every request size worth serving. Do **not** build a "
        "short-request surcharge on these numbers — at the bottom of the band there is "
        "nothing to surcharge, and at the top it is still small next to the "
        "0.14–0.28% that electricity costs against the market price. Revisit only if "
        "conversational traffic becomes the dominant workload, and re-measure on a "
        "quiet machine before doing so.",
    ]
    return lines


def _reconcile(swept: CostModel, metered: CostModel) -> list[str]:
    """Where the two fits disagree, and which one to believe about what.

    They disagreed sharply the first time this ran, and the reason is worth
    keeping rather than smoothing over. The metering fit is calibrated on
    29-142 characters — a 5x span — and over a span that narrow the slope is
    barely identified, so the intercept absorbs load, cold starts and noise and
    reports them as "fixed cost per request". The sweep spans 56-2036, a 36x
    range, which is what actually separates the two terms.

    Conclusion the first run overturned: there is *not* a large fixed overhead.
    The metering fit said 3.12 s per call and implied short requests cost ~5x per
    character; the sweep put it under 0.7 s, and a second sweep slightly below
    zero. Pricing built on the first number would have penalised conversational
    traffic for an artefact of a narrow-range regression.

    What the metering fit is still the authority on: real requests really are
    slower and more variable than the sweep, because production pays for
    reference conditioning, encoding and whatever else the machine is doing.
    Believe the sweep about the *machine* and the metering about the *service*.
    """
    ratio = (
        metered.fixed_seconds / swept.fixed_seconds
        if swept.fixed_seconds > 0.01
        else float("inf")
    )
    overstated = (
        f"~{ratio:.1f}x" if ratio != float("inf")
        else "immeasurably far"
    )
    return [
        "",
        "## Reconciling the two fits",
        "",
        "| | Fixed per request | Per 1000 chars | Fitted over |",
        "| --- | --- | --- | --- |",
        f"| Sweep | {swept.fixed_seconds:.2f} s | "
        f"{swept.marginal_seconds_per_char * 1000:.1f} s | "
        f"{swept.min_chars}–{swept.max_chars} chars |",
        f"| Metering | {metered.fixed_seconds:.2f} s | "
        f"{metered.marginal_seconds_per_char * 1000:.1f} s | "
        f"{metered.min_chars}–{metered.max_chars} chars |",
        "",
        "**Believe the sweep about the machine and the metering about the service.**",
        "The metering fit spans a 5× range of request sizes, which is not enough to",
        "separate a fixed term from a slope — the intercept absorbs load, cold starts",
        "and noise and then reports them as per-call overhead. The sweep spans 36×.",
        "",
        f"The narrow fit's intercept sits {overstated} above this one, and a price built",
        "on it would have penalised short-request conversational traffic for an artefact",
        "of the regression. That does not make the swept intercept correct either — see",
        "the bound above. It makes the *slope* the only part either fit agrees on.",
        "",
        "What the metering fit still shows truthfully is that production requests are",
        "slower and far more variable than the sweep. The swept marginal rate implies a",
        f"steady **RTF {swept.marginal_seconds_per_char * swept.chars_per_audio_second:.2f}**; "
        "real traffic measured 0.85 median with one request at 4.10.",
        "That gap is the real service overhead — reference conditioning, encoding, and",
        "whatever else the machine was doing — and it is the number capacity planning",
        "should use. The sweep is the floor, not the forecast.",
    ]


async def main_async(args) -> int:
    metered_samples = from_metering()
    swept: CostModel | None = None
    metered: CostModel | None = None

    if metered_samples:
        try:
            metered = fit(metered_samples)
        except ValueError:
            metered = None

    if args.sweep:
        swept_samples = await run_sweep(args.repeats)
        if not swept_samples:
            return 2
        try:
            swept = fit(by_bucket_median(swept_samples))
        except ValueError as exc:
            console.print(f"[red]{exc}[/]")
            return 2

    model = swept or metered
    if model is None:
        console.print("[red]no usable rows in the metering database.[/]")
        console.print("Run with --sweep to measure directly instead.")
        return 2

    source = (
        f"sweep, {args.repeats} repeat(s) per bucket"
        if swept
        else f"production metering ({USAGE_DB.name})"
    )

    lines = report(model, source)
    lines += _capacity_lines(model)
    if swept and metered:
        lines += _reconcile(swept, metered)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    header = [
        "# What a generation costs",
        "",
        "Generated by `uv run python -m voiceagent.eval.cogs`.",
        "",
        "Load-sensitive. Check `voiceagent.eval.specsheet --check` before trusting",
        "any figure here.",
        "",
    ]
    (OUT_DIR / "FINDINGS.md").write_text("\n".join(header + lines) + "\n")
    console.print(f"\nwritten: {OUT_DIR / 'FINDINGS.md'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true",
                        help="measure directly across length buckets instead of reading metering")
    parser.add_argument("--repeats", type=int, default=2,
                        help="generations per bucket in a sweep (default 2)")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
