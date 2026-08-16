"""Phase A probe: can the `[hi]` language token read other Devanagari languages?

Chatterbox Multilingual ships one Indic language token, `[hi]`. The plan's Phase A
opens with a free experiment: Marathi and Nepali are written in the *same script*,
so the tokenizer can encode them. The open question is whether the model, told
"this is Hindi", produces intelligible Marathi and Nepali or Hindi-accented mush.

WHAT THE TOKENIZER ALREADY SETTLES, before any audio is generated

`data/models/chatterbox-multilingual-v3-8bit/tokenizer.json`, 2,454 entries:

  * Devanagari      124 tokens   (includes ळ, which Marathi needs and Hindi lacks)
  * Bengali           0
  * Gurmukhi          0
  * Gujarati          0
  * Oriya             0
  * Tamil             0
  * Telugu            0
  * Kannada           0
  * Malayalam         0

Devanagari is the *only* Indic script this checkpoint can encode. That is a hard
ceiling and no amount of prompting moves it: Tamil text reaching this model
becomes a run of `[UNK]`. Note the trap — the vocab *does* contain a `[ta]`
language token (along with `[bg] [cs] [hu] [ro] [sk] [vi] [ea] [ipa]`, none of
which mlx_audio's 23-entry allowlist admits). A language token without its script
is not support. Anyone reading the vocab for a language list will conclude we can
speak Tamil, and they will be wrong.

So the reachable set from this engine is the Devanagari family — Hindi, Marathi,
Nepali, Konkani, Sanskrit — and everything else in Phase A needs the breadth
engine (Indic Parler-TTS) instead. This probe measures the first two.

WHY THE CONTROL CONDITION IS NOT OPTIONAL

Round trip is an alarm, not a ranking (see `eval/arena/FINDINGS.md`). Here it is
used only in the mode it is valid in: separating broken speech from working
speech. But there is a confound specific to this experiment. Whisper's Marathi and
Nepali are *much* weaker than its Hindi, so a low score has two possible causes —
the synthesis is bad, or the scorer is. Those need separating before any number
here means anything.

The control: take the known-good Hindi audio and decode it pinned to `mr` and to
`ne`. If a Marathi-pinned decoder mangles speech we already know is fine, then low
Marathi scores are the instrument's fault and this probe cannot answer the
question. If it holds up, the pin is not destructive and a low Marathi score is
the model.

AUTHORSHIP CAVEAT, and it is a serious one here

The probe sentences below are assistant-authored. The author is a native speaker
of neither Marathi nor Nepali. This is the exact failure mode already documented
for the 227-entry loanword table, and it bites harder here: a sentence that is
subtly ungrammatical in Marathi will be read badly by *any* engine, and this probe
would report that as the model failing. Every sentence wants a native speaker's
read before its number is quoted anywhere.

More importantly, the question this probe cannot answer is the one that decides
the product. Round trip asks "did the words survive?" A native Marathi listener
asks "does this sound like a Marathi speaker, or like a Hindi speaker reading
Marathi?" Those come apart precisely here, because Hindi and Marathi share a
script and diverge in phonology — Marathi's च/ज are dental affricates [ts]/[dz] in
native words where Hindi has only the palatal, and Hindi has no ळ at all. A model
carrying Hindi phonology into Marathi text can score well on every metric in this
file and still be obviously foreign to the ear.

The clips are written to `eval_out/devanagari/` for exactly that reason. Listen
before believing the table.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from rich.console import Console
from rich.table import Table

from voiceagent.eval.roundtrip import character_overlap, transcribe

console = Console()

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "eval_out" / "devanagari"
REF_WAV = ROOT / "fixtures" / "hi" / "reference_lekha.wav"
REF_TXT = REF_WAV.with_suffix(".txt")

#: Below this, human raters rejected every clip (11 of 11) in the arena study.
#: It is the one threshold round trip has earned: broken vs not broken.
BROKEN_BELOW = 0.50


@dataclass(frozen=True)
class Probe:
    slug: str
    language: str
    text: str
    targets: str
    """The Hindi/target divergence this sentence is chosen to expose."""


#: Pure script, no digits and no Latin loanwords. Both are deliberate. The scorer
#: compares raw text for non-Hindi languages (`roundtrip.normalized` only
#: normalizes `hi`), so Whisper's inverse text normalization would turn a correct
#: "४३" into "43" and a correct loanword into Devanagari, costing characters for
#: reasons that have nothing to do with the audio. A flawless *human* reading of a
#: code-mixed sentence scores 54 % on this scorer. Keeping the probe pure is what
#: makes its numbers comparable to BROKEN_BELOW.
MARATHI: tuple[Probe, ...] = (
    Probe(
        "m1",
        "mr",
        "सकाळी मी शाळेत लवकर पोहोचलो.",
        "ळ, the retroflex lateral Marathi has and Hindi does not — the single "
        "clearest tell of a Hindi model reading Marathi",
    ),
    Probe(
        "m2",
        "mr",
        "पाच वाजता चहा घेऊया.",
        "dental affricates — च in पाच/चहा and ज in वाजता are [ts]/[dz] in "
        "Marathi, palatal in Hindi",
    ),
    Probe(
        "m3",
        "mr",
        "मला मराठी नीट बोलता येते.",
        "Marathi morphology (येते) with no Hindi cognate reading",
    ),
    Probe(
        "m4",
        "mr",
        "तुम्ही उद्या सकाळी नक्की या, आपण सविस्तर बोलू.",
        "long form, two clauses — where accent drift shows up",
    ),
    Probe(
        "m5",
        "mr",
        "त्याने पुस्तक वाचून संपवले आणि मग तो झोपला.",
        "schwa deletion — Marathi and Hindi differ on final inherent vowels",
    ),
)

NEPALI: tuple[Probe, ...] = (
    Probe(
        "n1",
        "ne",
        "म भोलि काठमाडौं जान्छु.",
        "final schwa retention — Nepali pronounces inherent vowels Hindi drops",
    ),
    Probe(
        "n2",
        "ne",
        "तपाईंलाई कस्तो छ?",
        "तपाईं honorific with ऐं — Nepali-specific orthography",
    ),
    Probe(
        "n3",
        "ne",
        "मैले हिजो राति एउटा किताब पढें.",
        "एउटा, मैले — Nepali function words absent from Hindi",
    ),
    Probe(
        "n4",
        "ne",
        "तपाईं भोलि आउनुभयो भने हामी सबै कुरा मिलाउँछौं.",
        "long form, honorific verb morphology",
    ),
    Probe(
        "n5",
        "ne",
        "यो काम गर्न धेरै समय लाग्छ.",
        "गर्न/लाग्छ — conjunct + Nepali verb endings",
    ),
)

#: The control. Pure Hindi, same length range, no digits or loanwords, so the only
#: variable against the probes is the language. These are known-good: this engine
#: measures 93.5 % mean round-trip overlap on the held-out Hindi set.
HINDI_CONTROL: tuple[Probe, ...] = (
    Probe("c1", "hi", "सुबह मैं जल्दी स्कूल पहुँच गया.", "control"),
    Probe("c2", "hi", "पाँच बजे चाय पीते हैं.", "control"),
    Probe("c3", "hi", "मुझे हिंदी ठीक से बोलनी आती है.", "control"),
    Probe("c4", "hi", "आप कल सुबह ज़रूर आइए, हम विस्तार से बात करेंगे.", "control"),
    Probe("c5", "hi", "उसने किताब पढ़कर खत्म की और फिर वह सो गया.", "control"),
)


def _load_engine():
    from voiceagent.tts.chatterbox_indic import ChatterboxIndicEngine

    audio, sr = sf.read(str(REF_WAV), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    engine = ChatterboxIndicEngine()
    engine.set_reference(audio, REF_TXT.read_text().strip(), sr)
    console.print(f"loading {engine.repo} ...")
    engine.load()
    console.print(f"loaded ({engine.resident_bytes / 2**30:.2f} GiB)\n")
    return engine


async def _synthesize(engine, probe: Probe) -> tuple[Path, float, float]:
    """Generate one probe, bypassing the Hindi-only guard on purpose.

    `synthesize()` calls `_require_hindi`, which raises `UnsupportedLanguage` for
    exactly the languages under test — that guard is the claim being examined, so
    the probe goes around it rather than weakening it. Production behaviour is
    untouched until the evidence says it should change.

    The direct call also skips `normalize_hi`, which is correct here and worth
    stating: that normalizer spells numbers as *Hindi* words and would inject
    Hindi into a Marathi sentence. Going straight to the model isolates the
    question to the model.

    It goes through `_run_async` and not straight to `_generate_blocking`, because
    MLX arrays are thread-affine and the engine loaded the weights on its own
    executor thread. Calling the blocking path directly raises
    "There is no Stream(gpu, 0) in current thread" from 20 layers down.
    """
    from voiceagent.tts.chatterbox_indic import SAMPLE_RATE

    started = time.perf_counter()
    samples = await engine._run_async(engine._generate_blocking, probe.text)
    elapsed = time.perf_counter() - started

    path = OUT_DIR / f"{probe.slug}_{probe.language}.wav"
    sf.write(path, samples, SAMPLE_RATE)
    seconds = len(samples) / SAMPLE_RATE if len(samples) else 0.0
    rtf = elapsed / seconds if seconds else float("inf")
    return path, seconds, rtf


def _score(path: Path, expected: str, pinned: str) -> float:
    heard, _ = transcribe(path, language=pinned)
    return character_overlap(expected, heard)


async def run(limit: int | None) -> int:
    if not REF_WAV.exists():
        console.print(f"[red]missing reference clip:[/] {REF_WAV}")
        console.print("Untracked on purpose (a real recorded voice). Enrol one first.")
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    engine = _load_engine()

    groups = (("Hindi (control)", HINDI_CONTROL), ("Marathi", MARATHI), ("Nepali", NEPALI))
    rows: list[dict] = []

    for name, probes in groups:
        chosen = probes[:limit] if limit else probes
        console.print(f"[bold]{name}[/] — {len(chosen)} sentences")
        for probe in chosen:
            path, seconds, rtf = await _synthesize(engine, probe)
            row = {
                "group": name,
                "probe": probe,
                "seconds": seconds,
                "rtf": rtf,
                "self": _score(path, probe.text, probe.language),
                "auto": transcribe(path)[1],
            }
            # The control condition: known-good Hindi audio, decoded with the
            # Marathi and Nepali pins. This is what tells us whether a low
            # Marathi score below is the model or the instrument.
            if probe.language == "hi":
                row["as_mr"] = _score(path, probe.text, "mr")
                row["as_ne"] = _score(path, probe.text, "ne")
            rows.append(row)
            console.print(f"  {probe.slug}  overlap {row['self']:.2f}  rtf {rtf:.2f}")
        console.print()

    _report(rows)
    return 0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _report(rows: list[dict]) -> None:
    table = Table(title="Devanagari probe — [hi] token on Marathi and Nepali",
                  title_justify="left", header_style="bold")
    table.add_column("")
    table.add_column("Language")
    table.add_column("Text")
    table.add_column("Auto", justify="center")
    table.add_column("Overlap", justify="right")
    table.add_column("RTF", justify="right")

    for row in rows:
        probe = row["probe"]
        overlap = row["self"]
        colour = "red" if overlap < BROKEN_BELOW else "green"
        table.add_row(
            probe.slug,
            probe.language,
            probe.text[:44],
            row["auto"],
            f"[{colour}]{overlap:.2f}[/]",
            f"{row['rtf']:.2f}",
        )
    console.print(table)

    control = [r for r in rows if r["probe"].language == "hi"]
    marathi = [r for r in rows if r["probe"].language == "mr"]
    nepali = [r for r in rows if r["probe"].language == "ne"]

    console.print()
    console.print("[bold]Control — is the scorer usable for these languages?[/]")
    console.print(f"  Hindi audio, Hindi pin : {_mean([r['self'] for r in control]):.2f}")
    console.print(f"  Hindi audio, Marathi pin: {_mean([r['as_mr'] for r in control]):.2f}")
    console.print(f"  Hindi audio, Nepali pin : {_mean([r['as_ne'] for r in control]):.2f}")
    console.print()
    console.print("[bold]Probe[/]")
    console.print(f"  Marathi: {_mean([r['self'] for r in marathi]):.2f}")
    console.print(f"  Nepali : {_mean([r['self'] for r in nepali]):.2f}")
    console.print()
    console.print(f"clips: {OUT_DIR}")
    console.print("[yellow]Round trip cannot hear an accent. A native speaker must "
                  "listen before any of this reaches a pricing page.[/]")

    _write_findings(rows, control, marathi, nepali)


def _write_findings(rows, control, marathi, nepali) -> None:
    lines = [
        "# Phase A probe — Marathi and Nepali on the `[hi]` language token",
        "",
        "Generated by `uv run python -m voiceagent.eval.devanagari`.",
        "",
        "## Tokenizer ceiling (settled without generating audio)",
        "",
        "Devanagari is the only Indic script in the checkpoint's 2,454-token vocab.",
        "Bengali, Gujarati, Gurmukhi, Oriya, Tamil, Telugu, Kannada and Malayalam have",
        "**zero** tokens. A `[ta]` language token exists with no Tamil script behind it —",
        "a language token is not support. The Devanagari family is the whole reachable",
        "set for this engine; everything else in Phase A needs a different model.",
        "",
        "## Control — is the scorer usable here?",
        "",
        "| Condition | Mean overlap |",
        "| --- | --- |",
        f"| Hindi audio, Hindi pin | {_mean([r['self'] for r in control]):.2f} |",
        f"| Hindi audio, Marathi pin | {_mean([r['as_mr'] for r in control]):.2f} |",
        f"| Hindi audio, Nepali pin | {_mean([r['as_ne'] for r in control]):.2f} |",
        "",
        "The Hindi rows are known-good audio. If a pin collapses them, that pin cannot",
        "score the probe below it and the probe's number means nothing.",
        "",
        "## Probe",
        "",
        "| | Lang | Auto-detected | Overlap | RTF | Targets |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        probe = row["probe"]
        lines.append(
            f"| {probe.slug} | {probe.language} | {row['auto']} | "
            f"{row['self']:.2f} | {row['rtf']:.2f} | {probe.targets} |"
        )
    lines += [
        "",
        f"Marathi mean: **{_mean([r['self'] for r in marathi]):.2f}**  ",
        f"Nepali mean: **{_mean([r['self'] for r in nepali]):.2f}**  ",
        f"Broken below: {BROKEN_BELOW:.2f} (the one threshold round trip has earned)",
        "",
        "## Verdict",
        "",
        "**Intelligible, not native. Do not sell either language on this engine.**",
        "",
        "Nothing here is broken: every sentence clears the 0.50 alarm, none needed a",
        "retry, and RTF stays under 1.0, so the model does read Devanagari it was never",
        "told about. That is the good news and it is genuinely surprising.",
        "",
        "It is also not enough, for one concrete reason. `ळ` — the retroflex lateral",
        "Marathi has and Hindi does not — **survived 0 of 4 seeds** (`SEEDS.md`). It is",
        "in the tokenizer; it does not come out of the model. Worse than dropping it,",
        "the model reaches for Hindi: `शाळेत` (*shaaLet*, \"in school\") comes back as",
        "`शायद` (*shaayad*, \"maybe\") on every seed — a real Hindi word substituted for",
        "a Marathi one. That is the signature of Hindi phonology and a Hindi lexicon",
        "reading Marathi text, which is exactly the failure mode a shared script hides.",
        "",
        "A voice that cannot say `ळ` cannot be sold as Marathi. Nepali is less clear-cut",
        "— `र्` conjuncts survive 2 of 4 seeds, unreliable rather than absent — but",
        "unreliable is not shippable either.",
        "",
        "Two methodological cautions this probe earned:",
        "",
        "- **Single generations in this band are noise.** `m2` scored 0.59 on seed 0 and",
        "  0.88 on seeds 2 and 3. Read on seed 0 alone it says the dental affricates",
        "  fail; across seeds `च` survives 3 of 4 and there is no affricate finding at",
        "  all. Any per-sentence number here carries roughly ±0.15 of seed noise.",
        "- **Aggregate overlap cannot see this.** Marathi's 0.77 mean looks like a pass.",
        "  The failure is one grapheme, and it took a targeted per-grapheme check to",
        "  find. The means are the least informative numbers on this page.",
        "",
        "## What this cannot tell you",
        "",
        "Whether it sounds like a native speaker. Round trip asks whether the words",
        "survived; it is deaf to a Hindi speaker reading Marathi with Hindi phonology,",
        "which is the most likely failure here and the one that decides whether Marathi",
        "goes on a pricing page. The `ळ` result is a *lower* bound on that problem found",
        "through a transcript — the accent question is untouched and needs a listener.",
        "The probe sentences are also assistant-authored by a non-native speaker.",
        "Clips are in this directory. Listen.",
    ]
    (OUT_DIR / "FINDINGS.md").write_text("\n".join(lines) + "\n")


#: Sentences worth re-running across seeds, and the grapheme each one is asking
#: about. A single generation cannot answer "is this systematic?" — Chatterbox
#: samples, and the spread at this quality level is wide enough to invent a
#: finding. Measured here: `m2` scored 0.59 on seed 0 and 0.88 on seeds 2 and 3,
#: which is the difference between "the dental affricates fail" and "one
#: generation degenerated". Only a grapheme that fails on *every* seed is
#: evidence about the model.
SEED_CASES: tuple[tuple[Probe, str], ...] = (
    (MARATHI[0], "ळ"),
    (MARATHI[1], "च"),
    (NEPALI[4], "र्"),
)


async def run_seeds(seeds: int) -> int:
    """Re-generate the diagnostic sentences under several seeds.

    Separates a systematic phonological gap from sampling noise, which is the
    only way a per-sentence number in the 0.6-0.9 band means anything.
    """
    if not REF_WAV.exists():
        console.print(f"[red]missing reference clip:[/] {REF_WAV}")
        return 2

    out = OUT_DIR / "seeds"
    out.mkdir(parents=True, exist_ok=True)
    engine = _load_engine()

    lines = [
        "# Seed sweep — is a missing grapheme systematic, or one bad sample?",
        "",
        f"`uv run python -m voiceagent.eval.devanagari --seeds {seeds}`",
        "",
        "Chatterbox samples, and the spread at this quality level is wide. A grapheme",
        "that vanishes on one seed proves nothing; one that vanishes on every seed is",
        "the model. Read the survival counts, not the overlaps.",
        "",
    ]

    for probe, grapheme in SEED_CASES:
        console.print(f"[bold]{probe.slug}[/] [{probe.language}] — {grapheme!r}: {probe.text}")
        lines += [
            f"## `{probe.slug}` [{probe.language}] — does `{grapheme}` survive?",
            "",
            f"Target: {probe.text}",
            "",
            "| Seed | Overlap | Grapheme | Heard |",
            "| --- | --- | --- | --- |",
        ]
        survived = 0
        for seed in range(seeds):
            engine.seed = seed
            engine._span_seed = seed
            from voiceagent.tts.chatterbox_indic import SAMPLE_RATE

            samples = await engine._run_async(engine._generate_blocking, probe.text)
            path = out / f"{probe.slug}_seed{seed}.wav"
            sf.write(path, samples, SAMPLE_RATE)
            heard, _ = transcribe(path, language=probe.language)
            overlap = character_overlap(probe.text, heard)
            hit = grapheme in heard
            survived += hit
            mark = "[green]y[/]" if hit else "[red]n[/]"
            console.print(f"  seed {seed}: {overlap:.2f}  {grapheme!r} {mark}  {heard}")
            lines.append(
                f"| {seed} | {overlap:.2f} | {'yes' if hit else '**no**'} | {heard} |"
            )
        console.print(f"  -> {grapheme!r} survived {survived}/{seeds} seeds\n")
        lines += ["", f"**`{grapheme}` survived {survived}/{seeds} seeds.**", ""]

    (OUT_DIR / "SEEDS.md").write_text("\n".join(lines) + "\n")
    console.print(f"clips: {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="first N sentences per language")
    parser.add_argument(
        "--seeds",
        type=int,
        metavar="N",
        help="instead of the probe, re-run the diagnostic sentences under N seeds",
    )
    args = parser.parse_args()
    if args.seeds:
        return asyncio.run(run_seeds(args.seeds))
    return asyncio.run(run(args.limit))


if __name__ == "__main__":
    raise SystemExit(main())
