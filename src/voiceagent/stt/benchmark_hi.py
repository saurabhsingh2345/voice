"""Which STT engine can actually understand Hindi, and at what cost?

    uv run python -m voiceagent.stt.benchmark_hi
    uv run python -m voiceagent.stt.benchmark_hi --skip-moonshine

Reports character error rate, not just latency, because for Hindi the question
is comprehension rather than speed. CER rather than WER: Hindi orthography makes
word boundaries and matra placement vary between correct spellings, so WER
punishes cosmetic differences that a listener would not notice.

Moonshine is included on purpose, as a *demonstration of a hazard* rather than a
candidate. It is English-only here and does not refuse Hindi audio -- it invents
English words for it. Seeing that failure printed next to a real number is more
convincing than the warning in its docstring.

Audio comes from two sources, and they answer different questions:

  * The real recorded Hindi clip in fixtures/hi/ -- genuine human speech, so it
    is the honest measurement. There is only one, so it is a spot check.
  * The synthesized set in eval_out/hindi_tts/ -- 22 sentences across four
    registers, written by voiceagent.eval.hindi_tts. Breadth, but circular in one
    respect worth stating: it measures ASR against *this* TTS's Hindi, not
    against human speakers. Run the TTS eval first to populate it.

Both are untracked (real speech and clones of it), so this reports what is
present rather than failing.
"""

from __future__ import annotations

import argparse
import statistics
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from rich.console import Console
from rich.table import Table

from voiceagent.stt.base import SAMPLE_RATE, STTEngine

console = Console()

ROOT = Path(__file__).resolve().parents[3]
REAL_CLIP = ROOT / "fixtures" / "hi" / "reference_lekha.wav"
SYNTH_DIR = ROOT / "eval_out" / "hindi_tts"


@dataclass(frozen=True)
class Clip:
    path: Path
    text: str
    source: str


def _strip_for_scoring(text: str) -> str:
    """Normalize away differences that are not recognition errors.

    Punctuation, Devanagari danda, and whitespace are removed, and the string is
    NFC-composed so that a matra written as a combining sequence compares equal
    to its precomposed form. Without NFC two visually identical strings can score
    as completely different.
    """
    text = unicodedata.normalize("NFC", text)
    drop = {"।", "॥", ".", ",", "?", "!", ";", ":", '"', "'", "-", "—"}
    return "".join(ch for ch in text if ch not in drop and not ch.isspace())


def character_error_rate(expected: str, heard: str) -> float:
    """Levenshtein distance over characters, divided by expected length."""
    a, b = _strip_for_scoring(expected), _strip_for_scoring(heard)
    if not a:
        return 0.0 if not b else 1.0

    # Two-row DP; the strings here are single sentences, so this is cheap.
    previous = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, start=1):
        current = [i]
        for j, ch_b in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,                              # deletion
                current[j - 1] + 1,                           # insertion
                previous[j - 1] + (ch_a != ch_b),             # substitution
            ))
        previous = current
    return previous[-1] / len(a)


def devanagari_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "ऀ" <= c <= "ॿ") / len(letters)


def collect_clips() -> list[Clip]:
    clips: list[Clip] = []

    if REAL_CLIP.exists() and REAL_CLIP.with_suffix(".txt").exists():
        clips.append(Clip(REAL_CLIP, REAL_CLIP.with_suffix(".txt").read_text().strip(), "real"))
    else:
        console.print(f"[yellow]no real Hindi clip at {REAL_CLIP} -- skipping[/]")

    if SYNTH_DIR.exists():
        # The eval writes <slug>_<register>.wav; recover the text from the same
        # sentence table it used, so the two cannot drift apart.
        from voiceagent.eval import sentences as S
        from voiceagent.text.normalize_hi import normalize as normalize_hi

        by_slug = {s.slug: s for s in S.HINDI_ONLY}
        for wav in sorted(SYNTH_DIR.glob("*.wav")):
            slug = wav.stem.split("_")[0]
            sentence = by_slug.get(slug)
            if sentence is None:
                continue
            # The TTS spoke the normalized form, so that is the ground truth.
            clips.append(Clip(wav, normalize_hi(sentence.text), f"synth:{sentence.register}"))
    else:
        console.print(f"[yellow]no synthesized clips at {SYNTH_DIR}[/] "
                      "-- run voiceagent.eval.hindi_tts to populate it")

    return clips


def _load(path: Path) -> tuple[np.ndarray, float]:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        idx = (np.arange(int(len(audio) * SAMPLE_RATE / sr)) * sr / SAMPLE_RATE).astype(int)
        audio = audio[idx[idx < len(audio)]]
    return audio, len(audio) / SAMPLE_RATE


@dataclass
class Row:
    label: str
    cers: list[float]
    rtfs: list[float]
    languages: list[str]
    devanagari: list[float]
    resident_bytes: int
    load_s: float
    sample_heard: str = ""

    @property
    def median_cer(self) -> float:
        return statistics.median(self.cers) if self.cers else float("nan")

    @property
    def median_rtf(self) -> float:
        return statistics.median(self.rtfs) if self.rtfs else float("nan")


def bench(engine: STTEngine, label: str, clips: list[Clip]) -> Row:
    started = time.perf_counter()
    engine.load()
    load_s = time.perf_counter() - started

    row = Row(label, [], [], [], [], engine.resident_bytes, load_s)
    for clip in clips:
        audio, seconds = _load(clip.path)
        result = engine.transcribe(audio)
        row.cers.append(character_error_rate(clip.text, result.text))
        if result.latency_ms is not None and seconds > 0:
            row.rtfs.append((result.latency_ms / 1000) / seconds)
        row.languages.append(result.language or "?")
        row.devanagari.append(devanagari_share(result.text))
        if clip.source == "real" and not row.sample_heard:
            row.sample_heard = result.text

    row.resident_bytes = engine.resident_bytes
    engine.unload()
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-moonshine", action="store_true",
                        help="skip the English-only engine (it is here to show a hazard)")
    parser.add_argument("--limit", type=int, help="first N clips only")
    args = parser.parse_args()

    clips = collect_clips()
    if not clips:
        console.print("[red]no Hindi audio available to benchmark[/]")
        return 2
    if args.limit:
        clips = clips[: args.limit]

    console.print(f"{len(clips)} clips "
                  f"({sum(1 for c in clips if c.source == 'real')} real, "
                  f"{sum(1 for c in clips if c.source != 'real')} synthesized)\n")

    from voiceagent.stt.mlx_whisper_engine import MLXWhisperEngine

    candidates: list[tuple[str, callable]] = [
        ("whisper-large-v3-turbo (hi pinned)", lambda: MLXWhisperEngine(language="hi")),
        ("whisper-large-v3-turbo (auto-detect)", lambda: MLXWhisperEngine(language=None)),
    ]
    if not args.skip_moonshine:
        from voiceagent.stt.moonshine_engine import MoonshineEngine

        candidates.append(
            ("moonshine small-streaming (English-only)",
             lambda: MoonshineEngine(arch="SMALL_STREAMING"))
        )

    rows: list[Row] = []
    for label, factory in candidates:
        console.print(f"benchmarking [bold]{label}[/] ...")
        try:
            rows.append(bench(factory(), label, clips))
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [red]failed:[/] {type(exc).__name__}: {exc}")

    table = Table(title="Hindi STT", title_justify="left", header_style="bold")
    table.add_column("Engine")
    table.add_column("Load", justify="right")
    table.add_column("Model mem", justify="right")
    table.add_column("Median CER", justify="right")
    table.add_column("Median RTF", justify="right")
    table.add_column("Detected", justify="center")
    table.add_column("Devanagari out", justify="right")

    for row in rows:
        langs = sorted(set(row.languages))
        deva = statistics.median(row.devanagari) if row.devanagari else 0.0
        cer = row.median_cer
        table.add_row(
            row.label,
            f"{row.load_s:.1f} s",
            f"{row.resident_bytes / 2**20:.0f} MiB",
            f"{cer:.1%}" if cer < 0.5 else f"[red]{cer:.1%}[/]",
            f"{row.median_rtf:.3f}",
            ",".join(langs) if len(langs) <= 3 else f"{len(langs)} langs",
            f"{deva:.0%}" if deva > 0.9 else f"[red]{deva:.0%}[/]",
        )

    console.print(table)

    for row in rows:
        if row.sample_heard:
            console.print(f"\n[bold]{row.label}[/] on the real clip:\n  {row.sample_heard}")

    best = min((r for r in rows if r.cers), key=lambda r: r.median_cer, default=None)
    if best:
        console.print(f"\nLowest CER: [green]{best.label}[/] at {best.median_cer:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
