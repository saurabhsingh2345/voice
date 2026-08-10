"""Phase 1 benchmark: Moonshine vs. MLX-Whisper on identical audio.

Run with::

    uv run python -m voiceagent.stt.benchmark

Measures, per engine: model load time, resident memory, per-fixture transcribe
latency, and the real-time factor (RTF = compute time / audio duration). RTF
below 1.0 means the engine can keep up with live speech; the lower the better.

Each engine is loaded, measured, and unloaded in isolation so the two never
share residency.
"""

from __future__ import annotations

import gc
import statistics
import time
from dataclasses import dataclass, field

import numpy as np
import psutil
import soundfile as sf
from rich.console import Console
from rich.table import Table

from voiceagent.stt import fixtures as fx
from voiceagent.stt.base import SAMPLE_RATE, STTEngine

console = Console()
MIB = 1024**2

#: Each fixture is transcribed this many times; we report the median to blunt
#: the effect of macOS scheduling noise.
REPEATS = 3


@dataclass
class FixtureResult:
    slug: str
    duration_s: float
    latencies_ms: list[float] = field(default_factory=list)
    text: str = ""

    @property
    def median_ms(self) -> float:
        return statistics.median(self.latencies_ms)

    @property
    def rtf(self) -> float:
        return (self.median_ms / 1000) / self.duration_s


@dataclass
class EngineResult:
    name: str
    load_s: float
    resident_mib: float
    rss_delta_mib: float
    fixtures: list[FixtureResult] = field(default_factory=list)
    error: str | None = None

    @property
    def median_rtf(self) -> float:
        return statistics.median([f.rtf for f in self.fixtures]) if self.fixtures else float("nan")


def _load_audio(path) -> tuple[np.ndarray, float]:
    audio, sr = sf.read(path, dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path} is {sr} Hz, expected {SAMPLE_RATE} Hz")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, len(audio) / sr


def bench_engine(engine: STTEngine, items: list[fx.Fixture]) -> EngineResult:
    proc = psutil.Process()
    gc.collect()
    rss_before = proc.memory_info().rss

    console.print(f"[dim]loading {engine.name}...[/]")
    started = time.perf_counter()
    try:
        engine.load()
    except Exception as exc:  # noqa: BLE001 -- report, do not abort the whole run
        return EngineResult(engine.name, 0.0, 0.0, 0.0, error=str(exc))
    load_s = time.perf_counter() - started

    rss_after = proc.memory_info().rss
    result = EngineResult(
        name=engine.name,
        load_s=load_s,
        resident_mib=engine.resident_bytes / MIB,
        rss_delta_mib=(rss_after - rss_before) / MIB,
    )

    for item in items:
        audio, duration = _load_audio(item.path)
        fr = FixtureResult(slug=item.slug, duration_s=duration)
        for _ in range(REPEATS):
            out = engine.transcribe(audio)
            fr.latencies_ms.append(out.latency_ms or 0.0)
            fr.text = out.text
        result.fixtures.append(fr)
        console.print(
            f"  [dim]{item.slug:10s} {duration:4.1f}s audio -> "
            f"{fr.median_ms:7.1f} ms  RTF {fr.rtf:.3f}[/]"
        )

    engine.unload()
    gc.collect()
    return result


def report(results: list[EngineResult], items: list[fx.Fixture]) -> None:
    summary = Table(title="STT engine comparison", title_justify="left", header_style="bold")
    summary.add_column("Engine")
    summary.add_column("Load", justify="right")
    summary.add_column("Model mem", justify="right")
    summary.add_column("Proc RSS delta", justify="right")
    summary.add_column("Median RTF", justify="right")

    for r in results:
        if r.error:
            summary.add_row(r.name, "[red]FAILED[/]", "-", "-", "-")
            continue
        summary.add_row(
            r.name,
            f"{r.load_s:.1f}s",
            f"{r.resident_mib:.0f} MiB",
            f"{r.rss_delta_mib:.0f} MiB",
            f"{r.median_rtf:.3f}",
        )
    console.print(summary)
    console.print()

    detail = Table(title="Per-fixture latency (median of 3)", title_justify="left", header_style="bold")
    detail.add_column("Fixture")
    detail.add_column("Audio", justify="right")
    for r in results:
        detail.add_column(r.name, justify="right")

    for item in items:
        row = [item.slug]
        duration = next(
            (f.duration_s for r in results for f in r.fixtures if f.slug == item.slug), 0.0
        )
        row.append(f"{duration:.1f}s")
        for r in results:
            match = next((f for f in r.fixtures if f.slug == item.slug), None)
            row.append(f"{match.median_ms:.0f} ms" if match else "-")
        detail.add_row(*row)
    console.print(detail)
    console.print()

    for r in results:
        if r.error:
            console.print(f"[red]{r.name} failed:[/] {r.error}")
            continue
        console.print(f"[bold]{r.name}[/] transcripts:")
        for f in r.fixtures:
            expected = next(i.text for i in items if i.slug == f.slug)
            console.print(f"  [dim]expect[/] {expected}")
            console.print(f"  [cyan]got   [/] {f.text}\n")


def main() -> int:
    from voiceagent.stt.mlx_whisper_engine import MLXWhisperEngine
    from voiceagent.stt.moonshine_engine import MoonshineEngine

    console.print("[bold]Phase 1 -- STT benchmark[/]\n")
    items = fx.generate()
    console.print(f"[dim]{len(items)} fixtures in {fx.FIXTURE_DIR}[/]\n")

    results = [
        bench_engine(MoonshineEngine(), items),
        bench_engine(MLXWhisperEngine(), items),
    ]
    console.print()
    report(results, items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
