"""Phase 3 demo: speak text, and measure when the first audio actually starts.

    uv run python -m voiceagent.tts.demo                        # batch vs streamed
    uv run python -m voiceagent.tts.demo --text "Hello there."  # speak one line
    uv run python -m voiceagent.tts.demo --no-play              # measure only
    uv run python -m voiceagent.tts.demo --voices               # list voices

The number that matters is time-to-first-audio: how long the user waits in
silence. Synthesizing a whole reply before speaking pays for every sentence;
sentence-chunked streaming pays only for the first.
"""

from __future__ import annotations

import argparse
import asyncio
import queue
import threading
import time

import numpy as np
from rich.console import Console

from voiceagent.tts.kokoro_engine import SAMPLE_RATE, KokoroEngine

console = Console()

DEMO_TEXT = (
    "Sure, I can help with that. The quarterly numbers are ready for review. "
    "Would you like me to summarize them, or open the file directly?"
)


class Speaker:
    """Plays audio chunks as they arrive, on a background thread."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        import sounddevice as sd

        def run() -> None:
            with sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype="float32") as out:
                while True:
                    item = self._queue.get()
                    if item is None:
                        break
                    out.write(item.reshape(-1, 1))

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def play(self, samples: np.ndarray) -> None:
        if samples.size:
            self._queue.put(samples)

    def close(self) -> None:
        self._queue.put(None)
        if self._thread:
            self._thread.join()


async def _token_stream(text: str, tokens_per_second: float = 40.0):
    """Approximate an LLM writing at a realistic rate."""
    delay = 1.0 / tokens_per_second
    for word in text.split(" "):
        yield word + " "
        await asyncio.sleep(delay)


async def run_batch(engine: KokoroEngine, text: str, play: bool) -> float:
    speaker = Speaker()
    if play:
        speaker.start()

    started = time.perf_counter()
    first_audio = None
    total_samples = 0
    async for chunk in engine.synthesize(text):
        if first_audio is None:
            first_audio = (time.perf_counter() - started) * 1000
        total_samples += chunk.samples.size
        if play:
            speaker.play(chunk.samples)

    if play:
        speaker.close()
    console.print(
        f"  [bold]batch[/]     first audio [cyan]{first_audio:.0f} ms[/], "
        f"{total_samples / SAMPLE_RATE:.2f}s of speech"
    )
    return first_audio or 0.0


async def run_streamed(engine: KokoroEngine, text: str, play: bool) -> float:
    speaker = Speaker()
    if play:
        speaker.start()

    started = time.perf_counter()
    first_audio = None
    total_samples = 0
    async for chunk in engine.synthesize_stream(_token_stream(text)):
        if first_audio is None and chunk.samples.size:
            first_audio = (time.perf_counter() - started) * 1000
        total_samples += chunk.samples.size
        if play:
            speaker.play(chunk.samples)

    if play:
        speaker.close()
    console.print(
        f"  [bold]streamed[/]  first audio [green]{first_audio:.0f} ms[/], "
        f"{total_samples / SAMPLE_RATE:.2f}s of speech"
    )
    return first_audio or 0.0


async def main_async(args) -> None:
    engine = KokoroEngine(voice=args.voice)
    console.print(f"[dim]loading {engine.repo}...[/]")
    started = time.perf_counter()
    engine.load()
    console.print(
        f"[dim]loaded in {time.perf_counter() - started:.1f}s "
        f"({engine.resident_bytes / 1024**3:.2f} GiB peak)[/]\n"
    )

    text = args.text or DEMO_TEXT
    gaps = engine.check_coverage(text)
    if gaps:
        console.print(f"[yellow]words with no pronunciation (will be skipped): {gaps}[/]")

    console.print(f"[dim]{text}[/]\n")

    if args.text:
        await run_streamed(engine, text, play=not args.no_play)
    else:
        console.print("Same text, synthesized two ways:")
        batch = await run_batch(engine, text, play=not args.no_play)
        streamed = await run_streamed(engine, text, play=not args.no_play)
        if batch and streamed:
            console.print(
                f"\n[bold green]Streaming starts speaking {batch - streamed:.0f} ms "
                f"sooner ({(1 - streamed / batch) * 100:.0f}% less silence).[/]"
            )

    engine.unload()


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 Kokoro TTS demo.")
    parser.add_argument("--text", help="speak this text instead of the demo paragraph")
    parser.add_argument("--voice", default="af_heart", help="Kokoro voice id")
    parser.add_argument("--no-play", action="store_true", help="measure without audio output")
    parser.add_argument("--voices", action="store_true", help="list bundled voices and exit")
    args = parser.parse_args()

    if args.voices:
        from huggingface_hub import list_repo_files

        files = list_repo_files("mlx-community/Kokoro-82M-bf16")
        voices = sorted(
            f.split("/")[-1].removesuffix(".safetensors")
            for f in files
            if "voices/" in f
        )
        console.print(f"{len(voices)} voices: {', '.join(voices)}")
        return 0

    asyncio.run(main_async(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
