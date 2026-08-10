"""Live streaming transcription from the microphone.

Run with::

    uv run python -m voiceagent.stt.live                  # Moonshine (default)
    uv run python -m voiceagent.stt.live --engine mlx-whisper

Interim hypotheses overwrite the current line; finalized lines are committed
above it with the engine-reported latency. Ctrl-C to stop.

macOS will prompt for microphone access the first time. If no prompt appears,
grant it under System Settings > Privacy & Security > Microphone for whichever
app hosts the terminal.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import AsyncIterator

import numpy as np
import sounddevice as sd
from rich.console import Console

from voiceagent.stt.base import SAMPLE_RATE, STTEngine

console = Console()

#: 80 ms of audio per callback -- small enough to keep latency low, large
#: enough that we are not thrashing the event loop.
BLOCK_SAMPLES = int(0.08 * SAMPLE_RATE)


async def mic_chunks(stop: asyncio.Event) -> AsyncIterator[np.ndarray]:
    """Yield float32 mono blocks from the default input device."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[np.ndarray] = asyncio.Queue()

    def callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            console.print(f"[yellow]audio status: {status}[/]")
        # Copy: sounddevice reuses the buffer after this returns.
        loop.call_soon_threadsafe(queue.put_nowait, indata[:, 0].copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=BLOCK_SAMPLES,
        callback=callback,
    ):
        console.print("[green]Listening.[/] Speak now, Ctrl-C to stop.\n")
        while not stop.is_set():
            try:
                yield await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue


def build_engine(name: str) -> STTEngine:
    if name == "moonshine":
        from voiceagent.stt.moonshine_engine import MoonshineEngine

        return MoonshineEngine()
    if name == "mlx-whisper":
        from voiceagent.stt.mlx_whisper_engine import MLXWhisperEngine

        return MLXWhisperEngine()
    raise SystemExit(f"unknown engine {name!r}")


async def run(engine_name: str) -> None:
    engine = build_engine(engine_name)

    console.print(f"[dim]loading {engine.name}...[/]")
    started = time.perf_counter()
    engine.load()
    console.print(
        f"[dim]loaded in {time.perf_counter() - started:.1f}s "
        f"({engine.resident_bytes / 1024**2:.0f} MiB)[/]"
    )

    stop = asyncio.Event()
    finals = 0
    try:
        async for transcript in engine.stream(mic_chunks(stop)):
            if not transcript.text:
                continue
            if transcript.is_final:
                finals += 1
                latency = (
                    f"[dim]({transcript.latency_ms:.0f} ms)[/]"
                    if transcript.latency_ms
                    else ""
                )
                # \r clears the interim line before committing the final.
                sys.stdout.write("\r" + " " * 100 + "\r")
                console.print(f"[bold green]>[/] {transcript.text} {latency}")
            else:
                sys.stdout.write(f"\r\033[2m... {transcript.text[:90]}\033[0m")
                sys.stdout.flush()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        stop.set()
        engine.unload()
        console.print(f"\n[dim]stopped after {finals} final transcript(s).[/]")


def main() -> int:
    parser = argparse.ArgumentParser(description="Live microphone transcription.")
    parser.add_argument(
        "--engine",
        default="moonshine",
        choices=["moonshine", "mlx-whisper"],
        help="which STT backend to stream through",
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="print audio devices and exit"
    )
    args = parser.parse_args()

    if args.list_devices:
        console.print(str(sd.query_devices()))
        return 0

    try:
        asyncio.run(run(args.engine))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
