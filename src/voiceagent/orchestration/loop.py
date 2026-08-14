"""The full voice loop: VAD -> STT -> LLM -> TTS, with barge-in.

    uv run voice-chat                  # talk to it
    uv run voice-chat --replay f.wav   # drive it from a file, no mic needed

Deviation from the brief, stated plainly: this does not use Pipecat. Pipecat has
no MLX LLM service (only Ollama, which is llama.cpp), and its local audio
transport needs pyaudio, which needs a Homebrew portaudio that would then have
to be bundled for the desktop build. Using it would have meant writing custom
services for all three of our engines plus a sounddevice transport -- more
adapter code than the loop itself. The engines already sit behind interfaces, so
swapping this orchestrator for Pipecat later touches only this file.

Latency is reported per stage against the moment the user stopped talking, since
that is when they start waiting.
"""

from __future__ import annotations

import argparse
import asyncio
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from rich.console import Console

from voiceagent.llm.agent import Agent
from voiceagent.orchestration.audio_io import Microphone, Speaker
from voiceagent.orchestration.vad import SileroVAD, VADConfig, VADEvent
from voiceagent.stt.base import SAMPLE_RATE as STT_SR

console = Console()


@dataclass
class TurnTimings:
    """Milliseconds from end-of-speech to each downstream milestone."""

    speech_end: float = 0.0
    stt_final: float | None = None
    llm_first_token: float | None = None
    tts_first_audio: float | None = None
    turn_complete: float | None = None
    barged_in: bool = False

    def mark(self, field_name: str) -> None:
        setattr(self, field_name, (time.perf_counter() - self.speech_end) * 1000)

    def render(self) -> str:
        parts = []
        for label, value in (
            ("STT", self.stt_final),
            ("LLM 1st tok", self.llm_first_token),
            ("TTS 1st audio", self.tts_first_audio),
            ("done", self.turn_complete),
        ):
            parts.append(f"{label} {value:.0f}ms" if value is not None else f"{label} --")
        return "  ".join(parts)


@dataclass
class LoopConfig:
    #: While the agent is speaking, the mic hears the agent. Without acoustic
    #: echo cancellation the only defence is a stricter bar for what counts as
    #: a real interruption. Headphones remove the problem entirely.
    barge_in_threshold: float = 0.85
    #: Ignore the mic for this long after playback starts, to skip the attack
    #: transient of the agent's own first syllable.
    barge_in_grace_ms: float = 350.0
    allow_barge_in: bool = True
    max_tokens: int = 220
    auto_approve_tools: bool = False
    """Skip the confirmation prompt. For scripted runs only -- never a default."""


class VoiceLoop:
    def __init__(self, config: LoopConfig | None = None) -> None:
        self.config = config or LoopConfig()
        self.vad = SileroVAD(VADConfig())
        self.stt = None
        self.agent: Agent | None = None
        self.tts = None
        self.speaker = Speaker()

        self._utterance: list[np.ndarray] = []
        self._collecting = False
        self._turn_task: asyncio.Task | None = None
        self._playback_started_at = 0.0
        self._interrupt = threading.Event()
        self.sandbox = None
        self.store = None

    # --- setup ------------------------------------------------------------

    def load(self) -> None:
        from voiceagent.llm.mlx_engine import MLXLLMEngine
        from voiceagent.stt.moonshine_engine import MoonshineEngine
        from voiceagent.tools.files import ListFilesTool, ReadFileTool, Sandbox, WriteFileTool
        from voiceagent.storage.db import EncryptedStore
        from voiceagent.tools.http import HttpRequestTool
        from voiceagent.tools.memory import ForgetAllTool, RecallTool, RememberTool
        from voiceagent.tools.registry import ToolRegistry
        from voiceagent.tools.shell import ShellTool
        from voiceagent.tts.router import build_default_router

        started = time.perf_counter()
        console.print("[dim]loading VAD...[/]")
        self.vad.load()

        console.print("[dim]loading STT (moonshine)...[/]")
        self.stt = MoonshineEngine()
        self.stt.load()

        console.print("[dim]loading LLM (qwen3-4b)...[/]")
        llm = MLXLLMEngine()
        llm.load()

        sandbox = Sandbox()
        sandbox.ensure_root()
        self.store = EncryptedStore()
        registry = ToolRegistry(
            [
                ListFilesTool(sandbox),
                ReadFileTool(sandbox),
                WriteFileTool(sandbox),
                ShellTool(sandbox=sandbox),
                HttpRequestTool(),
                RememberTool(self.store),
                RecallTool(self.store),
                ForgetAllTool(self.store),
            ]
        )
        self.agent = Agent(
            llm, registry=registry, confirm=self._confirm, store=self.store
        )
        self.sandbox = sandbox
        console.print(f"[dim]  workspace: {sandbox.root}[/]")
        console.print(f"[dim]  tools: {', '.join(registry.names)}[/]")

        # The router, not KokoroEngine directly. That is the whole of what kept
        # this loop English-only: Devanagari sent to Kokoro produces nothing
        # usable, so a Hindi reply had to be typed and listened to elsewhere.
        # Hindi synthesis is now RTF 0.63, so it can keep up with playback.
        #
        # Nothing loads here. The router builds engines on first use and holds
        # one at a time -- Kokoro and Chatterbox Multilingual do not fit
        # alongside the LLM together. The cost is a reload when the reply
        # language changes; the alternative is not fitting.
        console.print("[dim]TTS: router (kokoro / chatterbox-multilingual, on demand)[/]")
        self.tts = build_default_router()

        # NOT priming the prefix cache here, though Agent.prime() exists.
        # Two reasons, both measured: it did not move turn-one latency
        # (1382 ms with vs 1457 ms without -- noise), and driving an async
        # generator through a throwaway asyncio.run() inside load() hangs on
        # executor shutdown. Turn one pays ~1.4s; every turn after is ~0.6s.

        total = time.perf_counter() - started
        # The router reports nothing until it has built an engine, which is
        # correct: quoting a TTS figure before one is loaded would be a guess.
        resident = (
            self.stt.resident_bytes + self.agent.engine.resident_bytes
        ) / 1024**3
        console.print(
            f"[dim]ready in {total:.1f}s  (models ~{resident:.2f} GiB resident)[/]\n"
        )

    def unload(self) -> None:
        for component in (self.stt, self.tts):
            if component is not None:
                component.unload()
        if self.agent is not None:
            self.agent.engine.unload()
        if self.store is not None:
            self.store.close()

    # --- the loop ---------------------------------------------------------

    async def run(self, replay: str | None = None) -> None:
        self.speaker.start()
        stop = threading.Event()

        source = self._replay_blocks(replay) if replay else Microphone().blocks(stop)
        console.print(
            "[green]Listening.[/] Speak, and interrupt any time. Ctrl-C to quit.\n"
            if not replay
            else f"[green]Replaying[/] {replay}\n"
        )

        try:
            async for block in source:
                for event, audio in self.vad.process(block):
                    await self._handle(event, audio)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            stop.set()
            if self._turn_task and not self._turn_task.done():
                self._turn_task.cancel()
            self.speaker.close()

    async def _replay_blocks(self, path: str):
        import soundfile as sf

        audio, sr = sf.read(path, dtype="float32")
        if sr != STT_SR:
            raise SystemExit(f"{path} is {sr} Hz, expected {STT_SR} Hz")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        # Lead-in and trailing silence so the VAD sees real boundaries.
        pad = np.zeros(int(0.7 * STT_SR), dtype=np.float32)
        audio = np.concatenate([pad, audio, pad, pad])

        block = 512
        for start in range(0, len(audio), block):
            yield audio[start : start + block]
            await asyncio.sleep(block / STT_SR)
        # Let the final turn finish before the generator ends.
        if self._turn_task:
            await asyncio.gather(self._turn_task, return_exceptions=True)

    async def _handle(self, event: VADEvent, audio: np.ndarray) -> None:
        if event is VADEvent.SPEECH_START:
            if self.speaker.is_playing and self._within_grace():
                return  # too soon after our own audio started; likely echo
            if self.speaker.is_playing or (self._turn_task and not self._turn_task.done()):
                if not self.config.allow_barge_in:
                    return
                self._barge_in()
            self._collecting = True
            self._utterance = [audio]

        elif event is VADEvent.NONE:
            if self._collecting:
                self._utterance.append(audio)

        elif event is VADEvent.SPEECH_END and self._collecting:
            self._collecting = False
            utterance = np.concatenate(self._utterance) if self._utterance else None
            self._utterance = []
            if utterance is None or len(utterance) < STT_SR * 0.25:
                return  # too short to be speech
            timings = TurnTimings(speech_end=time.perf_counter())
            self._interrupt.clear()
            self._turn_task = asyncio.create_task(self._run_turn(utterance, timings))

    async def _confirm(self, tool, arguments: dict) -> bool:
        """Ask before running anything that writes, executes, or leaves the machine.

        Typed rather than spoken on purpose: "yes" is exactly the kind of word a
        speech recogniser mishears, and this is the gate protecting the
        filesystem and the network.
        """
        if self.config.auto_approve_tools:
            console.print(f"[yellow]  auto-approved {tool.name}({arguments})[/]")
            return True

        console.print(
            f"\n[bold yellow]  Approve {tool.name}?[/] [dim]{arguments}[/]\n"
            f"  [dim]type 'y' then Enter to allow:[/] ",
            end="",
        )
        answer = await asyncio.to_thread(input)
        approved = answer.strip().lower() in ("y", "yes")
        console.print("[green]  approved[/]" if approved else "[red]  declined[/]")
        return approved

    def _within_grace(self) -> bool:
        elapsed = (time.perf_counter() - self._playback_started_at) * 1000
        return elapsed < self.config.barge_in_grace_ms

    def _barge_in(self) -> None:
        console.print("[yellow]  (interrupted)[/]")
        self._interrupt.set()
        if self.tts is not None:
            self.tts.cancel()
        self.speaker.flush()
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()

    # --- one turn ---------------------------------------------------------

    async def _run_turn(self, utterance: np.ndarray, timings: TurnTimings) -> None:
        try:
            transcript = await asyncio.to_thread(self.stt.transcribe, utterance)
            timings.mark("stt_final")
            text = transcript.text.strip()
            if not text:
                return

            console.print(f"[bold cyan]you[/]   {text}")
            console.print("[bold green]agent[/] ", end="")

            spoken = ""
            queue: asyncio.Queue[str | None] = asyncio.Queue()

            async def tokens():
                while True:
                    item = await queue.get()
                    if item is None:
                        return
                    yield item

            async def pump() -> None:
                nonlocal spoken
                async for event in self.agent.turn(text, max_tokens=self.config.max_tokens):
                    if self._interrupt.is_set():
                        break
                    if event.time_to_first_token_ms is not None and timings.llm_first_token is None:
                        timings.mark("llm_first_token")
                    if event.kind == "text" and event.text:
                        spoken += event.text
                        console.print(event.text, end="")
                        await queue.put(event.text)
                    elif event.kind == "tool_call":
                        console.print(f"\n[yellow]  -> {event.tool_call.name}({event.tool_call.arguments})[/]")
                await queue.put(None)

            pump_task = asyncio.create_task(pump())

            async for chunk in self.tts.synthesize_stream(tokens()):
                if self._interrupt.is_set():
                    break
                if chunk.samples.size:
                    if timings.tts_first_audio is None:
                        timings.mark("tts_first_audio")
                        self._playback_started_at = time.perf_counter()
                    self.speaker.play(chunk.samples)

            await pump_task
            timings.mark("turn_complete")
            console.print(f"\n[dim]{timings.render()}[/]\n")

        except asyncio.CancelledError:
            timings.barged_in = True
            raise
        except Exception as exc:  # noqa: BLE001
            console.print(f"\n[red]turn failed:[/] {exc}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Full local voice loop.")
    parser.add_argument("--replay", help="drive from a 16 kHz WAV instead of the mic")
    parser.add_argument("--no-barge-in", action="store_true", help="disable interruption")
    parser.add_argument(
        "--auto-approve-tools",
        action="store_true",
        help="run write/shell/network tools without asking (scripted runs only)",
    )
    args = parser.parse_args()

    loop = VoiceLoop(
        LoopConfig(
            allow_barge_in=not args.no_barge_in,
            auto_approve_tools=args.auto_approve_tools,
        )
    )
    loop.load()
    try:
        asyncio.run(loop.run(replay=args.replay))
    except KeyboardInterrupt:
        pass
    finally:
        loop.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
