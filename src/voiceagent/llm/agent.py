"""The tool-calling loop.

Drives an LLMEngine: stream a response, run any tools it calls, feed the results
back, and repeat until the model answers without calling a tool.

Run the Phase 2 console demo with::

    uv run python -m voiceagent.llm.agent
    uv run python -m voiceagent.llm.agent --prompt "what's the weather in Paris?"
"""

from __future__ import annotations

import argparse
import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

from rich.console import Console

from voiceagent.llm.base import LLMEngine, Message, ToolCall
from voiceagent.tools.base import Tool, ToolResult

console = Console()

SYSTEM_PROMPT = (
    "You are a local voice assistant. Your replies are spoken aloud by a "
    "speech synthesizer, so: keep them to one or two short sentences, write "
    "plain prose only, and never use emoji, markdown, bullet points, or "
    "special characters -- they get read out literally or mangled. Spell out "
    "symbols as words. Use the supplied tools when they are relevant instead "
    "of guessing."
)

#: Cap on tool round-trips per turn, so a confused model cannot loop forever.
MAX_ITERATIONS = 5


@dataclass
class AgentEvent:
    kind: Literal["text", "tool_call", "tool_result", "done"]
    text: str = ""
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    time_to_first_token_ms: float | None = None


class Agent:
    def __init__(
        self,
        engine: LLMEngine,
        tools: list[Tool] | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.engine = engine
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.history: list[Message] = [Message(role="system", content=system_prompt)]

    @property
    def tool_specs(self) -> list[dict[str, Any]] | None:
        if not self.tools:
            return None
        return [tool.to_openai_spec() for tool in self.tools.values()]

    async def _invoke(self, call: ToolCall) -> ToolResult:
        tool = self.tools.get(call.name)
        if tool is None:
            return ToolResult(content="", ok=False, error=f"unknown tool {call.name!r}")
        try:
            return await tool.run(**call.arguments)
        except TypeError as exc:
            # Wrong/missing arguments -- report back so the model can retry.
            return ToolResult(content="", ok=False, error=f"bad arguments: {exc}")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(content="", ok=False, error=str(exc))

    async def turn(self, user_text: str, max_tokens: int = 512) -> AsyncIterator[AgentEvent]:
        """Run one user turn to completion, yielding events as they happen."""
        self.history.append(Message(role="user", content=user_text))

        for _ in range(MAX_ITERATIONS):
            spoken = ""
            calls: list[ToolCall] = []

            async for chunk in self.engine.stream(
                self.history, tools=self.tool_specs, max_tokens=max_tokens
            ):
                if chunk.text:
                    spoken += chunk.text
                    yield AgentEvent(
                        kind="text",
                        text=chunk.text,
                        time_to_first_token_ms=chunk.time_to_first_token_ms,
                    )
                elif chunk.time_to_first_token_ms is not None:
                    # First token was withheld by the tool-call parser; still
                    # report when it arrived so TTFT stays honest.
                    yield AgentEvent(
                        kind="text",
                        time_to_first_token_ms=chunk.time_to_first_token_ms,
                    )
                if chunk.tool_call:
                    calls.append(chunk.tool_call)

            self.history.append(
                Message(role="assistant", content=spoken, tool_calls=calls)
            )

            if not calls:
                yield AgentEvent(kind="done")
                return

            for call in calls:
                yield AgentEvent(kind="tool_call", tool_call=call)
                result = await self._invoke(call)
                yield AgentEvent(kind="tool_result", tool_call=call, tool_result=result)
                self.history.append(
                    Message(
                        role="tool",
                        content=result.content if result.ok else f"ERROR: {result.error}",
                        tool_call_id=call.name,
                    )
                )

        yield AgentEvent(kind="done")


# --- console demo ---------------------------------------------------------


async def _run_turn(agent: Agent, prompt: str) -> None:
    console.print(f"\n[bold cyan]you[/] {prompt}")
    console.print("[bold green]agent[/] ", end="")

    ttft: float | None = None
    started = time.perf_counter()
    printed_any = False

    async for event in agent.turn(prompt):
        if event.time_to_first_token_ms is not None and ttft is None:
            ttft = event.time_to_first_token_ms
        if event.kind == "text" and event.text:
            console.print(event.text, end="")
            printed_any = True
        elif event.kind == "tool_call":
            prefix = "\n" if printed_any else ""
            console.print(
                f"{prefix}[yellow]-> calling {event.tool_call.name}"
                f"({event.tool_call.arguments})[/]"
            )
        elif event.kind == "tool_result":
            result = event.tool_result
            body = result.content if result.ok else f"ERROR: {result.error}"
            console.print(f"[yellow]<- {body}[/]")
            console.print("[bold green]agent[/] ", end="")

    total_ms = (time.perf_counter() - started) * 1000
    console.print(
        f"\n[dim]time to first token: {ttft:.0f} ms | turn total: {total_ms:.0f} ms[/]"
        if ttft
        else f"\n[dim]turn total: {total_ms:.0f} ms[/]"
    )


async def main_async(prompts: list[str], interactive: bool) -> None:
    from voiceagent.llm.mlx_engine import MLXLLMEngine
    from voiceagent.tools.dummy import DummyWeatherTool

    engine = MLXLLMEngine()
    console.print(f"[dim]loading {engine.repo}...[/]")
    started = time.perf_counter()
    engine.load()
    console.print(
        f"[dim]loaded in {time.perf_counter() - started:.1f}s "
        f"({engine.resident_bytes / 1024**3:.2f} GiB peak)[/]"
    )

    weather = DummyWeatherTool()
    agent = Agent(engine, tools=[weather])

    for prompt in prompts:
        await _run_turn(agent, prompt)

    if interactive:
        console.print("\n[dim]Type a message, or Ctrl-C to quit.[/]")
        while True:
            try:
                prompt = await asyncio.to_thread(input, "\nyou> ")
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.strip():
                await _run_turn(agent, prompt)

    console.print(f"\n[dim]dummy tool was invoked {len(weather.calls)} time(s): {weather.calls}[/]")
    engine.unload()


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 tool-calling demo.")
    parser.add_argument("--prompt", action="append", help="run this prompt (repeatable)")
    parser.add_argument("--chat", action="store_true", help="drop into an interactive loop")
    args = parser.parse_args()

    prompts = args.prompt or [
        "Hello! Say hi back in one short sentence.",
        "What's the weather in Paris right now?",
    ]
    try:
        asyncio.run(main_async(prompts, interactive=args.chat))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
