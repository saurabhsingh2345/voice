"""Phase 2 benchmark: where does the LLM's latency actually go?

Run with::

    uv run python -m voiceagent.llm.benchmark

Time-to-first-token is the number that matters for a voice loop -- it sets when
TTS can start speaking. TTFT is dominated by prompt processing, so this measures
how much the tool schemas and the growing conversation history cost us.
"""

from __future__ import annotations

import asyncio
import statistics
import time

from rich.console import Console
from rich.table import Table

from voiceagent.llm.base import Message
from voiceagent.llm.mlx_engine import MLXLLMEngine
from voiceagent.tools.dummy import DummyWeatherTool

console = Console()
GIB = 1024**3
REPEATS = 3

SYSTEM = (
    "You are a local voice assistant. Keep replies to one short sentence. "
    "No emoji or markdown."
)


async def measure(engine: MLXLLMEngine, messages: list[Message], tools, max_tokens: int = 60):
    """Return (ttft_ms, total_ms, generated_chars)."""
    started = time.perf_counter()
    ttft = None
    chars = 0
    async for chunk in engine.stream(messages, tools=tools, max_tokens=max_tokens):
        if chunk.time_to_first_token_ms is not None and ttft is None:
            ttft = chunk.time_to_first_token_ms
        chars += len(chunk.text)
    return ttft or 0.0, (time.perf_counter() - started) * 1000, chars


async def main_async() -> None:
    engine = MLXLLMEngine()
    console.print(f"[dim]loading {engine.repo}...[/]")
    started = time.perf_counter()
    engine.load()
    load_s = time.perf_counter() - started
    console.print(f"[dim]loaded in {load_s:.1f}s, {engine.resident_bytes / GIB:.2f} GiB peak[/]\n")

    tool_specs = [DummyWeatherTool().to_openai_spec()]
    base = [Message(role="system", content=SYSTEM)]

    # A conversation long enough to show how history inflates prompt processing.
    long_history = list(base)
    for i in range(6):
        long_history.append(Message(role="user", content=f"Question number {i} about the weather."))
        long_history.append(
            Message(role="assistant", content=f"That is answer number {i}, kept fairly short.")
        )

    scenarios = [
        ("no tools, short history", base + [Message(role="user", content="Say hello.")], None),
        ("with tools, short history", base + [Message(role="user", content="Say hello.")], tool_specs),
        ("no tools, 12-msg history", long_history + [Message(role="user", content="Say hello.")], None),
        (
            "with tools, 12-msg history",
            long_history + [Message(role="user", content="Say hello.")],
            tool_specs,
        ),
    ]

    table = Table(title="Time to first token", title_justify="left", header_style="bold")
    table.add_column("Scenario")
    table.add_column("Prompt tokens", justify="right")
    table.add_column("TTFT (median)", justify="right")
    table.add_column("Turn total", justify="right")

    for label, messages, tools in scenarios:
        prompt_text = engine._render(messages, tools)
        n_tokens = len(engine._tokenizer.encode(prompt_text))

        ttfts, totals = [], []
        for _ in range(REPEATS):
            # Cold: no shared prefix carried over, so this is worst-case TTFT.
            engine.reset_cache()
            ttft, total, _ = await measure(engine, messages, tools)
            ttfts.append(ttft)
            totals.append(total)

        table.add_row(
            label,
            str(n_tokens),
            f"{statistics.median(ttfts):.0f} ms",
            f"{statistics.median(totals):.0f} ms",
        )
        console.print(f"[dim]  {label}: {n_tokens} prompt tokens[/]")

    console.print()
    console.print(table)

    await _cache_comparison(engine, tool_specs)

    console.print(
        f"\n[bold]Load:[/] {load_s:.1f}s   "
        f"[bold]Peak memory:[/] {engine.resident_bytes / GIB:.2f} GiB"
    )
    engine.unload()


async def _cache_comparison(engine: MLXLLMEngine, tool_specs) -> None:
    """Same 4-turn conversation, with and without prefix cache reuse.

    This is the realistic case: in a voice loop the system prompt and tool
    schemas never change, so their KV should only ever be computed once.
    """
    turns = [
        "Say hello.",
        "What is the weather in Paris?",
        "And what about Rome?",
        "Thanks, that is all.",
    ]

    table = Table(
        title="Prefix cache: TTFT per turn of a 4-turn conversation",
        title_justify="left",
        header_style="bold",
    )
    table.add_column("Turn")
    table.add_column("Cache disabled", justify="right")
    table.add_column("Cache enabled", justify="right")
    table.add_column("Saved", justify="right")

    results: dict[bool, list[float]] = {}
    for cached in (False, True):
        engine.reset_cache()
        history = [Message(role="system", content=SYSTEM)]
        ttfts = []
        for turn in turns:
            history.append(Message(role="user", content=turn))
            if not cached:
                engine.reset_cache()
            ttft, _, _ = await measure(engine, history, tool_specs, max_tokens=40)
            ttfts.append(ttft)
            history.append(Message(role="assistant", content="Acknowledged."))
        results[cached] = ttfts

    for i, turn in enumerate(turns):
        cold, warm = results[False][i], results[True][i]
        table.add_row(
            f"{i + 1}. {turn[:28]}",
            f"{cold:.0f} ms",
            f"{warm:.0f} ms",
            f"[green]-{cold - warm:.0f} ms[/]" if warm < cold else f"{warm - cold:+.0f} ms",
        )

    console.print()
    console.print(table)


def main() -> int:
    asyncio.run(main_async())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
