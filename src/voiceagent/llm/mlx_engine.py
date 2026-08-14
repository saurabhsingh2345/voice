"""Qwen3-4B-Instruct on MLX via mlx-lm.

Qwen3 emits tool calls inline as ``<tool_call>{json}</tool_call>`` rather than
through a separate channel, so the token stream has to be parsed as it arrives:
text before a tool call must reach the user (and, later, the TTS) immediately,
while the call itself is withheld until it can be parsed as JSON.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from voiceagent.llm.base import Chunk, LLMEngine, Message, ToolCall

DEFAULT_REPO = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"

#: Qwen3-2507 Instruct's own recommended sampling, which this engine was not using.
#:
#: It sampled on temperature alone --- no truncation at all --- and that is the
#: setup a degenerate cycle survives in: once the model is looping,
#: `ऊपर से ऊपर ऊपर से ऊपर …`, the loop's own tokens dominate the distribution and
#: an untruncated tail keeps handing back the ones that continue it. Truncating to
#: the top 20 / 0.8 mass does not by itself break a cycle whose tokens *are* the
#: top of the distribution, which is why `_find_repetition_cycle` exists as well;
#: it does remove the long tail of unlikely tokens that make the model wander into
#: one. Qwen publishes these per variant, and 0.7 here already matched the
#: Instruct (non-thinking) row --- top_p and top_k were simply missing.
TOP_P = 0.8
TOP_K = 20

#: Qwen's documented lever for endless repetition is `presence_penalty` (0-2),
#: with an explicit warning that higher values cause **language mixing**. Left at
#: 0.0 by default, because language mixing is not a cosmetic risk in this project:
#: the reply's language selects the TTS engine (see the router), and neither
#: engine degrades gracefully into the other's language. A reply that drifts from
#: Hindi to English mid-sentence is routed on its first characters and then
#: mispronounced for the rest. Available, off, and documented rather than tuned
#: blind. Note Qwen also says to leave `repetition_penalty` disabled, so this
#: engine does not set one.
PRESENCE_PENALTY = 0.0

#: Longest cycle `_find_repetition_cycle` will look for, in tokens.
MAX_CYCLE_TOKENS = 12

#: How many times a cycle must repeat before generation is abandoned.
#:
#: A single token repeated needs a longer run than a phrase does: "!!!!" and a
#: row of newlines are ordinary, whereas the same four-token phrase four times in
#: a row is not language. Both thresholds are on *token* identity, not text, so
#: they do not fire on a legitimately repeated word in different contexts.
MIN_CYCLE_REPEATS = 4
MIN_SINGLE_TOKEN_REPEATS = 8


def _find_repetition_cycle(tokens: list[int]) -> int | None:
    """Length of the cycle the tail of `tokens` is stuck in, or None.

    Qwen3 degenerates into a repetition loop roughly once in 80 generations,
    observed as `ऊपर से ऊपर ऊपर से ऊपर …` running until the token budget is
    exhausted. It is not attributable to any prompt, no quality metric sees it,
    and in front of a listener it is the worst failure the system has: 512 tokens
    of the same phrase, read aloud.

    Sampling parameters lower the odds and cannot bound them, so this is the part
    that actually stops it. Checked after every token; comparing at most
    MAX_CYCLE_TOKENS x MIN_SINGLE_TOKEN_REPEATS integers, so it costs nothing
    against a forward pass.

    Deliberately exact-match on token ids rather than a similarity measure. A
    fuzzy detector would eventually truncate a legitimate answer, and cutting off
    a correct reply is a worse failure than the one being fixed --- it is silent,
    where the loop is obvious.
    """
    for period in range(1, MAX_CYCLE_TOKENS + 1):
        repeats = MIN_SINGLE_TOKEN_REPEATS if period == 1 else MIN_CYCLE_REPEATS
        span = period * repeats
        if len(tokens) < span:
            continue
        tail = tokens[-span:]
        cycle = tail[:period]
        if all(tail[i : i + period] == cycle for i in range(0, span, period)):
            return period
    return None


class _ToolCallParser:
    """Splits a raw token stream into plain text and completed tool calls.

    Text is emitted as soon as it cannot be part of a ``<tool_call>`` opening
    tag; anything that might still turn into one is held back. This keeps
    latency low without ever leaking a partial tag downstream.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._in_call = False

    @staticmethod
    def _longest_partial_suffix(text: str, marker: str) -> int:
        """Length of the longest suffix of `text` that prefixes `marker`."""
        for size in range(min(len(marker) - 1, len(text)), 0, -1):
            if text[-size:] == marker[:size]:
                return size
        return 0

    def feed(self, text: str) -> tuple[str, list[ToolCall]]:
        """Consume new tokens, returning (safe_text, completed_calls)."""
        self._buffer += text
        safe = ""
        calls: list[ToolCall] = []

        while True:
            if self._in_call:
                end = self._buffer.find(TOOL_CLOSE)
                if end == -1:
                    break
                payload = self._buffer[:end]
                self._buffer = self._buffer[end + len(TOOL_CLOSE) :]
                self._in_call = False
                call = self._parse(payload)
                if call is not None:
                    calls.append(call)
                continue

            start = self._buffer.find(TOOL_OPEN)
            if start != -1:
                safe += self._buffer[:start]
                self._buffer = self._buffer[start + len(TOOL_OPEN) :]
                self._in_call = True
                continue

            # No complete tag. Hold back only what could still become one.
            hold = self._longest_partial_suffix(self._buffer, TOOL_OPEN)
            if hold:
                safe += self._buffer[:-hold]
                self._buffer = self._buffer[-hold:]
            else:
                safe += self._buffer
                self._buffer = ""
            break

        return safe, calls

    def flush(self) -> str:
        """Return any text still held back at end of generation."""
        if self._in_call:
            # Generation stopped mid-call; the fragment is not usable text.
            self._buffer = ""
            return ""
        text, self._buffer = self._buffer, ""
        return text

    @staticmethod
    def _parse(payload: str) -> ToolCall | None:
        try:
            data = json.loads(payload.strip())
        except json.JSONDecodeError:
            return None
        name = data.get("name")
        if not name:
            return None
        arguments = data.get("arguments", {})
        if isinstance(arguments, str):
            # Some templates double-encode the arguments object.
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        return ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=name, arguments=arguments)


class MLXLLMEngine(LLMEngine):
    name = "qwen3-4b-mlx"

    def __init__(
        self,
        repo: str = DEFAULT_REPO,
        temperature: float = 0.7,
        system_prompt: str | None = None,
        top_p: float = TOP_P,
        top_k: int = TOP_K,
        presence_penalty: float = PRESENCE_PENALTY,
    ) -> None:
        self.repo = repo
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        #: How many generations were cut short by `_find_repetition_cycle`.
        #: Counted rather than logged-and-forgotten: the baseline is ~1 in 80, so
        #: a rate far above that means the sampling change made things worse, and
        #: there is no other signal that would show it.
        self.degenerations = 0
        self._model = None
        self._tokenizer = None
        self._peak_bytes = 0
        self._cache = None
        self._cached_tokens: list[int] = []

    # --- lifecycle --------------------------------------------------------

    def load(self, warmup: bool = True) -> None:
        import mlx.core as mx
        from mlx_lm import load

        mx.reset_peak_memory()
        self._model, self._tokenizer = load(self.repo)

        if warmup:
            # The first generation pays for Metal kernel compilation, which
            # showed up as ~1.6s of time-to-first-token on turn one versus
            # ~0.5s afterwards. Burn that cost at load time instead.
            from mlx_lm import stream_generate

            for _ in stream_generate(
                self._model,
                self._tokenizer,
                self._render([Message(role="user", content="hi")], None),
                max_tokens=1,
            ):
                pass

        self._peak_bytes = mx.get_peak_memory()

    def unload(self) -> None:
        import gc

        import mlx.core as mx

        self._model = None
        self._tokenizer = None
        gc.collect()
        mx.clear_cache()

    # --- prompting --------------------------------------------------------

    def _render(self, messages: list[Message], tools: list[dict[str, Any]] | None) -> str:
        payload: list[dict[str, Any]] = []
        if self.system_prompt and not any(m.role == "system" for m in messages):
            payload.append({"role": "system", "content": self.system_prompt})

        for msg in messages:
            if msg.role == "tool":
                payload.append(
                    {"role": "tool", "content": msg.content, "name": msg.tool_call_id or ""}
                )
            elif msg.role == "assistant" and msg.tool_calls:
                payload.append(
                    {
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": c.name, "arguments": c.arguments},
                            }
                            for c in msg.tool_calls
                        ],
                    }
                )
            else:
                payload.append({"role": msg.role, "content": msg.content})

        return self._tokenizer.apply_chat_template(
            payload,
            tools=tools or None,
            add_generation_prompt=True,
            tokenize=False,
        )

    # --- prompt cache -----------------------------------------------------

    def reset_cache(self) -> None:
        """Drop the KV cache. Call when starting an unrelated conversation."""
        self._cache = None
        self._cached_tokens = []

    def _prepare_prompt(self, prompt: str) -> list[int]:
        """Reuse KV for the shared prefix, returning only the tokens to process.

        The system prompt and tool schemas are identical on every turn, and
        earlier turns never change, so re-encoding them each time is pure waste
        -- and prompt processing is what dominates time-to-first-token.
        """
        from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache

        tokens = self._tokenizer.encode(prompt)

        if self._cache is None:
            self._cache = make_prompt_cache(self._model)
            self._cached_tokens = []

        common = 0
        for cached, incoming in zip(self._cached_tokens, tokens):
            if cached != incoming:
                break
            common += 1

        # Never reuse the entire prompt: generation needs at least one token to
        # run the forward pass on.
        common = min(common, len(tokens) - 1)

        extra = len(self._cached_tokens) - common
        if extra > 0:
            # The history diverged (edited/trimmed); roll the cache back.
            if can_trim_prompt_cache(self._cache):
                trim_prompt_cache(self._cache, extra)
            else:
                self._cache = make_prompt_cache(self._model)
                common = 0

        self._cached_tokens = list(tokens)
        return tokens[common:]

    # --- inference --------------------------------------------------------

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 512,
    ) -> AsyncIterator[Chunk]:
        if self._model is None:
            raise RuntimeError("load() must be called before stream()")

        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        prompt_tokens = self._prepare_prompt(self._render(messages, tools))
        sampler = make_sampler(temp=self.temperature, top_p=self.top_p, top_k=self.top_k)
        logits_processors = (
            make_logits_processors(presence_penalty=self.presence_penalty)
            if self.presence_penalty
            else None
        )
        parser = _ToolCallParser()

        queue: asyncio.Queue[Chunk | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        started = time.perf_counter()

        def produce() -> None:
            """Run the blocking generator on a worker thread."""
            first = True
            generated: list[int] = []
            try:
                for response in stream_generate(
                    self._model,
                    self._tokenizer,
                    prompt_tokens,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    logits_processors=logits_processors,
                    prompt_cache=self._cache,
                ):
                    # Generated tokens land in the cache too; track them so the
                    # next turn's prefix match lines up.
                    self._cached_tokens.append(response.token)
                    generated.append(response.token)
                    ttft = None
                    if first:
                        ttft = (time.perf_counter() - started) * 1000
                        first = False

                    text, calls = parser.feed(response.text)
                    if text or ttft is not None:
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            Chunk(text=text, time_to_first_token_ms=ttft),
                        )
                    for call in calls:
                        loop.call_soon_threadsafe(queue.put_nowait, Chunk(tool_call=call))

                    if response.peak_memory:
                        self._peak_bytes = max(self._peak_bytes, response.peak_memory)

                    # Stop a degenerate loop rather than reading 512 tokens of it
                    # aloud. Checked after the chunk is emitted, so the caller
                    # keeps the text generated up to this point -- a reply that
                    # ends a few repeats in beats one that never ends.
                    period = _find_repetition_cycle(generated)
                    if period is not None:
                        self.degenerations += 1
                        break

                tail = parser.flush()
                if tail:
                    loop.call_soon_threadsafe(queue.put_nowait, Chunk(text=tail))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, Chunk(done=True))
                loop.call_soon_threadsafe(queue.put_nowait, None)

        task = asyncio.create_task(asyncio.to_thread(produce))
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            task.cancel()

    @property
    def resident_bytes(self) -> int:
        return self._peak_bytes
