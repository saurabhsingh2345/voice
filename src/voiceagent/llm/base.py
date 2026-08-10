"""LLM interface, including tool calling.

Phase 2 implements this over mlx-lm. Keeping tool-call parsing behind the
interface means a different model or runtime can be swapped in later without
the orchestration layer noticing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    role: Role
    content: str
    tool_call_id: str | None = None
    tool_calls: list["ToolCall"] = field(default_factory=list)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Chunk:
    """One streamed unit of model output."""

    text: str = ""
    tool_call: ToolCall | None = None
    done: bool = False
    time_to_first_token_ms: float | None = None


class LLMEngine(ABC):
    """A local text-generation backend with OpenAI-style tool calling."""

    name: str

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 512,
    ) -> AsyncIterator[Chunk]:
        """Stream a response, emitting tool calls as they are parsed."""

    @property
    @abstractmethod
    def resident_bytes(self) -> int: ...
