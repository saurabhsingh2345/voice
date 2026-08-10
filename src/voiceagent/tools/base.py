"""Tool/plugin interface.

Phase 5 builds the registry and the concrete tools. The interface lives here
from the start so Phase 2's dummy tool and Phase 5's real tools share one
contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    content: str
    """Text handed back to the model."""
    ok: bool = True
    error: str | None = None


class Tool(ABC):
    """One capability the agent can invoke."""

    name: str
    description: str
    requires_confirmation: bool = False
    """If True, the orchestrator must get user approval before running this."""

    @property
    @abstractmethod
    def schema(self) -> dict[str, Any]:
        """JSON Schema for the arguments, in OpenAI function-calling format."""

    @abstractmethod
    async def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool. Implementations validate their own arguments."""

    def to_openai_spec(self) -> dict[str, Any]:
        """Render this tool in the format the LLM expects."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema,
            },
        }
