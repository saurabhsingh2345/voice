"""The tool registry.

Tools are how the agent acts rather than just talks, which makes them the part
of the system that can actually do damage. Two rules are enforced here rather
than left to each tool:

  * Every tool declares whether it needs confirmation. The orchestrator must
    ask before running one that does -- there is no "the model seemed sure"
    override.
  * Tool failures are returned to the model as text, never raised. A tool that
    throws would abort the turn; a tool that reports "permission denied" lets
    the model explain itself or try something else.
"""

from __future__ import annotations

from typing import Any

from voiceagent.tools.base import Tool, ToolResult


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[dict[str, Any]] | None:
        """OpenAI-format schemas for every registered tool."""
        if not self._tools:
            return None
        return [tool.to_openai_spec() for tool in self._tools.values()]

    async def invoke(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Run a tool, converting any failure into a result the model can read."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                content="",
                ok=False,
                error=f"unknown tool {name!r}; available: {', '.join(self.names)}",
            )
        try:
            return await tool.run(**arguments)
        except TypeError as exc:
            return ToolResult(content="", ok=False, error=f"bad arguments: {exc}")
        except Exception as exc:  # noqa: BLE001 -- a tool must never kill the turn
            return ToolResult(content="", ok=False, error=f"{type(exc).__name__}: {exc}")
