"""A dummy tool used to validate tool-call parsing before real tools exist.

Phase 2's job is to prove that this specific model, through this specific chat
template, produces tool calls we can parse reliably. Returning obviously fake
data makes it unmistakable when the model answers from the tool versus from its
own priors.
"""

from __future__ import annotations

from typing import Any

from voiceagent.tools.base import Tool, ToolResult


class DummyWeatherTool(Tool):
    name = "get_weather"
    description = (
        "Get the current weather for a city. Use this whenever the user asks "
        "about weather conditions or temperature."
    )

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        """Recorded invocations, so tests can assert the tool actually ran."""

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'Paris' or 'San Francisco'.",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit. Defaults to celsius.",
                },
            },
            "required": ["city"],
        }

    async def run(self, **kwargs: Any) -> ToolResult:
        city = kwargs.get("city")
        if not city:
            return ToolResult(content="", ok=False, error="missing required argument 'city'")

        unit = kwargs.get("unit", "celsius")
        self.calls.append({"city": city, "unit": unit})

        # Deliberately absurd so a hallucinated answer is easy to spot.
        temp = 42 if unit == "celsius" else 108
        return ToolResult(
            content=(
                f"{city}: {temp} degrees {unit}, thick purple fog, "
                f"wind 3 km/h from the northeast."
            )
        )
