"""Tools that let the agent write to and read from long-term memory.

Remembering is deliberately an explicit tool call rather than something the
system does automatically to every utterance. Silently persisting everything a
user says, encrypted or not, is a surprising default for a product whose whole
premise is privacy -- the user should be able to hear the agent decide to
remember something.
"""

from __future__ import annotations

from typing import Any

from voiceagent.storage.db import EncryptedStore
from voiceagent.tools.base import Tool, ToolResult


class RememberTool(Tool):
    name = "remember"
    description = (
        "Save a durable fact about the user for future conversations, such as a "
        "preference, a name, or a recurring detail. Use sparingly, for things "
        "worth recalling weeks later."
    )

    def __init__(self, store: EncryptedStore) -> None:
        self.store = store

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "fact": {
                    "type": "string",
                    "description": "The fact, written as a standalone sentence.",
                }
            },
            "required": ["fact"],
        }

    async def run(self, **kwargs: Any) -> ToolResult:
        fact = (kwargs.get("fact") or "").strip()
        if not fact:
            return ToolResult(content="", ok=False, error="a fact is required")
        self.store.remember(fact)
        return ToolResult(content=f"Saved: {fact}")


class RecallTool(Tool):
    name = "recall"
    description = (
        "Search previously saved facts about the user. Use when the user refers "
        "to something from an earlier conversation."
    )

    def __init__(self, store: EncryptedStore) -> None:
        self.store = store

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for."}
            },
            "required": ["query"],
        }

    async def run(self, **kwargs: Any) -> ToolResult:
        hits = self.store.recall(kwargs.get("query", ""), limit=5)
        if not hits:
            return ToolResult(content="Nothing relevant saved.")
        return ToolResult(content="\n".join(f"- {m.content}" for m in hits))


class ForgetAllTool(Tool):
    name = "forget_everything"
    description = (
        "Permanently delete all saved conversation history and memories. "
        "This cannot be undone."
    )
    requires_confirmation = True

    def __init__(self, store: EncryptedStore) -> None:
        self.store = store

    @property
    def schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def run(self, **kwargs: Any) -> ToolResult:
        counts = self.store.delete_all()
        return ToolResult(
            content=(
                f"Deleted {counts['messages']} messages, "
                f"{counts['conversations']} conversations and "
                f"{counts['memories']} memories, and destroyed the encryption key."
            )
        )
