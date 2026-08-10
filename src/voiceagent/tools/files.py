"""Sandboxed file read/write.

The sandbox is the whole point, so it is enforced on the *resolved* path. A
check on the string alone is defeated by `../`, by a symlink pointing outside
the workspace, and by an absolute path that merely starts with the right
prefix (``/tmp/workspace-evil`` passes a naive ``startswith('/tmp/workspace')``).
Resolving first and comparing path components defeats all three.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from voiceagent.tools.base import Tool, ToolResult

DEFAULT_WORKSPACE = Path.home() / "VoiceAgentWorkspace"

#: Refuse to read or write anything larger than this, so a stray request cannot
#: blow up the context window or fill the disk.
MAX_BYTES = 256 * 1024


class SandboxError(Exception):
    """Raised when a path escapes the workspace."""


class Sandbox:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or DEFAULT_WORKSPACE).expanduser().resolve()

    def ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative: str) -> Path:
        """Resolve a user-supplied path, refusing anything outside the root."""
        if not relative or not relative.strip():
            raise SandboxError("a path is required")

        candidate = Path(relative.strip()).expanduser()
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self.root / candidate).resolve()

        # Compare resolved paths by component. `is_relative_to` handles the
        # prefix-collision case that a string startswith() check would let past.
        if resolved != self.root and not resolved.is_relative_to(self.root):
            raise SandboxError(
                f"path escapes the workspace: {relative!r} resolves outside {self.root}"
            )
        return resolved


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read a UTF-8 text file from the user's workspace directory. "
        "Paths are relative to the workspace."
    )

    def __init__(self, sandbox: Sandbox | None = None) -> None:
        self.sandbox = sandbox or Sandbox()

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path relative to the workspace, e.g. 'notes.txt'.",
                }
            },
            "required": ["path"],
        }

    async def run(self, **kwargs: Any) -> ToolResult:
        try:
            target = self.sandbox.resolve(kwargs.get("path", ""))
        except SandboxError as exc:
            return ToolResult(content="", ok=False, error=str(exc))

        if not target.exists():
            return ToolResult(content="", ok=False, error=f"no such file: {kwargs['path']}")
        if target.is_dir():
            entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
            return ToolResult(content=f"{target.name} is a directory containing: {', '.join(entries) or '(empty)'}")
        if target.stat().st_size > MAX_BYTES:
            return ToolResult(
                content="", ok=False,
                error=f"file is {target.stat().st_size} bytes; limit is {MAX_BYTES}",
            )
        try:
            return ToolResult(content=target.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            return ToolResult(content="", ok=False, error="file is not UTF-8 text")


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Write a UTF-8 text file into the user's workspace directory, "
        "overwriting it if it exists."
    )
    #: Writing is destructive, so the orchestrator must confirm first.
    requires_confirmation = True

    def __init__(self, sandbox: Sandbox | None = None) -> None:
        self.sandbox = sandbox or Sandbox()

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the workspace."},
                "content": {"type": "string", "description": "Text to write."},
            },
            "required": ["path", "content"],
        }

    async def run(self, **kwargs: Any) -> ToolResult:
        content = kwargs.get("content", "")
        if len(content.encode()) > MAX_BYTES:
            return ToolResult(content="", ok=False, error=f"content exceeds {MAX_BYTES} bytes")
        try:
            target = self.sandbox.resolve(kwargs.get("path", ""))
        except SandboxError as exc:
            return ToolResult(content="", ok=False, error=str(exc))

        self.sandbox.ensure_root()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return ToolResult(content=f"Wrote {len(content)} characters to {target.name}.")


class ListFilesTool(Tool):
    name = "list_files"
    description = "List the files in the user's workspace directory."

    def __init__(self, sandbox: Sandbox | None = None) -> None:
        self.sandbox = sandbox or Sandbox()

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "subdirectory": {
                    "type": "string",
                    "description": "Optional subdirectory to list. Defaults to the workspace root.",
                }
            },
        }

    async def run(self, **kwargs: Any) -> ToolResult:
        try:
            target = self.sandbox.resolve(kwargs.get("subdirectory") or ".")
        except SandboxError as exc:
            return ToolResult(content="", ok=False, error=str(exc))

        if not target.exists():
            return ToolResult(content="The workspace is empty.")
        entries = sorted(
            f"{p.name}/" if p.is_dir() else f"{p.name} ({p.stat().st_size} bytes)"
            for p in target.iterdir()
            if not p.name.startswith(".")
        )
        return ToolResult(content=", ".join(entries) if entries else "(empty)")
