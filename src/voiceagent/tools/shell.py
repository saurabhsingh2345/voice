"""A shell tool constrained by an explicit allow-list.

This is the most dangerous tool in the system, so it is built to fail closed:

  * The command is parsed with ``shlex`` and the *executable* is matched against
    a fixed allow-list. There is no pattern matching and no "starts with" check.
  * It never runs through a shell, so ``;``, ``&&``, backticks, ``$(...)`` and
    redirection are inert -- they arrive as literal arguments to a program that
    will simply not understand them.
  * The working directory is the sandbox, and there is a hard timeout.

An allow-list of safe-looking binaries is still not a security boundary against
a determined attacker (``find -exec`` exists). It is a guard against an agent
doing something destructive by accident, which is the actual threat here.
"""

from __future__ import annotations

import asyncio
import shlex
from pathlib import Path
from typing import Any

from voiceagent.tools.base import Tool, ToolResult
from voiceagent.tools.files import Sandbox

#: Read-only, non-destructive commands. Anything that writes, installs,
#: escalates, or reaches the network stays out.
DEFAULT_ALLOWLIST = frozenset(
    {"ls", "cat", "head", "tail", "wc", "grep", "find", "file", "stat", "du", "date", "echo", "pwd"}
)

#: Arguments that turn an allow-listed command into an arbitrary-execution
#: primitive. Checked explicitly because the allow-list alone would let them by.
FORBIDDEN_ARGS = frozenset({"-exec", "-execdir", "-ok", "-okdir", "-fprintf", "-delete"})

TIMEOUT_SECONDS = 10.0
MAX_OUTPUT_CHARS = 8_000


class ShellTool(Tool):
    name = "run_command"
    description = (
        "Run a read-only shell command in the user's workspace. Only these "
        f"commands are permitted: {', '.join(sorted(DEFAULT_ALLOWLIST))}."
    )
    requires_confirmation = True

    def __init__(
        self,
        allowlist: frozenset[str] | None = None,
        sandbox: Sandbox | None = None,
    ) -> None:
        self.allowlist = allowlist or DEFAULT_ALLOWLIST
        self.sandbox = sandbox or Sandbox()

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "The command to run, e.g. 'ls -la' or 'grep -n TODO notes.txt'. "
                        "Shell operators are not interpreted."
                    ),
                }
            },
            "required": ["command"],
        }

    def _vet(self, command: str) -> list[str]:
        """Parse and authorise a command, or raise PermissionError."""
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            raise PermissionError(f"could not parse command: {exc}") from exc

        if not argv:
            raise PermissionError("empty command")

        executable = Path(argv[0]).name
        if executable not in self.allowlist:
            raise PermissionError(
                f"{executable!r} is not allow-listed. Permitted: {', '.join(sorted(self.allowlist))}"
            )

        forbidden = FORBIDDEN_ARGS.intersection(argv)
        if forbidden:
            raise PermissionError(f"argument(s) not permitted: {', '.join(sorted(forbidden))}")

        return argv

    async def run(self, **kwargs: Any) -> ToolResult:
        command = kwargs.get("command", "")
        try:
            argv = self._vet(command)
        except PermissionError as exc:
            return ToolResult(content="", ok=False, error=str(exc))

        self.sandbox.ensure_root()
        try:
            # create_subprocess_exec, never _shell: argv goes straight to
            # execve, so shell metacharacters are just bytes.
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=self.sandbox.root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError:
            return ToolResult(content="", ok=False, error=f"command not found: {argv[0]}")

        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return ToolResult(content="", ok=False, error=f"timed out after {TIMEOUT_SECONDS:.0f}s")

        output = stdout.decode("utf-8", errors="replace").strip()
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + f"\n... (truncated at {MAX_OUTPUT_CHARS} chars)"

        if process.returncode != 0:
            return ToolResult(
                content=output, ok=False, error=f"exit code {process.returncode}"
            )
        return ToolResult(content=output or "(no output)")
