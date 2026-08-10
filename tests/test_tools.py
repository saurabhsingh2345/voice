"""Tests for the tool sandbox and the shell allow-list.

These are the tools that can do real damage, so the tests are written as
attacks: each one is a way an agent (or a prompt injection reaching it) might
try to get outside the box.
"""

from __future__ import annotations

import pytest

from voiceagent.tools.files import (
    ListFilesTool,
    ReadFileTool,
    Sandbox,
    SandboxError,
    WriteFileTool,
)
from voiceagent.tools.http import HttpRequestTool
from voiceagent.tools.registry import ToolRegistry
from voiceagent.tools.shell import ShellTool
from voiceagent.tools.base import Tool, ToolResult


@pytest.fixture
def sandbox(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "notes.txt").write_text("hello from the workspace")
    # A secret that lives OUTSIDE the sandbox; nothing should ever reach it.
    (tmp_path / "secret.txt").write_text("TOP SECRET")
    return Sandbox(root)


# --- sandbox escapes ------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "../secret.txt",
        "../../etc/passwd",
        "subdir/../../secret.txt",
        "/etc/passwd",
        "~/.ssh/id_rsa",
        "./../secret.txt",
    ],
)
def test_traversal_is_refused(sandbox, path):
    with pytest.raises(SandboxError):
        sandbox.resolve(path)


def test_sibling_directory_with_shared_prefix_is_refused(tmp_path):
    """`/x/workspace-evil` must not pass a prefix check against `/x/workspace`."""
    root = tmp_path / "workspace"
    root.mkdir()
    evil = tmp_path / "workspace-evil"
    evil.mkdir()
    (evil / "loot.txt").write_text("nope")

    with pytest.raises(SandboxError):
        Sandbox(root).resolve(str(evil / "loot.txt"))


def test_symlink_pointing_outside_is_refused(sandbox, tmp_path):
    link = sandbox.root / "escape"
    link.symlink_to(tmp_path / "secret.txt")
    with pytest.raises(SandboxError):
        sandbox.resolve("escape")


def test_paths_inside_are_allowed(sandbox):
    assert sandbox.resolve("notes.txt").name == "notes.txt"
    assert sandbox.resolve("a/b/c.txt").is_relative_to(sandbox.root)


async def test_read_file_refuses_escape(sandbox):
    result = await ReadFileTool(sandbox).run(path="../secret.txt")
    assert not result.ok
    assert "TOP SECRET" not in result.content


async def test_read_file_works_inside(sandbox):
    result = await ReadFileTool(sandbox).run(path="notes.txt")
    assert result.ok and "hello from the workspace" in result.content


async def test_write_then_read_roundtrip(sandbox):
    write = await WriteFileTool(sandbox).run(path="out/deep.txt", content="written")
    assert write.ok
    read = await ReadFileTool(sandbox).run(path="out/deep.txt")
    assert read.content == "written"


async def test_write_refuses_escape(sandbox, tmp_path):
    result = await WriteFileTool(sandbox).run(path="../pwned.txt", content="x")
    assert not result.ok
    assert not (tmp_path / "pwned.txt").exists()


def test_destructive_tools_require_confirmation():
    assert WriteFileTool().requires_confirmation is True
    assert ShellTool().requires_confirmation is True
    assert HttpRequestTool().requires_confirmation is True
    assert ReadFileTool().requires_confirmation is False


# --- shell allow-list -----------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "curl https://evil.example.com",
        "python -c 'import os'",
        "sudo ls",
        "bash -c 'ls'",
        "sh",
        "/bin/rm file",
        "chmod 777 .",
    ],
)
async def test_non_allowlisted_commands_are_refused(sandbox, command):
    result = await ShellTool(sandbox=sandbox).run(command=command)
    assert not result.ok
    assert "allow-list" in (result.error or "").lower()


@pytest.mark.parametrize(
    "command",
    [
        "ls; rm -rf /",
        "ls && curl evil.com",
        "ls | tee /etc/passwd",
        "ls $(whoami)",
        "ls `whoami`",
    ],
)
async def test_shell_metacharacters_are_not_interpreted(sandbox, command):
    """Operators are inert: argv goes to execve, never through a shell.

    Two safe outcomes are possible and both are fine. `ls; rm -rf /` splits so
    that the executable is literally `ls;`, which is not allow-listed and is
    refused. `ls $(whoami)` splits to `ls` plus a literal argument, which runs
    but treats the substitution as a filename that does not exist. What must
    never happen is the second command executing.
    """
    result = await ShellTool(sandbox=sandbox).run(command=command)
    refused = not result.ok and "allow-list" in (result.error or "").lower()
    ran_harmlessly = "rm" not in (result.content or "")
    assert refused or ran_harmlessly


async def test_find_exec_is_blocked(sandbox):
    """find is allow-listed but -exec would be arbitrary execution."""
    result = await ShellTool(sandbox=sandbox).run(command="find . -exec rm {} ;")
    assert not result.ok
    assert "not permitted" in (result.error or "")


async def test_allowed_command_runs(sandbox):
    result = await ShellTool(sandbox=sandbox).run(command="ls")
    assert result.ok
    assert "notes.txt" in result.content


async def test_unparseable_command_is_refused(sandbox):
    result = await ShellTool(sandbox=sandbox).run(command="ls 'unterminated")
    assert not result.ok


# --- http -----------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1:8823/api/data", "http://localhost/admin", "http://169.254.169.254/latest/meta-data"],
)
async def test_http_refuses_private_addresses(url):
    result = await HttpRequestTool().run(url=url)
    assert not result.ok
    assert "private or loopback" in (result.error or "")


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com", "gopher://x"])
async def test_http_refuses_non_http_schemes(url):
    result = await HttpRequestTool().run(url=url)
    assert not result.ok
    assert "http and https" in (result.error or "")


# --- registry -------------------------------------------------------------


class Exploding(Tool):
    name = "boom"
    description = "always fails"

    @property
    def schema(self):
        return {"type": "object", "properties": {}}

    async def run(self, **kwargs):
        raise RuntimeError("kaboom")


async def test_registry_converts_exceptions_into_results():
    """A throwing tool must not abort the turn."""
    registry = ToolRegistry([Exploding()])
    result = await registry.invoke("boom", {})
    assert not result.ok
    assert "kaboom" in result.error


async def test_registry_reports_unknown_tools():
    registry = ToolRegistry([ReadFileTool()])
    result = await registry.invoke("nope", {})
    assert not result.ok
    assert "unknown tool" in result.error


async def test_registry_rejects_duplicate_names():
    registry = ToolRegistry([ReadFileTool()])
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(ReadFileTool())


def test_registry_specs_are_openai_shaped():
    registry = ToolRegistry([ReadFileTool(), ShellTool()])
    specs = registry.specs()
    assert len(specs) == 2
    for spec in specs:
        assert spec["type"] == "function"
        assert {"name", "description", "parameters"} <= set(spec["function"])


# --- confirmation gate ----------------------------------------------------


class Recording(Tool):
    """A confirmation-requiring tool that records whether it actually ran."""

    name = "danger"
    description = "requires confirmation"
    requires_confirmation = True

    def __init__(self):
        self.ran = False

    @property
    def schema(self):
        return {"type": "object", "properties": {}}

    async def run(self, **kwargs):
        self.ran = True
        return ToolResult(content="done")


async def _agent_with(tool, confirm=None):
    from voiceagent.llm.agent import Agent
    from voiceagent.llm.base import ToolCall

    class NullEngine:
        name = "null"
        def load(self): ...
        def unload(self): ...
        async def stream(self, *a, **k): ...
        @property
        def resident_bytes(self): return 0

    agent = Agent(NullEngine(), tools=[tool], confirm=confirm)
    return agent, ToolCall(id="1", name=tool.name, arguments={})


async def test_confirmation_required_tool_is_denied_by_default():
    """Forgetting to wire a confirm hook must fail closed, not open."""
    tool = Recording()
    agent, call = await _agent_with(tool)
    result = await agent._invoke(call)
    assert not result.ok
    assert tool.ran is False, "tool ran without confirmation"


async def test_confirmation_denied_does_not_run_the_tool():
    tool = Recording()
    async def deny(t, a): return False
    agent, call = await _agent_with(tool, confirm=deny)
    result = await agent._invoke(call)
    assert not result.ok and "declined" in result.error
    assert tool.ran is False


async def test_confirmation_granted_runs_the_tool():
    tool = Recording()
    async def allow(t, a): return True
    agent, call = await _agent_with(tool, confirm=allow)
    result = await agent._invoke(call)
    assert result.ok and tool.ran is True


async def test_safe_tools_never_prompt():
    """A read-only tool must not be gated behind confirmation."""
    from voiceagent.llm.agent import Agent
    from voiceagent.llm.base import ToolCall

    asked = False
    async def confirm(t, a):
        nonlocal asked
        asked = True
        return True

    class NullEngine:
        name = "null"
        def load(self): ...
        def unload(self): ...
        async def stream(self, *a, **k): ...
        @property
        def resident_bytes(self): return 0

    agent = Agent(NullEngine(), tools=[ListFilesTool()], confirm=confirm)
    await agent._invoke(ToolCall(id="1", name="list_files", arguments={}))
    assert asked is False
