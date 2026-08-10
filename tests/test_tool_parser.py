"""Tests for the streaming tool-call parser.

The parser sees the model's output one token at a time, so a `<tool_call>` tag
can be split across any number of chunks. These tests feed the same input at
every possible split to make sure nothing leaks or gets dropped.

Run with::

    uv run python -m pytest tests/ -q
"""

from __future__ import annotations

import pytest

from voiceagent.llm.mlx_engine import _ToolCallParser


def drain(chunks: list[str]) -> tuple[str, list]:
    """Feed chunks through a fresh parser, returning (text, calls)."""
    parser = _ToolCallParser()
    text = ""
    calls = []
    for chunk in chunks:
        safe, found = parser.feed(chunk)
        text += safe
        calls.extend(found)
    return text + parser.flush(), calls


def every_split(payload: str) -> list[list[str]]:
    """All one- and two-cut splittings of payload, plus per-character."""
    variants = [[payload], list(payload)]
    for i in range(1, len(payload)):
        variants.append([payload[:i], payload[i:]])
    return variants


def test_plain_text_passes_through():
    text, calls = drain(["Hello ", "there", "!"])
    assert text == "Hello there!"
    assert calls == []


def test_tool_call_is_extracted_and_hidden():
    payload = 'Sure.<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
    text, calls = drain([payload])
    assert text == "Sure."
    assert len(calls) == 1
    assert calls[0].name == "get_weather"
    assert calls[0].arguments == {"city": "Paris"}


@pytest.mark.parametrize(
    "chunks",
    every_split(
        'Sure.<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
    ),
)
def test_split_at_any_boundary(chunks):
    """A tag split across chunks must never leak into the spoken text."""
    text, calls = drain(chunks)
    assert text == "Sure."
    assert len(calls) == 1
    assert calls[0].arguments == {"city": "Paris"}


def test_no_partial_tag_ever_leaks():
    """Text emitted mid-stream must never contain a fragment of the tag."""
    payload = 'Hi<tool_call>{"name": "f", "arguments": {}}</tool_call>bye'
    parser = _ToolCallParser()
    emitted = []
    for char in payload:
        safe, _ = parser.feed(char)
        emitted.append(safe)
    emitted.append(parser.flush())
    assert "".join(emitted) == "Hibye"
    assert not any("<" in piece for piece in emitted)


def test_multiple_tool_calls():
    payload = (
        '<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>'
    )
    text, calls = drain([payload])
    assert text == ""
    assert [c.name for c in calls] == ["a", "b"]


def test_string_encoded_arguments_are_decoded():
    """Some chat templates double-encode the arguments object."""
    payload = '<tool_call>{"name": "f", "arguments": "{\\"city\\": \\"Rome\\"}"}</tool_call>'
    _, calls = drain([payload])
    assert calls[0].arguments == {"city": "Rome"}


def test_malformed_json_is_dropped_not_leaked():
    payload = "<tool_call>this is not json</tool_call>after"
    text, calls = drain([payload])
    assert calls == []
    assert text == "after"


def test_truncated_tool_call_yields_no_text():
    """Generation cut off mid-call must not spill the fragment as speech."""
    text, calls = drain(['ok<tool_call>{"name": "f", "argum'])
    assert text == "ok"
    assert calls == []


def test_text_resembling_tag_start_is_not_held_forever():
    text, calls = drain(["a < b and c > d"])
    assert text == "a < b and c > d"
    assert calls == []
