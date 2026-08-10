"""Tests for the streaming sentence chunker.

The chunker decides when TTS starts speaking, so two failures matter most:
cutting somewhere unnatural (an abbreviation, a decimal), and never cutting at
all so the user waits in silence.
"""

from __future__ import annotations

import pytest

from voiceagent.tts.chunker import SentenceChunker


def drain(text: str, chunk_size: int = 5) -> list[str]:
    """Feed text in fixed-size pieces, as a token stream would arrive."""
    chunker = SentenceChunker()
    out = []
    for i in range(0, len(text), chunk_size):
        out.extend(chunker.feed(text[i : i + chunk_size]))
    out.extend(chunker.flush())
    return out


def test_splits_on_sentences():
    assert drain("Hello there. How are you? I am fine!") == [
        "Hello there.",
        "How are you?",
        "I am fine!",
    ]


def test_first_chunk_is_allowed_to_be_short():
    """First audio latency matters more than a long opening phrase."""
    chunks = drain("Sure. I will look that up for you right away.")
    assert chunks[0] == "Sure."


def test_decimals_are_not_split():
    assert drain("The value is 3.5 degrees today.") == ["The value is 3.5 degrees today."]


def test_version_numbers_are_not_split():
    assert drain("Install version 1.2.3 now please.") == ["Install version 1.2.3 now please."]


@pytest.mark.parametrize("abbrev", ["Dr.", "Mr.", "Mrs.", "e.g.", "etc.", "a.m."])
def test_abbreviations_do_not_end_a_sentence(abbrev):
    text = f"Please ask {abbrev} Smith about the report now."
    assert drain(text) == [text]


def test_long_run_on_falls_back_to_clause_break():
    """A sentence with no terminal must still get spoken eventually."""
    text = "well " * 60
    chunks = drain(text)
    assert len(chunks) > 1
    assert all(len(c) <= 260 for c in chunks)


def test_flush_returns_trailing_text():
    chunker = SentenceChunker()
    assert chunker.feed("No terminal here") == []
    assert chunker.flush() == ["No terminal here"]


def test_nothing_is_lost_or_duplicated():
    text = (
        "Sure thing. The weather in Paris is 42.5 degrees, which is unusual. "
        "Dr. Smith says e.g. fog is likely! Shall I continue?"
    )
    for size in (1, 3, 7, 50):
        chunker = SentenceChunker()
        out = []
        for i in range(0, len(text), size):
            out.extend(chunker.feed(text[i : i + size]))
        out.extend(chunker.flush())
        # Whitespace normalizes, but no character may vanish or repeat.
        assert "".join(out).replace(" ", "") == text.replace(" ", "")


def test_trailing_quotes_stay_with_sentence():
    assert drain('He said "hello there." Then he left.') == [
        'He said "hello there."',
        "Then he left.",
    ]


def test_empty_stream_produces_nothing():
    chunker = SentenceChunker()
    assert chunker.feed("") == []
    assert chunker.flush() == []


def test_first_chunk_does_not_wait_for_a_long_sentence():
    """Perceived latency is set by first audio, so the opener cuts early."""
    text = "I can't provide information about quantized models or Apple Silicon performance."
    chunks = drain(text)
    assert len(chunks) > 1
    assert len(chunks[0]) <= 45
    assert not chunks[0].endswith(" ")
    # Later chunks are not cut short by the same rule.
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_first_chunk_prefers_a_clause_break():
    chunks = drain("Sure thing, I will look that up for you right now and report back.")
    assert chunks[0] == "Sure thing,"


def test_short_first_sentence_still_wins_over_early_cut():
    """A sentence that ends before the threshold is used as-is."""
    assert drain("Hello there. How are you doing today my friend?")[0] == "Hello there."
