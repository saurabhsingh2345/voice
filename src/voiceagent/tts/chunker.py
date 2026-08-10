"""Split a streaming token feed into speakable chunks.

The whole point of streaming TTS is to start speaking before the LLM has
finished writing. That means cutting the token stream at points where a
synthesizer will produce natural prosody -- sentence ends, or failing that
clause boundaries -- without ever cutting mid-word or mid-number.

The first chunk is deliberately allowed to be shorter than the rest: it sets
the user-perceived response latency, and a slightly clipped opening phrase is a
far better trade than half a second of silence.
"""

from __future__ import annotations

import re

#: Sentence-final punctuation.
TERMINALS = ".!?"

#: Weaker breaks, used only when a chunk is running long.
CLAUSE_BREAKS = ",;:"

#: Tokens that end in '.' but do not end a sentence.
ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st",
        "e.g", "i.e", "etc", "vs", "approx", "no", "fig",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
        "mon", "tue", "wed", "thu", "fri", "sat", "sun",
        "a.m", "p.m", "u.s", "u.k",
    }
)

#: A digit on both sides of the dot -- "3.5", "1.2.3" -- is never a boundary.
_DECIMAL = re.compile(r"\d\.\d?$")

#: Trailing token before a terminal, used for the abbreviation check.
_LAST_WORD = re.compile(r"([A-Za-z][A-Za-z.]*)\.$")


class SentenceChunker:
    """Accumulates text and yields speakable chunks as they become available."""

    def __init__(
        self,
        max_chars: int = 220,
        min_clause_chars: int = 40,
        first_chunk_max_chars: int = 45,
    ) -> None:
        self.max_chars = max_chars
        """Length past which we stop waiting for a sentence terminal."""
        self.min_clause_chars = min_clause_chars
        """Shortest acceptable chunk when falling back to a clause break."""
        self.first_chunk_max_chars = first_chunk_max_chars
        """The opening chunk cuts at a word boundary past this, without waiting
        for a sentence to finish. Perceived response time is set entirely by
        when the first audio starts, and a long opening sentence otherwise
        holds it back -- measured at 1.2s of extra silence in the live loop.
        Later chunks have no such pressure, so they wait for real boundaries."""
        self._buffer = ""
        self._emitted = 0

    def _is_boundary(self, upto: int) -> bool:
        """Is the character at `upto` a genuine sentence end?"""
        head = self._buffer[: upto + 1]
        if _DECIMAL.search(head):
            return False
        match = _LAST_WORD.search(head)
        if match and match.group(1).lower().rstrip(".") in ABBREVIATIONS:
            return False
        # A terminal must be followed by space or end-of-buffer, otherwise we
        # may be mid-token (a URL, a version number).
        rest = self._buffer[upto + 1 :]
        return rest == "" or rest[0].isspace() or rest[0] in "\"')]}"

    def _find_cut(self) -> int:
        """Index just past a good cut point, or -1.

        A completed sentence always cuts, however short. Withholding "Sure."
        until the following sentence arrives would delay first audio for no
        prosodic gain -- and a short opening reply is exactly the case where
        latency is most noticeable.
        """
        for i, char in enumerate(self._buffer):
            if char in TERMINALS and self._is_boundary(i):
                # Include trailing quotes/brackets that belong to the sentence.
                end = i + 1
                while end < len(self._buffer) and self._buffer[end] in "\"')]}":
                    end += 1
                return end

        # The opening chunk does not wait for a sentence: start speaking at the
        # first clean word boundary past the threshold.
        if self._emitted == 0 and len(self._buffer) >= self.first_chunk_max_chars:
            window = self._buffer[: self.first_chunk_max_chars]
            for i in range(len(window) - 1, 0, -1):
                if window[i] in CLAUSE_BREAKS:
                    return i + 1
            cut = window.rfind(" ")
            if cut > 0:
                return cut

        # Nothing terminal. If we are running long, fall back to a clause break
        # so the listener is not left waiting on a run-on sentence.
        if len(self._buffer) >= self.max_chars:
            for i in range(len(self._buffer) - 1, self.min_clause_chars - 1, -1):
                if self._buffer[i] in CLAUSE_BREAKS:
                    return i + 1
            # No punctuation at all -- cut at the last word break.
            cut = self._buffer.rfind(" ", self.min_clause_chars)
            if cut != -1:
                return cut
        return -1

    def feed(self, text: str) -> list[str]:
        """Add streamed text, returning any chunks that are ready to speak."""
        self._buffer += text
        chunks = []
        while True:
            cut = self._find_cut()
            if cut == -1:
                break
            chunk = self._buffer[:cut].strip()
            self._buffer = self._buffer[cut:].lstrip()
            if chunk:
                chunks.append(chunk)
                self._emitted += 1
        return chunks

    def flush(self) -> list[str]:
        """Return whatever is left once the stream ends."""
        tail = self._buffer.strip()
        self._buffer = ""
        if tail:
            self._emitted += 1
            return [tail]
        return []

    def reset(self) -> None:
        self._buffer = ""
        self._emitted = 0
