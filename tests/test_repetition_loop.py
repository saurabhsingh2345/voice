"""Qwen3's repetition loop, and the two things now standing between it and a listener.

The loop is the worst failure this system has in front of a person: roughly once
in 80 generations the model gets stuck --- `ऊपर से ऊपर ऊपर से ऊपर …` --- and runs
until the 512-token budget is exhausted, all of it read aloud. No quality metric
sees it. It is not attributable to any prompt.

Two changes: Qwen's own sampling parameters (which this engine was not using at
all), and a detector that stops generation. The detector is the part that
actually bounds it; sampling only lowers the odds. Both are tested here, and the
detector is testable without loading 2 GB of weights because it is a pure
function over token ids.
"""

from __future__ import annotations

from voiceagent.llm.mlx_engine import (
    MAX_CYCLE_TOKENS,
    MIN_CYCLE_REPEATS,
    MIN_SINGLE_TOKEN_REPEATS,
    PRESENCE_PENALTY,
    TOP_K,
    TOP_P,
    MLXLLMEngine,
    _find_repetition_cycle,
)


# --- the detector fires on real degeneration -------------------------------


def test_a_repeated_phrase_is_caught():
    """The observed failure. `ऊपर से ऊपर` is a short token cycle repeating until
    the budget runs out; four repeats is enough to be certain and early enough
    that a listener hears a stumble rather than a minute of it."""
    cycle = [901, 55, 901, 72]
    assert _find_repetition_cycle(cycle * MIN_CYCLE_REPEATS) == 4


def test_a_cycle_is_caught_after_ordinary_text():
    """It degenerates partway through a real answer, not from the first token."""
    prefix = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    assert _find_repetition_cycle(prefix + [7, 8] * MIN_CYCLE_REPEATS) == 2


def test_a_single_token_repeated_needs_a_longer_run():
    """"!!!!" and a row of newlines are ordinary text. Eight is not."""
    assert _find_repetition_cycle([42] * (MIN_SINGLE_TOKEN_REPEATS - 1)) is None
    assert _find_repetition_cycle([42] * MIN_SINGLE_TOKEN_REPEATS) == 1


def test_the_longest_cycle_it_looks_for_is_bounded():
    long_cycle = list(range(100, 100 + MAX_CYCLE_TOKENS))
    assert _find_repetition_cycle(long_cycle * MIN_CYCLE_REPEATS) == MAX_CYCLE_TOKENS

    too_long = list(range(200, 200 + MAX_CYCLE_TOKENS + 1))
    assert _find_repetition_cycle(too_long * MIN_CYCLE_REPEATS) is None


# --- and not on legitimate text --------------------------------------------
#
# Truncating a correct reply is a worse bug than the one being fixed: the loop
# is obvious, a silent truncation is not. These are the false positives worth
# being sure about.


def test_ordinary_text_is_left_alone():
    assert _find_repetition_cycle([5, 9, 3, 7, 1, 8, 2, 6, 4, 0, 11, 13]) is None


def test_three_repeats_of_a_phrase_are_allowed():
    """Below the threshold on purpose. Rhetorical repetition is real language --
    "बहुत बहुत बहुत धन्यवाद" -- and stopping on it would cut off a correct reply."""
    assert _find_repetition_cycle([21, 22] * (MIN_CYCLE_REPEATS - 1)) is None


def test_a_repeated_word_in_different_contexts_does_not_fire():
    """A list, or any answer that reuses a word. The token recurs constantly and
    never forms a cycle."""
    the = 464
    assert _find_repetition_cycle([the, 1, 2, the, 3, 4, the, 5, 6, the, 7, 8]) is None


def test_an_almost_cycle_does_not_fire():
    """Exact match on token ids, not similarity. One differing token anywhere in
    the run means it is not a loop."""
    tokens = [31, 32] * MIN_CYCLE_REPEATS
    tokens[-1] = 99
    assert _find_repetition_cycle(tokens) is None


def test_a_short_generation_cannot_trigger_it():
    assert _find_repetition_cycle([]) is None
    assert _find_repetition_cycle([1, 2, 3]) is None


def test_it_reports_the_shortest_cycle():
    """A run of one token repeated is also, technically, a run of a two-token
    cycle. Reporting the shortest keeps the number meaningful for debugging."""
    assert _find_repetition_cycle([8] * 16) == 1


# --- sampling ---------------------------------------------------------------


def test_the_engine_uses_qwens_published_sampling():
    """It sampled on temperature alone -- no truncation whatsoever -- which is
    the setup a degenerate cycle survives in. Qwen publishes these per variant
    and 0.7 already matched the Instruct row; top_p and top_k were missing."""
    engine = MLXLLMEngine()
    assert engine.temperature == 0.7
    assert engine.top_p == TOP_P == 0.8
    assert engine.top_k == TOP_K == 20


def test_presence_penalty_is_available_but_off():
    """Qwen's documented lever for endless repetition, with an explicit warning
    that higher values cause language mixing. Off by default because the reply's
    language *selects the TTS engine* here, so drifting mid-reply is not
    cosmetic -- it gets routed on the first characters and mispronounced after.
    """
    assert PRESENCE_PENALTY == 0.0
    assert MLXLLMEngine().presence_penalty == 0.0
    assert MLXLLMEngine(presence_penalty=1.0).presence_penalty == 1.0


def test_no_repetition_penalty_is_set():
    """Qwen says to leave it disabled. Named here so nobody adds one as an
    obvious-looking fix for the bug this file is about."""
    import inspect

    source = inspect.getsource(MLXLLMEngine.stream)
    assert "repetition_penalty" not in source


def test_degenerations_are_counted():
    """The baseline is ~1 in 80. A rate far above that means the sampling change
    made things worse, and nothing else would show it."""
    assert MLXLLMEngine().degenerations == 0


def test_generation_stops_on_a_detected_cycle():
    """The loop must break out, not merely record the fact. Asserted on the
    source because exercising it needs the model."""
    import inspect

    source = inspect.getsource(MLXLLMEngine.stream)
    assert "_find_repetition_cycle(generated)" in source
    assert "self.degenerations += 1" in source
    assert "break" in source
