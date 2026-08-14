"""The Hindi engine's pure logic, without loading 3 GB of weights.

Replaces `test_indic_reference.py`, which tested the same helpers on the IndicF5
path. Most of the assertions carried over unchanged because the helpers did ---
grouping, cross-fading and limiting are about joining spans of audio, not about
which model produced them. What is new here is the language guard and the
reference-clip contract, and what is *gone* is everything that tested f5-tts's
duration arithmetic, because Chatterbox has none.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from voiceagent.tts.chatterbox_indic import (
    CROSS_FADE_SECONDS,
    DEFAULT_SPAN_BYTES,
    MIN_FREE_GIB,
    REFERENCE_CLIP_SECONDS,
    ChatterboxIndicEngine,
    UnsupportedLanguage,
    _limit,
    concat_with_crossfade,
    group_sentences,
    required_free_gib,
)

HINDI = "आज मौसम बहुत सुहावना है।"


# --- language guard -------------------------------------------------------


def test_hindi_passes_the_language_guard():
    ChatterboxIndicEngine._require_hindi(HINDI)


def test_code_mixed_hindi_passes():
    """The detector treats a Latin loanword inside Devanagari as Hindi, and it has
    to: this is the register the engine is best at (95.4 % round-trip overlap on
    the code-mixed held-out subset, against IndicF5's 86.0 %)."""
    ChatterboxIndicEngine._require_hindi("मैंने एक documentary देखी जो अच्छी थी।")


def test_english_passes_the_guard_rather_than_being_rejected():
    """The guard rejects *other Indic languages*, not non-Indic text. English
    reaching this engine is a routing bug, and the router's own fallback comment
    explains why it is not this function's job to catch it."""
    ChatterboxIndicEngine._require_hindi("the weather is lovely today")


@pytest.mark.parametrize(
    "text,language",
    [
        ("বাংলা লেখা এখানে আছে", "bn"),
        ("தமிழ் உரை இங்கே உள்ளது", "ta"),
        ("ఇది తెలుగు వాక్యం", "te"),
        ("ಇದು ಕನ್ನಡ ವಾಕ್ಯ", "kn"),
    ],
)
def test_other_indic_languages_are_refused_by_name(text, language):
    """IndicF5 spoke 11 Indian languages; this checkpoint speaks 1. That is a real
    capability loss and the failure has to say so out loud --- the alternative,
    Bengali read aloud by a Hindi voice with nothing logged, is worse than an
    error."""
    with pytest.raises(UnsupportedLanguage) as caught:
        ChatterboxIndicEngine._require_hindi(text)
    assert language in str(caught.value)


def test_the_router_still_sends_every_indic_script_here():
    """Deliberate, and the reason is `route_for`'s fallback: with no matching
    route it returns `routes[0]`, which IS the Indic route. Narrowing the route to
    Hindi would therefore send Bengali to a Hindi voice silently instead of
    raising. Claiming the script and refusing loudly is the honest arrangement."""
    from voiceagent.tts.router import build_default_router

    indic = build_default_router().routes[0]
    assert {"hi", "bn", "ta", "te", "kn", "ml", "gu", "pa", "or"} <= indic.languages


# --- span grouping --------------------------------------------------------


def test_short_sentences_are_packed_into_one_span():
    """Fewer calls means fewer seams. Five sentences synthesized separately are
    four joins between independent generations."""
    assert group_sentences(["एक।", "दो।", "तीन।"], 600) == ["एक। दो। तीन।"]


def test_the_budget_is_bytes_not_characters():
    """Devanagari is 3 bytes per character in UTF-8, so a byte budget is about a
    third as many characters as it looks. Getting this backwards makes spans three
    times too long."""
    sentences = ["क" * 50 + "।"] * 4
    spans = group_sentences(sentences, 300)
    assert len(spans) > 1
    assert all(len(s.encode("utf-8")) <= 300 or " " not in s for s in spans)


def test_a_sentence_longer_than_the_budget_is_left_whole():
    """A cut we make blind is worse than a long call."""
    long = "क" * 500 + "।"
    assert group_sentences([long], 300) == [long]


def test_grouping_preserves_every_sentence():
    sentences = [f"वाक्य {i} है।" for i in range(12)]
    joined = " ".join(group_sentences(sentences, 120))
    for sentence in sentences:
        assert sentence in joined


# --- cross-fade -----------------------------------------------------------


def test_a_single_span_is_returned_unchanged():
    part = np.linspace(0, 0.5, 1000, dtype=np.float32)
    assert np.array_equal(concat_with_crossfade([part], 24_000), part)


def test_joined_length_is_shorter_than_the_sum_by_the_overlap():
    a = np.full(24_000, 0.5, dtype=np.float32)
    b = np.full(24_000, 0.5, dtype=np.float32)
    out = concat_with_crossfade([a, b], 24_000)
    assert len(out) == len(a) + len(b) - int(CROSS_FADE_SECONDS * 24_000)


def test_linear_ramps_do_not_overshoot_on_correlated_spans():
    """Equal-power ramps (sqrt of the linear ramp) are the textbook choice and are
    wrong here. They hold power constant for *uncorrelated* signals, but their
    gains sum to 1.414 where the spans are correlated --- and this model's output
    already reaches full scale, so that is clipping rather than a smoother join.
    Two identical constant spans is the exact case that exposes it.
    """
    a = np.full(12_000, 0.9, dtype=np.float32)
    out = concat_with_crossfade([a, a.copy()], 24_000)
    assert float(np.abs(out).max()) <= 0.9 + 1e-6


def test_empty_and_missing_parts_are_dropped():
    a = np.full(1000, 0.2, dtype=np.float32)
    assert len(concat_with_crossfade([None, np.zeros(0, dtype=np.float32), a], 24_000)) == 1000
    assert concat_with_crossfade([], 24_000).size == 0


def test_the_limiter_scales_down_but_never_up():
    """Scaling the whole span by one factor keeps relative levels inside a
    narration intact. Normalising *up* would make level depend on whatever the
    loudest moment happened to be, so two paragraphs of the same text would come
    back at different volumes."""
    quiet = np.full(100, 0.1, dtype=np.float32)
    assert np.array_equal(_limit(quiet), quiet)

    hot = np.full(100, 1.4, dtype=np.float32)
    assert float(np.abs(_limit(hot)).max()) == pytest.approx(0.99, abs=1e-6)


# --- memory guard ---------------------------------------------------------


def test_a_paragraph_of_short_sentences_needs_only_the_floor():
    """Sized on the longest *sentence*, not the total. Scaling on total length
    refused a 350-character paragraph at 4.0 GiB on a machine with 3.7 GiB free,
    for work that never allocated more than the floor."""
    paragraph = " ".join(["आज मौसम बहुत सुहावना है।"] * 8)
    assert len(paragraph) > 150
    assert required_free_gib(paragraph) == MIN_FREE_GIB


def test_one_very_long_sentence_raises_the_requirement():
    assert required_free_gib("क " * 400 + "।") > MIN_FREE_GIB


def test_the_floor_covers_the_default_checkpoint():
    """The 8-bit default peaks at 2.77 GiB during generation. A floor below that
    would let the wedge this guard exists to prevent happen again --- and it is
    not a crash, it is the model paging out mid-inference and the request neither
    finishing nor failing."""
    assert MIN_FREE_GIB >= 2.77


# --- reference clip -------------------------------------------------------


def test_the_reference_transcript_is_kept_but_does_not_trim():
    """f5-tts set output length from (generated chars / reference chars) x
    reference duration, so `set_reference` had to trim the transcript to match the
    12 s the model actually heard --- getting it wrong turned a 4 s sentence into
    25 s of audio. Chatterbox conditions on the audio alone. The transcript is
    stored verbatim because it is part of the consent record, and it no longer
    touches synthesis.
    """
    engine = ChatterboxIndicEngine()
    long_audio = np.zeros(24_000 * 30, dtype=np.float32)
    transcript = " ".join(["शब्द"] * 200)
    engine.set_reference(long_audio, transcript, 24_000)
    assert engine.reference_text == transcript


def test_a_new_reference_clip_invalidates_the_cached_conditionals():
    """The conditionals are derived from the clip. Reusing them across a change
    would answer in the previous speaker's voice --- which the web server's own
    lock comment names as the failure it is guarding against."""
    engine = ChatterboxIndicEngine()
    engine._conds = object()
    engine.set_reference(np.zeros(24_000, dtype=np.float32), "कुछ", 24_000)
    assert engine._conds is None


def test_reference_health_flags_a_clip_longer_than_the_conditioning_window():
    engine = ChatterboxIndicEngine()
    engine.set_reference(np.zeros(int(24_000 * 25), dtype=np.float32), "कुछ", 24_000)
    warning = engine.reference_health()
    assert warning and str(int(REFERENCE_CLIP_SECONDS)) in warning


def test_reference_health_flags_a_clip_too_short_to_clone_from():
    engine = ChatterboxIndicEngine()
    engine.set_reference(np.zeros(24_000, dtype=np.float32), "कुछ", 24_000)
    assert "only" in (engine.reference_health() or "")


def test_a_good_clip_is_silent():
    engine = ChatterboxIndicEngine()
    engine.set_reference(np.zeros(int(24_000 * 8), dtype=np.float32), "कुछ", 24_000)
    assert engine.reference_health() is None


def test_synthesizing_without_a_reference_is_refused():
    engine = ChatterboxIndicEngine()
    with pytest.raises(RuntimeError, match="set_reference"):
        engine._require_reference()


def test_synthesizing_before_load_is_refused():
    engine = ChatterboxIndicEngine()

    async def go():
        async for _ in engine.synthesize(HINDI):
            pass

    with pytest.raises(RuntimeError, match="load"):
        asyncio.run(go())


# --- configuration --------------------------------------------------------


def test_the_span_budget_does_not_depend_on_the_reference():
    """IndicF5 derived it from the reference clip's speaking rate because f5-tts
    sized its internal batches that way. Chatterbox does not batch on text length,
    so there is nothing to mirror and a constant is the honest implementation."""
    engine = ChatterboxIndicEngine()
    before = engine.batch_budget
    engine.set_reference(np.zeros(24_000 * 5, dtype=np.float32), "कुछ शब्द", 24_000)
    assert engine.batch_budget == before == DEFAULT_SPAN_BYTES


def test_the_engine_is_registered_as_permissively_licensed():
    """The entire reason for this engine's existence. IndicF5's weights were MIT;
    running it needed f5-tts, which drags in encodec (CC-BY-NC)."""
    from voiceagent.models import PERMISSIVE_LICENSES, REGISTRY

    spec = next(m for m in REGISTRY if "Multilingual" in m.name)
    assert spec.license in PERMISSIVE_LICENSES
    assert spec.measured


# --- which checkpoint actually loads ---------------------------------------


def test_the_default_is_the_local_8bit_build_when_present(tmp_path, monkeypatch):
    """8-bit is 1.9x faster than fp32 (RTF 0.63 vs 1.17) on 44% of the memory,
    at identical quality (93.5% mean round-trip either way). It is not an
    optimisation to opt into; it is what should run."""
    from voiceagent.tts import chatterbox_indic as ci

    built = tmp_path / "chatterbox-multilingual-v3-8bit"
    built.mkdir()
    monkeypatch.setattr(ci, "LOCAL_QUANTIZED", built)
    assert ci.resolve_checkpoint() == str(built)


def test_it_falls_back_to_fp32_rather_than_failing(tmp_path, monkeypatch):
    """A build step that can turn into "Hindi is broken" would be a bad trade for
    1.9x. Slower and hungrier beats absent."""
    from voiceagent.tts import chatterbox_indic as ci

    monkeypatch.setattr(ci, "LOCAL_QUANTIZED", tmp_path / "absent")

    def explode(**kwargs):
        raise RuntimeError("no disk space")

    monkeypatch.setattr("voiceagent.tts.quantize.quantize", explode)
    assert ci.resolve_checkpoint() == ci.CHATTERBOX_REPO


def test_build_can_be_declined(tmp_path, monkeypatch):
    """So a caller that must not spend seven seconds and 900 MB can say so."""
    from voiceagent.tts import chatterbox_indic as ci

    monkeypatch.setattr(ci, "LOCAL_QUANTIZED", tmp_path / "absent")
    assert ci.resolve_checkpoint(build=False) == ci.CHATTERBOX_REPO


def test_an_explicit_repo_is_not_overridden():
    """Pinning one is how the fp32-vs-8-bit comparison above was measured."""
    engine = ChatterboxIndicEngine(repo="some/other-checkpoint")
    assert engine.repo == "some/other-checkpoint"


def test_no_repo_means_resolve_at_load():
    assert ChatterboxIndicEngine().repo is None
