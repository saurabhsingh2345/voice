"""Tests for language detection and TTS routing.

Routing is what keeps Hindi inside the memory budget, so the tests cover both
that the right engine is picked and that the wrong one gets evicted.
"""

from __future__ import annotations

import pytest

from voiceagent.text.detect import detect
from voiceagent.tts.router import Route, TTSRouter


@pytest.mark.parametrize(
    "text,lang",
    [
        ("नमस्ते, आप कैसे हैं?", "hi"),
        ("Hello, how are you?", "en"),
        ("मैंने अभी email भेज दिया है, please check कर लेना।", "hi"),
        ("ये feature अभी beta में है।", "hi"),
        ("வணக்கம், எப்படி இருக்கிறீர்கள்?", "ta"),
        ("নমস্কার, আপনি কেমন আছেন?", "bn"),
    ],
)
def test_language_detection(text, lang):
    assert detect(text).language == lang


def test_code_mixing_routes_to_indic():
    """Hindi with English words is Hindi -- an Indic model handles both."""
    d = detect("मेरा laptop बहुत slow चल रहा है।")
    assert d.language == "hi" and d.is_code_mixed


def test_one_stray_devanagari_does_not_flip_english():
    d = detect("The word नमस्ते means hello in Hindi and is used very widely today")
    assert d.language == "en"


def test_empty_and_numeric_text_defaults_to_english():
    assert detect("").language == "en"
    assert detect("12345 !!!").language == "en"


# --- routing --------------------------------------------------------------


class FakeEngine:
    """Records load/unload so eviction can be asserted."""

    instances: list["FakeEngine"] = []

    def __init__(self, tag):
        self.tag = tag
        self.loaded = False
        self.unloaded = False
        FakeEngine.instances.append(self)

    def load(self): self.loaded = True
    def unload(self): self.unloaded = True
    @property
    def resident_bytes(self): return 0


@pytest.fixture
def router():
    FakeEngine.instances.clear()
    return TTSRouter([
        Route(frozenset({"hi", "ta"}), lambda: FakeEngine("indic"),
              normalizer=lambda t: t + " [normalized]", label="indic"),
        Route(frozenset({"en"}), lambda: FakeEngine("kokoro"), label="kokoro"),
    ])


def test_routes_hindi_to_indic_engine(router):
    engine, prepared, lang = router.prepare("नमस्ते")
    assert engine.tag == "indic" and lang == "hi"


def test_routes_english_to_kokoro(router):
    engine, prepared, lang = router.prepare("Hello there")
    assert engine.tag == "kokoro" and lang == "en"


def test_normalizer_applied_only_on_its_route(router):
    _, hindi_text, _ = router.prepare("नमस्ते")
    _, english_text, _ = router.prepare("Hello there")
    assert hindi_text.endswith("[normalized]")
    assert english_text == "Hello there"


def test_switching_language_evicts_the_previous_engine(router):
    """Both engines resident would blow the memory budget."""
    first, _, _ = router.prepare("नमस्ते")
    second, _, _ = router.prepare("Hello there")
    assert first.unloaded is True, "previous engine was not evicted"
    assert second.loaded is True


def test_same_language_does_not_reload(router):
    a, _, _ = router.prepare("नमस्ते")
    b, _, _ = router.prepare("आप कैसे हैं?")
    assert a is b
    assert a.unloaded is False


def test_keep_resident_disables_eviction():
    FakeEngine.instances.clear()
    r = TTSRouter([
        Route(frozenset({"hi"}), lambda: FakeEngine("indic"), label="indic"),
        Route(frozenset({"en"}), lambda: FakeEngine("kokoro"), label="kokoro"),
    ], keep_resident=True)
    first, _, _ = r.prepare("नमस्ते")
    r.prepare("Hello there")
    assert first.unloaded is False


def test_nothing_is_constructed_until_needed(router):
    assert FakeEngine.instances == []
    router.prepare("नमस्ते")
    assert len(FakeEngine.instances) == 1


# --- the Devanagari family ------------------------------------------------
#
# `detect` maps every Devanagari string to "hi", so Marathi and Nepali reach the
# Hindi engine without tripping `_require_hindi` --- they are not refused because
# they are not detected. Measured in `eval/devanagari.py`: both synthesize
# intelligibly and neither is spoken natively, so the honest handling is to name
# the case rather than to refuse it or to stay silent about it.


def test_marathi_and_nepali_still_detect_as_hindi():
    """Not a bug to fix here. Devanagari really is ambiguous, "hi" really is the
    right default, and the Indic route must keep claiming the script (see
    `test_the_router_still_sends_every_indic_script_here`). The fix belongs in a
    note to the caller, not in the script table."""
    assert detect("सकाळी मी शाळेत लवकर पोहोचलो.").language == "hi"
    assert detect("म भोलि काठमाडौं जान्छु.").language == "hi"


def test_marathi_is_named_by_the_one_letter_hindi_does_not_have():
    """`ळ` is not a letter of standard Hindi, which is what makes a one-character
    test precise enough to ship. The warning exists because the model was measured
    unable to produce it --- 0 of 4 seeds."""
    from voiceagent.text.detect import devanagari_language_note

    note = devanagari_language_note("सकाळी मी शाळेत लवकर पोहोचलो.")
    assert note is not None
    assert "Marathi" in note


def test_nepali_is_named_by_function_words():
    from voiceagent.text.detect import devanagari_language_note

    note = devanagari_language_note("तपाईंलाई कस्तो छ?")
    assert note is not None
    assert "Nepali" in note


@pytest.mark.parametrize(
    "text",
    [
        "नमस्ते, आप कैसे हैं?",
        "मैंने अभी email भेज दिया है, please check कर लेना।",
        "उसने किताब पढ़कर खत्म की और फिर वह सो गया.",
        "Hello, how are you?",
        "வணக்கம், எப்படி இருக்கிறீர்கள்?",
    ],
)
def test_ordinary_hindi_is_never_warned_about(text):
    """The check is biased to precision on purpose. Missing a Marathi sentence
    leaves today's behaviour untouched; warning about correct Hindi would train
    users to ignore the warning."""
    from voiceagent.text.detect import devanagari_language_note

    assert devanagari_language_note(text) is None
