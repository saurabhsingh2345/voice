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
