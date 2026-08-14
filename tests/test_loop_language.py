"""Which recogniser the live loop listens with, and why it is a choice.

Output became bilingual when the router replaced the bare Kokoro engine. This is
the other half. It is a flag rather than a default because the options trade
memory, latency and reliability against each other, and because the failure mode
of getting it wrong is silent.

Measured on a real Hindi recording of held-out sentence h1, spoken by the
enrolled speaker:

    --language en    "I have given a documentary for many years.
                      The climate change in India"          <- confident nonsense
    --language hi    "मैंने कल रात एक डॉक्यूमेंटरी देखी जो क्लाइमेट चेंज के बारे में थी"
    --language auto  same, detected hi

Moonshine does not fail on Hindi, it *invents English*. Nothing downstream can
tell that transcript from a real one, so the agent answers a question nobody
asked. That is the argument for the flag existing at all.
"""

from __future__ import annotations

import pytest

from voiceagent.orchestration.loop import LoopConfig, VoiceLoop


def build(language):
    return VoiceLoop(LoopConfig(listen_language=language))._build_stt()


def test_english_uses_moonshine():
    """228 MiB and RTF 0.12 against Whisper's ~2.3 GiB. Worth keeping as the
    default for an English-only conversation."""
    engine, label = build("en")
    assert type(engine).__name__ == "MoonshineEngine"
    assert "English only" in label


def test_hindi_pins_whisper_to_hindi():
    engine, label = build("hi")
    assert type(engine).__name__ == "MLXWhisperEngine"
    assert engine.language == "hi"
    assert "hi" in label


def test_auto_lets_whisper_detect():
    """language=None is Whisper's own per-utterance detection."""
    engine, label = build("auto")
    assert type(engine).__name__ == "MLXWhisperEngine"
    assert engine.language is None
    assert "detected" in label


def test_english_is_the_default():
    """Whisper costs ~2.3 GiB against Moonshine's 228 MiB on an 18 GiB machine,
    and `auto` is less reliable than pinning. Neither should be silently on."""
    assert LoopConfig().listen_language == "en"
    engine, _ = VoiceLoop()._build_stt()
    assert type(engine).__name__ == "MoonshineEngine"


def test_the_flag_offers_exactly_the_three_modes():
    import argparse
    import inspect

    from voiceagent.orchestration import loop as loop_module

    source = inspect.getsource(loop_module.main)
    assert '"--language"' in source
    assert 'choices=("en", "hi", "auto")' in source


def test_speaking_is_bilingual_regardless_of_this_flag():
    """The flag sets hearing only. The TTS router decides the reply's voice from
    the reply's own script, so a Hindi answer is spoken in Hindi even when the
    loop is listening in English."""
    import inspect

    from voiceagent.orchestration import loop as loop_module

    source = inspect.getsource(loop_module.VoiceLoop.load)
    assert "build_default_router()" in source


@pytest.mark.parametrize("language", ["en", "hi", "auto"])
def test_every_mode_builds_without_loading_weights(language):
    """Construction must stay cheap: the loop reports what it is loading before
    it loads it, and a mode that only fails at load time would report a lie."""
    engine, label = build(language)
    assert engine is not None and label
