"""STT must report which language it heard, and the web API must not call
methods that only exist on a different engine.

Both of these are about a failure that looks like success. A transcript with no
language attached gets routed to whichever TTS engine happens to be default, and
neither engine degrades gracefully: Kokoro cannot say Hindi at all in this build
and IndicF5 cannot pronounce English orthography.

These tests deliberately do not load any model. They assert the contract around
the engines, which is where the bugs were.
"""

from __future__ import annotations

import inspect

from voiceagent.stt.base import Transcript
from voiceagent.stt.mlx_whisper_engine import MLXWhisperEngine
from voiceagent.stt.moonshine_engine import LANGUAGE as MOONSHINE_LANGUAGE
from voiceagent.stt.moonshine_engine import MoonshineEngine


def test_transcript_carries_a_language():
    t = Transcript(text="नमस्ते", is_final=True, language="hi")
    assert t.language == "hi"


def test_language_defaults_to_none_for_engines_that_cannot_detect():
    assert Transcript(text="hi", is_final=True).language is None


def test_moonshine_declares_english():
    """Moonshine does not refuse Hindi audio -- it invents English words for it.

    Measured: on a real Hindi clip it returned "In Namaste, my name is Lekh. I am
    your Sahay Takeli..." and continued into an invented passage, at 121% CER.
    Declaring "en" is what lets a caller notice the mismatch, because the text
    itself looks like a perfectly good transcription.
    """
    assert MOONSHINE_LANGUAGE == "en"
    assert MoonshineEngine().language == "en"


def test_whisper_accepts_none_for_auto_detection():
    """None must be a legal language, since a bilingual loop cannot pin one."""
    assert MLXWhisperEngine(language=None).language is None
    signature = inspect.signature(MLXWhisperEngine.__init__)
    annotation = signature.parameters["language"].annotation
    assert "None" in str(annotation), annotation


def test_whisper_defaults_to_english():
    """Pinning is the default because detection costs a second pass: measured
    RTF 0.432 auto-detect vs 0.236 pinned, for identical 4.8% CER."""
    assert MLXWhisperEngine().language == "en"


def test_indic_engine_has_no_forget_reference():
    """Guards a real bug in web/server.py.

    The server used to call `_indic_engine.forget_reference(profile_id)` when a
    reference transcript was corrected. That method exists only on the Chatterbox
    VoiceCloneEngine, so once Hindi synthesis had loaded the Indic engine, fixing
    a transcript raised AttributeError -- on precisely the path that the
    output-length fix depends on.

    If IndicTTSEngine ever *gains* a per-profile cache, this test should fail and
    the server should start invalidating it again. It is asserting that the two
    engines have genuinely different contracts, not that the method is unwanted.
    """
    from voiceagent.tts.indic_engine import IndicTTSEngine
    from voiceagent.voice_clone.engine import ChatterboxCloneEngine

    assert not hasattr(IndicTTSEngine, "forget_reference")
    assert hasattr(ChatterboxCloneEngine, "forget_reference")


def test_server_does_not_call_forget_reference_on_the_indic_engine():
    """The source-level assertion, since exercising the route needs a model.

    Comments are stripped first: the fix left a comment naming the old call, and
    a naive substring search matches its own explanation.
    """
    import ast
    from pathlib import Path

    import voiceagent.web.server as server_module

    tree = ast.parse(Path(server_module.__file__).read_text())
    calls = {
        f"{node.func.value.id}.{node.func.attr}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
    }
    assert "_indic_engine.forget_reference" not in calls
