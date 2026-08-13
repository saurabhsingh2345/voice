"""Guards that keep one slow request from wedging the machine.

These cover a real incident, not a hypothetical. A pasted Hindi essay was
synthesized while a second request was already in flight, on an 18 GiB machine
with ~1.5 GiB free. Both needed the single shared model, both thrashed in swap,
and neither finished: the process sat in uninterruptible I/O wait at 0.3% CPU
with its RSS *shrinking* (94 -> 63 MiB, the model paging out), 68 MiB free
against 14.9 GiB of swap, no log output, no error.

Three things caused it and each is asserted here:

  1. Synthesis was not serialized, though both engines are single shared mutable
     objects -- the Indic path calls set_reference() on the shared instance.
  2. The Download button re-POSTed to /api/speak for any non-WebM format, so a
     download was a whole second synthesis, with its own independent
     button-disable. That is how two got in flight.
  3. There was no cap on input length, so a paste became several minutes of
     compute producing nothing until it finished.

No model is loaded: every path here is rejected before synthesis, or is pure
CPU encoding.
"""

from __future__ import annotations

import asyncio
import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from voiceagent.web import server as srv

client = TestClient(srv.app)


def wav_bytes(seconds: float = 0.5, sample_rate: int = 24_000) -> bytes:
    t = np.linspace(0, seconds, int(seconds * sample_rate), endpoint=False)
    tone = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, tone, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# --- 1. input length ------------------------------------------------------


def test_oversized_text_is_refused_before_any_synthesis():
    too_long = "यह एक परीक्षण वाक्य है। " * 200
    assert len(too_long) > srv.MAX_SPEAK_CHARS
    r = client.post("/api/speak", data={"text": too_long, "profile_id": "x"})
    assert r.status_code == 413
    # The message has to say what to do, not just that it failed.
    assert str(srv.MAX_SPEAK_CHARS) in r.json()["detail"]


def test_text_within_the_cap_is_not_rejected_for_length():
    r = client.post("/api/speak", data={"text": "नमस्ते।", "profile_id": "nope"})
    assert r.status_code != 413


def test_a_narration_length_paragraph_is_allowed():
    """Guards a regression I actually shipped. The cap was briefly 800, which
    refused narration-length text outright -- turning a slow-but-working feature
    into a hard error. The hang it was meant to prevent came from concurrency,
    not length.

    Built from the project's own test sentences rather than pasted user text, so
    no one's recorded or dictated content ends up in the repository.
    """
    from voiceagent.eval import sentences as S

    narration = " ".join(s.text for s in S.FORMAL + S.COLLOQUIAL)
    assert len(narration) > 400, "keep this representative of a real narration"
    r = client.post("/api/speak", data={"text": narration, "profile_id": "nope"})
    assert r.status_code != 413, f"narration of {len(narration)} chars was refused"


def test_the_cap_leaves_room_for_a_multi_paragraph_narration():
    assert srv.MAX_SPEAK_CHARS >= 2000


def test_empty_text_is_refused():
    assert client.post("/api/speak", data={"text": "  ", "profile_id": "x"}).status_code == 400


# --- 2. concurrency -------------------------------------------------------


def test_a_second_request_is_refused_while_one_is_running():
    """Refused, not queued. Queueing is what let clicks pile up and made the
    machine slower rather than the answer sooner."""
    async def hold():
        async with srv._synth_lock:
            return client.post(
                "/api/speak", data={"text": "नमस्ते।", "profile_id": "any"}
            )

    r = asyncio.run(hold())
    assert r.status_code == 429
    assert "at a time" in r.json()["detail"].lower()


def test_the_lock_is_released_afterwards():
    """A guard that leaks the lock would refuse every later request."""
    assert not srv._synth_lock.locked()
    client.post("/api/speak", data={"text": "नमस्ते।", "profile_id": "nope"})
    assert not srv._synth_lock.locked()


def test_format_is_validated_before_the_busy_check_wastes_a_slot():
    r = client.post(
        "/api/speak", data={"text": "नमस्ते।", "profile_id": "x", "format": "mp3x"}
    )
    assert r.status_code == 400
    assert "mp3x" in r.json()["detail"]


# --- 3. downloads must not re-synthesize ----------------------------------


@pytest.mark.parametrize("fmt,magic", [("flac", b"fLaC"), ("opus", b"OggS")])
def test_encode_reencodes_without_synthesizing(fmt, magic):
    r = client.post(
        "/api/encode", files={"audio": ("a.wav", wav_bytes(), "audio/wav")}, data={"format": fmt}
    )
    assert r.status_code == 200, r.text
    assert r.content.startswith(magic)
    assert r.headers["X-Audio-Format"] == fmt


def test_encode_preserves_duration():
    r = client.post(
        "/api/encode",
        files={"audio": ("a.wav", wav_bytes(seconds=1.5), "audio/wav")},
        data={"format": "flac"},
    )
    assert float(r.headers["X-Audio-Seconds"]) == pytest.approx(1.5, abs=0.05)


def test_encode_rejects_a_bad_format():
    r = client.post(
        "/api/encode", files={"audio": ("a.wav", wav_bytes(), "audio/wav")}, data={"format": "mp3x"}
    )
    assert r.status_code == 400


def test_encode_rejects_undecodable_audio():
    r = client.post(
        "/api/encode", files={"audio": ("a.wav", b"not audio", "audio/wav")}, data={"format": "flac"}
    )
    assert r.status_code == 400


def test_encode_does_not_take_the_synthesis_lock():
    """Encoding is milliseconds of CPU; making it wait behind a 40s synthesis
    would reintroduce the stall for no reason."""
    async def hold():
        async with srv._synth_lock:
            return client.post(
                "/api/encode",
                files={"audio": ("a.wav", wav_bytes(), "audio/wav")},
                data={"format": "flac"},
            )

    assert asyncio.run(hold()).status_code == 200


def test_the_ui_download_path_no_longer_posts_to_speak():
    """The bug was in the page, so assert on the page.

    Matches the fetch call rather than the bare path: the fix left a comment
    explaining the old behaviour, and a plain substring search matches its own
    explanation.
    """
    import re
    from pathlib import Path

    page = (Path(srv.__file__).parent / "static" / "index.html").read_text()
    download = page[page.index('$("download").onclick'):]
    download = download[: download.index("\n};")]
    # Strip // comments so prose about the old bug cannot satisfy or break this.
    code = re.sub(r"//[^\n]*", "", download)
    assert 'fetch("/api/encode"' in code
    assert "/api/speak" not in code


# --- memory guard ---------------------------------------------------------


def test_memory_threshold_is_above_the_model_size():
    """IndicF5 is ~1.4 GiB and needs headroom for activations on top; a
    threshold below that would let the wedge happen again."""
    assert srv.MIN_FREE_GIB_FOR_INDIC >= 2.0


def test_memory_requirement_scales_with_text_length():
    """A short sentence and a five-batch narration do not need the same headroom.

    This is why one succeeded and the other was killed with the same memory free.
    """
    assert srv.CHARS_PER_BATCH > 0 and srv.GIB_PER_EXTRA_BATCH > 0
    one = srv.MIN_FREE_GIB_FOR_INDIC
    five = srv.MIN_FREE_GIB_FOR_INDIC + srv.GIB_PER_EXTRA_BATCH * 4
    assert five > one


def test_an_ordinary_hindi_paragraph_needs_only_the_floor():
    """The bug this guards: a 200-character paragraph of short Hindi sentences was
    refused at 3.0 GiB, and a 350-character one at 4.0 GiB, on a machine with 3.7
    GiB free -- because the requirement scaled on total length while synthesis had
    moved to one f5-tts call per sentence. Nothing here needs more than the floor.
    """
    paragraph = (
        "आज मौसम बहुत सुहावना है। आसमान बिल्कुल साफ़ है। मैं आपकी मदद के लिए यहाँ हूँ। "
        "पेड़ हमें हवा देते हैं। हमें पर्यावरण का ध्यान रखना चाहिए। यह एक अच्छा दिन है।"
    )
    assert len(paragraph) > 150, "keep this representative of a real paragraph"
    assert srv.required_free_gib(paragraph) == srv.MIN_FREE_GIB_FOR_INDIC


def test_one_very_long_sentence_still_raises_the_requirement():
    """The guard must not become uniformly permissive -- a single long sentence is
    one large f5-tts call and genuinely does need more headroom."""
    long_sentence = "क " * 400 + "।"
    assert srv.required_free_gib(long_sentence) > srv.MIN_FREE_GIB_FOR_INDIC


def test_the_web_server_and_the_service_share_one_guard():
    """These were two copies that diverged within a day: the service was corrected
    for per-sentence synthesis and the web server was not, so the same text was
    refused locally at 4.0 GiB while the service would have run it at 2.5 GiB."""
    from voiceagent.web import tts_service as service

    assert srv.required_free_gib is service._required_gib
    assert srv.MIN_FREE_GIB_FOR_INDIC == service.MIN_FREE_GIB_FOR_INDIC


def test_swap_percentage_is_not_used_as_a_guard():
    """Tried, and it was wrong. macOS sizes its swap file to demand, so it read
    92% full with 113 GiB free on disk while the machine was healthy and
    `memory_pressure` reported 74% free. Guarding on it refused work that would
    have succeeded."""
    assert not hasattr(srv, "MAX_SWAP_FRACTION")


# --- fine-tuned weights are selected per voice ----------------------------


def test_a_voice_with_no_finetune_gets_stock_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(srv.dataset, "root", tmp_path / "voices")
    assert srv.indic_checkpoint_for("never-trained") is None


def test_a_voice_with_a_finetune_gets_its_own_weights(tmp_path, monkeypatch):
    """Looked up by profile id rather than configured globally: enrol two voices,
    train one, and the trained one should use its own weights without anybody
    selecting anything."""
    monkeypatch.setattr(srv.dataset, "root", tmp_path / "voices")
    ckpt = tmp_path / "f5tts_ckpts" / "trained" / "model_last.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"weights")
    assert srv.indic_checkpoint_for("trained") == ckpt


def test_model_last_is_preferred_over_numbered_checkpoints(tmp_path, monkeypatch):
    """The trainer writes model_last.pt every `last_per_updates`, so it is the most
    recent state; a numbered checkpoint can be hundreds of updates behind."""
    monkeypatch.setattr(srv.dataset, "root", tmp_path / "voices")
    d = tmp_path / "f5tts_ckpts" / "trained"
    d.mkdir(parents=True)
    (d / "model_200.pt").write_bytes(b"old")
    (d / "model_last.pt").write_bytes(b"new")
    assert srv.indic_checkpoint_for("trained").name == "model_last.pt"


def test_the_response_says_which_weights_answered():
    """Without this there is no way to tell a fine-tuned generation from a stock one
    by listening, and the whole point of training is that they differ."""
    import inspect

    source = inspect.getsource(srv.speak)
    assert '"X-Weights"' in source
    assert "fine-tuned" in source and "stock" in source
