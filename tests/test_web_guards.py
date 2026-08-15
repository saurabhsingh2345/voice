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


def test_a_request_past_capacity_is_refused_with_a_wait():
    """Queued now, not refused outright --- but still capped.

    The old behaviour was a flat 429 the moment anything was running, which is
    right for a developer tool and reads as broken to someone who paid. What had
    to survive the change is the cap: an unbounded line on a machine that serves
    one at a time is a lie with a spinner on it.
    """
    async def fill_the_queue():
        held = []
        for _ in range(srv.synth_queue._max_waiting + 1):
            slot = srv.synth_queue.slot()
            await slot.__aenter__()
            held.append(slot)
            break  # one running is enough; the rest are counted as waiting
        srv.synth_queue._waiting = srv.synth_queue._max_waiting
        try:
            return client.post(
                "/api/speak", data={"text": "नमस्ते।", "profile_id": "any"}
            )
        finally:
            srv.synth_queue._waiting = 0
            for slot in held:
                await slot.__aexit__(None, None, None)

    r = asyncio.run(fill_the_queue())
    assert r.status_code == 503
    assert "retry-after" in {k.lower() for k in r.headers}
    assert "waiting" in r.json()["detail"].lower()


def test_the_queue_is_released_afterwards():
    """A guard that leaks the slot would stall every later request."""
    assert not srv.synth_queue.running
    client.post("/api/speak", data={"text": "नमस्ते।", "profile_id": "nope"})
    assert not srv.synth_queue.running
    assert srv.synth_queue.waiting == 0


def test_the_queue_endpoint_reports_shape_without_identities():
    """Served to an unauthenticated page, so it must not leak who is generating."""
    body = client.get("/api/queue").json()
    assert set(body) >= {"running", "waiting", "depth", "capacity", "accepting"}
    assert not any("account" in k or "voice" in k for k in body)


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


def test_encode_does_not_wait_behind_the_synthesis_queue():
    """Encoding is milliseconds of CPU; making it wait behind a 40s synthesis
    would reintroduce the stall for no reason."""
    async def hold():
        slot = srv.synth_queue.slot()
        await slot.__aenter__()
        try:
            return client.post(
                "/api/encode",
                files={"audio": ("a.wav", wav_bytes(), "audio/wav")},
                data={"format": "flac"},
            )
        finally:
            await slot.__aexit__(None, None, None)

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
    """Chatterbox Multilingual measures 3.04 GiB after load and 5.09 GiB peak
    during generation, and needs headroom for activations on top; a threshold
    below that would let the wedge happen again. Raised from 2.0 when the Hindi
    path moved off IndicF5, whose checkpoint was less than half the size."""
    assert srv.MIN_FREE_GIB_FOR_INDIC >= 3.0


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
    GiB free -- because the requirement scaled on total length while synthesis
    had moved to one call per sentence. Nothing here needs more than the floor.
    """
    paragraph = (
        "आज मौसम बहुत सुहावना है। आसमान बिल्कुल साफ़ है। मैं आपकी मदद के लिए यहाँ हूँ। "
        "पेड़ हमें हवा देते हैं। हमें पर्यावरण का ध्यान रखना चाहिए। यह एक अच्छा दिन है।"
    )
    assert len(paragraph) > 150, "keep this representative of a real paragraph"
    assert srv.required_free_gib(paragraph) == srv.MIN_FREE_GIB_FOR_INDIC


def test_one_very_long_sentence_still_raises_the_requirement():
    """The guard must not become uniformly permissive -- a single long sentence is
    one large call and genuinely does need more headroom."""
    long_sentence = "क " * 400 + "।"
    assert srv.required_free_gib(long_sentence) > srv.MIN_FREE_GIB_FOR_INDIC


def test_the_memory_guard_has_exactly_one_definition():
    """These were two copies that diverged within a day: `web/tts_service.py` was
    corrected for per-sentence synthesis and `web/server.py` was not, so the same
    text was refused locally at 4.0 GiB while the service would have run it at
    2.5 GiB.

    The service is gone -- it existed to keep f5-tts off this machine, and f5-tts
    is gone too -- so the duplication cannot recur the same way. This now asserts
    the weaker, still-useful property: the server does not define its own copy,
    it re-exports the engine's.
    """
    from voiceagent.tts import chatterbox_indic

    assert srv.required_free_gib is chatterbox_indic.required_free_gib
    assert srv.MIN_FREE_GIB_FOR_INDIC == chatterbox_indic.MIN_FREE_GIB


def test_swap_percentage_is_not_used_as_a_guard():
    """Tried, and it was wrong. macOS sizes its swap file to demand, so it read
    92% full with 113 GiB free on disk while the machine was healthy and
    `memory_pressure` reported 74% free. Guarding on it refused work that would
    have succeeded."""
    assert not hasattr(srv, "MAX_SWAP_FRACTION")


# --- there are no longer per-voice weights --------------------------------


def test_there_is_no_per_voice_checkpoint_lookup():
    """IndicF5 was fine-tunable per voice, so the server looked up
    `data/f5tts_ckpts/<profile>/model_last.pt` and reloaded the engine whenever
    the resident weights belonged to a different speaker -- a ~15 s cost every
    time a caller alternated between a trained voice and a stock one.

    Chatterbox Multilingual clones zero-shot from the reference clip: one
    checkpoint serves every voice. The lookup, the reload and the whole
    `f5tts_ckpts` directory convention are gone. If a per-voice checkpoint ever
    comes back, this test should fail and `_ensure_indic` needs its eviction
    logic back with it.
    """
    assert not hasattr(srv, "indic_checkpoint_for")
    assert not hasattr(srv, "_indic_checkpoint")


def test_the_response_still_declares_which_engine_answered():
    """X-Weights used to distinguish a fine-tuned generation from a stock one,
    which mattered because the whole point of training was that they differ.
    There is nothing to distinguish now, so it reports "stock" unconditionally
    -- kept rather than dropped so existing clients do not see the header vanish.

    X-Engine is the header that still carries information, and it has to name
    the right one of the two Chatterbox checkpoints.
    """
    import inspect

    source = inspect.getsource(srv.speak)
    assert '"X-Weights"' in source
    assert "chatterbox-multilingual" in source and "chatterbox-turbo" in source


# --- 6. output formats ----------------------------------------------------


def test_mp3_is_offered():
    """What people expect to receive, and ~25x smaller than the WAV -- which
    matters over a home tunnel far more than it does in a datacentre."""
    assert "mp3" in srv.OUTPUT_FORMATS
    assert srv.OUTPUT_FORMATS["mp3"][2] == "audio/mpeg"


def test_only_the_mp3_layer_that_actually_encodes_is_offered():
    """libsndfile advertises Layers I and II and raises 'unimplemented format'
    on both. Offering them would be a 500 waiting for whoever tried."""
    assert srv.OUTPUT_FORMATS["mp3"][1] == "MPEG_LAYER_III"


def test_every_declared_format_can_actually_be_written():
    """The registry is the contract /api/config publishes, so a format listed
    there and unwritable is a promise broken at the last step of a request the
    caller already waited for."""
    import numpy as np

    audio = (0.1 * np.sin(np.arange(24_000) * 0.05)).astype("float32")
    for name in srv.OUTPUT_FORMATS:
        payload, media_type, ext = srv._encode(audio, 24_000, name)
        assert payload, f"{name} produced no bytes"
        assert media_type.startswith("audio/")


def test_mp3_is_much_smaller_than_wav():
    import numpy as np

    audio = (0.1 * np.sin(np.arange(24_000 * 2) * 0.05)).astype("float32")
    mp3, _, _ = srv._encode(audio, 24_000, "mp3")
    wav, _, _ = srv._encode(audio, 24_000, "wav")
    assert len(mp3) < len(wav) / 5


def test_an_unknown_format_is_refused_before_synthesis():
    """Rejecting after a 30s generation wastes all of it."""
    r = client.post(
        "/api/speak", data={"text": "नमस्ते।", "profile_id": "x", "format": "aiff"}
    )
    assert r.status_code == 400
