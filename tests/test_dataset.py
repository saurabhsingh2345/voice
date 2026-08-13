"""Properties of the training-clip store.

The consequential ones are about deletion and consent, not about counting clips.
Training audio is more of someone's voice than a single reference clip is, so the
guarantees that hold for the reference have to hold for the dataset too -- and the
easy way to break that is to store it somewhere the existing delete paths do not
reach.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voiceagent.voice_clone.dataset import (
    MAX_CLIP_SECONDS,
    MINIMUM_USEFUL_SECONDS,
    Clip,
    DatasetError,
    VoiceDataset,
)
from voiceagent.voice_clone.store import ConsentError, ConsentRecord, VoiceProfileStore

PHRASE = "I consent to cloning my voice"


def wav_bytes(seconds: float = 3.0, sample_rate: int = 24_000) -> bytes:
    t = np.linspace(0, seconds, int(seconds * sample_rate), endpoint=False)
    tone = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, tone, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture
def store(tmp_path):
    return VoiceProfileStore(root=tmp_path)


@pytest.fixture
def dataset(tmp_path, store):
    return VoiceDataset(root=tmp_path, profiles=store)


@pytest.fixture
def profile(store):
    return store.save(
        ConsentRecord.create("Atul", PHRASE), wav_bytes(6.0), 6.0, 24_000, reference_text="नमस्ते।"
    )


# --- consent --------------------------------------------------------------


def test_a_clip_cannot_be_added_without_a_consented_profile(dataset):
    """Inherited rather than re-implemented: the profile cannot exist without a
    ConsentRecord, so requiring a profile requires consent."""
    with pytest.raises(ConsentError):
        dataset.add_clip("nonexistent", wav_bytes(), "नमस्ते।", 3.0, 24_000)


def test_a_clip_attaches_to_a_consented_profile(dataset, profile):
    clip = dataset.add_clip(profile.profile_id, wav_bytes(), "नमस्ते।", 3.0, 24_000)
    assert isinstance(clip, Clip)
    assert dataset.summary(profile.profile_id).clip_count == 1


# --- deletion reaches the training data -----------------------------------


def test_deleting_the_voice_deletes_its_training_clips(dataset, store, profile):
    """The reason clips live inside the profile directory. A sibling
    data/datasets/ would look tidier and silently survive this."""
    dataset.add_clip(profile.profile_id, wav_bytes(), "एक।", 3.0, 24_000)
    assert dataset.summary(profile.profile_id).clip_count == 1

    store.delete(profile.profile_id)
    assert dataset.clips(profile.profile_id) == []


def test_delete_all_my_data_deletes_every_clip(dataset, store, profile, monkeypatch):
    monkeypatch.setattr("voiceagent.voice_clone.store._drop_key", lambda: None)
    dataset.add_clip(profile.profile_id, wav_bytes(), "एक।", 3.0, 24_000)
    dataset.add_clip(profile.profile_id, wav_bytes(), "दो।", 3.0, 24_000)

    store.delete_all()
    assert dataset.clips(profile.profile_id) == []


def test_clips_are_encrypted_at_rest(dataset, profile, tmp_path):
    raw = wav_bytes()
    clip = dataset.add_clip(profile.profile_id, raw, "नमस्ते।", 3.0, 24_000)
    on_disk = (tmp_path / profile.profile_id / "clips" / f"{clip.clip_id}.wav.enc").read_bytes()
    assert on_disk != raw
    assert b"RIFF" not in on_disk, "the WAV header must not be readable on disk"
    assert dataset.audio(profile.profile_id, clip.clip_id) == raw


def test_dropping_the_dataset_leaves_the_profile(dataset, store, profile):
    dataset.add_clip(profile.profile_id, wav_bytes(), "एक।", 3.0, 24_000)
    assert dataset.delete_all(profile.profile_id) == 1
    assert store.get(profile.profile_id) is not None, "the enrolled voice should survive"


# --- what makes a clip usable ---------------------------------------------


def test_an_untranscribed_clip_is_refused(dataset, profile):
    """f5-tts learns text -> voice. An untranscribed clip teaches nothing; a
    wrongly transcribed one teaches something false."""
    with pytest.raises(DatasetError):
        dataset.add_clip(profile.profile_id, wav_bytes(), "   ", 3.0, 24_000)


def test_a_too_short_clip_is_refused(dataset, profile):
    with pytest.raises(DatasetError):
        dataset.add_clip(profile.profile_id, wav_bytes(0.5), "नमस्ते।", 0.5, 24_000)


def test_a_too_long_clip_is_refused(dataset, profile):
    """Frame-based batching means one long clip crowds out many short ones and
    spikes memory on the machine this has to run on."""
    with pytest.raises(DatasetError) as exc:
        dataset.add_clip(profile.profile_id, wav_bytes(30.0), "नमस्ते।", 30.0, 24_000)
    assert str(int(MAX_CLIP_SECONDS)) in str(exc.value)


def test_a_transcript_can_be_corrected_without_touching_the_audio(dataset, profile):
    raw = wav_bytes()
    clip = dataset.add_clip(profile.profile_id, raw, "गलत", 3.0, 24_000)
    fixed = dataset.set_text(profile.profile_id, clip.clip_id, "सही")
    assert fixed.text == "सही"
    assert dataset.audio(profile.profile_id, clip.clip_id) == raw


# --- progress reporting ---------------------------------------------------


def test_summary_totals_duration_and_reports_usability(dataset, profile):
    for _ in range(4):
        dataset.add_clip(profile.profile_id, wav_bytes(5.0), "वाक्य।", 5.0, 24_000)
    s = dataset.summary(profile.profile_id)
    assert s.total_seconds == pytest.approx(20.0)
    assert s.clip_count == 4
    assert not s.usable, "20s is nowhere near enough to spend training compute on"
    assert 0 < s.fraction_of_target < 1


def test_usable_becomes_true_at_the_minimum(dataset, profile):
    from voiceagent.voice_clone.dataset import DatasetSummary

    assert DatasetSummary("x", "y", 1, MINIMUM_USEFUL_SECONDS).usable
    assert not DatasetSummary("x", "y", 1, MINIMUM_USEFUL_SECONDS - 1).usable


def test_clips_can_be_filtered_by_language(dataset, profile):
    dataset.add_clip(profile.profile_id, wav_bytes(), "नमस्ते।", 3.0, 24_000, language="hi")
    dataset.add_clip(profile.profile_id, wav_bytes(), "Hello.", 3.0, 24_000, language="en")
    assert len(dataset.clips(profile.profile_id)) == 2
    assert len(dataset.clips(profile.profile_id, language="hi")) == 1


# --- export ---------------------------------------------------------------


def test_export_writes_the_format_f5_tts_expects(dataset, profile, tmp_path):
    dataset.add_clip(profile.profile_id, wav_bytes(), "पहला वाक्य।", 3.0, 24_000)
    dataset.add_clip(profile.profile_id, wav_bytes(), "दूसरा वाक्य।", 3.0, 24_000)

    out = tmp_path / "train"
    metadata, count, seconds = dataset.export(profile.profile_id, out)

    assert count == 2 and seconds == pytest.approx(6.0)
    lines = metadata.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "audio_file|text"
    for line in lines[1:]:
        path, text = line.split("|", 1)
        assert path.startswith("/"), "prepare_csv_wavs.py requires absolute paths"
        assert Path(path).exists()
        assert text.strip()


def test_export_refuses_an_empty_dataset(dataset, profile, tmp_path):
    with pytest.raises(DatasetError):
        dataset.export(profile.profile_id, tmp_path / "train")


def test_purge_removes_the_decrypted_copy(dataset, profile, tmp_path):
    """Export necessarily writes plaintext audio -- the trainer is a separate
    process that reads files. Being able to remove it again is the mitigation."""
    dataset.add_clip(profile.profile_id, wav_bytes(), "नमस्ते।", 3.0, 24_000)
    out = tmp_path / "train"
    dataset.export(profile.profile_id, out)
    assert (out / "wavs").exists()

    assert VoiceDataset.purge(out) is True
    assert not out.exists()
    assert VoiceDataset.purge(out) is False


def test_export_can_be_restricted_to_one_language(dataset, profile, tmp_path):
    dataset.add_clip(profile.profile_id, wav_bytes(), "नमस्ते।", 3.0, 24_000, language="hi")
    dataset.add_clip(profile.profile_id, wav_bytes(), "Hello.", 3.0, 24_000, language="en")
    _, count, _ = dataset.export(profile.profile_id, tmp_path / "hi", language="hi")
    assert count == 1


def test_export_does_not_hand_out_the_stock_finetune_command(dataset, profile, tmp_path, monkeypatch):
    """The stock f5-tts CLI builds the model with text_mask_padding=True and
    pe_attn_head=null. IndicF5 needs False and 1, every tensor still loads with the
    wrong values, so it does not error -- it spends hours unlearning the pretrained
    weights. Printing that command in the product was worse than printing nothing."""
    from fastapi.testclient import TestClient

    from voiceagent.web import server as srv

    monkeypatch.setattr(srv, "dataset", dataset)
    dataset.add_clip(profile.profile_id, wav_bytes(), "नमस्ते।", 3.0, 24_000)

    steps = " ".join(
        TestClient(srv.app)
        .post(f"/api/dataset/{profile.profile_id}/export", data={"language": ""})
        .json()["next_steps"]
    )
    assert "voice-train-prep" in steps
    assert "f5-tts_finetune-cli" not in steps, "the stock CLI cannot fine-tune IndicF5 correctly"
    assert "prepare_csv_wavs" not in steps, "its output path depends on the tokenizer; prep prints it"


# --- splitting a long take -------------------------------------------------


def words(*specs):
    """(text, start, end) triples as Whisper's word timestamps."""
    return [{"word": t, "start": s, "end": e} for t, s, e in specs]


def test_a_sentence_end_cuts_once_the_span_is_usable():
    from voiceagent.voice_clone.dataset import plan_segments

    spans = plan_segments(words(("नमस्ते", 0.0, 1.0), ("दोस्तों।", 1.0, 2.0),
                                ("आज", 2.0, 3.0), ("मौसम", 3.0, 4.0), ("अच्छा है।", 4.0, 5.0)))
    assert len(spans) == 2
    assert spans[0][2].endswith("।") and spans[1][2].endswith("।")


def test_commas_and_pauses_cut_because_whisper_writes_them_for_dandas():
    """The measured failure: an 8.26s fixture with three clear Hindi sentences came
    back as one 8.14s clip, because Whisper wrote commas where the speaker said
    dandas. Sentence punctuation alone is not a sufficient boundary rule."""
    from voiceagent.voice_clone.dataset import plan_segments

    only_commas = words(*[(f"शब्द{i}," if i % 4 == 3 else f"शब्द{i}", i * 0.5, i * 0.5 + 0.5)
                          for i in range(20)])
    assert len(plan_segments(only_commas)) > 1


def test_nothing_exceeds_the_batchable_length():
    """A clip longer than batch_size_per_gpu in frames cannot be batched at all, so
    it is skipped data rather than more data."""
    from voiceagent.voice_clone.dataset import MAX_SEGMENT_SECONDS, plan_segments

    # Continuous speech, no punctuation and no pauses at all.
    run = words(*[(f"शब्द{i}", i * 0.4, i * 0.4 + 0.4) for i in range(80)])
    spans = plan_segments(run)
    assert spans
    for begin, finish, _ in spans:
        assert finish - begin <= MAX_SEGMENT_SECONDS + 0.5


def test_a_short_tail_is_dropped_not_saved():
    """A fragment labelled with a fragment of a transcript is exactly the false
    mapping the store refuses when a clip is added by hand."""
    from voiceagent.voice_clone.dataset import plan_segments

    spans = plan_segments(words(("पहला", 0.0, 2.0), ("वाक्य।", 2.0, 4.0), ("अरे", 4.0, 4.3)))
    assert len(spans) == 1, "the 0.3s tail must not become a clip"


def test_no_words_yields_no_spans():
    from voiceagent.voice_clone.dataset import plan_segments

    assert plan_segments([]) == []
    assert plan_segments(words(("  ", 0.0, 1.0))) == []


def test_spans_do_not_overlap_and_stay_in_order():
    from voiceagent.voice_clone.dataset import plan_segments

    spans = plan_segments(words(*[(f"शब्द{i}" + ("।" if i % 5 == 4 else ""), i * 0.6, i * 0.6 + 0.6)
                                  for i in range(30)]))
    assert len(spans) > 2
    for earlier, later in zip(spans, spans[1:]):
        assert earlier[1] <= later[0], "a clip must not include audio from the next one"


def test_the_max_segment_is_shorter_than_the_hand_added_clip_cap():
    """The splitter is bound by what a training batch holds, which is stricter than
    what a person may upload by hand."""
    from voiceagent.voice_clone.dataset import MAX_CLIP_SECONDS, MAX_SEGMENT_SECONDS

    assert MAX_SEGMENT_SECONDS < MAX_CLIP_SECONDS
