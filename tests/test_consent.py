"""Tests that the voice-cloning consent gate is structural, not cosmetic.

The claim being tested is that there is no path to a stored voice profile, or to
synthesized speech, that bypasses a recorded ConsentRecord.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from voiceagent.voice_clone import store as S


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    """A store rooted in a temp dir, with the Keychain stubbed out."""
    key_slot: dict[str, str] = {}

    def fake_get(service, user):
        return key_slot.get(f"{service}/{user}")

    def fake_set(service, user, value):
        key_slot[f"{service}/{user}"] = value

    def fake_delete(service, user):
        key_slot.pop(f"{service}/{user}", None)

    import keyring

    monkeypatch.setattr(keyring, "get_password", fake_get)
    monkeypatch.setattr(keyring, "set_password", fake_set)
    monkeypatch.setattr(keyring, "delete_password", fake_delete)
    return S.VoiceProfileStore(root=tmp_path / "voices")


def wav_bytes(seconds: float = 8.0, sr: int = 24000) -> bytes:
    t = np.linspace(0, seconds, int(seconds * sr), dtype=np.float32)
    audio = 0.1 * np.sin(2 * np.pi * 140 * t)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def valid_consent() -> S.ConsentRecord:
    return S.ConsentRecord.create("Alex", S.CONSENT_PHRASE)


# --- consent construction -------------------------------------------------


def test_correct_phrase_grants_consent():
    record = valid_consent()
    assert record.speaker_name == "Alex"
    assert record.granted_at.endswith("+00:00")


@pytest.mark.parametrize(
    "phrase",
    ["", "yes", "i consent", "I consent to cloning my voices", "I CONSENT TO CLONING"],
)
def test_wrong_phrase_is_refused(phrase):
    with pytest.raises(S.ConsentError):
        S.ConsentRecord.create("Alex", phrase)


def test_phrase_is_case_insensitive_but_exact():
    assert S.ConsentRecord.create("Alex", S.CONSENT_PHRASE.upper())


def test_anonymous_consent_is_refused():
    with pytest.raises(S.ConsentError):
        S.ConsentRecord.create("   ", S.CONSENT_PHRASE)


# --- storage gate ---------------------------------------------------------


def test_save_requires_a_real_consent_record(tmp_store):
    """Passing something consent-shaped must not work."""
    class Faker:
        speaker_name = "Alex"
        granted_at = "now"

    with pytest.raises(S.ConsentError):
        tmp_store.save(Faker(), wav_bytes(), 8.0, 24000)
    with pytest.raises(S.ConsentError):
        tmp_store.save(None, wav_bytes(), 8.0, 24000)


def test_short_clips_are_refused(tmp_store):
    with pytest.raises(ValueError, match="at least"):
        tmp_store.save(valid_consent(), wav_bytes(2.0), 2.0, 24000)


def test_roundtrip_and_encryption_at_rest(tmp_store):
    raw = wav_bytes()
    profile = tmp_store.save(valid_consent(), raw, 8.0, 24000)

    on_disk = (tmp_store.root / profile.profile_id / "reference.wav.enc").read_bytes()
    assert on_disk != raw, "reference audio must not be stored in plaintext"

    # A wav file *begins* with RIFF. Asserting the four bytes are absent
    # anywhere was flaky at ~4 %, and measurably so: Fernet output is urlsafe
    # base64, "RIFF" is four characters of that alphabet, and a 512 KB blob has
    # roughly len/64^4 chances to contain it. A security assertion that cries
    # wolf once in twenty-five runs is worse than no assertion, because the way
    # people make it stop is to stop reading it.
    assert not on_disk.startswith(b"RIFF"), "stored file is a plaintext wav"
    assert raw not in on_disk, "plaintext audio is embedded in the stored file"

    assert tmp_store.reference_audio(profile.profile_id) == raw


def test_consent_is_persisted_with_the_profile(tmp_store):
    profile = tmp_store.save(valid_consent(), wav_bytes(), 8.0, 24000)
    reloaded = tmp_store.get(profile.profile_id)
    assert reloaded.consent.speaker_name == "Alex"
    assert reloaded.consent.phrase_typed == S.CONSENT_PHRASE


# --- deletion -------------------------------------------------------------


def test_delete_removes_profile(tmp_store):
    profile = tmp_store.save(valid_consent(), wav_bytes(), 8.0, 24000)
    assert tmp_store.delete(profile.profile_id) is True
    assert tmp_store.get(profile.profile_id) is None
    assert not (tmp_store.root / profile.profile_id).exists()


def test_delete_all_wipes_everything_and_the_key(tmp_store):
    for _ in range(3):
        tmp_store.save(valid_consent(), wav_bytes(), 8.0, 24000)
    assert len(tmp_store.list()) == 3

    assert tmp_store.delete_all() == 3
    assert tmp_store.list() == []
    assert not tmp_store.root.exists()

    # The key is destroyed too, so any stray ciphertext is unrecoverable.
    import keyring

    assert keyring.get_password(S.KEYRING_SERVICE, S.KEYRING_USER) is None


def test_reference_for_unknown_profile_raises(tmp_store):
    with pytest.raises(KeyError):
        tmp_store.reference_audio("does-not-exist")
