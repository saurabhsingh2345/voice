"""Consent-gated, encrypted storage for cloned voice profiles.

Voice is biometric data, and a cloning model turns a few seconds of it into the
ability to make someone appear to say anything. So the consent gate here is
structural, not cosmetic:

  * A profile cannot be written without a ConsentRecord. There is no code path
    that creates one otherwise -- `save()` requires it as an argument.
  * The reference audio is encrypted at rest with a key held in the macOS
    Keychain, never on disk beside the data.
  * Deletion removes the audio, the metadata, and (for a full wipe) the key,
    so remaining ciphertext is unrecoverable rather than merely unreferenced.

None of this leaves the machine.
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from voiceagent import paths

#: Resolved through `voiceagent.paths`, not from `__file__`. Deriving it from
#: the module location put a packaged app's voice profiles inside its own .app
#: bundle -- read-only, and replaced wholesale on every update. Consented
#: recordings are not a cache.
DATA_DIR = paths.data_dir() / "voices"

KEYRING_SERVICE = "voiceagent.voice_clone"
KEYRING_USER = "profile-encryption-key"

#: Consent must be given by typing this exactly. A checkbox is too easy to
#: click past for something this consequential.
CONSENT_PHRASE = "I consent to cloning my voice"

#: Shortest reference clip we will accept. Too little audio produces a poor
#: clone, and encourages re-recording rather than silent low quality.
MIN_REFERENCE_SECONDS = 5.0
MAX_REFERENCE_SECONDS = 60.0


class ConsentError(RuntimeError):
    """Raised when a voice operation is attempted without valid consent."""


@dataclass(frozen=True)
class ConsentRecord:
    """Evidence that a specific person agreed to have their voice cloned."""

    speaker_name: str
    phrase_typed: str
    granted_at: str
    """ISO-8601 UTC timestamp."""
    method: str = "typed-phrase"

    @staticmethod
    def create(speaker_name: str, phrase_typed: str) -> "ConsentRecord":
        if phrase_typed.strip().lower() != CONSENT_PHRASE.lower():
            raise ConsentError(
                f"Consent phrase does not match. Type exactly: {CONSENT_PHRASE!r}"
            )
        if not speaker_name.strip():
            raise ConsentError("A speaker name is required to record consent.")
        return ConsentRecord(
            speaker_name=speaker_name.strip(),
            phrase_typed=phrase_typed.strip(),
            granted_at=datetime.now(timezone.utc).isoformat(),
        )


@dataclass(frozen=True)
class VoiceProfile:
    profile_id: str
    speaker_name: str
    created_at: str
    duration_seconds: float
    sample_rate: int
    consent: ConsentRecord
    reference_text: str = ""
    """What the speaker actually said. IndicF5 conditions on the reference
    transcript as well as the audio, so Hindi synthesis needs it; Chatterbox
    does not, which is why it is optional."""

    @property
    def dir(self) -> Path:
        return DATA_DIR / self.profile_id


# --- encryption -----------------------------------------------------------


def _get_key() -> bytes:
    """Fetch the encryption key from the Keychain, creating it on first use."""
    import keyring
    from cryptography.fernet import Fernet

    existing = keyring.get_password(KEYRING_SERVICE, KEYRING_USER)
    if existing:
        return existing.encode()

    key = Fernet.generate_key()
    keyring.set_password(KEYRING_SERVICE, KEYRING_USER, key.decode())
    return key


def _fernet():
    from cryptography.fernet import Fernet

    return Fernet(_get_key())


def _drop_key() -> None:
    import keyring

    try:
        keyring.delete_password(KEYRING_SERVICE, KEYRING_USER)
    except keyring.errors.PasswordDeleteError:
        pass


# --- store ----------------------------------------------------------------


class VoiceProfileStore:
    def __init__(self, root: Path = DATA_DIR) -> None:
        self.root = root

    def save(self, consent: ConsentRecord, wav_bytes: bytes, duration_seconds: float,
             sample_rate: int, reference_text: str = "") -> VoiceProfile:
        """Persist a reference clip. Impossible to call without consent."""
        if not isinstance(consent, ConsentRecord):
            raise ConsentError("A valid ConsentRecord is required to store a voice.")
        if duration_seconds < MIN_REFERENCE_SECONDS:
            raise ValueError(
                f"Reference clip is {duration_seconds:.1f}s; at least "
                f"{MIN_REFERENCE_SECONDS:.0f}s is needed for a usable clone."
            )

        profile = VoiceProfile(
            profile_id=uuid.uuid4().hex[:12],
            speaker_name=consent.speaker_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            consent=consent,
            reference_text=reference_text.strip(),
        )

        target = self.root / profile.profile_id
        target.mkdir(parents=True, exist_ok=True)
        (target / "reference.wav.enc").write_bytes(_fernet().encrypt(wav_bytes))
        (target / "profile.json").write_text(
            json.dumps(
                {**asdict(profile), "consent": asdict(consent)},
                indent=2,
                default=str,
            )
        )
        return profile

    def list(self) -> list[VoiceProfile]:
        if not self.root.exists():
            return []
        profiles = []
        for meta in sorted(self.root.glob("*/profile.json")):
            data = json.loads(meta.read_text())
            consent = ConsentRecord(**data.pop("consent"))
            data.pop("dir", None)
            # Profiles enrolled before Hindi support have no transcript.
            data.setdefault("reference_text", "")
            profiles.append(VoiceProfile(**data, consent=consent))
        return profiles

    def get(self, profile_id: str) -> VoiceProfile | None:
        return next((p for p in self.list() if p.profile_id == profile_id), None)

    def reference_audio(self, profile_id: str) -> bytes:
        """Decrypt and return the reference clip.

        Held in memory only -- the plaintext is never written back to disk.
        """
        profile = self.get(profile_id)
        if profile is None:
            raise KeyError(f"no such voice profile: {profile_id}")
        blob = (self.root / profile_id / "reference.wav.enc").read_bytes()
        return _fernet().decrypt(blob)

    def set_reference_text(self, profile_id: str, reference_text: str) -> VoiceProfile:
        """Attach a transcript to an existing profile.

        Voices enrolled before Indic support have no transcript, and IndicF5
        needs one. Re-recording a minute of audio just to add a line of text
        would be a poor trade, so this edits the metadata in place. The audio
        and the consent record are untouched.
        """
        profile = self.get(profile_id)
        if profile is None:
            raise KeyError(f"no such voice profile: {profile_id}")

        meta = self.root / profile_id / "profile.json"
        data = json.loads(meta.read_text())
        data["reference_text"] = reference_text.strip()
        meta.write_text(json.dumps(data, indent=2, default=str))
        return self.get(profile_id)

    def delete(self, profile_id: str) -> bool:
        target = self.root / profile_id
        if not target.exists():
            return False
        shutil.rmtree(target)
        return True

    def delete_all(self) -> int:
        """Wipe every profile and destroy the key, per the brief's requirement."""
        count = len(self.list())
        if self.root.exists():
            shutil.rmtree(self.root)
        _drop_key()
        return count
