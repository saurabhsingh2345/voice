"""Many clips per speaker, for fine-tuning rather than zero-shot cloning.

`VoiceProfileStore` holds ONE reference clip per profile, which is all zero-shot
cloning needs. Zero-shot has a ceiling though: it transfers timbre from a 12-second
prompt and nothing else -- not rhythm, not breath, not the way a particular person
lands on the end of a sentence. Getting past that ceiling means fine-tuning the
model on a real dataset of one voice, and a dataset is what this module stores.

WHERE THE CLIPS LIVE, AND WHY IT MATTERS

Under `data/voices/<profile_id>/clips/`, deliberately inside the profile directory
rather than a sibling `data/datasets/`. Every existing deletion path is a
`shutil.rmtree` of the profile directory or of `data/voices` entirely, so nesting
the dataset means "delete this voice" and "delete all my data" already wipe the
training clips -- including `delete_all()`, which also destroys the Keychain key
and so makes any stray ciphertext unrecoverable.

A sibling directory would have looked tidier and silently survived both. For
biometric data that is the wrong kind of tidy.

CONSENT

A clip cannot be added except against an existing profile, and a profile cannot
exist without a `ConsentRecord` -- `VoiceProfileStore.save()` requires one as an
argument. So the gate is inherited structurally rather than re-implemented, and
there is no path that accumulates training audio for a voice nobody consented to.
"""

from __future__ import annotations

import csv
import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from voiceagent.voice_clone.store import DATA_DIR, ConsentError, VoiceProfileStore, _fernet

#: Fine-tuning targets, in seconds of usable speech.
#:
#: These are guidance for the UI, not enforcement. The honest shape of the curve:
#: below ~10 minutes a fine-tune mostly learns timbre, which zero-shot already
#: gives you, so it is not worth the compute. Around 30 minutes prosody starts to
#: transfer. "Nobody can tell, including my mother" is an hour and up, and depends
#: far more on the *variety* of the speech than the total -- an hour of flat
#: read-aloud sentences teaches the model to sound flat.
TARGET_SECONDS = 1800.0
MINIMUM_USEFUL_SECONDS = 600.0

#: Per-clip bounds. Long clips are the enemy of a good batch: f5-tts trains on
#: frames, so one 60-second clip crowds out many short ones and spikes memory on
#: the machine this has to run on.
MIN_CLIP_SECONDS = 1.5
MAX_CLIP_SECONDS = 20.0


class DatasetError(RuntimeError):
    """Raised when a clip cannot be accepted into the dataset."""


@dataclass(frozen=True)
class Clip:
    """One utterance and what was said in it."""

    clip_id: str
    profile_id: str
    text: str
    duration_seconds: float
    sample_rate: int
    created_at: str
    language: str = "hi"


@dataclass(frozen=True)
class DatasetSummary:
    profile_id: str
    speaker_name: str
    clip_count: int
    total_seconds: float
    target_seconds: float = TARGET_SECONDS

    @property
    def fraction_of_target(self) -> float:
        return min(1.0, self.total_seconds / self.target_seconds) if self.target_seconds else 0.0

    @property
    def usable(self) -> bool:
        """Enough to be worth spending compute on."""
        return self.total_seconds >= MINIMUM_USEFUL_SECONDS


class VoiceDataset:
    """Encrypted, consent-gated training clips for one or more voice profiles."""

    def __init__(self, root: Path = DATA_DIR, profiles: VoiceProfileStore | None = None) -> None:
        self.root = root
        self.profiles = profiles or VoiceProfileStore(root)

    # --- paths ------------------------------------------------------------

    def _dir(self, profile_id: str) -> Path:
        return self.root / profile_id / "clips"

    # --- writing ----------------------------------------------------------

    def add_clip(
        self,
        profile_id: str,
        wav_bytes: bytes,
        text: str,
        duration_seconds: float,
        sample_rate: int,
        language: str = "hi",
    ) -> Clip:
        """Store one clip. Requires a consented profile to attach it to.

        The consent check is a profile lookup rather than a second typed phrase:
        the profile cannot exist without a ConsentRecord, so requiring one here
        inherits that guarantee instead of duplicating a gate that could drift
        from it.
        """
        profile = self.profiles.get(profile_id)
        if profile is None:
            raise ConsentError(
                f"No consented voice profile {profile_id!r}. Training clips can only "
                "be added to a voice that has already been enrolled with consent."
            )

        text = text.strip()
        if not text:
            raise DatasetError(
                "A transcript is required. f5-tts learns the mapping from text to "
                "your voice, so an untranscribed clip teaches it nothing and a "
                "wrongly transcribed one teaches it something false."
            )
        if duration_seconds < MIN_CLIP_SECONDS:
            raise DatasetError(
                f"Clip is {duration_seconds:.1f}s; at least {MIN_CLIP_SECONDS}s is needed."
            )
        if duration_seconds > MAX_CLIP_SECONDS:
            raise DatasetError(
                f"Clip is {duration_seconds:.1f}s; the limit is {MAX_CLIP_SECONDS:.0f}s. "
                "Long clips crowd out short ones in a frame-based batch and spike "
                "memory during training. Split it into sentences."
            )

        clip = Clip(
            clip_id=uuid.uuid4().hex[:12],
            profile_id=profile_id,
            text=text,
            duration_seconds=round(duration_seconds, 3),
            sample_rate=sample_rate,
            created_at=datetime.now(timezone.utc).isoformat(),
            language=language,
        )

        target = self._dir(profile_id)
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{clip.clip_id}.wav.enc").write_bytes(_fernet().encrypt(wav_bytes))
        (target / f"{clip.clip_id}.json").write_text(json.dumps(asdict(clip), indent=2))
        return clip

    def set_text(self, profile_id: str, clip_id: str, text: str) -> Clip:
        """Correct a transcript. The audio is untouched."""
        path = self._dir(profile_id) / f"{clip_id}.json"
        if not path.exists():
            raise KeyError(f"no such clip: {clip_id}")
        if not text.strip():
            raise DatasetError("A transcript is required.")
        data = json.loads(path.read_text())
        data["text"] = text.strip()
        path.write_text(json.dumps(data, indent=2))
        return Clip(**data)

    # --- reading ----------------------------------------------------------

    def clips(self, profile_id: str, language: str | None = None) -> list[Clip]:
        directory = self._dir(profile_id)
        if not directory.exists():
            return []
        out = []
        for meta in sorted(directory.glob("*.json")):
            clip = Clip(**json.loads(meta.read_text()))
            if language is None or clip.language == language:
                out.append(clip)
        return out

    def audio(self, profile_id: str, clip_id: str) -> bytes:
        """Decrypt one clip. Held in memory only."""
        path = self._dir(profile_id) / f"{clip_id}.wav.enc"
        if not path.exists():
            raise KeyError(f"no such clip: {clip_id}")
        return _fernet().decrypt(path.read_bytes())

    def summary(self, profile_id: str, language: str | None = None) -> DatasetSummary:
        profile = self.profiles.get(profile_id)
        clips = self.clips(profile_id, language=language)
        return DatasetSummary(
            profile_id=profile_id,
            speaker_name=profile.speaker_name if profile else "?",
            clip_count=len(clips),
            total_seconds=round(sum(c.duration_seconds for c in clips), 2),
        )

    # --- deleting ---------------------------------------------------------

    def delete_clip(self, profile_id: str, clip_id: str) -> bool:
        directory = self._dir(profile_id)
        found = False
        for suffix in (".wav.enc", ".json"):
            path = directory / f"{clip_id}{suffix}"
            if path.exists():
                path.unlink()
                found = True
        return found

    def delete_all(self, profile_id: str) -> int:
        """Drop every clip for one voice, leaving the profile itself intact."""
        count = len(self.clips(profile_id))
        directory = self._dir(profile_id)
        if directory.exists():
            shutil.rmtree(directory)
        return count

    # --- export -----------------------------------------------------------

    def export(
        self, profile_id: str, dest: Path, language: str | None = None
    ) -> tuple[Path, int, float]:
        """Write a plaintext training set f5-tts can read.

        Produces `dest/wavs/*.wav` and `dest/metadata.csv` in the format
        `prepare_csv_wavs.py` expects: a `audio_file|text` header, pipe delimiter,
        and absolute paths.

        THIS WRITES DECRYPTED AUDIO TO DISK. It has to -- the trainer is a separate
        process that reads files, so there is no way to fine-tune on this data
        without it existing in the clear somewhere. That is a real widening of the
        threat model compared to everything else in this project, so it is explicit
        rather than incidental: the caller chooses the location, and `purge()`
        removes it. Put it somewhere `data/` covers and it stays gitignored.
        """
        import io

        import soundfile as sf

        clips = self.clips(profile_id, language=language)
        if not clips:
            raise DatasetError(f"No clips to export for {profile_id!r}.")

        dest = Path(dest)
        wavs = dest / "wavs"
        wavs.mkdir(parents=True, exist_ok=True)

        rows, total = [], 0.0
        for clip in clips:
            samples, rate = sf.read(io.BytesIO(self.audio(profile_id, clip.clip_id)), dtype="float32")
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            path = wavs / f"{clip.clip_id}.wav"
            sf.write(path, samples, rate, format="WAV", subtype="PCM_16")
            rows.append((str(path.resolve()), clip.text))
            total += clip.duration_seconds

        metadata = dest / "metadata.csv"
        with metadata.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="|", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["audio_file", "text"])
            writer.writerows(rows)

        return metadata, len(rows), round(total, 2)

    @staticmethod
    def purge(dest: Path) -> bool:
        """Remove an exported training set, i.e. the decrypted copy."""
        dest = Path(dest)
        if dest.exists():
            shutil.rmtree(dest)
            return True
        return False
