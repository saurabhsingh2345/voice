"""Generate local test audio for STT benchmarking.

Uses the macOS ``say`` command so the fixtures are reproducible and nothing has
to be downloaded. Synthetic speech is not a substitute for real-world accuracy
testing, but it is a fair, identical input for comparing two engines' latency.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parents[3] / "fixtures"

#: Target format for every fixture: 16 kHz mono 16-bit PCM, what both engines expect.
SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class Fixture:
    slug: str
    text: str
    """Ground-truth transcript, used to sanity-check engine output."""

    @property
    def path(self) -> Path:
        return FIXTURE_DIR / f"{self.slug}.wav"


FIXTURES: tuple[Fixture, ...] = (
    Fixture(
        "short",
        "What is the weather like today?",
    ),
    Fixture(
        "medium",
        "Set a timer for twenty five minutes and then remind me to review "
        "the quarterly numbers before the meeting.",
    ),
    Fixture(
        "long",
        "I want you to read the configuration file in my projects directory, "
        "summarize the database settings, and tell me whether the connection "
        "pool size looks reasonable for a service handling about two thousand "
        "requests per second. If it looks too small, suggest a better value.",
    ),
    Fixture(
        "technical",
        "Benchmark the quantized model on Apple Silicon and report the "
        "time to first token in milliseconds.",
    ),
)


def generate(force: bool = False) -> list[Fixture]:
    """Render every fixture to a 16 kHz mono WAV. Returns the fixtures."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    for fixture in FIXTURES:
        if fixture.path.exists() and not force:
            continue

        aiff = fixture.path.with_suffix(".aiff")
        subprocess.run(
            ["say", "-o", str(aiff), fixture.text],
            check=True,
            capture_output=True,
        )
        # afconvert is the reliable way to hit exactly 16 kHz mono LEI16.
        subprocess.run(
            [
                "afconvert",
                str(aiff),
                str(fixture.path),
                "-f", "WAVE",
                "-d", f"LEI16@{SAMPLE_RATE}",
                "-c", "1",
            ],
            check=True,
            capture_output=True,
        )
        aiff.unlink()

    return list(FIXTURES)


if __name__ == "__main__":
    import soundfile as sf

    for f in generate(force=True):
        info = sf.info(f.path)
        print(f"{f.slug:10s} {info.duration:5.2f}s  {info.samplerate} Hz  {f.path}")
