"""Audio conditioning for the blind benchmark.

Every condition has to reach the listener differing in exactly the thing being
tested. Loudness is the loudest tell and was handled from the start; sample rate was
not, and the resampler that did exist aliased. Both are cheap to get wrong silently
and impossible to notice in a score, which is what these pin down.
"""

from __future__ import annotations

import numpy as np
import pytest

from voiceagent.eval.build_benchmark import TARGET_RMS, match_loudness, resample

RATE = 48_000


def tone(hz: float, seconds: float = 1.0, rate: int = RATE, amplitude: float = 0.5):
    t = np.linspace(0, seconds, int(seconds * rate), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt((audio.astype(np.float64) ** 2).mean()))


# --- resampling ------------------------------------------------------------


def test_content_above_the_new_nyquist_is_filtered_not_folded_down():
    """The original code took every other sample. An 18 kHz tone survived that at
    full energy, reappearing at 6 kHz — audible aliasing, in the vocoder control
    condition and nowhere else, which is precisely the condition whose whole job is
    to be free of artefacts the other conditions do not have."""
    above_nyquist = tone(18_000)
    out = resample(above_nyquist, 48_000, 24_000)

    dropped = above_nyquist[::2]
    assert rms(dropped) == pytest.approx(rms(above_nyquist), rel=0.05), "the old bug"
    assert rms(out) < rms(above_nyquist) / 100, "at least 40 dB down"


def test_speech_band_content_survives_intact():
    speech = tone(1_000)
    out = resample(speech, 48_000, 24_000)
    assert rms(out) == pytest.approx(rms(speech), rel=0.02)
    spectrum = np.abs(np.fft.rfft(out))
    peak = np.fft.rfftfreq(len(out), 1 / 24_000)[spectrum.argmax()]
    assert peak == pytest.approx(1_000, abs=5)


def test_duration_is_preserved():
    out = resample(tone(1_000, seconds=2.5), 48_000, 24_000)
    assert len(out) == pytest.approx(2.5 * 24_000, rel=0.001)


def test_a_matching_rate_is_a_no_op():
    audio = tone(1_000, rate=24_000)
    assert np.array_equal(resample(audio, 24_000, 24_000), audio)


def test_empty_audio_does_not_raise():
    assert resample(np.zeros(0, dtype=np.float32), 48_000, 24_000).size == 0


def test_odd_rate_pairs_resample_by_ratio_not_by_integer_factor():
    """44.1 kHz is not a multiple of 24 kHz. The gcd reduction is what keeps that
    from silently becoming a wrong-length clip."""
    out = resample(tone(1_000, seconds=1.0, rate=44_100), 44_100, 24_000)
    assert len(out) == pytest.approx(24_000, rel=0.001)


# --- loudness --------------------------------------------------------------


def test_loudness_is_matched_across_conditions():
    """Level is the single loudest tell available to a listener and would otherwise
    swamp the thing being measured."""
    quiet, loud = tone(1_000, amplitude=0.01), tone(1_000, amplitude=0.9)
    assert rms(match_loudness(quiet)) == pytest.approx(TARGET_RMS, rel=0.01)
    assert rms(match_loudness(loud)) == pytest.approx(TARGET_RMS, rel=0.01)


def test_loudness_matching_never_clips():
    peaky = tone(1_000, amplitude=0.02)
    peaky[0] = 0.99
    assert float(np.abs(match_loudness(peaky)).max()) <= 0.99


def test_silence_is_left_alone_rather_than_amplified():
    """Dividing by a near-zero RMS would turn a silent clip into full-scale noise."""
    silence = np.zeros(1000, dtype=np.float32)
    assert float(np.abs(match_loudness(silence)).max()) == 0.0
