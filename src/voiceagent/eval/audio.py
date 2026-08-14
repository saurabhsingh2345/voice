"""Sample-rate conversion shared by the evaluation tools.

One function, in its own module, because the wrong version of it was written twice
independently — once feeding the vocoder control and once feeding Whisper — and both
times as index arithmetic:

    idx = (np.arange(int(len(audio) * target / rate)) * rate / target).astype(int)
    audio = audio[idx[idx < len(audio)]]

That is decimation with no anti-alias filter. Everything above the new Nyquist folds
back into the audible band instead of being removed: an 18 kHz tone resampled 48 →
24 kHz this way survives at full energy as 6 kHz. It is quiet enough to miss by ear
on speech and it corrupts exactly the measurements these tools exist to make, so it
is worth the module.

scipy rather than soxr: scipy is BSD-3-Clause and already required by mlx-whisper
and librosa, so it is present wherever these tools run. libsoxr is LGPL, which the
licence rule in `models.py` does not admit.
"""

from __future__ import annotations

from math import gcd

import numpy as np


def resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Band-limited resample. Returns float32, length scaled by the rate ratio."""
    audio = np.asarray(audio, dtype=np.float32)
    if from_rate == to_rate or audio.size == 0:
        return audio

    from scipy.signal import resample_poly

    divisor = gcd(int(from_rate), int(to_rate))
    out = resample_poly(audio, int(to_rate) // divisor, int(from_rate) // divisor)
    return np.asarray(out, dtype=np.float32)
