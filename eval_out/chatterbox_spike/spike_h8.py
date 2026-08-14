"""Is the h8 failure a flake or systematic? Short utterances, repeated.

h8 ("बिल्कुल, हो जाएगा।") transcribed as Korean at 0% overlap. Generation is
stochastic (temperature 0.6, min_p 0.1) and mlx-audio exposes no seed, so a
single sample cannot tell a degenerate mode from bad luck. Also tries the two
other short sentences and a lower-temperature setting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path("/Users/enfecsolutions/Desktop/self/voice")
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "eval_out" / "chatterbox_spike" / "h8_repeats"
REF = ROOT / "fixtures" / "hi" / "reference_lekha.wav"
REPO = "mlx-community/chatterbox-multilingual-v3"

CASES = [
    ("h8_t06", "बिल्कुल, हो जाएगा।", dict(exaggeration=0.7, temperature=0.6, min_p=0.1)),
    ("h8_t03", "बिल्कुल, हो जाएगा।", dict(exaggeration=0.5, temperature=0.3, min_p=0.1)),
    ("h7_t06", "अरे! यह तो मैंने सोचा भी नहीं था।",
     dict(exaggeration=0.7, temperature=0.6, min_p=0.1)),
]
REPEATS = 5


def main() -> int:
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model

    OUT.mkdir(parents=True, exist_ok=True)
    ref, ref_sr = sf.read(str(REF), dtype="float32")
    if ref.ndim > 1:
        ref = ref.mean(axis=1)
    ref = mx.array(ref)

    model = load_model(REPO)
    paths = []
    for name, text, params in CASES:
        for i in range(REPEATS):
            chunks, rate = [], 24_000
            for r in model.generate(text=text, audio_prompt=ref, audio_prompt_sr=ref_sr,
                                    lang_code="hi", verbose=False, **params):
                chunks.append(np.asarray(r.audio, dtype=np.float32).reshape(-1))
                if getattr(r, "sample_rate", None):
                    rate = r.sample_rate
            audio = np.concatenate(chunks)
            p = OUT / f"{name}_{i}.wav"
            sf.write(str(p), audio, rate)
            paths.append((name, text, p, len(audio) / rate))
            print(f"{name}_{i}: {len(audio)/rate:.2f}s")

    del model
    mx.clear_cache()

    from voiceagent.eval.roundtrip import (
        EQUIVALENT_LANGUAGES, character_overlap, normalized, transcribe,
    )
    print()
    for name, text, p, dur in paths:
        heard, lang = transcribe(p)
        if lang in EQUIVALENT_LANGUAGES.get("hi", frozenset()) and lang != "hi":
            heard, _ = transcribe(p, language="hi")
            lang = "hi"
        ov = character_overlap(normalized(text, "hi"), normalized(heard, "hi"))
        flag = "" if ov >= 0.7 else "   <-- FAIL"
        print(f"{p.stem:12s} {ov:5.0%} [{lang:2s}] {dur:5.2f}s  {heard[:48]!r}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
