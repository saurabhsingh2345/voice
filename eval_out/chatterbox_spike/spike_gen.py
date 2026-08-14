"""Phase 0 spike: render the 12 held-out sentences with Chatterbox Multilingual (hi).

Generation only. Scoring is a separate pass so the two models are never
co-resident -- this machine has ~4 GB free.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path("/Users/enfecsolutions/Desktop/self/voice")
sys.path.insert(0, str(ROOT / "src"))

from voiceagent.eval.heldout import SENTENCES  # noqa: E402
from voiceagent.text.translit_en import transliterate  # noqa: E402

REPO = "mlx-community/chatterbox-multilingual-v3"
REF = ROOT / "fixtures" / "hi" / "reference_lekha.wav"
OUT = ROOT / "eval_out" / "chatterbox_spike"

# Praxy Voice's published recipe for Indic Chatterbox (arXiv 2604.25441).
PARAMS = dict(exaggeration=0.7, temperature=0.6, min_p=0.1)

# Sentences containing Latin script -- the code-mixing cases. Rendered twice:
# raw (does the multilingual tokenizer handle Latin natively?) and
# transliterated (what the IndicF5 path had to do).
CODE_MIXED = {"h1", "h3", "h12"}


def main() -> int:
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model

    OUT.mkdir(parents=True, exist_ok=True)
    ref, ref_sr = sf.read(str(REF), dtype="float32")
    if ref.ndim > 1:
        ref = ref.mean(axis=1)
    print(f"reference: {len(ref)/ref_sr:.1f}s @ {ref_sr} Hz")
    # prepare_conditionals feeds this straight into mx ops -- numpy fails there.
    ref = mx.array(ref)

    t0 = time.perf_counter()
    model = load_model(REPO)
    print(f"model loaded in {time.perf_counter()-t0:.1f}s")

    rows = []
    for s in SENTENCES:
        variants = [("raw", s.text)]
        if s.slug in CODE_MIXED:
            variants.append(("translit", transliterate(s.text)))

        for tag, text in variants:
            name = s.slug if tag == "raw" else f"{s.slug}_{tag}"
            t0 = time.perf_counter()
            chunks = []
            rate = 24_000
            for r in model.generate(
                text=text,
                audio_prompt=ref,
                audio_prompt_sr=ref_sr,
                lang_code="hi",
                verbose=False,
                **PARAMS,
            ):
                a = np.asarray(r.audio, dtype=np.float32).reshape(-1)
                chunks.append(a)
                if getattr(r, "sample_rate", None):
                    rate = r.sample_rate
            wall = time.perf_counter() - t0

            audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
            dur = len(audio) / rate
            path = OUT / f"{name}.wav"
            sf.write(str(path), audio, rate)
            rtf = wall / dur if dur else float("nan")
            print(f"{name:12s} {dur:6.2f}s audio  {wall:6.2f}s wall  RTF {rtf:5.2f}")
            rows.append(
                dict(slug=s.slug, variant=tag, file=path.name, text=text,
                     seconds=dur, wall=wall, rtf=rtf, rate=rate)
            )

    (OUT / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    total_audio = sum(r["seconds"] for r in rows if r["variant"] == "raw")
    total_wall = sum(r["wall"] for r in rows if r["variant"] == "raw")
    print(f"\n12 sentences: {total_audio:.1f}s audio in {total_wall:.1f}s "
          f"-> aggregate RTF {total_wall/total_audio:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
