"""Round-trip score the spike against the IndicF5 baseline and the human ceiling.

Uses the project's own scorer -- same normalization, same ur->hi re-decode --
so the numbers are comparable to everything already in the README.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path("/Users/enfecsolutions/Desktop/self/voice")
sys.path.insert(0, str(ROOT / "src"))

from voiceagent.eval.heldout import SENTENCES  # noqa: E402
from voiceagent.eval.roundtrip import (  # noqa: E402
    EQUIVALENT_LANGUAGES,
    character_overlap,
    normalized,
    transcribe,
)

BENCH = ROOT / "eval_out" / "benchmark_samples"
CONDITIONS = {
    "human": BENCH / "real_heldout",      # the ceiling this metric can reach
    "indicf5": BENCH / "stock",           # current Hindi path
    "chatterbox": ROOT / "eval_out" / "chatterbox_spike",
}
TEXT = {s.slug: s.text for s in SENTENCES}
TARGETS = {s.slug: s.targets for s in SENTENCES}


def score_one(path: Path, expected: str) -> tuple[float, str, str]:
    heard, lang = transcribe(path)
    accepted = EQUIVALENT_LANGUAGES.get("hi", frozenset())
    if lang in accepted and lang != "hi":
        heard, _ = transcribe(path, language="hi")
        lang = "hi"
    overlap = character_overlap(normalized(expected, "hi"), normalized(heard, "hi"))
    return overlap, lang, heard


def duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


def main() -> int:
    results: dict[str, dict[str, dict]] = {}
    for cond, d in CONDITIONS.items():
        results[cond] = {}
        for slug in TEXT:
            p = d / f"{slug}.wav"
            if not p.exists():
                continue
            ov, lang, heard = score_one(p, TEXT[slug])
            results[cond][slug] = dict(
                overlap=ov, lang=lang, heard=heard, seconds=duration(p)
            )
            print(f"{cond:11s} {slug:4s} {ov:5.0%} [{lang}] {duration(p):5.2f}s")

    # transliterated code-mixed variants, chatterbox only
    translit = {}
    for slug in ("h1", "h3", "h12"):
        p = CONDITIONS["chatterbox"] / f"{slug}_translit.wav"
        if p.exists():
            ov, lang, heard = score_one(p, TEXT[slug])
            translit[slug] = dict(overlap=ov, lang=lang, heard=heard,
                                  seconds=duration(p))
            print(f"{'ctbx-tl':11s} {slug:4s} {ov:5.0%} [{lang}] {duration(p):5.2f}s")

    print("\n" + "=" * 74)
    print(f"{'slug':5s} {'target':38s} {'human':>7s} {'indicF5':>8s} {'chatterbox':>11s}")
    print("-" * 74)
    for slug in TEXT:
        row = [results[c].get(slug, {}).get("overlap") for c in
               ("human", "indicf5", "chatterbox")]
        cells = "".join(f"{v:>{w}.0%}" if v is not None else f"{'-':>{w}}"
                        for v, w in zip(row, (8, 9, 12)))
        print(f"{slug:5s} {TARGETS[slug][:37]:38s}{cells}")

    print("-" * 74)
    for cond, w in (("human", 8), ("indicf5", 9), ("chatterbox", 12)):
        vals = [r["overlap"] for r in results[cond].values()]
        secs = sum(r["seconds"] for r in results[cond].values())
        n_hi = sum(1 for r in results[cond].values() if r["lang"] == "hi")
        pass70 = sum(1 for v in vals if v >= 0.70)
        print(f"{cond:11s} mean {sum(vals)/len(vals):5.1%}   "
              f">=70%: {pass70}/{len(vals)}   Hindi-detected: {n_hi}/{len(vals)}   "
              f"total audio {secs:5.1f}s")

    hs = sum(r["seconds"] for r in results["human"].values())
    for cond in ("indicf5", "chatterbox"):
        cs = sum(r["seconds"] for r in results[cond].values())
        print(f"speed ratio {cond:11s}: {cs/hs:.2f}x human duration "
              f"({'faster' if cs < hs else 'slower'} than the speaker)")

    if translit:
        print("\ncode-mixed: raw Latin vs transliterated (chatterbox)")
        for slug, r in translit.items():
            raw = results["chatterbox"][slug]["overlap"]
            print(f"  {slug}: raw {raw:.0%}  ->  translit {r['overlap']:.0%}")

    out = ROOT / "eval_out" / "chatterbox_spike" / "scores.json"
    out.write_text(json.dumps(dict(conditions=results, translit=translit),
                              ensure_ascii=False, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
