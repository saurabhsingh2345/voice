"""Quantize the Hindi checkpoint locally, from the MIT source.

    uv run python -m voiceagent.tts.quantize --bits 8

Writes to `data/models/chatterbox-multilingual-v3-<bits>bit`, which is gitignored;
point `ChatterboxIndicEngine(repo=...)` at that path to use it.

WHY LOCALLY, AND NOT A COMMUNITY REQUANT

Four 8-bit and 4-bit Chatterbox Multilingual conversions exist on the Hub and all
of them are usable in mlx-audio's layout. Every one of them declares **no
licence**. This project's rule is permissive-only and `voice-doctor` enforces it,
so an untagged checkpoint cannot go in `models.py` regardless of how obvious its
provenance looks -- a quantization of an MIT model is presumably MIT, but
"presumably" is exactly the word the audit exists to remove.

Quantizing from `mlx-community/chatterbox-multilingual-v3` ourselves keeps the
licence chain intact and takes about seven seconds.

WHAT IT BUYS, MEASURED

Memory, unambiguously. These are `mx.get_peak_memory()` and are not sensitive to
machine load:

    fp32   3.04 GiB resident   5.09 GiB peak during generation
    8-bit  1.33 GiB resident   2.77 GiB peak

That more than halves the floor, and the floor is the binding constraint on an
18 GiB machine -- `MIN_FREE_GIB` is 4.0 for the fp32 path.

Speed is **not yet established**. The first five held-out sentences measured RTF
0.63-0.95 against fp32's 1.24, which would put Hindi under real time for the
first time. Then the same engine, same seeds, re-measured the remaining sentences
at RTF 6-12 and stayed there across three repeats. That is not the model: the
machine had a virtualization process thrashing it, load average 537, 13 GiB of
swap in use, and Whisper could no longer transcribe 24 short clips inside ten
minutes. Do not quote an RTF from that session.

Quality is also unmeasured for the 8-bit path, for the same reason -- the scorer
never finished. Both need re-running on an idle machine before this is adopted:

    uv run python -m voiceagent.eval.hindi_tts     # round-trip, per register
"""

from __future__ import annotations

import argparse
from pathlib import Path

from voiceagent.tts.chatterbox_indic import CHATTERBOX_REPO

#: Under `data/`, which is gitignored. A 900 MB artifact is a build output, not
#: a source file, and it is reproducible from this script in seconds.
DEFAULT_OUT = Path("data/models")

#: 64 matches what mlx-lm uses by default and divides every Linear in this model
#: that is eligible; a group size the weights do not divide silently skips layers.
GROUP_SIZE = 64


def quantize(bits: int = 8, group_size: int = GROUP_SIZE, out: Path | None = None) -> Path:
    from mlx_audio.tts.utils import convert

    destination = (out or DEFAULT_OUT) / f"chatterbox-multilingual-v3-{bits}bit"
    convert(
        CHATTERBOX_REPO,
        mlx_path=str(destination),
        quantize=True,
        q_bits=bits,
        q_group_size=group_size,
    )
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bits", type=int, default=8, choices=(4, 8))
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    destination = quantize(args.bits, args.group_size, args.out)
    print(f"\nwrote {destination}")
    print("Not adopted by default: measure it first on an idle machine.")
    print("  uv run python -m voiceagent.eval.hindi_tts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
