"""Is Hindi synthesis actually intelligible, across registers?

    uv run python -m voiceagent.eval.hindi_tts                 # all registers
    uv run python -m voiceagent.eval.hindi_tts --register formal
    uv run python -m voiceagent.eval.hindi_tts --limit 2

Drives the real `IndicTTSEngine` -- not a hand-built model -- so it catches a
regression in the engine's own construction, which is exactly where the babble
bug lived. Each sentence is synthesized and then transcribed back with Whisper;
a sentence passes only if the transcript overlaps the input and Whisper agrees
the language is Hindi.

This exists because judging Hindi TTS by ear does not scale and spectral
measures actively mislead: the babble this catches measured 0.06-0.10 spectral
flatness against 0.088 for real speech, i.e. indistinguishable from genuine
speech by that metric. See voiceagent.eval.roundtrip.

Exits non-zero if any sentence fails, so it works as a regression gate.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np
import soundfile as sf
from rich.console import Console
from rich.table import Table

from voiceagent.eval import sentences as S
from voiceagent.eval.roundtrip import character_overlap, transcribe
from voiceagent.text.normalize_hi import normalize as normalize_hi

console = Console()

OUT_DIR = Path(__file__).resolve().parents[3] / "eval_out" / "hindi_tts"
REF_WAV = Path(__file__).resolve().parents[3] / "fixtures" / "hi" / "reference_lekha.wav"
REF_TXT = REF_WAV.with_suffix(".txt")

#: A sentence passes at >=0.7 rather than the roundtrip module's 0.5. That
#: looser bar exists there to separate "different language" from "works"; here
#: the fix is already in, so the bar is set where garbled-but-Hindi fails. The
#: rope-fix-only variant scored 0.55 and was not usable speech.
PASS_OVERLAP = 0.70

REGISTERS = {
    "formal": S.FORMAL,
    "colloquial": S.COLLOQUIAL,
    "code-mixed": S.CODE_MIXED,
    "numeric": S.NUMERIC,
}


async def run(register: str | None, limit: int | None) -> int:
    from voiceagent.tts.indic_engine import IndicTTSEngine

    if not REF_WAV.exists():
        console.print(f"[red]missing reference clip:[/] {REF_WAV}")
        console.print("This file is untracked on purpose (it is a real recorded voice).")
        return 2

    chosen = REGISTERS if register is None else {register: REGISTERS[register]}
    cases = [s for group in chosen.values() for s in (group[:limit] if limit else group)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audio, sr = sf.read(str(REF_WAV), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    engine = IndicTTSEngine()
    engine.set_reference(audio, REF_TXT.read_text().strip(), sr)
    if warning := engine.reference_health():
        console.print(f"[yellow]reference warning:[/] {warning}")

    console.print(f"loading {engine.repo} ...")
    engine.load()
    console.print(f"loaded on [bold]{engine._device}[/] "
                  f"({engine.resident_bytes / 2**30:.2f} GiB)\n")

    table = Table(title="Hindi TTS round-trip", title_justify="left", header_style="bold")
    table.add_column("")
    table.add_column("Register")
    table.add_column("Expected")
    table.add_column("Heard back")
    table.add_column("Lang", justify="center")
    table.add_column("Overlap", justify="right")
    table.add_column("RTF", justify="right")

    failures = 0
    rtfs: list[float] = []
    for case in cases:
        out_path = OUT_DIR / f"{case.slug}_{case.register}.wav"
        # Normalize exactly as the router does, so digits and symbols reach the
        # model as Devanagari words. Skipping this makes numeric cases fail for
        # a reason that has nothing to do with the model.
        spoken = normalize_hi(case.text)
        chunks = [c async for c in engine.synthesize(spoken)]
        if not chunks:
            table.add_row("[red]FAIL[/]", case.register, spoken, "[red]no audio[/]", "-", "-")
            failures += 1
            continue

        samples = np.concatenate([c.samples for c in chunks])
        sf.write(out_path, samples, chunks[0].sample_rate)

        # RTF = synthesis time / audio duration. Above 1 means it cannot keep up
        # with speech, which decides whether this engine can serve the live loop
        # or only type-and-listen.
        seconds = len(samples) / chunks[0].sample_rate
        latency_ms = chunks[0].latency_ms
        rtf = (latency_ms / 1000) / seconds if latency_ms and seconds else float("nan")
        if rtf == rtf:  # not NaN
            rtfs.append(rtf)

        heard, language = transcribe(out_path)
        # Score against the normalized text -- that is what was actually spoken
        # -- and normalize the transcript too, because Whisper applies *inverse*
        # text normalization: it writes spoken numerals back as digits. The TTS
        # correctly said "एक हज़ार दो सौ निन्यानवे" and Whisper returned "1299",
        # which is a perfect round trip that scores 39% if compared raw. Running
        # both sides through the same pass puts them in one canonical form.
        overlap = character_overlap(spoken, normalize_hi(heard))
        ok = overlap >= PASS_OVERLAP and language == "hi"
        failures += 0 if ok else 1

        canonical = normalize_hi(heard)
        shown = heard if canonical == heard else f"{heard}\n[dim]-> {canonical}[/]"

        table.add_row(
            "[green]PASS[/]" if ok else "[red]FAIL[/]",
            case.register,
            spoken,
            shown or "[dim](silence)[/]",
            language if language == "hi" else f"[red]{language}[/]",
            f"{overlap:.0%}" if ok else f"[red]{overlap:.0%}[/]",
            f"{rtf:.2f}",
        )

    console.print(table)
    engine.unload()

    total = len(cases)
    if failures:
        console.print(f"\n[red]{failures}/{total} sentences are not intelligible Hindi.[/]")
        console.print("If all of them failed, check pe_attn_head / text_mask_padding in "
                      "tts/indic_engine.py -- see OLD_SEMANTICS there.")
    else:
        console.print(f"\n[green]all {total} sentences came back as intelligible Hindi[/] "
                      f"(>={PASS_OVERLAP:.0%} overlap)")
    if rtfs:
        median_rtf = sorted(rtfs)[len(rtfs) // 2]
        console.print(f"median RTF {median_rtf:.2f} "
                      f"(synthesis time / audio duration; >1 cannot keep up with speech)")
        if median_rtf > 1:
            console.print("[yellow]This is type-and-listen speed, not live-loop speed.[/] "
                          "Kokoro runs ~0.1 for English.")
    console.print(f"audio written to {OUT_DIR}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", choices=sorted(REGISTERS), help="only this register")
    parser.add_argument("--limit", type=int, help="first N sentences per register")
    args = parser.parse_args()
    return asyncio.run(run(args.register, args.limit))


if __name__ == "__main__":
    raise SystemExit(main())
