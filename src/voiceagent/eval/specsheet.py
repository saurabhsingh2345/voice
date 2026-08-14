"""Measure the whole pipeline and write the numbers down.

    uv run python -m voiceagent.eval.specsheet            # measure and write
    uv run python -m voiceagent.eval.specsheet --check    # refuse to run if the machine is busy

Writes `eval_out/specsheet.json` and `eval_out/SPECSHEET.md`.

WHY THIS IS A PROGRAM AND NOT A DOCUMENT

Every number here has been wrong at least once in this project's history, and
each time it was wrong in the same direction: someone measured, wrote it in a
file, changed the code, and the file kept saying the old thing. The Hindi RTF was
3.40 for months after it stopped being true. A sheet that regenerates cannot
drift from the code.

The second reason is the one that matters commercially. Publishing latency,
memory and a quality ceiling *including the parts that look bad* is the credible
move --- anyone can claim a number, and almost nobody publishes the ceiling their
own metric cannot exceed. That only stays true if the numbers are reproducible by
whoever is reading them, on their own machine, with one command.

REFUSING TO MEASURE A BUSY MACHINE

`--check` exits non-zero when the host is loaded, and it exists because of a real
mistake. A first attempt at the Hindi RTF measured 6-12 and was nearly written
down; the host had a virtualization process on it at load average 537 with 13 GiB
of swap in use. On an idle machine the same code measured 0.63. `voice-doctor`
already warns about this and the warning is easy to scroll past, so here it is a
gate rather than a note.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics as st
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from voiceagent.eval.heldout import SENTENCES

ROOT = Path(__file__).resolve().parents[3]
OUT_JSON = ROOT / "eval_out" / "specsheet.json"
OUT_MD = ROOT / "eval_out" / "SPECSHEET.md"

#: Load average **per core**. A raw load average means nothing without knowing
#: how many cores it is spread across.
#:
#: This took three tries and each wrong answer taught something. 12.0 absolute
#: was far too loose: a sheet passed the gate at load 4.5 and reported Hindi
#: RTF 1.18 for an engine that measures 0.63 idle --- wrong by 1.9x with a green
#: light on it. 3.0 absolute was then too tight: this 12-core machine sits at
#: 2.4-4.3 when genuinely idle, so the gate refused to measure a fine machine.
#:
#: Per-core is the quantity that actually means something. 4.1 on 12 cores is
#: 0.34 --- a third busy. The catastrophic run was 537, which is 44 per core.
#: Those are not two points on one scale; they are different states of matter,
#: and no absolute threshold separates them without also being wrong about one.
MAX_LOAD_PER_CORE = 1.0

#: Below this much free memory the models page out mid-inference.
MIN_FREE_GIB = 4.5

#: Swap is REPORTED, never gated on. Learned the hard way, twice.
#:
#: The README argues that swap *percentage* is useless on macOS because the file
#: is sized to demand. I initially thought absolute swap-in-use escaped that
#: argument -- 14 GiB against 18 GiB of RAM plainly means the machine has been
#: thrashing -- and gated on it. Then I watched it after the load was removed:
#: load fell 3.8 -> 2.4 over four minutes while swap moved 14.11 -> 13.78. It
#: does not drain. Pages stay in the swapfile until something reclaims them, so
#: the number describes the machine's *history*, not its present. Gating on it
#: means a machine that has ever thrashed can never be measured again.
#:
#: It stays in the sheet because it is real context for a reader. It just cannot
#: be a gate.
REPORT_SWAP = True

#: A fixed MLX workload, timed. Reported in the sheet so two runs are comparable
#: even when the load average looked fine in both.
CALIBRATION_ITERATIONS = 60


def host() -> dict:
    import psutil

    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return dict(
        platform=platform.platform(),
        machine=platform.machine(),
        cpus=os.cpu_count(),
        ram_gib=round(memory.total / 2**30, 1),
        free_gib=round(memory.available / 2**30, 2),
        swap_used_gib=round(swap.used / 2**30, 2),
        load_1m=round(os.getloadavg()[0], 1),
    )


def compute_calibration() -> float:
    """Milliseconds for a fixed matmul. Small, fixed, and not model-dependent.

    A direct measurement of how fast this machine is *right now*, which is the
    question the load average is only a proxy for. Two sheets can be compared on
    this number even if both passed the gate.
    """
    import mlx.core as mx

    a = mx.random.normal((512, 512))
    b = mx.random.normal((512, 512))
    mx.eval(a, b)
    started = time.perf_counter()
    for _ in range(CALIBRATION_ITERATIONS):
        a = mx.matmul(a, b) * 0.001
    mx.eval(a)
    return round((time.perf_counter() - started) * 1000, 1)


def machine_is_quiet(state: dict) -> tuple[bool, str]:
    per_core = state["load_1m"] / max(state["cpus"] or 1, 1)
    if per_core > MAX_LOAD_PER_CORE:
        return False, (
            f"load is {state['load_1m']} across {state['cpus']} cores "
            f"({per_core:.2f} per core, above {MAX_LOAD_PER_CORE}). Timings taken "
            "now will be wrong -- and wrong by a factor, not a margin."
        )
    if state["free_gib"] < MIN_FREE_GIB:
        return False, (
            f"only {state['free_gib']} GiB free, below {MIN_FREE_GIB}. The models "
            "will page out mid-inference."
        )
    return True, (
        f"quiet enough to measure ({per_core:.2f} load per core, "
        f"{state['free_gib']} GiB free)"
    )


def _hindi_reference() -> tuple[np.ndarray, int]:
    audio, rate = sf.read(str(ROOT / "fixtures/hi/reference_lekha.wav"), dtype="float32")
    return (audio.mean(axis=1) if audio.ndim > 1 else audio), rate


def measure_hindi_tts(repeats: int = 2) -> dict:
    from voiceagent.tts.chatterbox_indic import ChatterboxIndicEngine, concat_with_crossfade

    ref, rate = _hindi_reference()
    engine = ChatterboxIndicEngine()
    started = time.perf_counter()
    engine.load()
    load_s = time.perf_counter() - started
    engine.set_reference(ref, "", rate)

    rtfs, audio_s = [], 0.0
    for sentence in SENTENCES:
        runs = []
        for _ in range(repeats):
            t0 = time.perf_counter()

            async def go():
                return concat_with_crossfade(
                    [c.samples async for c in engine.synthesize(sentence.text)], 24_000
                )

            audio = asyncio.run(go())
            seconds = len(audio) / 24_000
            runs.append((time.perf_counter() - t0) / seconds)
        rtfs.append(st.median(runs))
        audio_s += seconds

    resident = engine.resident_bytes / 2**30
    engine.unload()
    return dict(
        engine="Chatterbox Multilingual v3 (MLX, 8-bit)",
        checkpoint=str(engine.repo),
        load_s=round(load_s, 1),
        resident_gib=round(resident, 2),
        rtf_median=round(st.median(rtfs), 2),
        rtf_worst=round(max(rtfs), 2),
        sentences=len(rtfs),
        audio_seconds=round(audio_s, 1),
    )


def measure_hindi_stt() -> dict:
    """CER against the speaker's own reading of the held-out sentences."""
    from voiceagent.eval.audio import resample
    from voiceagent.stt.mlx_whisper_engine import MLXWhisperEngine
    from voiceagent.text.normalize_hi import normalize

    recorded = ROOT / "eval_out" / "benchmark_samples" / "real_heldout"
    text = {s.slug: s.text for s in SENTENCES}
    engine = MLXWhisperEngine(language="hi")
    started = time.perf_counter()
    engine.load()
    load_s = time.perf_counter() - started

    errors = total = 0
    rtfs = []
    for slug, expected in text.items():
        path = recorded / f"{slug}.wav"
        if not path.exists():
            continue
        audio, rate = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = resample(audio, rate, 16_000)
        t0 = time.perf_counter()
        heard = engine.transcribe(audio).text
        rtfs.append((time.perf_counter() - t0) / (len(audio) / 16_000))
        errors += _levenshtein(normalize(expected), normalize(heard))
        total += len(normalize(expected))

    resident = engine.resident_bytes / 2**30
    engine.unload()
    return dict(
        engine="whisper-large-v3-turbo (MLX), pinned hi",
        load_s=round(load_s, 1),
        resident_gib=round(resident, 2),
        cer_percent=round(100 * errors / total, 1) if total else None,
        rtf_median=round(st.median(rtfs), 2) if rtfs else None,
        clips=len(rtfs),
    )


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def measure_roundtrip() -> dict:
    """Intelligibility of synthesis, against the ceiling this metric can reach.

    The human row is the point. A flawless recording by the speaker scores 90.2 %,
    not 100 %, because the scorer compares Whisper's spelling against the
    transliterator's. Every synthetic number has to be read against that, and for
    a long time they were not.
    """
    from voiceagent.eval.roundtrip import (
        character_overlap,
        decode_for_scoring,
        normalized,
    )

    conditions = {
        "human recording (metric ceiling)": ROOT / "eval_out/benchmark_samples/real_heldout",
        "IndicF5 (previous engine)": ROOT / "eval_out/benchmark_samples/stock",
        "Chatterbox 8-bit (current)": ROOT / "eval_out/chatterbox_8bit",
    }
    text = {s.slug: s.text for s in SENTENCES}
    code_mixed = {"h1", "h3", "h12"}
    out = {}
    for label, directory in conditions.items():
        scores, mixed = {}, []
        for slug, expected in text.items():
            path = directory / f"{slug}.wav"
            if not path.exists():
                continue
            heard, _, _ = decode_for_scoring(path, "hi")
            score = character_overlap(normalized(expected, "hi"), normalized(heard, "hi"))
            scores[slug] = score
            if slug in code_mixed:
                mixed.append(score)
        if scores:
            values = list(scores.values())
            out[label] = dict(
                mean_percent=round(100 * sum(values) / len(values), 1),
                worst_percent=round(100 * min(values), 1),
                code_mixed_percent=round(100 * sum(mixed) / len(mixed), 1) if mixed else None,
                sentences=len(values),
            )
    return out


def licence_audit() -> dict:
    from voiceagent.models import audit_installed_packages, audit_licenses

    violations, accepted = audit_installed_packages()
    return dict(
        model_violations=audit_licenses(),
        dependency_violations=violations,
        accepted_exceptions=[note.split(":")[0] for note in accepted],
        clean=not violations and not audit_licenses(),
    )


def build(skip_slow: bool = False) -> dict:
    state = host()
    quiet, why = machine_is_quiet(state)
    state["calibration_ms"] = compute_calibration()
    report = dict(host=state, machine_quiet=quiet, machine_note=why, licences=licence_audit())
    if skip_slow:
        return report
    print("measuring Hindi TTS ...")
    report["hindi_tts"] = measure_hindi_tts()
    print("measuring Hindi STT ...")
    report["hindi_stt"] = measure_hindi_stt()
    print("scoring round trip ...")
    report["roundtrip"] = measure_roundtrip()
    return report


def render(report: dict) -> str:
    state = report["host"]
    lines = [
        "# Measured spec sheet",
        "",
        "Generated by `uv run python -m voiceagent.eval.specsheet`. Every number "
        "below is measured on this machine, now --- none is copied from a document.",
        "",
        f"- **Host** — {state['machine']}, {state['cpus']} cores, {state['ram_gib']} GiB RAM",
        f"- **State when measured** — {state['free_gib']} GiB free, "
        f"load {state['load_1m']}, swap {state['swap_used_gib']} GiB, "
        f"calibration {state.get('calibration_ms', '?')} ms",
        f"- **Verdict** — {report['machine_note']}",
        "",
    ]
    if not report["machine_quiet"]:
        lines += [
            "> **These timings are not trustworthy.** The host was busy when they "
            "were taken. Re-run on an idle machine.",
            "",
        ]

    audit = report["licences"]
    lines += [
        "## Licences",
        "",
        f"- Permissive-only audit: **{'clean' if audit['clean'] else 'FAILING'}**",
        f"- Model violations: {len(audit['model_violations'])}",
        f"- Dependency violations: {len(audit['dependency_violations'])}",
        f"- Recorded exceptions: {', '.join(audit['accepted_exceptions']) or 'none'}",
        "",
        "Enforced in code, not policy: `voice-doctor` exits non-zero on a "
        "violation anywhere in the installed dependency tree.",
        "",
    ]

    if tts := report.get("hindi_tts"):
        lines += [
            "## Hindi speech synthesis",
            "",
            f"- Engine — {tts['engine']}",
            f"- **RTF {tts['rtf_median']} median**, {tts['rtf_worst']} worst "
            f"({tts['sentences']} held-out sentences, {tts['audio_seconds']}s of audio)",
            f"- {tts['resident_gib']} GiB resident, {tts['load_s']}s to load",
            "",
            "RTF below 1.0 means synthesis keeps up with playback. This figure is "
            "strongly load-sensitive: the same engine measures 0.63 on an idle "
            "host and 1.18 with a virtual machine running alongside it. Compare "
            "the calibration figure above between runs before comparing these.",
            "",
        ]

    if stt := report.get("hindi_stt"):
        lines += [
            "## Hindi speech recognition",
            "",
            f"- Engine — {stt['engine']}",
            f"- **CER {stt['cer_percent']} %** over {stt['clips']} recordings by "
            "the speaker, on the *held-out* set",
            f"- RTF {stt['rtf_median']} median, {stt['resident_gib']} GiB resident",
            "",
            "Higher than the 4.8 % quoted elsewhere in this project, and not a "
            "contradiction: that figure is on ordinary Hindi, this one is on the "
            "held-out set, which was built to be hard --- code-switching, digits, "
            "retroflex clusters and nuqta consonants, chosen because they break "
            "things.",
            "",
        ]

    if rt := report.get("roundtrip"):
        lines += [
            "## Intelligibility, against the ceiling",
            "",
            "| Condition | Mean | Worst | Code-mixed |",
            "| --- | --- | --- | --- |",
        ]
        for label, row in rt.items():
            mixed = f"{row['code_mixed_percent']} %" if row["code_mixed_percent"] else "—"
            lines.append(
                f"| {label} | {row['mean_percent']} % | {row['worst_percent']} % | {mixed} |"
            )
        lines += [
            "",
            "**Read the human row first.** A flawless recording by the speaker "
            "scores well under 100 %, because the scorer compares Whisper's "
            "spelling against the transliterator's. That is the ceiling this "
            "metric can reach, and every other row has to be read against it. A "
            "synthetic score at or above it means the metric has stopped "
            "discriminating --- not that the model sounds better than a person.",
            "",
            "Intelligibility is not naturalness. Nothing here measures prosody.",
            "",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="only report whether the machine is quiet enough; exit 1 if not")
    parser.add_argument("--fast", action="store_true",
                        help="skip everything that loads a model")
    args = parser.parse_args(argv)

    if args.check:
        state = host()
        quiet, why = machine_is_quiet(state)
        print(f"load {state['load_1m']} on {state['cpus']} cores, "
              f"free {state['free_gib']} GiB, swap {state['swap_used_gib']} GiB "
              f"(reported, never gated on) -> {why}")
        return 0 if quiet else 1

    report = build(skip_slow=args.fast)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2))
    OUT_MD.write_text(render(report))
    print(f"\nwrote {OUT_MD}")
    print(f"wrote {OUT_JSON}")
    if not report["machine_quiet"]:
        print("\nWARNING: the machine was busy; the timings are not trustworthy.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
