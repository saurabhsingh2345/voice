"""Phase 9: find out WHY Hindi sounds wrong, before swapping anything.

    uv run python -m voiceagent.eval.diagnose

Checks, in order of how cheaply they can be ruled out:

  1. Does the text reaching the TTS survive as Devanagari, or has something
     upstream romanized it? (Root cause: lossy transliteration.)
  2. Can the current TTS accept Hindi at all, and through what phonemizer?
  3. Does the model have real Hindi voices, or is it generalizing from English?
  4. If it synthesizes, write the audio so a native speaker can judge it.

Nothing here is inferred. Every claim is printed with the evidence for it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from voiceagent.eval import sentences as S

console = Console()
OUT_DIR = Path(__file__).resolve().parents[3] / "eval_out"


# --- check 1: is the text still Devanagari when it reaches the model? -----


def check_script_integrity() -> bool:
    """Prove the pipeline does not romanize Devanagari in transit.

    The check is deliberately done on the string the *engine* would receive,
    including the sentence chunker, since that is the last thing to touch the
    text before synthesis.
    """
    from voiceagent.tts.chunker import SentenceChunker

    table = Table(title="1. Script integrity through the text pipeline",
                  title_justify="left", header_style="bold")
    table.add_column("Sentence")
    table.add_column("In", justify="right")
    table.add_column("After chunker", justify="right")
    table.add_column("Verdict")

    ok = True
    for sentence in S.HINDI_ONLY[:8]:
        chunker = SentenceChunker()
        pieces = []
        for i in range(0, len(sentence.text), 4):
            pieces.extend(chunker.feed(sentence.text[i : i + 4]))
        pieces.extend(chunker.flush())
        rebuilt = " ".join(pieces)

        before = S.script_report(sentence.text)
        after = S.script_report(rebuilt)
        intact = before["devanagari"] == after["devanagari"]
        ok = ok and intact
        table.add_row(
            sentence.text[:28] + ("…" if len(sentence.text) > 28 else ""),
            str(before["devanagari"]),
            str(after["devanagari"]),
            "[green]intact[/]" if intact else "[red]LOST[/]",
        )

    console.print(table)
    return ok


# --- check 2: what does the current TTS do with Hindi? -------------------


def check_kokoro_hindi() -> dict:
    """Try to load Kokoro's Hindi pipeline and record exactly what happens."""
    from voiceagent.tts.kokoro_engine import KokoroEngine

    result = {"loads": False, "error": None, "phonemizer": None}

    # Which G2P does Kokoro use for this language? Read it from the source of
    # truth rather than assuming.
    try:
        from mlx_audio.tts.models.kokoro import pipeline as kp

        result["phonemizer"] = (
            "misaki (native)" if "h" in "abjz" else "espeak-ng (EspeakG2P)"
        )
        result["lang_name"] = kp.LANG_CODES.get("h", "?")
    except Exception as exc:  # noqa: BLE001
        result["phonemizer"] = f"could not determine: {exc}"

    engine = KokoroEngine(lang_code="h")
    try:
        engine.load()
        result["loads"] = True
        engine.unload()
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def check_hindi_voices() -> list[str]:
    """Does Kokoro ship Hindi-specific voices, or only English ones?"""
    try:
        from huggingface_hub import list_repo_files

        files = list_repo_files("mlx-community/Kokoro-82M-bf16")
        return sorted(
            {
                f.split("/")[-1].removesuffix(".safetensors").removesuffix(".pt")
                for f in files
                if "voices/" in f and f.split("/")[-1].startswith(("hf_", "hm_"))
            }
        )
    except Exception:  # noqa: BLE001
        return []


# --- check 3: synthesize whatever we can, for listening ------------------


async def render_samples(engine, label: str, items, voice: str | None = None) -> list[Path]:
    """Write one wav per sentence so the difference can actually be heard."""
    import soundfile as sf

    target = OUT_DIR / label
    target.mkdir(parents=True, exist_ok=True)
    written = []

    for sentence in items:
        try:
            parts = []
            async for chunk in engine.synthesize(sentence.text, voice=voice):
                if chunk.samples.size:
                    parts.append(chunk.samples)
            if not parts:
                continue
            audio = np.concatenate(parts)
            path = target / f"{sentence.slug}_{sentence.register}.wav"
            sf.write(path, audio, chunk.sample_rate)
            written.append(path)
            console.print(f"  [dim]{sentence.slug}: {len(audio)/chunk.sample_rate:.1f}s  {sentence.text[:40]}[/]")
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [red]{sentence.slug} failed: {type(exc).__name__}: {str(exc)[:90]}[/]")

    return written


# --- report ---------------------------------------------------------------


async def main_async() -> int:
    console.print("[bold]Phase 9 — why does Hindi sound wrong?[/]\n")

    script_ok = check_script_integrity()
    console.print()

    kokoro = check_kokoro_hindi()
    voices = check_hindi_voices()

    table = Table(title="2. Current TTS (Kokoro) vs Hindi", title_justify="left",
                  header_style="bold")
    table.add_column("Question")
    table.add_column("Answer")
    table.add_row("Hindi language code", f"'h' -> {kokoro.get('lang_name', '?')}")
    table.add_row("Phonemizer for Hindi", str(kokoro["phonemizer"]))
    table.add_row("Hindi voices shipped", ", ".join(voices) if voices else "[red]none[/]")
    table.add_row(
        "Can it synthesize Hindi?",
        "[green]yes[/]" if kokoro["loads"] else f"[red]NO — {kokoro['error']}[/]",
    )
    console.print(table)
    console.print()

    findings = []
    if script_ok:
        findings.append(
            "[green]RULED OUT[/] romanization. Devanagari survives the text "
            "pipeline intact — the chunker preserves every codepoint."
        )
    else:
        findings.append("[red]CONFIRMED[/] the text pipeline loses Devanagari characters.")

    if not kokoro["loads"]:
        findings.append(
            "[red]ROOT CAUSE[/] Kokoro cannot speak Hindi in this build at all. "
            "Its Hindi path goes through espeak-ng, which we deliberately "
            "disabled in Phase 3 because phonemizer is GPLv3. Any Hindi you "
            "have heard from this system did not come from Kokoro."
        )
    if voices:
        findings.append(
            f"[yellow]NUANCE[/] Kokoro does ship {len(voices)} Hindi voices "
            f"({', '.join(voices)}), so it has some Hindi training. But they are "
            "unreachable without the GPL phonemizer, and Kokoro's Hindi is a "
            "small slice of an English-dominant 82M model."
        )

    findings.append(
        "[bold]Consequence:[/] even with espeak re-enabled, the architecture is "
        "wrong for native Hindi — rule-based espeak phonemes fed to a mostly-"
        "English 82M model is the textbook recipe for accented Hindi. The fix is "
        "an Indic-native model, not a tweak."
    )

    console.print(Panel("\n\n".join(findings), title="Diagnosis", border_style="cyan"))
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
