"""Generate the samples for a blind benchmark and assemble it.

    uv run python -m voiceagent.eval.build_benchmark <profile_id>

Produces four conditions, and the fourth is the one that makes the test
interpretable rather than merely suggestive:

  real      untouched recordings of the speaker            (upper anchor)
  vocoded   those same recordings passed through the       (channel control)
            mel-spectrogram and vocoder, nothing else
  ours      the fine-tuned model on held-out sentences
  stock     stock IndicF5 on the same sentences            (baseline)

WHY THE VOCODER CONTROL EXISTS

A real recording carries microphone and room character that model output does not.
Without a control, a listener can learn to spot the *channel* instead of the voice,
and "they could tell" becomes uninterpretable — was it the voice, or was it that one
set of clips has room in it? Passing real speech through the same mel-and-vocoder
path the model synthesises through removes the channel difference while keeping
genuine human prosody. So:

  listeners call `vocoded` synthetic too    -> the tell is the vocoder, not our model,
                                              and no amount of fine-tuning fixes it
  listeners call `vocoded` real but `ours`  -> the tell is genuinely our model's
  synthetic                                   prosody or timbre, which is fixable

Everything is loudness-matched to a common RMS. Level differences are the single
loudest tell available to a listener and would otherwise swamp the thing being
measured.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from voiceagent.eval import heldout
from voiceagent.eval.abtest import IDENTITY, NATURALNESS, build
from voiceagent.tts.indic_engine import concat_with_crossfade
from voiceagent.voice_clone.dataset import VoiceDataset

#: Common loudness for every condition. -20 dBFS RMS is a conventional speech
#: target: loud enough to judge, quiet enough that nothing clips after alignment.
TARGET_RMS = 0.1

WORK = Path("eval_out/benchmark_samples")


def match_loudness(audio: np.ndarray, target: float = TARGET_RMS) -> np.ndarray:
    rms = float(np.sqrt((audio.astype(np.float64) ** 2).mean())) if audio.size else 0.0
    if rms <= 1e-9:
        return audio
    scaled = audio * (target / rms)
    peak = float(np.abs(scaled).max())
    if peak > 0.99:
        scaled = scaled * (0.99 / peak)
    return scaled.astype(np.float32)


def write(path: Path, audio: np.ndarray, rate: int = 24_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, match_loudness(audio), rate, subtype="PCM_16")
    return path


def real_clips(profile_id: str, count: int) -> list[tuple[str, np.ndarray, int]]:
    """The speaker's own recordings, longest first so they are substantial enough to judge."""
    dataset = VoiceDataset()
    picked = []
    for clip in sorted(dataset.clips(profile_id), key=lambda c: -c.duration_seconds):
        if len(picked) >= count:
            break
        if clip.duration_seconds < 3.0:
            continue
        audio, rate = sf.read(io.BytesIO(dataset.audio(profile_id, clip.clip_id)), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        picked.append((clip.clip_id, audio, rate))
    return picked


def vocode(audio: np.ndarray, rate: int) -> np.ndarray | None:
    """Real speech through the same mel-and-vocoder path the model synthesises through.

    Returns None if the vocoder is unavailable, so a missing control degrades the
    benchmark rather than breaking it.
    """
    try:
        import torch
        from f5_tts.infer import utils_infer as U

        vocoder = U.load_vocoder(vocoder_name="vocos", is_local=False, device="cpu")
        if rate != U.target_sample_rate:
            idx = (np.arange(int(len(audio) * U.target_sample_rate / rate))
                   * rate / U.target_sample_rate).astype(int)
            audio = audio[idx[idx < len(audio)]]
        wave = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
        from f5_tts.model.modules import MelSpec

        mel_spec = MelSpec(
            n_fft=U.n_fft, hop_length=U.hop_length, win_length=U.win_length,
            n_mel_channels=U.n_mel_channels, target_sample_rate=U.target_sample_rate,
            mel_spec_type="vocos",
        )
        with torch.no_grad():
            mel = mel_spec(wave)
            out = vocoder.decode(mel)
        return np.asarray(out.squeeze().cpu().numpy(), dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        print(f"  vocoder control unavailable ({type(exc).__name__}: {exc})")
        return None


def synthesize(engine, reference, reference_text, rate, sentences) -> dict[str, np.ndarray]:
    engine.load()
    engine.set_reference(reference, reference_text, rate)
    out = {}
    for sentence in sentences:
        started = time.perf_counter()
        parts: list[np.ndarray] = []

        async def go():
            async for chunk in engine.synthesize(sentence.text):
                parts.append(chunk.samples)

        asyncio.run(go())
        if not parts:
            print(f"    {sentence.slug}: no audio, skipped")
            continue
        audio = concat_with_crossfade(parts, 24_000)
        out[sentence.slug] = audio
        print(f"    {sentence.slug}: {len(audio)/24000:5.2f}s in {time.perf_counter()-started:5.1f}s")
    engine.unload()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("profile_id")
    parser.add_argument("--limit", type=int, default=len(heldout.SENTENCES),
                        help="how many held-out sentences to use")
    parser.add_argument("--skip-stock", action="store_true",
                        help="skip the stock baseline (halves the runtime)")
    args = parser.parse_args(argv)

    from voiceagent.tts.indic_engine import IndicTTSEngine
    from voiceagent.voice_clone.store import VoiceProfileStore

    store = VoiceProfileStore()
    profile = store.get(args.profile_id)
    if profile is None:
        print(f"no such profile: {args.profile_id}")
        return 1

    reference, rate = sf.read(io.BytesIO(store.reference_audio(args.profile_id)), dtype="float32")
    if reference.ndim > 1:
        reference = reference.mean(axis=1)

    sentences = list(heldout.SENTENCES)[: args.limit]
    checkpoint = Path("data/f5tts_ckpts") / args.profile_id / "model_last.pt"
    samples: dict[str, dict[str, Path]] = {}

    print(f"profile   : {profile.speaker_name} ({args.profile_id})")
    print(f"sentences : {len(sentences)} held out")
    print(f"checkpoint: {'present' if checkpoint.exists() else 'MISSING — no fine-tuned condition'}\n")

    # 1. real
    print("real recordings:")
    reals = real_clips(args.profile_id, len(sentences))
    samples["real"] = {}
    for i, (clip_id, audio, clip_rate) in enumerate(reals):
        slug = f"r{i+1}"
        samples["real"][slug] = write(WORK / "real" / f"{slug}.wav", audio, clip_rate)
        print(f"    {slug}: {len(audio)/clip_rate:5.2f}s")

    # 2. vocoded control
    print("\nvocoder control (real speech through mel + vocoder):")
    samples["vocoded"] = {}
    for i, (clip_id, audio, clip_rate) in enumerate(reals):
        done = vocode(audio, clip_rate)
        if done is None:
            break
        slug = f"r{i+1}"
        samples["vocoded"][slug] = write(WORK / "vocoded" / f"{slug}.wav", done)
        print(f"    {slug}: {len(done)/24000:5.2f}s")
    if not samples["vocoded"]:
        del samples["vocoded"]
        print("    none produced — the channel confound will be uncontrolled")

    # 3. fine-tuned
    if checkpoint.exists():
        print("\nfine-tuned:")
        got = synthesize(IndicTTSEngine(checkpoint=checkpoint), reference,
                         profile.reference_text, rate, sentences)
        samples["ours"] = {s: write(WORK / "ours" / f"{s}.wav", a) for s, a in got.items()}

    # 4. stock baseline
    if not args.skip_stock:
        print("\nstock IndicF5:")
        got = synthesize(IndicTTSEngine(), reference, profile.reference_text, rate, sentences)
        samples["stock"] = {s: write(WORK / "stock" / f"{s}.wav", a) for s, a in got.items()}

    if len(samples) < 2:
        print("\nnot enough conditions to compare")
        return 1

    bench = build(
        samples,
        real_systems=("real",),
        # Identity asks "did a person say this". It is meaningful for the speaker's
        # own voice and its imitations, and meaningless for a third-party voice --
        # so competitor samples dropped in later join naturalness only.
        identity_systems=tuple(k for k in samples if k in ("real", "vocoded", "ours", "stock")),
    )

    print(f"\nbenchmark {bench.benchmark_id}")
    print(f"  conditions : {', '.join(sorted(samples))}")
    print(f"  items      : {sum(1 for i in bench.items if i.kind == IDENTITY)} identity, "
          f"{sum(1 for i in bench.items if i.kind == NATURALNESS)} naturalness")
    print(f"  listen at  : http://127.0.0.1:8823/listen")
    print(f"  results at : http://127.0.0.1:8823/api/listen/latest/results")
    print("\nTo add a competitor: put WAVs of the same held-out sentences in")
    print(f"  {WORK}/<name>/<slug>.wav   then re-run with --skip-stock")
    return 0


if __name__ == "__main__":
    sys.exit(main())
