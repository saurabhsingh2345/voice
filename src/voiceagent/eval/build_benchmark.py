"""Generate the samples for a blind benchmark and assemble it.

    uv run python -m voiceagent.eval.build_benchmark <profile_id>

Produces three conditions, and the second is the one that makes the test
interpretable rather than merely suggestive:

  real      untouched recordings of the speaker            (upper anchor)
  vocoded   those same recordings passed through the       (channel control)
            mel-spectrogram and vocoder, nothing else
  ours      Chatterbox Multilingual on held-out sentences

There used to be a fourth. `ours` was a per-voice IndicF5 fine-tune and `stock`
was unmodified IndicF5, so the pair measured what fine-tuning bought. Chatterbox
clones zero-shot --- there is one checkpoint for every voice --- so the two
conditions would be byte-identical and the comparison is gone. What replaces it
is a *baseline*: drop the old IndicF5 renders (or ElevenLabs, or Sarvam) into
`eval_out/benchmark_samples/<name>/<slug>.wav` and re-run. Those join the
naturalness test only; see `identity_systems` below for why.

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
from voiceagent.eval.audio import resample as _resample
from voiceagent.tts.chatterbox_indic import concat_with_crossfade
from voiceagent.voice_clone.dataset import VoiceDataset

#: Common loudness for every condition. -20 dBFS RMS is a conventional speech
#: target: loud enough to judge, quiet enough that nothing clips after alignment.
TARGET_RMS = 0.1

WORK = Path("eval_out/benchmark_samples")

#: The speaker reading the held-out sentences. Preferred over training clips for
#: the "real" condition, because it makes real and synthetic differ in exactly one
#: thing. See the confound note in `web.server.benchmark_record`.
REAL_HELDOUT = WORK / "real_heldout"


def resample(audio: np.ndarray, from_rate: int, to_rate: int = 24_000) -> np.ndarray:
    """Every condition reaches the listener at one rate.

    The microphone records at 48 kHz and the model synthesises at 24 kHz. Left alone
    that is a second channel difference sitting next to the vocoder — `real` would be
    the only condition carrying anything above 12 kHz, which is a tell that has
    nothing to do with whether a person spoke.
    """
    return _resample(audio, from_rate, to_rate)


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


def vocode(audio: np.ndarray, rate: int, engine=None) -> np.ndarray | None:
    """Real speech through the same mel-and-vocoder path the model synthesises through.

    The control that makes the rest of the benchmark interpretable. A listener
    who marks our samples as synthetic may be hearing the *clone* or merely the
    *channel* --- every generated sample passes through a mel bottleneck and a
    neural vocoder, and that alone leaves an audible signature. Passing a real
    recording through the identical path and rating it too separates the two. If
    vocoded-real scores as badly as ours, the tell is the channel and improving
    the clone will not move it.

    Reimplemented on Chatterbox's own HiFiGAN when the Hindi path moved off
    IndicF5. It used to run f5-tts's vocos at 24 kHz; it now takes the mel
    parameters straight from `s3gen.embed_ref` (n_fft 1920, 80 mels, hop 480,
    fmax 8000, center=False) so the control passes through the same bottleneck
    the model actually uses. Those constants are not free choices --- a mel that
    does not match what `mel2wav` was trained on reconstructs badly, and the
    control would then overstate the channel's contribution.

    Returns None if the vocoder is unavailable, so a missing control degrades the
    benchmark rather than breaking it.
    """
    try:
        import mlx.core as mx

        from mlx_audio.tts.models.chatterbox.s3gen.mel import mel_spectrogram

        if engine is None or engine._model is None:
            print("  vocoder control needs a loaded engine; skipped")
            return None

        def run():
            wave = mx.array(np.asarray(resample(audio, rate, 24_000), dtype=np.float32))
            mel = mel_spectrogram(
                mx.expand_dims(wave, 0),
                n_fft=1920, num_mels=80, sampling_rate=24_000,
                hop_size=480, win_size=1920, fmin=0, fmax=8000, center=False,
            )
            # (B, D, T) -> (B, T, D), the layout mel2wav.inference expects.
            out, _ = engine._model.s3gen.hift_inference(
                speech_feat=mx.transpose(mel, [0, 2, 1])
            )
            return np.asarray(out, dtype=np.float32).reshape(-1)

        # MLX arrays are thread-affine, so this has to run on the same thread
        # that loaded the weights. See ChatterboxIndicEngine._run.
        return engine._run(run)
    except Exception as exc:  # noqa: BLE001
        print(f"  vocoder control unavailable ({type(exc).__name__}: {exc})")
        return None


def synthesize(engine, reference, reference_text, rate, sentences) -> dict[str, np.ndarray]:
    """Render every held-out sentence. Assumes the engine is already loaded.

    Loading and unloading moved out to `main`, because the vocoder control now
    runs on the same model's HiFiGAN and must share the loaded instance --- and
    its thread. See `vocode`.
    """
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
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("profile_id")
    parser.add_argument("--limit", type=int, default=len(heldout.SENTENCES),
                        help="how many held-out sentences to use")
    parser.add_argument("--skip-synthesis", action="store_true",
                        help="rebuild from WAVs already on disk, without synthesizing")
    parser.add_argument("--allow-unmatched-real", action="store_true",
                        help="build from training clips when the held-out set is not "
                             "recorded — produces an uninterpretable identity test")
    args = parser.parse_args(argv)

    from voiceagent.tts.chatterbox_indic import ChatterboxIndicEngine
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
    samples: dict[str, dict[str, Path]] = {}

    print(f"profile   : {profile.speaker_name} ({args.profile_id})")
    print(f"sentences : {len(sentences)} held out\n")

    # One engine for both the synthesis and the vocoder control. It is loaded
    # here rather than inside `synthesize` because `vocode` reaches into the same
    # model's HiFiGAN, and MLX arrays cannot cross threads --- so both have to go
    # through this instance's worker.
    engine = ChatterboxIndicEngine()
    if not args.skip_synthesis:
        print("loading Chatterbox Multilingual ...")
        engine.load()

    # 1. real -- content-matched if the speaker has read the held-out sentences.
    #
    # The first benchmark used training clips here, and that made both tests
    # uninterpretable in opposite directions. Content and speaking style differed
    # from the synthetic condition, so identity could be called on familiarity
    # rather than voice; and naturalness scored the synthetic clips ABOVE the real
    # ones, because clean read sentences sound tidier than spontaneous speech with
    # disfluencies. Same sentences on both sides removes it.
    matched = {s.slug: REAL_HELDOUT / f"{s.slug}.wav" for s in sentences}
    matched = {slug: path for slug, path in matched.items() if path.exists()}

    samples["real"] = {}
    reals: list[tuple[str, np.ndarray, int]] = []
    if len(matched) >= max(3, len(sentences) // 2):
        print(f"real recordings: {len(matched)} content-matched (held-out sentences)")
        for slug, path in sorted(matched.items()):
            audio, clip_rate = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            # To 24 kHz here, not at playback: the browser would resample it anyway,
            # but then `real` would be the only condition carrying content above
            # 12 kHz. Doing it now leaves the vocoder as the only thing separating
            # `real` from `vocoded`, which is what that control is for.
            audio = resample(audio, clip_rate, 24_000)
            samples["real"][slug] = write(WORK / "real" / f"{slug}.wav", audio)
            reals.append((slug, audio, 24_000))
            print(f"    {slug}: {len(audio)/24_000:5.2f}s")
    elif not args.allow_unmatched_real:
        # Refuses rather than warns, because the warning was not enough. The first
        # run took this path, printed this text, scrolled it away, and two people
        # spent their afternoon rating a test whose identity result could not mean
        # anything. Listener time is the scarcest input here; spending it on a
        # confounded benchmark is the expensive mistake, not stopping early.
        print("real recordings: only "
              f"{len(matched)} of {len(sentences)} held-out sentences are recorded.")
        print()
        print("  Refusing to fall back to training clips. Those are different")
        print("  sentences, spoken spontaneously, and about twice as long as the")
        print("  synthetic clips -- so a listener can sort them without hearing a")
        print("  voice, and the fooled rate stops being about the voice.")
        print()
        print("  Record them:  uv run voice-web  ->  http://127.0.0.1:8823/record-benchmark")
        print("  Or override:  --allow-unmatched-real  (naturalness only; identity")
        print("                will be reported as uninterpretable)")
        return 1
    else:
        print("real recordings: falling back to TRAINING CLIPS -- content will not match")
        print(f"  the identity test will be reported as uninterpretable")
        print(f"  ({len(matched)} of {len(sentences)} recorded so far)")
        for i, (clip_id, audio, clip_rate) in enumerate(real_clips(args.profile_id, len(sentences))):
            slug = f"r{i+1}"
            audio = resample(audio, clip_rate, 24_000)
            samples["real"][slug] = write(WORK / "real" / f"{slug}.wav", audio)
            reals.append((slug, audio, 24_000))
            print(f"    {slug}: {len(audio)/24_000:5.2f}s")

    # 2. vocoded control
    print("\nvocoder control (real speech through mel + vocoder):")
    samples["vocoded"] = {}
    for slug, audio, clip_rate in reals:
        done = vocode(audio, clip_rate, engine)
        if done is None:
            break
        samples["vocoded"][slug] = write(WORK / "vocoded" / f"{slug}.wav", done)
        print(f"    {slug}: {len(done)/24000:5.2f}s")
    if not samples["vocoded"]:
        del samples["vocoded"]
        print("    none produced — the channel confound will be uncontrolled")

    # 3. ours
    if not args.skip_synthesis:
        print("\nChatterbox Multilingual:")
        got = synthesize(engine, reference, profile.reference_text, rate, sentences)
        samples["ours"] = {s: write(WORK / "ours" / f"{s}.wav", a) for s, a in got.items()}
        engine.unload()

    # 4. anything else already on disk -- an IndicF5 baseline kept from before the
    # switch, or a competitor's renders of the same sentences.
    for extra in sorted(WORK.glob("*/")):
        name = extra.name
        if name in samples or name == REAL_HELDOUT.name:
            continue
        found = {s.slug: extra / f"{s.slug}.wav" for s in sentences}
        found = {slug: path for slug, path in found.items() if path.exists()}
        if len(found) >= max(3, len(sentences) // 2):
            samples[name] = found
            print(f"\nbaseline on disk: {name} ({len(found)} sentences)")

    if len(samples) < 2:
        print("\nnot enough conditions to compare")
        return 1

    bench = build(
        samples,
        real_systems=("real",),
        # Identity asks "did a person say this". It is meaningful for the speaker's
        # own voice and its imitations, and meaningless for a third-party voice --
        # so competitor samples dropped in later join naturalness only.
        identity_systems=tuple(k for k in samples if k in ("real", "vocoded", "ours")),
    )

    print(f"\nbenchmark {bench.benchmark_id}")
    print(f"  conditions : {', '.join(sorted(samples))}")
    print(f"  items      : {sum(1 for i in bench.items if i.kind == IDENTITY)} identity, "
          f"{sum(1 for i in bench.items if i.kind == NATURALNESS)} naturalness")
    print(f"  listen at  : http://127.0.0.1:8823/listen")
    print(f"  results at : http://127.0.0.1:8823/api/listen/latest/results")
    print("\nTo add a competitor or baseline: put WAVs of the same held-out")
    print(f"  sentences in {WORK}/<name>/<slug>.wav, then re-run.")
    print("  Add --skip-synthesis to reassemble without re-rendering.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
