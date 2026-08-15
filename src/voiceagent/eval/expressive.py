"""Sweep the expressiveness knobs, so prosody can be chosen instead of assumed.

`EXAGGERATION = 0.7` has been the setting for every Hindi sentence this project
has ever produced. It was inherited from the Praxy recipe, never swept, and
never listened to against an alternative. This module renders the same sentences
across a grid of settings so the choice can be made by ear, blind.

**Why this cannot be scored automatically, and why that is the point.** The
project's round-trip scorer is blind to prosody by construction: it asks whether
Whisper recovers the words, and a flat monotone reading with every phoneme
correct scores at ceiling. Phase 2 then measured what that scorer is worth ---
AUC 0.625 against human verdicts among systems that work, near the 0.500 coin
flip. So there is no instrument here that can rank expressiveness, and inventing
one by reusing overlap would be the exact error `eval_out/arena/FINDINGS.md`
documents. Round trip is still computed per clip, but only as an *alarm*: a
setting that mangles the words is disqualified regardless of how lively it
sounds, and that is the only question overlap is allowed to answer.

The verdict comes from `abtest`, blind, from listeners. This module's job is to
produce the audio and get out of the way.

    uv run python -m voiceagent.eval.expressive render     # audio + manifest
    uv run python -m voiceagent.eval.expressive listen     # blind benchmark
    uv run python -m voiceagent.eval.expressive report     # what was rated

One thing to know before reading a result: **exaggeration is not a quality
dial.** Upstream describes it as emotion intensity, and it is widely said to
speed delivery up as well as liven it. Measured here over 6 sentences x 5
settings, that is not what happens: RTF is flat across the grid (0.52--0.58) and
mean duration *rises* slightly with exaggeration, 3.71 s at 0.3 to 3.83 s at
0.9 --- though the longest sentence inverts the trend, so the honest statement is
that duration moves a little and not consistently in one direction. Speed is
recorded per condition anyway, because a setting that sounds better and misses
real time is a different decision from one that is free.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from voiceagent.eval import abtest, heldout

ROOT = Path(__file__).resolve().parents[3] / "eval_out" / "expressive"

#: The default grid. Deliberately one axis at a time around the shipped setting
#: (0.7 / 0.6), plus one off-axis point:
#:
#:   * Four exaggeration values bracket the current one in both directions, so
#:     the sweep can say "lower is better" rather than only "0.9 is worse".
#:   * `warm` moves temperature alone, to separate liveliness from variability.
#:     They are easily confused by ear and they fail differently --- temperature
#:     buys variation at the cost of stability, and an unstable setting shows up
#:     as a bad clip in twenty, which a six-sentence sweep will under-sample.
#:
#: Kept to five: every condition multiplies listening time, and a listener who
#: stops paying attention produces worse data than one who was never asked.
CONDITIONS: tuple[dict, ...] = (
    {"name": "flat", "exaggeration": 0.3, "temperature": 0.6},
    {"name": "mid", "exaggeration": 0.5, "temperature": 0.6},
    {"name": "shipped", "exaggeration": 0.7, "temperature": 0.6},
    {"name": "high", "exaggeration": 0.9, "temperature": 0.6},
    {"name": "warm", "exaggeration": 0.7, "temperature": 0.8},
)

#: Six of the twelve held-out sentences. Chosen for spread across the failure
#: modes those sentences target rather than for being flattering --- prosody is
#: most audible where the sentence is hard, and `h12` in particular is the long
#: one with an internal clause, which is where a flat reading gives itself away.
DEFAULT_SLUGS: tuple[str, ...] = ("h1", "h4", "h6", "h8", "h10", "h12")

#: Below this, the words did not survive and the setting is out regardless of
#: how it sounds. Not a ranking threshold --- see the module docstring. The value
#: is the one Phase 2 validated: every arena clip under 0.5 overlap was rejected
#: by human raters too, 11 of 11.
DISQUALIFYING_OVERLAP = 0.5


@dataclass(frozen=True)
class Rendered:
    """One clip: which setting produced it, and what it cost."""

    condition: str
    slug: str
    path: str
    seconds: float
    synthesis_seconds: float
    rtf: float
    overlap: float | None = None

    @property
    def suspect(self) -> bool:
        return self.overlap is not None and self.overlap < DISQUALIFYING_OVERLAP


def _reference() -> tuple[np.ndarray, str, int]:
    """The enrolled Indic voice, decrypted.

    Fails loudly rather than falling back to an English clip: conditioning Hindi
    on an English reference is the accent-bleed failure the held-out set was
    built to expose, and silently doing it would poison every row of the sweep.
    """
    import soundfile as sf

    from voiceagent.tts.router import pick_indic_profile
    from voiceagent.voice_clone.store import VoiceProfileStore

    store = VoiceProfileStore()
    profile = pick_indic_profile(store.list())
    if profile is None:
        raise SystemExit(
            "No enrolled voice. Chatterbox clones and ships no default speaker, "
            "so there is nothing to sweep. Enrol one in `voice-web` first."
        )
    audio, rate = sf.read(io.BytesIO(store.reference_audio(profile.profile_id)), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, profile.reference_text or "", rate


def render(
    slugs: tuple[str, ...] = DEFAULT_SLUGS,
    conditions: tuple[dict, ...] = CONDITIONS,
    score: bool = True,
) -> Path:
    """Render every condition over every sentence into a timestamped directory.

    The model is loaded **once** and the knobs are moved between renders, which
    is what makes a five-way sweep affordable: a reload per condition would cost
    more than the synthesis. Conditioning is rebuilt when exaggeration changes
    --- upstream does update a cached `emotion_adv` in place, but relying on a
    dependency mutating this engine's private cache is exactly the coupling that
    breaks quietly on upgrade, and rebuilding costs one embedding per condition.
    """
    import soundfile as sf

    from voiceagent.tts.chatterbox_indic import SAMPLE_RATE, ChatterboxIndicEngine

    out = ROOT / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    (out / "audio").mkdir(parents=True, exist_ok=True)

    sentences = [heldout.by_slug(s) for s in slugs]
    audio, ref_text, rate = _reference()

    engine = ChatterboxIndicEngine()
    engine.set_reference(audio, ref_text, rate)
    engine.load()

    rows: list[Rendered] = []
    for condition in conditions:
        engine.exaggeration = condition["exaggeration"]
        engine.temperature = condition["temperature"]
        #: Force the conditionals to be rebuilt at the new exaggeration. See the
        #: docstring: correctness here is worth one mel per condition.
        engine._conds = None

        for sentence in sentences:
            started = time.perf_counter()
            #: Through `_run`, never called directly. MLX arrays are thread-affine
            #: and MLX is lazy, so building a graph on the wrong thread succeeds
            #: and only `mx.eval` touches the stream --- an off-thread bug then
            #: surfaces twenty layers from its cause. The engine owns one thread
            #: for load and generate and everything must go through it.
            samples = engine._run(engine._generate_blocking, sentence.text)
            elapsed = time.perf_counter() - started

            path = out / "audio" / f"{condition['name']}__{sentence.slug}.wav"
            sf.write(path, samples, SAMPLE_RATE)
            seconds = len(samples) / SAMPLE_RATE

            rows.append(
                Rendered(
                    condition=condition["name"],
                    slug=sentence.slug,
                    path=str(path.relative_to(out)),
                    seconds=round(seconds, 3),
                    synthesis_seconds=round(elapsed, 3),
                    rtf=round(elapsed / seconds, 3) if seconds else 0.0,
                )
            )
            print(f"  {condition['name']:>8} {sentence.slug:>4}  {seconds:5.2f}s  RTF {rows[-1].rtf:.2f}")

    if score:
        rows = _score(rows, out, sentences)

    (out / "manifest.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "conditions": list(conditions),
                "slugs": list(slugs),
                "reference_text": ref_text,
                "rows": [asdict(r) for r in rows],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _summarise(rows, conditions)
    print(f"\n  {out}")
    return out


def _score(rows: list[Rendered], out: Path, sentences: list) -> list[Rendered]:
    """Attach round-trip overlap, as an alarm and never as a ranking.

    A failure to score is recorded as `None` rather than as a zero: zero is a
    disqualifying value, and a scorer that could not run is not evidence that a
    clip is bad.
    """
    from voiceagent.eval.roundtrip import character_overlap, decode_for_scoring, normalized

    texts = {s.slug: s.text for s in sentences}
    scored: list[Rendered] = []
    for row in rows:
        try:
            #: Pinned to `hi`. Auto-detect mislabelled a 1.7 s Hindi clip as
            #: Korean and scored it 0 %, and several clips here are short.
            heard, _language, _note = decode_for_scoring(out / row.path, "hi")
            overlap = character_overlap(
                normalized(texts[row.slug], "hi"), normalized(heard, "hi")
            )
        except Exception:  # noqa: BLE001
            overlap = None
        scored.append(Rendered(**{**asdict(row), "overlap": overlap}))
    return scored


def _summarise(rows: list[Rendered], conditions: tuple[dict, ...]) -> None:
    print("\n  condition   exag  temp    RTF   mean s   overlap   suspect")
    for condition in conditions:
        mine = [r for r in rows if r.condition == condition["name"]]
        if not mine:
            continue
        overlaps = [r.overlap for r in mine if r.overlap is not None]
        rtf = sum(r.rtf for r in mine) / len(mine)
        secs = sum(r.seconds for r in mine) / len(mine)
        mean_overlap = sum(overlaps) / len(overlaps) if overlaps else float("nan")
        suspect = sum(1 for r in mine if r.suspect)
        print(
            f"  {condition['name']:>9}  {condition['exaggeration']:.1f}   "
            f"{condition['temperature']:.1f}   {rtf:.2f}   {secs:5.2f}   "
            f"{mean_overlap:6.3f}    {suspect}"
        )
    print(
        "\n  Overlap is an alarm, not a ranking: it is blind to prosody and cannot\n"
        "  order these. Anything in `suspect` mangled the words and is out. The\n"
        "  rest is a listening question --- run `listen`."
    )


def latest() -> Path:
    """The most recent render, or a clear error saying to make one."""
    runs = sorted(p for p in ROOT.glob("*") if (p / "manifest.json").exists())
    if not runs:
        raise SystemExit("No sweep yet. Run `expressive render` first.")
    return runs[-1]


def listen(run: Path | None = None) -> str:
    """Load a rendered sweep into the blind harness and return its id.

    Conditions ride in as `abtest` "systems", which is what buys the blinding for
    free: the listener is served an opaque item id, so the setting cannot leak
    through a filename or a play order.

    `identity_systems=()` suppresses the real-or-synthetic question. Every clip
    here is synthetic and the listener knows it, so asking would only teach them
    to answer "synthetic" and waste half the session.
    """
    run = run or latest()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))

    samples: dict[str, dict[str, Path]] = {}
    for row in manifest["rows"]:
        if row.get("overlap") is not None and row["overlap"] < DISQUALIFYING_OVERLAP:
            #: Kept out of the listening set rather than rated and discarded: a
            #: clip whose words are mangled is not a prosody question, and it
            #: spends a listener's attention on a foregone conclusion.
            continue
        samples.setdefault(row["condition"], {})[row["slug"]] = run / row["path"]

    benchmark = abtest.build(samples, real_systems=(), identity_systems=())
    print(f"  benchmark {benchmark.benchmark_id}: {len(benchmark.items)} clips")
    print(f"  rate them at  http://127.0.0.1:8823/listen  ->  How natural")
    return benchmark.benchmark_id


def report(benchmark_id: str | None = None) -> None:
    """Print what listeners said, with the harness's own refusal to over-call.

    `abtest.results` reports an interval and declines a verdict below 20 ratings
    per system, which for a five-way sweep is 100 ratings. That refusal is the
    feature: one listener's opinion of five similar clips is a hypothesis.
    """
    if benchmark_id is None:
        runs = sorted(p.name for p in abtest.ROOT.glob("*") if (p / "manifest.json").exists())
        if not runs:
            raise SystemExit("No benchmark yet. Run `expressive listen` first.")
        benchmark_id = runs[-1]
    benchmark = abtest.Benchmark.load(benchmark_id)
    print(f"  benchmark {benchmark_id}")
    print(json.dumps(benchmark.results(abtest.NATURALNESS), ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    render_cmd = sub.add_parser("render", help="synthesize the grid")
    render_cmd.add_argument("--slugs", nargs="*", default=list(DEFAULT_SLUGS))
    render_cmd.add_argument(
        "--no-score", action="store_true", help="skip the round-trip alarm (faster)"
    )

    sub.add_parser("listen", help="load the latest sweep into the blind harness")

    report_cmd = sub.add_parser("report", help="what listeners said")
    report_cmd.add_argument("benchmark_id", nargs="?")

    args = parser.parse_args(argv)
    if args.command == "render":
        render(slugs=tuple(args.slugs), score=not args.no_score)
    elif args.command == "listen":
        listen()
    else:
        report(args.benchmark_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
