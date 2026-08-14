"""SpeechArenaBench: the human evaluation this project did not have to run.

    uv run python -m voiceagent.eval.arena votes      # fetch + fit the ranking
    uv run python -m voiceagent.eval.arena clips -n 300   # fetch audio + ratings

AI4Bharat released 120K pairwise comparisons from 1,900 native raters over seven
TTS systems (arXiv 2604.21481, dataset MIT, gated behind a click-through). Its
README names our exact use as intended: "Benchmarking new TTS systems against
the released leaderboard using Bradley-Terry-style modeling."

This project's own blind harness has been stuck at zero listeners for months and
`abtest.results()` correctly refuses a verdict under 20 ratings per system.
Recruiting raters was costed at 2-3 weeks. This module is the alternative.

WHAT IS AND IS NOT POSSIBLE HERE. Nobody has ever heard this project's engine,
and this dataset cannot change that -- it contains preferences over *their*
seven systems. What it does contain is, for every clip, both the audio and the
six-axis human rating of that audio. That makes it a calibration set: run our
objective scorer over their clips, and we learn what our number is worth in
human terms, on the one axis our scorer measures. Then our own score means
something.

So the honest shape of the verdict is:

  - Intelligibility: placeable. Our round-trip scorer measures it, their raters
    rated it, and the two can be regressed against each other on identical audio.
  - Naturalness / expressiveness: NOT placeable, at any sample size, because we
    have no instrument for it. Round-trip overlap is blind to prosody -- a
    monotone robot that pronounces every phoneme correctly scores at ceiling.

Publishing the second half of that is the point. §1.1 of plan.md is what happens
to projects that claim a quality win they cannot support.

Data volume is the reason this module exists at all rather than a `datasets`
one-liner. The Hindi config is 32.5 GiB because every pairwise row embeds both
clips; the whole release is 241 GiB. The votes are a few MiB of that. pyarrow
reads Parquet column-wise, so projecting to the text columns fetches those
column chunks over HTTP range requests and never touches the audio.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Literal

from voiceagent.eval.arena_bt import Comparison, fit, format_table

REPO = "ai4bharat/SpeechArenaBench"
SHARDS = 70
LANGUAGE = "hi"

#: Where the cached slice lands. Text only -- audio is written beside it on
#: demand by `clips`, and is deliberately not committed.
CACHE = Path("eval_out/arena")

#: The six axes each rater scored, from the dataset README.
AXES = (
    "intelligibility",
    "expressiveness",
    "voice_quality",
    "liveliness",
    "hallucinations",
    "noise",
)

#: Published Bradley-Terry scores, Table 4 of arXiv 2604.21481, all ten
#: languages pooled. Kept here as the reference our Hindi-only fit is checked
#: against -- not as the answer, because a pooled-language number is not a Hindi
#: number and this table is the only one the paper published. Hindi-only scores
#: appear in their Figure 1 as a picture with no table behind it, which is why
#: fitting them ourselves is additive rather than duplicative.
PUBLISHED_BT: dict[str, dict[str, float]] = {
    "Gemini 2.5 Pro TTS": {"codemixed": 1135.45, "normalized": 1120.12, "symbolic": 1143.68},
    "Eleven Labs v3": {"codemixed": 1054.00, "normalized": 1059.28, "symbolic": 1044.37},
    "Sonic 3": {"codemixed": 1054.74, "normalized": 1049.68, "symbolic": 1049.42},
    "Bulbul V3 Beta": {"codemixed": 1031.28, "normalized": 1012.58, "symbolic": 1048.20},
    "Speech 2.8 HD": {"codemixed": 982.76, "normalized": 1011.02, "symbolic": 958.15},
    "GPT 4o Mini TTS": {"codemixed": 951.42, "normalized": 934.76, "symbolic": 970.75},
    "Indic F5": {"codemixed": 812.54, "normalized": 849.75, "symbolic": 785.42},
}

#: What this project's engine is called in the output tables. Named rather than
#: "ours" so a reader of a pasted table can tell what it was.
OUR_SYSTEM = "Chatterbox Multilingual 8-bit (this repo)"

#: The six axes are BINARY, whatever the documentation says.
#:
#: The dataset card and README present them as perceptual rating scales, and the
#: values are int64 in a range that looks like 1-5. Measured over 654 rated
#: clips, every axis takes exactly two values -- 1 and 5, with nothing in
#: between:
#:
#:   intelligibility {1: 70, 5: 160}   expressiveness {1: 100, 5: 130}
#:   voice_quality   {1: 94, 5: 136}   liveliness     {1: 96, 5: 134}
#:   hallucinations  {1: 59, 5: 171}   noise          {1: 42, 5: 188}
#:
#: So a rater gave each axis a thumbs up or down, and a per-system "mean rating
#: of 4.48" is not a score out of five -- it is an 87% good-rate wearing a
#: disguise. Treating it as continuous invites two mistakes: reading small
#: differences in the mean as quality gradations, and reaching for Pearson,
#: which on a two-valued outcome is a point-biserial coefficient whose magnitude
#: is capped by the class balance and is not comparable across systems with
#: different balances.
#:
#: The honest instrument for "does our number predict this judgement" is AUC.
GOOD_RATING = 4

Subset = Literal["codemixed", "symbolic", "normalized"]

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")
#: Digits in either script, plus the operators and sub/superscripts the
#: "symbolic" subset is defined by.
_SYMBOLIC = re.compile(r"[0-9०-९=+×÷%⁻₀-₉⁰-⁹°]")


def classify_subset(sentence: str) -> Subset:
    """Which of the three input conditions a sentence belongs to.

    THIS IS A HEURISTIC, NOT A DATASET LABEL. The paper defines three structured
    subsets and reports Table 4 against them, but the released Hindi parquet has
    no subset column -- there is `academic_prompt_id` and nothing that maps it.
    So the label is reconstructed from the text, using the README's own
    definitions:

      Code-mixed  intra-sentential English insertions, transliteration-based
                  mixing, and mixed-script sentences
      Symbolic    raw numerals, formulas and operators retained
      Normalized  numerals, equations and acronyms fully verbalized

    Latin script is tested first because it is the defining feature of all three
    code-mixed varieties the README lists, including fully romanised Hinglish
    ("Organic synthesis lab mein, students aksar...") which contains no
    Devanagari at all. A sentence with both Latin text and a formula is counted
    code-mixed; that ordering is a choice, and it is the one that matches the
    README's "mixed-script" clause.

    Where this can be wrong: a Devanagari sentence quoting a bare acronym in
    Latin gets called code-mixed, and a verbalised-numeral sentence that happens
    to contain a stray digit gets called symbolic. Report subset counts next to
    any subset-split result so a reader can see the classifier's split rather
    than trusting it.
    """
    if _LATIN.search(sentence):
        return "codemixed"
    if _SYMBOLIC.search(sentence):
        return "symbolic"
    return "normalized"


def parse_preference(preference: str, model_a: str, model_b: str) -> str | None:
    """Map the released `preference_model` string onto a Bradley-Terry outcome.

    The column holds model *names*, not "Model A"/"Model B", and encodes the two
    tie kinds structurally rather than with a keyword:

        "Speech 2.8 HD"                     -> that side won
        "Bulbul V3 Beta, Speech 2.8 HD"     -> both good (both names present)
        "Tie / No Preference"               -> both bad / no preference

    Returns None for a row that names neither system, which is a data problem
    rather than a tie and must not be silently counted as one.
    """
    text = (preference or "").strip()
    if not text:
        return None
    if text.lower().startswith("tie"):
        return "both_bad"

    named = {part.strip() for part in text.split(",") if part.strip()}
    hit_a, hit_b = model_a in named, model_b in named
    if hit_a and hit_b:
        return "both_good"
    if hit_a:
        return "a"
    if hit_b:
        return "b"
    return None


@dataclass(frozen=True)
class Vote:
    """One rater's judgement of one pair, with the audio left behind."""

    prompt_id: str
    sentence: str
    subset: str
    model_a: str
    model_b: str
    user_id: str
    outcome: str
    ratings_a: dict[str, int]
    ratings_b: dict[str, int]


def _shard_paths() -> list[str]:
    return [
        f"datasets/{REPO}/{LANGUAGE}/val-{i:05d}-of-{SHARDS:05d}.parquet" for i in range(SHARDS)
    ]


def _filesystem():
    from huggingface_hub import HfFileSystem

    return HfFileSystem()


#: Seconds to wait on a single shard read before giving up and retrying.
#:
#: Not a guess. The 70-shard fetch hung twice with an ESTABLISHED TCP connection
#: to the CDN delivering 880 bytes/second -- a dead socket that neither side
#: closed, with the process parked in `_ssl__SSLSocket_read` inside pyarrow's
#: footer read. There is no timeout anywhere in that stack by default, so the
#: job waits forever and looks identical to a slow one. A cold shard read takes
#: about 8 seconds, so this is generous and still bounds the failure.
SHARD_TIMEOUT_SECONDS = 120
SHARD_ATTEMPTS = 4


def _with_deadline(work, label: str) -> list[dict]:
    """Run a blocking read on a worker thread, retrying if it stalls.

    The thread cannot be killed and may linger on the dead connection. Accepted:
    it holds one socket and no lock, and the process exits once the useful work
    is done.
    """
    import concurrent.futures as futures

    last: Exception | None = None
    for attempt in range(1, SHARD_ATTEMPTS + 1):
        pool = futures.ThreadPoolExecutor(max_workers=1)
        try:
            return pool.submit(work).result(timeout=SHARD_TIMEOUT_SECONDS)
        except futures.TimeoutError as exc:
            last = exc
            print(f"    stalled after {SHARD_TIMEOUT_SECONDS}s, retry {attempt}", flush=True)
        except Exception as exc:  # transient 5xx / reset connection
            last = exc
            print(f"    {type(exc).__name__}, retry {attempt}", flush=True)
        finally:
            pool.shutdown(wait=False)

    raise RuntimeError(f"{label} failed after {SHARD_ATTEMPTS} attempts") from last


def _read_row_group(fs, pq, path: str, columns: list[str], seed: int) -> list[dict]:
    def work() -> list[dict]:
        parquet = pq.ParquetFile(fs.open(path, "rb"))
        group = seed % parquet.metadata.num_row_groups
        return parquet.read_row_group(group, columns=columns).to_pylist()

    return _with_deadline(work, path)


def _read_shard(fs, pq, path: str, columns: list[str]) -> list[dict]:
    """Read one shard's columns, retrying a stalled connection.

    The read runs on a worker thread so a hung socket can be abandoned. The
    thread itself cannot be killed and may linger blocked on the dead
    connection, which is accepted: it holds one socket and no lock, and the
    process exits normally once the useful work is done.
    """
    return _with_deadline(
        lambda: pq.ParquetFile(fs.open(path, "rb")).read(columns=columns).to_pylist(), path
    )


def _ratings(blob: dict | None, side: str) -> dict[str, int]:
    if not blob:
        return {}
    inner = blob.get(f"detailed_ratings_{side}") or {}
    return {axis: inner[axis] for axis in AXES if isinstance(inner.get(axis), int)}


def stream_votes(limit_shards: int | None = None, *, progress: bool = True) -> Iterator[Vote]:
    """Yield every Hindi vote, audio columns never requested.

    Slower than its byte count suggests, and worth knowing why before assuming
    it has hung: projecting to the text columns means fsspec issues a separate
    authenticated range request per column chunk per shard, so the cost is
    thousands of HTTPS round-trips rather than bandwidth. Measured ~20 minutes
    for all 70 shards while the local HF cache grew by 28 KB -- that 28 KB is
    the proof it is range-reading and not quietly pulling 32.5 GiB.
    """
    import pyarrow.parquet as pq

    fs = _filesystem()
    columns = [
        "academic_prompt_id",
        "sentence",
        "model_a",
        "model_b",
        "user_id",
        "preference_model",
        "fine_grained_eval",
    ]
    paths = _shard_paths()[:limit_shards]
    for index, path in enumerate(paths, 1):
        # Per-shard cache. A 70-shard fetch over a link that stalls is a job
        # that will be interrupted, and losing 20 minutes of completed shards to
        # the 60th one hanging is the difference between a tool that finishes
        # and one that is restarted all evening.
        part = CACHE / "shards" / f"{LANGUAGE}-{index:03d}.json"
        part.parent.mkdir(parents=True, exist_ok=True)
        if part.exists():
            rows = json.loads(part.read_text())
            if progress:
                print(f"  shard {index}/{len(paths)}: {len(rows)} rows (cached)", flush=True)
        else:
            rows = _read_shard(fs, pq, path, columns)
            part.write_text(json.dumps(rows, ensure_ascii=False))
            if progress:
                print(f"  shard {index}/{len(paths)}: {len(rows)} rows", flush=True)
        for row in rows:
            outcome = parse_preference(
                row.get("preference_model", ""), row["model_a"], row["model_b"]
            )
            if outcome is None:
                continue
            yield Vote(
                prompt_id=row.get("academic_prompt_id", ""),
                sentence=row["sentence"],
                subset=classify_subset(row["sentence"]),
                model_a=row["model_a"],
                model_b=row["model_b"],
                user_id=row.get("user_id", ""),
                outcome=outcome,
                ratings_a=_ratings(row.get("fine_grained_eval"), "a"),
                ratings_b=_ratings(row.get("fine_grained_eval"), "b"),
            )


def cached_votes(refresh: bool = False, limit_shards: int | None = None) -> list[Vote]:
    """Votes from the local cache, fetching once if it is not there."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"votes_{LANGUAGE}.json"
    if path.exists() and not refresh:
        return [Vote(**row) for row in json.loads(path.read_text())]

    votes = list(stream_votes(limit_shards=limit_shards))
    path.write_text(json.dumps([asdict(v) for v in votes], ensure_ascii=False))
    return votes


def comparisons_for(votes: list[Vote], subset: str | None = None) -> list[Comparison]:
    return [
        Comparison(v.model_a, v.model_b, v.outcome)  # type: ignore[arg-type]
        for v in votes
        if subset is None or v.subset == subset
    ]


@dataclass(frozen=True)
class Clip:
    """One system's rendering of one sentence, with the rating a human gave it.

    The unit of the calibration is the clip, not the system. That is forced by
    the data and it is also the stronger design: sentences are almost never
    reused across comparisons (712 distinct sentences in the first 717 votes),
    so there is no matched sentence set to put seven systems on. Pairing each
    clip with its own rater's score sidesteps that entirely and gives thousands
    of paired points instead of seven.
    """

    clip_id: str
    system: str
    sentence: str
    subset: str
    ratings: dict[str, int]
    wav: str


def harvest_clips(target: int, *, seed: int = 0) -> list[Clip]:
    """Download audio for `target` clips, spread across shards.

    One row group per shard rather than many row groups from one shard. The
    arena samples its pairs randomly, so any single shard is a thin and uneven
    draw over the seven systems -- shard 0 alone has 152 appearances of Speech
    2.8 HD against 14 of Eleven Labs v3. Spreading the reads costs nothing extra
    and evens that out.

    Audio is ~0.74 MiB per clip, so this is the expensive call in the module and
    the only one that touches the 32.5 GiB. It writes wavs under
    `eval_out/arena/clips/` and is resumable: clips already on disk are kept.
    """
    import pyarrow.parquet as pq

    fs = _filesystem()
    out_dir = CACHE / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        "sentence",
        "model_a",
        "model_b",
        "fine_grained_eval",
        "audio_a",
        "audio_b",
    ]
    manifest = CACHE / "clips.json"
    clips: list[Clip] = []
    for shard_index, path in enumerate(_shard_paths()):
        if len(clips) >= target:
            break
        # Same 120s deadline as the vote fetch, for the same reason and after
        # the same failure: this loop stalled at 562 of 600 clips on a socket
        # that stayed ESTABLISHED and delivered nothing.
        try:
            rows = _read_row_group(fs, pq, path, columns, seed)
        except RuntimeError as exc:
            print(f"  shard {shard_index}: {exc}", flush=True)
            continue

        for row in rows:
            for side in ("a", "b"):
                audio = row.get(f"audio_{side}") or {}
                blob, name = audio.get("bytes"), audio.get("path")
                if not blob or not name:
                    continue
                wav = out_dir / name
                if not wav.exists():
                    wav.write_bytes(blob)
                clips.append(
                    Clip(
                        clip_id=name,
                        system=row[f"model_{side}"],
                        sentence=row["sentence"],
                        subset=classify_subset(row["sentence"]),
                        ratings=_ratings(row.get("fine_grained_eval"), side),
                        wav=str(wav),
                    )
                )
        # After every shard, not at the end. The ratings live only in the
        # parquet, so a manifest written once at the end means an interrupted
        # harvest leaves 562 wavs on disk that are unusable -- audio with no
        # human score attached to it is not a calibration set, and rebuilding it
        # costs the whole download again.
        manifest.write_text(json.dumps([asdict(c) for c in clips], ensure_ascii=False, indent=1))
        print(f"  shard {shard_index}: {len(clips)} clips", flush=True)

    return clips


def _harvest(args: argparse.Namespace) -> int:
    clips = harvest_clips(args.n)
    systems: dict[str, int] = {}
    rated = 0
    for c in clips:
        systems[c.system] = systems.get(c.system, 0) + 1
        if c.ratings.get("intelligibility") is not None:
            rated += 1
    print(f"\n{len(clips)} clips, {rated} with an intelligibility rating")
    for name, count in sorted(systems.items(), key=lambda kv: -kv[1]):
        print(f"  {name:22s} {count:5d}")
    return 0


def score_clips(clips: list[Clip], *, resume: bool = True) -> list[dict]:
    """Run this project's round-trip scorer over arena audio.

    The same scorer, unmodified, that every Hindi claim in this repo rests on:
    transcribe with Whisper, normalise both sides into Devanagari, take
    character overlap. Running it over *their* clips is what converts it from a
    number about us into a number with a known human meaning.

    Scoring is the slow half of this module (Whisper at RTF 0.24 over clips of a
    few seconds) so results are checkpointed after every clip and `resume`
    picks up where an interrupted run stopped.
    """
    from voiceagent.eval.roundtrip import character_overlap, decode_for_scoring, normalized

    path = CACHE / "scores.json"
    done: dict[str, dict] = {}
    if resume and path.exists():
        done = {row["clip_id"]: row for row in json.loads(path.read_text())}

    for index, clip in enumerate(clips, 1):
        if clip.clip_id in done:
            continue
        try:
            heard, language, note = decode_for_scoring(clip.wav, "hi")
        except Exception as exc:  # a corrupt clip must not lose the whole run
            done[clip.clip_id] = {**asdict(clip), "error": f"{type(exc).__name__}: {exc}"}
            continue

        overlap = character_overlap(normalized(clip.sentence, "hi"), normalized(heard, "hi"))
        done[clip.clip_id] = {
            **asdict(clip),
            "heard": heard,
            "language": language,
            "note": note,
            "overlap": overlap,
        }
        if index % 10 == 0:
            path.write_text(json.dumps(list(done.values()), ensure_ascii=False, indent=1))
            print(f"  scored {len(done)}/{len(clips)}", flush=True)

    path.write_text(json.dumps(list(done.values()), ensure_ascii=False, indent=1))
    return list(done.values())


def auc(scores: list[float], labels: list[bool]) -> float:
    """Probability our score ranks a human-approved clip above a rejected one.

    The Mann-Whitney form, so ties are handled by mid-ranks rather than being
    silently broken. 0.5 is a coin flip; 1.0 is perfect separation.
    """
    from scipy import stats as st

    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return float("nan")
    ranks = st.rankdata(scores)
    positive_rank_sum = sum(r for r, is_good in zip(ranks, labels) if is_good)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def correlate(rows: list[dict]) -> dict:
    """Can our round-trip score predict a native speaker's verdict?

    This is the question the whole module exists to answer, and a null result is
    a real one: if our number cannot separate clips humans approved from clips
    they rejected, then it cannot place a system against these raters, and the
    honest move is to say so rather than publish a position it does not support.

    Reported three ways, because the first one alone is misleading:

      overall AUC     over every clip, including the broken ones
      cloud-only AUC  with IndicF5 dropped

    The second is the one that matters. A correlation across all systems can be
    manufactured entirely by a single broken outlier: six systems in a narrow
    band plus one far below on both axes is a line through a cluster and a dot.
    It reports a healthy number while having no resolving power anywhere a real
    decision gets made. IndicF5 is precisely that dot -- 0.53 mean overlap and a
    14% good-rate against a pack at 0.85-0.89 and 70-87%.
    """
    from scipy import stats

    # `is not None`, not truthiness: a rating of 0 is a rating, and the axis
    # scale is not documented as 1-based. Dropping zeros would quietly discard
    # exactly the clips the raters thought were worst, which is the end of the
    # range the correlation most depends on.
    usable = [
        r
        for r in rows
        if "overlap" in r and r.get("ratings", {}).get("intelligibility") is not None
    ]
    if len(usable) < 3:
        return {"clips": len(usable), "error": "not enough rated clips to correlate"}

    ours = [r["overlap"] for r in usable]
    human = [r["ratings"]["intelligibility"] for r in usable]
    pearson = stats.pearsonr(ours, human)
    spearman = stats.spearmanr(ours, human)

    by_system: dict[str, list[tuple[float, bool]]] = {}
    for r in usable:
        by_system.setdefault(r["system"], []).append(
            (r["overlap"], r["ratings"]["intelligibility"] >= GOOD_RATING)
        )
    systems = {
        name: {
            "clips": len(pairs),
            "our_overlap": sum(p[0] for p in pairs) / len(pairs),
            # The share of clips a native speaker called intelligible. This is
            # what the "mean rating" actually was.
            "human_good_rate": sum(1 for p in pairs if p[1]) / len(pairs),
        }
        for name, pairs in sorted(by_system.items())
    }

    result = {
        "clips": len(usable),
        "clip_pearson_r": pearson.statistic,
        "clip_pearson_p": pearson.pvalue,
        "clip_spearman_r": spearman.statistic,
        "clip_spearman_p": spearman.pvalue,
        "systems": systems,
    }
    if len(systems) >= 3:
        sys_ours = [s["our_overlap"] for s in systems.values()]
        sys_human = [s["human_good_rate"] for s in systems.values()]
        sys_r = stats.pearsonr(sys_ours, sys_human)
        result["system_pearson_r"] = sys_r.statistic
        result["system_pearson_p"] = sys_r.pvalue

    labels = [r["ratings"]["intelligibility"] >= GOOD_RATING for r in usable]
    result["good_rate"] = sum(labels) / len(labels)
    result["auc"] = auc([r["overlap"] for r in usable], labels)

    # The check that decides what may be claimed. See the docstring.
    cloud = [r for r in usable if r["system"] != "Indic F5"]
    if cloud:
        cloud_labels = [r["ratings"]["intelligibility"] >= GOOD_RATING for r in cloud]
        result["cloud_clips"] = len(cloud)
        result["cloud_good_rate"] = sum(cloud_labels) / len(cloud_labels)
        result["cloud_auc"] = auc([r["overlap"] for r in cloud], cloud_labels)
    return result


def _score(args: argparse.Namespace) -> int:
    manifest = CACHE / "clips.json"
    if not manifest.exists():
        print("no clips yet -- run `arena clips` first", file=sys.stderr)
        return 1
    clips = [Clip(**row) for row in json.loads(manifest.read_text())][: args.n]
    rows = score_clips(clips, resume=not args.restart)
    stats_out = correlate(rows)
    (CACHE / "correlation.json").write_text(json.dumps(stats_out, indent=1))

    print(json.dumps({k: v for k, v in stats_out.items() if k != "systems"}, indent=1))
    print("\ncan our round-trip score predict a native speaker's verdict?")
    print(
        f"  all clips      AUC {stats_out['auc']:.3f}   "
        f"(n={stats_out['clips']}, {stats_out['good_rate']:.0%} rated intelligible)"
    )
    if "cloud_auc" in stats_out:
        print(
            f"  IndicF5 removed AUC {stats_out['cloud_auc']:.3f}   "
            f"(n={stats_out['cloud_clips']}, {stats_out['cloud_good_rate']:.0%} rated intelligible)"
        )

    print("\nper system:")
    print(f"  {'system':22s} {'clips':>6} {'our overlap':>12} {'human good-rate':>16}")
    for name, row in sorted(
        stats_out.get("systems", {}).items(),
        key=lambda kv: -kv[1]["human_good_rate"],
    ):
        print(
            f"  {name:22s} {row['clips']:6d} {row['our_overlap']:12.3f} "
            f"{row['human_good_rate']:15.0%}"
        )
    return 0


REFERENCE_WAV = Path("fixtures/hi/reference_lekha.wav")


async def render_ours(sentences: list[tuple[str, str]]) -> list[dict]:
    """Synthesize arena sentences with this project's engine and score them.

    Same engine, same normalisation and same scorer the rest of the repo uses --
    `ChatterboxIndicEngine` driven exactly as `eval.hindi_tts` drives it, so a
    regression in engine construction shows up here too rather than being
    papered over by a bespoke path.

    Hindi has no built-in speaker: Chatterbox clones, so it is silent until a
    reference voice is enrolled. That makes the enrolled clip part of the
    result, and it is the single largest caveat on comparing our number to a
    cloud system's -- they were not conditioned on one amateur recording of one
    speaker, and we were.
    """
    import numpy as np
    import soundfile as sf

    from voiceagent.eval.roundtrip import character_overlap, decode_for_scoring, normalized
    from voiceagent.text.normalize_hi import normalize as normalize_hi
    from voiceagent.tts.chatterbox_indic import ChatterboxIndicEngine

    out_dir = CACHE / "ours"
    out_dir.mkdir(parents=True, exist_ok=True)

    reference, sample_rate = sf.read(str(REFERENCE_WAV), dtype="float32")
    if reference.ndim > 1:
        reference = reference.mean(axis=1)

    engine = ChatterboxIndicEngine()
    engine.set_reference(reference, REFERENCE_WAV.with_suffix(".txt").read_text().strip(), sample_rate)
    engine.load()

    rows: list[dict] = []
    for index, (prompt_id, sentence) in enumerate(sentences, 1):
        wav = out_dir / f"{prompt_id or index}.wav"
        spoken = normalize_hi(sentence)
        chunks = [c async for c in engine.synthesize(spoken)]
        if not chunks:
            rows.append({"prompt_id": prompt_id, "sentence": sentence, "error": "no audio"})
            continue

        samples = np.concatenate([c.samples for c in chunks])
        sf.write(wav, samples, chunks[0].sample_rate)
        seconds = len(samples) / chunks[0].sample_rate
        latency_ms = chunks[0].latency_ms

        heard, language, note = decode_for_scoring(wav, "hi")
        rows.append(
            {
                "prompt_id": prompt_id,
                "sentence": sentence,
                "subset": classify_subset(sentence),
                "system": OUR_SYSTEM,
                "heard": heard,
                "language": language,
                "note": note,
                "overlap": character_overlap(normalized(sentence, "hi"), normalized(heard, "hi")),
                "rtf": (latency_ms / 1000) / seconds if latency_ms and seconds else None,
                "wav": str(wav),
            }
        )
        if index % 5 == 0:
            print(f"  rendered {index}/{len(sentences)}", flush=True)
            (CACHE / "ours.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))

    (CACHE / "ours.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    return rows


def _render(args: argparse.Namespace) -> int:
    import asyncio

    if not REFERENCE_WAV.exists():
        print(
            f"missing {REFERENCE_WAV} -- Hindi needs an enrolled voice, and this "
            "clip is untracked because it is a real person's recording.",
            file=sys.stderr,
        )
        return 2

    votes = cached_votes()
    seen: dict[str, str] = {}
    for v in votes:
        if v.subset == args.subset and v.prompt_id not in seen:
            seen[v.prompt_id] = v.sentence
    chosen = list(seen.items())[: args.n]
    if not chosen:
        print(f"no {args.subset} sentences in the cache", file=sys.stderr)
        return 1

    rows = asyncio.run(render_ours(chosen))
    scored = [r for r in rows if "overlap" in r]
    if scored:
        mean = sum(r["overlap"] for r in scored) / len(scored)
        rtfs = [r["rtf"] for r in scored if r.get("rtf")]
        print(f"\n{OUR_SYSTEM} on {len(scored)} {args.subset} arena sentences")
        print(f"  mean round-trip overlap : {mean:.3f}")
        if rtfs:
            print(f"  median RTF              : {sorted(rtfs)[len(rtfs) // 2]:.2f}")
    return 0


def _compare(args: argparse.Namespace) -> int:
    """Place this project's engine beside the seven, and say what that is worth.

    Paired where possible: most of the sentences we rendered were also rendered
    by the arena systems, so the comparison can hold the sentence fixed instead
    of comparing two different sentence samples and hoping they average out.
    """
    import statistics

    scored = [r for r in json.loads((CACHE / "scores.json").read_text()) if "overlap" in r]
    ours = [r for r in json.loads((CACHE / "ours.json").read_text()) if "overlap" in r]
    if not scored or not ours:
        print("run `arena score` and `arena ours` first", file=sys.stderr)
        return 1

    code_mixed = [r for r in scored if r["subset"] == "codemixed"]
    stats_out = correlate(scored)

    print("=== 1. Same scorer, same input condition (code-mixed) ===\n")
    print(f"  {'system':44s} {'n':>4} {'overlap':>8} {'human':>10}")
    table: list[tuple[str, int, float, float | None]] = []
    grouped: dict[str, list[dict]] = {}
    for r in code_mixed:
        grouped.setdefault(r["system"], []).append(r)
    for name, rows in grouped.items():
        good = sum(1 for r in rows if r["ratings"]["intelligibility"] >= GOOD_RATING) / len(rows)
        table.append((name, len(rows), statistics.mean(r["overlap"] for r in rows), good))
    table.append((OUR_SYSTEM, len(ours), statistics.mean(r["overlap"] for r in ours), None))
    for name, count, overlap, good in sorted(table, key=lambda t: -t[2]):
        human = f"{good:9.0%}" if good is not None else "  no rater"
        print(f"  {name:44s} {count:4d} {overlap:8.3f} {human}")

    # Paired: hold the sentence fixed.
    by_sentence: dict[str, list[dict]] = {}
    for r in scored:
        by_sentence.setdefault(r["sentence"], []).append(r)
    wins: dict[str, list[int]] = {}
    for row in ours:
        for other in by_sentence.get(row["sentence"], []):
            wins.setdefault(other["system"], []).append(
                1 if row["overlap"] > other["overlap"] else 0
            )
    if wins:
        print("\n=== 2. Paired on identical sentences: how often do we score higher? ===\n")
        for name, results in sorted(wins.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
            print(
                f"  vs {name:42s} {sum(results):3d}/{len(results):<3d} "
                f"({sum(results) / len(results):.0%})"
            )

    print("\n=== 3. What that is worth ===\n")
    print(f"  Our scorer predicts a native speaker's verdict at AUC {stats_out['auc']:.3f}")
    print(
        f"  With the one broken system removed, AUC {stats_out['cloud_auc']:.3f} "
        f"-- a coin flip is 0.500"
    )
    print(
        "\n  So table 1 ranks nothing inside the working band. It is read as: this\n"
        "  engine is in the band, and is nowhere near the failure case. Any\n"
        "  stronger reading is unsupported by the instrument that produced it."
    )
    return 0


def _report_votes(args: argparse.Namespace) -> int:
    votes = cached_votes(refresh=args.refresh, limit_shards=args.shards)
    if not votes:
        print("no votes -- is the dataset gate accepted for this account?", file=sys.stderr)
        return 1

    counts: dict[str, int] = {}
    for v in votes:
        counts[v.subset] = counts.get(v.subset, 0) + 1
    raters = len({v.user_id for v in votes if v.user_id})
    sentences = len({v.prompt_id for v in votes if v.prompt_id})

    print(f"{len(votes)} Hindi votes, {raters} raters, {sentences} sentences")
    print("subsets (heuristic, see classify_subset): " + ", ".join(
        f"{k}={v}" for k, v in sorted(counts.items())
    ))

    for subset in (None, "codemixed"):
        label = "ALL HINDI" if subset is None else "HINDI CODE-MIXED"
        scored = fit(comparisons_for(votes, subset), bootstrap=args.bootstrap)
        print(f"\n=== {label} ===")
        print(format_table(scored))
        if subset:
            print("\npublished Table 4 codemixed (all 10 languages pooled), for reference:")
            for name, row in sorted(
                PUBLISHED_BT.items(), key=lambda kv: kv[1]["codemixed"], reverse=True
            ):
                print(f"  {name:22s} {row['codemixed']:8.2f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    votes = sub.add_parser("votes", help="fetch the preference data and fit the ranking")
    votes.add_argument("--refresh", action="store_true", help="re-fetch instead of using the cache")
    votes.add_argument("--shards", type=int, default=None, help="stop after N shards (a smoke test)")
    votes.add_argument("--bootstrap", type=int, default=500, help="bootstrap resamples for the CI")
    votes.set_defaults(func=_report_votes)

    clips = sub.add_parser("clips", help="download arena audio and its human ratings")
    clips.add_argument("-n", type=int, default=300, help="how many clips to fetch")
    clips.set_defaults(func=_harvest)

    score = sub.add_parser("score", help="round-trip the arena clips and correlate with raters")
    score.add_argument("-n", type=int, default=None, help="score only the first N clips")
    score.add_argument("--restart", action="store_true", help="ignore checkpointed scores")
    score.set_defaults(func=_score)

    ours = sub.add_parser("ours", help="render arena sentences with this project's engine")
    ours.add_argument("-n", type=int, default=40, help="how many sentences")
    ours.add_argument("--subset", default="codemixed", help="which input condition")
    ours.set_defaults(func=_render)

    compare = sub.add_parser("compare", help="the verdict: place our engine and caveat it")
    compare.set_defaults(func=_compare)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
