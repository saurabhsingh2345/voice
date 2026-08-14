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


def _ratings(blob: dict | None, side: str) -> dict[str, int]:
    if not blob:
        return {}
    inner = blob.get(f"detailed_ratings_{side}") or {}
    return {axis: inner[axis] for axis in AXES if isinstance(inner.get(axis), int)}


def stream_votes(limit_shards: int | None = None) -> Iterator[Vote]:
    """Yield every Hindi vote, audio columns never requested."""
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
    for path in _shard_paths()[:limit_shards]:
        table = pq.ParquetFile(fs.open(path, "rb")).read(columns=columns)
        rows = table.to_pylist()
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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
