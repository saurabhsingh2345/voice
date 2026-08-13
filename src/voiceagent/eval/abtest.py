"""Blind A/B benchmark for synthetic voice.

The project has had no automated measure of voice quality, and the two obvious
proxies actively misled: F0 and spectral centroid ranked the sample the speaker
identified with *worst*. Listener judgement is the ground truth, so this makes
collecting it cheap and repeatable instead of ad hoc.

TWO QUESTIONS, DELIBERATELY SEPARATE

  identity     "Is this a real recording of a person, or synthesised?"
               Systems: the real recordings, our fine-tune, the stock base.
               Reports a *fooled rate* — how often a synthetic sample was called
               real. This is the claim "it sounds like me", made falsifiable.

  naturalness  "How natural is this Hindi, 1-5?"
               Systems: ours and any competitor whose samples are dropped in.
               Reports a mean opinion score. This is the claim "better Hindi than
               the incumbent", made comparable.

They are separate because they have different answers. A clone can be
indistinguishable from its speaker (high fooled rate) while still being less
natural Hindi than a competitor's stock voice, and conflating the two would hide
that.

BLINDING IS STRUCTURAL, NOT POLICY

Item ids are random and carry no information about which system produced them. The
mapping from item to system lives in `manifest.json`, which the listening UI never
reads — the UI is served item ids and audio URLs only. A blind test that depends on
the experimenter remembering not to leak the answer is not blind, and the natural
implementation (`ours_h1.wav`) leaks it in the URL bar.

Order is randomised per listener but *deterministically* from the listener id, so
reloading the page resumes the same sequence rather than reshuffling and
re-presenting items already rated.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "eval_out" / "abtest"

#: Question kinds. See the module docstring for why they are separate.
IDENTITY = "identity"
NATURALNESS = "naturalness"

#: A rating faster than this means the listener did not hear the clip out. Kept as
#: quality control rather than a hard reject: the scores are reported both ways so
#: an inattentive listener is visible instead of silently averaged in.
MIN_LISTEN_MS = 1200

#: Below this many ratings per system, report the interval and refuse to call it.
#: Ten listeners on twelve sentences is 120 ratings, which is plenty; two friends on
#: three sentences is not, and the difference should not be a judgement call.
MIN_RATINGS_FOR_A_VERDICT = 20


@dataclass(frozen=True)
class Item:
    """One audio file, with the answer attached — never served to a listener."""

    item_id: str
    system: str
    slug: str
    kind: str
    is_real: bool = False


@dataclass
class Benchmark:
    benchmark_id: str
    created_at: str
    items: list[Item] = field(default_factory=list)

    # --- paths ------------------------------------------------------------

    @property
    def dir(self) -> Path:
        return ROOT / self.benchmark_id

    @property
    def audio_dir(self) -> Path:
        return self.dir / "audio"

    @property
    def ratings_dir(self) -> Path:
        return self.dir / "ratings"

    def audio_path(self, item_id: str) -> Path:
        return self.audio_dir / f"{item_id}.wav"

    # --- persistence ------------------------------------------------------

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ratings_dir.mkdir(exist_ok=True)
        (self.dir / "manifest.json").write_text(
            json.dumps(
                {
                    "benchmark_id": self.benchmark_id,
                    "created_at": self.created_at,
                    "items": [asdict(i) for i in self.items],
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    @classmethod
    def load(cls, benchmark_id: str) -> "Benchmark":
        path = ROOT / benchmark_id / "manifest.json"
        if not path.exists():
            raise KeyError(f"no such benchmark: {benchmark_id}")
        data = json.loads(path.read_text())
        return cls(
            benchmark_id=data["benchmark_id"],
            created_at=data["created_at"],
            items=[Item(**i) for i in data["items"]],
        )

    @staticmethod
    def latest() -> str | None:
        if not ROOT.exists():
            return None
        candidates = [p.parent.name for p in ROOT.glob("*/manifest.json")]
        return sorted(candidates)[-1] if candidates else None

    # --- what a listener sees ---------------------------------------------

    def order_for(self, listener_id: str, kind: str) -> list[str]:
        """Item ids for one listener, shuffled per-listener but stably.

        Seeded from the listener id so a page reload resumes rather than reshuffles.
        Without that, a listener who refreshes is re-presented items they already
        rated, and the duplicates quietly weight their opinion twice.
        """
        ids = [i.item_id for i in self.items if i.kind == kind]
        random.Random(f"{self.benchmark_id}:{listener_id}").shuffle(ids)
        return ids

    def system_of(self, item_id: str) -> str:
        for item in self.items:
            if item.item_id == item_id:
                return item.system
        raise KeyError(item_id)

    def item(self, item_id: str) -> Item:
        for candidate in self.items:
            if candidate.item_id == item_id:
                return candidate
        raise KeyError(item_id)

    # --- ratings ----------------------------------------------------------

    def ratings_path(self, listener_id: str) -> Path:
        safe = "".join(c for c in listener_id if c.isalnum() or c in "-_")[:40] or "anon"
        return self.ratings_dir / f"{safe}.json"

    def record(self, listener_id: str, item_id: str, answer: dict) -> int:
        """Append one rating. Returns how many that listener has now given.

        Re-rating an item overwrites rather than appends, so a listener who goes
        back does not count twice.
        """
        self.item(item_id)  # raises on an unknown id
        path = self.ratings_path(listener_id)
        existing = json.loads(path.read_text()) if path.exists() else []
        existing = [r for r in existing if r["item_id"] != item_id]
        existing.append(
            {
                "item_id": item_id,
                "at": datetime.now(timezone.utc).isoformat(),
                **answer,
            }
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
        return len(existing)

    def all_ratings(self) -> list[dict]:
        out = []
        if not self.ratings_dir.exists():
            return out
        for path in sorted(self.ratings_dir.glob("*.json")):
            for rating in json.loads(path.read_text()):
                out.append({**rating, "listener": path.stem})
        return out

    # --- results ----------------------------------------------------------

    def results(self, kind: str, include_rushed: bool = False) -> dict:
        """Per-system scores with intervals.

        `include_rushed=False` drops ratings given faster than MIN_LISTEN_MS. Both
        views are worth looking at: a large gap between them means the numbers rest
        on people clicking through.
        """
        ratings = [r for r in self.all_ratings() if self.item(r["item_id"]).kind == kind]
        rushed = [r for r in ratings if r.get("ms", 10**9) < MIN_LISTEN_MS]
        if not include_rushed:
            ratings = [r for r in ratings if r.get("ms", 10**9) >= MIN_LISTEN_MS]

        by_system: dict[str, list[dict]] = {}
        for rating in ratings:
            by_system.setdefault(self.system_of(rating["item_id"]), []).append(rating)

        systems = {}
        for system, group in sorted(by_system.items()):
            if kind == IDENTITY:
                called_real = [1 if r.get("called_real") else 0 for r in group]
                low, high = wilson(sum(called_real), len(called_real))
                systems[system] = {
                    "n": len(group),
                    "metric": "fooled_rate",
                    "value": round(sum(called_real) / len(called_real), 4) if called_real else None,
                    "ci95": [round(low, 4), round(high, 4)],
                    "is_real": self.item(group[0]["item_id"]).is_real,
                }
            else:
                scores = [float(r["score"]) for r in group if r.get("score") is not None]
                mean, low, high = mean_ci(scores)
                systems[system] = {
                    "n": len(scores),
                    "metric": "mos",
                    "value": round(mean, 3) if scores else None,
                    "ci95": [round(low, 3), round(high, 3)],
                }

        listeners = {r["listener"] for r in ratings}
        enough = all(s["n"] >= MIN_RATINGS_FOR_A_VERDICT for s in systems.values()) and systems
        return {
            "kind": kind,
            "systems": systems,
            "listeners": len(listeners),
            "ratings": len(ratings),
            "rushed_dropped": len(rushed) if not include_rushed else 0,
            "verdict_supported": bool(enough),
            "note": (
                None
                if enough
                else f"fewer than {MIN_RATINGS_FOR_A_VERDICT} ratings for some system; "
                "read the intervals, do not call a winner"
            ),
        }


# --- statistics -----------------------------------------------------------


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Not the normal approximation, which is wrong in exactly the regime this runs
    in: at n = 10 with 9 successes it produces an upper bound above 1.0 and an
    interval that excludes plausible values. Wilson stays inside [0, 1] and behaves
    at small n and extreme proportions.
    """
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def mean_ci(values: list[float], z: float = 1.96) -> tuple[float, float, float]:
    """Mean and a normal-approximation interval. Returns (mean, low, high)."""
    if not values:
        return (0.0, 0.0, 0.0)
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return (mean, mean, mean)
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    margin = z * math.sqrt(variance / n)
    return (mean, mean - margin, mean + margin)


# --- construction ---------------------------------------------------------


def build(
    samples: dict[str, dict[str, Path]],
    real_systems: tuple[str, ...] = ("real",),
    identity_systems: tuple[str, ...] | None = None,
) -> Benchmark:
    """Assemble a benchmark from `{system: {slug: wav_path}}`.

    Audio is *copied* into the benchmark directory under its opaque item id rather
    than referenced in place. That is what makes the blinding hold: the served URL
    carries no system name, and a later edit to the source files cannot silently
    change what listeners already scored.

    `identity_systems` limits the identity test to systems where the question makes
    sense — asking "real or synthetic?" about a competitor's voice, which is not the
    speaker's voice at all, measures nothing.
    """
    benchmark = Benchmark(
        benchmark_id=datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    benchmark.audio_dir.mkdir(parents=True, exist_ok=True)
    benchmark.ratings_dir.mkdir(parents=True, exist_ok=True)

    for system, per_slug in sorted(samples.items()):
        for slug, source in sorted(per_slug.items()):
            source = Path(source)
            if not source.exists():
                continue
            for kind in (NATURALNESS, IDENTITY):
                if kind == IDENTITY and identity_systems is not None and system not in identity_systems:
                    continue
                item = Item(
                    item_id=uuid.uuid4().hex[:16],
                    system=system,
                    slug=slug,
                    kind=kind,
                    is_real=system in real_systems,
                )
                shutil.copyfile(source, benchmark.audio_path(item.item_id))
                benchmark.items.append(item)

    benchmark.save()
    return benchmark
