"""Declarative registry of every model the pipeline may load.

This module is the single enforcement point for two of the project's hard
constraints:

  1. Open-source / commercially-usable licenses only (Apache-2.0, MIT, BSD).
  2. The whole resident pipeline must stay under the memory budget.

Nothing here downloads or loads a model. It is pure metadata, so the diagnostic
script can prove we are within budget *before* anything is fetched.

Sizes are recorded per phase. Values marked ``measured=False`` are published or
computed estimates and MUST be replaced with real numbers as each phase lands.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

# --- License policy -------------------------------------------------------

#: Licenses that permit unrestricted commercial use. Anything outside this set
#: is rejected by :func:`audit_licenses`.
PERMISSIVE_LICENSES = frozenset(
    {
        "Apache-2.0",
        "MIT",
        "BSD-2-Clause",
        "BSD-3-Clause",
    }
)

#: Models explicitly ruled out, kept here so nobody re-adds them by accident.
#: The blueprint calls these out because most tutorials reach for them first.
DENYLIST = {
    "XTTS-v2": "CPML (Coqui Public Model License) -- non-commercial.",
    "F5-TTS": "Trained on Emilia (CC-BY-NC-4.0) -- non-commercial.",
    "Fish-Speech": "Weights are CC-BY-NC-SA-4.0 -- non-commercial.",
}


class Stage(str, Enum):
    VAD = "VAD"
    STT = "STT"
    LLM = "LLM"
    TTS = "TTS"


@dataclass(frozen=True)
class ModelSpec:
    """Metadata for one model considered for the pipeline."""

    name: str
    stage: Stage
    license: str
    repo: str
    weights_gb: float
    """Resident size of the weights at the stated quantization, in GiB."""
    runtime_overhead_gb: float = 0.0
    """Activations, KV cache, audio buffers -- anything resident beyond weights."""
    phase: int = 0
    """Project phase that introduces this model."""
    default: bool = False
    """True if this is the chosen model for its stage (vs. a benchmarked rival)."""
    measured: bool = False
    """False means the number is an estimate awaiting a real measurement."""
    notes: str = ""

    @property
    def total_gb(self) -> float:
        return self.weights_gb + self.runtime_overhead_gb


# --- The registry ---------------------------------------------------------
#
# Estimate provenance:
#   * weights_gb for 4-bit MLX models ~= params * 0.5 bytes + embedding overhead.
#   * Qwen3-4B KV cache: 36 layers x 8 KV heads x 128 head_dim x 2 (K+V) x 2 bytes
#     ~= 147 KiB/token; budgeted here for an 8k-token context (~1.15 GiB).

REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="Silero VAD v5",
        stage=Stage.VAD,
        license="MIT",
        repo="snakers4/silero-vad",
        weights_gb=0.002,
        runtime_overhead_gb=0.05,
        phase=4,
        default=True,
        notes="~2 MB ONNX model; overhead is the audio ring buffer.",
    ),
    ModelSpec(
        name="Moonshine tiny-streaming (en)",
        stage=Stage.STT,
        license="MIT",
        repo="moonshine-voice:TINY_STREAMING",
        weights_gb=0.128,
        runtime_overhead_gb=0.01,
        phase=1,
        default=False,
        measured=True,
        notes="Measured 131 MiB, median RTF 0.063, load 9.8s. Fastest and smallest, weakest accuracy.",
    ),
    ModelSpec(
        name="Moonshine small-streaming (en)",
        stage=Stage.STT,
        license="MIT",
        repo="moonshine-voice:SMALL_STREAMING",
        weights_gb=0.223,
        runtime_overhead_gb=0.34,
        phase=1,
        default=True,
        measured=True,
        notes=(
            "CHOSEN. Measured 228 MiB weights, median RTF 0.120, load 15.8s; "
            "process RSS during streaming settles near 570 MiB including the "
            "ONNX runtime. Native streaming, no torch dependency. "
            "CAUTION: only the code and the ENGLISH models are MIT. Non-English "
            "models ship under the 'Moonshine Community License' and are NOT "
            "cleared by this project's allow-list."
        ),
    ),
    ModelSpec(
        name="Moonshine medium-streaming (en)",
        stage=Stage.STT,
        license="MIT",
        repo="moonshine-voice:MEDIUM_STREAMING",
        weights_gb=0.470,
        runtime_overhead_gb=0.10,
        phase=1,
        default=False,
        measured=True,
        notes="Measured 481 MiB, median RTF 0.131. Best Moonshine accuracy; upgrade path if small proves weak.",
    ),
    ModelSpec(
        name="Whisper Large-v3-Turbo (MLX, fp16)",
        stage=Stage.STT,
        license="MIT",
        repo="mlx-community/whisper-large-v3-turbo",
        weights_gb=2.31,
        runtime_overhead_gb=0.0,
        phase=1,
        default=False,
        measured=True,
        notes=(
            "Measured 2362 MiB MLX peak, median RTF 0.104, load 4.6s. Most "
            "accurate by a clear margin, but 10x Moonshine's memory, no true "
            "streaming (re-decodes a growing buffer), and it pulls the entire "
            "torch stack in as a dependency. Kept as an opt-in high-accuracy "
            "batch backend, not the live-loop default."
        ),
    ),
    ModelSpec(
        name="Qwen3-4B-Instruct-2507 (MLX, 4-bit)",
        stage=Stage.LLM,
        license="Apache-2.0",
        repo="mlx-community/Qwen3-4B-Instruct-2507-4bit",
        weights_gb=2.16,
        runtime_overhead_gb=1.15,
        phase=2,
        default=True,
        measured=True,
        notes=(
            "Weights measured at 2.16 GiB MLX peak, load 5.3s. Runtime figure "
            "is the computed KV-cache worst case at an 8k context "
            "(36 layers x 8 KV heads x 128 dim x 2 x fp16 ~= 147 KiB/token); "
            "halve it at 4k. TTFT 185 ms with a bare prompt, ~500 ms once tool "
            "schemas are attached, ~150 ms on later turns with prefix caching."
        ),
    ),
    ModelSpec(
        name="Kokoro-82M (MLX, bf16)",
        stage=Stage.TTS,
        license="Apache-2.0",
        repo="mlx-community/Kokoro-82M-bf16",
        weights_gb=0.68,
        runtime_overhead_gb=0.10,
        phase=3,
        default=True,
        measured=True,
        notes=(
            "Measured 0.68 GiB MLX peak (above the 0.33 GiB estimate: bf16, not "
            "4-bit, and the figure includes the vocoder), load 3.3s, 24 kHz "
            "output. First audio 822 ms synthesizing a whole reply vs 280 ms "
            "sentence-streamed. Runs with the GPL espeak fallback DISABLED -- "
            "see voiceagent.tts.kokoro_engine. 8-bit/4-bit variants exist if "
            "the budget ever tightens."
        ),
    ),
    ModelSpec(
        name="Chatterbox Turbo (MLX, fp16)",
        stage=Stage.TTS,
        license="MIT",
        repo="mlx-community/chatterbox-turbo-fp16",
        weights_gb=2.82,
        runtime_overhead_gb=0.64,
        phase=7,
        default=False,
        measured=True,
        notes=(
            "Voice cloning. 350M params, MIT (Resemble AI), zero-shot from a "
            "~10s reference clip, 24 kHz. Measured 2.82 GiB after load, 3.46 GiB "
            "peak during generation. Loaded only when a cloned voice is used -- "
            "it is NOT co-resident with Kokoro in the default pipeline. Chosen "
            "over Fish Speech, whose weights are CC-BY-NC-SA-4.0."
        ),
    ),
    ModelSpec(
        name="Chatterbox Multilingual v3 (MLX, 8-bit)",
        stage=Stage.TTS,
        license="MIT",
        repo="mlx-community/chatterbox-multilingual-v3",
        weights_gb=1.33,
        runtime_overhead_gb=1.44,
        phase=9,
        default=False,
        measured=True,
        notes=(
            "Hindi. Replaced IndicF5 in Phase 0 of the commercial-readiness "
            "pass -- not because IndicF5's weights were a problem (they are "
            "MIT) but because running it required `f5-tts`, which drags in "
            "encodec (CC-BY-NC), Unidecode (GPL), frozendict (LGPL-3) and soxr "
            "(LGPL-2.1). That tree is what made voice-doctor exit non-zero. "
            "Quantized to 8-bit locally from the MIT fp32 repo on first use "
            "(tts/quantize.py) -- community requants exist and all declare no "
            "licence. Measured idle, medians of 3: 1.33 GiB after load, 2.77 "
            "GiB peak, aggregate RTF 0.63, against fp32's 3.04 / 4.55 / 1.17. "
            "Quality is unchanged by quantization -- 93.5 % mean round-trip "
            "overlap either way, against IndicF5's 88.7 % -- and every sentence "
            "is now under RTF 1.0, so Hindi keeps up with playback. The cost is "
            "coverage: this checkpoint speaks Hindi and no other Indic "
            "language, where IndicF5 spoke 11. Loaded only for Hindi; NOT "
            "co-resident with Kokoro or Chatterbox Turbo."
        ),
    ),
)


# --- Budget ---------------------------------------------------------------

#: Ceiling for the resident pipeline (models + Python + app), in GiB.
#: The brief sets 12-13 GB; we hold the line at the bottom of that range.
PIPELINE_BUDGET_GB = 12.0

#: Reserve for the Python runtime, PyTorch/MLX libs, and the desktop shell.
FRAMEWORK_OVERHEAD_GB = 1.20

#: Headroom macOS needs to avoid swapping/memory pressure on an 18 GiB machine.
OS_RESERVE_GB = 4.0


# --- Dependency licenses --------------------------------------------------
#
# Model licenses are only half the problem. Phase 3 nearly pulled in
# `phonemizer-fork` (GPLv3) as a transitive dependency of Kokoro's text
# frontend, which would have forced the whole desktop app to be GPL. Copyleft
# arrives through the dependency tree, so the tree gets audited too.

#: Tokens that mark a copyleft or non-commercial license, matched on word
#: boundaries against a SHORT license identifier -- never against free text.
#: Packages often stuff their entire LICENSE file into the metadata (scipy's is
#: 47 KB and quotes the GPL because it bundles libgfortran, which is BSD-licensed
#: code shipping alongside a GPL-with-runtime-exception binary). Substring
#: matching on that produces confident nonsense.
COPYLEFT_PATTERN = re.compile(
    r"\b(?:A?GPL|LGPL|GNU (?:Affero |Lesser )?General Public|SSPL|CPML|"
    r"CC[- ]BY[- ]NC|noncommercial|non-commercial)\b",
    re.IGNORECASE,
)

#: A License field longer than this is a license *text*, not an identifier.
MAX_LICENSE_ID_CHARS = 80

#: Packages whose non-permissive license we have consciously accepted, with why.
#:
#: Empty, and that is the point. It held exactly one entry for most of this
#: project's life -- `num2words`, LGPL-2.1, imported by misaki to turn digits
#: into words for Kokoro. The note there said to revisit it "if the Tauri build
#: ever bundles Python into one archive", because the LGPL's one real obligation
#: is that recipients can replace the library, and an opaque frozen binary
#: cannot satisfy that.
#:
#: That day arrived: a self-contained desktop bundle is the whole point of the
#: packaging work, so the exception had to go rather than be renewed.
#: `text/numbers.py` replaces it -- written against the library's observed
#: output rather than its source, so nothing LGPL is derived or vendored, and
#: verified at full parity over every integer to 10,000, every year to 20,000,
#: every ordinal to 2,000, and several thousand floats. `text/num2words_shim.py`
#: registers it under the name misaki imports.
#:
#: Adding an entry here is a real decision. It means the audit passes on
#: something the rule says it should not, and the reason has to survive being
#: read back a year later by someone deciding whether to ship.
ACCEPTED_EXCEPTIONS: dict[str, str] = {}

#: Packages that must never be installed, and why.
FORBIDDEN_PACKAGES: dict[str, str] = {
    "phonemizer": "GPLv3 -- would make the packaged app GPL.",
    "phonemizer-fork": "GPLv3 -- would make the packaged app GPL.",
}


def audit_installed_packages() -> tuple[list[str], list[str]]:
    """Scan the live environment. Returns (violations, accepted_notes)."""
    import importlib.metadata as md

    violations: list[str] = []
    accepted: list[str] = []

    for dist in md.distributions():
        name = (dist.metadata["Name"] or "").strip()
        if not name:
            continue

        if name.lower() in FORBIDDEN_PACKAGES:
            violations.append(f"{name}: INSTALLED but forbidden -- {FORBIDDEN_PACKAGES[name.lower()]}")
            continue

        # Prefer structured signals: PEP 639 License-Expression, then the
        # license classifiers, then a short License field.
        declared = (dist.metadata["License-Expression"] or "").strip()
        if not declared:
            raw = (dist.metadata["License"] or "").strip()
            declared = raw if len(raw) <= MAX_LICENSE_ID_CHARS else ""
        classifiers = " ".join(
            c for c in (dist.metadata.get_all("Classifier") or []) if "License" in c
        )
        identifier = f"{declared} {classifiers}".strip()

        if not COPYLEFT_PATTERN.search(identifier):
            continue
        if name.lower() in ACCEPTED_EXCEPTIONS:
            accepted.append(f"{name}: {ACCEPTED_EXCEPTIONS[name.lower()]}")
        else:
            violations.append(f"{name}: non-permissive license detected ({declared or classifiers})")

    return violations, accepted


def default_models() -> list[ModelSpec]:
    """The models that will actually be co-resident in the full pipeline."""
    return [m for m in REGISTRY if m.default]


def projected_resident_gb() -> float:
    """Total projected RAM for the full pipeline once every phase has landed."""
    return sum(m.total_gb for m in default_models()) + FRAMEWORK_OVERHEAD_GB


def audit_licenses() -> list[str]:
    """Return a list of license violations. Empty list means we are compliant."""
    violations = []
    for spec in REGISTRY:
        if spec.license not in PERMISSIVE_LICENSES:
            violations.append(
                f"{spec.name}: license {spec.license!r} is not in the permissive allow-list."
            )
        for banned, reason in DENYLIST.items():
            if banned.lower().replace("-", "") in spec.name.lower().replace("-", ""):
                violations.append(f"{spec.name}: denylisted -- {reason}")
    return violations
