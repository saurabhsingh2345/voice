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
        weights_gb=2.30,
        runtime_overhead_gb=1.15,
        phase=2,
        default=True,
        notes="Overhead is the KV cache at an 8k context. Halve it at 4k.",
    ),
    ModelSpec(
        name="Kokoro-82M",
        stage=Stage.TTS,
        license="Apache-2.0",
        repo="hexgrad/Kokoro-82M",
        weights_gb=0.33,
        runtime_overhead_gb=0.15,
        phase=3,
        default=True,
        notes="82M params. Overhead is the vocoder + output audio buffer.",
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
