"""Phase 0 diagnostic: prove this machine can host the pipeline before we build it.

Run with::

    uv run voice-doctor

Exits non-zero if a hard constraint is violated (bad license, over budget, no
Metal), so it can be wired into CI or a pre-flight check later.
"""

from __future__ import annotations

import platform
import subprocess
import sys

import psutil
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from voiceagent import models as M

console = Console()

GIB = 1024**3


# --- Probes ---------------------------------------------------------------


def probe_memory() -> dict[str, float]:
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "total_gb": vm.total / GIB,
        "available_gb": vm.available / GIB,
        "used_gb": vm.used / GIB,
        "swap_used_gb": swap.used / GIB,
    }


def probe_metal() -> tuple[bool, dict[str, str], str | None]:
    """Return (ok, info, error). Runs a real GPU op -- not just an import."""
    try:
        import mlx.core as mx
    except ImportError as exc:
        return False, {}, f"mlx is not installed ({exc})"

    info: dict[str, str] = {"mlx_version": getattr(mx, "__version__", "unknown")}

    # device_info() moved from mx.metal to the top level around MLX 0.22.
    raw = {}
    for getter in (
        lambda: mx.device_info(),
        lambda: mx.metal.device_info(),
    ):
        try:
            raw = getter()
            break
        except (AttributeError, RuntimeError):
            continue

    for key in ("architecture", "memory_size", "max_recommended_working_set_size"):
        if key in raw:
            value = raw[key]
            info[key] = f"{value / GIB:.2f} GiB" if isinstance(value, int) and value > GIB else str(value)

    # Actually exercise the GPU so we know Metal is functional, not just present.
    try:
        a = mx.random.normal((512, 512))
        b = mx.random.normal((512, 512))
        result = (a @ b).sum()
        mx.eval(result)
        info["matmul_smoke_test"] = "passed (512x512 on GPU)"
    except Exception as exc:  # noqa: BLE001 -- surface whatever Metal complains about
        return False, info, f"Metal compute failed: {exc}"

    return True, info, None


def probe_gpu_cores() -> str:
    """Best-effort GPU core count from system_profiler."""
    try:
        out = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return "unknown"
    for line in out.splitlines():
        if "Total Number of Cores" in line:
            return line.split(":", 1)[1].strip()
    return "unknown"


# --- Reports --------------------------------------------------------------


def report_system(mem: dict[str, float], metal_ok: bool, metal: dict[str, str]) -> None:
    table = Table(title="System", title_justify="left", header_style="bold")
    table.add_column("Property")
    table.add_column("Value", style="cyan")

    table.add_row("Machine", f"{platform.machine()} / {platform.processor() or 'Apple Silicon'}")
    table.add_row("macOS", platform.mac_ver()[0] or platform.release())
    table.add_row("Python", sys.version.split()[0])
    table.add_row("CPU cores", str(psutil.cpu_count(logical=True)))
    table.add_row("GPU cores", probe_gpu_cores())
    table.add_row("Total RAM", f"{mem['total_gb']:.2f} GiB")
    table.add_row("Available RAM", f"{mem['available_gb']:.2f} GiB")
    table.add_row("Swap in use", f"{mem['swap_used_gb']:.2f} GiB")
    table.add_row("Metal/MLX", "[green]OK[/]" if metal_ok else "[red]FAILED[/]")
    for key, value in metal.items():
        table.add_row(f"  {key}", value)

    console.print(table)


def report_budget(mem: dict[str, float]) -> bool:
    table = Table(
        title=f"Memory budget (ceiling {M.PIPELINE_BUDGET_GB:.1f} GiB)",
        title_justify="left",
        header_style="bold",
    )
    table.add_column("Stage")
    table.add_column("Model")
    table.add_column("License")
    table.add_column("Weights", justify="right")
    table.add_column("Runtime", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Phase", justify="right")
    table.add_column("Src")

    for spec in M.REGISTRY:
        chosen = spec.default
        table.add_row(
            spec.stage.value,
            f"[bold]{spec.name}[/]" if chosen else f"[dim]{spec.name} (alt)[/]",
            spec.license,
            f"{spec.weights_gb:.2f}",
            f"{spec.runtime_overhead_gb:.2f}",
            f"{spec.total_gb:.2f}",
            str(spec.phase),
            "meas" if spec.measured else "[yellow]est[/]",
            style=None if chosen else "dim",
        )

    models_gb = sum(s.total_gb for s in M.default_models())
    projected = M.projected_resident_gb()

    table.add_section()
    table.add_row("", "[bold]Default models combined[/]", "", "", "", f"[bold]{models_gb:.2f}[/]", "", "")
    table.add_row("", "Python + MLX + app overhead", "", "", "", f"{M.FRAMEWORK_OVERHEAD_GB:.2f}", "", "est")
    table.add_row("", "[bold]Projected resident pipeline[/]", "", "", "", f"[bold]{projected:.2f}[/]", "", "")

    console.print(table)

    headroom = mem["total_gb"] - projected
    within_budget = projected <= M.PIPELINE_BUDGET_GB
    os_ok = headroom >= M.OS_RESERVE_GB

    summary = [
        f"Projected pipeline : {projected:.2f} GiB",
        f"Budget ceiling     : {M.PIPELINE_BUDGET_GB:.2f} GiB  "
        + ("[green]WITHIN BUDGET[/]" if within_budget else "[red]OVER BUDGET[/]"),
        f"Left for macOS     : {headroom:.2f} GiB  "
        + (
            f"[green]OK (>= {M.OS_RESERVE_GB:.1f} GiB reserve)[/]"
            if os_ok
            else f"[red]TIGHT (< {M.OS_RESERVE_GB:.1f} GiB reserve)[/]"
        ),
    ]
    console.print(Panel("\n".join(summary), title="Verdict", border_style="green" if within_budget and os_ok else "red"))

    # The budget above is theoretical -- it compares against *total* RAM. What
    # matters when we actually load models is what is free right now.
    _report_pressure(mem, projected)
    return within_budget and os_ok


def _report_pressure(mem: dict[str, float], projected: float) -> None:
    """Warn if the machine's current state would force swapping."""
    warnings = []
    if mem["available_gb"] < projected:
        warnings.append(
            f"Only {mem['available_gb']:.2f} GiB is available right now, but the full "
            f"pipeline needs {projected:.2f} GiB. Other processes must be closed before "
            f"loading models, or macOS will swap and latency targets will not hold."
        )
    if mem["swap_used_gb"] > 2.0:
        warnings.append(
            f"{mem['swap_used_gb']:.2f} GiB of swap is already in use. This machine is "
            f"under memory pressure from something other than this project. Benchmarks "
            f"taken in this state will be misleading."
        )
    if warnings:
        console.print(
            Panel(
                "\n\n".join(warnings),
                title="Live memory pressure (advisory, not a budget failure)",
                border_style="yellow",
            )
        )


def report_licenses() -> bool:
    violations = M.audit_licenses()
    if violations:
        console.print(Panel("\n".join(violations), title="LICENSE VIOLATIONS", border_style="red"))
        return False
    allowed = ", ".join(sorted(M.PERMISSIVE_LICENSES))
    denied = "\n".join(f"  - {name}: {reason}" for name, reason in M.DENYLIST.items())
    console.print(
        Panel(
            f"All {len(M.REGISTRY)} registered models use a permissive license.\n"
            f"Allow-list: {allowed}\n\nExplicitly denylisted:\n{denied}",
            title="License audit",
            border_style="green",
        )
    )
    return True


def main() -> int:
    console.print("[bold]Local Voice Agent -- Phase 0 diagnostic[/]\n")

    mem = probe_memory()
    metal_ok, metal_info, metal_err = probe_metal()

    report_system(mem, metal_ok, metal_info)
    console.print()
    licenses_ok = report_licenses()
    console.print()
    budget_ok = report_budget(mem)

    if metal_err:
        console.print(f"\n[red]Metal error:[/] {metal_err}")

    ok = metal_ok and licenses_ok and budget_ok
    console.print(
        f"\n[bold]{'[green]PHASE 0 PASS[/]' if ok else '[red]PHASE 0 FAIL[/]'}[/] "
        f"(metal={metal_ok}, licenses={licenses_ok}, budget={budget_ok})"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
