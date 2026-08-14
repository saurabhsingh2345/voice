"""The gate that decides whether a measurement is worth believing.

This exists because of a specific mistake and then three wrong fixes for it. A
Hindi RTF was measured at 6-12 and nearly written down; the host was at load
average 537 with 13 GiB of swap in use, and the same code reads 0.63 idle. So the
sheet refuses to measure a busy machine.

Getting the *threshold* right took three tries, and each wrong answer is encoded
in a test below, because each one is a way to be confidently wrong again.
"""

from __future__ import annotations

from voiceagent.eval.specsheet import (
    MAX_LOAD_PER_CORE,
    MIN_FREE_GIB,
    machine_is_quiet,
    render,
)


def state(load=2.0, cpus=12, free=8.0, swap=1.0):
    return dict(load_1m=load, cpus=cpus, free_gib=free, swap_used_gib=swap,
                machine="arm64", ram_gib=18.0, platform="test")


def test_an_idle_machine_passes():
    quiet, why = machine_is_quiet(state())
    assert quiet and "quiet enough" in why


def test_the_catastrophic_case_is_refused():
    """Load 537 on 12 cores: 44 per core. This is the run that started all of it."""
    quiet, why = machine_is_quiet(state(load=537.0))
    assert not quiet and "per core" in why


def test_load_is_judged_per_core_not_absolute():
    """The first fix used an absolute threshold and there is no value that works.
    12.0 passed a machine that then measured 1.9x slow; 3.0 refused a genuinely
    idle 12-core box that idles at 2.4-4.3. The same raw number means opposite
    things on 4 cores and on 12."""
    assert machine_is_quiet(state(load=6.0, cpus=12))[0]
    assert not machine_is_quiet(state(load=6.0, cpus=4))[0]


def test_the_threshold_is_full_utilisation():
    assert MAX_LOAD_PER_CORE == 1.0


def test_a_machine_with_no_headroom_is_refused():
    quiet, why = machine_is_quiet(state(free=MIN_FREE_GIB - 0.5))
    assert not quiet and "free" in why


def test_swap_is_never_gated_on():
    """Learned by watching it. After the load was removed, load fell 3.8 -> 2.4
    over four minutes while swap moved 14.11 -> 13.78: it does not drain. The
    number describes the machine's history, not its present, so gating on it
    means a machine that has ever thrashed can never be measured again.

    The README makes the same argument about swap *percentage*; absolute
    swap-in-use turns out not to escape it.
    """
    assert machine_is_quiet(state(swap=15.0))[0], "swap must not veto a quiet machine"

    import inspect

    from voiceagent.eval import specsheet

    source = inspect.getsource(specsheet.machine_is_quiet)
    assert "swap" not in source.lower()


def test_a_busy_sheet_says_so_at_the_top():
    """The whole point. A sheet with untrustworthy numbers and no warning is
    worse than no sheet -- it looks exactly like a good one."""
    report = dict(
        host=state(load=537.0, cpus=12) | {"calibration_ms": 900.0},
        machine_quiet=False,
        machine_note="load is 537.0 across 12 cores",
        licences=dict(clean=True, model_violations=[], dependency_violations=[],
                      accepted_exceptions=[]),
    )
    body = render(report)
    lowered = body.lower()
    assert "not trustworthy" in lowered
    # Before any section that reports a measurement -- "first half of the file"
    # is a proxy that breaks as soon as a section is added or removed.
    assert lowered.index("not trustworthy") < lowered.index("## licences")


def test_a_quiet_sheet_carries_no_warning():
    report = dict(
        host=state() | {"calibration_ms": 19.4},
        machine_quiet=True,
        machine_note="quiet enough to measure",
        licences=dict(clean=True, model_violations=[], dependency_violations=[],
                      accepted_exceptions=["num2words"]),
    )
    assert "not trustworthy" not in render(report).lower()


def test_the_sheet_reports_the_metric_ceiling_alongside_the_result():
    """Publishing a synthetic score without the human anchor is the thing this
    project got wrong for months: every Hindi number in the README was being read
    against an implicit 100% that the scorer cannot reach."""
    report = dict(
        host=state() | {"calibration_ms": 19.4},
        machine_quiet=True,
        machine_note="ok",
        licences=dict(clean=True, model_violations=[], dependency_violations=[],
                      accepted_exceptions=[]),
        roundtrip={
            "human recording (metric ceiling)": dict(
                mean_percent=90.2, worst_percent=80.6, code_mixed_percent=91.5, sentences=12),
            "Chatterbox 8-bit (current)": dict(
                mean_percent=93.5, worst_percent=83.9, code_mixed_percent=94.5, sentences=12),
        },
    )
    body = render(report)
    assert "metric ceiling" in body
    assert "90.2" in body and "93.5" in body
    assert "not naturalness" in body
