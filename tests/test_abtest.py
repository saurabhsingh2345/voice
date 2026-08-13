"""Properties of the blind benchmark.

The ones that matter are about blinding and about not overclaiming. A benchmark that
leaks which system made a sample produces a number that feels like evidence and is
not, which is worse than having no number — and this project has already been misled
once by a measurement it trusted (F0 ranked the preferred sample worst).
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
import soundfile as sf

from voiceagent.eval import abtest


@pytest.fixture(autouse=True)
def isolated_root(tmp_path, monkeypatch):
    monkeypatch.setattr(abtest, "ROOT", tmp_path / "abtest")
    return tmp_path


def wav(path, seconds: float = 1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(seconds * 24_000), endpoint=False)
    sf.write(path, (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 24_000)
    return path


@pytest.fixture
def samples(tmp_path):
    return {
        "ours": {s: wav(tmp_path / "src" / f"ours_{s}.wav") for s in ("h1", "h2", "h3")},
        "stock": {s: wav(tmp_path / "src" / f"stock_{s}.wav") for s in ("h1", "h2", "h3")},
        "real": {s: wav(tmp_path / "src" / f"real_{s}.wav") for s in ("h1", "h2", "h3")},
    }


# --- blinding --------------------------------------------------------------


def test_item_ids_do_not_contain_the_system_name(samples):
    """The natural implementation names files ours_h1.wav and leaks the answer in
    the URL bar. Item ids are random for exactly this reason."""
    bench = abtest.build(samples)
    for item in bench.items:
        assert item.system not in item.item_id
        assert item.slug not in item.item_id


def test_audio_is_copied_under_the_opaque_id_not_referenced(samples, tmp_path):
    bench = abtest.build(samples)
    for item in bench.items:
        assert bench.audio_path(item.item_id).exists()
        assert bench.audio_path(item.item_id).name == f"{item.item_id}.wav"
    # Editing a source file must not change what listeners already scored.
    original = bench.audio_path(bench.items[0].item_id).read_bytes()
    wav(tmp_path / "src" / "ours_h1.wav", seconds=3.0)
    assert bench.audio_path(bench.items[0].item_id).read_bytes() == original


def test_the_listener_order_contains_ids_only(samples):
    bench = abtest.build(samples)
    order = bench.order_for("listener-1", abtest.NATURALNESS)
    assert order and all(isinstance(i, str) for i in order)
    assert all(i in {x.item_id for x in bench.items} for i in order)


def test_order_is_stable_across_reloads_but_differs_between_listeners(samples):
    """Stable so a refresh resumes instead of re-presenting rated items, which would
    silently double-weight that listener. Different per listener so a shared
    presentation order cannot bias the comparison."""
    bench = abtest.build(samples)
    assert bench.order_for("a", abtest.NATURALNESS) == bench.order_for("a", abtest.NATURALNESS)
    assert bench.order_for("a", abtest.NATURALNESS) != bench.order_for("b", abtest.NATURALNESS)


def test_identity_can_be_restricted_to_systems_where_the_question_makes_sense(samples):
    """Asking 'is this the real person?' about a competitor's stock voice measures
    nothing -- it is not the speaker's voice at all."""
    samples["competitor"] = {"h1": next(iter(samples["ours"].values()))}
    bench = abtest.build(samples, identity_systems=("ours", "stock", "real"))
    identity_systems = {i.system for i in bench.items if i.kind == abtest.IDENTITY}
    assert "competitor" not in identity_systems
    assert "competitor" in {i.system for i in bench.items if i.kind == abtest.NATURALNESS}


# --- ratings ---------------------------------------------------------------


def test_ratings_round_trip(samples):
    bench = abtest.build(samples)
    item = bench.items[0].item_id
    bench.record("ravi", item, {"score": 4, "ms": 3000})
    assert len(bench.all_ratings()) == 1
    assert bench.all_ratings()[0]["listener"] == "ravi"


def test_re_rating_overwrites_instead_of_counting_twice(samples):
    bench = abtest.build(samples)
    item = bench.items[0].item_id
    bench.record("ravi", item, {"score": 2, "ms": 3000})
    bench.record("ravi", item, {"score": 5, "ms": 3000})
    ratings = bench.all_ratings()
    assert len(ratings) == 1
    assert ratings[0]["score"] == 5


def test_an_unknown_item_is_refused(samples):
    bench = abtest.build(samples)
    with pytest.raises(KeyError):
        bench.record("ravi", "not-an-item", {"score": 4})


# --- results ---------------------------------------------------------------


def test_naturalness_reports_a_mean_with_an_interval(samples):
    bench = abtest.build(samples)
    for item in [i for i in bench.items if i.kind == abtest.NATURALNESS]:
        score = 5 if item.system == "ours" else 2
        for listener in range(8):
            bench.record(f"l{listener}", item.item_id, {"score": score, "ms": 4000})

    out = bench.results(abtest.NATURALNESS)
    assert out["systems"]["ours"]["value"] == pytest.approx(5.0)
    assert out["systems"]["stock"]["value"] == pytest.approx(2.0)
    assert out["systems"]["ours"]["metric"] == "mos"
    assert len(out["systems"]["ours"]["ci95"]) == 2


def test_identity_reports_a_fooled_rate(samples):
    bench = abtest.build(samples)
    for item in [i for i in bench.items if i.kind == abtest.IDENTITY]:
        # Our clone is called real most of the time; the stock voice rarely.
        called = item.system in ("ours", "real")
        for listener in range(8):
            bench.record(f"l{listener}", item.item_id, {"called_real": called, "ms": 4000})

    out = bench.results(abtest.IDENTITY)
    assert out["systems"]["ours"]["metric"] == "fooled_rate"
    assert out["systems"]["ours"]["value"] == pytest.approx(1.0)
    assert out["systems"]["stock"]["value"] == pytest.approx(0.0)
    assert out["systems"]["real"]["is_real"] is True


def test_rushed_ratings_are_dropped_and_counted(samples):
    """A rating faster than the clip cannot be a judgement of the clip."""
    bench = abtest.build(samples)
    item = next(i for i in bench.items if i.kind == abtest.NATURALNESS)
    bench.record("careful", item.item_id, {"score": 5, "ms": 4000})
    bench.record("clicker", item.item_id, {"score": 1, "ms": 200})

    out = bench.results(abtest.NATURALNESS)
    assert out["rushed_dropped"] == 1
    assert out["systems"][item.system]["value"] == pytest.approx(5.0)

    both = bench.results(abtest.NATURALNESS, include_rushed=True)
    assert both["systems"][item.system]["n"] == 2


def test_a_thin_sample_refuses_to_call_a_winner(samples):
    """The project has already been burned by trusting a measurement. Three ratings
    must not read the same as three hundred."""
    bench = abtest.build(samples)
    item = next(i for i in bench.items if i.kind == abtest.NATURALNESS)
    bench.record("one", item.item_id, {"score": 5, "ms": 4000})

    out = bench.results(abtest.NATURALNESS)
    assert out["verdict_supported"] is False
    assert "do not call a winner" in out["note"]


def test_enough_ratings_supports_a_verdict(samples):
    bench = abtest.build(samples)
    for item in [i for i in bench.items if i.kind == abtest.NATURALNESS]:
        for listener in range(abtest.MIN_RATINGS_FOR_A_VERDICT):
            bench.record(f"l{listener}", item.item_id, {"score": 4, "ms": 4000})
    out = bench.results(abtest.NATURALNESS)
    assert out["verdict_supported"] is True
    assert out["note"] is None


# --- statistics ------------------------------------------------------------


def test_wilson_stays_inside_zero_and_one_at_the_extremes():
    """The normal approximation gives an upper bound above 1.0 at 9 of 10, which is
    the regime this actually runs in."""
    low, high = abtest.wilson(9, 10)
    assert 0.0 <= low <= high <= 1.0
    assert high < 1.0, "9 of 10 is not certainty"
    low, high = abtest.wilson(10, 10)
    assert high == 1.0 and low < 1.0


def test_wilson_handles_zero_trials():
    assert abtest.wilson(0, 0) == (0.0, 0.0)


def test_wilson_interval_narrows_with_more_data():
    narrow = abtest.wilson(80, 100)
    wide = abtest.wilson(8, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_mean_ci_on_a_single_value_has_no_spread():
    mean, low, high = abtest.mean_ci([4.0])
    assert mean == low == high == 4.0


def test_mean_ci_widens_with_variance():
    _, low_tight, high_tight = abtest.mean_ci([3, 3, 3, 3, 3])
    _, low_loose, high_loose = abtest.mean_ci([1, 2, 3, 4, 5])
    assert (high_tight - low_tight) < (high_loose - low_loose)


# --- persistence -----------------------------------------------------------


def test_a_benchmark_reloads_from_disk(samples):
    bench = abtest.build(samples)
    again = abtest.Benchmark.load(bench.benchmark_id)
    assert {i.item_id for i in again.items} == {i.item_id for i in bench.items}
    assert again.system_of(bench.items[0].item_id) == bench.items[0].system


def test_latest_finds_the_most_recent(samples):
    first = abtest.build(samples)
    assert abtest.Benchmark.latest() == first.benchmark_id


def test_the_manifest_holds_the_answers_and_the_audio_dir_does_not(samples):
    """The UI is served from the audio directory and item ids; the mapping lives in
    the manifest, which the UI never reads."""
    bench = abtest.build(samples)
    manifest = json.loads((bench.dir / "manifest.json").read_text())
    assert any(i["system"] == "ours" for i in manifest["items"])
    names = {p.stem for p in bench.audio_dir.glob("*.wav")}
    assert not any("ours" in n or "stock" in n or "real" in n for n in names)
