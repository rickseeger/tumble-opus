"""The sustainability invariants of a long soak, as an automated test.

``tools/analyze_soak.py`` is the judgement; this is the test that runs it.
Two jobs:

1. **Unit-test the analyser itself.**  A checker that cannot fail is not a
   checker.  Synthetic series with a known defect - a cap breach, a creeping
   live count, rotting frame time, a linear memory leak, a run that did no
   work - must each be caught, and a healthy synthetic series must pass.
   Without these, a green board proves only that the analyser is quiet.

2. **Run it against the real committed soak log.**  ``soak_runs/node20_long/
   metrics.jsonl`` is the artifact of the node-20 long soak (240 demolitions,
   28,681 frames, 478 s simulated).  The test asserts each invariant against
   it individually, so the failure message names the invariant that broke and
   quotes its numbers.

Two real logs are committed, from the same seed and the same 240-demolition
horizon, differing in exactly one flag:

* ``soak_runs/node20_long``    - the driver's DEFAULT course behaviour.  Every
  debris invariant passes; ``memory`` FAILS: RSS 104 -> 391 MB, +286 MB,
  growing dead-linear at ~10 MB per 1000 frames with no deceleration.
* ``soak_runs/node20_release`` - identical, plus ``--release-chunks``.  All six
  invariants pass: RSS 111.6 -> 114.6 MB, +3.0 MB over 28,681 frames.

The two runs are bit-identical in physics (same 32,705 chunks spawned, 32,299
despawned, 46,215 retired, peak live 260), so the 283 MB difference is
attributable to one holder and one holder only: spent
:class:`~game.fracture.FractureResult` chunk descriptors retained by
demolished structures still sitting in ``app.destructibles``.  At the 1.22 MB
per structure that ``Destructible.release_chunks`` documents, 240 structures
predicts 293 MB against 286 MB measured.

That matters for what this node can and cannot claim.  The node-15 debris
budget is sound - the cap held, retirement fired in every quarter, frame time
was flat.  The leak is in *course authoring*, not in debris, and the fix
already exists in the codebase; the endless-campaign path simply has to call
it.  The default-configuration failure is therefore asserted here as a known,
quantified defect via :data:`KNOWN_MEMORY_REGRESSION`, NOT hidden by loosening
a bound: the leak may shrink, never grow, and
``test_memory_invariant_is_still_failing`` will itself fail the day someone
fixes it, so the waiver cannot outlive the defect it documents.
"""

from __future__ import annotations

import json
import math
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tools"))

from analyze_soak import (                                  # noqa: E402
    DEFAULT_FRAME_TIME_TOL,
    DEFAULT_LIVE_DRIFT_TOL,
    DEFAULT_MIN_DEMOLITIONS,
    DEFAULT_RSS_GROWTH_MB,
    DEFAULT_RSS_PLATEAU_RATIO,
    SoakSeries,
    analyze,
    check_frame_time,
    check_live_cap,
    check_live_drift,
    check_memory,
    check_retirement,
    render,
)
from game import config                                      # noqa: E402

def _log(name):
    return os.path.join(_ROOT, "soak_runs", name, "metrics.jsonl")


#: Default course behaviour: debris bounded, course authoring leaks.
SOAK_LOG = _log("node20_long")
#: Same run with --release-chunks: the fully sustainable configuration.
SOAK_LOG_RELEASE = _log("node20_release")

#: The measured memory defect, recorded as a number so it can only get better.
#: Growth at the time of measurement: 286.3 MB over 28,681 frames (254.2 MB
#: measured from the post-warm-up window the analyser uses); second-half
#: acquisition rate 0.98x the first half (i.e. dead linear - no plateau).
#: These waivers are ~10% looser than measured, to absorb host noise, and no
#: looser.  See tools/analyze_soak.py and NODE20_SOAK_REPORT.md.
KNOWN_MEMORY_REGRESSION = {
    "max_growth_mb": 315.0,
    "max_plateau_ratio": 1.10,
    "max_mb_per_1k_frames": 11.0,
}


# --------------------------------------------------------------- synthetic


def _series(n=400, live=250, cap=260, frame_ms=5.0, rss0=100.0,
            rss_rate=0.0, live_slope=0.0, frame_slope=0.0,
            demolitions=240, retire_rate=40, spawn_rate=45):
    """A synthetic soak series with knobs for each way a run can go wrong."""
    recs = []
    retired = 0
    spawned = 0
    for i in range(n):
        f = (i + 1) * 60
        t = i / max(1, n - 1)
        retired += retire_rate
        spawned += spawn_rate
        recs.append({
            "record": "sample", "frame": f,
            "elapsed_s": round(f / 170.0, 3), "sim_time_s": round(f / 60.0, 3),
            "live": int(min(cap, live + live_slope * t * live)),
            "stepped": int(min(cap, live + live_slope * t * live)),
            "frozen": 100,
            "frame_ms": frame_ms * (1.0 + frame_slope * t),
            "total_spawned": spawned, "total_frozen": retired // 2,
            "total_despawned": retired - retired // 2,
            "total_evicted": retired // 3,
            "retired_total": retired,
            "rss_mb": rss0 + rss_rate * f / 1000.0,
            "structures_placed": int(demolitions * (i + 1) / n),
            "structures_destroyed": int(demolitions * (i + 1) / n),
        })
    return SoakSeries(recs, "<synthetic>")


def _verdict(series, **kw):
    kw.setdefault("cap", 260)
    return analyze(series, **kw)


def test_healthy_synthetic_run_passes_every_invariant():
    v = _verdict(_series())
    assert v["passed"], v["failed"]
    assert v["failed"] == []
    assert len(v["invariants"]) == 6


def test_analyser_catches_a_cap_breach():
    s = _series()
    # One sample, one body over. A hard ceiling gets no tolerance at all.
    s.samples[len(s.samples) // 2]["live"] = 261
    r = check_live_cap(s, 260)
    assert not r.passed
    assert r.numbers["samples_over_cap"] == 1
    assert r.numbers["peak_live"] == 261
    assert not _verdict(s)["passed"]


def test_analyser_catches_a_breach_in_the_stepped_count_too():
    s = _series()
    s.samples[3]["stepped"] = 400
    assert not check_live_cap(s, 260).passed


def test_analyser_catches_creeping_live_debris():
    # +30% across the run: the cap holds, but retirement is losing the race.
    s = _series(live=190, live_slope=0.30)
    r = check_live_drift(s.warm(0.10), DEFAULT_LIVE_DRIFT_TOL)
    assert not r.passed
    assert r.numbers["ratio"] > DEFAULT_LIVE_DRIFT_TOL
    assert check_live_cap(s, 260).passed, "cap alone must not catch this"


def test_analyser_catches_frame_time_rot():
    s = _series(frame_slope=0.60)
    r = check_frame_time(s.warm(0.10), DEFAULT_FRAME_TIME_TOL)
    assert not r.passed
    assert r.numbers["ratio"] > DEFAULT_FRAME_TIME_TOL


def test_analyser_catches_a_linear_memory_leak():
    # Small absolute growth, but dead linear: the plateau clause must bite
    # even when the absolute bound is nowhere near.
    s = _series(n=400, rss_rate=4.0)          # 4 MB/1k frames over 24k frames
    r = check_memory(s.warm(0.10), DEFAULT_RSS_GROWTH_MB,
                     DEFAULT_RSS_PLATEAU_RATIO)
    assert not r.passed
    assert r.numbers["absolute_ok"], "absolute bound should not be what fails"
    assert not r.numbers["plateau_ok"]
    assert r.numbers["plateau_ratio"] > DEFAULT_RSS_PLATEAU_RATIO


def test_analyser_accepts_memory_that_plateaus():
    # A saturating curve: warm caches fill early, then the process stops
    # acquiring. This is what "bounded" actually looks like, and it must pass.
    # (A sqrt curve does NOT pass, and should not: over the measured window it
    # still acquires at 0.61x its early rate. The bound is 0.50x on purpose -
    # boundedness means approaching an asymptote, not merely slowing down.)
    s = _series(n=400)
    for i, rec in enumerate(s.samples):
        t = i / len(s.samples)
        rec["rss_mb"] = 100.0 + 60.0 * (1.0 - math.exp(-6.0 * t))
    r = check_memory(s.warm(0.10), DEFAULT_RSS_GROWTH_MB,
                     DEFAULT_RSS_PLATEAU_RATIO)
    assert r.passed, r.headline
    assert r.numbers["plateau_ratio"] < DEFAULT_RSS_PLATEAU_RATIO


def test_analyser_rejects_merely_slowing_growth():
    """Deceleration is not a plateau. A sqrt curve still grows forever."""
    s = _series(n=400)
    for i, rec in enumerate(s.samples):
        rec["rss_mb"] = 100.0 + 60.0 * (i / len(s.samples)) ** 0.5
    r = check_memory(s.warm(0.10), DEFAULT_RSS_GROWTH_MB,
                     DEFAULT_RSS_PLATEAU_RATIO)
    assert not r.passed
    assert 0.5 < r.numbers["plateau_ratio"] < 1.0


def test_analyser_catches_a_run_that_stopped_retiring():
    s = _series(retire_rate=0)
    assert not check_retirement(s.warm(0.10)).passed


def test_analyser_catches_a_run_that_did_no_work():
    # Flat and bounded and useless: 5 demolitions is not a soak.
    s = _series(demolitions=5)
    v = _verdict(s, min_demolitions=DEFAULT_MIN_DEMOLITIONS)
    assert not v["passed"]
    assert "demolitions" in v["failed"]


def test_analyser_catches_a_demolition_counter_that_lies():
    # The counter climbs but no debris is spawned: nothing was demolished.
    s = _series(spawn_rate=0)
    v = _verdict(s)
    assert not v["passed"]
    assert "demolitions" in v["failed"]


def test_cap_is_checked_over_the_full_series_including_warmup():
    # A breach in the first 10% must not be excused by warm-up trimming.
    s = _series()
    s.samples[1]["live"] = 999
    assert not _verdict(s)["passed"]


def test_truncated_final_line_is_tolerated(tmp_path):
    p = tmp_path / "metrics.jsonl"
    good = _series(n=40)
    with open(p, "w") as fh:
        for rec in good.samples:
            fh.write(json.dumps(rec) + "\n")
        fh.write('{"record": "sample", "frame": 99999, "li')   # torn write
    s = SoakSeries.from_jsonl(str(p))
    assert len(s) == 40


# ------------------------------------------------------------- the real log


@pytest.fixture(scope="module")
def real_soak():
    if not os.path.exists(SOAK_LOG):
        pytest.skip(f"no committed soak log at {SOAK_LOG}")
    return SoakSeries.from_jsonl(SOAK_LOG)


def test_real_soak_is_a_long_run(real_soak):
    assert real_soak.demolitions >= DEFAULT_MIN_DEMOLITIONS, (
        f"only {real_soak.demolitions} demolitions")
    assert real_soak.frames >= 20000
    assert real_soak.sim_s >= 120.0


def test_real_soak_never_exceeded_the_debris_cap(real_soak):
    r = check_live_cap(real_soak, config.DEBRIS_MAX_LIVE)
    assert r.passed, r.headline
    assert r.numbers["peak_live"] <= config.DEBRIS_MAX_LIVE
    assert r.numbers["peak_stepped"] <= config.DEBRIS_MAX_LIVE


def test_real_soak_live_debris_does_not_trend_upward(real_soak):
    r = check_live_drift(real_soak.warm(0.10), DEFAULT_LIVE_DRIFT_TOL)
    assert r.passed, r.headline


def test_real_soak_frame_time_does_not_degrade(real_soak):
    r = check_frame_time(real_soak.warm(0.10), DEFAULT_FRAME_TIME_TOL)
    assert r.passed, r.headline


def test_real_soak_kept_retiring_debris_throughout(real_soak):
    r = check_retirement(real_soak.warm(0.10))
    assert r.passed, r.headline
    assert r.numbers["retired_end"] > 10000


def test_memory_invariant_is_still_failing(real_soak):
    """The known defect, asserted as a defect.

    This test PASSES while the leak exists and FAILS when it is fixed. That is
    deliberate: it is a tripwire on the waiver below, so the waiver cannot
    quietly outlive the bug it documents. If this fails, delete
    KNOWN_MEMORY_REGRESSION and this test, and let
    ``test_real_soak_memory_leak_is_no_worse_than_measured`` become a plain
    strict assertion.
    """
    r = check_memory(real_soak.warm(0.10), DEFAULT_RSS_GROWTH_MB,
                     DEFAULT_RSS_PLATEAU_RATIO)
    assert not r.passed, (
        "The soak memory leak appears to be FIXED: " + r.headline
        + "  -> remove KNOWN_MEMORY_REGRESSION and this tripwire.")


def test_real_soak_memory_leak_is_no_worse_than_measured(real_soak):
    """Ratchet. The leak may shrink; it may not grow."""
    r = check_memory(real_soak.warm(0.10), DEFAULT_RSS_GROWTH_MB,
                     DEFAULT_RSS_PLATEAU_RATIO)
    n = r.numbers
    frames = real_soak.frames
    per_1k = n["rss_growth_mb"] / (frames / 1000.0)
    assert n["rss_growth_mb"] <= KNOWN_MEMORY_REGRESSION["max_growth_mb"], (
        f"memory leak got WORSE: {n['rss_growth_mb']} MB > waiver "
        f"{KNOWN_MEMORY_REGRESSION['max_growth_mb']} MB")
    assert n["plateau_ratio"] <= KNOWN_MEMORY_REGRESSION["max_plateau_ratio"], (
        f"memory growth is accelerating: ratio {n['plateau_ratio']}")
    assert per_1k <= KNOWN_MEMORY_REGRESSION["max_mb_per_1k_frames"], (
        f"leak rate got worse: {per_1k:.2f} MB/1k frames")


def test_full_verdict_reports_exactly_the_known_failure(real_soak):
    v = analyze(real_soak, cap=config.DEBRIS_MAX_LIVE)
    assert v["failed"] == ["memory"], (
        "expected the memory invariant to be the only failure; got "
        f"{v['failed']}\n" + render(v))
    assert not v["passed"]


def test_render_names_every_invariant(real_soak):
    text = render(analyze(real_soak, cap=config.DEBRIS_MAX_LIVE))
    for name in ("demolitions", "live_cap", "live_no_drift", "frame_time",
                 "retirement", "memory"):
        assert name in text
    assert "VERDICT" in text


# ------------------------------------------- the fully sustainable real run


@pytest.fixture(scope="module")
def release_soak():
    if not os.path.exists(SOAK_LOG_RELEASE):
        pytest.skip(f"no committed soak log at {SOAK_LOG_RELEASE}")
    return SoakSeries.from_jsonl(SOAK_LOG_RELEASE)


def test_release_soak_passes_every_invariant_strictly(release_soak):
    """The headline result: with spent chunks released, Tumble is bounded.

    No waiver, no relaxed threshold - the default thresholds, all six
    invariants, over 240 real demolitions.
    """
    v = analyze(release_soak, cap=config.DEBRIS_MAX_LIVE)
    assert v["passed"], render(v)
    assert v["failed"] == []


def test_release_soak_memory_actually_plateaus(release_soak):
    r = check_memory(release_soak.warm(0.10), DEFAULT_RSS_GROWTH_MB,
                     DEFAULT_RSS_PLATEAU_RATIO)
    assert r.passed, r.headline
    n = r.numbers
    assert n["rss_growth_mb"] < 20.0, f"expected single-digit MB, got {n}"
    assert n["plateau_ratio"] < 0.25, "growth should be near-flat, not merely slower"


def test_the_two_runs_differ_only_in_memory(real_soak, release_soak):
    """Same seed, same horizon, same physics: so the delta is one holder.

    If this ever fails, the A/B is no longer controlled and the attribution
    of the leak to retained chunk descriptors is no longer supported.
    """
    for key in ("total_spawned", "total_despawned", "retired_total",
                "structures_destroyed"):
        a = real_soak.icol(key)[-1]
        b = release_soak.icol(key)[-1]
        assert a == b, f"{key}: baseline {a} != release {b}"
    assert real_soak.frames == release_soak.frames
    assert max(real_soak.icol("live")) == max(release_soak.icol("live"))

    leaked = (real_soak.col("rss_mb")[-1] - real_soak.col("rss_mb")[0])
    clean = (release_soak.col("rss_mb")[-1] - release_soak.col("rss_mb")[0])
    assert leaked > 200.0 and clean < 20.0, (
        f"expected a large baseline/release RSS split; got {leaked:.1f} MB "
        f"vs {clean:.1f} MB")


def test_leak_matches_the_documented_per_structure_cost(real_soak,
                                                        release_soak):
    """Quantitative attribution, not a hand-wave.

    ``Destructible.release_chunks`` documents 1.22 MB of chunk descriptors per
    structure. 240 structures predicts ~293 MB; the baseline/release split
    must land within 25% of that or the diagnosis is wrong.
    """
    per_structure_mb = 1.22
    n = real_soak.demolitions
    predicted = per_structure_mb * n
    observed = ((real_soak.col("rss_mb")[-1] - real_soak.col("rss_mb")[0])
                - (release_soak.col("rss_mb")[-1]
                   - release_soak.col("rss_mb")[0]))
    assert abs(observed - predicted) / predicted < 0.25, (
        f"attribution mismatch: predicted ~{predicted:.0f} MB from {n} "
        f"structures, observed {observed:.0f} MB")
