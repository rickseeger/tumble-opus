"""The sustained-demolition soak, as assertions (node 16).

`tools_soak.py` *measures*: it drives the real game through many consecutive
structure destructions and records live/active/slept body counts, per-step
physics time, debris positions and process RSS. This module is where those
measurements become **bounds that fail the build**.

The division of labour matters, and it is deliberate:

* every bound lives in exactly one place - a module-level constant in
  `tools_soak` - and is checked in exactly one place, `SoakResult.failures()`.
  So `tools_soak.py` exiting non-zero and this test file failing can never
  disagree about what "sustainable" means.
* each test below then asserts on ONE named bound against a shared run, so a
  failure names the property that broke rather than dumping a wall of numbers.

Would these assertions actually catch a regression?
---------------------------------------------------
That is the question a soak test has to answer, so
`test_the_bounds_would_fail_if_the_budget_were_broken` answers it directly: it
takes the healthy measurement and mutates each field the way the corresponding
bug would move it - debris unbounded, debris never slept, debris leaking - and
requires `failures()` to name the right bound. It does not weaken anything; it
proves the bounds have teeth.

Cost
----
The `--ci` shape (12 structures, 15 s of simulated settle) runs in ~8 s wall on
this machine and spawns ~2000 debris bodies, 7.7x the 260-body cap, so the cap
is genuinely overflowed several times over. One session-scoped run is shared by
every test here. The longer shape is the documented command:

    .venv/bin/python tools_soak.py --structures 60 --settle-seconds 20

Neither shape sleeps, waits on wall clock, or opens a window: the run is a
fixed frame count at a fixed timestep, so its duration is bounded by
construction.
"""

import math

import pytest

import tools_soak
from tools_soak import (
    ACTIVE_BASELINE_MAX,
    LIVE_TREND_FACTOR,
    RECOVERY_SECONDS_MAX,
    RSS_GROWTH_MAX_MB,
    RSS_PLATEAU_RATIO_MAX,
    STEP_BUDGET_MS,
    STEP_DEGRADATION_FACTOR,
    STEP_MAX_MS,
    STEP_P95_DEGRADATION_FACTOR,
    WORLD_ABS_X_MAX,
    WORLD_Z_MAX,
    WORLD_Z_MIN,
)

#: The CI shape. 12 structures is the smallest count that still overflows the
#: 260-body cap several times over (measured: 1998 bodies spawned, 7.7x) while
#: keeping the whole file under ~10 s.
CI_STRUCTURES = 12
CI_SETTLE_FRAMES = 60 * 15


@pytest.fixture(scope="module")
def soak():
    """One real soak run, shared by every bound below."""
    return tools_soak.run_soak(
        structures_count=CI_STRUCTURES,
        settle_frames=CI_SETTLE_FRAMES,
        verbose=False,
    )


# ------------------------------------------------------- the run is a real run
def test_the_soak_actually_demolished_things(soak):
    """Guard against a vacuous pass: no destruction, no proof."""
    assert soak.structures_destroyed == CI_STRUCTURES, (
        f"only {soak.structures_destroyed} of {CI_STRUCTURES} structures "
        f"were destroyed"
    )
    assert soak.physics_steps > 3000, (
        f"only {soak.physics_steps} physics steps ran"
    )
    # The load bound: the cap must have been overflowed several times over,
    # or nothing was under pressure and every bound below is trivial.
    assert soak.cap_overflows >= 2.0, (
        f"only spawned {soak.cap_overflows:.1f}x the cap of {soak.cap}"
    )
    assert soak.total_evicted > 0, "the budget never had to evict anything"
    assert soak.total_frozen > 0, "nothing ever settled and froze"


# ------------------------------------------------------------- bound 1: the cap
def test_live_debris_never_exceeds_the_configured_cap(soak):
    """Bound 1. Sampled every frame of every cycle, and during settling."""
    assert soak.cap_breaches == [], (
        f"live debris exceeded the cap of {soak.cap} on "
        f"{len(soak.cap_breaches)} sampled frames; worst was "
        f"{max(n for _, n in soak.cap_breaches) if soak.cap_breaches else 0}"
    )
    assert soak.peak_live <= soak.cap, (
        f"peak live debris {soak.peak_live} exceeded the cap {soak.cap}"
    )
    # And the cap was actually reached - a cap nothing approaches proves
    # nothing about the cap.
    assert soak.peak_live == soak.cap, (
        f"peak live debris only reached {soak.peak_live} of the {soak.cap} "
        f"cap, so the ceiling was never actually tested"
    )


# ------------------------------------- bound 2: return toward baseline / sleep
def test_active_debris_returns_to_baseline_after_the_bursts(soak):
    """Bound 2. Settled debris is retired, not accumulated forever."""
    assert soak.final_active <= ACTIVE_BASELINE_MAX, (
        f"{soak.final_active} debris bodies are still being solved after "
        f"settling (baseline allows {ACTIVE_BASELINE_MAX})"
    )
    # Peaked high, came back down: that is the shape of a burst that recovers.
    assert soak.peak_active > 10 * max(ACTIVE_BASELINE_MAX, 1), (
        f"active debris only peaked at {soak.peak_active}; there was no burst "
        f"to recover from"
    )


def test_the_return_to_baseline_is_prompt_not_eventual(soak):
    """Bound 2, with teeth: recovery must be fast in SIMULATED time.

    `final_active` on its own could be satisfied by settling for an hour. This
    bounds how long the recovery takes, so debris that retires only after an
    unreasonable wait still fails.
    """
    assert soak.recovery_seconds >= 0.0, (
        "active debris never returned to baseline during the settle phase"
    )
    assert soak.recovery_seconds <= RECOVERY_SECONDS_MAX, (
        f"took {soak.recovery_seconds:.2f} simulated s to return to "
        f"{ACTIVE_BASELINE_MAX} active bodies (allowed "
        f"{RECOVERY_SECONDS_MAX} s)"
    )


def test_settled_debris_is_slept_rather_than_despawned(soak):
    """Retirement must be a freeze, not a disappearance.

    A budget that met every bound by deleting rubble the instant it landed
    would be cheating: the locked 1A art direction is glowing debris that
    *stays*. So the run must end with real rubble present and none of it
    costing the solver anything.
    """
    assert soak.final_frozen > 0, (
        "no frozen rubble survived the run - debris is being deleted, not "
        "slept"
    )
    assert soak.final_total_bodies == soak.final_frozen + soak.final_live
    assert soak.final_active == 0 or soak.final_active <= ACTIVE_BASELINE_MAX
    # Slept bodies are the bulk of what is left, and they are free.
    assert soak.final_frozen >= soak.final_live, (
        f"{soak.final_live} live vs {soak.final_frozen} frozen: the pile is "
        f"not actually retiring"
    )


# --------------------------------------------------- bound 3: per-step budget
def test_physics_step_time_stays_within_the_frame_budget(soak):
    """Bound 3. The numeric budget is STEP_BUDGET_MS = 8.333 ms per step.

    Derivation, not a vibe: the sim runs a fixed 1/120 s timestep and the game
    targets 60 fps, so one rendered frame is two physics steps and a step's
    share of a 16.667 ms frame is 8.333 ms. A mean above that means physics
    alone cannot keep up with real time.
    """
    assert STEP_BUDGET_MS == pytest.approx(8.333, abs=1e-3)
    assert soak.step_mean_ms <= STEP_BUDGET_MS, (
        f"mean physics step {soak.step_mean_ms:.3f} ms exceeds the "
        f"{STEP_BUDGET_MS} ms budget - the sim cannot hold 60 fps"
    )
    assert soak.step_max_ms <= STEP_MAX_MS, (
        f"worst physics step {soak.step_max_ms:.3f} ms exceeds the "
        f"{STEP_MAX_MS} ms stutter ceiling"
    )


def test_per_step_cost_does_not_grow_as_demolitions_accumulate(soak):
    """Bound 3b. No unbounded growth in per-step work over the run."""
    assert soak.degradation <= STEP_DEGRADATION_FACTOR, (
        f"late steps average {soak.degradation:.2f}x the early mean "
        f"({soak.early_mean_ms:.3f} -> {soak.late_mean_ms:.3f} ms); "
        f"allowed {STEP_DEGRADATION_FACTOR}x"
    )
    assert soak.p95_degradation <= STEP_P95_DEGRADATION_FACTOR, (
        f"late p95 is {soak.p95_degradation:.2f}x the early p95 "
        f"({soak.early_p95_ms:.3f} -> {soak.late_p95_ms:.3f} ms); "
        f"allowed {STEP_P95_DEGRADATION_FACTOR}x"
    )


# ------------------------------------------------------- bound 4: no leak
def test_neither_body_count_nor_memory_grows_monotonically(soak):
    """Bound 4. The no-leak check, in three independent forms."""
    # (a) body count plateaus rather than trending up
    assert soak.live_trend <= LIVE_TREND_FACTOR, (
        f"per-cycle peak live debris trends up {soak.live_trend:.2f}x "
        f"early-to-late (allowed {LIVE_TREND_FACTOR}x)"
    )

    # (b) RSS plateaus. This is the form that does not depend on run length:
    # a real leak is linear, so its late per-cycle growth rate equals its
    # early rate and the ratio sits near 1.0. Measured before the DebrisBody
    # release fix: 1.08 MB/cycle late vs 1.10 early = 0.99x, i.e. no
    # convergence at all. After: ~0.04 MB/cycle.
    assert soak.rss_plateau_ratio <= RSS_PLATEAU_RATIO_MAX, (
        f"RSS is still growing at {soak.rss_late_slope:.2f} MB/cycle late vs "
        f"{soak.rss_early_slope:.2f} MB/cycle early "
        f"({soak.rss_plateau_ratio:.2f}x, allowed {RSS_PLATEAU_RATIO_MAX}x) - "
        f"memory is not plateauing"
    )

    # (c) and an absolute ceiling, as a backstop
    assert soak.rss_growth_mb <= RSS_GROWTH_MAX_MB, (
        f"RSS grew {soak.rss_growth_mb:.1f} MB "
        f"({soak.rss_start_mb:.1f} -> {soak.rss_settled_mb:.1f})"
    )


def test_everything_leaves_when_the_field_is_cleared(soak):
    """Teardown leaves nothing behind - bodies or bookkeeping."""
    assert soak.live_bodies_after_clear == 0, (
        f"{soak.live_bodies_after_clear} debris bodies survived clear()"
    )
    assert soak.tracked_records_after_clear == 0, (
        f"{soak.tracked_records_after_clear} debris records still tracked "
        f"after clear()"
    )


# ------------------------------------------- bound 5: no NaN, no escapees
def test_no_debris_position_is_nan_or_out_of_the_world(soak):
    """Bound 5. Every body's live Bullet pose, swept throughout the run."""
    assert soak.bodies_swept > 10000, (
        f"only {soak.bodies_swept} body-poses were checked; the sanity sweep "
        f"barely ran"
    )
    assert soak.nan_bodies == 0, (
        f"{soak.nan_bodies} debris bodies had a non-finite position - the "
        f"solver blew up"
    )
    assert soak.out_of_bounds == [], (
        f"{len(soak.out_of_bounds)} debris bodies left the sane world box; "
        f"first was {soak.out_of_bounds[0] if soak.out_of_bounds else None}"
    )
    # The recorded extents are themselves finite and sane.
    for value in (soak.max_abs_x, soak.max_abs_y, soak.max_z, soak.min_z):
        assert math.isfinite(value)
    assert soak.max_abs_x <= WORLD_ABS_X_MAX
    assert WORLD_Z_MIN <= soak.min_z and soak.max_z <= WORLD_Z_MAX


# --------------------------------------- the bounds have teeth (mutation test)
def test_the_bounds_would_fail_if_the_budget_were_broken(soak):
    """The meta-test: each bug the soak exists to catch must trip a bound.

    Without this, a soak test's green tick means "nothing crashed". With it,
    the green tick means "and the checks can tell the difference".

    Each case takes the real, healthy measurement, mutates the one field the
    named bug would move, and requires `failures()` to name it. The healthy
    run is never mutated in place - `dataclasses.replace` gives each case its
    own copy.
    """
    import dataclasses

    assert soak.failures() == [], (
        f"the reference run is not healthy: {soak.failures()}"
    )

    def broken(**changes):
        return dataclasses.replace(soak, **changes).failures()

    # 1. debris unbounded: the cap is breached
    f = broken(peak_live=soak.cap + 1,
               cap_breaches=[(100, soak.cap + 1), (101, soak.cap + 40)])
    assert any(x.startswith("CAP:") for x in f), f

    # 2. debris never slept: active bodies never come back down
    f = broken(final_active=180, recovery_seconds=-1.0)
    assert any(x.startswith("BASELINE:") for x in f), f

    # 3. slept, but only after an unreasonable wait
    f = broken(recovery_seconds=RECOVERY_SECONDS_MAX + 5.0)
    assert any("return to" in x for x in f), f

    # 4. per-step work grows without bound as demolitions accumulate
    f = broken(step_ms=([0.5] * 1000) + ([9.0] * 1000) + ([40.0] * 1000))
    assert any(x.startswith("DEGRADATION:") for x in f), f

    # 5. physics blows the frame budget outright
    f = broken(step_ms=[STEP_BUDGET_MS + 1.0] * 1000)
    assert any("STEP BUDGET" in x for x in f), f

    # 6. a memory leak - linear, so early and late rates match
    f = broken(rss_early_slope=1.10, rss_late_slope=1.08,
               rss_settled_mb=soak.rss_start_mb + 200.0)
    assert any("not plateauing" in x for x in f), f
    assert any("RSS grew" in x for x in f), f

    # 7. debris accumulating across cycles. `live_trend` is derived, not
    # stored, so this mutates the real per-cycle samples it is computed from -
    # a cap that drifted upward every cycle instead of holding.
    climbing = [
        dataclasses.replace(c, peak_live=50 + 20 * i)
        for i, c in enumerate(soak.cycles)
    ]
    f = broken(cycles=climbing)
    assert any("trends up" in x for x in f), f
    # ...and the healthy, flat-at-the-cap shape must NOT trip it.
    flat = [dataclasses.replace(c, peak_live=soak.cap) for c in soak.cycles]
    assert not any("trends up" in x for x in broken(cycles=flat))

    # 8. the solver blows up
    f = broken(nan_bodies=3)
    assert any("non-finite" in x for x in f), f
    f = broken(out_of_bounds=[("debris_x", 1e9, 0.0, 0.0)])
    assert any("left the" in x for x in f), f

    # 9. bodies or bookkeeping survive teardown
    f = broken(live_bodies_after_clear=12)
    assert any("after DebrisField.clear()" in x for x in f), f
    f = broken(tracked_records_after_clear=12)
    assert any("still tracked" in x for x in f), f

    # 10. a vacuous run that never pressured the budget
    f = broken(total_spawned=10)
    assert any(x.startswith("LOAD:") for x in f), f
    f = broken(structures_destroyed=0, total_spawned=0)
    assert any(x.startswith("LOAD:") for x in f), f


def test_the_cli_and_the_test_suite_share_one_definition_of_passing(soak):
    """No second opinion: `failures()` is the only verdict in the codebase."""
    assert soak.ok is (soak.failures() == [])
    assert soak.to_dict()["ok"] is soak.ok
    assert soak.to_dict()["failures"] == soak.failures()
    # And the report text agrees.
    text = tools_soak.report(soak)
    assert "VERDICT: PASS" in text
    assert str(soak.cap) in text
