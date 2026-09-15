#!/usr/bin/env python3
"""Sustained-demolition soak: is Tumble's debris budget actually sustainable?

Node 15 landed the debris budget - one global cap on simultaneously simulated
bodies, a settle-freeze, an eviction policy, despawning. This harness is the
PROOF, under load the game will never see in a real play session:

    .venv/bin/python tools_soak.py                 # the real soak (~40 structures)
    .venv/bin/python tools_soak.py --ci            # short mode, lives in pytest
    .venv/bin/python tools_soak.py --structures 80 # longer
    .venv/bin/python tools_soak.py --json out.json

What it drives is the REAL game, not a stand-in:

* :class:`game.app.TumbleApp` in ``window-type none`` - the same class, the
  same ``step_frame()`` the windowed playtest runs;
* :meth:`game.app.TumbleApp.strike` -> :class:`game.damage.DamageSystem` ->
  ``on_destroy`` -> :meth:`game.app.TumbleApp.demolish` -> the real
  :meth:`game.debris.DebrisField.demolish` - so destruction happens the only
  way it happens in the game: accumulated damage crossing an integrity
  threshold;
* real Bullet: :meth:`game.physics.PhysicsWorld.advance` stepping a real
  :class:`~panda3d.bullet.BulletWorld` at the shipping 1/120 s timestep, with
  real convex-hull debris bodies. Nothing here is mocked, monkeypatched or
  reimplemented - the harness only *reads* counters and *times* steps.

Structures are authored down the course at runtime (the shipping five are not
enough to overflow a 260-body cap 'several times over'), placed and registered
through the same :func:`game.structures.generate` /
:func:`game.structures.attach_proxies` / :meth:`DamageSystem.register` path the
course build uses.

The bounds it asserts are in :mod:`tests.test_soak`; this module measures and
reports. `SoakResult.failures()` is the single source of truth for both.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field as _dc_field
from typing import List, Optional, Tuple

from game import config

config.bootstrap_display(headless=True)

from game import structures                      # noqa: E402
from game.player import InputState               # noqa: E402

# --------------------------------------------------------------- the bounds
#: Per-physics-step wall-time budget, in milliseconds. A step is 1/120 s and
#: the game targets 60 fps, so a frame is 2 steps: the honest budget is half a
#: 60 fps frame, 16.667/2 = 8.333 ms per step. Exceeding this MEAN means the
#: sim cannot keep up with real time and the game is in slow motion.
STEP_BUDGET_MS = 8.333

#: Bound on the *worst single step*. Generous relative to the mean because one
#: step containing a fresh 250-body spawn is legitimately expensive, and the
#: fixed-timestep accumulator (MAX_STEPS_PER_FRAME) is designed to absorb an
#: occasional hitch. A max above this is a visible stutter, not a hitch.
STEP_MAX_MS = 50.0

#: Late-window mean step time may not exceed early-window mean by more than
#: this factor. This is the no-degradation bound: if debris accumulated, or
#: frozen rubble kept costing the solver, late steps would get steadily slower.
STEP_DEGRADATION_FACTOR = 1.6
#: Same, on p95 - catches a growing tail that a mean can hide.
STEP_P95_DEGRADATION_FACTOR = 2.0

#: After demolition stops and debris is given time to settle, the number of
#: ACTIVE (still being solved) debris bodies must come back to at most this.
#: Zero is the real target and what we measure; a small slack keeps the bound
#: from being decided by one shard balanced on a corner.
ACTIVE_BASELINE_MAX = 4

#: Process RSS growth allowance, in MB, across the WHOLE soak, measured after
#: the field is cleared and everything has settled. This is the no-leak bound,
#: and it is deliberately an absolute figure rather than a per-cycle rate: a
#: real leak is linear in cycles, so a run of 40+ structures blows any fixed
#: allowance, while a bounded system plateaus and passes at any length.
#:
#: Calibration, measured on this stack (40 structures, ~5800 bodies spawned):
#: three separate real leaks had to be fixed to get under this - Panda's
#: interned TransformState table (+266 MB), the unbounded ShatterEvent history
#: pinning every body ever spawned (+20 MB/cycle), and retained fracture
#: descriptors (+1.22 MB/structure). With those fixed the same run settles at
#: +17 MB and a 100-structure run at +21 MB, i.e. it plateaus. 45 MB leaves
#: room for allocator noise while still failing loudly if any of the three
#: regress: the smallest of them alone would put a 40-structure run at +49 MB.
RSS_GROWTH_MAX_MB = 45.0

#: The live-body count must not trend upward across cycles. Compared as the
#: mean of the last third of per-cycle peaks against the first third; the cap
#: means peaks plateau, so any real upward trend is a budget failure.
LIVE_TREND_FACTOR = 1.15

#: Archetypes cycled down the course, and how far apart they are placed.
SOAK_ARCHETYPES = ("cluster", "arch", "slab", "tower")
SOAK_SPACING = 30.0
SOAK_FIRST_Y = 40.0
#: Frames (at 60 fps) spent on each structure: strike it, then watch the
#: debris fly. Long enough that debris from consecutive structures coexists,
#: which is what puts the cap under real pressure.
FRAMES_PER_STRUCTURE = 90
#: Frames of quiet at the end, for the return-to-baseline measurement.
SETTLE_FRAMES = 60 * 45
FRAME_DT = 1.0 / 60.0


def rss_mb() -> float:
    """Process resident set size in MB, from /proc. No psutil dependency."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:                                    # pragma: no cover
        pass
    return -1.0


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


# --------------------------------------------------------------- the result
@dataclass
class CycleSample:
    """One structure's demolition, as measured."""

    index: int
    archetype: str
    chunks: int
    spawned: int
    skipped_for_budget: int
    peak_live: int
    peak_active: int
    end_live: int
    end_frozen: int
    rss_mb: float
    destroyed: bool


@dataclass
class SoakResult:
    """Everything measured, plus the bound checks. No printing, no asserting."""

    structures_destroyed: int = 0
    structures_placed: int = 0
    cap: int = 0
    peak_live: int = 0
    final_live: int = 0
    peak_active: int = 0
    final_active: int = 0
    peak_total_bodies: int = 0
    final_total_bodies: int = 0
    total_spawned: int = 0
    total_despawned: int = 0
    total_frozen: int = 0
    total_evicted: int = 0
    physics_steps: int = 0
    step_ms: List[float] = _dc_field(default_factory=list, repr=False)
    cycles: List[CycleSample] = _dc_field(default_factory=list, repr=False)
    rss_start_mb: float = 0.0
    rss_peak_mb: float = 0.0
    rss_end_mb: float = 0.0
    rss_settled_mb: float = 0.0
    rss_trace_mb: List[float] = _dc_field(default_factory=list, repr=False)
    live_bodies_after_clear: int = 0
    tracked_records_after_clear: int = 0
    wall_seconds: float = 0.0
    sim_seconds: float = 0.0
    chunks_released: int = 0
    cap_breaches: List[Tuple[int, int]] = _dc_field(default_factory=list)
    states_reclaimed: int = 0

    # ------------------------------------------------------------- stats
    @property
    def step_max_ms(self) -> float:
        return max(self.step_ms) if self.step_ms else 0.0

    @property
    def step_mean_ms(self) -> float:
        return _mean(self.step_ms)

    def window(self, which: str) -> List[float]:
        """Early or late third of the step-time series."""
        n = len(self.step_ms)
        if n < 6:
            return list(self.step_ms)
        third = n // 3
        return self.step_ms[:third] if which == "early" else self.step_ms[-third:]

    @property
    def early_mean_ms(self) -> float:
        return _mean(self.window("early"))

    @property
    def late_mean_ms(self) -> float:
        return _mean(self.window("late"))

    @property
    def early_p95_ms(self) -> float:
        return _percentile(self.window("early"), 0.95)

    @property
    def late_p95_ms(self) -> float:
        return _percentile(self.window("late"), 0.95)

    @property
    def degradation(self) -> float:
        e = self.early_mean_ms
        return (self.late_mean_ms / e) if e > 0 else 0.0

    @property
    def p95_degradation(self) -> float:
        e = self.early_p95_ms
        return (self.late_p95_ms / e) if e > 0 else 0.0

    @property
    def rss_growth_mb(self) -> float:
        """Growth measured at the settled end of the run, not at the peak."""
        return self.rss_settled_mb - self.rss_start_mb

    @property
    def live_trend(self) -> float:
        """Late-cycle mean peak-live over early-cycle mean peak-live."""
        peaks = [c.peak_live for c in self.cycles]
        if len(peaks) < 6:
            return 1.0
        third = len(peaks) // 3
        early = _mean(peaks[:third])
        return (_mean(peaks[-third:]) / early) if early > 0 else 0.0

    @property
    def cap_overflows(self) -> float:
        """How many times over the cap the run actually spawned."""
        return (self.total_spawned / self.cap) if self.cap else 0.0

    # ------------------------------------------------------------ verdict
    def failures(self) -> List[str]:
        """Every violated bound, as a human-readable string. Empty == healthy.

        The single source of truth: `tests/test_soak.py` asserts on this and
        the CLI exits non-zero on it, so the command and the test suite can
        never disagree about what 'passing' means.
        """
        out: List[str] = []

        if self.cap_breaches:
            worst = max(n for _, n in self.cap_breaches)
            out.append(
                f"CAP: live debris exceeded the cap of {self.cap} on "
                f"{len(self.cap_breaches)} sampled ticks (worst {worst})"
            )
        if self.peak_live > self.cap:
            out.append(f"CAP: peak live {self.peak_live} > cap {self.cap}")
        if self.structures_destroyed < 1:
            out.append("LOAD: nothing was destroyed - the soak did not run")
        if self.cap_overflows < 2.0:
            out.append(
                f"LOAD: only spawned {self.cap_overflows:.1f}x the cap; the "
                f"budget was never put under sustained pressure"
            )
        if self.final_active > ACTIVE_BASELINE_MAX:
            out.append(
                f"BASELINE: {self.final_active} debris bodies still active "
                f"after settling (allowed {ACTIVE_BASELINE_MAX})"
            )
        if self.step_mean_ms > STEP_BUDGET_MS:
            out.append(
                f"STEP BUDGET: mean step {self.step_mean_ms:.3f} ms exceeds "
                f"the {STEP_BUDGET_MS} ms budget"
            )
        if self.step_max_ms > STEP_MAX_MS:
            out.append(
                f"STEP BUDGET: worst step {self.step_max_ms:.3f} ms exceeds "
                f"the {STEP_MAX_MS} ms ceiling"
            )
        if self.degradation > STEP_DEGRADATION_FACTOR:
            out.append(
                f"DEGRADATION: late steps are {self.degradation:.2f}x the "
                f"early mean (allowed {STEP_DEGRADATION_FACTOR}x)"
            )
        if self.p95_degradation > STEP_P95_DEGRADATION_FACTOR:
            out.append(
                f"DEGRADATION: late p95 is {self.p95_degradation:.2f}x the "
                f"early p95 (allowed {STEP_P95_DEGRADATION_FACTOR}x)"
            )
        if self.rss_growth_mb > RSS_GROWTH_MAX_MB:
            out.append(
                f"LEAK: RSS grew {self.rss_growth_mb:.1f} MB "
                f"({self.rss_start_mb:.1f} -> {self.rss_settled_mb:.1f}) "
                f"over {self.structures_destroyed} demolitions "
                f"(allowed {RSS_GROWTH_MAX_MB} MB)"
            )
        if self.live_trend > LIVE_TREND_FACTOR:
            out.append(
                f"LEAK: per-cycle peak live debris trends up "
                f"{self.live_trend:.2f}x early-to-late "
                f"(allowed {LIVE_TREND_FACTOR}x)"
            )
        if self.live_bodies_after_clear != 0:
            out.append(
                f"LEAK: {self.live_bodies_after_clear} debris bodies still in "
                f"the world after DebrisField.clear()"
            )
        if self.tracked_records_after_clear != 0:
            out.append(
                f"LEAK: {self.tracked_records_after_clear} debris records "
                f"still tracked in the field's registries after clear()"
            )
        return out

    @property
    def ok(self) -> bool:
        return not self.failures()

    def to_dict(self) -> dict:
        return {
            "structures_destroyed": self.structures_destroyed,
            "cap": self.cap,
            "cap_overflows": round(self.cap_overflows, 2),
            "peak_live": self.peak_live,
            "final_live": self.final_live,
            "peak_active": self.peak_active,
            "final_active": self.final_active,
            "total_spawned": self.total_spawned,
            "total_despawned": self.total_despawned,
            "total_evicted": self.total_evicted,
            "physics_steps": self.physics_steps,
            "step_mean_ms": round(self.step_mean_ms, 4),
            "step_max_ms": round(self.step_max_ms, 4),
            "step_budget_ms": STEP_BUDGET_MS,
            "early_mean_ms": round(self.early_mean_ms, 4),
            "late_mean_ms": round(self.late_mean_ms, 4),
            "degradation": round(self.degradation, 3),
            "early_p95_ms": round(self.early_p95_ms, 4),
            "late_p95_ms": round(self.late_p95_ms, 4),
            "p95_degradation": round(self.p95_degradation, 3),
            "rss_start_mb": round(self.rss_start_mb, 1),
            "rss_peak_mb": round(self.rss_peak_mb, 1),
            "rss_settled_mb": round(self.rss_settled_mb, 1),
            "rss_growth_mb": round(self.rss_growth_mb, 1),
            "rss_trace_mb": [round(v, 1) for v in self.rss_trace_mb],
            "live_trend": round(self.live_trend, 3),
            "live_bodies_after_clear": self.live_bodies_after_clear,
            "tracked_records_after_clear": self.tracked_records_after_clear,
            "states_reclaimed": self.states_reclaimed,
            "chunks_released": self.chunks_released,
            "wall_seconds": round(self.wall_seconds, 2),
            "sim_seconds": round(self.sim_seconds, 2),
            "failures": self.failures(),
            "ok": self.ok,
        }


# ---------------------------------------------------------------- the soak
def _place(app, index: int, seed_base: int):
    """Author one more destructible on the course, the shipping way."""
    archetype = SOAK_ARCHETYPES[index % len(SOAK_ARCHETYPES)]
    x = -8.0 if index % 2 else 8.0
    y = SOAK_FIRST_Y + index * SOAK_SPACING
    d = structures.generate(
        archetype, (x, y, 0.0), seed_base + index,
        name=f"soak_{index:03d}_{archetype}",
    )
    structures.attach_proxies(app.physics, d)      # real static proxies
    app.destructibles.append(d)
    app.damage.register(d)                         # real integrity threshold
    return d


def run_soak(
    structures_count: int = 40,
    frames_per_structure: int = FRAMES_PER_STRUCTURE,
    settle_frames: int = SETTLE_FRAMES,
    seed_base: int = 40000,
    verbose: bool = True,
) -> SoakResult:
    """Demolish *structures_count* structures back to back and measure.

    Drives `TumbleApp.step_frame`, the real frame function, and destroys
    through `app.strike` -> DamageSystem -> app.demolish. The only
    instrumentation is a timing wrapper around `PhysicsWorld.step_fixed`,
    which calls straight through - no behaviour is replaced.
    """
    from game.app import TumbleApp

    app = TumbleApp(headless=True)
    result = SoakResult(cap=app.debris.max_live)
    result.rss_start_mb = rss_mb()
    result.rss_peak_mb = result.rss_start_mb
    result.rss_trace_mb.append(result.rss_start_mb)

    # Time every fixed substep Bullet actually runs. This wraps the real
    # method and calls it - the physics is untouched.
    real_step_fixed = app.physics.step_fixed

    def timed_step_fixed(steps: int = 1) -> int:
        t0 = time.perf_counter()
        taken = real_step_fixed(steps)
        per = (time.perf_counter() - t0) * 1000.0 / max(int(steps), 1)
        result.step_ms.extend([per] * int(steps))
        return taken

    app.physics.step_fixed = timed_step_fixed
    app.input_state = InputState()

    if verbose:
        print(f"[soak] cap={result.cap} bodies  "
              f"step budget={STEP_BUDGET_MS} ms  "
              f"structures={structures_count}")
        print(f"[soak] {'cyc':>4} {'archetype':>10} {'chunks':>7} {'spawn':>6} "
              f"{'skip':>5} {'pkLive':>7} {'pkAct':>6} {'live':>5} "
              f"{'frozen':>7} {'rssMB':>7}")

    started = time.perf_counter()

    for i in range(structures_count):
        d = _place(app, i, seed_base)
        # Stand the player just short of the structure: in weapon range, and
        # close enough that the eviction protection radius is genuinely in
        # play (evicting debris in front of the player is forbidden, which is
        # the hard case for the budget).
        app.player.np.setPos(0.0, d.world_bounds()[0][1] - 14.0,
                             config.SPAWN_POS[2])

        peak_live = peak_active = 0
        for _ in range(frames_per_structure):
            app.step_frame(FRAME_DT)             # the REAL frame function
            if d.intact:
                # The real gameplay path: damage accumulates, crosses the
                # integrity threshold, and destruction is the consequence.
                app.strike(d.default_impact_point())

            live = app.debris.live_count
            active = app.debris.active_count()
            peak_live = max(peak_live, live)
            peak_active = max(peak_active, active)
            # Sampled EVERY frame, and recorded rather than asserted, so the
            # report can say how badly and how often rather than just dying.
            if live > result.cap:
                result.cap_breaches.append((app.physics.step_count, live))
            result.peak_total_bodies = max(result.peak_total_bodies,
                                           app.debris.total_bodies)

        snap = app.debris.snapshot()
        rss = rss_mb()
        result.rss_peak_mb = max(result.rss_peak_mb, rss)
        result.rss_trace_mb.append(rss)
        result.peak_live = max(result.peak_live, peak_live)
        result.peak_active = max(result.peak_active, peak_active)

        last = app.debris.events[-1] if app.debris.events else None
        sample = CycleSample(
            index=i, archetype=d.archetype, chunks=d.chunk_count,
            spawned=last.spawned if last else 0,
            skipped_for_budget=last.skipped_for_budget if last else 0,
            peak_live=peak_live, peak_active=peak_active,
            end_live=snap["live"], end_frozen=snap["frozen"],
            rss_mb=rss, destroyed=not d.intact,
        )
        result.cycles.append(sample)
        if not d.intact:
            result.structures_destroyed += 1
            # The structure is rubble: its pre-generated chunk descriptors are
            # spent (1.22 MB per structure, measured) and an endless run must
            # let them go. This is the real API - `Destructible.release_chunks`
            # - not a harness-local trick.
            result.chunks_released += d.release_chunks()
        result.structures_placed += 1

        if verbose:
            print(f"[soak] {i:4d} {d.archetype:>10} {d.chunk_count:7d} "
                  f"{sample.spawned:6d} {sample.skipped_for_budget:5d} "
                  f"{peak_live:7d} {peak_active:6d} {snap['live']:5d} "
                  f"{snap['frozen']:7d} {rss:7.1f}")

    # ---- demolition stops. Does it come back to baseline? ---------------
    if verbose:
        print(f"[soak] demolition done; settling for "
              f"{settle_frames / 60.0:.0f} s of simulated quiet ...")
    for _ in range(settle_frames):
        app.step_frame(FRAME_DT)
        live = app.debris.live_count
        if live > result.cap:
            result.cap_breaches.append((app.physics.step_count, live))

    snap = app.debris.snapshot()
    result.final_live = snap["live"]
    result.final_active = app.debris.active_count()
    result.final_total_bodies = app.debris.total_bodies
    result.total_spawned = app.debris.total_spawned
    result.total_despawned = app.debris.total_despawned
    result.total_frozen = app.debris.total_frozen
    result.total_evicted = app.debris.total_evicted
    result.physics_steps = app.physics.step_count
    result.sim_seconds = app.physics.sim_time
    result.states_reclaimed = getattr(app.physics, "states_reclaimed", 0)
    result.rss_settled_mb = rss_mb()
    result.rss_trace_mb.append(result.rss_settled_mb)

    # ---- and does everything actually leave, registries included? -------
    app.debris.clear()
    result.live_bodies_after_clear = (
        app.debris.live_count + app.debris.frozen_count
    )
    result.tracked_records_after_clear = len(app.debris.all_bodies)
    result.rss_end_mb = rss_mb()
    result.wall_seconds = time.perf_counter() - started

    app.physics.step_fixed = real_step_fixed
    app.destroy()
    return result


def report(result: SoakResult) -> str:
    r = result
    lines = [
        "",
        "================ Tumble sustained-demolition soak ================",
        f"structures destroyed     : {r.structures_destroyed}"
        f" (of {r.structures_placed} placed)",
        f"debris bodies spawned    : {r.total_spawned}"
        f"  = {r.cap_overflows:.1f}x the cap",
        f"  despawned / evicted    : {r.total_despawned} / {r.total_evicted}",
        f"  frozen (settled)       : {r.total_frozen}",
        "",
        f"live debris cap          : {r.cap}",
        f"  peak live              : {r.peak_live}"
        f"   {'OK' if r.peak_live <= r.cap else 'OVER CAP'}",
        f"  final live             : {r.final_live}",
        f"  cap breaches (sampled) : {len(r.cap_breaches)}",
        f"  peak bodies (live+froz): {r.peak_total_bodies}",
        f"  final bodies           : {r.final_total_bodies}",
        "",
        f"active (solved) bodies   : peak {r.peak_active}"
        f" -> final {r.final_active}  (baseline allowed {ACTIVE_BASELINE_MAX})",
        "",
        f"physics steps timed      : {r.physics_steps}"
        f"  ({r.sim_seconds:.1f} s simulated in {r.wall_seconds:.1f} s wall)",
        f"  step budget            : {STEP_BUDGET_MS} ms"
        f"  (1/120 s step, 2 steps per 60 fps frame)",
        f"  mean / max             : {r.step_mean_ms:.4f} ms / "
        f"{r.step_max_ms:.4f} ms",
        f"  early mean -> late mean: {r.early_mean_ms:.4f} -> "
        f"{r.late_mean_ms:.4f} ms  ({r.degradation:.2f}x, "
        f"allowed {STEP_DEGRADATION_FACTOR}x)",
        f"  early p95  -> late p95 : {r.early_p95_ms:.4f} -> "
        f"{r.late_p95_ms:.4f} ms  ({r.p95_degradation:.2f}x, "
        f"allowed {STEP_P95_DEGRADATION_FACTOR}x)",
        f"  real-time headroom     : "
        f"{STEP_BUDGET_MS / r.step_mean_ms if r.step_mean_ms else 0:.2f}x",
        "",
        f"memory (RSS)             : start {r.rss_start_mb:.1f} MB"
        f" -> peak {r.rss_peak_mb:.1f} MB"
        f" -> settled {r.rss_settled_mb:.1f} MB",
        f"  growth                 : {r.rss_growth_mb:+.1f} MB"
        f"  (allowed +{RSS_GROWTH_MAX_MB} MB)",
        f"  interned states reaped : {r.states_reclaimed}",
        f"  spent chunk descriptors released : {r.chunks_released}",
        f"per-cycle live trend     : {r.live_trend:.2f}x early-to-late"
        f"  (allowed {LIVE_TREND_FACTOR}x)",
        f"after field.clear()      : {r.live_bodies_after_clear} bodies in "
        f"world, {r.tracked_records_after_clear} records tracked",
        "",
    ]
    failures = r.failures()
    if failures:
        lines.append(f"VERDICT: FAIL - {len(failures)} bound(s) violated")
        lines += [f"  * {f}" for f in failures]
    else:
        lines.append("VERDICT: PASS - every sustainability bound held")
    lines.append("==================================================================")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Tumble demolition soak test")
    p.add_argument("--structures", type=int, default=40,
                   help="how many structures to demolish back to back")
    p.add_argument("--frames-per-structure", type=int,
                   default=FRAMES_PER_STRUCTURE)
    p.add_argument("--settle-seconds", type=float, default=45.0,
                   help="simulated quiet after the last demolition")
    p.add_argument("--ci", action="store_true",
                   help="short mode: 12 structures, still overflows the cap "
                        "several times over (runs in ~1 min)")
    p.add_argument("--seed", type=int, default=40000)
    p.add_argument("--json", metavar="PATH",
                   help="also write the measurements as JSON")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    count = 12 if args.ci else args.structures
    settle = int((15.0 if args.ci else args.settle_seconds) * 60)

    result = run_soak(
        structures_count=count,
        frames_per_structure=args.frames_per_structure,
        settle_frames=settle,
        seed_base=args.seed,
        verbose=not args.quiet,
    )
    print(report(result))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(result.to_dict(), fh, indent=2)
        print(f"[soak] measurements written to {args.json}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
