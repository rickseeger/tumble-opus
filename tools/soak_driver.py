#!/usr/bin/env python3
"""Headless soak *driver* for Tumble: instrumentation only.

What this is
------------
A measuring instrument, not a test. It drives the REAL integrated game -
:class:`game.app.TumbleApp` in headless mode - through sustained demolition
and writes one row of numbers per frame to a machine-readable file. It makes
no judgements: there are no pass/fail assertions here and no tuning of the
debris budget. (The *judging* soak with its bounds is `tools_soak.py`; this
driver is the raw data feed that an analysis node can point at.)

Everything it drives is the shipping path:

* ``TumbleApp(headless=True)`` - ``window-type none``, no graphics pipe, no
  audio, so it runs over SSH on a display-less server.
* ``app.step_frame(dt)`` - the real host frame: input, damage resolution,
  fixed-step physics, then ``DebrisField.update`` (the node-15 budgeting).
* ``app.strike(point)`` - the real gameplay destruction path. Damage
  accumulates through :class:`game.damage.DamageSystem`, crosses the
  structure's integrity threshold, and destruction is the *consequence*; the
  driver never calls ``demolish()`` behind damage's back.
* ``game.structures.generate`` / ``attach_proxies`` / ``DamageSystem.register``
  - structures are authored onto the course exactly the way the shipping
  course is authored.

No game module was modified to make this work, and no accessor was added:
every count recorded below was already public on
:class:`game.debris.DebrisField` (``live_count``, ``active_count``,
``stepped_count``, ``frozen_count``, ``total_spawned``, ``total_frozen``,
``total_despawned``, ``total_evicted``, ``snapshot``) or on
:class:`game.physics.PhysicsWorld` (``step_count``, ``sim_time``).

The one instrumentation seam
---------------------------
``PhysicsWorld.step_fixed`` is wrapped with a timer that calls straight
through to the real method and records elapsed milliseconds. It adds two
``perf_counter`` reads per substep and changes nothing about the simulation.

Usage
-----
    ./run.sh --soak-driver                       # short default, ~20 s
    .venv/bin/python tools/soak_driver.py --frames 3600 --destroy-every 120
    .venv/bin/python tools/soak_driver.py --seconds 60 --format jsonl \
        --out /tmp/soak.jsonl

Flags that matter: ``--frames`` / ``--seconds`` (run length),
``--destroy-every`` (destruction schedule: author + strike one structure
every N frames), ``--seed``, ``--out``, ``--format``. ``--help`` lists them
all.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional

# Importable as a script from anywhere: the repo root holds the `game` package.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from game import config, structures  # noqa: E402

#: Host frame delta the driver feeds `step_frame`. Fixed, so the run is a
#: function of the frame count and not of how fast this machine happens to be.
FRAME_DT = 1.0 / 60.0

#: Archetypes cycled through by the destruction schedule.
ARCHETYPES = ("cluster", "arch", "slab", "tower")
#: Where the first authored structure stands, and the gap between them.
FIRST_Y = 30.0
SPACING = 26.0
#: How far short of a structure the player is stood when it is struck. Inside
#: the eviction protection radius on purpose: that is the interesting case for
#: the budget, and the driver should observe the game under it, not around it.
PLAYER_STANDOFF = 14.0

#: Columns written to the CSV, in order. JSONL uses the same keys.
COLUMNS = (
    "frame",
    "ts_unix",
    "elapsed_s",
    "sim_time_s",
    "frame_ms",
    "physics_ms",
    "substeps",
    "live",
    "active",
    "stepped",
    "frozen",
    "total_spawned",
    "total_frozen",
    "total_despawned",
    "total_evicted",
    "retired_total",
    "rss_mb",
    "structures_placed",
    "structures_destroyed",
    "struck_this_frame",
    "spawned_this_event",
    "skipped_for_budget",
)


# ----------------------------------------------------------------- utilities
def rss_mb() -> float:
    """Process resident set size in MB, straight from /proc.

    No psutil dependency: this is Linux-only software by design.
    """
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:                                     # pragma: no cover
        pass
    return 0.0


def percentile(values: List[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 1]. 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ------------------------------------------------------------------- writers
class MetricsWriter:
    """Streams samples to CSV or JSON Lines, flushing as it goes.

    Streaming rather than buffering: a soak that is killed halfway through
    should still leave behind every frame it managed to measure.
    """

    def __init__(self, path: str, fmt: str) -> None:
        self.path = path
        self.fmt = fmt
        self.rows = 0
        self._fh = open(path, "w", newline="")
        self._csv = None
        if fmt == "csv":
            self._csv = csv.DictWriter(self._fh, fieldnames=list(COLUMNS))
            self._csv.writeheader()

    def write(self, sample: Dict) -> None:
        if self._csv is not None:
            self._csv.writerow({k: sample.get(k, "") for k in COLUMNS})
        else:
            self._fh.write(json.dumps(sample) + "\n")
        self.rows += 1

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()


# -------------------------------------------------------------------- driver
def place_structure(app, index: int, seed_base: int):
    """Author one more destructible onto the course, the shipping way."""
    archetype = ARCHETYPES[index % len(ARCHETYPES)]
    x = -8.0 if index % 2 else 8.0
    y = FIRST_Y + index * SPACING
    d = structures.generate(
        archetype, (x, y, 0.0), seed_base + index,
        name=f"soakdrv_{index:03d}_{archetype}",
    )
    structures.attach_proxies(app.physics, d)   # real static collision proxies
    app.destructibles.append(d)
    app.damage.register(d)                      # real integrity threshold
    return d


def run(
    frames: int,
    destroy_every: int,
    seed: int,
    writer: Optional[MetricsWriter],
    max_structures: Optional[int] = None,
    release_chunks: bool = False,
    progress_every: int = 0,
) -> Dict:
    """Step the real headless game for *frames* frames, measuring every one.

    Returns a summary dict. Writes nothing to stdout unless *progress_every*
    is positive; the caller owns the reporting.
    """
    from game.app import TumbleApp

    app = TumbleApp(headless=True)

    # Seed the debris field's own RNG so launch velocities and spins are
    # reproducible for a given --seed. This is an assignment from outside, not
    # a change to game code: DebrisField.rng/.seed are plain public attributes
    # and the field reads them exactly as it did before.
    app.debris.rng = random.Random(seed)
    app.debris.seed = seed

    # --- the one instrumentation seam: time the real step_fixed -----------
    per_frame_physics_ms = [0.0]
    real_step_fixed = app.physics.step_fixed

    def timed_step_fixed(steps: int = 1) -> int:
        t0 = time.perf_counter()
        taken = real_step_fixed(steps)
        per_frame_physics_ms[0] += (time.perf_counter() - t0) * 1000.0
        return taken

    app.physics.step_fixed = timed_step_fixed

    frame_ms: List[float] = []
    physics_ms: List[float] = []
    rss_trace: List[float] = []
    destroy_frames: List[int] = []

    placed = 0
    destroyed = 0
    pending = None            # structure authored and not yet brought down
    peak_live = 0
    peak_stepped = 0
    peak_frozen = 0
    peak_total_bodies = 0
    chunks_released = 0

    rss0 = rss_mb()
    started = time.perf_counter()

    for frame in range(1, frames + 1):
        struck = 0
        spawned_this_event = 0
        skipped_for_budget = 0

        # ---- destruction schedule ---------------------------------------
        # Every N frames: author a fresh structure, stand the player in range,
        # and start hitting it. Damage does the rest.
        if destroy_every > 0 and (frame - 1) % destroy_every == 0:
            if max_structures is None or placed < max_structures:
                pending = place_structure(app, placed, seed)
                placed += 1
                app.player.np.setPos(
                    0.0,
                    pending.world_bounds()[0][1] - PLAYER_STANDOFF,
                    config.SPAWN_POS[2],
                )

        events_before = len(app.debris.events)
        spawned_before = app.debris.total_spawned

        per_frame_physics_ms[0] = 0.0
        t0 = time.perf_counter()
        substeps = app.step_frame(FRAME_DT)      # the REAL frame function
        # Strike after the frame's own `damage.resolve()`, which is exactly
        # where the windowed game's debug key sits in the frame. `app.strike`
        # -> `DamageSystem.apply_damage` defaults to resolve=True, so a blow
        # that crosses the integrity threshold demolishes inside this call
        # (damage -> threshold -> resolve -> app.demolish -> field.shatter).
        # The driver never calls demolish() itself: destruction is always the
        # consequence of damage, which is why `structures_destroyed` can rise
        # on the very first frame a structure is struck.
        if pending is not None and pending.intact:
            app.strike(pending.default_impact_point())
            struck = 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if pending is not None and not pending.intact:
            destroyed += 1
            destroy_frames.append(frame)
            if release_chunks:
                chunks_released += pending.release_chunks()
            pending = None

        if len(app.debris.events) > events_before or \
                app.debris.total_spawned > spawned_before:
            ev = app.debris.events[-1] if app.debris.events else None
            if ev is not None:
                spawned_this_event = ev.spawned
                skipped_for_budget = ev.skipped_for_budget

        live = app.debris.live_count
        stepped = app.debris.stepped_count()
        frozen = app.debris.frozen_count
        rss = rss_mb()

        peak_live = max(peak_live, live)
        peak_stepped = max(peak_stepped, stepped)
        peak_frozen = max(peak_frozen, frozen)
        peak_total_bodies = max(peak_total_bodies, app.debris.total_bodies)

        frame_ms.append(elapsed_ms)
        physics_ms.append(per_frame_physics_ms[0])
        rss_trace.append(rss)

        sample = {
            "frame": frame,
            "ts_unix": round(time.time(), 3),
            "elapsed_s": round(time.perf_counter() - started, 4),
            "sim_time_s": round(app.physics.sim_time, 4),
            "frame_ms": round(elapsed_ms, 4),
            "physics_ms": round(per_frame_physics_ms[0], 4),
            "substeps": substeps,
            "live": live,
            "active": app.debris.active_count(),
            "stepped": stepped,
            "frozen": frozen,
            "total_spawned": app.debris.total_spawned,
            "total_frozen": app.debris.total_frozen,
            "total_despawned": app.debris.total_despawned,
            "total_evicted": app.debris.total_evicted,
            # What the budget has retired, all mechanisms together: frozen out
            # of the solver plus removed from the world entirely.
            "retired_total": (app.debris.total_frozen
                              + app.debris.total_despawned),
            "rss_mb": round(rss, 2),
            "structures_placed": placed,
            "structures_destroyed": destroyed,
            "struck_this_frame": struck,
            "spawned_this_event": spawned_this_event,
            "skipped_for_budget": skipped_for_budget,
        }
        if writer is not None:
            writer.write(sample)

        if progress_every and frame % progress_every == 0:
            print(f"[driver] frame {frame:6d}/{frames}  "
                  f"sim {app.physics.sim_time:7.2f}s  "
                  f"live {live:4d}  frozen {frozen:4d}  "
                  f"spawned {app.debris.total_spawned:6d}  "
                  f"retired {sample['retired_total']:6d}  "
                  f"rss {rss:7.1f} MB  "
                  f"frame {elapsed_ms:6.2f} ms",
                  flush=True)

    wall = time.perf_counter() - started
    snap = app.debris.snapshot()

    return {
        "frames": frames,
        "wall_seconds": round(wall, 3),
        "sim_seconds": round(app.physics.sim_time, 3),
        "physics_steps": app.physics.step_count,
        "frame_dt": FRAME_DT,
        "destroy_every": destroy_every,
        "seed": seed,
        "cap": app.debris.max_live,
        "structures_placed": placed,
        "structures_destroyed": destroyed,
        "chunks_released": chunks_released,
        "destroy_frames": destroy_frames,
        "frame_ms_mean": round(mean(frame_ms), 4),
        "frame_ms_p50": round(percentile(frame_ms, 0.50), 4),
        "frame_ms_p95": round(percentile(frame_ms, 0.95), 4),
        "frame_ms_max": round(max(frame_ms) if frame_ms else 0.0, 4),
        "physics_ms_mean": round(mean(physics_ms), 4),
        "physics_ms_p95": round(percentile(physics_ms, 0.95), 4),
        "physics_ms_max": round(max(physics_ms) if physics_ms else 0.0, 4),
        "live_end": snap["live"],
        "live_peak": peak_live,
        "stepped_peak": peak_stepped,
        "frozen_end": snap["frozen"],
        "frozen_peak": peak_frozen,
        "total_bodies_peak": peak_total_bodies,
        "total_spawned": app.debris.total_spawned,
        "total_frozen": app.debris.total_frozen,
        "total_despawned": app.debris.total_despawned,
        "total_evicted": app.debris.total_evicted,
        "retired_total": app.debris.total_frozen + app.debris.total_despawned,
        "states_reclaimed": getattr(app.physics, "states_reclaimed", 0),
        "rss_start_mb": round(rss0, 2),
        "rss_peak_mb": round(max(rss_trace) if rss_trace else rss0, 2),
        "rss_end_mb": round(rss_trace[-1] if rss_trace else rss0, 2),
        "rss_growth_mb": round((rss_trace[-1] if rss_trace else rss0) - rss0, 2),
    }


def report(s: Dict, out_path: Optional[str], rows: int) -> str:
    """The compact stdout summary. Description only - it judges nothing."""
    sim_rate = (s["sim_seconds"] / s["wall_seconds"]
                if s["wall_seconds"] else 0.0)
    lines = [
        "[driver] ---- tumble headless soak driver: measurements ----",
        f"  run              : {s['frames']} frames @ dt {s['frame_dt']:.5f} s"
        f"  -> {s['sim_seconds']:.2f} s simulated"
        f" in {s['wall_seconds']:.2f} s wall ({sim_rate:.2f}x realtime)",
        f"  physics steps    : {s['physics_steps']}"
        f"   (fixed {config.FIXED_DT:.6f} s)",
        f"  destruction      : every {s['destroy_every']} frames"
        f"  -> {s['structures_placed']} structures placed,"
        f" {s['structures_destroyed']} destroyed",
        f"  frame time (ms)  : mean {s['frame_ms_mean']:.3f}"
        f"   p50 {s['frame_ms_p50']:.3f}"
        f"   p95 {s['frame_ms_p95']:.3f}"
        f"   max {s['frame_ms_max']:.3f}",
        f"  of which physics : mean {s['physics_ms_mean']:.3f}"
        f"   p95 {s['physics_ms_p95']:.3f}"
        f"   max {s['physics_ms_max']:.3f}",
        f"  debris live      : end {s['live_end']}"
        f"   peak {s['live_peak']}"
        f"   (cap {s['cap']}, peak stepped {s['stepped_peak']})",
        f"  debris frozen    : end {s['frozen_end']}   peak {s['frozen_peak']}",
        f"  bodies spawned   : {s['total_spawned']} total"
        f"   (peak resident {s['total_bodies_peak']})",
        f"  bodies retired   : {s['retired_total']} total"
        f"   = {s['total_frozen']} frozen + {s['total_despawned']} despawned"
        f"   (of which {s['total_evicted']} evicted for budget)",
        f"  pose cache       : {s['states_reclaimed']} interned states"
        f" reclaimed",
        f"  memory (RSS)     : start {s['rss_start_mb']:.1f} MB"
        f"  -> peak {s['rss_peak_mb']:.1f} MB"
        f"  -> end {s['rss_end_mb']:.1f} MB"
        f"  ({s['rss_growth_mb']:+.1f} MB)",
        f"  seed             : {s['seed']}",
    ]
    if out_path:
        lines.append(f"  metrics          : {rows} rows -> {out_path}")
    lines.append("[driver] no assertions were made: this node measures, it "
                 "does not judge.")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="soak_driver.py",
        description="Headless instrumentation driver for Tumble's real "
                    "destruction path. Measures; asserts nothing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ---- run length ----
    p.add_argument("--frames", type=int, default=1200,
                   help="how many host frames to run (1/60 s each)")
    p.add_argument("--seconds", type=float, default=None,
                   help="run length in SIMULATED seconds; overrides --frames")
    # ---- destruction schedule ----
    p.add_argument("--destroy-every", type=int, default=120, metavar="N",
                   help="author and strike one structure every N frames; "
                        "0 disables destruction entirely")
    p.add_argument("--structures", type=int, default=None, metavar="N",
                   help="stop authoring after N structures (default: no "
                        "limit, the schedule runs for the whole run)")
    # ---- determinism / output ----
    p.add_argument("--seed", type=int, default=90210,
                   help="seeds structure generation and the debris field RNG")
    p.add_argument("--out", default="soak_metrics.csv", metavar="PATH",
                   help="per-frame metrics file ('-' or '' to skip writing)")
    p.add_argument("--format", choices=("csv", "jsonl"), default="csv",
                   help="metrics file format")
    p.add_argument("--summary-json", default=None, metavar="PATH",
                   help="also write the run summary as JSON")
    p.add_argument("--progress-every", type=int, default=120, metavar="N",
                   help="print a progress line every N frames (0 = silent)")
    p.add_argument("--release-chunks", action="store_true",
                   help="call Destructible.release_chunks() on rubble, as the "
                        "long-run soak does. Off by default so the driver "
                        "observes the integrated path untouched.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    frames = args.frames
    if args.seconds is not None:
        frames = max(1, int(round(args.seconds / FRAME_DT)))
    frames = max(1, int(frames))

    out_path = args.out if args.out not in ("", "-") else None
    writer = MetricsWriter(out_path, args.format) if out_path else None

    print(f"[driver] headless Tumble ({config.PRC_HEADLESS.split()[1]}"
          f" window) - {frames} frames, destroying every "
          f"{args.destroy_every} frames, seed {args.seed}", flush=True)
    try:
        summary = run(
            frames=frames,
            destroy_every=args.destroy_every,
            seed=args.seed,
            writer=writer,
            max_structures=args.structures,
            release_chunks=args.release_chunks,
            progress_every=max(0, args.progress_every),
        )
    finally:
        if writer is not None:
            writer.close()

    print(report(summary, out_path, writer.rows if writer else 0))

    if args.summary_json:
        with open(args.summary_json, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"[driver] summary -> {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
