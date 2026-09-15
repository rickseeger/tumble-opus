#!/usr/bin/env python3
"""Headless shatter demo: blow up a structure and print the debris state.

No window, no GL, no vision required - it prints what the simulation is
actually doing, phase by phase, so the behaviour can be inspected on a server.

    .venv/bin/python tools_debris_demo.py
    .venv/bin/python tools_debris_demo.py --structure arch --seed 3
    .venv/bin/python tools_debris_demo.py --structure all --seconds 20
    .venv/bin/python tools_debris_demo.py --benchmark
"""

from __future__ import annotations

import argparse
import sys
import time

from game import config

config.bootstrap_display(headless=True)

from game import fracture                      # noqa: E402
from game.debris import DebrisField            # noqa: E402
from game.physics import PhysicsWorld          # noqa: E402

#: Where the report takes a reading, in seconds of simulated time.
#: Where the report takes a reading, in seconds of simulated time. The last
#: two are far enough out to be real: measured on this stack, the heaviest
#: structure (a 380-chunk / 8589 t tower, capped to 260 bodies) needs ~25 s to
#: come fully to rest under pure Bullet, so a run that stopped at 15 s would
#: honestly report "not at rest" and look like a failure when it is merely
#: unfinished.
PHASES = (
    ("launch", 0.0),
    ("airborne", 0.25),
    ("first-impacts", 1.0),
    ("tumbling", 3.0),
    ("rolling", 6.0),
    ("settling", 10.0),
    ("quiet", 18.0),
    ("at-rest", 30.0),
)

HEADER = (
    f"{'phase':>14} {'t(s)':>6} {'live':>5} {'frozen':>7} {'active':>7} "
    f"{'KE(J)':>13} {'|v|max':>8} {'|w|max':>8} {'minZ':>8}"
)


def _row(label: str, t: float, snap: dict) -> str:
    return (
        f"{label:>14} {t:6.2f} {snap['live']:5d} {snap['frozen']:7d} "
        f"{snap['active']:7d} {snap['kinetic_energy']:13.1f} "
        f"{snap['max_speed']:8.3f} {snap['max_spin']:8.3f} {snap['min_z']:8.3f}"
    )


def run_structure(name: str, seed: int, seconds: float, verbose: bool = True) -> dict:
    """Shatter one structure and walk it through every phase."""
    spec = fracture.ARCHETYPES[name]()
    generated = fracture.fracture(spec, seed=seed)

    physics = PhysicsWorld()
    physics.add_ground_plane(0.0)
    field = DebrisField(physics, seed=seed)

    lo, hi = generated.bounds
    impact = (0.0, 0.0, lo[2] + (hi[2] - lo[2]) * 0.22)

    if verbose:
        print(f"\n=== {spec.name} (seed {seed}) ===")
        print(f"pre-generated chunks : {len(generated.chunks)}")
        print(f"source volume        : {generated.source_volume:.1f} m^3")
        print(f"impact point         : ({impact[0]:.1f}, {impact[1]:.1f}, "
              f"{impact[2]:.1f})")

    event = field.shatter(generated, impact_point=impact,
                          impulse=config.DEBRIS_IMPULSE)

    if verbose:
        print(f"spawned bodies       : {event.spawned}"
              f"  (cap {field.max_live}, skipped {event.skipped_for_budget})")
        print(f"debris total mass    : {event.total_mass / 1000.0:.1f} t")
        print(f"shatter build time   : {event.build_seconds * 1000.0:.2f} ms")
        masses = sorted(b.base_mass for b in event.bodies)
        print(f"chunk mass range     : {masses[0]:.1f} kg .. {masses[-1]:.1f} kg "
              f"(x{masses[-1] / max(masses[0], 1e-9):.0f})")
        print()
        print(HEADER)

    dt = physics.fixed_dt
    total_steps = int(seconds / dt)
    phases = [(lbl, t) for lbl, t in PHASES if t <= seconds]
    next_phase = 0
    worst_z = float("inf")
    settled_at = None
    sim_started = time.perf_counter()

    if phases and phases[0][1] == 0.0:
        if verbose:
            print(_row(phases[0][0], 0.0, field.snapshot()))
        next_phase = 1

    for step in range(total_steps):
        physics.step_fixed(1)
        field.update(dt, player_y=None)
        t = (step + 1) * dt
        if settled_at is None and not any(
            b.node.isActive() for b in field.live
        ):
            settled_at = t
        while next_phase < len(phases) and t >= phases[next_phase][1]:
            snap = field.snapshot()
            worst_z = min(worst_z, snap["min_z"])
            if verbose:
                print(_row(phases[next_phase][0], t, snap))
            next_phase += 1

    wall = time.perf_counter() - sim_started
    final = field.snapshot()
    bodies = field.live + field.frozen
    all_asleep = all(b.is_asleep() for b in bodies)

    if verbose:
        print()
        print(f"simulated {seconds:.1f} s in {wall:.2f} s wall clock "
              f"-> {seconds / max(wall, 1e-9):.2f}x real time "
              f"({wall / max(total_steps, 1) * 1000.0:.3f} ms/step)")
        if settled_at is None:
            print(f"every body at rest   : False - STILL MOVING after "
                  f"{seconds:.1f} s (try --seconds 30; the heaviest tower "
                  f"needs ~25 s)")
        else:
            print(f"every body at rest   : {all_asleep} "
                  f"(came to rest at t={settled_at:.2f} s)")
        print(f"final live/frozen    : {final['live']} / {final['frozen']}")
        print(f"lowest debris vertex : {final['min_z']:.4f} m "
              f"(ground is 0.0 - negative means penetration)")

    return {
        "structure": spec.name,
        "chunks": len(generated.chunks),
        "spawned": event.spawned,
        "seconds": seconds,
        "wall": wall,
        "realtime_factor": seconds / max(wall, 1e-9),
        "ms_per_step": wall / max(total_steps, 1) * 1000.0,
        "all_asleep": all_asleep,
        "settled_at": settled_at,
        "final": final,
    }


def benchmark(seed: int = 7, seconds: float = 10.0) -> dict:
    """Step a full structure's worth of debris and report the throughput."""
    print("=== stepping benchmark: a full tower's debris ===")
    spec = fracture.tower_spec()
    generated = fracture.fracture(spec, seed=seed)
    physics = PhysicsWorld()
    physics.add_ground_plane(0.0)
    field = DebrisField(physics, seed=seed)
    event = field.shatter(generated, impact_point=(0.0, 0.0, 6.0),
                          impulse=config.DEBRIS_IMPULSE)

    dt = physics.fixed_dt
    steps = int(seconds / dt)
    started = time.perf_counter()
    for _ in range(steps):
        physics.step_fixed(1)
        field.update(dt, player_y=None)
    wall = time.perf_counter() - started

    budget_ms = dt * 1000.0
    ms = wall / steps * 1000.0
    print(f"bodies spawned       : {event.spawned}")
    print(f"steps                : {steps} at dt={dt * 1000.0:.3f} ms "
          f"({seconds:.1f} s simulated)")
    print(f"wall clock           : {wall:.3f} s")
    print(f"per step             : {ms:.4f} ms  (budget {budget_ms:.3f} ms)")
    print(f"real-time factor     : {seconds / wall:.2f}x")
    print(f"headroom             : {budget_ms / ms:.2f}x the step budget")
    return {"spawned": event.spawned, "steps": steps, "wall": wall,
            "ms_per_step": ms, "realtime_factor": seconds / wall,
            "budget_ms": budget_ms}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Tumble headless debris demo")
    p.add_argument("--structure", default="tower",
                   choices=sorted(fracture.ARCHETYPES) + ["all"])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--seconds", type=float, default=30.0,
                   help="simulated seconds; the heaviest tower needs ~25 s "
                        "to come fully to rest")
    p.add_argument("--benchmark", action="store_true",
                   help="run the stepping benchmark instead of the demo")
    args = p.parse_args(argv)

    if args.benchmark:
        benchmark(seed=args.seed)
        return 0

    names = sorted(fracture.ARCHETYPES) if args.structure == "all" \
        else [args.structure]
    for name in names:
        run_structure(name, args.seed, args.seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
