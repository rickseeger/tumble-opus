#!/usr/bin/env python3
"""Diagnostic probe: WHERE does a Tumble soak's resident memory go?

Node 20's sustainability analysis found live debris pinned flat at the cap and
frame time flat, but RSS climbing linearly at ~9.8 MB per 1000 frames with no
deceleration.  That is a leak, and "RSS grew" is not a diagnosis.  This script
drives the same real soak loop as ``tools/soak_driver.py`` and, at intervals,
attributes the growth to a specific holder:

  * ``transform_states`` / ``render_states`` - Panda's interned state tables
    (``game.physics.reclaim_interned_states`` is supposed to bound the first)
  * ``gc_objects`` + per-type deltas       - Python-side retention
  * ``rss_after_trim``                     - RSS after ``malloc_trim(0)``; if
    the gap to ``rss`` is large, the growth is glibc arena fragmentation
    (freed but unreturned), not live allocation
  * debris field totals                    - live/frozen/spawned, for context

Writes a JSONL of probe records.  This is instrumentation, like the driver:
it measures and reports, it asserts nothing.
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import gc
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import soak_driver                                            # noqa: E402


def _libc():
    try:
        return ctypes.CDLL("libc.so.6")
    except OSError:                                           # pragma: no cover
        return None


def type_histogram(top: int = 25):
    c = collections.Counter(type(o).__name__ for o in gc.get_objects())
    return dict(c.most_common(top))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", type=int, default=12000)
    p.add_argument("--max-seconds", type=float, default=300.0)
    p.add_argument("--destroy-every", type=int, default=120)
    p.add_argument("--probe-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=90210)
    p.add_argument("--out", default="soak_runs/diagnose/probe.jsonl")
    a = p.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    fh = open(a.out, "w")
    libc = _libc()

    from panda3d.core import TransformState, RenderState

    state = {"app": None, "n": 0, "t0": time.perf_counter()}

    def factory():
        from game.app import TumbleApp
        app = TumbleApp(headless=True)
        state["app"] = app
        real_step = app.step_frame

        def probed_step(dt):
            taken = real_step(dt)
            state["n"] += 1
            if state["n"] % a.probe_every == 0:
                rss = soak_driver.rss_mb()
                if libc is not None:
                    libc.malloc_trim(0)
                rss_trim = soak_driver.rss_mb()
                gc.collect()
                rec = {
                    "frame": state["n"],
                    "elapsed_s": round(time.perf_counter() - state["t0"], 3),
                    "rss_mb": round(rss, 2),
                    "rss_after_trim_mb": round(rss_trim, 2),
                    "trim_returned_mb": round(rss - rss_trim, 2),
                    "transform_states": int(TransformState.getNumStates()),
                    "render_states": int(RenderState.getNumStates()),
                    "gc_objects": len(gc.get_objects()),
                    "states_reclaimed": int(
                        getattr(app.physics, "states_reclaimed", 0)),
                    "live": app.debris.live_count,
                    "frozen": app.debris.frozen_count,
                    "total_spawned": app.debris.total_spawned,
                    "total_despawned": app.debris.total_despawned,
                    "total_bodies": app.debris.total_bodies,
                    "types": type_histogram(),
                }
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(f"[probe] f{rec['frame']:6d} rss {rec['rss_mb']:7.1f} "
                      f"(trim -> {rec['rss_after_trim_mb']:7.1f}, returned "
                      f"{rec['trim_returned_mb']:6.1f})  TS "
                      f"{rec['transform_states']:8d}  RS "
                      f"{rec['render_states']:6d}  gcobj "
                      f"{rec['gc_objects']:8d}  spawned "
                      f"{rec['total_spawned']:6d}", flush=True)
            return taken

        app.step_frame = probed_step
        return app

    control = soak_driver.RunControl(max_seconds=a.max_seconds,
                                     max_frames=a.frames)
    soak_driver.run(frames=a.frames, destroy_every=a.destroy_every,
                    seed=a.seed, writer=None, progress_every=0,
                    control=control, app_factory=factory)
    fh.close()
    print(f"[probe] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
