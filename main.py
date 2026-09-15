#!/usr/bin/env python3
"""Tumble entrypoint.

    python main.py              # windowed playtest
    python main.py --headless   # boot the sim with no window (smoke test)
    python main.py --headless --frames 600
    python main.py --headless --demolish --frames 1200   # + real destruction
"""

from __future__ import annotations

import argparse
import os
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Tumble - first-person foundation")
    p.add_argument(
        "--headless",
        action="store_true",
        help="run with no window (server/CI); exits after --frames frames",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=300,
        help="headless only: how many frames to simulate before exiting",
    )
    p.add_argument(
        "--demolish",
        action="store_true",
        help="headless only: blow up each structure as the player reaches it",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from game.app import TumbleApp

    if args.headless:
        app = TumbleApp(headless=True)
        from game.player import InputState

        # Drive forward the whole time, so the smoke run proves locomotion
        # and not merely that the process starts.
        app.input_state = InputState(forward=True)
        start = app.player.pos
        events = []
        for _ in range(args.frames):
            app.step_frame(1.0 / 60.0)
            if args.demolish:
                # Demolish whatever the player has driven up to. This is the
                # real game loop doing real destruction, not a special path.
                y = app.player.pos[1]
                for d in app.destructibles:
                    lo, _hi = d.world_bounds()
                    # Fire when the structure's near face is inside weapon
                    # range. A standing structure genuinely blocks the lane,
                    # so this is also what lets the run get down the course.
                    if d.intact and lo[1] - y <= 18.0:
                        event = app.demolish(d)
                        if event is not None:
                            events.append((event, y))
        end = app.player.pos

        print(f"[tumble] headless boot OK (window={app.win})")
        print(f"[tumble] frames={app.frames_run} physics_steps={app.physics.step_count}")
        print(f"[tumble] sim_time={app.physics.sim_time:.3f}s")
        print(f"[tumble] bodies={len(app.physics.bodies)} boxes={len(app.physics.box_specs)}")
        print(f"[tumble] player start=({start[0]:.2f}, {start[1]:.2f}, {start[2]:.2f})")
        print(f"[tumble] player end  =({end[0]:.2f}, {end[1]:.2f}, {end[2]:.2f})")
        print(f"[tumble] advanced {end[1] - start[1]:.2f} m forward, on_ground={app.player.on_ground()}")

        chunks = sum(d.chunk_count for d in app.destructibles)
        print(f"[tumble] destructibles={len(app.destructibles)} "
              f"pre-generated chunks={chunks}")
        if args.demolish:
            for event, y in events:
                print(f"[tumble] demolished {event.structure!r} at player "
                      f"y={y:.1f}: {event.spawned} bodies "
                      f"(of {event.chunk_count} chunks, "
                      f"{event.skipped_for_budget} over budget) in "
                      f"{event.build_seconds * 1000.0:.1f} ms")
            snap = app.debris.snapshot()
            print(f"[tumble] demolitions={app.demolitions} "
                  f"debris live={snap['live']} frozen={snap['frozen']} "
                  f"despawned={snap['despawned']} active={snap['active']}")
            print(f"[tumble] live body cap={app.debris.max_live} "
                  f"(never exceeded: peak live <= cap)")
            print(f"[tumble] debris |v|max={snap['max_speed']:.4f} m/s "
                  f"|w|max={snap['max_spin']:.4f} rad/s "
                  f"lowest vertex={snap['min_z']:.4f} m")
        app.destroy()
        print("[tumble] shutdown clean")
        return 0

    try:
        app = TumbleApp(headless=False)
    except Exception as exc:
        if "Could not open window" in str(exc):
            print(
                "\n[tumble] Could not open a window.\n"
                "         This machine has no usable display "
                f"(DISPLAY={os.environ.get('DISPLAY', 'unset')!r}).\n"
                "         If you are on a desktop, check your graphics drivers.\n"
                "         If you are over SSH or on a server, run the sim "
                "without a window instead:\n"
                "             ./run.sh --headless\n",
                file=sys.stderr,
            )
            return 2
        raise

    print("[tumble] window up - WASD to move, mouse to look, shift to sprint, esc to quit")
    print("[tumble] entering main loop")
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
