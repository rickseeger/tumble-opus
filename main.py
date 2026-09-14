#!/usr/bin/env python3
"""Tumble entrypoint.

    python main.py              # windowed playtest
    python main.py --headless   # boot the sim with no window (smoke test)
    python main.py --headless --frames 600
"""

from __future__ import annotations

import argparse
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
        for _ in range(args.frames):
            app.step_frame(1.0 / 60.0)
        end = app.player.pos

        print(f"[tumble] headless boot OK (window={app.win})")
        print(f"[tumble] frames={app.frames_run} physics_steps={app.physics.step_count}")
        print(f"[tumble] sim_time={app.physics.sim_time:.3f}s")
        print(f"[tumble] bodies={len(app.physics.bodies)} boxes={len(app.physics.box_specs)}")
        print(f"[tumble] player start=({start[0]:.2f}, {start[1]:.2f}, {start[2]:.2f})")
        print(f"[tumble] player end  =({end[0]:.2f}, {end[1]:.2f}, {end[2]:.2f})")
        print(f"[tumble] advanced {end[1] - start[1]:.2f} m forward, on_ground={app.player.on_ground()}")
        app.destroy()
        print("[tumble] shutdown clean")
        return 0

    app = TumbleApp(headless=False)
    print("[tumble] window up - WASD to move, mouse to look, shift to sprint, esc to quit")
    print("[tumble] entering main loop")
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
