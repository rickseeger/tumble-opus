"""The soak *harness* under test: bounds, watchdog, checkpoints, detach.

Why this file exists
--------------------
Node 18 was dispatched twice and both worker sessions died at their 1800 s
timeout, because each started a long soak in the foreground and sat waiting
on it. The fix is not "be more careful next time" - it is that a soak must be
structurally incapable of blocking anybody: it bounds itself, it checkpoints
as it goes, and it can be detached and read back from a session that never
touched it.

That is what is tested here, and it is tested *fast*. Nothing in this file
runs the real game for more than a couple of hundred frames; the loop-level
tests drive `run()` with a cheap fake app through its `app_factory` seam, and
the watchdog is driven by an injected clock, so escalation that would take
minutes of wall time is exercised in microseconds. The whole file is seconds,
not minutes - a test suite that is slow to run is a test suite that stops
being run.

The real-game path is not left unexercised: `test_real_game_run_is_bounded`
and the CLI smoke tests below do drive `TumbleApp(headless=True)`, just
briefly.
"""

import json
import os
import subprocess
import sys
import threading
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import soak_driver as sd  # noqa: E402

DRIVER = os.path.join(ROOT, "tools", "soak_driver.py")


# ------------------------------------------------------------ a cheap stand-in
class FakeDebris:
    """Enough DebrisField surface for the run loop, and nothing more."""

    def __init__(self):
        self.rng = None
        self.seed = 0
        self.live_count = 7
        self.frozen_count = 2
        self.total_spawned = 0
        self.total_frozen = 0
        self.total_despawned = 0
        self.total_evicted = 0
        self.total_bodies = 9
        self.max_live = 260
        self.events = []

    def active_count(self):
        return 5

    def stepped_count(self):
        return 5

    def snapshot(self):
        return {"live": self.live_count, "frozen": self.frozen_count}


class FakePhysics:
    def __init__(self):
        self.step_count = 0
        self.sim_time = 0.0
        self.states_reclaimed = 0

    def step_fixed(self, steps=1):
        self.step_count += steps
        return steps


class FakeApp:
    """A TumbleApp-shaped object whose frame costs nothing.

    `frame_cost` lets a test make frames arbitrarily slow in wall-clock terms
    without actually sleeping for that long anywhere it matters.
    """

    def __init__(self, frame_cost=0.0):
        self.debris = FakeDebris()
        self.physics = FakePhysics()
        self.destructibles = []
        self.frame_cost = frame_cost
        self.frames = 0

        class _Damage:
            def register(self, d):
                pass

        self.damage = _Damage()

        class _Player:
            class _NP:
                def setPos(self, *a):
                    pass

            np = _NP()

        self.player = _Player()

    def step_frame(self, dt):
        self.frames += 1
        self.physics.sim_time += dt
        self.physics.step_fixed(2)
        if self.frame_cost:
            time.sleep(self.frame_cost)
        return 2

    def strike(self, point):
        pass


def fake_run(**kw):
    """run() over a FakeApp, with destruction switched off (no structures)."""
    kw.setdefault("destroy_every", 0)
    kw.setdefault("seed", 1)
    kw.setdefault("writer", None)
    kw.setdefault("progress_every", 0)
    cost = kw.pop("frame_cost", 0.0)
    kw.setdefault("app_factory", lambda: FakeApp(frame_cost=cost))
    kw.setdefault("frames", 10 ** 9)
    return sd.run(**kw)


# ------------------------------------------------------------------- bounds
def test_max_frames_stops_at_exactly_that_many_frames():
    """--max-frames K means at most K frames stepped. Not K+1."""
    for k in (1, 5, 37, 100):
        control = sd.RunControl(max_frames=k)
        summary = fake_run(control=control)
        assert summary["frames"] == k, "asked for %d, stepped %d" % (
            k, summary["frames"])
        assert summary["stop_reason"] == "max-frames"


def test_max_demolitions_bound_is_honoured():
    """The demolition bound ends the run even with frames left to burn."""
    control = sd.RunControl(max_frames=10 ** 6, max_demolitions=3)
    # Destroy on a schedule with the real-ish path stubbed: the FakeApp's
    # structures come down the frame they are placed, so 3 demolitions is 3
    # schedule ticks.
    calls = {"n": 0}

    class Pending:
        intact = True

        def world_bounds(self):
            return ((0.0, 10.0, 0.0), (1.0, 11.0, 1.0))

        def default_impact_point(self):
            return (0.0, 0.0, 0.0)

        def release_chunks(self):
            return 0

    def place(app, index, seed_base):
        calls["n"] += 1
        p = Pending()
        # comes down one frame later, via the loop's own intact check
        p.intact = False
        return p

    original = sd.place_structure
    sd.place_structure = place
    try:
        summary = fake_run(control=control, destroy_every=5)
    finally:
        sd.place_structure = original
    assert summary["structures_destroyed"] == 3
    assert summary["stop_reason"] == "max-demolitions"
    # And it did not run away burning frames after the bound tripped.
    assert summary["frames"] <= 5 * 3 + 5


def test_max_seconds_stops_a_run_that_would_otherwise_continue():
    """The wall-clock bound ends a nominally unbounded run, promptly."""
    control = sd.RunControl(max_frames=10 ** 9, max_seconds=0.35)
    t0 = time.monotonic()
    summary = fake_run(control=control, frame_cost=0.002)
    elapsed = time.monotonic() - t0
    assert summary["stop_reason"] == "max-seconds"
    assert elapsed < 3.0, "wall bound overshot badly: %.2fs" % elapsed
    assert summary["frames"] > 0


def test_cli_rejects_a_run_with_no_wall_clock_bound():
    """--max-seconds 0 is refused: there is no such thing as an unbounded run."""
    rc = sd.main(["--max-seconds", "0", "--out", "-"])
    assert rc == 2


# ---------------------------------------------------------------- watchdog
def test_watchdog_escalates_from_soft_stop_to_hard_kill():
    """Stage one asks; stage two, after the grace, kills.

    Driven by an injected clock so an escalation that takes 30 real seconds
    is exercised instantly and, crucially, the "hard kill" is a callback the
    test can observe instead of os._exit taking the test runner with it.
    """
    now = [0.0]
    fired = {"soft": False, "hard": False}
    stop = threading.Event()

    def clock():
        return now[0]

    def sleep(_):
        now[0] += 1.0      # each tick advances the fake clock one second

    wd = sd.Watchdog(
        deadline_s=5.0,
        grace_s=3.0,
        tick=0.001,
        stop_event=stop,
        on_soft=lambda: fired.__setitem__("soft", True),
        on_hard=lambda: fired.__setitem__("hard", True),
        clock=clock,
        sleep=sleep,
    )
    wd.start()
    wd.join(timeout=5.0)

    assert fired["soft"], "soft stop never fired"
    assert stop.is_set(), "the run loop was never asked to stop"
    assert fired["hard"], "hard kill never fired after the grace period"
    assert now[0] >= 8.0, "escalated before deadline + grace"


def test_watchdog_that_is_cancelled_never_kills():
    """The clean-exit path cancels the guard; it must then stay silent."""
    now = [0.0]
    fired = {"hard": False}

    def sleep(_):
        now[0] += 1.0

    wd = sd.Watchdog(
        deadline_s=1.0, grace_s=1.0, tick=0.001,
        on_hard=lambda: fired.__setitem__("hard", True),
        clock=lambda: now[0], sleep=sleep,
    )
    wd.cancel()
    wd.start()
    wd.join(timeout=2.0)
    assert not fired["hard"]


def test_watchdog_deadline_is_hard_even_when_the_loop_ignores_the_flag():
    """A wedged frame cannot outlive the deadline: this is the whole point."""
    now = [0.0]
    killed = threading.Event()

    def sleep(_):
        now[0] += 0.5

    wd = sd.Watchdog(
        deadline_s=2.0, grace_s=1.0, tick=0.001,
        on_soft=None,                      # nobody is listening to the flag
        on_hard=killed.set,
        clock=lambda: now[0], sleep=sleep,
    )
    wd.start()
    assert killed.wait(timeout=5.0), "a stuck run was never terminated"


# ------------------------------------------------------------- checkpoints
def test_checkpoint_records_parse_and_carry_every_required_key(tmp_path):
    path = str(tmp_path / "metrics.jsonl")
    cp = sd.Checkpointer(path, every_frames=10, every_seconds=0)
    control = sd.RunControl(max_frames=55, checkpointer=cp)
    fake_run(control=control)
    cp.close()

    records = sd.read_jsonl(path)
    assert len(records) >= 5, "expected several samples, got %d" % len(records)
    for rec in records:
        missing = [k for k in sd.CHECKPOINT_KEYS if k not in rec]
        assert not missing, "record missing %s: %r" % (missing, rec)
    frames = [r["frame"] for r in records]
    assert frames == sorted(frames), "samples are not in frame order"


def test_every_checkpoint_is_flushed_so_a_partial_file_is_readable(tmp_path):
    """Read the file mid-run, from a separate handle: rows must already be there."""
    path = str(tmp_path / "metrics.jsonl")
    cp = sd.Checkpointer(path, every_frames=1, every_seconds=0)
    cp.sample({"record": "sample", "frame": 1}, 1, 0.0)
    cp.sample({"record": "sample", "frame": 2}, 2, 0.0)
    # No close() - the run is notionally still going.
    assert len(sd.read_jsonl(path)) == 2


def test_read_jsonl_survives_a_truncated_final_line(tmp_path):
    """A hard kill can sever the last line. The rest must still parse."""
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps({"record": "sample", "frame": 1}) + "\n"
        + json.dumps({"record": "sample", "frame": 2}) + "\n"
        + '{"record": "sample", "frame": 3, "live": 2'      # severed
    )
    recs = sd.read_jsonl(str(path))
    assert [r["frame"] for r in recs] == [1, 2]


def test_checkpoint_cadence_honours_frames_and_seconds():
    cp = sd.Checkpointer(None, every_frames=100, every_seconds=2.0)
    cp.sample({"record": "sample"}, 100, 10.0)
    assert not cp.due(150, 10.5), "fired early on both counts"
    assert cp.due(200, 10.5), "frame cadence did not fire"
    assert cp.due(101, 12.5), "time cadence did not fire"


# ------------------------------------------------------------------ status
def test_status_distinguishes_running_from_finished_from_killed(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    metrics = d / "metrics.jsonl"
    metrics.write_text(json.dumps({"record": "sample", "frame": 10}) + "\n")

    # Our own PID is certainly alive -> running.
    (d / "soak.pid").write_text("%d\n" % os.getpid())
    assert sd.read_status(str(d))["state"] == "running"

    # A PID that cannot be alive, with no terminal record -> killed hard.
    (d / "soak.pid").write_text("2147480000\n")
    st = sd.read_status(str(d))
    assert st["state"] == "killed"
    assert st["samples"] == 1, "a killed run must still report its samples"

    # Terminal record present -> finished, whatever the PID says.
    with open(metrics, "a") as fh:
        fh.write(json.dumps(
            {"record": "final", "status": "finished",
             "stop_reason": "max-frames"}) + "\n")
    st = sd.read_status(str(d))
    assert st["state"] == "finished"
    assert st["final"]["stop_reason"] == "max-frames"
    assert "state" in sd.format_status(st)


def test_status_on_an_empty_directory_is_not_an_error(tmp_path):
    st = sd.read_status(str(tmp_path))
    assert st["state"] == "unknown"
    assert st["samples"] == 0
    sd.format_status(st)          # must not raise


# ------------------------------------------------- rolling stats stay bounded
def test_rolling_stats_are_bounded_and_say_when_they_are_windowed():
    rs = sd.RollingStats(window=50)
    for i in range(500):
        rs.add(float(i))
    assert len(rs._w) == 50, "the window grew: the instrument itself leaks"
    assert rs.n == 500
    assert rs.max == 499.0                   # max stays exact over the run
    assert abs(rs.mean - 249.5) < 1e-9       # mean stays exact too
    assert not rs.exact                      # percentiles are windowed, said so
    small = sd.RollingStats(window=50)
    for i in range(10):
        small.add(1.0)
    assert small.exact


# --------------------------------------------------------- the CLI, for real
def test_cli_short_run_writes_checkpoints_and_a_summary(tmp_path):
    """The real game, briefly: bounds honoured end-to-end through the CLI."""
    out = tmp_path / "run"
    rc = subprocess.run(
        [sys.executable, DRIVER,
         "--results-dir", str(out),
         "--max-frames", "120", "--max-seconds", "90",
         "--destroy-every", "60", "--sample-every", "40",
         "--out", "-", "--progress-every", "0"],
        capture_output=True, text=True, timeout=180, cwd=ROOT)
    assert rc.returncode == 0, rc.stderr[-3000:]

    summary = json.loads((out / "summary.json").read_text())
    assert summary["frames"] == 120, "CLI overran its frame bound"
    assert summary["stop_reason"] in ("max-frames", "frames")

    records = sd.read_jsonl(str(out / "metrics.jsonl"))
    samples = [r for r in records if r["record"] == "sample"]
    assert len(samples) >= 3
    for rec in samples:
        for key in sd.CHECKPOINT_KEYS:
            assert key in rec, "sample missing %s" % key
    assert records[-1]["record"] == "final"
    assert records[-1]["status"] == "finished"
    assert sd.read_status(str(out))["state"] == "finished"


def test_detach_returns_immediately_and_the_child_outlives_the_call(tmp_path):
    """The parent call must return in a small fraction of the child's runtime.

    This is the property node 18 needed and did not have: launching a soak
    costs the caller nothing, so no session ever has to sit on one.
    """
    out = tmp_path / "detached"
    t0 = time.monotonic()
    rc = subprocess.run(
        [sys.executable, DRIVER, "--detach",
         "--results-dir", str(out),
         "--max-seconds", "20", "--max-frames", "100000",
         "--destroy-every", "120", "--sample-every", "30",
         "--out", "-", "--progress-every", "0"],
        capture_output=True, text=True, timeout=60, cwd=ROOT)
    launch_s = time.monotonic() - t0
    assert rc.returncode == 0, rc.stderr[-2000:]
    assert launch_s < 5.0, "--detach blocked for %.1fs" % launch_s

    pid = int((out / "soak.pid").read_text().strip())
    assert "pid" in rc.stdout and str(pid) in rc.stdout
    assert (out / "soak.log").exists()

    # It is genuinely running, detached, in its own session.
    assert sd.pid_alive(pid)
    assert os.getsid(pid) != os.getsid(os.getpid()), \
        "child shares our session: losing this session would kill it"

    # Let it get a little way in, then use the documented stop path.
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if sd.read_status(str(out))["samples"] >= 2:
            break
        time.sleep(0.25)
    st = sd.read_status(str(out))
    assert st["state"] == "running"
    assert st["samples"] >= 2, "no checkpoints appeared while detached"

    sd.stop_run(str(out))
    deadline = time.monotonic() + 20.0
    while sd.pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.25)
    assert not sd.pid_alive(pid), "SIGTERM did not stop the detached run"

    st = sd.read_status(str(out))
    assert st["state"] == "finished", \
        "a SIGTERMed run left no terminal record: %s" % st["state"]
    assert st["final"]["stop_reason"].startswith("signal:")


def test_a_hard_killed_run_still_leaves_a_parseable_jsonl(tmp_path):
    """SIGKILL: no terminal record, but every flushed sample survives."""
    out = tmp_path / "killed"
    proc = subprocess.Popen(
        [sys.executable, DRIVER,
         "--results-dir", str(out),
         "--max-seconds", "60", "--max-frames", "100000",
         "--destroy-every", "120", "--sample-every", "20",
         "--out", "-", "--progress-every", "0"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=ROOT)
    try:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if sd.read_status(str(out))["samples"] >= 2:
                break
            time.sleep(0.25)
        proc.kill()                         # SIGKILL: no handler can run
        proc.wait(timeout=15)
    finally:
        if proc.poll() is None:             # pragma: no cover
            proc.kill()

    records = sd.read_jsonl(str(out / "metrics.jsonl"))
    samples = [r for r in records if r.get("record") == "sample"]
    assert len(samples) >= 2, "a killed run lost its checkpoints"
    for rec in samples:
        for key in sd.CHECKPOINT_KEYS:
            assert key in rec
    assert not any(r.get("record") == "final" for r in records), \
        "a SIGKILLed run cannot have written a terminal record"
    assert sd.read_status(str(out))["state"] == "killed"


def test_real_game_run_is_bounded_and_drives_the_real_app():
    """The driver still drives TumbleApp, not a stub - and stops on its bound."""
    from game.app import TumbleApp

    control = sd.RunControl(max_frames=40, max_seconds=60)
    summary = sd.run(frames=10 ** 9, destroy_every=20, seed=90210,
                     writer=None, control=control, progress_every=0)
    assert summary["frames"] == 40
    assert summary["stop_reason"] == "max-frames"
    # Real destruction happened through the real damage path.
    assert summary["structures_destroyed"] >= 1
    assert summary["total_spawned"] > 0
    assert summary["physics_steps"] > 0
    assert TumbleApp is not None
    # ...and it cleaned up after itself. A ShowBase left in builtins.base
    # poisons every app built later in this interpreter, which is how a
    # harmless-looking driver takes the rest of the suite down with it.
    import builtins
    assert not hasattr(builtins, "base"), \
        "run() leaked a ShowBase into builtins"


def test_status_and_stop_flags_never_start_a_run(tmp_path):
    """The read path must be safe to call from any session at any time."""
    d = tmp_path / "nothing"
    d.mkdir()
    rc = subprocess.run(
        [sys.executable, DRIVER, "--status", str(d)],
        capture_output=True, text=True, timeout=60, cwd=ROOT)
    assert "soak status" in rc.stdout
    assert not (d / "metrics.jsonl").exists(), "--status started a run"

    rc = subprocess.run(
        [sys.executable, DRIVER, "--stop", str(d)],
        capture_output=True, text=True, timeout=60, cwd=ROOT)
    assert "no PID file" in rc.stdout
    assert not (d / "metrics.jsonl").exists(), "--stop started a run"
