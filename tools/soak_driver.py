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

Bounded, checkpointed, detachable (node 19)
-------------------------------------------
The driver above measures. What it did *not* do was guarantee it would ever
stop, or leave anything behind if it was killed - so a long soak could only
be run by a session that sat and waited on it, and two worker sessions died
at their harness timeout doing exactly that. Three additions fix it, and they
are the contract for every long run from here on:

* **Bounds.** ``--max-seconds`` (WALL clock), ``--max-frames``,
  ``--max-demolitions``. Whichever trips first ends the run cleanly. A
  wall-clock bound is always in force - the default is small - and it is
  enforced by a two-stage :class:`Watchdog` thread, so even a frame wedged
  inside C cannot outlive the deadline: at ``--max-seconds`` the loop is
  asked to stop, at ``--max-seconds + --watchdog-grace`` the process is
  terminated outright with exit code 75.
* **Checkpoints.** ``--metrics-out run.jsonl`` appends one JSON object every
  ``--sample-every`` frames (and at least every ``--sample-seconds``),
  flushed on every write, so a partial run is still a parseable file. A
  terminal ``{"record": "final", ...}`` line is written on clean exit and,
  best effort, on SIGTERM/SIGINT and on watchdog kill. ``--summary-out
  run.json`` writes the full summary. ``--results-dir DIR`` sets sensible
  defaults for all of these at once.
* **Detachment.** ``--detach`` starts the soak in its own session
  (``setsid``), redirects stdout/stderr into ``DIR/soak.log``, writes
  ``DIR/soak.pid``, prints the PID and the paths and *returns immediately*.
  Losing the calling shell does not kill the soak. Plain ``nohup ... &``
  works for the same reason: the driver writes its own PID file and
  checkpoints whenever ``--results-dir`` is given.

Reading a run back from a fresh session (this is what node 20 polls)::

    python3 tools/soak_driver.py --status soak_runs/long
    python3 tools/soak_driver.py --stop   soak_runs/long   # documented kill

A long detached soak is launched exactly like this::

    python3 tools/soak_driver.py --detach --results-dir soak_runs/long \
        --max-seconds 3600 --max-frames 400000 --destroy-every 120 \
        --sample-every 120

and nothing needs to stay attached to it.
"""

from __future__ import annotations

import argparse
import csv
import errno
import json
import math
import os
import random
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Callable, Deque, Dict, List, Optional

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

#: Exit code the watchdog uses when it has to terminate a wedged process.
#: Distinct from anything the driver itself returns, so a supervisor can tell
#: "it ran out of time and was killed" from "it finished".
EXIT_WATCHDOG = 75

#: Keys every checkpoint sample record is guaranteed to carry. Anything
#: reading a soak from a later session may rely on exactly these; the test
#: suite asserts each written line contains them.
CHECKPOINT_KEYS = (
    "record",
    "frame",
    "ts_unix",
    "ts_iso",
    "elapsed_s",
    "sim_time_s",
    "live",
    "peak_live",
    "total_spawned",
    "total_culled",
    "total_frozen",
    "total_despawned",
    "total_evicted",
    "structures_placed",
    "structures_destroyed",
    "frame_ms",
    "frame_ms_mean",
    "frame_ms_p95",
    "frame_ms_max",
    "rss_mb",
    "peak_rss_mb",
    "pid",
)

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



# --------------------------------------------------------- bounded statistics
class RollingStats:
    """Exact count/mean/max, windowed percentile - O(window) memory.

    The original driver kept every frame time, every physics time and every
    RSS reading in a plain list. Over a short run that is free; over the
    multi-hour run this harness exists to enable it is a slow leak in the
    *instrument*, which is precisely the signal the instrument is there to
    measure. So: running aggregates for count/mean/max (exact, over every
    frame) and a bounded window for the percentile - exact while the run is
    shorter than the window, and the last `window` frames beyond it. The
    `exact` flag says which you are looking at, so nobody has to guess.
    """

    def __init__(self, window: int = 3600) -> None:
        self.window = max(1, int(window))
        self._w: Deque[float] = deque(maxlen=self.window)
        self.n = 0
        self.total = 0.0
        self.max = 0.0
        self.last = 0.0

    def add(self, value: float) -> None:
        v = float(value)
        self._w.append(v)
        self.n += 1
        self.total += v
        self.last = v
        if v > self.max:
            self.max = v

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else 0.0

    def p(self, q: float) -> float:
        return percentile(list(self._w), q)

    @property
    def exact(self) -> bool:
        """True while the window still holds every sample ever added."""
        return self.n <= self.window


# ------------------------------------------------------------------ watchdog
class Watchdog:
    """A two-stage hard wall-clock guard. Nothing outlives the deadline.

    Stage one, at `deadline_s`: set the stop event. The run loop notices
    between frames and exits cleanly, flushing a final checkpoint. That is
    the normal path and it is the one that happens.

    Stage two, at `deadline_s + grace_s`: assume the process is wedged - a
    frame stuck inside Bullet, a signal handler that never got scheduled, a
    disk write blocked on a dead mount - and terminate it outright. A
    cooperative flag cannot stop a stuck frame. This stage is why the class
    exists: node 18 lost two worker sessions to a soak that could not be made
    to stop, and "it will probably finish" is not a bound.

    The clock, the sleep, and both callbacks are injectable so the escalation
    can be tested in milliseconds without killing the test runner.
    """

    def __init__(
        self,
        deadline_s: float,
        grace_s: float = 10.0,
        tick: float = 0.25,
        stop_event: Optional[threading.Event] = None,
        on_soft: Optional[Callable[[], None]] = None,
        on_hard: Optional[Callable[[], None]] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.deadline_s = float(deadline_s)
        self.grace_s = max(0.0, float(grace_s))
        self.tick = max(0.001, float(tick))
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self.on_soft = on_soft
        self.on_hard = on_hard if on_hard is not None else self._default_hard
        self._clock = clock
        self._sleep = sleep
        self._done = threading.Event()
        self.soft_fired = False
        self.hard_fired = False
        self.started_at = 0.0
        self._thread: Optional[threading.Thread] = None

    # -- the default stage two: leave no doubt that the process is gone ----
    @staticmethod
    def _default_hard() -> None:                        # pragma: no cover
        sys.stderr.write(
            "[driver] WATCHDOG: wall-clock deadline exceeded and the run did "
            "not stop on its own - terminating.\n")
        sys.stderr.flush()
        os._exit(EXIT_WATCHDOG)

    def start(self) -> "Watchdog":
        self.started_at = self._clock()
        self._thread = threading.Thread(
            target=self._loop, name="soak-watchdog", daemon=True)
        self._thread.start()
        return self

    def cancel(self) -> None:
        """Stop watching. Called on the clean-exit path."""
        self._done.set()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _loop(self) -> None:
        soft_at = self.started_at + self.deadline_s
        hard_at = soft_at + self.grace_s
        while not self._done.is_set():
            now = self._clock()
            if not self.soft_fired and now >= soft_at:
                self.soft_fired = True
                self.stop_event.set()
                if self.on_soft is not None:
                    try:
                        self.on_soft()
                    except Exception:                   # pragma: no cover
                        pass
            if now >= hard_at:
                self.hard_fired = True
                self.on_hard()
                return
            self._sleep(self.tick)


# -------------------------------------------------------------- checkpointing
class Checkpointer:
    """Appends JSON Lines progress records and flushes every single one.

    Flushing per record is the entire value: a soak that is killed - by a
    signal, by the OOM killer, by the machine going away - must still leave a
    file that parses line for line up to the moment it died. Nothing is held
    in a buffer waiting for a clean exit that may never come.

    Writes are taken under a lock because the watchdog thread may need to
    write the terminal record while the main thread is mid-sample.
    """

    def __init__(
        self,
        path: Optional[str],
        every_frames: int = 60,
        every_seconds: float = 2.0,
        fsync: bool = False,
    ) -> None:
        self.path = path
        self.every_frames = max(1, int(every_frames))
        self.every_seconds = max(0.0, float(every_seconds))
        self.fsync = bool(fsync)
        self.records = 0
        self.samples = 0
        self.last_record: Optional[Dict] = None
        self._lock = threading.Lock()
        self._last_frame = 0
        self._last_time = 0.0
        self._closed = False
        self._fh = open(path, "w") if path else None

    # -- writing ----------------------------------------------------------
    def record(self, rec: Dict) -> None:
        with self._lock:
            self.last_record = rec
            self.records += 1
            if rec.get("record") == "sample":
                self.samples += 1
            if self._fh is None or self._closed:
                return
            self._fh.write(json.dumps(rec) + "\n")
            self._fh.flush()
            if self.fsync:
                try:
                    os.fsync(self._fh.fileno())
                except OSError:                         # pragma: no cover
                    pass

    def due(self, frame: int, now: float) -> bool:
        """Cadence test: every N frames, or every S seconds, whichever first."""
        if frame - self._last_frame >= self.every_frames:
            return True
        if self.every_seconds and (now - self._last_time) >= self.every_seconds:
            return True
        return False

    def sample(self, rec: Dict, frame: int, now: float) -> None:
        self._last_frame = frame
        self._last_time = now
        self.record(rec)

    def close(self) -> None:
        with self._lock:
            if self._fh is not None and not self._closed:
                self._fh.flush()
                self._fh.close()
            self._closed = True


# ---------------------------------------------------------------- run control
class RunControl:
    """The bounds, in one object the run loop asks once per frame.

    `should_stop` is checked *before* a frame is stepped, so `--max-frames K`
    means the run steps at most K frames - never K+1.
    """

    def __init__(
        self,
        max_frames: Optional[int] = None,
        max_seconds: Optional[float] = None,
        max_demolitions: Optional[int] = None,
        stop_event: Optional[threading.Event] = None,
        checkpointer: Optional[Checkpointer] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_frames = int(max_frames) if max_frames else None
        self.max_seconds = float(max_seconds) if max_seconds else None
        self.max_demolitions = int(max_demolitions) if max_demolitions else None
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self.checkpointer = checkpointer
        self._clock = clock
        self.started = clock()
        self.stop_reason: Optional[str] = None
        self.signal_name: Optional[str] = None
        self.last_sample: Optional[Dict] = None

    @property
    def elapsed(self) -> float:
        return self._clock() - self.started

    def request_stop(self, reason: str) -> None:
        if self.stop_reason is None:
            self.stop_reason = reason
        self.stop_event.set()

    def should_stop(self, frames_done: int, demolitions: int) -> bool:
        if self.max_frames is not None and frames_done >= self.max_frames:
            self.request_stop("max-frames")
            return True
        if self.max_demolitions is not None and demolitions >= self.max_demolitions:
            self.request_stop("max-demolitions")
            return True
        if self.max_seconds is not None and self.elapsed >= self.max_seconds:
            self.request_stop("max-seconds")
            return True
        if self.stop_event.is_set():
            if self.stop_reason is None:
                self.stop_reason = "stop-requested"
            return True
        return False

    def observe(self, rec: Dict, frame: int) -> None:
        """Offer a checkpoint record; written only when the cadence is due."""
        self.last_sample = rec
        cp = self.checkpointer
        if cp is None:
            return
        now = self._clock()
        if cp.records == 0 or cp.due(frame, now):
            cp.sample(rec, frame, now)


# ------------------------------------------------------------- status reading
def pid_alive(pid: Optional[int]) -> bool:
    """Is this PID a live process? signal 0, the usual way."""
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True


def read_jsonl(path: str) -> List[Dict]:
    """Every line of a JSONL file that parses. A truncated tail is skipped.

    A killed run can leave a half-written final line: that is not corruption,
    it is the price of streaming, and it must not make the file unreadable.
    """
    out: List[Dict] = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return out


def read_status(results_dir: str) -> Dict:
    """Everything a fresh session can learn about a run it never attached to."""
    d = os.path.abspath(results_dir)
    metrics = os.path.join(d, "metrics.jsonl")
    if not os.path.exists(metrics) and os.path.isfile(d):
        metrics, d = d, os.path.dirname(d)
    pid_path = os.path.join(d, "soak.pid")
    summary_path = os.path.join(d, "summary.json")
    log_path = os.path.join(d, "soak.log")

    pid = None
    if os.path.exists(pid_path):
        try:
            pid = int(open(pid_path).read().strip().split()[0])
        except (ValueError, IndexError, OSError):
            pid = None

    records = read_jsonl(metrics)
    samples = [r for r in records if r.get("record") == "sample"]
    final = None
    for r in reversed(records):
        if r.get("record") == "final":
            final = r
            break
    last = samples[-1] if samples else None

    summary = None
    if os.path.exists(summary_path):
        try:
            summary = json.load(open(summary_path))
        except (ValueError, OSError):
            summary = None

    alive = pid_alive(pid)
    if final is not None:
        state = "finished"
    elif alive:
        state = "running"
    elif pid is not None or records:
        # Progress on disk, no terminal record, nobody home: it was killed
        # hard (SIGKILL / OOM / power). The partial file is still the truth.
        state = "killed"
    else:
        state = "unknown"

    return {
        "results_dir": d,
        "metrics_path": metrics,
        "summary_path": summary_path if summary else None,
        "log_path": log_path if os.path.exists(log_path) else None,
        "pid": pid,
        "pid_alive": alive,
        "state": state,
        "records": len(records),
        "samples": len(samples),
        "last_sample": last,
        "final": final,
        "summary": summary,
    }


def format_status(st: Dict) -> str:
    """The concise read-out a polling session sees."""
    last = st.get("last_sample") or {}
    fin = st.get("final") or {}
    lines = [
        "[driver] soak status: %s" % st["results_dir"],
        "  state            : %-10s (pid %s, %s)" % (
            st["state"], st["pid"],
            "alive" if st["pid_alive"] else "not running"),
    ]
    if not last and not fin:
        lines.append("  (no checkpoint records yet)")
        return "\n".join(lines)
    # Prefer the terminal record when there is one: it is the last word on
    # the run, taken after the final frame, where the last periodic sample is
    # only as fresh as the checkpoint cadence allowed.
    src = fin or last
    lines += [
        "  progress         : frame %s   elapsed %.1f s   sim %.1f s" % (
            src.get("frame"), src.get("elapsed_s") or 0.0,
            src.get("sim_time_s") or 0.0),
        "  structures       : %s placed / %s destroyed" % (
            src.get("structures_placed"), src.get("structures_destroyed")),
        "  debris live      : %s   (peak %s)" % (
            src.get("live"), src.get("peak_live")),
        "  debris totals    : %s spawned / %s culled" % (
            src.get("total_spawned"), src.get("total_culled")),
        "  frame time (ms)  : mean %.3f   p95 %.3f   max %.3f" % (
            src.get("frame_ms_mean") or 0.0, src.get("frame_ms_p95") or 0.0,
            src.get("frame_ms_max") or 0.0),
        "  memory (RSS)     : %.1f MB   (peak %.1f MB)" % (
            src.get("rss_mb") or 0.0, src.get("peak_rss_mb") or 0.0),
        "  checkpoints      : %d records (%d samples) -> %s" % (
            st["records"], st["samples"], st["metrics_path"]),
    ]
    if fin:
        lines.append("  terminal record  : status=%s  reason=%s" % (
            fin.get("status"), fin.get("stop_reason")))
    if st.get("summary_path"):
        lines.append("  summary          : %s" % st["summary_path"])
    if st.get("log_path"):
        lines.append("  log              : %s" % st["log_path"])
    if st["state"] == "killed":
        lines.append("  NOTE: no terminal record - this run was killed hard; "
                     "the samples above are still valid.")
    return "\n".join(lines)


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
    control: Optional[RunControl] = None,
    stats_window: int = 3600,
    app_factory: Optional[Callable[[], object]] = None,
    destroy_app: bool = True,
) -> Dict:
    """Step the real headless game, measuring every frame, under *control*.

    *frames* is the nominal length; *control* (a :class:`RunControl`) holds
    the real bounds and is consulted BEFORE each frame, so ``max_frames=K``
    stops at exactly K frames stepped and never K+1. The loop also offers
    every frame's record to the control's checkpointer, which writes at its
    own cadence - that is what makes a killed run still readable.

    Returns a summary dict. Writes nothing to stdout unless *progress_every*
    is positive; the caller owns the reporting.

    *app_factory* exists only so the harness tests can drive this loop with a
    cheap fake app in milliseconds. Left at None - which is how every real
    invocation calls it - it builds the real ``TumbleApp(headless=True)``.

    The app this function creates, this function tears down (*destroy_app*).
    A ShowBase installs itself as the global ``builtins.base``; leaving one
    behind poisons every later app in the same interpreter, which is exactly
    what the rest of the suite's ``a.destroy()`` calls are avoiding. A
    detached soak process would get away with it - it exits anyway - but a
    driver that cannot be called twice in one process is a broken tool, so it
    cleans up after itself.
    """
    if control is None:
        control = RunControl()
    owns_app = app_factory is None
    if app_factory is None:
        def app_factory():
            from game.app import TumbleApp
            return TumbleApp(headless=True)

    app = app_factory()

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

    # Bounded aggregates, not unbounded lists: a multi-hour run must not
    # leak inside the instrument measuring the leak. See RollingStats.
    frame_ms = RollingStats(stats_window)
    physics_ms = RollingStats(stats_window)
    rss_stats = RollingStats(stats_window)
    destroy_frames: List[int] = []
    DESTROY_FRAMES_CAP = 10000     # keep the summary a fixed size, too
    destroy_frames_truncated = 0

    placed = 0
    destroyed = 0
    pending = None            # structure authored and not yet brought down
    peak_live = 0
    peak_stepped = 0
    peak_frozen = 0
    peak_total_bodies = 0
    peak_rss = 0.0
    chunks_released = 0
    frames_done = 0

    rss0 = rss_mb()
    started = time.perf_counter()
    control.started = time.monotonic()

    frame = 0
    while True:
        # The bounds are checked BEFORE the frame is stepped. `max_frames=K`
        # therefore means "at most K frames stepped", which is exactly what
        # the tests assert and what a supervisor is entitled to assume.
        if frames is not None and frames_done >= frames:
            control.request_stop("frames")
            break
        if control.should_stop(frames_done, destroyed):
            break
        frame += 1
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
            if len(destroy_frames) < DESTROY_FRAMES_CAP:
                destroy_frames.append(frame)
            else:
                destroy_frames_truncated += 1
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

        frame_ms.add(elapsed_ms)
        physics_ms.add(per_frame_physics_ms[0])
        rss_stats.add(rss)
        peak_rss = max(peak_rss, rss)
        frames_done = frame

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

        # Offer the checkpoint. The checkpointer decides whether the cadence
        # is due; it flushes every record it does write.
        control.observe(
            {
                "record": "sample",
                "frame": frame,
                "ts_unix": sample["ts_unix"],
                "ts_iso": datetime.now(timezone.utc).isoformat(
                    timespec="milliseconds"),
                "elapsed_s": sample["elapsed_s"],
                "sim_time_s": sample["sim_time_s"],
                "live": live,
                "peak_live": peak_live,
                "active": sample["active"],
                "stepped": stepped,
                "frozen": frozen,
                "total_spawned": app.debris.total_spawned,
                # "culled" = bodies taken out of the world entirely. Freezing
                # is reported separately: a frozen body is still resident.
                "total_culled": app.debris.total_despawned,
                "total_frozen": app.debris.total_frozen,
                "total_despawned": app.debris.total_despawned,
                "total_evicted": app.debris.total_evicted,
                "retired_total": sample["retired_total"],
                "structures_placed": placed,
                "structures_destroyed": destroyed,
                "frame_ms": round(elapsed_ms, 4),
                "frame_ms_mean": round(frame_ms.mean, 4),
                "frame_ms_p95": round(frame_ms.p(0.95), 4),
                "frame_ms_max": round(frame_ms.max, 4),
                "frame_ms_p95_exact": frame_ms.exact,
                "physics_ms": round(per_frame_physics_ms[0], 4),
                "physics_ms_mean": round(physics_ms.mean, 4),
                "rss_mb": round(rss, 2),
                "peak_rss_mb": round(peak_rss, 2),
                "rss_growth_mb": round(rss - rss0, 2),
                "pid": os.getpid(),
            },
            frame,
        )

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

    summary = {
        "frames": frames_done,
        "frames_requested": frames,
        "stop_reason": control.stop_reason or "frames",
        "signal": control.signal_name,
        "max_frames": control.max_frames,
        "max_seconds": control.max_seconds,
        "max_demolitions": control.max_demolitions,
        "pid": os.getpid(),
        "finished_iso": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
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
        "frame_ms_mean": round(frame_ms.mean, 4),
        "frame_ms_p50": round(frame_ms.p(0.50), 4),
        "frame_ms_p95": round(frame_ms.p(0.95), 4),
        "frame_ms_max": round(frame_ms.max, 4),
        # p50/p95 are over the last `stats_window` frames once a run outgrows
        # it; mean and max are exact over every frame. Say so, do not imply
        # a precision the aggregate does not have.
        "frame_ms_percentiles_exact": frame_ms.exact,
        "frame_ms_window": frame_ms.window,
        "physics_ms_mean": round(physics_ms.mean, 4),
        "physics_ms_p95": round(physics_ms.p(0.95), 4),
        "physics_ms_max": round(physics_ms.max, 4),
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
        "rss_peak_mb": round(peak_rss if rss_stats.n else rss0, 2),
        "rss_end_mb": round(rss_stats.last if rss_stats.n else rss0, 2),
        "rss_growth_mb": round(
            (rss_stats.last if rss_stats.n else rss0) - rss0, 2),
        "destroy_frames_truncated": destroy_frames_truncated,
    }

    if destroy_app and owns_app:
        destroyer = getattr(app, "destroy", None)
        if callable(destroyer):
            destroyer()

    return summary


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

    # ---- bounds: a run must always be able to end without supervision ----
    b = p.add_argument_group(
        "bounds",
        "Whichever trips first ends the run cleanly. A wall-clock bound is "
        "ALWAYS in force; --max-seconds 0 is rejected.")
    b.add_argument("--max-seconds", type=float, default=60.0, metavar="S",
                   help="hard WALL-clock bound on the whole run")
    b.add_argument("--max-frames", type=int, default=None, metavar="N",
                   help="stop after N host frames (default: --frames)")
    b.add_argument("--max-demolitions", type=int, default=None, metavar="N",
                   help="stop once N structures have been brought down")
    b.add_argument("--watchdog-grace", type=float, default=10.0, metavar="S",
                   help="seconds after --max-seconds before the watchdog "
                        "terminates the process outright (exit %d)"
                        % EXIT_WATCHDOG)

    # ---- checkpointing: partial results must survive a kill --------------
    c = p.add_argument_group("checkpointing")
    c.add_argument("--results-dir", default=None, metavar="DIR",
                   help="directory for metrics.jsonl, summary.json, soak.log "
                        "and soak.pid; sets the defaults for all of them")
    c.add_argument("--metrics-out", default=None, metavar="PATH",
                   help="JSONL checkpoint stream (one record per sample, "
                        "flushed on every write)")
    c.add_argument("--summary-out", default=None, metavar="PATH",
                   help="final run summary as JSON (alias of --summary-json)")
    c.add_argument("--sample-every", type=int, default=60, metavar="N",
                   help="write a checkpoint sample every N frames")
    c.add_argument("--sample-seconds", type=float, default=2.0, metavar="S",
                   help="...and at least this often in wall seconds (0 off)")
    c.add_argument("--stats-window", type=int, default=3600, metavar="N",
                   help="rolling window for frame-time percentiles; mean and "
                        "max stay exact over the whole run")

    # ---- detachment: no session need ever wait on a soak -----------------
    d = p.add_argument_group("detached execution")
    d.add_argument("--detach", action="store_true",
                   help="run the soak in its own session (setsid), log to "
                        "DIR/soak.log, write DIR/soak.pid, print them and "
                        "return immediately. Requires --results-dir.")
    d.add_argument("--pid-file", default=None, metavar="PATH",
                   help="where to write this run's PID (default: "
                        "DIR/soak.pid when --results-dir is given)")
    d.add_argument("--status", default=None, metavar="DIR",
                   help="do not run: read DIR's checkpoints + PID and print "
                        "a status summary, then exit")
    d.add_argument("--status-json", action="store_true",
                   help="with --status, print raw JSON instead of text")
    d.add_argument("--stop", default=None, metavar="DIR",
                   help="do not run: SIGTERM the run recorded in DIR's PID "
                        "file so it flushes a final checkpoint and exits")
    return p


# ------------------------------------------------------------- detach plumbing
def spawn_detached(argv: List[str], results_dir: str, log_path: str,
                   pid_path: str) -> int:
    """Start this same driver in its own session and return its PID at once.

    ``setsid`` + a new process group means the child survives the death of
    whatever started it - a shell, an SSH connection, a worker session that
    hit its timeout. stdout and stderr go to *log_path*; nothing is left
    connected to our terminal, so nothing can block on a pipe nobody reads.

    This is the same guarantee as ``nohup python3 tools/soak_driver.py ... &``
    and is documented as such; ``--detach`` just removes the chance of
    getting the incantation wrong.
    """
    os.makedirs(results_dir, exist_ok=True)
    child = [sys.executable, os.path.abspath(__file__)] + argv
    log = open(log_path, "a", buffering=1)
    log.write("[driver] --- detached run started %s ---\n"
              % datetime.now(timezone.utc).isoformat(timespec="seconds"))
    log.flush()
    proc = subprocess.Popen(
        child,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,           # setsid: detached from our session
        cwd=_ROOT,
        close_fds=True,
    )
    log.close()
    with open(pid_path, "w") as fh:
        fh.write("%d\n" % proc.pid)
    return proc.pid


def strip_detach_args(argv: List[str]) -> List[str]:
    """The child's argv: everything we were given except --detach itself."""
    return [a for a in argv if a != "--detach"]


def stop_run(results_dir: str) -> int:
    """The documented kill path: SIGTERM, so the run flushes a final record."""
    st = read_status(results_dir)
    pid = st["pid"]
    if not pid:
        print("[driver] no PID file in %s" % st["results_dir"])
        return 1
    if not st["pid_alive"]:
        print("[driver] pid %d is not running (state: %s)" % (pid, st["state"]))
        return 0
    os.kill(pid, signal.SIGTERM)
    print("[driver] SIGTERM -> pid %d; it will flush a final checkpoint to %s"
          % (pid, st["metrics_path"]))
    return 0


def resolve_paths(args) -> Dict[str, Optional[str]]:
    """--results-dir is the one flag that sets sane defaults for the rest."""
    rd = os.path.abspath(args.results_dir) if args.results_dir else None
    metrics = args.metrics_out
    summary = args.summary_out or args.summary_json
    pid_file = args.pid_file
    if rd:
        os.makedirs(rd, exist_ok=True)
        metrics = metrics or os.path.join(rd, "metrics.jsonl")
        summary = summary or os.path.join(rd, "summary.json")
        pid_file = pid_file or os.path.join(rd, "soak.pid")
    return {
        "results_dir": rd,
        "metrics": metrics,
        "summary": summary,
        "pid_file": pid_file,
        "log": os.path.join(rd, "soak.log") if rd else None,
    }


def main(argv=None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw)

    # ---- read-only paths: never start a run ------------------------------
    if args.status is not None:
        st = read_status(args.status)
        print(json.dumps(st, indent=2) if args.status_json
              else format_status(st))
        return 0 if st["state"] in ("running", "finished") else 1
    if args.stop is not None:
        return stop_run(args.stop)

    paths = resolve_paths(args)

    # ---- detach: fork, print, return. This call must be instant. ---------
    if args.detach:
        if not paths["results_dir"]:
            print("[driver] --detach requires --results-dir", file=sys.stderr)
            return 2
        pid = spawn_detached(strip_detach_args(raw), paths["results_dir"],
                             paths["log"], paths["pid_file"])
        print("[driver] detached soak started")
        print("  pid      : %d" % pid)
        print("  results  : %s" % paths["results_dir"])
        print("  log      : %s" % paths["log"])
        print("  metrics  : %s" % paths["metrics"])
        print("  summary  : %s" % paths["summary"])
        print("  pid file : %s" % paths["pid_file"])
        print("  status   : %s %s --status %s"
              % (os.path.basename(sys.executable),
                 os.path.relpath(os.path.abspath(__file__), _ROOT),
                 paths["results_dir"]))
        print("  stop     : %s %s --stop %s"
              % (os.path.basename(sys.executable),
                 os.path.relpath(os.path.abspath(__file__), _ROOT),
                 paths["results_dir"]))
        return 0

    # ---- bounds ----------------------------------------------------------
    frames = args.frames
    if args.seconds is not None:
        frames = max(1, int(round(args.seconds / FRAME_DT)))
    if args.max_frames is not None:
        frames = args.max_frames
    frames = max(1, int(frames))

    if args.max_seconds is None or args.max_seconds <= 0:
        print("[driver] --max-seconds must be > 0: every run is bounded by "
              "wall clock, without exception.", file=sys.stderr)
        return 2

    # ---- checkpointing ---------------------------------------------------
    checkpointer = Checkpointer(
        paths["metrics"],
        every_frames=args.sample_every,
        every_seconds=args.sample_seconds,
    )
    control = RunControl(
        max_frames=args.max_frames if args.max_frames is not None else frames,
        max_seconds=args.max_seconds,
        max_demolitions=args.max_demolitions,
        checkpointer=checkpointer,
    )

    if paths["pid_file"]:
        with open(paths["pid_file"], "w") as fh:
            fh.write("%d\n" % os.getpid())

    out_path = args.out if args.out not in ("", "-") else None
    writer = MetricsWriter(out_path, args.format) if out_path else None

    finalised = [False]

    def write_final(status: str, extra: Optional[Dict] = None) -> None:
        """The terminal record. Written exactly once, whatever path we die on.

        Clean exit, SIGTERM, SIGINT, watchdog: all four land here. Its
        presence is what lets a later session tell "finished" from "killed
        hard" without ever having been attached.
        """
        if finalised[0]:
            return
        finalised[0] = True
        base = dict(control.last_sample or {})
        base.update({
            "record": "final",
            "status": status,
            "stop_reason": control.stop_reason,
            "signal": control.signal_name,
            "ts_unix": round(time.time(), 3),
            "ts_iso": datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"),
            "wall_seconds": round(control.elapsed, 3),
            "pid": os.getpid(),
        })
        for key in CHECKPOINT_KEYS:
            base.setdefault(key, None)
        if extra:
            base.update(extra)
        checkpointer.record(base)

    # ---- signals: flush, then go ----------------------------------------
    def on_signal(signum, _frame):
        control.signal_name = signal.Signals(signum).name
        control.request_stop("signal:%s" % control.signal_name)
        # Do NOT write the final record here and do NOT exit here: the run
        # loop notices the stop between frames, returns normally, and the
        # ordinary exit path writes the terminal record and the summary. A
        # signal handler that longjmps out of a Bullet step is how you get a
        # corrupt tail instead of a clean one.

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, on_signal)
        except (ValueError, OSError):        # not the main thread: fine
            pass

    # ---- the hard guard --------------------------------------------------
    def on_hard_kill():
        # Last breath: say so in the checkpoint file, then die. If we are
        # here the main thread is wedged, so nothing else will get the
        # chance to record it.
        try:
            write_final("watchdog-killed",
                        {"stop_reason": "watchdog:max-seconds+grace"})
            checkpointer.close()
        except Exception:                                # pragma: no cover
            pass
        sys.stderr.write(
            "[driver] WATCHDOG: exceeded --max-seconds %.1f + grace %.1f "
            "and did not stop - terminating (exit %d).\n"
            % (args.max_seconds, args.watchdog_grace, EXIT_WATCHDOG))
        sys.stderr.flush()
        os._exit(EXIT_WATCHDOG)

    watchdog = Watchdog(
        deadline_s=args.max_seconds,
        grace_s=args.watchdog_grace,
        stop_event=control.stop_event,
        on_soft=lambda: control.request_stop("max-seconds"),
        on_hard=on_hard_kill,
    ).start()

    print("[driver] headless Tumble (%s window) - pid %d, bounds: "
          "<=%d frames, <=%.1f s wall, demolitions %s; destroying every %d "
          "frames, seed %d"
          % (config.PRC_HEADLESS.split()[1], os.getpid(), frames,
             args.max_seconds,
             args.max_demolitions if args.max_demolitions else "unbounded",
             args.destroy_every, args.seed), flush=True)
    if paths["metrics"]:
        print("[driver] checkpoints -> %s (every %d frames / %.1f s)"
              % (paths["metrics"], args.sample_every, args.sample_seconds),
              flush=True)

    status = "ok"
    summary: Optional[Dict] = None
    try:
        summary = run(
            frames=frames,
            destroy_every=args.destroy_every,
            seed=args.seed,
            writer=writer,
            max_structures=args.structures,
            release_chunks=args.release_chunks,
            progress_every=max(0, args.progress_every),
            control=control,
            stats_window=args.stats_window,
        )
    except BaseException as exc:                          # noqa: BLE001
        status = "error"
        write_final("error", {"error": "%s: %s" % (type(exc).__name__, exc)})
        raise
    finally:
        watchdog.cancel()
        if writer is not None:
            writer.close()
        if summary is not None:
            write_final("finished", {"summary": summary})
        checkpointer.close()

    print(report(summary, out_path, writer.rows if writer else 0))
    print("[driver] stopped on: %s  (%d frames, %.2f s wall)"
          % (summary["stop_reason"], summary["frames"],
             summary["wall_seconds"]))

    if paths["summary"]:
        with open(paths["summary"], "w") as fh:
            json.dump(summary, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        print("[driver] summary -> %s" % paths["summary"])
    if paths["metrics"]:
        print("[driver] checkpoints: %d records (%d samples) -> %s"
              % (checkpointer.records, checkpointer.samples, paths["metrics"]))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
