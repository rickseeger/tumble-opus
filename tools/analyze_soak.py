#!/usr/bin/env python3
"""Judge a Tumble soak: the sustainability invariants, in code, with numbers.

The soak driver (``tools/soak_driver.py``) is deliberately instrumentation
only - it measures and it writes, it never decides.  This module is the other
half: it reads the metrics JSONL a soak leaves behind and decides, against
explicit numeric thresholds, whether sustained demolition stayed bounded.

Five invariants, each one a thing that could plausibly be false:

1. ``live_cap``        - the simulated debris body count NEVER exceeds the
                         global cap (:data:`game.config.DEBRIS_MAX_LIVE`), at
                         any sample.  A single violation fails the run; this
                         is a hard ceiling, not a target.
2. ``live_no_drift``   - live debris does not trend upward.  Mean live count
                         over the last quarter <= mean over the first quarter
                         times ``--live-drift-tol``.  Catches the leak the cap
                         cannot: a field that creeps toward its ceiling and
                         parks there because retirement has stopped keeping up.
3. ``frame_time``      - per-frame step time does not degrade.  Mean frame_ms
                         over the last quarter <= first-quarter mean times
                         ``--frame-time-tol``.  Catches O(n) bookkeeping that
                         grows with total-spawned-ever rather than with live.
4. ``retirement``      - retired_total actually climbs in every quarter.  A run
                         where nothing is retired but nothing grows either is
                         a run that quietly stopped spawning debris; that is
                         not sustainability, that is an empty world.
5. ``memory``          - resident memory stays bounded.  Two ways to be
                         bounded, BOTH required:
                           (a) absolute: total RSS growth < ``--rss-growth-mb``
                           (b) plateau : the MB-per-1000-frames acquisition
                               rate over the last half is at most
                               ``--rss-plateau-ratio`` times the rate over the
                               first half.  Linear growth scores ~1.0 and
                               fails, which is the point: "it only grew 250 MB
                               in five minutes" is not boundedness, it is a
                               leak with a short observation window.

Plus a sanity gate (``demolitions``) so a green board cannot be bought by
simply not doing any work: the run must have demolished at least
``--min-demolitions`` structures, the counter must be monotone, and debris
must actually have been spawned by those demolitions.

Usage::

    tools/analyze_soak.py soak_runs/node20_long/metrics.jsonl
    tools/analyze_soak.py <jsonl> --json verdict.json --quiet

Exit code is 0 only if every invariant holds.  Any failure exits 1 and says
which, with the numbers.  Thresholds are CLI flags so they are visible and
auditable - but relaxing one to make a red run green is falsifying the
result, not fixing it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# --- default thresholds ---------------------------------------------------
# Every one of these is a judgement call, so every one is written down with
# the reason it has the value it has.

#: Live-debris drift.  5%: the live count is a sampled instantaneous value and
#: bounces with the demolition phase, so demanding last-quarter <= first-
#: quarter exactly would fail on noise.  It may not, however, *climb*: 5% of a
#: 260-body cap is 13 bodies, well inside sampling jitter and far below any
#: real leak, which shows up as a march toward the ceiling.
DEFAULT_LIVE_DRIFT_TOL = 1.05

#: Frame-time degradation.  25%: the soak shares a machine with whatever else
#: is running, and frame_ms is wall time, so it carries scheduler noise.  A
#: genuine algorithmic degradation (per-frame work proportional to bodies ever
#: spawned) grows without limit and blows past 25% long before a run ends; 25%
#: is loose enough to survive a noisy host and tight enough that real rot
#: cannot hide under it.
DEFAULT_FRAME_TIME_TOL = 1.25

#: Absolute RSS growth ceiling for a run of this shape, in MB.  The process
#: starts around 100 MB (Panda3D + Bullet + the pre-generated chunk library).
#: 250 MB of growth means ending near 350 MB - already generous for a headless
#: run with a 260-body cap, and the point past which "bounded" stops being a
#: defensible word.
DEFAULT_RSS_GROWTH_MB = 250.0

#: RSS plateau ratio.  Memory acquisition must be *decelerating*.  A process
#: that caches, pools and reuses shows a big first-half rate and a small
#: second-half one; a process that leaks shows the same rate forever.  0.5
#: demands the second half acquire at most half the rate of the first - a
#: weak demand of a plateauing curve and an impossible one for a linear leak.
DEFAULT_RSS_PLATEAU_RATIO = 0.5

#: Warm-up fraction discarded before the drift/degradation comparisons.  The
#: first frames pay import, JIT-free interpreter warm-up, the first structure
#: build and the first shatter; including them makes the first quarter look
#: artificially slow and would *hide* later degradation.  Discarding 10% is
#: the conservative direction: it makes the tests harder to pass, not easier.
#: The cap invariant (1) is checked over the FULL series, warm-up included -
#: a hard ceiling gets no grace period.
DEFAULT_WARMUP_FRAC = 0.10

#: Minimum demolitions for the run to count as a sustained soak at all.
DEFAULT_MIN_DEMOLITIONS = 60

#: Minimum mean chunks spawned per demolition.  Proves the demolition counter
#: is counting real shatters, not incrementing past empty structures.
DEFAULT_MIN_CHUNKS_PER_DEMOLITION = 20.0


def mean(xs: Sequence[float]) -> float:
    return float(sum(xs)) / len(xs) if xs else 0.0


def pct(xs: Sequence[float], q: float) -> float:
    """Nearest-rank percentile; exact, the series is small enough to sort."""
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return float(s[i])


# --------------------------------------------------------------------------


@dataclass
class Result:
    """One invariant's verdict."""
    name: str
    passed: bool
    headline: str
    numbers: Dict[str, Any] = field(default_factory=dict)

    @property
    def mark(self) -> str:
        return "PASS" if self.passed else "FAIL"


class SoakSeries:
    """The sample records of one soak, in frame order."""

    def __init__(self, samples: List[Dict[str, Any]], path: str = "<memory>"):
        if not samples:
            raise ValueError(f"{path}: no 'sample' records found")
        self.path = path
        self.samples = sorted(samples, key=lambda r: r.get("frame", 0))

    @classmethod
    def from_jsonl(cls, path: str) -> "SoakSeries":
        samples: List[Dict[str, Any]] = []
        with open(path, "r") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # A run killed mid-write can leave one torn final line.
                    # Every earlier line was flushed and is valid; drop the
                    # tail rather than throwing the whole run away.
                    sys.stderr.write(
                        f"[analyze_soak] {path}:{lineno}: truncated record "
                        f"ignored (run ended mid-write)\n")
                    continue
                if rec.get("record") == "sample":
                    samples.append(rec)
        return cls(samples, path)

    # -- accessors ---------------------------------------------------------
    def col(self, key: str, default: float = 0.0) -> List[float]:
        return [float(r.get(key, default) or 0.0) for r in self.samples]

    def icol(self, key: str, default: int = 0) -> List[int]:
        return [int(r.get(key, default) or 0) for r in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def warm(self, frac: float) -> "SoakSeries":
        """The series with the first *frac* of samples discarded."""
        cut = int(len(self.samples) * frac)
        cut = min(cut, max(0, len(self.samples) - 4))
        return SoakSeries(self.samples[cut:], self.path)

    def split(self, n: int) -> List["SoakSeries"]:
        """Split into *n* contiguous, roughly equal segments."""
        L = len(self.samples)
        if L < n:
            raise ValueError(f"{self.path}: {L} samples cannot form {n} parts")
        bounds = [round(i * L / n) for i in range(n + 1)]
        return [SoakSeries(self.samples[bounds[i]:bounds[i + 1]], self.path)
                for i in range(n)]

    # -- derived -----------------------------------------------------------
    @property
    def frames(self) -> int:
        return self.samples[-1].get("frame", 0)

    @property
    def wall_s(self) -> float:
        return float(self.samples[-1].get("elapsed_s", 0.0))

    @property
    def sim_s(self) -> float:
        return float(self.samples[-1].get("sim_time_s", 0.0))

    @property
    def demolitions(self) -> int:
        return int(self.samples[-1].get("structures_destroyed", 0))


# --- the invariants -------------------------------------------------------


def check_demolitions(s: SoakSeries, min_demolitions: int,
                      min_chunks_per: float) -> Result:
    d = s.icol("structures_destroyed")
    spawned = s.icol("total_spawned")
    monotone = all(b >= a for a, b in zip(d, d[1:]))
    spawn_monotone = all(b >= a for a, b in zip(spawned, spawned[1:]))
    total_d = d[-1]
    total_spawned = spawned[-1]
    per = (total_spawned / total_d) if total_d else 0.0
    ok = (total_d >= min_demolitions and monotone and spawn_monotone
          and per >= min_chunks_per)
    return Result(
        "demolitions", ok,
        f"{total_d} structures demolished (need >= {min_demolitions}), "
        f"{total_spawned} chunks spawned = {per:.1f}/demolition "
        f"(need >= {min_chunks_per:.0f})",
        {"demolitions": total_d, "min_demolitions": min_demolitions,
         "total_spawned": total_spawned, "chunks_per_demolition": round(per, 2),
         "min_chunks_per_demolition": min_chunks_per,
         "demolition_counter_monotone": monotone,
         "spawn_counter_monotone": spawn_monotone,
         "frames": s.frames, "wall_s": round(s.wall_s, 1),
         "sim_s": round(s.sim_s, 1)})


def check_live_cap(s: SoakSeries, cap: int) -> Result:
    """Hard ceiling, checked over every sample including warm-up."""
    live = s.icol("live")
    stepped = s.icol("stepped")
    # `live` is the field's live body count; `stepped` is what the solver
    # actually integrates.  The cap binds the simulated set, so check both -
    # whichever is larger is the honest number to compare against the cap.
    peak_live = max(live)
    peak_stepped = max(stepped) if any(stepped) else 0
    worst = max(peak_live, peak_stepped)
    over = [(s.samples[i].get("frame"), live[i], stepped[i])
            for i in range(len(live))
            if live[i] > cap or stepped[i] > cap]
    ok = not over
    head = (f"peak live {peak_live}, peak stepped {peak_stepped}, "
            f"cap {cap} -> {'never exceeded' if ok else f'EXCEEDED at {len(over)} sample(s)'}")
    return Result("live_cap", ok, head,
                  {"cap": cap, "peak_live": peak_live,
                   "peak_stepped": peak_stepped, "worst": worst,
                   "samples_over_cap": len(over),
                   "first_violations": over[:5],
                   "mean_live": round(mean(live), 1),
                   "samples": len(s)})


def check_live_drift(s: SoakSeries, tol: float) -> Result:
    q = s.split(4)
    m = [mean(p.col("live")) for p in q]
    ratio = (m[3] / m[0]) if m[0] else float("inf")
    ok = m[3] <= m[0] * tol
    return Result(
        "live_no_drift", ok,
        f"mean live Q1 {m[0]:.1f} -> Q4 {m[3]:.1f} (ratio {ratio:.3f}, "
        f"tolerance {tol:.2f})",
        {"quarter_means": [round(x, 2) for x in m], "ratio": round(ratio, 4),
         "tolerance": tol, "q1_mean": round(m[0], 2), "q4_mean": round(m[3], 2)})


def check_frame_time(s: SoakSeries, tol: float) -> Result:
    q = s.split(4)
    m = [mean(p.col("frame_ms")) for p in q]
    p95 = [pct(p.col("frame_ms"), 0.95) for p in q]
    ratio = (m[3] / m[0]) if m[0] else float("inf")
    ok = m[3] <= m[0] * tol
    return Result(
        "frame_time", ok,
        f"mean frame_ms Q1 {m[0]:.3f} -> Q4 {m[3]:.3f} (ratio {ratio:.3f}, "
        f"tolerance {tol:.2f}); p95 Q1 {p95[0]:.3f} -> Q4 {p95[3]:.3f}",
        {"quarter_mean_ms": [round(x, 4) for x in m],
         "quarter_p95_ms": [round(x, 4) for x in p95],
         "ratio": round(ratio, 4), "tolerance": tol,
         "q1_mean_ms": round(m[0], 4), "q4_mean_ms": round(m[3], 4)})


def check_retirement(s: SoakSeries) -> Result:
    q = s.split(4)
    ends = [p.icol("retired_total")[-1] for p in q]
    starts = [p.icol("retired_total")[0] for p in q]
    gains = [ends[i] - starts[i] for i in range(4)]
    total = s.icol("retired_total")
    monotone = all(b >= a for a, b in zip(total, total[1:]))
    ok = monotone and all(g > 0 for g in gains) and total[-1] > 0
    return Result(
        "retirement", ok,
        f"retired_total {total[0]} -> {total[-1]}; gain per quarter {gains} "
        f"(each must be > 0)",
        {"retired_start": total[0], "retired_end": total[-1],
         "quarter_gains": gains, "monotone": monotone,
         "total_frozen_end": s.icol("total_frozen")[-1],
         "total_despawned_end": s.icol("total_despawned")[-1],
         "total_evicted_end": s.icol("total_evicted")[-1]})


def check_memory(s: SoakSeries, growth_mb: float, plateau_ratio: float) -> Result:
    rss = s.col("rss_mb")
    frames = s.icol("frame")
    start, end, peak = rss[0], rss[-1], max(rss)
    growth = end - start
    half = len(rss) // 2
    df1 = max(1, frames[half] - frames[0])
    df2 = max(1, frames[-1] - frames[half])
    rate1 = (rss[half] - rss[0]) / df1 * 1000.0      # MB per 1000 frames
    rate2 = (rss[-1] - rss[half]) / df2 * 1000.0
    ratio = (rate2 / rate1) if rate1 > 1e-9 else (0.0 if rate2 <= 0 else float("inf"))
    abs_ok = growth < growth_mb
    plateau_ok = ratio <= plateau_ratio
    ok = abs_ok and plateau_ok
    why = []
    if not abs_ok:
        why.append(f"absolute growth {growth:.1f} MB >= bound {growth_mb:.0f} MB")
    if not plateau_ok:
        why.append(f"no plateau: 2nd-half rate is {ratio:.2f}x the 1st-half "
                   f"rate (bound {plateau_ratio:.2f}x) - growth is not decelerating")
    return Result(
        "memory", ok,
        (f"RSS {start:.1f} -> {end:.1f} MB (peak {peak:.1f}, growth "
         f"{growth:+.1f} MB over {frames[-1]} frames); acquisition rate "
         f"{rate1:.2f} -> {rate2:.2f} MB/1k frames (ratio {ratio:.2f})"
         + ("" if ok else "; " + "; ".join(why))),
        {"rss_start_mb": round(start, 2), "rss_end_mb": round(end, 2),
         "rss_peak_mb": round(peak, 2), "rss_growth_mb": round(growth, 2),
         "rss_growth_bound_mb": growth_mb, "absolute_ok": abs_ok,
         "rate_first_half_mb_per_1k_frames": round(rate1, 4),
         "rate_second_half_mb_per_1k_frames": round(rate2, 4),
         "plateau_ratio": round(ratio, 4),
         "plateau_ratio_bound": plateau_ratio, "plateau_ok": plateau_ok,
         "mb_per_demolition": round(growth / s.demolitions, 3)
                              if s.demolitions else None})


# --------------------------------------------------------------------------


def default_cap() -> int:
    """The cap, from the one place it is defined."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        from game import config           # noqa: WPS433 (deliberate late import)
        return int(config.DEBRIS_MAX_LIVE)
    except Exception:                                    # pragma: no cover
        return 260


def analyze(series: SoakSeries, *, cap: int,
            live_drift_tol: float = DEFAULT_LIVE_DRIFT_TOL,
            frame_time_tol: float = DEFAULT_FRAME_TIME_TOL,
            rss_growth_mb: float = DEFAULT_RSS_GROWTH_MB,
            rss_plateau_ratio: float = DEFAULT_RSS_PLATEAU_RATIO,
            warmup_frac: float = DEFAULT_WARMUP_FRAC,
            min_demolitions: int = DEFAULT_MIN_DEMOLITIONS,
            min_chunks_per: float = DEFAULT_MIN_CHUNKS_PER_DEMOLITION,
            ) -> Dict[str, Any]:
    warm = series.warm(warmup_frac)
    results = [
        check_demolitions(series, min_demolitions, min_chunks_per),
        check_live_cap(series, cap),           # full series: no grace period
        check_live_drift(warm, live_drift_tol),
        check_frame_time(warm, frame_time_tol),
        check_retirement(warm),
        check_memory(warm, rss_growth_mb, rss_plateau_ratio),
    ]
    return {
        "source": series.path,
        "samples": len(series),
        "samples_after_warmup": len(warm),
        "warmup_frac": warmup_frac,
        "frames": series.frames,
        "wall_seconds": round(series.wall_s, 2),
        "sim_seconds": round(series.sim_s, 2),
        "demolitions": series.demolitions,
        "cap": cap,
        "passed": all(r.passed for r in results),
        "failed": [r.name for r in results if not r.passed],
        "invariants": [
            {"name": r.name, "passed": r.passed, "headline": r.headline,
             "numbers": r.numbers}
            for r in results
        ],
    }


def render(v: Dict[str, Any]) -> str:
    L = ["=" * 74,
         "TUMBLE SOAK SUSTAINABILITY ANALYSIS",
         "=" * 74,
         f"  source        : {v['source']}",
         f"  samples       : {v['samples']} "
         f"({v['samples_after_warmup']} after {v['warmup_frac']:.0%} warm-up)",
         f"  horizon       : {v['frames']} frames, {v['wall_seconds']:.1f} s wall, "
         f"{v['sim_seconds']:.1f} s simulated",
         f"  demolitions   : {v['demolitions']}",
         f"  debris cap    : {v['cap']} (game.config.DEBRIS_MAX_LIVE)",
         "-" * 74]
    for inv in v["invariants"]:
        L.append(f"  [{'PASS' if inv['passed'] else 'FAIL'}] {inv['name']}")
        L.append(f"         {inv['headline']}")
    L.append("-" * 74)
    if v["passed"]:
        L.append("  VERDICT: PASS - debris stayed bounded and performance held.")
    else:
        L.append(f"  VERDICT: FAIL - {', '.join(v['failed'])}")
        L.append("  Do not relax a threshold to turn this green. Fix the cause")
        L.append("  or report the failure with these numbers.")
    L.append("=" * 74)
    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Assert Tumble's soak sustainability invariants.")
    p.add_argument("metrics", help="soak metrics JSONL (from soak_driver.py)")
    p.add_argument("--cap", type=int, default=None,
                   help="global live-debris cap (default: game.config."
                        "DEBRIS_MAX_LIVE)")
    p.add_argument("--live-drift-tol", type=float, default=DEFAULT_LIVE_DRIFT_TOL)
    p.add_argument("--frame-time-tol", type=float, default=DEFAULT_FRAME_TIME_TOL)
    p.add_argument("--rss-growth-mb", type=float, default=DEFAULT_RSS_GROWTH_MB)
    p.add_argument("--rss-plateau-ratio", type=float,
                   default=DEFAULT_RSS_PLATEAU_RATIO)
    p.add_argument("--warmup-frac", type=float, default=DEFAULT_WARMUP_FRAC)
    p.add_argument("--min-demolitions", type=int, default=DEFAULT_MIN_DEMOLITIONS)
    p.add_argument("--min-chunks-per-demolition", type=float,
                   default=DEFAULT_MIN_CHUNKS_PER_DEMOLITION)
    p.add_argument("--json", default=None, metavar="PATH",
                   help="also write the full verdict as JSON")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)

    series = SoakSeries.from_jsonl(a.metrics)
    verdict = analyze(
        series,
        cap=a.cap if a.cap is not None else default_cap(),
        live_drift_tol=a.live_drift_tol,
        frame_time_tol=a.frame_time_tol,
        rss_growth_mb=a.rss_growth_mb,
        rss_plateau_ratio=a.rss_plateau_ratio,
        warmup_frac=a.warmup_frac,
        min_demolitions=a.min_demolitions,
        min_chunks_per=a.min_chunks_per_demolition,
    )
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(verdict, fh, indent=2)
            fh.write("\n")
    if not a.quiet:
        print(render(verdict))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
