# Node 20 — the sustained soak: what held, and the one thing that did not

Two long soaks, 240 demolitions each, same seed, same 28,681-frame horizon,
differing in exactly one flag. Five of six sustainability invariants held in
both. The sixth — memory — failed in the default configuration and passed in
the other, which is what let the leak be attributed to a single holder rather
than guessed at.

**The node-15 debris budget is sound.** The cap was never exceeded, live
debris did not drift, retirement fired in every quarter, and frame time was
flat. The leak this soak found is not in debris at all; it is in course
authoring, and the fix already exists in the codebase.

---

## 1. The exact commands

Launch (returns in ~0.01 s; the soak outlives the shell that started it):

```bash
# Baseline: the driver's default course behaviour
./tools/soak_launch.sh soak_runs/node20_long \
    --max-demolitions 240 --max-seconds 1500 --max-frames 400000 \
    --destroy-every 120 --sample-every 60 --sample-seconds 2.0 \
    --progress-every 600 --out -

# Controlled A/B: identical, plus --release-chunks
./tools/soak_launch.sh soak_runs/node20_release \
    --max-demolitions 240 --max-seconds 1500 --max-frames 400000 \
    --destroy-every 120 --sample-every 60 --sample-seconds 2.0 \
    --progress-every 0 --release-chunks --out -
```

Poll (short, foreground-safe, never blocks):

```bash
.venv/bin/python tools/soak_driver.py --status soak_runs/node20_long
```

Analyse (the judgement; exit 0 only if every invariant holds):

```bash
.venv/bin/python tools/analyze_soak.py soak_runs/node20_long/metrics.jsonl
.venv/bin/python tools/analyze_soak.py soak_runs/node20_release/metrics.jsonl
.venv/bin/python -m pytest tests/test_soak_sustainability.py
```

Both runs terminated cleanly on `reason=max-demolitions` — they hit the
demolition bound, not a timeout and not the watchdog.

## 2. What was achieved

| | baseline (`node20_long`) | with `--release-chunks` |
|---|---|---|
| demolitions | **240** (target ≥ 60) | **240** |
| frames stepped | 28,681 | 28,681 |
| simulated time | **478.0 s** | 478.0 s |
| wall time | 168.4 s (2.84× real time) | 167.6 s |
| chunks spawned | 32,705 (136.3/demolition) | 32,705 |
| chunks despawned | 32,299 | 32,299 |
| retirements total | 46,215 | 46,215 |
| peak live debris | **260** | **260** |
| RSS | 104.3 → **390.6 MB** | 111.6 → **114.6 MB** |

The physics counters are *identical* across the two runs. That is what makes
the memory column an attribution and not a correlation.

## 3. The invariants, with numbers

Thresholds live in `tools/analyze_soak.py`, each with the reason it has the
value it has. The live-debris cap is checked over the **full** series
including warm-up — a hard ceiling gets no grace period. The drift and
degradation comparisons discard the first 10% of samples, which is the
conservative direction: it makes them harder to pass, not easier.

### Baseline — `soak_runs/node20_long/analysis.txt` (exit 1)

```
  [PASS] demolitions
         240 structures demolished (need >= 60), 32705 chunks spawned
         = 136.3/demolition (need >= 20)
  [PASS] live_cap
         peak live 260, peak stepped 260, cap 260 -> never exceeded
  [PASS] live_no_drift
         mean live Q1 257.7 -> Q4 257.7 (ratio 1.000, tolerance 1.05)
  [PASS] frame_time
         mean frame_ms Q1 8.061 -> Q4 8.328 (ratio 1.033, tolerance 1.25);
         p95 Q1 15.542 -> Q4 15.184
  [PASS] retirement
         retired_total 4670 -> 46215; gain per quarter
         [10464, 10408, 10376, 10278] (each must be > 0)
  [FAIL] memory
         RSS 136.3 -> 390.6 MB (peak 390.6, growth +254.2 MB over 28681
         frames); acquisition rate 9.91 -> 9.76 MB/1k frames (ratio 0.98);
         absolute growth 254.2 MB >= bound 250 MB; no plateau: 2nd-half rate
         is 0.98x the 1st-half rate (bound 0.50x) - growth is not decelerating

  VERDICT: FAIL - memory
```

### With `--release-chunks` — `soak_runs/node20_release/analysis.txt` (exit 0)

```
  [PASS] demolitions   240 structures, 136.3 chunks/demolition
  [PASS] live_cap      peak live 260, peak stepped 260, cap 260 -> never exceeded
  [PASS] live_no_drift mean live Q1 257.7 -> Q4 257.7 (ratio 1.000)
  [PASS] frame_time    Q1 8.367 -> Q4 8.633 ms (ratio 1.032, tolerance 1.25)
  [PASS] retirement    4670 -> 46215; per quarter [10464, 10408, 10376, 10278]
  [PASS] memory        RSS 111.6 -> 114.6 MB (growth +3.0 MB over 28681 frames);
                       acquisition rate 0.22 -> 0.01 MB/1k frames (ratio 0.04)

  VERDICT: PASS - debris stayed bounded and performance held.
```

Note the memory row is not merely *smaller*: the acquisition rate collapses
from 0.22 to 0.01 MB per 1000 frames. That is a curve reaching an asymptote,
which is what "bounded" has to mean. The baseline's 9.91 → 9.76 is a straight
line, and a straight line does not become bounded by being observed for a
shorter time.

## 4. Diagnosis: where the 286 MB went

`tools/diagnose_soak_rss.py` (committed) probes a real soak in-process and
attributes growth to a holder. Over 12,000 frames, RSS 115 → 227 MB:

| suspect | verdict |
|---|---|
| Panda `TransformState` interned table | **not it** — 487 → 453 states. `reclaim_interned_states()` from the earlier nodes is working. |
| Panda `RenderState` table | **not it** — 8 → 8. |
| glibc arena fragmentation | **not it** — `malloc_trim(0)` returned 1.5 MB of 227. |
| Python-side retention | **this** — gc objects 59,393 → 94,914, at 9.24 KB and 2.88 objects per chunk spawned. |

Per-type deltas name the objects: `Chunk` +27,360 and `DebrisBody` +7,033,
while the debris field itself stayed flat (`live` 260 → 230, `all_bodies` 366).
A direct referrer walk at frame 6,000 settles it:

```
destructibles in course: 55
chunks pinned by app.destructibles: 16340
destructibles still holding a fracture result: 55   <-- all 55, every one demolished
Chunk objects alive total: 16340                    <-- every live Chunk is pinned here
DebrisBody alive: 7191   field.all_bodies: 358      <-- field is clean; 6833 pinned elsewhere
bodies pinned by bounded event history: 3233
```

**Every single `Chunk` alive in the process is pinned by a demolished
structure still holding its spent `FractureResult`.** A structure's fracture
result is load-time data, read exactly once on the frame it shatters. After
that it is dead weight — and the soak driver's `place_structure()` appends
each new structure to `app.destructibles` and never releases it, so an endless
course retains every descriptor set it ever authored.

`Destructible.release_chunks()` already exists for precisely this and
documents the cost: **1.22 MB per structure**.

```
predicted: 1.22 MB x 240 structures = 293 MB
observed : 286.3 MB baseline growth - 10.4 MB release growth = 275.9 MB delta
           (baseline total growth 286.3 MB)
```

Within 6% of the documented figure. The diagnosis is quantitative, and
`test_leak_matches_the_documented_per_structure_cost` asserts it stays so.

The secondary holder — 3,233 `DebrisBody` shells pinned by the bounded
24-event `ShatterEvent` history — is **not** a leak: it is bounded by
`DEBRIS_EVENT_HISTORY`, and since node 16's `DebrisBody.release()` each shell
is ~150 bytes, not ~7 KB. It is flat, and it stays flat.

## 5. The honest verdict

- **Debris budgeting (node 15) is proven.** Over 240 demolitions and 32,705
  spawned chunks, the simulated set never once exceeded its 260-body cap,
  never drifted, and never slowed the frame down. Retirement fired 46,215
  times, in every quarter. That is the thing this node was asked to prove, and
  it holds.
- **The default course configuration leaks 1.22 MB per structure demolished.**
  Not in debris — in course authoring. A five-structure shipping course pays
  6 MB and nobody notices; an endless campaign grows without bound.
- **The fix is one call, already written.** Whoever owns the endless-campaign
  path should call `Destructible.release_chunks()` when a structure finishes
  collapsing. `--release-chunks` proves the result: +3.0 MB over 478 simulated
  seconds, with physics unchanged to the body.

I did not relax a threshold to turn the red run green. The failing
configuration is committed alongside the passing one, the failure is asserted
as a known defect with a ratchet that only permits improvement, and
`test_memory_invariant_is_still_failing` will break the day the leak is fixed
so that the waiver cannot outlive the bug it documents.

## 6. Not verified here

No visual inspection was performed — this is a headless instrumentation run.
Rendering, glow, debris readability and audio remain for the human playtest.
The soak proves the simulation stays bounded; it says nothing about how the
rubble looks.

## 7. Artifacts

```
tools/analyze_soak.py                      the invariants, in code
tools/diagnose_soak_rss.py                 the attribution probe
tests/test_soak_sustainability.py          26 tests: analyser + both real logs
soak_runs/node20_long/metrics.jsonl        479 samples, baseline
soak_runs/node20_long/{summary,verdict}.json, analysis.txt, soak.log
soak_runs/node20_release/metrics.jsonl     479 samples, --release-chunks
soak_runs/node20_release/{summary,verdict}.json, analysis.txt, soak.log
soak_runs/diagnose_probe.jsonl             per-1000-frame attribution probe
```
