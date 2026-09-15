# Tumble

A first-person 3D game built on Panda3D + Bullet physics. Linux.

This repo is the **project foundation**: a ground-based first-person player
who drives forward along a linear course, real gravity and collision, a few
placeholder box structures to bump into, and a headless test harness so every
later feature has an automated path to verify against.

## Play it

```
git clone git@github.com:rickseeger/tumble-opus.git
cd tumble-opus
./run.sh
```

That's it. `run.sh` creates the virtualenv, installs the dependencies on first
run, and launches the game. Later runs skip straight to launching.

### Controls

| Input          | Action                          |
| -------------- | ------------------------------- |
| `W` `A` `S` `D`| move (forward is much faster)   |
| Mouse          | look                            |
| `Shift`        | sprint                          |
| `Space`        | jump                            |
| `F`            | strike the nearest structure    |
| `Esc`          | quit                            |

You are on the ground with real gravity — no flying, no noclip. Forward is
deliberately the fastest direction: the course is meant to be driven down.

## Other commands

```
./run.sh --headless              # boot the sim with no window, print state, exit
./run.sh --headless --demolish   # same, but blow up each structure on approach
./run.sh --test                  # run the test suite
```

`--headless` is the useful one on a server with no display: it builds the
world, drives the player forward for 300 frames and reports how far it got.

## Requirements

- Linux
- Python 3.8+ with `venv` (on Debian/Ubuntu: `sudo apt install python3-venv`)
- A GPU/driver capable of OpenGL for the windowed mode (headless mode needs none)

Dependencies (`panda3d`, `numpy`, `pytest`) install into `.venv/` and are
pinned loosely in `requirements.txt`.

**Python version note:** verified working on Python **3.14.4** (Ubuntu 26.04)
with Panda3D 1.10.16 — upstream publishes a `cp314` manylinux wheel, so no
pinned older interpreter is needed. Panda3D ships wheels back to 3.8, so
`run.sh` works on older distros unchanged.

## Layout

```
main.py            entrypoint (windowed or --headless)
run.sh             one-command install + launch
game/config.py     tunables; PRC display config for windowed vs headless
game/physics.py    Bullet world, fixed timestep, body authoring  (no render deps)
game/player.py     FPS capsule character + InputState             (no render deps)
game/world.py      course authoring: ground plane + box towers    (no render deps)
game/fracture.py   fracture generation library: spec -> convex chunks (pure data)
game/structures.py destructible placement: static proxies + pre-generated chunks
game/debris.py     runtime shatter + debris physics: chunks -> Bullet bodies
game/damage.py     accumulated damage + integrity thresholds: the destruction TRIGGER
game/render_debris.py the 1A glowing-wireframe treatment (render layer only)
game/app.py        ShowBase host: visuals, lights, camera, input binding
tools_fracture_report.py  headless per-archetype fracture summary table
tools_debris_demo.py      headless shatter demo + stepping benchmark
tests/             pytest suite, runs with no display
```

The split matters: `physics.py`, `player.py` and `world.py` never import
`ShowBase` or the render graph, so the simulation can be constructed and
stepped with no window at all. `app.py` is the only module that knows a
window might exist, and every render call in it is guarded — the same class
boots cleanly headless.

## Design notes

**Fixed timestep.** The sim only ever advances in whole 1/120 s steps.
`PhysicsWorld.advance(dt)` accumulates wall-clock time and runs as many whole
steps as fit, capped at 8 per frame so a hitch can't spiral into a death
loop. Framerate changes how many steps run per frame, never the step size, so
physics behaviour doesn't drift with the renderer.

**Continuous collision detection is on for dynamic bodies.** Measured on this
stack: without CCD a body moving at 200 m/s passes straight through the ground
plane (it ends up at z=+106, or below the floor entirely). With CCD it rests
correctly at every speed tested up to 500 m/s. Destruction debris moves fast,
so this is load-bearing rather than decorative.

**The player is a `BulletCharacterControllerNode`** — a swept kinematic
capsule. Vertical motion comes only from Bullet, so the player is subject to
gravity and cannot tunnel through the ground or a static structure. Input is
funnelled through a plain `InputState` dataclass, which is what lets tests
drive the character with no keyboard and no window.

## Fracture library

`game/fracture.py` is a standalone, engine-free module: it imports nothing from
Panda3D or Bullet, so it can be tested and profiled headlessly and reused by
the runtime destruction layer. It takes a `StructureSpec` (one or more
axis-aligned `Block`s) plus an integer seed and pre-generates, at load time, the
full list of convex `Chunk` descriptors that tile that structure's volume.

```python
from game import fracture

result = fracture.fracture(fracture.tower_spec(), seed=7)
for chunk in result.chunks:
    chunk.center          # centroid in structure-local space
    chunk.vertices        # convex hull vertices, relative to the centroid
    chunk.edges           # explicit edge list - the glowing wireframe draws these
    chunk.faces           # vertex-index polygons, outward wound
    chunk.volume, chunk.mass, chunk.half_extents
    chunk.launch_dir      # unit outward hint from the structure's fracture origin
```

Four archetypes ship by default — `tower_spec`, `slab_spec`, `arch_spec`,
`cluster_spec` — but the spec dataclass is general, so new structures are just
data.

**How it fractures.** Each block is carved by recursive half-space splitting
with jittered, *oblique* planes. Cut placement is deliberately biased: a
sizeable minority of cuts run across a piece's shortest axis (which yields long
slabs and shards rather than cubes) and land well off-centre (which spreads the
size distribution), and the next piece to split is chosen with probability
proportional to its volume rather than always being the largest — so a few big
slabs survive alongside a swarm of small debris. Because every cut is a plane
through a convex solid, the result is exactly space-filling: chunks do not
overlap and volume is conserved to floating-point round-off, not to a
hand-waved tolerance.

Measured output at the shipped budgets (seed 20260914, `tools_fracture_report.py`):

| structure | chunks | source vol | chunk vol | vol err | min/med/max chunk vol | max:min | median aspect | gen |
|---|---|---|---|---|---|---|---|---|
| tall_tower | 380 | 3840.0 | 3840.0 | 1.2e-16 | 0.36 / 7.00 / 72.66 | 204:1 | 3.28 | 69 ms |
| wide_slab | 320 | 7128.0 | 7128.0 | 0.0 | 0.93 / 16.02 / 126.06 | 136:1 | 2.74 | 53 ms |
| arch | 260 | 1830.0 | 1830.0 | 2.5e-16 | 0.27 / 5.44 / 55.98 | 209:1 | 2.56 | 47 ms |
| block_cluster | 240 | 1179.0 | 1179.0 | 1.9e-16 | 0.17 / 3.93 / 45.18 | 273:1 | 2.63 | 40 ms |

All four together pre-generate in about 190 ms — comfortably a load-time cost,
never a shatter-time one. `max_chunks` is a hard ceiling that is never exceeded,
and `fracture()` refuses a budget smaller than the block count rather than
silently dropping geometry.

**Spec validation.** `fracture()` calls `spec.validate()` first, which rejects
a structure whose source blocks *interpenetrate*. Overlapping blocks are a
double fault: the shared region is counted twice in `spec.volume`, and the
chunks carved from each block occupy the same space. Blocks may touch
face-to-face; only shared volume is an error. Use
`spec.overlapping_block_pairs()` to inspect a spec directly.

**Non-overlap is proven exactly, not sampled.** `convex_pair_overlap()` is a
separating-axis test over both solids' face normals and edge-edge cross
products. Monte-Carlo point sampling can miss a thin interpenetration; the SAT
check cannot, so the suite uses it for the real disjointness proof and keeps
sampling only as a coverage measure.

Print the table yourself:

```
.venv/bin/python tools_fracture_report.py --seed 7
```

`tests/test_fracture.py` asserts determinism (byte-identical output for the
same spec+seed, stable across `PYTHONHASHSEED`, and genuinely different chunk
centres across seeds), volume conservation, containment inside both the
structure bounds and the source block (so the arch's opening stays open),
zero sampled interior overlap, full interior coverage, closed-manifold geometry
(V - E + F = 2 on every chunk), size spread of at least one order of magnitude
with all three log-size bands populated, aspect-ratio spread, budget bounds,
and per-structure generation time.

## Runtime destruction

`game/debris.py` is the layer between the pre-generated chunk descriptors and
the live simulation. One public verb:

```python
from game.debris import DebrisField

field = DebrisField(physics)
event = field.shatter(structure, impact_point=(0, 40, 6), impulse=9e4)
event = field.demolish(destructible)      # proxies out, debris in, one call
...
field.update(dt, player_y=player.pos[1])  # once per frame - this retires debris
```

Each chunk becomes an independent `BulletConvexHullShape` rigid body at its
own local offset, with **mass derived from its volume** (so a 56 t slab and a
0.4 t shard behave nothing alike), an outward impulse from the blast point
with distance falloff, randomised direction spread and a random 3-axis spin.
It then falls under real gravity, bounces, rolls and settles.

`bodies_near()`, `contact_test()` and `body_for_node()` are the hooks the
later damage node asked for; frozen rubble stays attached to the world, so
ray tests and dynamic-body contacts still find it.

**An intact structure is not 380 sleeping bodies.** `game/structures.py` gives
each destructible a handful of static box proxies while it is standing, and
`demolish()` removes them on the same frame its debris appears. Miss that and
the player is walled off by a tower that has visibly collapsed — there is a
test for exactly that.

### The tuning, and how it was found

Three findings that were measured on this stack, not guessed:

1. **The collision margin has to scale with the chunk.** Bullet's default
   convex margin is a flat 0.04 m. The fracture library tiles a structure's
   volume *exactly*, so with a flat margin every chunk starts life ~8 cm
   interpenetrated with each neighbour and the pile detonates on frame one.

2. **Restitution is a product.** Bullet combines restitution
   multiplicatively, so with the ground plane left at its default 0, *nothing
   bounces* regardless of the debris setting — measured, a 9.4 m/s impact
   rebounded at 0.46–0.48 m/s for every body restitution from 0.0 to 0.6,
   i.e. pure penetration-recovery noise, completely decoupled from the knob.
   `DebrisField.prepare_ground()` gives the ground a real restitution.

3. **Glow cannot come from an over-bright colour.** `game/render_debris.py`
   originally multiplied vertex colour by 1.85 and called that glow. Panda
   *clamps* vertex colour and colour scale: the same chunk rendered at scale
   1.0, 1.85 and 3.0 produced byte-identical frames (mean brightness 0.030404
   for all three). The multiplier did nothing. The glow is now a thick dim
   halo pass under a thin bright core pass, additively blended — which
   measurably works: differenced against a baseline frame, the halo takes one
   chunk's wireframe from 1119 to 2678 brightened pixels (2.39x) and 1.69x
   total added light.

### Settling is real physics

Settled debris is frozen to mass-0 static geometry, which zeroes its
velocity — so "everything has near-zero velocity and is asleep" would be
trivially true even if the physics jittered forever. So
`tests/test_settling_is_real.py` runs with **the freeze disabled** and
requires pure Bullet to bring the pile to rest unaided. It does, for every
archetype: a 150-chunk cluster collapse decays from ~10 m/s peak at t=2 s to
exactly 0 m/s and 0 rad/s with every body deactivated by t≈24 s, and the
worst case — a 380-chunk, 8589 t tower capped to 260 bodies — reaches exact
zero at t≈25 s. Kinetic energy reaches 0 J with no body removed, everything
resting above the ground plane.

That last number is worth stating plainly, because it bit the demo: the
headless report used to stop at 15 s and print `every body at rest: False`
for the tower. That looked like broken physics and was not — it was an
unfinished run. The demo now defaults to 30 s, *measures* when rest was
reached, and prints it (`came to rest at t=20.10 s`).

The freeze earns its keep on *promptness and cost*, not on rescuing divergent
physics: it takes those ~20 s of a few hundred solver-resident bodies (which
the next collapse would wake straight back up) down to zero solver cost.

### Reaping debris

`DebrisField.reap()` is the lifecycle mechanism, and it is deliberately
*untuned* — every criterion is off unless the caller asks for it, because
choosing the actual budget is the game's call, not the physics core's:

```python
field.reap(max_age=20.0)     # older than 20 simulated seconds
field.reap(behind_y=cutoff)  # driven past and never to be looked at again
field.reap(cap=200)          # hard ceiling, oldest debris goes first
```

Age is measured in **simulated** seconds against `field.clock`, which
`update(dt)` advances — so reaping behaves identically at any frame rate, and
is reproducible in a test. `update(dt, max_age=...)` runs the age pass inline.
`live_count`, `frozen_count` and `total_bodies` reflect a reap immediately, and
the bodies are genuinely detached from the Bullet world rather than merely
dropped from a list.

### Performance budget

`DEBRIS_MAX_LIVE = 260` is a hard ceiling on dynamic debris bodies, never
exceeded. A shatter that would blow it first freezes genuinely settled debris,
then despawns the oldest, and only then declines to spawn a structure's
smallest chunks — the budget goes to the biggest pieces, because losing a
shard is invisible and losing the corner slab is not. Frozen rubble more than
70 m behind the player is despawned outright.

Measured on this machine (`tools_debris_demo.py --benchmark`):

| scenario | bodies | per step | budget | real-time factor |
|---|---|---|---|---|
| full tower's debris | 260 | 2.96 ms | 8.33 ms | **2.81x** |
| 3 simultaneous collapses | 260 (capped) | 2.81 ms | 8.33 ms | **2.97x** |
| whole course, 5 structures in sequence | 260 (capped) | 2.93 ms | 8.33 ms | **2.85x** |

Pre-generating all five placed structures (1440 chunks) costs ~228 ms, paid
once at load. A shatter itself builds 240–260 bodies in 6–9 ms, so a
demolition does not stall the frame it happens on.

Watch it happen, phase by phase, with no display:

```
.venv/bin/python tools_debris_demo.py --structure cluster
.venv/bin/python tools_debris_demo.py --benchmark
```

```
         phase   t(s)  live  frozen  active         KE(J)   |v|max   |w|max     minZ
        launch   0.00   240       0     240     6423025.0   28.218   18.475    0.000
 first-impacts   1.00   237       3     237    12723408.6   11.028    5.510    0.007
      tumbling   3.00   178      62     178      486053.4    6.139    4.469    0.014
       rolling   6.00   128     112     128      264321.1    3.129    1.520    0.014
      settling  10.00   124     116     124        3724.1    0.425    0.439    0.014
       at-rest  15.00   122     118       0           0.0    0.000    0.000    0.014
```


## Damage and the destruction trigger

`game/damage.py` is the decision layer. Before it, the game had two halves
that never met: the debris layer knew *how* to turn a structure into rigid
bodies, and nothing ever decided that it *should* — destruction only happened
because a test, a demo or a keypress asked for it directly.

`DamageSystem` holds one damage tally per placed structure, with an integrity
threshold proportional to the structure's authored volume
(`STRUCTURE_INTEGRITY_PER_M3`, floored at `STRUCTURE_INTEGRITY_MIN`), so the
60 m tower is genuinely harder to bring down than a low cluster:

```python
app.strike((0.0, 30.0, 4.0), amount=500.0, radius=14.0)   # the gameplay path
```

`apply_damage` measures distance to each structure's *volume*, not its
centroid — a blast against the face of a 60 m tower is a direct hit, not a
30 m near miss — and lerps the damage from full strength at the impact point
down to `DAMAGE_FALLOFF_AT_EDGE` at the rim. When a tally crosses its
threshold the structure is demolished through the game's real path:
`TumbleApp.demolish` → `DebrisField.demolish`, which pulls the static
collision proxies out of the Bullet world *and* the scene graph and spawns the
pre-generated chunk set in their place, inheriting the structure's world
transform.

Three properties are deliberate:

* **Damage is only ever caused by an explicit call.** Nothing watches
  contacts. Debris raining onto the next tower must not demolish it by
  accident, and a collapse must not chain down the course — there is a test
  for exactly that.
* **Destruction resolves before physics advances.** Damage enqueues;
  `step_frame` calls `damage.resolve()` *before* `physics.advance()`, so a
  structure damaged during a frame is rubble before a single substep of that
  frame runs. The intact proxies and their own debris never coexist for one
  tick — which is what stops the chunks spawning interpenetrated with the
  boxes they replace. `apply_damage` resolves immediately by default too; the
  deferral exists so damage can safely be raised from inside a physics
  callback, where mutating the world mid-step is illegal.
* **Freshly spawned debris is queryable immediately.** `damage.debris_near()`
  and `damage.debris_in_contact()` are the hooks the later "debris hurts the
  player" node reads, and the tests assert the new bodies show up there on the
  frame they spawn — not one frame later.

Direct `demolish()` calls (campaign script, tests, the debug key) call
`damage.note_destroyed()`, so the tally can never claim a pile of rubble is
still standing at 0 % damage.

No budgeting lives in this module. How much debris the world may hold is
`game/debris.py`'s business.

## Tests

```
./run.sh --test
```

The suite asserts on numbers, not screenshots. Among other things it checks
that a dropped body's fall matches the free-fall equation to within 8%, that
bodies come to rest at exactly their half-height and stay there, that fast
bodies don't tunnel, that the player can't sink through the floor over 10 s of
walking, that walking into a static wall stops the player short of it, that
forward input advances position along the *heading* rather than a world axis,
that the player never rises without a jump input, that the eye/camera offset
holds while moving and turning, and that the app boots headless and shuts
down cleanly.

`tests/test_damage_destruction.py` covers the damage trigger end to end
through the real headless game loop: that damage below the threshold destroys
nothing, that crossing it removes every collision proxy from both the Bullet
world and the scene graph while spawning exactly `chunk_count` debris bodies
inside the original structure's bounding volume, that those bodies launch with
varied non-zero velocity and spin, that after 45 simulated seconds they have
fallen, settled above the ground plane with no tunnelling and no non-finite
coordinates, that they are reachable through the damage system's query path on
the frame they spawn, that a collapse does not chain into its neighbours, and
that a whole burst down the course runs through `step_frame` without
exception. Run just that file with:

```
./run.sh --test tests/test_damage_destruction.py
```

### Not verified automatically

There is no vision check in this repo, so the following need a human eye:

- **Whether the 1A glowing wireframe actually looks good.** The tests prove
  the geometry is drawn, carries additive/unlit/no-depth-write state, and
  measurably brightens real rendered frames. Line thickness, palette, halo
  strength and whether a 240-piece collapse reads as spectacular or as visual
  noise are Rick's call at playtest.
- **Whether a collapse *feels* right** — impulse strength, how far debris
  flies, how long it tumbles before settling. The physics is correct and
  convergent; "satisfying" is a judgement.
- Mouse-look feel and sensitivity, field-of-view comfort, and the rubble
  pile's readability as cover or as an obstacle (a settled cluster collapse
  leaves ~43 chunks above 1 m inside the lane).
- The non-debris course geometry is still placeholder grey boxes by design.
