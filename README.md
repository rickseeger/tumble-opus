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
| `Esc`          | quit                            |

You are on the ground with real gravity — no flying, no noclip. Forward is
deliberately the fastest direction: the course is meant to be driven down.

## Other commands

```
./run.sh --headless     # boot the sim with no window, print state, exit
./run.sh --test         # run the test suite
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
game/app.py        ShowBase host: visuals, lights, camera, input binding
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

### Not verified automatically

There is no vision check in this repo, so the following need a human eye:
actual visual appearance, lighting and colour, mouse-look feel and
sensitivity, and field-of-view comfort. The geometry is placeholder grey
boxes by design at this stage.
