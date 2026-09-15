"""Central tunables plus display bootstrap for Tumble.

Everything here is plain data or a pure function, so it is importable with no
graphics context, no window and no ShowBase instance.
"""

from __future__ import annotations

# ---------------------------------------------------------------- simulation
GRAVITY_Z = -9.81
#: The sim always advances in whole steps of this size, regardless of how
#: fast (or slow) the renderer happens to be running.
FIXED_DT = 1.0 / 120.0
#: Upper bound on catch-up steps per frame, so a hitch cannot spiral.
MAX_STEPS_PER_FRAME = 8

# -------------------------------------------------------------------- player
PLAYER_RADIUS = 0.4
#: Height of the capsule's cylindrical mid-section (total height = H + 2R).
PLAYER_HEIGHT = 1.1
PLAYER_STEP_HEIGHT = 0.35
#: Eye/camera offset above the capsule's centre, in the player's local space.
EYE_HEIGHT = 0.7
SPAWN_POS = (0.0, -10.0, 1.5)

#: Forward-biased locomotion: you drive down the course, you don't dance.
FORWARD_SPEED = 8.0
BACK_SPEED = 2.5
STRAFE_SPEED = 3.5
SPRINT_MULTIPLIER = 1.6

MOUSE_SENSITIVITY = 0.12
PITCH_LIMIT = 85.0

# -------------------------------------------------------------------- course
COURSE_LENGTH = 200.0
COURSE_HALF_WIDTH = 24.0

# --------------------------------------------------------------- destruction
# Everything below tunes the runtime shatter/debris layer (`game.debris`).
# The numbers are measured on this stack, not guessed: see the docstrings in
# `game/debris.py` and the assertions in `tests/test_debris.py`.

#: Solver settings applied to every Bullet world the game builds. More
#: iterations than Bullet's default 10 is what keeps a 250-body rubble pile
#: from jittering itself awake forever; split impulse keeps the penetration
#: recovery out of the restitution channel so bounces stay proportional to
#: the configured restitution instead of exploding off a deep overlap.
BULLET_SOLVER_ITERATIONS = 12
BULLET_SPLIT_IMPULSE = True

#: kg/m^3 for debris mass. Mass is ALWAYS volume * this, never a constant:
#: that is what makes a big slab thud and a small shard skitter.
DEBRIS_DENSITY = 2400.0
#: Nothing lighter than this, so a sliver shard still has a stable inertia
#: tensor rather than being flung to infinity by a rounding error.
DEBRIS_MIN_MASS = 0.5

#: Convex-hull collision margin, as a fraction of the chunk's smallest half
#: extent, clamped to [MIN, MAX]. Bullet's flat 0.04 m default margin is
#: catastrophic here: chunks tile space EXACTLY, so a fixed margin makes every
#: neighbour start life 8 cm interpenetrated and the pile detonates on frame
#: one. Scaling the margin to the chunk kills that.
DEBRIS_MARGIN_FRACTION = 0.08
DEBRIS_MARGIN_MIN = 0.002
DEBRIS_MARGIN_MAX = 0.02

#: Restitution/friction are interpolated across the chunk size range: heavy
#: slabs are dead and grippy (thud), light shards are livelier and slipperier
#: (skitter). `size_blend` in debris.py is the 0 (heaviest) .. 1 (lightest)
#: parameter.
DEBRIS_RESTITUTION_HEAVY = 0.18
DEBRIS_RESTITUTION_LIGHT = 0.42
DEBRIS_FRICTION_HEAVY = 1.05
DEBRIS_FRICTION_LIGHT = 0.55
#: The ground's own restitution. Bullet multiplies the two, so a ground of 0
#: means nothing ever bounces no matter what the debris is set to.
DEBRIS_GROUND_RESTITUTION = 0.45
DEBRIS_GROUND_FRICTION = 0.95

#: Damping bleeds off the residual solver energy that keeps a rubble pile
#: twitching. Angular damping is much stronger than linear: spin is what
#: refuses to die in a contact-rich pile.
DEBRIS_LINEAR_DAMPING = 0.08
DEBRIS_ANGULAR_DAMPING = 0.34

#: How far above the ground a chunk's lowest vertex may sit and still count as
#: touching down. One fixed step of travel at settle speed, with slack for the
#: collision margin.
DEBRIS_GROUNDED_TOLERANCE = 0.12

#: Bullet's own sleep thresholds (m/s and rad/s) and the dwell time before it
#: deactivates a body.
DEBRIS_LINEAR_SLEEP_THRESHOLD = 0.30
DEBRIS_ANGULAR_SLEEP_THRESHOLD = 0.60
DEBRIS_DEACTIVATION_TIME = 0.8

#: Our own, stricter settle detector, which runs on top of Bullet's. Bullet
#: will happily leave a body nominally "active" forever in a dense pile; this
#: is what actually retires debris.
DEBRIS_SETTLE_LINEAR = 0.28
DEBRIS_SETTLE_ANGULAR = 0.70
DEBRIS_SETTLE_TIME = 0.7

#: Hard ceiling on *simulated* debris bodies. Never exceeded, ever - a shatter
#: that would blow the budget retires settled debris first and then drops its
#: smallest chunks.
DEBRIS_MAX_LIVE = 260
#: Settled debris is frozen to mass-0 static geometry: still visible, still
#: collidable, costs the solver nothing. This caps how many we keep around.
DEBRIS_MAX_FROZEN = 900
#: Frozen debris this far *behind* the player (in -Y) is despawned outright.
DEBRIS_DESPAWN_BEHIND = 70.0

#: Launch shaping. The blast is modelled as an impulse (N*s) that falls off
#: with distance from the impact point; the resulting speed is v = J/m, then
#: clamped, then jittered.
DEBRIS_IMPULSE = 9.0e4
DEBRIS_IMPULSE_FALLOFF = 1.2
DEBRIS_MIN_RADIUS = 0.6
DEBRIS_SPEED_MIN = 1.5
DEBRIS_SPEED_MAX = 30.0
#: How far the launch direction is randomised off the pure radial (unitless
#: perturbation added per axis before renormalising), plus a standing upward
#: bias so a collapse throws material up and out rather than only sideways.
DEBRIS_LAUNCH_SPREAD = 0.28
DEBRIS_LAUNCH_UP_BIAS = 0.22
DEBRIS_SPEED_JITTER = 0.30
#: Peak random spin, rad/s, per axis. Smaller chunks spin faster.
DEBRIS_SPIN_HEAVY = 5.0
DEBRIS_SPIN_LIGHT = 14.0


# ------------------------------------------------------------------- display
#: No window, no graphics pipe at all: the sim runs bare. Used by pytest.
#: Solver settings, shared by every display mode. These are global Bullet
#: settings rather than per-body ones, so they have to go through PRC.
PRC_BULLET = f"""
bullet-solver-iterations {BULLET_SOLVER_ITERATIONS}
bullet-split-impulse {'#t' if BULLET_SPLIT_IMPULSE else '#f'}
"""

PRC_HEADLESS = """
window-type none
audio-library-name null
notify-level-util error
"""

#: Real window for the human playtest.
PRC_WINDOWED = """
window-title Tumble
win-size 1280 720
framebuffer-multisample 1
multisamples 4
sync-video #t
"""


#: Renders into an offscreen buffer: a real graphics pipe and camera, but no
#: visible window. Lets the tests exercise the *render* path (and sample the
#: resulting pixels) on a display-less server.
PRC_OFFSCREEN = """
window-type offscreen
audio-library-name null
win-size 320 240
notify-level-display error
"""


def display_prc(headless: bool, offscreen: bool = False) -> str:
    """Return the PRC fragment appropriate for the requested mode.

    Bullet's solver settings ride along in every mode: debris behaviour must
    not depend on whether a window happened to open.
    """
    if headless:
        return PRC_BULLET + PRC_HEADLESS
    if offscreen:
        return PRC_BULLET + PRC_OFFSCREEN
    return PRC_BULLET + PRC_WINDOWED


_bullet_config_loaded = False


def ensure_bullet_config() -> bool:
    """Load Bullet's global solver settings into Panda, exactly once.

    Bullet reads ``bullet-solver-iterations`` and ``bullet-split-impulse``
    when a :class:`BulletWorld` is constructed, so this has to run *before*
    the first world exists. `PhysicsWorld.__init__` calls it, which means a
    test that never touches the display layer still gets the same solver the
    game ships with. Returns True the first time it actually loaded.
    """
    global _bullet_config_loaded
    if _bullet_config_loaded:
        return False
    from panda3d.core import loadPrcFileData

    loadPrcFileData("tumble-bullet", PRC_BULLET)
    _bullet_config_loaded = True
    return True


def bootstrap_display(headless: bool, offscreen: bool = False) -> str:
    """Load display config into Panda.

    Must be called *before* a ShowBase is constructed. Returns the PRC text
    that was applied (handy for tests and for logging).
    """
    from panda3d.core import loadPrcFileData

    prc = display_prc(headless, offscreen)
    loadPrcFileData("tumble-display", prc)
    return prc
