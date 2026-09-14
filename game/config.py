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

# ------------------------------------------------------------------- display
#: No window, no graphics pipe at all: the sim runs bare. Used by pytest.
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
    """Return the PRC fragment appropriate for the requested mode."""
    if headless:
        return PRC_HEADLESS
    if offscreen:
        return PRC_OFFSCREEN
    return PRC_WINDOWED


def bootstrap_display(headless: bool, offscreen: bool = False) -> str:
    """Load display config into Panda.

    Must be called *before* a ShowBase is constructed. Returns the PRC text
    that was applied (handy for tests and for logging).
    """
    from panda3d.core import loadPrcFileData

    prc = display_prc(headless, offscreen)
    loadPrcFileData("tumble-display", prc)
    return prc
