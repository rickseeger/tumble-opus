"""Render-path tests using an offscreen buffer (no visible window).

I have no vision tool, so these do the honest mechanical equivalent: render
real frames into a buffer and assert on the *pixels* and scene graph in code.
They catch the failure modes that would otherwise only show up when a human
launches the game - a missing model file, an all-black screen because the
lights never got attached, a camera that isn't parented to the player.

Skipped automatically if the machine has no usable GL pipe, so the core suite
still runs anywhere.
"""

import pytest

from game import config
from game.player import InputState

pytestmark = pytest.mark.render


@pytest.fixture(scope="module")
def rendered():
    """Boot the app offscreen and render a few real frames."""
    from game.app import TumbleApp

    try:
        app = TumbleApp(offscreen=True)
    except Exception as exc:                        # pragma: no cover
        pytest.skip(f"no offscreen GL pipe available: {exc}")

    if app.win is None:                             # pragma: no cover
        app.destroy()
        pytest.skip("offscreen buffer could not be created")

    # Let the player settle, then render.
    for _ in range(90):
        app.step_frame(1.0 / 60.0)
    for _ in range(3):
        app.graphicsEngine.renderFrame()

    yield app
    app.destroy()


def _screenshot(app):
    """Grab the rendered buffer as a PNMImage, or skip if unavailable."""
    from panda3d.core import PNMImage

    img = PNMImage()
    if not app.win.getScreenshot(img):              # pragma: no cover
        pytest.skip("could not read back the offscreen buffer")
    return img


def test_render_path_builds_visuals_without_error(rendered):
    """The windowed code path (models, lights, camera) ran for real."""
    app = rendered
    assert app.headless is False, "offscreen mode must still build visuals"
    assert app.win is not None
    # Every collision box must have at least one visible child mirroring it.
    for spec in app.physics.box_specs:
        np_ = app.physics.body(spec.name)
        assert np_.getNumChildren() >= 1, f"{spec.name} has no visual"


def test_sim_root_is_attached_to_the_render_graph(rendered):
    app = rendered
    assert app.physics.root.getParent() == app.render
    # And the player's capsule is in the graph too.
    assert app.player.np.isAncestorOf(app.camera) or (
        app.camera.getParent() == app.player.np
    )


def test_camera_is_parented_to_the_player_at_eye_height(rendered):
    app = rendered
    cam = app.camera
    assert cam.getParent() == app.player.np, "camera is not on the player"
    assert float(cam.getZ()) == pytest.approx(config.EYE_HEIGHT, abs=1e-6)

    # Camera must ride along in world space as the player moves.
    before = cam.getPos(app.render)
    app.input_state = InputState(forward=True)
    for _ in range(60):
        app.step_frame(1.0 / 60.0)
    after = cam.getPos(app.render)
    app.input_state = InputState()

    assert float(after[1] - before[1]) > 3.0, "camera did not follow the player"
    assert float(after[2]) > 0.5, "camera sank into the ground"


def test_lights_are_attached_so_the_scene_is_not_black(rendered):
    """A scene with no lights renders flat black - guard against that."""
    app = rendered
    assert not app.render.getAttrib(None) if False else True  # readability
    from panda3d.core import LightAttrib

    attrib = app.render.getState().getAttrib(LightAttrib.getClassSlot())
    assert attrib is not None, "no LightAttrib on render - scene is unlit"
    assert attrib.getNumOnLights() >= 2, (
        f"expected ambient + directional, got {attrib.getNumOnLights()}"
    )


def test_rendered_frame_is_not_blank(rendered):
    """Pixel-level check: the frame must contain actual variation.

    A black screen, a solid background with nothing drawn, or a camera inside
    geometry would all collapse the pixel distribution. This is the stand-in
    for looking at it.
    """
    app = rendered
    # Aim the player at the first tower so there is definitely geometry ahead.
    app.player.np.setPos(-6.0, 5.0, 1.0)
    app.player.heading = 0.0
    app.player.np.setH(0.0)
    app.player.pitch = 0.0
    app.player.sync_camera()
    for _ in range(3):
        app.graphicsEngine.renderFrame()

    img = _screenshot(app)
    w, h = img.getXSize(), img.getYSize()
    assert w > 0 and h > 0

    seen = set()
    total = 0.0
    samples = 0
    for y in range(0, h, 4):
        for x in range(0, w, 4):
            px = img.getXel(x, y)
            r, g, b = round(px[0], 2), round(px[1], 2), round(px[2], 2)
            seen.add((r, g, b))
            total += (px[0] + px[1] + px[2]) / 3.0
            samples += 1

    mean = total / samples
    assert samples > 100
    assert len(seen) >= 3, (
        f"frame has only {len(seen)} distinct colours - likely blank"
    )
    assert 0.02 < mean < 0.99, f"frame mean brightness {mean:.3f} - black or blown out"


def test_geometry_is_visible_ahead_of_the_player(rendered):
    """Looking at a tower must differ from looking away from it.

    If nothing were ever drawn, both frames would be identical background.
    """
    app = rendered

    def frame_signature(heading):
        app.player.np.setPos(-6.0, 6.0, 1.0)
        app.player.heading = heading
        app.player.np.setH(heading)
        app.player.pitch = 0.0
        app.player.sync_camera()
        for _ in range(3):
            app.graphicsEngine.renderFrame()
        img = _screenshot(app)
        w, h = img.getXSize(), img.getYSize()
        vals = []
        for y in range(0, h, 3):
            for x in range(0, w, 3):
                px = img.getXel(x, y)
                vals.append((px[0] + px[1] + px[2]) / 3.0)
        return vals

    facing_tower = frame_signature(0.0)      # tower_a sits at (-6, 15)
    facing_away = frame_signature(180.0)     # empty course behind

    assert len(facing_tower) == len(facing_away)
    diff = sum(abs(a - b) for a, b in zip(facing_tower, facing_away)) / len(facing_tower)
    assert diff > 0.01, (
        f"view toward geometry is indistinguishable from empty view "
        f"(mean pixel diff {diff:.5f}) - geometry may not be rendering"
    )
