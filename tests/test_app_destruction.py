"""The real game loop demolishing real structures, headless.

`TumbleApp` in `window-type none` mode: no window, no GL, but the same class,
the same frame function and the same destruction path the human playtest runs.
These are the tests that would catch "it works in the test harness but the
game itself never calls it".
"""

import math

import pytest

from game import config, structures
from game.player import InputState


@pytest.fixture
def app():
    from game.app import TumbleApp

    a = TumbleApp(headless=True)
    yield a
    a.destroy()


def test_app_boots_with_destructibles_and_a_debris_field(app):
    assert app.headless is True
    assert app.win is None
    assert len(app.destructibles) == len(structures.PLACEMENTS)
    assert app.debris is not None
    assert app.debris.live_count == 0
    assert app.demolitions == 0
    # Every structure standing, every proxy in the world.
    for d in app.destructibles:
        assert d.intact
        for name in d.proxy_names:
            assert name in app.physics.bodies


def test_the_frame_function_updates_the_debris_field(app):
    """If `step_frame` does not call `debris.update`, nothing is ever retired
    and the budget leaks. This asserts the loop really does it."""
    d = app.destructibles[0]
    event = app.demolish(d)
    assert event is not None

    for _ in range(60 * 40):                 # 40 s at 60 fps
        app.step_frame(1.0 / 60.0)

    snap = app.debris.snapshot()
    assert snap["frozen"] > 0, (
        "nothing was ever retired - step_frame is not updating the field"
    )
    assert snap["active"] == 0, f"{snap['active']} bodies still awake"
    assert snap["max_speed"] < 1e-9
    assert snap["max_spin"] < 1e-9


def test_demolish_nearest_is_the_weapon_node_hook(app):
    """One call, no arguments: the surface a trigger/weapon will use."""
    first = app.demolish_nearest()
    assert first is not None
    assert app.demolitions == 1
    assert app.debris.live_count == first.spawned
    # The nearest structure to the spawn point is the one that went down.
    down = [d for d in app.destructibles if not d.intact]
    assert len(down) == 1
    assert down[0].world_bounds()[0][1] == min(
        d.world_bounds()[0][1] for d in app.destructibles
    )


def test_demolishing_everything_returns_none_when_nothing_is_left(app):
    seen = 0
    while True:
        event = app.demolish_nearest()
        if event is None:
            break
        seen += 1
        assert seen <= len(app.destructibles) + 1
    assert seen == len(app.destructibles)
    assert all(not d.intact for d in app.destructibles)
    assert app.demolish_nearest() is None


def test_the_body_cap_holds_while_the_loop_runs(app):
    """Drive the course, demolishing as you go, at the real frame rate."""
    app.input_state = InputState(forward=True)
    peak_live = 0
    for _ in range(60 * 60):                 # 60 s at 60 fps
        app.step_frame(1.0 / 60.0)
        y = app.player.pos[1]
        for d in app.destructibles:
            if d.intact and d.world_bounds()[0][1] - y <= 18.0:
                app.demolish(d)
        peak_live = max(peak_live, app.debris.live_count)
        assert app.debris.live_count <= app.debris.max_live, (
            f"cap breached in the real loop: {app.debris.live_count}"
        )

    assert app.demolitions >= 1, "the run never demolished anything"
    assert peak_live > 0


def test_a_demolished_structure_stops_blocking_the_player(app):
    """The proxy removal is what lets the player through. Prove it moves."""
    target = min(app.destructibles, key=lambda d: d.world_bounds()[0][1])
    lo, hi = target.world_bounds()

    # Park the player right up against the structure and push forward.
    app.player.np.setPos(
        (lo[0] + hi[0]) * 0.5, lo[1] - 2.0, config.SPAWN_POS[2]
    )
    app.player.heading = 0.0
    app.player.np.setH(0.0)
    app.input_state = InputState(forward=True)

    for _ in range(180):
        app.step_frame(1.0 / 60.0)
    blocked_y = app.player.pos[1]
    assert blocked_y < lo[1] + 0.5, (
        f"the player walked to y={blocked_y:.2f} through an intact structure "
        f"whose near face is at y={lo[1]:.2f}"
    )

    app.demolish(target)
    # Let the rubble fall out of the way, then push on.
    for _ in range(60 * 12):
        app.step_frame(1.0 / 60.0)
    after_y = app.player.pos[1]

    assert after_y > blocked_y + 1.0, (
        f"the player is still stuck at y={after_y:.2f} (was {blocked_y:.2f}) "
        f"after the structure was demolished - the collision proxy is still "
        f"in the world"
    )


def test_debris_leaves_no_nans_in_the_world(app):
    app.demolish_nearest()
    for _ in range(60 * 30):
        app.step_frame(1.0 / 60.0)
    for body in app.debris.live + app.debris.frozen:
        for v in tuple(body.pos) + tuple(body.quat):
            assert not math.isnan(float(v)), f"{body.name} has a NaN pose"
        assert abs(float(body.pos.getX())) < 1e4
        assert abs(float(body.pos.getZ())) < 1e4


def test_headless_demolition_smoke_run_exits_cleanly(capsys):
    """The scripted entry point Rick and CI actually run."""
    import main as entry

    rc = entry.main(["--headless", "--demolish", "--frames", "900"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "headless boot OK" in out
    assert "demolished" in out, "the smoke run never demolished anything"
    assert "pre-generated chunks" in out
    assert "shutdown clean" in out
    # And it reported real debris state, not just that it started.
    assert "debris live=" in out
    assert "live body cap=" in out


def test_headless_run_without_demolition_still_works(capsys):
    """The destruction layer must not break the plain smoke run."""
    import main as entry

    rc = entry.main(["--headless", "--frames", "300"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "shutdown clean" in out
    assert "demolished" not in out
