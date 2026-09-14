"""App-layer smoke tests: boot with no window, step, shut down cleanly."""

import pytest

from game import config
from game.player import InputState
from game.world import block_count


@pytest.fixture
def app():
    from game.app import TumbleApp

    a = TumbleApp(headless=True)
    yield a
    a.destroy()


def test_app_boots_headless_with_no_window(app):
    assert app.win is None, "a window was created in headless mode"
    assert app.headless is True
    assert app.physics is not None and app.player is not None


def test_course_is_populated_on_boot(app):
    names = app.physics.bodies
    assert "ground" in names
    assert "player" not in names          # the character is not a rigid body
    # Tower blocks + 2 boundary walls.
    assert len(app.physics.box_specs) == block_count() + 2
    assert all(s.static for s in app.physics.box_specs), "course must be static"
    assert any(n.startswith("tower_c") for n in names)


def test_step_frame_runs_fixed_steps_and_moves_the_player(app):
    app.input_state = InputState()
    # Let the spawn drop settle first.
    for _ in range(120):
        app.step_frame(1.0 / 60.0)
    assert app.physics.step_count > 0
    assert app.player.on_ground()

    start_y = float(app.player.pos[1])
    app.input_state = InputState(forward=True)
    for _ in range(120):                   # 2 s at 60 fps
        app.step_frame(1.0 / 60.0)
    end_y = float(app.player.pos[1])

    assert end_y - start_y > config.FORWARD_SPEED * 1.0, (
        f"advanced only {end_y - start_y:.3f} m in 2 s"
    )
    assert app.frames_run == 240


def test_sim_time_tracks_wall_clock_regardless_of_frame_rate():
    """Same total dt delivered in big or small frames -> same sim time."""
    from game.app import TumbleApp

    def total(frame_dt, frames):
        a = TumbleApp(headless=True)
        for _ in range(frames):
            a.step_frame(frame_dt)
        t = a.physics.sim_time
        a.destroy()
        return t

    fine = total(1.0 / 240.0, 240)     # 1.0 s in tiny frames
    coarse = total(1.0 / 60.0, 60)     # 1.0 s in bigger frames
    assert fine == pytest.approx(1.0, abs=config.FIXED_DT)
    assert coarse == pytest.approx(1.0, abs=config.FIXED_DT)
    assert fine == pytest.approx(coarse, abs=config.FIXED_DT * 2)


def test_headless_step_never_drops_the_player_through_the_world(app):
    app.input_state = InputState(forward=True, sprint=True)
    worst = float("inf")
    for _ in range(600):
        app.step_frame(1.0 / 60.0)
        worst = min(worst, app.player.feet_z)
    assert worst > -0.06, f"player feet reached {worst:.4f}"
    assert app.player.pos[2] > 0.0


def test_headless_prc_requests_no_window():
    prc = config.display_prc(headless=True)
    assert "window-type none" in prc
    assert "window-type none" not in config.display_prc(headless=False)


def test_main_headless_entrypoint_returns_zero(capsys):
    import main

    rc = main.main(["--headless", "--frames", "120"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "headless boot OK" in out
    assert "shutdown clean" in out
    assert "m forward" in out
