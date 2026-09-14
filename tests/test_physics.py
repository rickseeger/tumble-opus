"""Physics-layer assertions. No window, no ShowBase, no rendering."""

import pytest

from panda3d.core import Vec3

from game.physics import PhysicsWorld


@pytest.fixture
def world():
    w = PhysicsWorld()
    w.add_ground_plane(0.0)
    return w


def test_gravity_pulls_a_body_down(world):
    """A released body must accelerate downward, not hover or rise."""
    box = world.add_dynamic_box("faller", (0, 0, 20.0), (0.5, 0.5, 0.5), mass=1.0)
    start_z = box.getZ()

    world.step_fixed(12)          # 0.1 s
    z_after_short = box.getZ()
    world.step_fixed(48)          # 0.5 s total
    z_after_long = box.getZ()

    assert z_after_short < start_z, "body did not move down at all"
    assert z_after_long < z_after_short, "descent did not continue"

    # Free fall for 0.5 s under -9.81 m/s^2 is ~1.226 m. Bullet's semi-
    # implicit Euler overshoots slightly; demand real physics, not a drift.
    drop = float(start_z - z_after_long)
    expected = 0.5 * 9.81 * 0.5 ** 2
    assert drop == pytest.approx(expected, rel=0.08), (
        f"drop {drop:.4f} m does not match free fall {expected:.4f} m"
    )


def test_falling_body_comes_to_rest_on_the_ground(world):
    """It must stop at its own half-height, and stay stopped."""
    half = 0.5
    box = world.add_dynamic_box("rester", (0, 0, 8.0), (half, half, half), mass=1.0)

    world.step_fixed(900)         # 7.5 s: plenty of time to land and settle
    resting_z = float(box.getZ())

    assert resting_z == pytest.approx(half, abs=0.03), (
        f"box centre rests at {resting_z:.4f}, expected ~{half}"
    )
    assert resting_z > 0.0, "box sank to or below the ground plane"

    world.step_fixed(300)         # another 2.5 s of contact
    assert float(box.getZ()) == pytest.approx(resting_z, abs=0.01), "body crept"


def test_body_does_not_tunnel_through_ground_over_long_run(world):
    """Sample every step: contact dip is bounded and fully recovered.

    A discrete solver always overshoots by up to one step of travel on the
    impact frame. Dropped from 12 m the body arrives at ~15.1 m/s, so one
    1/120 s step is ~0.126 m - that is the floor on any honest tolerance
    here. What must NOT happen is passing through, or staying sunk.
    """
    half = 0.4
    box = world.add_dynamic_box("probe", (0, 0, 12.0), (half, half, half), mass=2.0)

    worst = float("inf")
    for _ in range(1200):         # 10 s
        world.step_fixed(1)
        worst = min(worst, float(box.getZ()) - half)

    impact_speed = (2 * 9.81 * (12.0 - half)) ** 0.5
    one_step_travel = impact_speed * world.fixed_dt
    assert worst > -(one_step_travel + 0.02), (
        f"body sank {-worst:.4f} m, more than one step of travel "
        f"({one_step_travel:.4f} m)"
    )
    # And it must be sitting cleanly on the surface at the end.
    assert float(box.getZ()) == pytest.approx(half, abs=0.01)


def test_fast_body_does_not_pass_through_the_ground(world):
    """CCD guard: high-speed bodies must not escape the world.

    Measured without CCD on this stack, a body launched downward at 200 m/s
    ends up at z=+106 (spurious impulse) or straight through the floor. Later
    destruction nodes will fling debris fast, so this is load-bearing.
    """
    half = 0.3
    for speed in (50.0, 200.0, 500.0):
        box = world.add_dynamic_box(
            f"bullet_{int(speed)}", (speed, 0, 5.0), (half, half, half), mass=1.0
        )
        box.node().setLinearVelocity(Vec3(0, 0, -speed))

    world.step_fixed(900)

    for speed in (50.0, 200.0, 500.0):
        z = float(world.body(f"bullet_{int(speed)}").getZ())
        assert z == pytest.approx(half, abs=0.05), (
            f"body launched at {speed} m/s ended at z={z:.3f}, not resting "
            f"on the ground at {half}"
        )


def test_static_box_is_immovable_under_impact(world):
    """Mass-0 geometry must not be shoved by a falling dynamic body."""
    static = world.add_static_box("anvil", (0, 0, 1.0), (1.0, 1.0, 1.0))
    before = static.getPos()
    world.add_dynamic_box("hammer", (0, 0, 6.0), (0.5, 0.5, 0.5), mass=25.0)

    world.step_fixed(600)

    after = static.getPos()
    assert (after - before).length() < 1e-4, "static body moved"


def test_fixed_step_is_deterministic():
    """Same inputs, same steps -> bit-comparable results across two worlds."""
    def run():
        w = PhysicsWorld()
        w.add_ground_plane(0.0)
        b = w.add_dynamic_box("d", (0.3, -0.2, 9.0), (0.5, 0.5, 0.5), mass=1.0)
        w.step_fixed(400)
        return tuple(round(float(v), 6) for v in b.getPos())

    assert run() == run()


def test_step_accounting_and_render_decoupling():
    """`advance` must only ever run whole fixed steps, and clamp backlog."""
    w = PhysicsWorld()
    w.add_ground_plane(0.0)
    dt = w.fixed_dt

    assert w.advance(dt * 0.4) == 0, "a partial step was simulated"
    assert w.step_count == 0
    assert w.pending_time == pytest.approx(dt * 0.4)

    # 0.4 + 0.7 = 1.1 fixed steps -> exactly one step runs, 0.1 is retained.
    assert w.advance(dt * 0.7) == 1
    assert w.step_count == 1
    assert w.pending_time == pytest.approx(dt * 0.1, abs=1e-9)
    assert w.sim_time == pytest.approx(dt)

    # A huge hitch must not run unbounded steps.
    taken = w.advance(dt * 500)
    assert taken == w_max_steps(), f"ran {taken} steps in one frame"
    assert w.pending_time == 0.0, "backlog was not dropped"


def w_max_steps():
    from game import config
    return config.MAX_STEPS_PER_FRAME


def test_gravity_sign_and_magnitude(world):
    """Guard against a mis-signed gravity vector reaching later nodes."""
    g = world.world.getGravity()
    assert g.getZ() < 0.0
    assert abs(g.getZ()) == pytest.approx(9.81, rel=0.01)
    assert g.getX() == 0.0 and g.getY() == 0.0
