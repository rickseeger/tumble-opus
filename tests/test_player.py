"""Player-layer assertions: locomotion, collision, look, camera offset."""

import math

import pytest

from game import config
from game.physics import PhysicsWorld
from game.player import InputState, Player
from game.world import build_course


@pytest.fixture
def sim():
    w = PhysicsWorld()
    build_course(w)
    return w, Player(w)


def simulate(player, world, state, seconds):
    """Hold `state` for `seconds` of fixed steps, as the app would."""
    steps = int(round(seconds / world.fixed_dt))
    for _ in range(steps):
        player.apply_input(state)
        world.step_fixed(1)
    return steps


def settle(player, world, seconds=1.5):
    simulate(player, world, InputState(), seconds)


def test_player_spawns_falling_then_stands_on_ground(sim):
    world, player = sim
    assert player.pos[2] == pytest.approx(config.SPAWN_POS[2])

    settle(player, world, 2.0)

    assert player.on_ground(), "player never registered ground contact"
    # Capsule centre should sit ~half its total height above the plane.
    assert player.feet_z == pytest.approx(0.0, abs=0.12), (
        f"feet at z={player.feet_z:.4f}, expected ~0"
    )


def test_player_does_not_sink_through_the_floor(sim):
    """Sample feet Z every step for 10 s of standing and walking."""
    world, player = sim
    settle(player, world, 2.0)

    worst = float("inf")
    for i in range(1200):
        state = InputState(forward=(i % 200) < 100)
        player.apply_input(state)
        world.step_fixed(1)
        worst = min(worst, player.feet_z)

    assert worst > -0.06, f"player feet reached z={worst:.4f} (below floor)"


def test_forward_input_advances_player_along_forward_axis(sim):
    world, player = sim
    settle(player, world, 2.0)
    start = player.pos

    simulate(player, world, InputState(forward=True), 1.0)
    moved = player.pos

    dy = float(moved[1] - start[1])
    assert dy > 0.0, "forward input did not move the player forward"
    # 1 s at FORWARD_SPEED, on flat open ground: expect close to the nominal.
    assert dy == pytest.approx(config.FORWARD_SPEED, rel=0.15), (
        f"advanced {dy:.3f} m/s, expected ~{config.FORWARD_SPEED}"
    )
    assert abs(float(moved[0] - start[0])) < 0.3, "forward input caused drift in X"


def test_no_input_means_no_horizontal_movement(sim):
    """Distinguishes 'forward works' from 'the player always slides'."""
    world, player = sim
    settle(player, world, 2.0)
    start = player.pos

    simulate(player, world, InputState(), 2.0)

    assert abs(float(player.pos[0] - start[0])) < 0.05
    assert abs(float(player.pos[1] - start[1])) < 0.05


def test_movement_follows_heading_not_world_axis(sim):
    """Turn 90 degrees, then 'forward' must become world -X, not +Y."""
    world, player = sim
    settle(player, world, 2.0)
    # +90 deg heading in Panda turns the body to face world -X.
    player.apply_look(-90.0 / config.MOUSE_SENSITIVITY, 0.0)
    assert player.heading == pytest.approx(90.0, abs=0.5)
    start = player.pos

    simulate(player, world, InputState(forward=True), 1.0)

    dx = float(player.pos[0] - start[0])
    dy = float(player.pos[1] - start[1])
    assert dx < -3.0, f"expected motion toward -X, got dx={dx:.3f}"
    assert abs(dy) < abs(dx) * 0.2, f"unexpected Y motion dy={dy:.3f}"


def test_player_cannot_pass_through_static_structure():
    """Walk straight into a wall of static boxes and get stopped short."""
    world = PhysicsWorld()
    world.add_ground_plane(0.0)
    # Solid barrier across the path at y = +6.
    for i in range(-2, 3):
        world.add_static_box(f"bar_{i}", (i * 2.0, 6.0, 1.0), (1.0, 0.5, 1.0))

    player = Player(world, spawn=(0.0, 0.0, 1.5))
    settle(player, world, 2.0)

    # Push forward for 5 s: unobstructed that is ~40 m, far past the wall.
    simulate(player, world, InputState(forward=True), 5.0)

    y = float(player.pos[1])
    barrier_face = 6.0 - 0.5          # near face of the barrier boxes
    limit = barrier_face - config.PLAYER_RADIUS + 0.12
    assert y < limit, f"player reached y={y:.3f}, passing into/through the wall"
    assert y > 2.0, f"player only reached y={y:.3f} - it never travelled"


def test_player_is_blocked_by_course_boundary_wall(sim):
    """The authored course walls must actually contain the player."""
    world, player = sim
    settle(player, world, 2.0)
    player.apply_look(-90.0 / config.MOUSE_SENSITIVITY, 0.0)   # face -X

    simulate(player, world, InputState(forward=True, sprint=True), 6.0)

    x = float(player.pos[0])
    wall_face = -config.COURSE_HALF_WIDTH + 0.5
    assert x > wall_face - 0.2, f"player escaped the course at x={x:.3f}"
    assert x < -5.0, f"player never travelled toward the wall (x={x:.3f})"


def test_player_never_flies_without_jumping(sim):
    """No noclip/fly: with only WASD held, the player stays grounded."""
    world, player = sim
    settle(player, world, 2.0)
    grounded_z = float(player.pos[2])

    highest = -1e9
    for i in range(900):
        player.apply_input(InputState(forward=True, sprint=(i % 2 == 0)))
        world.step_fixed(1)
        highest = max(highest, float(player.pos[2]))

    assert highest < grounded_z + 0.45, (
        f"player rose to z={highest:.3f} from {grounded_z:.3f} with no jump input"
    )


def test_forward_is_faster_than_backward_and_strafe():
    """The 'driving forward' bias is a real asymmetry, not a comment."""
    p = Player.__new__(Player)     # pure-function check, no physics needed
    fwd = Player.desired_velocity(p, InputState(forward=True))
    back = Player.desired_velocity(p, InputState(backward=True))
    strafe = Player.desired_velocity(p, InputState(right=True))
    sprint = Player.desired_velocity(p, InputState(forward=True, sprint=True))

    assert fwd.getY() > 0.0 and back.getY() < 0.0
    assert fwd.getY() > abs(back.getY()) * 2.0
    assert fwd.getY() > strafe.getX()
    assert sprint.getY() > fwd.getY()
    # Sprint must not create free backward speed.
    assert Player.desired_velocity(
        p, InputState(backward=True, sprint=True)
    ).getY() == pytest.approx(back.getY())


def test_pitch_is_clamped_and_yaw_wraps(sim):
    world, player = sim
    for _ in range(50):
        player.apply_look(0.0, -500.0)
    assert player.pitch == pytest.approx(config.PITCH_LIMIT)
    for _ in range(100):
        player.apply_look(0.0, 500.0)
    assert player.pitch == pytest.approx(-config.PITCH_LIMIT)

    for _ in range(200):
        player.apply_look(100.0, 0.0)
    assert -180.0 <= player.heading <= 180.0
    # Body yaw must track the look heading.
    assert float(player.np.getH()) == pytest.approx(player.heading, abs=1e-3)


def test_eye_point_stays_at_expected_offset_above_body(sim):
    """Camera/eye offset holds while moving, turning and falling."""
    world, player = sim
    for i in range(600):
        player.apply_input(InputState(forward=True, mouse_dx=3.0 if i < 100 else 0.0))
        world.step_fixed(1)
        offset = player.eye_pos - player.pos
        assert offset[2] == pytest.approx(config.EYE_HEIGHT, abs=1e-6)
        assert abs(offset[0]) < 1e-6 and abs(offset[1]) < 1e-6
    assert config.EYE_HEIGHT < player.half_total_height + 0.5


def test_eye_sits_at_a_human_standing_height_above_the_ground(sim):
    """Absolute check, independent of config: the view must be head-high.

    Deliberately written against literal metres rather than EYE_HEIGHT, so
    that retuning the constant to something unplayable (a camera in the dirt
    or floating above the capsule) fails here instead of passing silently.
    """
    world, player = sim
    settle(player, world, 2.0)

    eye_above_ground = float(player.eye_pos[2])
    assert 1.2 < eye_above_ground < 2.0, (
        f"eye is {eye_above_ground:.3f} m above the ground - not head height"
    )

    # The eye must be inside the capsule's vertical extent, not floating.
    top_of_head = float(player.pos[2]) + player.half_total_height
    assert float(player.eye_pos[2]) <= top_of_head + 1e-6, "eye above the head"
    assert float(player.eye_pos[2]) > float(player.pos[2]), "eye below body centre"


def test_camera_nodepath_is_parented_at_eye_height(sim):
    """Render-layer contract, verified with a stand-in NodePath."""
    from panda3d.core import NodePath

    world, player = sim
    cam = NodePath("stand-in-camera")
    player.attach_camera(cam)

    assert cam.getParent() == player.np
    # Compare component-wise: LPoint3f is not a sequence pytest.approx reads.
    cam_pos = cam.getPos()
    assert float(cam_pos[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(cam_pos[1]) == pytest.approx(0.0, abs=1e-6)
    assert float(cam_pos[2]) == pytest.approx(config.EYE_HEIGHT, abs=1e-6)

    settle(player, world, 1.0)
    player.apply_look(200.0, -300.0)
    player.sync_camera()

    assert float(cam.getP()) == pytest.approx(player.pitch, abs=1e-3)
    # Camera must ride the body in world space.
    world_cam = cam.getPos(world.root)
    assert world_cam[2] == pytest.approx(
        float(player.pos[2]) + config.EYE_HEIGHT, abs=1e-4
    )
    assert math.isfinite(world_cam[0])
