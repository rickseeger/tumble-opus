"""Explicit debris budgeting: cap, settle-freeze, eviction, despawn (node 15).

Four mechanisms, each asserted against a real Bullet world with no window:

1. **The cap.** :data:`config.DEBRIS_MAX_LIVE` is the single global ceiling on
   simultaneously *stepped* debris bodies. Throw far more chunks at the field
   than the cap and the stepped count stays at or below it - at every instant,
   not just at the end.
2. **Settle-freeze.** A body whose linear and angular velocity stay under
   :data:`config.DEBRIS_SETTLE_LINEAR` / ``_ANGULAR`` for
   :data:`config.DEBRIS_SETTLE_TIME` becomes mass-0 static geometry: still
   present, still collidable, no longer stepped.
3. **Eviction.** When the cap is hit, the field retires settled/far/old debris
   and *never* the fast-moving chunk near the player - that one is the
   gameplay threat, and deleting it mid-flight in front of the player would be
   a visual lie.
4. **Despawn.** Below the world floor, or far enough behind the player, and it
   is gone.

The policy functions (:meth:`eviction_rank`, :meth:`eviction_candidates`,
:meth:`eviction_protected`) are directly callable and read-only, so the
selection can be asserted exactly rather than inferred from side effects.

No frame-rate or throughput claim is made anywhere in this file; the soak/load
proof belongs to a sibling node.
"""

import pytest
from panda3d.core import Quat, Vec3

from game import config, fracture
from game.debris import DESPAWNED, FROZEN, LIVE, DebrisField
from game.physics import PhysicsWorld

DT = config.FIXED_DT


# --------------------------------------------------------------- fixtures
def make_field(ground=True, seed=20260915, **kw):
    w = PhysicsWorld()
    if ground:
        w.add_ground_plane(0.0)
    return w, DebrisField(w, seed=seed, **kw)


def cube_spec(name="budget_cube", half=0.5):
    """One chunk, one clean cube. The simplest possible debris body."""
    return fracture.StructureSpec(
        name=name,
        kind="probe",
        blocks=(fracture.Block("b", (0.0, 0.0, half), (half, half, half)),),
        max_chunks=1,
    )


def tag(body):
    """The bare label `place` was called with.

    `DebrisField` names bodies `debris_<structure>_<chunk>_<serial>`, which is
    the right thing for it to do; these tests just want the label back.
    """
    return body.name.split("_")[1]


def settle(world, field, body, limit=3.0):
    """Step until *body* is genuinely quiet, and return the sim time it took.

    A freshly-placed cube takes a handful of steps to shed the solver's
    penetration-recovery twitch, and that twitch correctly resets the settle
    timer. So the settle-window assertions are measured from the moment the
    body is actually quiet, not from the moment it was created.
    """
    for i in range(int(limit / DT)):
        world.step_fixed(1)
        field.update(DT)
        if body.state is not LIVE:
            raise AssertionError(f"froze after {i} steps, before it was quiet")
        if field.settled(body) and body.quiet_time > 0.0:
            return (i + 1) * DT
    raise AssertionError(f"never came to rest within {limit}s")


def place(field, name, pos, *, moving=False, spin=False,
          spawn_time=None, half=0.5):
    """Spawn exactly one debris body at *pos* and set its motion by hand.

    Bypasses the launch shaping so a test can state precisely what it wants:
    a body at rest on the ground, or one tearing through the air.
    """
    result = fracture.fracture(cube_spec(name, half), seed=1)
    event = field.shatter(result, impact_point=(pos[0], pos[1], pos[2] + 10.0),
                          origin=(pos[0], pos[1], pos[2] - half))
    assert event.spawned == 1, "the probe itself hit the budget"
    body = event.bodies[0]
    body.np.setPos(Vec3(*pos))
    body.np.setQuat(Quat(1, 0, 0, 0))
    body.node.setLinearVelocity(Vec3(0, 12.0, 4.0) if moving else Vec3(0, 0, 0))
    body.node.setAngularVelocity(Vec3(6.0, 0, 0) if spin else Vec3(0, 0, 0))
    if spawn_time is not None:
        body.spawn_time = float(spawn_time)
    return body


# ================================================================ (1) cap
# One constant, one ceiling, never breached.
# =========================================================================
def test_the_cap_is_a_single_named_constant_and_is_the_field_default():
    """config.DEBRIS_MAX_LIVE is THE cap - defined once, used everywhere."""
    assert isinstance(config.DEBRIS_MAX_LIVE, int)
    assert config.DEBRIS_MAX_LIVE > 0
    w, f = make_field()
    assert f.max_live == config.DEBRIS_MAX_LIVE, (
        "DebrisField must default to the one configured cap, not its own number"
    )


def test_spawning_far_beyond_the_cap_leaves_the_stepped_count_at_or_below_it():
    """A 380-chunk tower, four times over, against a 30-body cap."""
    cap = 30
    w, f = make_field(max_live=cap)
    spec = fracture.tower_spec()
    total_chunks = 0

    for i in range(4):
        result = fracture.fracture(spec, seed=300 + i)
        total_chunks += len(result.chunks)
        f.shatter(result, impact_point=(0.0, i * 12.0, 6.0),
                  origin=(0.0, i * 12.0, 0.0))
        assert f.stepped_count() <= cap, (
            f"cap breached the instant shatter {i} landed: "
            f"{f.stepped_count()} > {cap}"
        )
        for _ in range(int(0.5 / DT)):
            w.step_fixed(1)
            f.update(DT, player_pos=(0.0, i * 12.0, 1.5))
            assert f.stepped_count() <= cap, (
                f"cap breached mid-sim: {f.stepped_count()} > {cap}"
            )

    assert total_chunks > cap * 10, "the test did not actually stress the cap"
    assert f.stepped_count() <= cap
    assert f.stepped_count() > 0, "everything was thrown away, not budgeted"
    # The cap was enforced by retiring/refusing, not by luck.
    assert (f.total_frozen + f.total_despawned
            + sum(e.skipped_for_budget for e in f.events)) > 0


def test_stepped_count_counts_dynamic_bodies_only_not_frozen_rubble():
    w, f = make_field()
    resting = place(f, "rest", (0.0, 0.0, 0.5))
    flying = place(f, "fly", (0.0, 4.0, 9.0), moving=True)
    assert f.stepped_count() == 2
    f._freeze(resting)
    assert resting.state is FROZEN
    assert f.is_stepped(resting) is False
    assert f.is_stepped(flying) is True
    assert f.stepped_count() == 1
    # Still there, still visible, still collidable - just not simulated.
    assert resting.name in f.all_bodies
    assert resting.mass == 0.0


# ======================================================= (2) settle-freeze
# Sub-threshold linear AND angular velocity, sustained, -> slept/frozen.
# =========================================================================
def test_a_quiet_body_freezes_after_the_settle_window_and_stops_being_stepped():
    w, f = make_field()
    body = place(f, "quiet", (0.0, 0.0, 0.5))
    assert body.state is LIVE

    settle(w, f, body)
    assert body.speed() <= config.DEBRIS_SETTLE_LINEAR
    assert body.spin() <= config.DEBRIS_SETTLE_ANGULAR
    assert f.settled(body), "a cube sitting still on the ground is settled"

    # Halfway through the window: the timer is running, the body is not yet
    # frozen. The dwell time is a real requirement, not a formality.
    half = int((config.DEBRIS_SETTLE_TIME * 0.5) / DT)
    for _ in range(half):
        w.step_fixed(1)
        f.update(DT)
    assert body.state is LIVE, "froze before the settle window elapsed"
    assert 0.0 < body.quiet_time < config.DEBRIS_SETTLE_TIME
    assert f.stepped_count() == 1

    # Run out the rest of the window, plus a little slack.
    for _ in range(int((config.DEBRIS_SETTLE_TIME * 0.5 + 0.1) / DT)):
        w.step_fixed(1)
        f.update(DT)
        if body.state is FROZEN:
            break
    assert body.state is FROZEN, (
        f"still {body.state} after {config.DEBRIS_SETTLE_TIME}s below threshold"
    )
    assert body.quiet_time >= config.DEBRIS_SETTLE_TIME, (
        "froze without the full dwell time actually elapsing"
    )
    assert body.mass == 0.0, "frozen debris must be mass-0 static geometry"
    assert body.is_asleep()
    assert f.stepped_count() == 0, "frozen debris is still being stepped"
    assert f.frozen_count == 1, "frozen debris vanished instead of resting"


def test_the_settle_timer_resets_when_the_body_is_disturbed():
    w, f = make_field()
    body = place(f, "jostled", (0.0, 0.0, 0.5))

    settle(w, f, body)
    for _ in range(int((config.DEBRIS_SETTLE_TIME * 0.5) / DT)):
        w.step_fixed(1)
        f.update(DT)
    assert body.state is LIVE
    partial = body.quiet_time
    assert partial > 0.0

    # Something hits it: well over both thresholds.
    body.node.setActive(True)
    body.node.setLinearVelocity(Vec3(0.0, 0.0, 8.0))
    f.update(DT)
    assert body.quiet_time == 0.0, (
        f"settle timer kept {body.quiet_time}s of credit through an impact"
    )
    assert body.state is LIVE


def test_spin_alone_is_enough_to_keep_a_body_out_of_the_frozen_state():
    """Angular velocity is checked, not just linear."""
    w, f = make_field()
    body = place(f, "spinner", (0.0, 0.0, 0.5))

    for _ in range(int((config.DEBRIS_SETTLE_TIME * 3.0) / DT)):
        body.node.setActive(True)
        body.node.setLinearVelocity(Vec3(0, 0, 0))
        body.node.setAngularVelocity(
            Vec3(0.0, 0.0, config.DEBRIS_SETTLE_ANGULAR * 4.0)
        )
        f.update(DT)
    assert body.state is LIVE, "a fast-spinning body was treated as settled"
    assert body.at_rest() is False


def test_a_frozen_body_is_no_longer_advanced_by_the_solver():
    """Frozen means frozen: the pose does not move again."""
    w, f = make_field()
    body = place(f, "still", (0.0, 0.0, 0.5))
    settle(w, f, body)
    for _ in range(int((config.DEBRIS_SETTLE_TIME + 0.2) / DT)):
        w.step_fixed(1)
        f.update(DT)
        if body.state is FROZEN:
            break
    assert body.state is FROZEN
    before = tuple(body.np.getPos())

    for _ in range(int(2.0 / DT)):
        w.step_fixed(1)
        f.update(DT)
    after = tuple(body.np.getPos())
    assert after == pytest.approx(before, abs=1e-9), (
        "a frozen body drifted, so it is still in the solver"
    )
    assert body.mass == 0.0


# =========================================================== (3) eviction
# Settled / far / old goes first. In-flight near the player never goes.
# =========================================================================
def test_eviction_protects_fast_moving_debris_near_the_player():
    w, f = make_field()
    f.set_player(pos=(0.0, 0.0, 1.5))

    threat = place(f, "threat", (0.0, 5.0, 8.0), moving=True, spin=True)
    settled_far = place(f, "settled_far", (0.0, -50.0, 0.5))
    settled_near = place(f, "settled_near", (0.0, -5.0, 0.5))
    ahead = place(f, "ahead", (0.0, 10.0, 0.5))

    # In flight within the protection radius: off-limits.
    assert f.in_flight(threat) is True
    assert f.player_distance(threat) < config.DEBRIS_PROTECT_RADIUS
    assert f.eviction_protected(threat) is True

    # Sitting in front of the player, in view: off-limits even though settled.
    assert f.in_flight(ahead) is False
    assert f.behind_distance(ahead) < 0.0
    assert f.eviction_protected(ahead) is True

    # Settled behind the player: fair game, near or far.
    assert f.eviction_protected(settled_near) is False
    assert f.eviction_protected(settled_far) is False

    candidates = f.eviction_candidates()
    assert threat not in candidates, "the in-flight threat was evictable"
    assert ahead not in candidates, "debris in the player's view was evictable"
    assert settled_far in candidates and settled_near in candidates
    # Farthest behind goes first.
    assert candidates[0] is settled_far, (
        f"eviction picked {candidates[0].name}, not the farthest-behind rubble"
    )


def test_eviction_takes_the_settled_debris_and_leaves_the_threat_alone():
    w, f = make_field()
    f.set_player(pos=(0.0, 0.0, 1.5))
    threat = place(f, "threat", (0.0, 6.0, 9.0), moving=True, spin=True)
    rubble = [place(f, f"rubble{i}", (0.0, -40.0 - i, 0.5), spawn_time=i)
              for i in range(5)]

    evicted = f.evict_for_budget(3)
    assert evicted == 3
    assert threat.state is LIVE, "the in-flight threat was evicted"
    gone = [b for b in rubble if b.state is DESPAWNED]
    assert len(gone) == 3
    # Farthest behind (largest -Y) went first: rubble4, rubble3, rubble2.
    assert {tag(b) for b in gone} == {"rubble4", "rubble3", "rubble2"}
    assert f.total_evicted == 3


def test_in_flight_debris_just_behind_the_player_is_still_protected():
    """In flight near the player is protected even *behind* them.

    Two independent reasons to protect a body near the player: it is in
    flight, or it is in view. This asserts the first one on its own, with the
    second deliberately switched off - the body is behind the player, so only
    the in-flight clause can save it. A chunk tumbling past the player's
    shoulder is exactly the debris they are reacting to; vanishing it because
    it happens to have crossed their Y is the same lie as vanishing it in
    front of them.
    """
    w, f = make_field()
    f.set_player(pos=(0.0, 0.0, 1.5))
    r = config.DEBRIS_PROTECT_RADIUS
    tumbling = place(f, "tumbling", (0.0, -r * 0.3, 6.0), moving=True, spin=True)
    resting = place(f, "resting", (0.0, -r * 0.3, 0.5))

    # Behind the player, so the in-view clause cannot be what protects it.
    assert f.behind_distance(tumbling) > 0.0
    assert f.player_distance(tumbling) < r
    assert f.in_flight(tumbling) is True
    assert f.eviction_protected(tumbling) is True, (
        "an in-flight chunk beside the player was evictable"
    )

    # Same spot, at rest: that one is rubble, and rubble is fair game.
    assert f.in_flight(resting) is False
    assert f.eviction_protected(resting) is False

    assert tumbling not in f.eviction_candidates()
    assert f.evict_for_budget(5) == 1, "only the settled body should have gone"
    assert resting.state is DESPAWNED
    assert tumbling.state is LIVE


def test_eviction_prefers_settled_over_moving_when_both_are_unprotected():
    """Beyond the protection radius a moving chunk *can* go - but last."""
    w, f = make_field()
    f.set_player(pos=(0.0, 0.0, 1.5))
    far = config.DEBRIS_PROTECT_RADIUS + 60.0
    mover = place(f, "far_mover", (0.0, -far, 6.0), moving=True, spin=True)
    settled = place(f, "far_settled", (0.0, -far, 0.5))

    assert f.eviction_protected(mover) is False
    assert f.eviction_protected(settled) is False
    candidates = f.eviction_candidates()
    assert candidates[0] is settled, (
        "a moving chunk was ranked above settled rubble"
    )
    assert f.evict_for_budget(1) == 1
    assert settled.state is DESPAWNED
    assert mover.state is LIVE


def test_eviction_breaks_ties_by_age_oldest_first():
    w, f = make_field()
    f.set_player(pos=(0.0, 0.0, 1.5))
    # Identical position, so only age can order them.
    bodies = [place(f, f"aged{i}", (0.0, -50.0, 0.5), spawn_time=10.0 - i)
              for i in range(4)]
    order = [tag(b) for b in f.eviction_candidates()]
    assert order == ["aged3", "aged2", "aged1", "aged0"], order
    assert bodies[3].spawn_time < bodies[0].spawn_time


def test_eviction_ranking_is_deterministic():
    """Same field state, same order, every time."""
    w, f = make_field()
    f.set_player(pos=(0.0, 0.0, 1.5))
    for i in range(6):
        place(f, f"r{i}", (float(i), -30.0 - i * 2, 0.5), spawn_time=i * 0.5)
    first = [b.name for b in f.eviction_candidates()]
    for _ in range(5):
        assert [b.name for b in f.eviction_candidates()] == first


def test_eviction_candidates_is_read_only():
    """The policy query must not mutate the field - tests rely on that."""
    w, f = make_field()
    f.set_player(pos=(0.0, 0.0, 1.5))
    for i in range(4):
        place(f, f"q{i}", (0.0, -40.0 - i, 0.5))
    before = (f.live_count, f.frozen_count, f.total_despawned, f.total_evicted)
    f.eviction_candidates()
    f.eviction_candidates()
    assert (f.live_count, f.frozen_count, f.total_despawned,
            f.total_evicted) == before


def test_when_everything_left_is_protected_the_shatter_spawns_less():
    """The cap is never broken to keep a protected body alive.

    With a tiny cap and the whole field in flight next to the player, the
    right answer is to refuse to spawn - not to evict the debris the player
    is watching.
    """
    cap = 4
    w, f = make_field(max_live=cap)
    f.set_player(pos=(0.0, 0.0, 1.5))
    threats = [place(f, f"t{i}", (0.0, 3.0 + i, 9.0), moving=True, spin=True)
               for i in range(cap)]
    assert f.stepped_count() == cap
    assert all(f.eviction_protected(b) for b in threats)
    assert f.eviction_candidates() == []

    result = fracture.fracture(fracture.tower_spec(), seed=11)
    event = f.shatter(result, impact_point=(0.0, 5.0, 6.0))
    assert event.spawned == 0, "the cap was breached to spawn a new shatter"
    assert event.skipped_for_budget == len(result.chunks)
    assert f.stepped_count() == cap
    assert all(b.state is LIVE for b in threats), (
        "a protected in-flight body was evicted anyway"
    )


def test_protection_radius_is_the_configured_one():
    w, f = make_field()
    f.set_player(pos=(0.0, 0.0, 1.5))
    r = config.DEBRIS_PROTECT_RADIUS
    inside = place(f, "inside", (0.0, r * 0.5, 6.0), moving=True)
    outside = place(f, "outside", (0.0, -(r + 5.0), 6.0), moving=True)
    assert f.player_distance(inside) < r
    assert f.player_distance(outside) > r
    assert f.eviction_protected(inside) is True
    assert f.eviction_protected(outside) is False


def test_with_no_known_player_position_nothing_is_protected_but_order_holds():
    w, f = make_field()
    assert f.player_pos is None
    old = place(f, "old", (0.0, 0.0, 0.5), spawn_time=1.0)
    new = place(f, "new", (0.0, 0.0, 0.5), spawn_time=9.0)
    assert f.player_distance(old) is None
    assert f.eviction_protected(old) is False
    assert [tag(b) for b in f.eviction_candidates()] == ["old", "new"]


# ============================================================ (4) despawn
# Out of the world, or far enough behind, and it is gone.
# =========================================================================
def test_debris_below_the_world_floor_is_despawned():
    w, f = make_field(ground=False)
    fallen = place(f, "fallen", (0.0, 0.0, config.DEBRIS_WORLD_FLOOR_Z - 5.0))
    kept = place(f, "kept", (0.0, 0.0, config.DEBRIS_WORLD_FLOOR_Z + 5.0))

    assert f.despawn_out_of_world() == 1
    assert fallen.state is DESPAWNED
    assert kept.state is LIVE
    assert f.total_floor_despawned == 1
    assert fallen.name not in f.all_bodies


def test_update_despawns_debris_that_falls_out_of_the_world():
    """End to end: no ground, so it falls, and eventually it is reaped."""
    w, f = make_field(ground=False)
    body = place(f, "faller", (0.0, 0.0, 2.0))
    body.node.setActive(True)
    body.node.setLinearVelocity(Vec3(0.0, 0.0, -30.0))

    for _ in range(int(6.0 / DT)):
        w.step_fixed(1)
        f.update(DT)
        if body.state is DESPAWNED:
            break
    assert body.state is DESPAWNED, (
        f"body at z={body.np.getZ():.1f} was never despawned"
    )
    assert f.total_bodies == 0
    assert f.total_floor_despawned >= 1
    assert w.world.getNumRigidBodies() == 0, "the node was left in the world"


def test_frozen_debris_below_the_world_floor_is_also_despawned():
    w, f = make_field(ground=False)
    body = place(f, "frozen_faller", (0.0, 0.0, 0.5))
    f._freeze(body)
    assert body.state is FROZEN
    body.np.setZ(config.DEBRIS_WORLD_FLOOR_Z - 1.0)
    assert f.despawn_out_of_world() == 1
    assert body.state is DESPAWNED
    assert f.frozen_count == 0


def test_debris_far_enough_behind_the_player_is_despawned():
    w, f = make_field()
    behind = config.DEBRIS_DESPAWN_BEHIND
    far = place(f, "far_behind", (0.0, -(behind + 10.0), 0.5))
    near = place(f, "near_behind", (0.0, -(behind * 0.5), 0.5))
    ahead = place(f, "ahead", (0.0, 20.0, 0.5))

    assert f.despawn_behind_player(player_y=0.0) == 1
    assert far.state is DESPAWNED
    assert near.state is LIVE, "rubble the player can still turn and see"
    assert ahead.state is LIVE
    assert f.total_behind_despawned == 1


def test_the_behind_despawn_distance_is_outside_the_protection_radius():
    """Structural: nothing can ever be despawned-behind while in view.

    The behind-player despawn does not consult the protection radius, so the
    two numbers have to be ordered for the guarantee to hold.
    """
    assert config.DEBRIS_DESPAWN_BEHIND > config.DEBRIS_PROTECT_RADIUS


def test_update_retires_debris_the_player_drives_past():
    w, f = make_field()
    result = fracture.fracture(
        fracture.with_budget(fracture.cluster_spec(), 60), seed=6
    )
    f.shatter(result, impact_point=(0.0, 0.0, 3.0))
    for _ in range(int(6.0 / DT)):
        w.step_fixed(1)
        f.update(DT, player_pos=(0.0, 0.0, 1.5))
    standing = f.total_bodies
    assert standing > 0, "the debris was gone before the player moved on"

    for _ in range(int(0.5 / DT)):
        w.step_fixed(1)
        f.update(DT, player_pos=(0.0, config.DEBRIS_DESPAWN_BEHIND + 150.0, 1.5))
    assert f.total_bodies == 0, f"{f.total_bodies} bodies survived the drive-by"
    assert f.total_behind_despawned >= standing
    assert w.world.getNumRigidBodies() == 1, "only the ground should remain"


def test_update_accepts_a_full_player_position_and_remembers_it():
    w, f = make_field()
    f.update(DT, player_pos=(3.0, 25.0, 1.5))
    assert f.player_y == pytest.approx(25.0)
    assert tuple(f.player_pos) == pytest.approx((3.0, 25.0, 1.5))
    # Y-only still works, for callers that only have the corridor position.
    f.update(DT, player_y=40.0)
    assert f.player_y == pytest.approx(40.0)


def test_the_snapshot_reports_the_budget():
    w, f = make_field(max_live=25)
    place(f, "one", (0.0, 0.0, 0.5))
    snap = f.snapshot()
    assert snap["max_live"] == 25
    assert snap["stepped"] == 1
    for key in ("evicted", "floor_despawned", "behind_despawned"):
        assert snap[key] == 0
