"""Damage-triggered destruction, driven through the real game loop, headless.

This is the node-13 integration: a structure's *accumulated damage* reaching
its integrity threshold is what removes the intact building from the world and
replaces it, in place, with its pre-generated chunk set as independently
simulated rigid bodies.

Everything here drives `TumbleApp` in `window-type none` mode - the same class,
the same `step_frame`, the same destruction path the windowed playtest runs.
There is no test-only shim: the tests call `app.strike(...)` (the surface the
weapon node will call) or `damage.damage_structure(...)`, and the demolition
happens because the damage system decided it should.

Deliberately NOT tested here: body caps, pooling, retirement budgets or frame
timing. Those belong to the performance node and are not this node's claim.
"""

import math

import pytest

from game import config, structures
from game.damage import DamageSystem, integrity_for
from game.player import InputState


# The cluster at y=30 is the first structure on the course and its chunk count
# (240) is inside the debris field's live budget, so a single demolition
# spawns every chunk and "debris count == chunk count" is an exact claim
# rather than a budget-dependent one. Budget behaviour is node 14's problem.
TARGET_INDEX = 0


@pytest.fixture
def app():
    from game.app import TumbleApp

    a = TumbleApp(headless=True)
    yield a
    a.destroy()


def _world_body_nodes(app):
    """Every rigid body Bullet currently has in the world, by name."""
    return {b.getName() for b in app.physics.world.getRigidBodies()}


def _debris_of(app, structure_name):
    return [b for b in app.debris.live + app.debris.frozen
            if b.structure == structure_name]


# ----------------------------------------------------------------- the wiring
def test_the_app_owns_a_damage_system_wired_to_its_own_destruction_path(app):
    """The decision layer exists in the running game, not just in a test."""
    assert isinstance(app.damage, DamageSystem)
    assert app.damage.debris is app.debris
    assert app.damage.physics is app.physics
    # The trigger goes through the app's real demolish(), which is what
    # removes proxies from the Bullet world and swaps the visuals.
    assert app.damage.on_destroy == app.demolish
    assert len(app.damage.states) == len(app.destructibles)
    for d in app.destructibles:
        st = app.damage.state(d)
        assert st.threshold > 0.0
        assert st.accumulated == 0.0
        assert st.destroyed is False
        assert st.fraction == 0.0


def test_integrity_threshold_scales_with_structure_volume(app):
    """A 60 m tower must not be as fragile as a low cluster."""
    by_vol = sorted(app.destructibles, key=lambda d: d.spec.volume)
    thresholds = [integrity_for(d) for d in by_vol]
    assert thresholds == sorted(thresholds)
    assert thresholds[-1] > thresholds[0]
    assert min(thresholds) >= config.STRUCTURE_INTEGRITY_MIN


def test_damage_below_the_threshold_does_not_destroy_anything(app):
    d = app.destructibles[TARGET_INDEX]
    st = app.damage.state(d)
    app.damage.damage_structure(d, st.threshold * 0.4)
    app.damage.damage_structure(d, st.threshold * 0.4)

    assert st.accumulated == pytest.approx(st.threshold * 0.8)
    assert st.hits == 2
    assert 0.79 < st.fraction < 0.81
    assert st.destroyed is False
    assert d.intact is True
    assert app.debris.total_bodies == 0
    assert app.demolitions == 0
    # And the proxies are all still solid.
    live = _world_body_nodes(app)
    for name in d.proxy_names:
        assert name in live


# ------------------------------------------- (a)(b) the swap itself, in a frame
def test_reaching_the_threshold_swaps_the_structure_for_its_chunks(app):
    """(a) intact + zero debris before; (b) gone + chunk-count debris after."""
    d = app.destructibles[TARGET_INDEX]
    st = app.damage.state(d)

    # ---- (a) before -------------------------------------------------------
    assert d.intact
    assert _debris_of(app, d.result.spec_name) == []
    before_world = _world_body_nodes(app)
    assert d.proxy_names
    for name in d.proxy_names:
        assert name in before_world
        assert name in app.physics.bodies
    assert app.debris.total_bodies == 0

    # ---- the destruction event, through the real path ---------------------
    report = app.strike(d.default_impact_point())

    assert report.any_destroyed
    assert d.name in report.destroyed
    assert st.destroyed is True
    assert st.at_threshold
    assert st.fraction == 1.0
    assert app.demolitions == 1

    # ---- (b) after --------------------------------------------------------
    assert d.intact is False
    after_world = _world_body_nodes(app)
    for name in d.proxy_names:
        assert name not in app.physics.bodies, f"{name} still in the registry"
        assert name not in after_world, f"{name} still a body in the world"
    # ...and gone from the scene graph too, not merely detached from the
    # solver. A body left parented under sim-root would still be drawn.
    for name in d.proxy_names:
        assert app.physics.root.find(f"**/{name}").isEmpty(), (
            f"{name} is still in the scene graph after demolition"
        )

    event = report.events[0]
    assert event.chunk_count == d.chunk_count
    assert event.skipped_for_budget == 0, (
        "this structure is meant to fit inside the budget so the count is exact"
    )
    assert event.spawned == d.chunk_count
    spawned = _debris_of(app, d.result.spec_name)
    assert len(spawned) == d.chunk_count
    assert app.debris.total_bodies == d.chunk_count
    # Every one of them is a real body in the Bullet world.
    for body in spawned:
        assert body.name in after_world
        assert float(body.node.getMass()) > 0.0


# ------------------------------------------------------- (c) spawn positions
def test_debris_spawns_inside_the_original_structures_volume(app):
    d = app.destructibles[TARGET_INDEX]
    lo, hi = d.world_bounds()

    app.strike(d.default_impact_point())
    bodies = _debris_of(app, d.result.spec_name)
    assert len(bodies) > 1

    for body in bodies:
        p = body.pos
        for i, v in enumerate((p.getX(), p.getY(), p.getZ())):
            assert lo[i] - 1e-6 <= v <= hi[i] + 1e-6, (
                f"{body.name} spawned at {tuple(p)} outside the structure "
                f"bounds {lo}..{hi}"
            )

    # In place, not at the origin: the chunks inherit the structure's world
    # transform, so their centroid must sit inside the footprint, far from (0,0).
    cx = sum(float(b.pos.getX()) for b in bodies) / len(bodies)
    cy = sum(float(b.pos.getY()) for b in bodies) / len(bodies)
    assert lo[0] <= cx <= hi[0] and lo[1] <= cy <= hi[1]
    assert abs(cy - d.origin[1]) < (hi[1] - lo[1])
    # Distinct positions: this is a shattered building, not a stack at one point.
    distinct = {(round(float(b.pos.getX()), 3),
                 round(float(b.pos.getY()), 3),
                 round(float(b.pos.getZ()), 3)) for b in bodies}
    assert len(distinct) == len(bodies)


# ------------------------------------------------- (d) launch velocity + spin
def test_debris_launches_with_varied_velocity_and_spin(app):
    d = app.destructibles[TARGET_INDEX]
    app.strike(d.default_impact_point())
    bodies = _debris_of(app, d.result.spec_name)

    speeds = [b.speed() for b in bodies]
    spins = [b.spin() for b in bodies]

    assert min(speeds) > 0.0, "a chunk was spawned with zero linear velocity"
    assert min(spins) > 0.0, "a chunk was spawned with zero angular velocity"
    # Varied, not one canned impulse copied N times.
    assert len(set(round(s, 6) for s in speeds)) > len(bodies) // 2
    assert len(set(round(s, 6) for s in spins)) > len(bodies) // 2
    assert max(speeds) > min(speeds) * 1.5
    assert max(spins) > min(spins) * 1.5
    # Directions differ too: the net momentum of an outward blast is small
    # compared with the total speed flying around.
    vx = sum(float(b.linear_velocity().getX()) for b in bodies)
    vy = sum(float(b.linear_velocity().getY()) for b in bodies)
    assert abs(vx) + abs(vy) < sum(speeds)


# ------------------------------------ (e) gravity, settling, no exploded coords
def test_debris_falls_settles_above_the_ground_and_stays_finite(app):
    d = app.destructibles[TARGET_INDEX]
    app.strike(d.default_impact_point())
    bodies = _debris_of(app, d.result.spec_name)
    spawn_mean_z = sum(float(b.pos.getZ()) for b in bodies) / len(bodies)
    spawn_max_z = max(float(b.pos.getZ()) for b in bodies)

    # The real frame function, for 45 simulated seconds at 60 fps.
    for _ in range(60 * 45):
        app.step_frame(1.0 / 60.0)

    alive = [b for b in app.debris.live + app.debris.frozen
             if b.structure == d.result.spec_name]
    assert alive, "every chunk vanished - nothing left to assert on"

    mean_z = sum(float(b.pos.getZ()) for b in alive) / len(alive)
    assert mean_z < spawn_mean_z, (
        f"debris did not fall: mean z {spawn_mean_z:.2f} -> {mean_z:.2f}"
    )
    assert max(float(b.pos.getZ()) for b in alive) < spawn_max_z

    ground = app.physics.ground_z or 0.0
    for b in alive:
        for v in tuple(b.pos) + tuple(b.quat):
            assert math.isfinite(float(v)), f"{b.name} has a non-finite pose"
        assert abs(float(b.pos.getX())) < 1e3
        assert abs(float(b.pos.getY())) < 1e3
        assert abs(float(b.pos.getZ())) < 1e3
        # No tunnelling: the lowest vertex is on the floor, not under it.
        assert b.lowest_z() >= ground - config.DEBRIS_GROUNDED_TOLERANCE, (
            f"{b.name} sank to {b.lowest_z():.3f} below ground {ground}"
        )

    snap = app.debris.snapshot()
    assert snap["active"] == 0, f"{snap['active']} bodies never came to rest"
    assert snap["max_speed"] < 1e-9
    assert snap["max_spin"] < 1e-9


# --------------------------------- (f) reachable by the damage/collision query
def test_spawned_debris_is_immediately_reachable_by_the_damage_query_path(app):
    """The hook the 'debris hurts the player' node will read.

    It is not enough that the bodies exist; the damage system has to be able
    to find them through the same query surface it uses for everything else,
    on the frame they spawn.
    """
    d = app.destructibles[TARGET_INDEX]
    lo, hi = d.world_bounds()
    centre = tuple((lo[i] + hi[i]) * 0.5 for i in range(3))
    span = max(hi[i] - lo[i] for i in range(3))

    assert app.damage.debris_near(centre, span) == []

    report = app.strike(d.default_impact_point())
    spawned = list(report.events[0].bodies)
    assert len(spawned) > 1

    found = app.damage.debris_near(centre, span * 2.0)
    assert len(found) == len(spawned)
    assert {b.name for b in found} == {b.name for b in spawned}

    # The report itself counts what the blast could see, and a Bullet node off
    # a contact manifold maps back to its DebrisBody.
    assert report.debris_in_radius >= 0
    for body in spawned[:20]:
        assert app.debris.body_for_node(body.node) is body

    # contactTest against a debris body finds its neighbours in the pile:
    # this is the live broadphase, so the bodies are genuinely collidable.
    touching = app.damage.debris_in_contact(spawned[0].node)
    assert isinstance(touching, list)

    # Let them land, then confirm they are still queryable once settled
    # (frozen debris is static geometry, but it is still in the world).
    for _ in range(60 * 20):
        app.step_frame(1.0 / 60.0)
    settled = app.damage.debris_near(centre, span * 3.0)
    assert len(settled) > 1
    world_names = _world_body_nodes(app)
    for b in settled:
        assert b.name in world_names


# ------------------------------------------------------ the loop and the burst
def test_the_main_update_loop_runs_a_burst_without_exception(app):
    """Drive the course, striking each structure as the player reaches it."""
    app.input_state = InputState(forward=True)
    destroyed_names = []
    for _ in range(60 * 50):
        app.step_frame(1.0 / 60.0)
        y = app.player.pos[1]
        for d in app.destructibles:
            if d.intact and d.world_bounds()[0][1] - y <= 18.0:
                report = app.strike(d.default_impact_point())
                destroyed_names.extend(report.destroyed)

    assert destroyed_names, "the run never destroyed anything"
    assert app.damage.total_destroyed == len(destroyed_names)
    assert app.demolitions == len(destroyed_names)
    assert app.damage.snapshot()["destroyed"] == len(destroyed_names)
    # No structure went down twice, and no NaN escaped into the world.
    assert len(set(destroyed_names)) == len(destroyed_names)
    for b in app.debris.live + app.debris.frozen:
        assert all(math.isfinite(float(v)) for v in tuple(b.pos))


def test_destruction_does_not_chain_into_neighbouring_structures(app):
    """Falling rubble must not demolish the next tower by itself."""
    target = app.destructibles[TARGET_INDEX]
    others = [d for d in app.destructibles if d is not target]

    app.strike(target.default_impact_point())
    for _ in range(60 * 25):
        app.step_frame(1.0 / 60.0)

    assert app.damage.total_destroyed == 1
    assert app.demolitions == 1
    for d in others:
        assert d.intact, f"{d.name} came down on its own"
        st = app.damage.state(d)
        assert st.accumulated == 0.0
        assert st.destroyed is False
        for name in d.proxy_names:
            assert name in app.physics.bodies


def test_a_blast_between_two_structures_damages_both_with_falloff(app):
    """Radius + falloff, and the near one takes more than the far one."""
    a, b = app.destructibles[0], app.destructibles[1]
    lo_a, hi_a = a.world_bounds()
    lo_b, _ = b.world_bounds()
    # Just outside a's far face, pointed down the course towards b.
    point = ((lo_a[0] + hi_a[0]) * 0.5, hi_a[1] + 1.0, 4.0)
    reach = (lo_b[1] - point[1]) + 2.0

    report = app.strike(point, amount=1.0, radius=reach)

    assert a.name in report.damaged and b.name in report.damaged
    assert report.damaged[a.name] > report.damaged[b.name]
    assert report.destroyed == ()
    assert a.intact and b.intact


def test_destroying_a_structure_twice_is_a_no_op(app):
    d = app.destructibles[TARGET_INDEX]
    first = app.strike(d.default_impact_point())
    assert first.any_destroyed
    count = app.debris.total_bodies

    second = app.strike(d.default_impact_point())
    assert second.destroyed == ()
    assert second.damaged == {}
    assert app.damage.total_destroyed == 1
    assert app.demolitions == 1
    assert app.debris.total_bodies == count


def test_direct_demolish_keeps_the_damage_tally_honest(app):
    """The scripted path (campaign, tests, debug key) must not desync it."""
    d = app.destructibles[TARGET_INDEX]
    app.demolish(d)
    st = app.damage.state(d)
    assert st.destroyed is True
    assert st.fraction == 1.0
    assert app.damage.total_destroyed == 1
    # And a later blast on the rubble does nothing.
    report = app.strike(d.default_impact_point())
    assert report.destroyed == ()
    assert app.damage.total_destroyed == 1


def test_damage_resolves_before_physics_advances_in_the_same_frame(app):
    """Deferred damage is rubble before a single substep of that frame runs.

    An intact structure and its own debris must never coexist for even one
    physics tick, or the chunks spawn interpenetrated with the proxies they
    replace and the pile detonates.
    """
    d = app.destructibles[TARGET_INDEX]
    st = app.damage.state(d)
    app.damage.damage_structure(d, st.threshold, resolve=False)

    assert st.at_threshold
    assert d.intact, "resolve=False should have deferred the demolition"
    assert app.damage.pending == [d]
    assert app.debris.total_bodies == 0
    steps_before = app.physics.step_count

    app.step_frame(1.0 / 60.0)

    assert app.physics.step_count > steps_before, "the frame ran no physics"
    assert d.intact is False
    assert app.damage.pending == []
    assert app.debris.total_bodies == d.chunk_count
    for name in d.proxy_names:
        assert name not in app.physics.bodies
    # Spawned this frame, so at most the frame's substeps have elapsed for it.
    for body in app.debris.live:
        assert body.spawn_step >= steps_before
