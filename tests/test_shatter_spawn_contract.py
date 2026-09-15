"""Node 11 contract assertions for the shatter-spawn physics module.

Every test here steps a **real** headless Bullet world - no window, no render
graph, no game loop - and asserts on measured numbers. This file exists to
prove the node 11 completion contract clause by clause, so each test is named
after the clause it discharges rather than after the code it touches.

I have no vision tool. Nothing here claims the debris *looks* right; that is
Rick's call at playtest. What is proven is behaviour: one body per descriptor,
masses ordered by volume, launch away from the blast, nonzero and non-identical
spin, a settled pile above the floor with finite coordinates, seed determinism,
and a reap API that actually empties the world.
"""

import math

import pytest
from panda3d.core import Vec3

from game import config, fracture
from game.debris import DESPAWNED, FROZEN, LIVE, DebrisField
from game.physics import PhysicsWorld

DT = config.FIXED_DT


# ----------------------------------------------------------------- helpers
def make_field(seed=20260914, ground=True, **kw):
    w = PhysicsWorld()
    if ground:
        w.add_ground_plane(0.0)
    return w, DebrisField(w, seed=seed, **kw)


def contract_spec(max_chunks=24, name="contract_tower"):
    """A modest tower: enough chunks to be a real pile, quick enough to step."""
    return fracture.StructureSpec(
        name=name,
        kind="tower",
        blocks=(fracture.Block("shaft", (0.0, 0.0, 6.0), (2.0, 2.0, 6.0)),),
        max_chunks=max_chunks,
    )


def run(world, field, seconds, player_y=None, max_age=None):
    for _ in range(int(round(seconds / DT))):
        world.step_fixed(1)
        field.update(DT, player_y=player_y, max_age=max_age)


def finite(v):
    return all(math.isfinite(float(x)) for x in v)


@pytest.fixture
def shattered():
    """A fresh tower, shattered from below-centre, not yet stepped."""
    w, f = make_field()
    result = fracture.fracture(contract_spec(24), seed=7)
    impact = (0.0, 0.0, 2.0)
    event = f.shatter(result, impact_point=impact, seed=42)
    return w, f, result, event, impact


# ============================================================== clause (1)
# one rigid body per chunk descriptor, masses ordered by chunk volume
# =========================================================================
def test_one_rigid_body_per_chunk_descriptor(shattered):
    w, f, result, event, _ = shattered
    assert len(result.chunks) > 1
    assert event.chunk_count == len(result.chunks)
    assert event.skipped_for_budget == 0, "budget must not distort this test"
    assert event.spawned == len(result.chunks)
    assert f.live_count == len(result.chunks)

    # Each body is a distinct Bullet node, attached to the world exactly once.
    nodes = [b.node for b in event.bodies]
    assert len({id(n) for n in nodes}) == len(nodes)
    assert len({b.name for b in event.bodies}) == len(nodes)

    # Each body maps back to a distinct source descriptor.
    assert sorted(b.chunk.index for b in event.bodies) == \
        sorted(c.index for c in result.chunks)


def test_masses_are_monotonically_ordered_by_chunk_volume(shattered):
    _, _, _, event, _ = shattered
    pairs = sorted((b.volume, b.base_mass) for b in event.bodies)
    assert len(pairs) > 2

    # Monotonic non-decreasing: a bigger chunk is never lighter than a
    # smaller one. (Non-strict because the minimum-mass floor can tie the
    # very smallest shards together.)
    for (v0, m0), (v1, m1) in zip(pairs, pairs[1:]):
        assert m1 >= m0 - 1e-9, f"volume {v1} -> mass {m1} < volume {v0} -> {m0}"

    assert pairs[-1][1] > pairs[0][1], "all masses identical - not volume-derived"
    for b in event.bodies:
        assert b.base_mass > 0.0
        assert float(b.node.getMass()) == pytest.approx(b.base_mass, rel=1e-6)


def test_collision_shape_matches_the_descriptor_dimensions(shattered):
    """Each body's hull is built from its own descriptor, at its own size.

    ``half_extents`` is the descriptor's conservative bounding box, not the
    exact vertex span, so the assertion is: the hull's real extent is positive,
    never exceeds the descriptor's box, and the shape's bounding sphere matches
    the descriptor's furthest vertex to within the collision margin.
    """
    _, _, _, event, _ = shattered
    for b in event.bodies:
        shapes = b.node.getShapes()
        assert len(shapes) == 1, "one convex shape per chunk, not a compound"

        for axis in range(3):
            span = max(float(v[axis]) for v in b.chunk.vertices) - \
                min(float(v[axis]) for v in b.chunk.vertices)
            assert span > 0.0, f"{b.name} is degenerate on axis {axis}"
            assert span <= float(b.chunk.half_extents[axis]) * 2.0 + 1e-6, (
                f"{b.name} hull exceeds its descriptor box on axis {axis}"
            )

        bounds = b.node.getShapeBounds()
        centre = bounds.getCenter()
        centre = (float(centre[0]), float(centre[1]), float(centre[2]))
        furthest = max(
            math.dist(centre, (float(v[0]), float(v[1]), float(v[2])))
            for v in b.chunk.vertices
        )
        # Bullet inflates the hull by the collision margin, hence the 12%%
        # tolerance rather than an exact match.
        assert float(bounds.getRadius()) == pytest.approx(furthest, rel=0.12), (
            f"{b.name} shape size does not track its descriptor"
        )

    # Different descriptors produce different shapes - not one shape reused.
    radii = {round(float(b.node.getShapeBounds().getRadius()), 4)
             for b in event.bodies}
    assert len(radii) > len(event.bodies) // 2


def test_total_mass_is_conserved_against_the_source_structure(shattered):
    _, _, result, event, _ = shattered
    expected = sum(
        max(c.volume * config.DEBRIS_DENSITY, config.DEBRIS_MIN_MASS)
        for c in result.chunks
    )
    assert event.total_mass == pytest.approx(expected, rel=1e-9)

    # And against the structure own volume, at the runtime debris density.
    assert event.total_mass == pytest.approx(
        result.chunk_volume * config.DEBRIS_DENSITY, rel=0.02
    )


# ============================================================== clause (2)
# immediately after shatter: outward velocities, nonzero varied spin
# =========================================================================
def test_velocities_point_away_from_the_blast_origin(shattered):
    _, _, _, event, impact = shattered
    origin = Vec3(*impact)

    radial = []
    for b in event.bodies:
        out = b.pos - origin
        if out.lengthSquared() < 1e-12:
            continue
        out.normalize()
        radial.append(float(b.linear_velocity().dot(out)))

    assert len(radial) > 5
    positive = sum(1 for r in radial if r > 0.0)
    assert positive / len(radial) >= 0.9, (
        f"only {positive}/{len(radial)} chunks launched outward"
    )
    assert sum(radial) / len(radial) > 1.0, "no net outward push"


def test_initial_speeds_are_nonzero_and_within_configured_bounds(shattered):
    _, _, _, event, _ = shattered
    jitter = 1.0 + config.DEBRIS_SPEED_JITTER
    for b in event.bodies:
        s = b.speed()
        assert s > 0.0, f"{b.name} spawned dead"
        assert s >= config.DEBRIS_SPEED_MIN * (1.0 - config.DEBRIS_SPEED_JITTER)
        assert s <= config.DEBRIS_SPEED_MAX * jitter + 1e-6
        assert finite(b.linear_velocity())


def test_angular_velocities_are_nonzero_and_not_identical(shattered):
    _, _, _, event, _ = shattered
    spins = [b.spin() for b in event.bodies]
    assert len(spins) > 5
    assert all(s > 0.0 for s in spins), "a chunk spawned without spin"
    assert all(s <= config.DEBRIS_SPIN_LIGHT * math.sqrt(3.0) + 1e-6
               for s in spins)

    # Non-identical: distinct magnitudes AND distinct axes.
    assert len({round(s, 6) for s in spins}) > len(spins) // 2
    axes = {tuple(round(float(c), 5) for c in b.angular_velocity())
            for b in event.bodies}
    assert len(axes) == len(event.bodies), "chunks share an angular velocity"


def test_speed_falls_off_with_distance_from_the_blast(shattered):
    """Chunks of comparable mass move slower the further out they start."""
    _, _, _, event, impact = shattered
    origin = Vec3(*impact)
    # Compare within a narrow mass band so mass is not the variable.
    masses = sorted(b.base_mass for b in event.bodies)
    mid = masses[len(masses) // 2]
    band = [b for b in event.bodies if 0.5 * mid <= b.base_mass <= 2.0 * mid]
    if len(band) < 4:
        pytest.skip("not enough same-mass chunks in this structure to compare")

    band.sort(key=lambda b: float((b.pos - origin).length()))
    near = band[: max(1, len(band) // 3)]
    far = band[-max(1, len(band) // 3):]
    near_speed = sum(b.speed() for b in near) / len(near)
    far_speed = sum(b.speed() for b in far) / len(far)
    assert near_speed > far_speed, (
        f"no distance falloff: near {near_speed:.2f} <= far {far_speed:.2f}"
    )


# ============================================================== clause (3)
# after a fixed simulated duration: everything at rest, above the floor,
# finite, nothing tunnelled
# =========================================================================
REST_SECONDS = 12.0


def test_every_body_comes_to_rest_above_the_ground_and_stays_finite():
    w, f = make_field()
    result = fracture.fracture(contract_spec(24), seed=7)
    f.shatter(result, impact_point=(0.0, 0.0, 2.0), seed=11)
    spawned = f.live_count
    assert spawned > 1

    run(w, f, REST_SECONDS)

    bodies = f.live + f.frozen
    assert len(bodies) == spawned, "bodies vanished without being reaped"

    for b in bodies:
        assert finite(b.pos), f"{b.name} has a non-finite position"
        assert finite(b.quat), f"{b.name} has a non-finite orientation"

        # Above the ground plane: the lowest vertex may sink only as far as
        # Bullet collision margin allows.
        assert b.lowest_z() >= -config.DEBRIS_GROUNDED_TOLERANCE, (
            f"{b.name} fell through the floor to z={b.lowest_z():.3f}"
        )
        assert float(b.pos.getZ()) > -1.0

        # At rest: live bodies below threshold, frozen bodies static by
        # construction.
        if b.state is LIVE:
            assert b.speed() <= config.DEBRIS_SETTLE_LINEAR, (
                f"{b.name} still moving at {b.speed():.3f} m/s after "
                f"{REST_SECONDS}s"
            )
            assert b.spin() <= config.DEBRIS_SETTLE_ANGULAR, (
                f"{b.name} still spinning at {b.spin():.3f} rad/s"
            )
        else:
            assert b.state is FROZEN
            assert b.is_asleep()

    assert f.kinetic_energy() <= 1e-3, "the pile still has kinetic energy"


def test_debris_arcs_up_before_it_lands_rather_than_dropping_straight():
    """Proves gravity is doing the work: a real arc, not a teleport to rest."""
    w, f = make_field()
    result = fracture.fracture(contract_spec(16), seed=5)
    f.shatter(result, impact_point=(0.0, 0.0, 2.0), seed=3)
    start = {b.name: float(b.pos.getZ()) for b in f.live}

    peak = dict(start)
    for _ in range(int(round(3.0 / DT))):
        w.step_fixed(1)
        f.update(DT)
        for b in f.live:
            peak[b.name] = max(peak[b.name], float(b.pos.getZ()))

    risers = sum(1 for n, z in peak.items() if z > start[n] + 0.05)
    assert risers > 0, "nothing was thrown upward at all"

    run(w, f, REST_SECONDS)
    lows = [b.lowest_z() for b in f.live + f.frozen]
    assert min(lows) >= -config.DEBRIS_GROUNDED_TOLERANCE
    assert max(lows) < 30.0, "something is stuck in the sky"


def test_nothing_tunnels_through_the_floor_at_any_point_in_the_flight():
    """Sampled every step, not just at the end - tunnelling is transient."""
    w, f = make_field()
    result = fracture.fracture(contract_spec(16), seed=5)
    f.shatter(result, impact_point=(0.0, 0.0, 1.5), seed=9)

    worst = math.inf
    for _ in range(int(round(8.0 / DT))):
        w.step_fixed(1)
        f.update(DT)
        for b in f.live + f.frozen:
            worst = min(worst, b.lowest_z())

    assert worst >= -config.DEBRIS_GROUNDED_TOLERANCE, (
        f"debris reached z={worst:.4f}, below the floor"
    )


# ============================================================== clause (4)
# seed determinism: same seed identical, different seeds measurably differ
# =========================================================================
def debris_field_state(seed, seconds=3.0):
    """Final pose of every chunk, keyed by descriptor index."""
    w, f = make_field()
    result = fracture.fracture(contract_spec(24), seed=7)
    f.shatter(result, impact_point=(0.0, 0.0, 2.0), seed=seed)
    run(w, f, seconds)
    return {
        b.chunk.index: tuple(round(float(v), 6) for v in b.pos)
        for b in f.live + f.frozen
    }


def test_the_same_seed_reproduces_an_identical_debris_field():
    a = debris_field_state(4242)
    b = debris_field_state(4242)
    assert a.keys() == b.keys()
    assert a == b, "identical seeds diverged"


def test_different_seeds_produce_a_measurably_different_debris_field():
    a = debris_field_state(4242)
    b = debris_field_state(9001)
    assert a.keys() == b.keys(), "seed changed the chunk set, not just the throw"
    assert a != b, "different seeds produced the same debris field"

    # "Measurably" different, not just floating-point noise: the mean chunk
    # displacement between the two fields is a real distance.
    deltas = [math.dist(a[i], b[i]) for i in a]
    mean = sum(deltas) / len(deltas)
    assert mean > 0.25, f"debris fields differ by only {mean:.4f} m on average"
    assert max(deltas) > 1.0


def test_field_seed_alone_also_drives_the_throw():
    """With no per-shatter seed, the field own seed is the determinant."""
    def go(field_seed):
        w, f = make_field(seed=field_seed)
        result = fracture.fracture(contract_spec(16), seed=7)
        f.shatter(result, impact_point=(0.0, 0.0, 2.0))
        return [tuple(round(float(v), 6) for v in b.linear_velocity())
                for b in sorted(f.live, key=lambda b: b.chunk.index)]

    assert go(1234) == go(1234)
    assert go(1234) != go(5678)


# ================================================================== reaping
# the lifecycle mechanism exists and empties the world (budget is node 12 job)
# =========================================================================
def test_reap_by_age_removes_old_debris_and_leaves_the_young():
    w, f = make_field()
    result = fracture.fracture(contract_spec(12), seed=5)
    f.shatter(result, impact_point=(0.0, 0.0, 2.0), seed=1)
    first = f.total_bodies
    assert first > 1

    run(w, f, 2.0)                      # advance the field clock
    f.shatter(result, impact_point=(0.0, 0.0, 2.0), seed=2)
    second = f.total_bodies - first
    assert second > 1

    removed = f.reap(max_age=1.0)       # only the first wave is older than 1s
    assert removed == first
    assert f.total_bodies == second
    assert all(b.age(f.clock) <= 1.0 for b in f.live + f.frozen)


def test_reap_by_distance_behind_the_player_removes_debris():
    w, f = make_field()
    result = fracture.fracture(contract_spec(12), seed=5)
    f.shatter(result, impact_point=(0.0, 0.0, 2.0), seed=1)
    run(w, f, 4.0)
    assert f.total_bodies > 1

    ahead = max(float(b.np.getY()) for b in f.live + f.frozen) + 10.0
    removed = f.reap(behind_y=ahead)
    assert removed > 0
    assert f.total_bodies == 0


def test_reap_by_cap_trims_to_the_ceiling_oldest_first():
    w, f = make_field()
    result = fracture.fracture(contract_spec(12), seed=5)
    f.shatter(result, impact_point=(0.0, 0.0, 2.0), seed=1)
    run(w, f, 1.0)
    old = {b.name for b in f.live + f.frozen}
    f.shatter(result, impact_point=(0.0, 0.0, 2.0), seed=2)
    total = f.total_bodies

    keep = total // 3
    removed = f.reap(cap=keep)
    assert removed == total - keep
    assert f.total_bodies == keep
    # The survivors are the young ones.
    survivors = {b.name for b in f.live + f.frozen}
    assert survivors.isdisjoint(old), "cap reaped the newest, not the oldest"


def test_reap_drives_the_live_count_to_zero_and_detaches_every_body():
    w, f = make_field()
    result = fracture.fracture(contract_spec(16), seed=5)
    event = f.shatter(result, impact_point=(0.0, 0.0, 2.0), seed=1)
    run(w, f, 2.0)
    assert f.live_count + f.frozen_count > 0

    before = w.world.getNumRigidBodies()
    removed = f.reap(cap=0)
    assert removed > 0
    assert f.live_count == 0
    assert f.frozen_count == 0
    assert f.total_bodies == 0
    assert f.snapshot()["live"] == 0

    # Actually detached from Bullet, not merely dropped from a list.
    assert w.world.getNumRigidBodies() == before - removed
    for b in event.bodies:
        assert b.state is DESPAWNED

    # And the world still steps cleanly with nothing in it.
    run(w, f, 0.5)
    assert f.total_bodies == 0


def test_reap_with_no_criteria_is_a_no_op():
    w, f = make_field()
    result = fracture.fracture(contract_spec(12), seed=5)
    f.shatter(result, impact_point=(0.0, 0.0, 2.0), seed=1)
    n = f.total_bodies
    assert f.reap() == 0
    assert f.total_bodies == n


def test_update_can_reap_by_age_inline():
    w, f = make_field()
    result = fracture.fracture(contract_spec(12), seed=5)
    f.shatter(result, impact_point=(0.0, 0.0, 2.0), seed=1)
    assert f.total_bodies > 1
    run(w, f, 3.0, max_age=1.0)
    assert f.total_bodies == 0, "age reaping did not run from update()"


def test_the_field_clock_tracks_simulated_time_not_wall_clock():
    w, f = make_field()
    assert f.clock == 0.0
    run(w, f, 2.0)
    assert f.clock == pytest.approx(2.0, abs=DT * 2)


# ========================================================== module hygiene
def test_the_module_is_free_of_game_loop_and_render_concerns():
    """Scope fence, asserted on the import graph rather than on prose.

    The physics core may import Panda3D and Bullet. It may not import the
    application framework (``direct.*``, ShowBase), and it must not reach for a
    global ``base`` / ``render`` / ``taskMgr``. Checked by parsing the module,
    so a docstring that merely *mentions* ShowBase does not trip it.
    """
    import ast

    import game.debris as debris

    tree = ast.parse(open(debris.__file__, encoding="utf-8").read())

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module)
                imported.update(f"{node.module}.{a.name}" for a in node.names)

    for name in imported:
        assert not name.startswith("direct"), f"{name} leaked into the core"
        assert "ShowBase" not in name, f"{name} leaked into the core"
        assert not name.startswith("game.app"), "the physics core imports the app"
        assert not name.startswith("game.render"), "the physics core renders"

    # No reliance on Panda's implicit globals, which only exist under
    # ShowBase. A name counts only if it is never bound anywhere in the module
    # - otherwise a local parameter innocently called `base` would trip this.
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                bound.add(a.asname or a.name.split(".")[0])

    loaded = {n.id for n in ast.walk(tree)
              if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    free = loaded - bound - set(dir(__builtins__))

    for banned in ("base", "render", "taskMgr", "loadPrcFileData", "globalClock"):
        assert banned not in free, (
            f"module reaches for the ShowBase global {banned!r}"
        )
