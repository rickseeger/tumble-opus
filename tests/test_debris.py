"""Runtime shatter / debris physics assertions.

Every test here steps a **real** Bullet world with no rendering: the chunk
descriptors from `game.fracture` become live rigid bodies and the assertions
are on measured positions, velocities, masses and wall-clock timings.

I have no vision tool, so nothing here claims the debris *looks* right - that
is Rick's call at playtest. What these prove is that it *behaves* right:
bodies are independent, they launch away from the blast, they fall at g, they
bounce proportionally to the restitution they were given, they come to a true
stop rather than jittering forever, mass tracks volume, the live-body cap is
never breached, and the whole thing steps faster than real time.
"""

import math
import time

import pytest
from panda3d.core import Vec3

from game import config, fracture
from game.debris import DESPAWNED, FROZEN, LIVE, DebrisField
from game.physics import PhysicsWorld

DT = config.FIXED_DT


# --------------------------------------------------------------- fixtures
def make_world(ground=True):
    w = PhysicsWorld()
    if ground:
        w.add_ground_plane(0.0)
    return w


def make_field(ground=True, seed=20260914, **kw):
    w = make_world(ground)
    return w, DebrisField(w, seed=seed, **kw)


def small_spec(max_chunks=24, name="probe_tower"):
    """A modest structure, so the tests stay quick but stay real."""
    return fracture.StructureSpec(
        name=name,
        kind="tower",
        blocks=(fracture.Block("shaft", (0.0, 0.0, 6.0), (2.0, 2.0, 6.0)),),
        max_chunks=max_chunks,
    )


def unit_cube_spec():
    """One chunk, exactly a 2x2x2 cube. Used for clean single-body physics."""
    return fracture.StructureSpec(
        name="probe_cube",
        kind="probe",
        blocks=(fracture.Block("b", (0.0, 0.0, 1.0), (1.0, 1.0, 1.0)),),
        max_chunks=1,
    )


def run(world, field, seconds, player_y=None):
    """Step the sim for `seconds` of simulated time, updating the field."""
    for _ in range(int(round(seconds / DT))):
        world.step_fixed(1)
        field.update(DT, player_y=player_y)


@pytest.fixture
def field_with_ground():
    w, f = make_field()
    return w, f


# ==================================================================== (1)
# shatter spawns one independent rigid body per chunk, within the budget
# =========================================================================
def test_shatter_spawns_one_rigid_body_per_chunk():
    w, f = make_field()
    result = fracture.fracture(small_spec(24), seed=7)
    assert len(result.chunks) > 1

    before = w.world.getNumRigidBodies()
    event = f.shatter(result, impact_point=(0, 0, 2.0))

    assert event.spawned == len(result.chunks), (
        f"spawned {event.spawned} bodies for {len(result.chunks)} chunks"
    )
    assert event.skipped_for_budget == 0
    assert f.live_count == len(result.chunks)
    # Every one of them is really in the Bullet world, not just in our list.
    assert w.world.getNumRigidBodies() == before + len(result.chunks)
    # And each body wraps a distinct chunk.
    assert len({b.chunk.index for b in event.bodies}) == len(result.chunks)
    assert len({b.name for b in event.bodies}) == len(result.chunks)


def test_shatter_never_exceeds_the_live_body_cap():
    """A 380-chunk tower against a 40-body cap: the cap wins."""
    w, f = make_field(max_live=40)
    result = fracture.fracture(fracture.tower_spec(), seed=7)
    assert len(result.chunks) > 40

    event = f.shatter(result, impact_point=(0, 0, 6.0))
    assert f.live_count <= 40
    assert event.spawned == 40
    assert event.skipped_for_budget == len(result.chunks) - 40
    # The budget goes to the biggest chunks: losing a shard is invisible,
    # losing the corner slab is not.
    kept = sorted((b.volume for b in event.bodies), reverse=True)
    expected = sorted((c.volume for c in result.chunks), reverse=True)[:40]
    assert kept == pytest.approx(expected)


def test_chunks_are_independently_simulated_not_one_welded_body():
    """After a second of sim every body must have its own distinct pose."""
    w, f = make_field()
    result = fracture.fracture(small_spec(24), seed=3)
    event = f.shatter(result, impact_point=(0, 0, 3.0))

    start = [tuple(round(float(v), 4) for v in b.pos) for b in event.bodies]
    run(w, f, 1.0)
    end = [tuple(round(float(v), 4) for v in b.pos) for b in event.bodies]

    assert all(a != b for a, b in zip(start, end)), "some chunk never moved"
    assert len(set(end)) == len(end), "two chunks share a position"
    # Orientations must diverge too - they are tumbling, not sliding as a slab.
    quats = {tuple(round(float(c), 3) for c in b.quat) for b in event.bodies}
    assert len(quats) >= len(event.bodies) - 1, "chunks are not rotating apart"

    # Displacements must differ: a welded body would move as one.
    disp = [
        math.dist(a, b) for a, b in zip(start, end)
    ]
    assert max(disp) - min(disp) > 0.25, (
        "every chunk moved the same distance - they are not independent"
    )


# ==================================================================== (2)
# launch: velocities point away from the impact, spins are nonzero + varied
# =========================================================================
def test_launch_velocities_point_away_from_the_impact_point():
    w, f = make_field()
    result = fracture.fracture(fracture.tower_spec(), seed=7)
    impact = Vec3(0.0, 0.0, 8.0)
    event = f.shatter(result, impact_point=impact)

    outward = 0
    for body in event.bodies:
        radial = body.pos - impact
        if radial.lengthSquared() < 1e-12:      # pragma: no cover
            continue
        radial.normalize()
        vel = Vec3(body.linear_velocity())
        assert vel.length() > 0.0, f"{body.name} was launched with zero speed"
        vel.normalize()
        if radial.dot(vel) > 0.0:
            outward += 1

    frac = outward / len(event.bodies)
    assert frac >= 0.95, (
        f"only {frac:.1%} of chunks launched away from the blast"
    )


def test_launch_direction_is_randomised_not_purely_radial():
    """Outward, yes - but not a sterile radial fan."""
    w, f = make_field()
    result = fracture.fracture(fracture.tower_spec(), seed=7)
    impact = Vec3(0.0, 0.0, 8.0)
    event = f.shatter(result, impact_point=impact)

    dots = []
    for body in event.bodies:
        radial = body.pos - impact
        radial.normalize()
        vel = Vec3(body.linear_velocity())
        vel.normalize()
        dots.append(float(radial.dot(vel)))

    assert min(dots) < 0.99, "every chunk flew exactly radially - no spread"
    spread = max(dots) - min(dots)
    assert spread > 0.05, f"launch direction spread is only {spread:.4f}"


def test_angular_velocity_is_nonzero_and_varied_across_chunks():
    w, f = make_field()
    result = fracture.fracture(fracture.tower_spec(), seed=7)
    event = f.shatter(result, impact_point=(0, 0, 8.0))

    spins = [b.spin() for b in event.bodies]
    assert min(spins) > 0.0, "a chunk was launched with no spin at all"
    assert max(spins) > 3.0, f"peak spin is only {max(spins):.3f} rad/s"

    # Varied, not one shared value.
    mean = sum(spins) / len(spins)
    sd = (sum((s - mean) ** 2 for s in spins) / len(spins)) ** 0.5
    assert sd > 0.5, f"spin standard deviation is only {sd:.4f} rad/s"
    assert len({round(s, 3) for s in spins}) > len(spins) * 0.9

    # Spin must be genuinely 3D, not all about one axis.
    for axis in range(3):
        comps = [abs(float(b.angular_velocity()[axis])) for b in event.bodies]
        assert max(comps) > 1.0, f"no chunk spins about axis {axis}"


def test_small_shards_spin_faster_than_heavy_slabs():
    """The 'skitter vs thud' contract, measured at launch."""
    w, f = make_field()
    result = fracture.fracture(fracture.tower_spec(), seed=7)
    event = f.shatter(result, impact_point=(0, 0, 8.0))

    ranked = sorted(event.bodies, key=lambda b: b.volume)
    n = max(1, len(ranked) // 5)
    light = sum(b.spin() for b in ranked[:n]) / n
    heavy = sum(b.spin() for b in ranked[-n:]) / n
    assert light > heavy * 1.3, (
        f"small shards spin at {light:.2f} rad/s, heavy slabs at "
        f"{heavy:.2f} - shards are not livelier"
    )


# ==================================================================== (3)
# gravity: with no ground, altitude decreases at approximately g
# =========================================================================
def _isolate(field, bodies, spacing=30.0):
    """Spread chunks far apart so nothing touches anything else.

    The fracture library tiles a structure's volume EXACTLY, so a freshly
    shattered structure's chunks all start face-to-face. That is correct for
    a collapse, but it makes a free-fall measurement meaningless: neighbours
    push on each other and the measured acceleration comes out well above g.
    Scattering them onto a wide grid is what isolates gravity.
    """
    for i, body in enumerate(bodies):
        col = i % 8
        row = i // 8
        body.np.setPos(col * spacing, row * spacing, float(body.np.getZ()))
        body.node.setLinearVelocity(Vec3(0, 0, 0))
        body.node.setAngularVelocity(Vec3(0, 0, 0))
    # Bullet caches broadphase AABBs; force them to catch up with the teleport.
    field.world.force_update_all_aabbs = True


def _free_fall_reference(z0, v0, seconds, damping=config.DEBRIS_LINEAR_DAMPING):
    """Integrate the exact model Bullet uses, in pure Python.

    Per substep Bullet does: ``v += g*dt``; ``v *= (1-damping)**dt``;
    ``x += v*dt``. Debris carries a little linear damping (it is what stops a
    rubble pile twitching), so the honest reference for "falls under real
    gravity" is damped free fall, not the textbook undamped one.

    One subtlety, measured rather than assumed: the transform Panda exposes on
    the NodePath lags Bullet's internal state by exactly one substep, because
    the motion state is synced at the *start* of a tick. So the position that
    pairs with a velocity read after N steps is the model's position after
    N-1. With that accounted for, this reference matches Bullet to better
    than 0.1% over a four second fall.

    Returns ``(z_as_panda_reports_it, v)``.
    """
    k = (1.0 - damping) ** DT
    v = float(v0)
    z = float(z0)
    steps = int(round(seconds / DT))
    z_lagged = z
    for _ in range(steps):
        z_lagged = z
        v = (v + config.GRAVITY_Z * DT) * k
        z += v * DT
    return z_lagged, v


def test_chunks_accelerate_downward_at_g_with_no_ground_beneath_them():
    """Short window, from rest: acceleration is g to within 1%.

    Measured over a quarter second the damping term is still negligible, so
    this is the clean 'altitude decreases at approximately g' assertion.
    """
    w, f = make_field(ground=False)
    result = fracture.fracture(small_spec(16), seed=5)
    event = f.shatter(result, impact_point=(0, 0, 6.0), impulse=0.0)

    # Scatter them and start from rest, so gravity is the only term acting.
    _isolate(f, event.bodies)

    z0 = [float(b.pos.getZ()) for b in event.bodies]
    seconds = 0.25
    w.step_fixed(int(seconds / DT))

    # What the configured damping costs over this window, stated up front so
    # the tolerance below is a measured fact rather than a fudge factor.
    _, model_v = _free_fall_reference(0.0, 0.0, seconds)
    damping_loss = abs(model_v / (config.GRAVITY_Z * seconds) - 1.0)
    assert damping_loss < 0.02, (
        f"debris damping costs {damping_loss:.2%} of g over {seconds}s - too "
        f"much to call this free fall any more"
    )

    for body, start in zip(event.bodies, z0):
        vz = float(body.linear_velocity().getZ())
        accel = vz / seconds
        # 'Approximately g': within the damping loss plus a little solver
        # noise. Nothing here is loosened to make a number pass - the exact
        # trajectory is pinned by the model test below.
        assert accel == pytest.approx(config.GRAVITY_Z, rel=0.02), (
            f"{body.name} accelerated at {accel:.4f} m/s^2, not "
            f"{config.GRAVITY_Z}"
        )
        assert accel == pytest.approx(model_v / seconds, rel=0.005), (
            f"{body.name} accelerated at {accel:.4f} m/s^2, damped free fall "
            f"predicts {model_v / seconds:.4f}"
        )
        drop = start - float(body.pos.getZ())
        model_z, _ = _free_fall_reference(start, 0.0, seconds)
        assert drop == pytest.approx(start - model_z, rel=0.01), (
            f"{body.name} fell {drop:.5f} m, the model predicts "
            f"{start - model_z:.5f} m"
        )
        textbook = 0.5 * 9.81 * seconds ** 2
        assert drop == pytest.approx(textbook, rel=0.06), (
            f"{body.name} fell {drop:.5f} m, textbook free fall says "
            f"{textbook:.5f} m"
        )
        assert float(body.pos.getZ()) < start, f"{body.name} did not descend"


def test_long_fall_matches_the_damped_free_fall_model_exactly():
    """Four seconds of fall, checked against the integrated model.

    Stronger than 'roughly g': it pins the trajectory, so a wrong gravity
    vector, a wrong timestep or a stray force would all show up.
    """
    w, f = make_field(ground=False)
    result = fracture.fracture(small_spec(12), seed=5)
    event = f.shatter(result, impact_point=(0, 0, 6.0), impulse=0.0)
    _isolate(f, event.bodies)

    z0 = [float(b.pos.getZ()) for b in event.bodies]
    seconds = 4.0
    w.step_fixed(int(seconds / DT))

    for body, start in zip(event.bodies, z0):
        want_z, want_v = _free_fall_reference(start, 0.0, seconds)
        got_z = float(body.pos.getZ())
        got_v = float(body.linear_velocity().getZ())
        assert got_v == pytest.approx(want_v, rel=0.005), (
            f"{body.name} vz={got_v:.5f}, damped model says {want_v:.5f}"
        )
        assert got_z == pytest.approx(want_z, rel=0.005), (
            f"{body.name} z={got_z:.5f}, damped model says {want_z:.5f}"
        )
        # Sanity: it really did fall a long way, this is not a no-op.
        assert start - got_z > 60.0


def test_gravity_is_the_only_vertical_force_on_a_lone_chunk():
    """No chunk hovers, rises, or is nudged sideways by nothing."""
    w, f = make_field(ground=False)
    result = fracture.fracture(unit_cube_spec(), seed=1)
    event = f.shatter(result, impact_point=(0, 0, 0), impulse=0.0,
                      origin=(0.0, 0.0, 50.0))
    body = event.bodies[0]
    body.node.setLinearVelocity(Vec3(0, 0, 0))
    body.node.setAngularVelocity(Vec3(0, 0, 0))
    x0, y0 = float(body.pos.getX()), float(body.pos.getY())

    # The NodePath transform lags Bullet by one substep, so the very first
    # read still shows the spawn position. Step once before sampling.
    w.step_fixed(1)
    prev = float(body.pos.getZ())
    for _ in range(int(3.0 / DT)):
        w.step_fixed(1)
        z = float(body.pos.getZ())
        assert z < prev, "the chunk stopped descending in open air"
        prev = z

    assert float(body.pos.getX()) == pytest.approx(x0, abs=1e-6)
    assert float(body.pos.getY()) == pytest.approx(y0, abs=1e-6)


def test_debris_never_tunnels_through_the_ground():
    """Sampled every step over a full collapse: no chunk escapes downward."""
    w, f = make_field()
    result = fracture.fracture(fracture.with_budget(fracture.arch_spec(), 90),
                               seed=4)
    f.shatter(result, impact_point=(0, 0, 4.0))

    worst = math.inf
    for _ in range(int(8.0 / DT)):
        w.step_fixed(1)
        f.update(DT, player_y=None)
        worst = min(worst, f.snapshot()["min_z"])

    # A discrete solver always dips a little on the impact step; passing
    # through the floor is what must never happen.
    assert worst > -0.15, f"a chunk sank {-worst:.4f} m below the ground"
    assert f.snapshot()["min_z"] > -0.05, "debris ended up below the ground"


# ==================================================================== (4)
# bounce: post-impact upward velocity positive, < impact speed, and it
#         tracks the configured restitution
# =========================================================================
def _drop_and_measure_bounce(restitution, height=6.0):
    """Drop one cube chunk from rest and measure impact/rebound speeds.

    Damping is switched off for this body only: damping is a settling aid,
    and leaving it on would confound the restitution measurement.
    """
    w, f = make_field()
    result = fracture.fracture(unit_cube_spec(), seed=1)
    assert len(result.chunks) == 1
    event = f.shatter(result, impact_point=(0, 0, 0), impulse=0.0,
                      origin=(0.0, 0.0, height - 1.0))
    body = event.bodies[0]
    node = body.node
    node.setRestitution(restitution)
    node.setLinearVelocity(Vec3(0, 0, 0))
    node.setAngularVelocity(Vec3(0, 0, 0))
    node.setLinearDamping(0.0)
    node.setAngularDamping(0.0)
    node.setDeactivationEnabled(False)

    impact = 0.0
    hit = None
    rebound = 0.0
    for i in range(int(4.0 / DT)):
        w.step_fixed(1)
        vz = float(body.linear_velocity().getZ())
        if hit is None:
            impact = min(impact, vz)
            if vz > -0.01 and impact < -3.0:
                hit = i
        elif i - hit <= 40:
            rebound = max(rebound, vz)
        else:
            break
    assert hit is not None, "the chunk never reached the ground"
    return -impact, rebound


def test_a_dropped_chunk_bounces_upward_but_slower_than_it_arrived():
    impact, rebound = _drop_and_measure_bounce(0.42)

    assert impact == pytest.approx(math.sqrt(2 * 9.81 * 5.0), rel=0.05), (
        f"impact speed {impact:.3f} m/s does not match the 5 m drop"
    )
    assert rebound > 0.0, "the chunk did not bounce at all - it just splatted"
    assert rebound < impact, (
        f"rebound {rebound:.3f} m/s >= impact {impact:.3f} m/s - energy was "
        f"created by the collision"
    )


def test_rebound_speed_is_consistent_with_the_configured_restitution():
    """Bounce must be *caused* by restitution, not by penetration recovery.

    This is the test that caught the real bug: Bullet combines restitution
    multiplicatively, so with the ground plane left at its default 0 the
    rebound was ~0.47 m/s for EVERY body restitution from 0.0 to 0.6 - pure
    solver noise, completely decoupled from the setting. The fix is
    `DebrisField.prepare_ground`, which gives the ground a real restitution.
    """
    samples = [(e, _drop_and_measure_bounce(e)) for e in (0.0, 0.2, 0.6, 1.0)]

    ratios = [rebound / impact for _, (impact, rebound) in samples]

    # Monotone in restitution: more bouncy setting -> more bounce. This is
    # what proves the knob is actually connected.
    for (e_a, _), (e_b, _), r_a, r_b in zip(
        samples, samples[1:], ratios, ratios[1:]
    ):
        assert r_b > r_a, (
            f"restitution {e_b} rebounded at ratio {r_b:.4f}, no better than "
            f"{e_a} at {r_a:.4f} - restitution is not wired through"
        )

    # A real signal, not noise: the bounciest setting must clearly beat the
    # dead one.
    assert ratios[-1] > ratios[0] * 4.0, (
        f"restitution 1.0 ({ratios[-1]:.4f}) barely beats 0.0 "
        f"({ratios[0]:.4f})"
    )

    # And never above the physical ceiling e_body * e_ground: a collision may
    # not manufacture energy.
    for (e, (impact, rebound)), ratio in zip(samples, ratios):
        ceiling = e * config.DEBRIS_GROUND_RESTITUTION
        assert ratio <= ceiling + 0.05, (
            f"restitution {e} rebounded at ratio {ratio:.4f}, above the "
            f"e_body*e_ground ceiling of {ceiling:.4f}"
        )


def test_the_ground_has_a_real_restitution_so_debris_can_bounce_at_all():
    w, f = make_field()
    assert w.ground_np is not None
    assert float(w.ground_np.node().getRestitution()) == pytest.approx(
        config.DEBRIS_GROUND_RESTITUTION
    )
    assert config.DEBRIS_GROUND_RESTITUTION > 0.0, (
        "a ground restitution of 0 multiplies every bounce to nothing"
    )


# ==================================================================== (5)
# settle: after a bounded duration everything is at rest and asleep
# =========================================================================
@pytest.mark.parametrize("archetype", sorted(fracture.ARCHETYPES))
def test_every_structure_settles_completely_within_the_time_budget(archetype):
    """Thirty seconds of simulated time, then absolute stillness.

    'Near zero' is not good enough here - settled debris is frozen to static
    geometry, so the velocities are exactly 0 and nothing can jitter.
    """
    w, f = make_field()
    spec = fracture.ARCHETYPES[archetype]()
    result = fracture.fracture(spec, seed=7)
    lo, hi = result.bounds
    f.shatter(result, impact_point=(0.0, 0.0, lo[2] + (hi[2] - lo[2]) * 0.22))

    run(w, f, 30.0)

    snap = f.snapshot()
    bodies = f.live + f.frozen
    assert bodies, "all the debris vanished"
    assert snap["max_speed"] < 1e-9, (
        f"{archetype}: still moving at {snap['max_speed']:.6f} m/s after 30 s"
    )
    assert snap["max_spin"] < 1e-9, (
        f"{archetype}: still spinning at {snap['max_spin']:.6f} rad/s"
    )
    assert snap["kinetic_energy"] == 0.0
    assert snap["active"] == 0, f"{archetype}: {snap['active']} bodies awake"
    assert all(b.is_asleep() for b in bodies)


def test_settled_debris_does_not_creep_or_jitter_afterwards():
    """Freeze the pose, run another 20 s, compare. Zero drift allowed."""
    w, f = make_field()
    result = fracture.fracture(fracture.with_budget(fracture.cluster_spec(), 120),
                               seed=9)
    f.shatter(result, impact_point=(0, 0, 3.0))
    run(w, f, 30.0)

    bodies = f.live + f.frozen
    before = [(tuple(float(v) for v in b.pos),
               tuple(float(c) for c in b.quat)) for b in bodies]

    run(w, f, 20.0)

    after = [(tuple(float(v) for v in b.pos),
              tuple(float(c) for c in b.quat)) for b in bodies]

    worst = 0.0
    for (p0, _), (p1, _) in zip(before, after):
        worst = max(worst, math.dist(p0, p1))
    assert worst < 1e-6, f"settled debris crept {worst:.8f} m over 20 s"
    assert before == after, "settled debris changed pose"


def test_kinetic_energy_decays_toward_rest():
    """The pile must calm down, not wind itself up."""
    w, f = make_field()
    result = fracture.fracture(fracture.with_budget(fracture.tower_spec(), 150),
                               seed=7)
    f.shatter(result, impact_point=(0, 0, 8.0))

    samples = []
    for _ in range(30):                         # 30 x 1 s
        run(w, f, 1.0)
        samples.append(f.kinetic_energy())

    peak = max(samples)
    assert peak > 0.0
    # The tail must be genuinely dead, and the second half must be far
    # quieter than the first: no runaway.
    assert samples[-1] == 0.0, (
        f"kinetic energy is still {samples[-1]:.3f} J after 30 s"
    )
    first_half = sum(samples[:15]) / 15.0
    second_half = sum(samples[15:]) / 15.0
    assert second_half < first_half * 0.05, (
        f"energy barely decayed: {first_half:.1f} J -> {second_half:.1f} J"
    )


def test_frozen_debris_is_static_geometry_that_still_collides():
    """Retired debris must stay queryable for the later damage node."""
    w, f = make_field()
    result = fracture.fracture(fracture.with_budget(fracture.cluster_spec(), 60),
                               seed=2)
    f.shatter(result, impact_point=(0, 0, 3.0))
    run(w, f, 25.0)

    assert f.frozen_count > 0, "nothing ever settled"
    for body in f.frozen:
        assert body.state is FROZEN
        assert body.mass == 0.0, "frozen debris still has mass"
        assert not body.node.isActive()
        assert body.is_asleep()
        # Still attached to the world, so contact/ray tests still find it.
        assert not body.np.isEmpty()
    names = {w.world.getRigidBody(i).getName()
             for i in range(w.world.getNumRigidBodies())}
    for body in f.frozen:
        assert body.name in names, (
            f"{body.name} was frozen out of the Bullet world entirely"
        )


def test_settle_requires_ground_contact_so_nothing_freezes_in_mid_air():
    """A chunk at the apex of its arc is slow - it must not freeze there."""
    w, f = make_field()
    result = fracture.fracture(unit_cube_spec(), seed=1)
    event = f.shatter(result, impact_point=(0, 0, 0), impulse=0.0,
                      origin=(0.0, 0.0, 39.0))
    body = event.bodies[0]
    body.node.setLinearVelocity(Vec3(0, 0, 0))
    body.node.setAngularVelocity(Vec3(0, 0, 0))

    # It starts stationary, high above the ground, for longer than the
    # settle dwell time. It must still be live and falling.
    run(w, f, config.DEBRIS_SETTLE_TIME + 0.3)
    assert body.state is LIVE, "a chunk froze in mid-air"
    assert float(body.pos.getZ()) < 40.0
    assert body.linear_velocity().getZ() < -1.0

    run(w, f, 20.0)
    assert body.state is FROZEN, "the chunk never settled once it landed"
    assert body.lowest_z() == pytest.approx(0.0, abs=0.15)


# ==================================================================== (6)
# mass scales with chunk volume
# =========================================================================
def test_mass_scales_linearly_with_chunk_volume():
    w, f = make_field()
    result = fracture.fracture(fracture.tower_spec(), seed=7)
    event = f.shatter(result, impact_point=(0, 0, 8.0))

    for body in event.bodies:
        expected = max(body.volume * config.DEBRIS_DENSITY,
                       config.DEBRIS_MIN_MASS)
        assert body.base_mass == pytest.approx(expected, rel=1e-9)
        assert float(body.node.getMass()) == pytest.approx(expected, rel=1e-6)

    # Doubling the volume doubles the mass - a real ratio, not a constant.
    ranked = sorted(event.bodies, key=lambda b: b.volume)
    light, heavy = ranked[0], ranked[-1]
    assert heavy.volume > light.volume * 5.0, "chunk sizes are not varied"
    assert heavy.base_mass / light.base_mass == pytest.approx(
        heavy.volume / light.volume, rel=1e-6
    ), "mass is not proportional to volume"

    masses = [b.base_mass for b in event.bodies]
    assert max(masses) / min(masses) > 5.0, (
        "every chunk has effectively the same mass - nothing will feel heavy"
    )


def test_heavy_slabs_are_grippier_and_deader_than_light_shards():
    """The material response is blended by size, not one-size-fits-all."""
    w, f = make_field()
    result = fracture.fracture(fracture.tower_spec(), seed=7)
    event = f.shatter(result, impact_point=(0, 0, 8.0))

    ranked = sorted(event.bodies, key=lambda b: b.volume)
    n = max(1, len(ranked) // 5)
    light = ranked[:n]
    heavy = ranked[-n:]

    lr = sum(float(b.node.getRestitution()) for b in light) / n
    hr = sum(float(b.node.getRestitution()) for b in heavy) / n
    lf = sum(float(b.node.getFriction()) for b in light) / n
    hf = sum(float(b.node.getFriction()) for b in heavy) / n

    assert lr > hr, f"shards ({lr:.3f}) are no bouncier than slabs ({hr:.3f})"
    assert hf > lf, f"slabs ({hf:.3f}) grip no harder than shards ({lf:.3f})"


def test_heavy_chunks_launch_slower_than_light_ones_for_one_blast():
    """v = J/m, so the same impulse moves a shard far more than a slab."""
    w, f = make_field()
    result = fracture.fracture(fracture.tower_spec(), seed=7)
    event = f.shatter(result, impact_point=(0, 0, 8.0))

    ranked = sorted(event.bodies, key=lambda b: b.volume)
    n = max(1, len(ranked) // 5)
    light = sum(b.speed() for b in ranked[:n]) / n
    heavy = sum(b.speed() for b in ranked[-n:]) / n
    assert light > heavy, (
        f"light chunks launch at {light:.2f} m/s, heavy at {heavy:.2f} - "
        f"mass is not affecting the launch"
    )


def test_collision_margin_scales_with_the_chunk():
    """Bullet's flat default margin detonates an exactly-tiled structure."""
    result = fracture.fracture(fracture.tower_spec(), seed=7)
    margins = [DebrisField.margin_for(c) for c in result.chunks]
    assert min(margins) >= config.DEBRIS_MARGIN_MIN
    assert max(margins) <= config.DEBRIS_MARGIN_MAX
    assert max(margins) > min(margins), "margin is not scaling at all"
    # Never larger than the chunk it wraps.
    for chunk, m in zip(result.chunks, margins):
        assert m < min(chunk.half_extents), (
            f"chunk {chunk.index} has margin {m} but half extent "
            f"{min(chunk.half_extents)}"
        )


# ==================================================================== (7)
# debris cap and retirement across several structures in sequence
# =========================================================================
def test_several_structures_in_sequence_never_exceed_the_cap():
    w, f = make_field()
    specs = [fracture.tower_spec(), fracture.slab_spec(), fracture.arch_spec(),
             fracture.cluster_spec(), fracture.tower_spec()]

    peak = 0
    total_chunks = 0
    for i, spec in enumerate(specs):
        result = fracture.fracture(spec, seed=100 + i)
        total_chunks += len(result.chunks)
        f.shatter(result, impact_point=(0.0, i * 40.0, 5.0),
                  origin=(0.0, i * 40.0, 0.0))
        assert f.live_count <= f.max_live, (
            f"cap breached immediately after shatter {i}: {f.live_count}"
        )
        for _ in range(int(2.5 / DT)):
            w.step_fixed(1)
            f.update(DT, player_y=i * 40.0)
            peak = max(peak, f.live_count)
            assert f.live_count <= f.max_live, (
                f"cap breached mid-sim: {f.live_count} > {f.max_live}"
            )

    assert total_chunks > f.max_live * 3, "the test did not actually stress it"
    assert peak <= f.max_live
    assert peak > f.max_live * 0.5, "the cap was never even approached"
    # Retirement actually happened, rather than shatters being refused.
    assert f.total_frozen + f.total_despawned > 0


def test_debris_behind_the_player_is_retired():
    w, f = make_field()
    result = fracture.fracture(fracture.with_budget(fracture.cluster_spec(), 120),
                               seed=6)
    f.shatter(result, impact_point=(0, 0, 3.0))
    run(w, f, 20.0, player_y=0.0)

    remaining = f.total_bodies
    assert remaining > 0, "debris vanished before the player moved on"

    # Drive well past it.
    run(w, f, 3.0, player_y=config.DEBRIS_DESPAWN_BEHIND + 200.0)

    assert f.total_bodies == 0, (
        f"{f.total_bodies} bodies survived the player driving 200 m past"
    )
    assert f.total_despawned >= remaining
    assert w.world.getNumRigidBodies() == 1, "only the ground should remain"


def test_debris_just_behind_the_player_is_kept():
    """Retirement must not delete rubble the player can still turn and see."""
    w, f = make_field()
    result = fracture.fracture(fracture.with_budget(fracture.cluster_spec(), 60),
                               seed=6)
    f.shatter(result, impact_point=(0, 0, 3.0))
    run(w, f, 20.0, player_y=0.0)
    before = f.total_bodies
    assert before > 0

    # Half a despawn radius ahead: still in view, must survive.
    run(w, f, 2.0, player_y=config.DEBRIS_DESPAWN_BEHIND * 0.5)
    assert f.total_bodies == before, "debris still in view was despawned"


def test_frozen_pile_is_bounded():
    w, f = make_field(max_frozen=50)
    for i in range(4):
        result = fracture.fracture(
            fracture.with_budget(fracture.cluster_spec(), 100), seed=200 + i
        )
        f.shatter(result, impact_point=(0.0, i * 30.0, 3.0),
                  origin=(0.0, i * 30.0, 0.0))
        run(w, f, 12.0)
        assert f.frozen_count <= 50, f"frozen pile grew to {f.frozen_count}"


def test_clear_removes_every_body():
    w, f = make_field()
    result = fracture.fracture(fracture.with_budget(fracture.tower_spec(), 80),
                               seed=8)
    f.shatter(result, impact_point=(0, 0, 8.0))
    run(w, f, 3.0)
    assert f.total_bodies > 0

    f.clear()
    assert f.total_bodies == 0
    assert w.world.getNumRigidBodies() == 1
    w.step_fixed(60)                             # must not crash afterwards


# ==================================================================== (8)
# performance: a full structure's debris steps faster than real time
# =========================================================================
def test_full_structure_debris_steps_faster_than_real_time(capsys):
    """The budget assertion, with the measured numbers printed.

    Panda's fixed step is 1/120 s; a frame at 60 fps runs two of them. If a
    step costs more than the wall-clock time it represents, the sim can never
    keep up and a tower collapse turns into a slideshow.
    """
    w, f = make_field()
    result = fracture.fracture(fracture.tower_spec(), seed=7)
    event = f.shatter(result, impact_point=(0, 0, 6.0))
    assert event.spawned >= 200, "not a representative debris load"

    seconds = 10.0
    steps = int(seconds / DT)
    started = time.perf_counter()
    for _ in range(steps):
        w.step_fixed(1)
        f.update(DT, player_y=None)
    wall = time.perf_counter() - started

    ms_per_step = wall / steps * 1000.0
    budget_ms = DT * 1000.0
    factor = seconds / wall

    with capsys.disabled():
        print()
        print(f"  [debris benchmark] bodies={event.spawned} "
              f"steps={steps} dt={budget_ms:.3f} ms")
        print(f"  [debris benchmark] wall={wall:.3f} s "
              f"per_step={ms_per_step:.4f} ms")
        print(f"  [debris benchmark] real-time factor={factor:.2f}x "
              f"headroom={budget_ms / ms_per_step:.2f}x")

    assert factor > 1.0, (
        f"a full tower's debris runs at {factor:.2f}x real time - slower than "
        f"the wall clock ({ms_per_step:.3f} ms per {budget_ms:.3f} ms step)"
    )
    assert ms_per_step < budget_ms, (
        f"{ms_per_step:.4f} ms per step exceeds the {budget_ms:.3f} ms budget"
    )


def test_shatter_itself_is_cheap_enough_to_do_mid_frame():
    """Building the bodies must not stall the frame it happens on."""
    w, f = make_field()
    result = fracture.fracture(fracture.tower_spec(), seed=7)
    event = f.shatter(result, impact_point=(0, 0, 6.0))
    assert event.build_seconds < 0.05, (
        f"shatter took {event.build_seconds * 1000:.1f} ms to build "
        f"{event.spawned} bodies"
    )


def test_several_simultaneous_collapses_stay_real_time(capsys):
    """Worst case: three structures shattered inside one second."""
    w, f = make_field()
    for i in range(3):
        result = fracture.fracture(fracture.tower_spec(), seed=300 + i)
        f.shatter(result, impact_point=(i * 30.0, 0.0, 6.0),
                  origin=(i * 30.0, 0.0, 0.0))
        w.step_fixed(int(0.3 / DT))

    assert f.live_count <= f.max_live

    steps = int(5.0 / DT)
    started = time.perf_counter()
    for _ in range(steps):
        w.step_fixed(1)
        f.update(DT, player_y=0.0)
    wall = time.perf_counter() - started
    factor = (steps * DT) / wall

    with capsys.disabled():
        print(f"  [debris benchmark] 3 simultaneous collapses: "
              f"{factor:.2f}x real time, {wall / steps * 1000:.4f} ms/step")

    assert factor > 1.0, (
        f"three simultaneous collapses run at {factor:.2f}x real time"
    )


# ==================================================================== API
# the surface the weapon/damage node and the campaign node call
# =========================================================================
def test_shatter_accepts_a_spec_directly():
    """Convenience path: hand it a spec and it fractures on the spot."""
    w, f = make_field()
    spec = small_spec(20)
    event = f.shatter(spec, impact_point=(0, 0, 3.0))
    assert event.structure == spec.name
    assert event.spawned > 1
    assert f.live_count == event.spawned


def test_shatter_rejects_something_that_is_not_a_structure():
    w, f = make_field()
    with pytest.raises(TypeError):
        f.shatter("a tower, honest", impact_point=(0, 0, 1.0))


def test_impulse_controls_how_hard_the_debris_is_thrown():
    speeds = []
    for impulse in (2.0e4, 9.0e4, 4.0e5):
        w, f = make_field()
        result = fracture.fracture(small_spec(24), seed=7)
        event = f.shatter(result, impact_point=(0, 0, 3.0), impulse=impulse)
        speeds.append(sum(b.speed() for b in event.bodies) / event.spawned)
    assert speeds[0] < speeds[1] < speeds[2], (
        f"mean launch speed did not rise with impulse: {speeds}"
    )


def test_bodies_near_finds_debris_for_the_damage_node():
    w, f = make_field()
    result = fracture.fracture(small_spec(24), seed=7)
    event = f.shatter(result, impact_point=(0, 0, 3.0))

    near = f.bodies_near((0.0, 0.0, 6.0), 100.0)
    assert len(near) == event.spawned, "the broad phase missed live debris"
    assert f.bodies_near((500.0, 500.0, 0.0), 1.0) == []

    # Frozen debris is still findable - that is the whole point of freezing
    # rather than despawning.
    run(w, f, 25.0)
    assert f.frozen_count > 0
    still = f.bodies_near((0.0, 0.0, 0.0), 500.0)
    assert len(still) == f.total_bodies


def test_body_for_node_round_trips():
    w, f = make_field()
    result = fracture.fracture(small_spec(12), seed=7)
    event = f.shatter(result, impact_point=(0, 0, 3.0))
    for body in event.bodies:
        assert f.body_for_node(body.node) is body
    assert f.body_for_node(w.ground_np.node()) is None


def test_contact_test_reports_live_debris_touching_a_node():
    """The hook the damage node will use on a weapon/projectile body."""
    w, f = make_field()
    result = fracture.fracture(fracture.with_budget(fracture.cluster_spec(), 60),
                               seed=2)
    f.shatter(result, impact_point=(0, 0, 3.0))
    run(w, f, 3.0)

    touching = f.contact_test(w.ground_np.node())
    assert touching, "no live debris reported as touching the ground"
    assert all(t.state in (LIVE, FROZEN) for t in touching)
    assert all(f.body_for_node(t.node) is t for t in touching)


def test_settled_debris_is_still_hittable_by_a_dynamic_probe():
    """Frozen debris is static, and Bullet generates no static-static
    manifolds - so a ground `contactTest` finds nothing once the pile has
    settled. That is correct Bullet behaviour, not a bug in the freeze: what
    matters for the damage node is that a *dynamic* body (a projectile, the
    player, a fresh chunk) still collides with retired rubble. This proves it
    does, both by contact query and by the probe physically landing on top of
    the pile instead of falling through it.
    """
    w, f = make_field()
    result = fracture.fracture(fracture.with_budget(fracture.cluster_spec(), 60),
                               seed=2)
    f.shatter(result, impact_point=(0, 0, 3.0))
    run(w, f, 25.0)
    assert f.frozen_count > 0 and f.live_count == 0

    # Static/static really does report nothing - documented, not papered over.
    assert f.contact_test(w.ground_np.node()) == []

    # 1. A ray fired straight down at a piece of rubble must hit that rubble,
    #    not sail through it to the ground. This is the query the damage node
    #    will actually use for a hitscan weapon.
    target = max(f.frozen, key=lambda b: float(b.np.getZ()))
    tx, ty, tz = (float(v) for v in target.np.getPos())
    hit = w.world.rayTestClosest(Vec3(tx, ty, tz + 40.0), Vec3(tx, ty, -1.0))
    assert hit.hasHit(), "a ray fired at settled debris hit nothing at all"
    hit_body = f.body_for_node(hit.getNode())
    assert hit_body is not None, (
        f"the ray hit {hit.getNode().getName()!r} instead of any debris - "
        f"frozen rubble is not collidable"
    )
    assert hit_body.state is FROZEN
    assert float(hit.getHitPos().getZ()) > 0.1, "the ray hit the bare ground"

    # 2. And a dynamic body dropped onto it registers real contacts with the
    #    frozen rubble while falling past / landing on it.
    half = 0.25
    probe = w.add_dynamic_box("probe", (tx, ty, tz + 6.0),
                              (half, half, half), mass=20.0)
    saw_frozen_contact = False
    for _ in range(int(5.0 / DT)):
        w.step_fixed(1)
        f.update(DT, player_y=None)
        touching = f.contact_test(probe.node())
        if touching:
            saw_frozen_contact = True
            assert all(t.state is FROZEN for t in touching)
    assert saw_frozen_contact, (
        "a dynamic probe dropped onto the rubble never touched any of it"
    )
    # It may slide off the pile onto the floor, which is fine - what must not
    # happen is it passing through the world.
    assert float(probe.getZ()) >= half - 0.05, "the probe fell through the floor"
    assert float(probe.node().getLinearVelocity().length()) < 1.0


def test_shatter_event_reports_what_it_did():
    w, f = make_field(max_live=50)
    result = fracture.fracture(fracture.tower_spec(), seed=7)
    event = f.shatter(result, impact_point=(1.0, 2.0, 3.0), impulse=5.0e4)

    assert event.structure == "tall_tower"
    assert event.impact_point == (1.0, 2.0, 3.0)
    assert event.impulse == 5.0e4
    assert event.chunk_count == len(result.chunks)
    assert event.spawned + event.skipped_for_budget == event.chunk_count
    assert event.total_mass > 0.0
    assert event.build_seconds > 0.0
    assert f.events == [event]


def test_wireframe_segments_track_the_live_pose():
    """The 1A glowing-wireframe render hook: real edges, real transform.

    Whether it *looks* good is Rick's call. What is checkable is that the
    segments exist, match the chunk's edge list, and move with the body.
    """
    w, f = make_field()
    result = fracture.fracture(small_spec(8), seed=7)
    event = f.shatter(result, impact_point=(0, 0, 3.0))
    body = event.bodies[0]

    segs = body.wireframe_segments()
    assert len(segs) == len(body.chunk.edges)
    assert all(len(s) == 2 and len(s[0]) == 3 for s in segs)
    assert any(math.dist(a, b) > 1e-6 for a, b in segs), "degenerate wireframe"

    # Segments are in world space around the body.
    pos = tuple(float(v) for v in body.pos)
    assert max(math.dist(pos, a) for a, _ in segs) < 20.0

    run(w, f, 1.0)
    moved = body.wireframe_segments()
    assert moved != segs, "the wireframe did not follow the body"


def test_field_does_not_disturb_the_existing_course():
    """Adding debris must not break the world the player walks on."""
    from game.world import build_course

    w = PhysicsWorld()
    build_course(w)
    before = {name: tuple(float(v) for v in np_.getPos())
              for name, np_ in w.bodies.items()}

    f = DebrisField(w)
    f.shatter(small_spec(20), impact_point=(0.0, 20.0, 3.0),
              origin=(0.0, 20.0, 0.0))
    run(w, f, 5.0)

    for name, pos in before.items():
        now = tuple(float(v) for v in w.bodies[name].getPos())
        assert now == pytest.approx(pos, abs=1e-6), (
            f"static course body {name} was moved by the debris"
        )


def test_shatter_is_deterministic_for_a_given_seed():
    def go():
        w, f = make_field()
        result = fracture.fracture(small_spec(24), seed=7)
        f.shatter(result, impact_point=(0, 0, 3.0), seed=42)
        run(w, f, 2.0)
        return [tuple(round(float(v), 5) for v in b.pos)
                for b in sorted(f.live + f.frozen, key=lambda b: b.chunk.index)]

    assert go() == go()


# ==================================================================== demo
def test_headless_demo_entrypoint_runs_and_reports_each_phase(capsys):
    import tools_debris_demo

    rc = tools_debris_demo.main(
        ["--structure", "cluster", "--seed", "7", "--seconds", "6"]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "block_cluster" in out
    assert "spawned bodies" in out
    for phase in ("launch", "airborne", "first-impacts", "tumbling", "rolling"):
        assert phase in out, f"demo never reported the {phase!r} phase"
    assert "real time" in out


def test_demo_benchmark_mode_prints_numbers(capsys):
    import tools_debris_demo

    stats = tools_debris_demo.benchmark(seed=7, seconds=4.0)
    out = capsys.readouterr().out

    assert stats["spawned"] > 200
    assert stats["realtime_factor"] > 1.0, (
        f"benchmark reports {stats['realtime_factor']:.2f}x real time"
    )
    assert stats["ms_per_step"] < stats["budget_ms"]
    assert "real-time factor" in out
    assert "per step" in out
