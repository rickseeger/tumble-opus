"""The destruction path as the game actually runs it.

`test_debris.py` proves the physics. This file proves the *wiring*: that a
placed structure is solid until it is demolished, that demolishing it removes
its collision proxy on the same frame its debris appears, that the real game
loop can bring several structures down in sequence without breaching the
body budget, and that the whole thing steps faster than real time while doing
it.

Everything here is headless - `window-type none`, no GL, no window. Numbers
are measured, not asserted into existence.
"""

import math
import time

import pytest
from panda3d.core import Vec3

from game import config, fracture, structures
from game.debris import FROZEN, LIVE, DebrisField
from game.physics import PhysicsWorld

DT = config.FIXED_DT


def make_world():
    w = PhysicsWorld()
    w.add_ground_plane(0.0)
    return w


def run(world, field, seconds, player_y=None):
    for _ in range(int(round(seconds / DT))):
        world.step_fixed(1)
        field.update(DT, player_y=player_y)


# ===================================================== placement / authoring
def test_every_placement_generates_a_destructible_with_chunks():
    w = make_world()
    ds = structures.build_destructibles(w)

    assert len(ds) == len(structures.PLACEMENTS)
    for d in ds:
        assert d.intact
        assert d.chunk_count > 1, f"{d.name} generated no chunks"
        assert d.mass > 0.0
        assert d.proxy_names, f"{d.name} has no collision proxy"
        # Chunk descriptors must tile the source volume, not approximate it.
        assert d.result.volume_error < 1e-9, (
            f"{d.name} chunk volume drifted by {d.result.volume_error:.2e}"
        )
    assert structures.total_chunks(ds) > 1000, "not a representative load"


def test_generation_is_deterministic_in_the_seed():
    a = structures.generate("arch", (0.0, 10.0, 0.0), seed=77)
    b = structures.generate("arch", (0.0, 10.0, 0.0), seed=77)
    c = structures.generate("arch", (0.0, 10.0, 0.0), seed=78)
    assert a.result.serialize() == b.result.serialize()
    assert a.result.serialize() != c.result.serialize()


def test_placements_do_not_overlap_each_other():
    """Two structures sharing space would interpenetrate on demolition."""
    w = make_world()
    ds = structures.build_destructibles(w)
    for i in range(len(ds)):
        for j in range(i + 1, len(ds)):
            alo, ahi = ds[i].world_bounds()
            blo, bhi = ds[j].world_bounds()
            sep = any(ahi[k] <= blo[k] or bhi[k] <= alo[k] for k in range(3))
            assert sep, f"{ds[i].name} overlaps {ds[j].name}"


def test_proxies_are_static_and_match_the_source_blocks():
    w = make_world()
    d = structures.attach_proxies(
        w, structures.generate("arch", (3.0, 40.0, 0.0), seed=5)
    )
    assert len(d.proxy_names) == len(d.spec.blocks)
    for name, block in zip(d.proxy_names, d.spec.blocks):
        np_ = w.body(name)
        assert float(np_.node().getMass()) == 0.0, "a proxy is dynamic"
        want = tuple(block.center[i] + d.origin[i] for i in range(3))
        got = tuple(float(v) for v in np_.getPos())
        assert got == pytest.approx(want, abs=1e-6)


def test_an_intact_structure_is_solid():
    """A dropped probe must land on the structure, not fall through it."""
    w = make_world()
    d = structures.attach_proxies(
        w, structures.generate("slab", (0.0, 0.0, 0.0), seed=5)
    )
    lo, hi = d.world_bounds()
    probe = w.add_dynamic_box("probe", (0.0, 0.0, hi[2] + 3.0),
                              (0.3, 0.3, 0.3), mass=40.0)
    w.step_fixed(int(4.0 / DT))
    z = float(probe.getZ())
    assert z > hi[2] - 0.5, (
        f"the probe ended at z={z:.3f}; the slab's top is {hi[2]:.3f} - it "
        f"fell through an intact structure"
    )


# ===================================================== demolish: the handoff
def test_demolish_removes_the_proxies_and_spawns_the_debris():
    w = make_world()
    d = structures.attach_proxies(
        w, structures.generate("cluster", (0.0, 0.0, 0.0), seed=11)
    )
    field = DebrisField(w)

    for name in d.proxy_names:
        assert name in w.bodies
    before = w.world.getNumRigidBodies()

    event = field.demolish(d)

    assert event is not None
    assert not d.intact
    for name in d.proxy_names:
        assert name not in w.bodies, f"proxy {name} survived demolition"
    assert event.spawned == min(d.chunk_count, field.max_live)
    assert field.live_count == event.spawned
    assert w.world.getNumRigidBodies() == (
        before - len(d.proxy_names) + event.spawned
    )


def test_demolished_structure_no_longer_blocks_the_lane():
    """The proxy really is gone: a probe now falls past where it stood."""
    w = make_world()
    d = structures.attach_proxies(
        w, structures.generate("slab", (0.0, 0.0, 0.0), seed=5)
    )
    field = DebrisField(w)
    lo, hi = d.world_bounds()

    field.demolish(d)
    # Put the probe outside the debris footprint but inside where the slab
    # used to be, so this is about the proxy and not about the rubble.
    probe = w.add_dynamic_box("probe", (hi[0] - 0.5, 0.0, hi[2] - 1.0),
                              (0.3, 0.3, 0.3), mass=40.0)
    w.step_fixed(int(3.0 / DT))
    assert float(probe.getZ()) < hi[2] - 1.0, (
        "the probe is still held up where the demolished slab used to be"
    )


def test_demolishing_twice_is_a_no_op():
    """A damage node can land two hits on the same frame."""
    w = make_world()
    d = structures.attach_proxies(
        w, structures.generate("cluster", (0.0, 0.0, 0.0), seed=11)
    )
    field = DebrisField(w)
    first = field.demolish(d)
    count = field.live_count
    second = field.demolish(d)
    assert first is not None
    assert second is None, "the same structure was demolished twice"
    assert field.live_count == count


def test_debris_spawns_at_the_structures_world_position():
    """Debris must appear where the building was, not at the origin."""
    w = make_world()
    d = structures.attach_proxies(
        w, structures.generate("cluster", (9.0, 120.0, 0.0), seed=11)
    )
    field = DebrisField(w)
    event = field.demolish(d)

    lo, hi = d.world_bounds()
    for body in event.bodies:
        x, y, z = (float(v) for v in body.pos)
        assert lo[0] - 1.0 <= x <= hi[0] + 1.0, f"{body.name} x={x}"
        assert lo[1] - 1.0 <= y <= hi[1] + 1.0, f"{body.name} y={y}"
        assert lo[2] - 1.0 <= z <= hi[2] + 1.0, f"{body.name} z={z}"
    ys = [float(b.pos.getY()) for b in event.bodies]
    assert sum(ys) / len(ys) > 100.0, "debris spawned back at the origin"


def test_demolished_debris_settles_on_the_ground_where_it_stood():
    w = make_world()
    d = structures.attach_proxies(
        w, structures.generate("cluster", (0.0, 60.0, 0.0), seed=11)
    )
    field = DebrisField(w)
    field.demolish(d)
    run(w, field, 30.0)

    snap = field.snapshot()
    bodies = field.live + field.frozen
    assert bodies
    assert snap["max_speed"] < 1e-9, f"still moving at {snap['max_speed']}"
    assert snap["max_spin"] < 1e-9
    assert snap["active"] == 0
    assert snap["asleep"] == len(bodies), (
        f"{len(bodies) - snap['asleep']} bodies are not asleep"
    )
    assert snap["min_z"] > -0.05, "debris ended below the ground"
    # And it stayed put: rubble on the spot, not scattered to infinity.
    assert all(abs(float(b.pos.getY()) - 60.0) < 40.0 for b in bodies)


def test_nearest_intact_targets_and_then_skips_rubble():
    w = make_world()
    ds = structures.build_destructibles(w)
    field = DebrisField(w, max_live=60)

    first = structures.nearest_intact(ds, (0.0, 0.0, 0.0))
    assert first is not None
    field.demolish(first)
    second = structures.nearest_intact(ds, (0.0, 0.0, 0.0))
    assert second is not None and second is not first, (
        "nearest_intact returned a structure that is already rubble"
    )

    for d in ds:
        field.demolish(d)
    assert structures.nearest_intact(ds, (0.0, 0.0, 0.0)) is None


# ===================================================== budget under sequence
def test_the_whole_course_demolished_in_sequence_holds_the_cap():
    """Every structure on the course, one after another, cap never breached."""
    w = make_world()
    ds = structures.build_destructibles(w)
    field = DebrisField(w)
    assert structures.total_chunks(ds) > field.max_live * 4, (
        "the course is not big enough to stress the budget"
    )

    peak = 0
    for d in ds:
        player_y = d.world_bounds()[0][1] - 12.0
        field.demolish(d)
        assert field.live_count <= field.max_live, (
            f"cap breached on the demolition frame: {field.live_count}"
        )
        for _ in range(int(3.0 / DT)):
            w.step_fixed(1)
            field.update(DT, player_y=player_y)
            peak = max(peak, field.live_count)
            assert field.live_count <= field.max_live, (
                f"cap breached mid-sim: {field.live_count} > {field.max_live}"
            )

    assert peak > field.max_live * 0.5, "the cap was never approached"
    assert field.total_frozen + field.total_despawned > 0, (
        "nothing was ever retired - the budget held only because it was lucky"
    )


def test_driving_the_course_retires_the_rubble_behind_you():
    w = make_world()
    ds = structures.build_destructibles(w)
    field = DebrisField(w)

    for d in ds:
        field.demolish(d)
        run(w, field, 4.0, player_y=d.world_bounds()[1][1])

    mid = field.total_bodies
    assert mid > 0

    # Drive off the end of the course: everything behind is retired.
    run(w, field, 5.0, player_y=400.0)
    assert field.total_bodies == 0, (
        f"{field.total_bodies} bodies survived the player leaving the course"
    )
    assert w.world.getNumRigidBodies() == 1 + sum(
        len(d.proxy_names) for d in ds if d.intact
    ), "something other than the ground was left in the world"


def test_sequential_demolition_stays_faster_than_real_time(capsys):
    """The performance budget, with the numbers printed.

    Demolishes the entire course while stepping, and reports throughput. A
    step that costs more than the wall time it represents can never keep up.
    """
    w = make_world()
    ds = structures.build_destructibles(w)
    field = DebrisField(w)

    seconds_each = 3.0
    steps_each = int(seconds_each / DT)
    total_steps = 0
    build_ms = []
    started = time.perf_counter()
    for d in ds:
        event = field.demolish(d)
        build_ms.append(event.build_seconds * 1000.0)
        player_y = d.world_bounds()[0][1] - 12.0
        for _ in range(steps_each):
            w.step_fixed(1)
            field.update(DT, player_y=player_y)
        total_steps += steps_each
    wall = time.perf_counter() - started

    simulated = total_steps * DT
    ms_per_step = wall / total_steps * 1000.0
    budget_ms = DT * 1000.0
    factor = simulated / wall

    with capsys.disabled():
        print()
        print(f"  [course benchmark] {len(ds)} structures, "
              f"{structures.total_chunks(ds)} pre-generated chunks, "
              f"cap {field.max_live}")
        print(f"  [course benchmark] simulated {simulated:.1f} s in "
              f"{wall:.3f} s wall -> {factor:.2f}x real time")
        print(f"  [course benchmark] per step {ms_per_step:.4f} ms "
              f"(budget {budget_ms:.3f} ms, "
              f"headroom {budget_ms / ms_per_step:.2f}x)")
        print(f"  [course benchmark] shatter build time: "
              f"min {min(build_ms):.1f} ms, max {max(build_ms):.1f} ms")

    assert factor > 1.0, (
        f"demolishing the course runs at {factor:.2f}x real time"
    )
    assert ms_per_step < budget_ms, (
        f"{ms_per_step:.4f} ms per step exceeds the {budget_ms:.3f} ms budget"
    )
    # A demolition must not stall the frame it happens on.
    assert max(build_ms) < 50.0, f"shatter took {max(build_ms):.1f} ms"


def test_load_time_fracture_cost_is_paid_once_at_startup(capsys):
    """Generation is a load-time cost. Measured, and reported."""
    w = make_world()
    started = time.perf_counter()
    ds = structures.build_destructibles(w)
    gen = time.perf_counter() - started

    with capsys.disabled():
        print(f"  [load benchmark] fractured {len(ds)} structures / "
              f"{structures.total_chunks(ds)} chunks in {gen * 1000.0:.0f} ms")

    assert gen < 5.0, f"load-time fracture took {gen:.2f} s"
    # And it is genuinely not repeated per shatter: the field only converts
    # existing descriptors.
    field = DebrisField(w)
    t = time.perf_counter()
    field.demolish(ds[0])
    shatter = time.perf_counter() - t
    assert shatter < gen, (
        f"shatter ({shatter * 1000:.1f} ms) costs as much as generation "
        f"({gen * 1000:.1f} ms) - the pre-generation is being thrown away"
    )
