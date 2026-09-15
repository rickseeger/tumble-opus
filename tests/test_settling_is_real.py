"""Is the settling real physics, or is the freeze doing all the work?

This is the file that keeps the settle claim honest. `DebrisField` retires
settled debris by freezing it to mass-0 static geometry, which sets its
velocities to exactly zero. That makes "all debris has near-zero velocity and
is asleep" trivially true - it would pass even if the underlying physics
jittered forever.

So these tests run with the freeze **disabled** (settle dwell time pushed
beyond the run) and require pure Bullet to bring the pile to rest on its own.
If that fails, the tuning is wrong and no amount of freezing fixes it.
"""

import pytest

from game import config, fracture
from game.debris import DebrisField
from game.physics import PhysicsWorld

DT = config.FIXED_DT


class no_freeze:
    """Disable the field's own retirement, leaving pure Bullet behaviour."""

    def __enter__(self):
        self._settle = config.DEBRIS_SETTLE_TIME
        config.DEBRIS_SETTLE_TIME = 1.0e9
        return self

    def __exit__(self, *exc):
        config.DEBRIS_SETTLE_TIME = self._settle
        return False


def make_field(**kw):
    w = PhysicsWorld()
    w.add_ground_plane(0.0)
    return w, DebrisField(w, **kw)


def test_bullet_settles_the_pile_on_its_own_with_no_freezing():
    """The core honesty check: real physics, no retirement, real rest.

    Measured on this machine: a 150-chunk cluster collapse decays from
    ~10 m/s peak at t=2 s to exactly 0 with every body deactivated by
    t~24 s. The 45 s bound below is that measurement with headroom, not a
    number chosen to make the test pass.
    """
    w, f = make_field()
    result = fracture.fracture(
        fracture.with_budget(fracture.cluster_spec(), 150), seed=9
    )
    f.shatter(result, impact_point=(0.0, 0.0, 3.0))

    with no_freeze():
        settled_at = None
        t = 0.0
        while t < 45.0:
            w.step_fixed(1)
            f.update(DT)
            t += DT
            if settled_at is None and not any(
                b.node.isActive() for b in f.live
            ):
                settled_at = t

        assert f.frozen_count == 0, "the freeze fired; this proves nothing"
        assert f.live_count == len(result.chunks)

        speeds = [b.speed() for b in f.live]
        spins = [b.spin() for b in f.live]
        active = sum(1 for b in f.live if b.node.isActive())

        assert settled_at is not None, (
            "no body ever deactivated in 45 s of real simulation - the pile "
            "is jittering forever and the freeze was hiding it"
        )
        assert active == 0, f"{active} bodies still awake after 45 s"
        assert max(speeds) == 0.0, (
            f"pure Bullet left debris moving at {max(speeds):.6f} m/s"
        )
        assert max(spins) == 0.0, (
            f"pure Bullet left debris spinning at {max(spins):.6f} rad/s"
        )


def test_energy_really_decays_rather_than_being_zeroed_by_the_freeze():
    """Kinetic energy must fall monotonically-ish, with the freeze off."""
    w, f = make_field()
    result = fracture.fracture(
        fracture.with_budget(fracture.tower_spec(), 150), seed=7
    )
    f.shatter(result, impact_point=(0.0, 0.0, 8.0))

    with no_freeze():
        samples = []
        for _ in range(30):                     # 30 x 1 s
            for _ in range(int(1.0 / DT)):
                w.step_fixed(1)
                f.update(DT)
            samples.append(f.kinetic_energy())

        assert f.frozen_count == 0
        peak = max(samples)
        assert peak > 1000.0, "the collapse carried no energy to begin with"
        # The tail must be dead, and the decay must be real, not a cliff
        # caused by bodies being removed.
        assert samples[-1] == 0.0, (
            f"{samples[-1]:.3f} J of kinetic energy left after 30 s"
        )
        early = sum(samples[:10]) / 10.0
        late = sum(samples[20:]) / 10.0
        assert late < early * 0.01, (
            f"energy barely decayed under pure physics: {early:.1f} J -> "
            f"{late:.1f} J"
        )
        assert f.live_count == len(result.chunks), "bodies were removed"


def test_debris_rests_above_the_ground_not_sunk_into_it():
    """Settled means resting on the floor, not tunnelled halfway through it."""
    w, f = make_field()
    result = fracture.fracture(
        fracture.with_budget(fracture.arch_spec(), 120), seed=4
    )
    f.shatter(result, impact_point=(0.0, 0.0, 4.0))

    with no_freeze():
        for _ in range(int(40.0 / DT)):
            w.step_fixed(1)
            f.update(DT)

        lows = [b.lowest_z() for b in f.live]
        assert min(lows) > -0.05, (
            f"a chunk's lowest vertex settled {min(lows):.4f} m below the "
            f"ground plane"
        )
        # Something must actually be touching down, not all hovering.
        assert min(lows) < config.DEBRIS_GROUNDED_TOLERANCE, (
            f"nothing reached the ground; lowest vertex is {min(lows):.4f} m"
        )
        # And nothing was flung out of the world.
        assert max(abs(float(b.pos.getX())) for b in f.live) < 500.0
        assert max(float(b.pos.getZ()) for b in f.live) < 200.0


def test_bullet_deactivates_a_meaningful_share_of_the_pile_unaided():
    """Sanity on the tuning: sleep thresholds must actually be reachable.

    If Bullet never deactivated anything, the sleep thresholds would be
    mistuned and the freeze would be papering over it.
    """
    w, f = make_field()
    result = fracture.fracture(fracture.tower_spec(), seed=7)
    event = f.shatter(result, impact_point=(0.0, 0.0, 6.0))

    peak_asleep = 0
    for _ in range(int(40.0 / DT)):
        w.step_fixed(1)
        # Count what Bullet slept on its own BEFORE our retirement runs.
        peak_asleep = max(
            peak_asleep, sum(1 for b in f.live if not b.node.isActive())
        )
        f.update(DT)

    assert peak_asleep > event.spawned * 0.25, (
        f"Bullet only ever deactivated {peak_asleep} of {event.spawned} "
        f"bodies by itself - the sleep thresholds are not reachable"
    )


def test_settling_is_not_a_timed_despawn_in_disguise():
    """Retired debris is still there. It settled; it did not evaporate."""
    w, f = make_field()
    result = fracture.fracture(
        fracture.with_budget(fracture.cluster_spec(), 120), seed=9
    )
    event = f.shatter(result, impact_point=(0.0, 0.0, 3.0))

    for _ in range(int(30.0 / DT)):
        w.step_fixed(1)
        f.update(DT)            # player_y omitted: no distance culling

    assert f.total_despawned == 0, "debris was despawned, not settled"
    assert f.total_bodies == event.spawned, (
        f"{event.spawned - f.total_bodies} bodies vanished"
    )
    # Each one is at a real resting pose, on the ground, near where it fell.
    for body in f.live + f.frozen:
        assert body.lowest_z() > -0.05
        assert abs(float(body.pos.getX())) < 100.0
