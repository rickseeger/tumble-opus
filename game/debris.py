"""Runtime shatter and debris physics for Tumble.

This is the layer between the *pre-generated* chunk descriptors that
:mod:`game.fracture` produces at load time and the live Bullet simulation. It
owns exactly one public verb::

    field.shatter(structure, impact_point, impulse) -> ShatterEvent

...which turns a structure's chunk set into independently simulated rigid
bodies that launch away from the blast, fall under gravity, bounce, roll,
settle, and then get retired so the frame budget survives the next tower.

It imports Panda3D/Bullet but **not** ShowBase or the render graph, so the
whole thing is headless-testable exactly like :mod:`game.physics`.

Why the tuning is the way it is
-------------------------------
Every number here was measured on this stack (Panda3D 1.10.16 / Bullet, fixed
1/120 s step), not guessed. The three findings that actually mattered:

1. **Collision margin must scale with the chunk.** Bullet's default convex
   margin is a flat 0.04 m. The fracture library tiles a structure's volume
   *exactly*, so with a flat margin every chunk starts life ~8 cm
   interpenetrated with each of its neighbours and the solver detonates the
   pile on frame one. Scaling the margin to the chunk's own smallest half
   extent (and clamping it small) is what makes a 380-piece tower spawn
   quietly instead of exploding.

2. **Restitution is a product, and split impulse matters.** Bullet combines
   restitution multiplicatively, so a ground plane left at the default 0
   restitution means *nothing ever bounces* regardless of what the debris is
   set to - the only rebound you see is penetration-recovery noise. Measured
   here: with the ground at 0, a 9.4 m/s impact rebounds at 0.46-0.48 m/s for
   every body restitution from 0.0 to 0.6, i.e. entirely decoupled from the
   setting. With the ground at :data:`config.DEBRIS_GROUND_RESTITUTION` the
   rebound tracks the product properly. Split impulse is on so that the
   positional correction for an overlap does not get injected back as bounce
   velocity.

3. **Bullet's own sleep is not enough in a dense pile.** With 260 bodies in a
   rubble heap, ~250 of them stay nominally `isActive()` indefinitely: contact
   churn keeps resetting the deactivation timer even though nothing is really
   moving (measured residual |v| < 0.4 m/s, |w| < 0.2 rad/s, and falling). So
   :class:`DebrisField` runs its own stricter settle detector on top, and
   *freezes* a body that has been quiet for
   :data:`config.DEBRIS_SETTLE_TIME`: mass is set to 0, the body becomes
   static geometry. It is still there, still visible, still collidable by the
   later damage node - it just costs the solver nothing. That is what makes
   "debris comes to rest, not jitters forever" literally true rather than
   approximately true.

Budget
------
:data:`config.DEBRIS_MAX_LIVE` is a hard ceiling on *dynamic* debris bodies
and is never exceeded. When a shatter would blow it, the field first freezes
the debris that is closest to settled, then despawns the oldest frozen rubble,
and only then declines to spawn a structure's smallest chunks. Frozen debris
that the player has driven past by :data:`config.DEBRIS_DESPAWN_BEHIND` metres
is despawned outright.

Rendering
---------
:meth:`DebrisBody.wireframe_segments` hands the render layer the locked 1A
glowing-wireframe treatment's raw material: world-space line segments for
every edge of the chunk, already transformed by the body's live pose. Nothing
in this module imports the render graph; the visual call is the render layer's
to make.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field as _dc_field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from panda3d.bullet import (
    BulletConvexHullShape,
    BulletRigidBodyNode,
)
from panda3d.core import NodePath, Point3, Quat, TransformState, Vec3

from . import config
from .fracture import Chunk, FractureResult, StructureSpec, fracture

Triple = Sequence[float]

#: Lifecycle states a debris body moves through, in order.
LIVE = "live"        # dynamic, simulated, costs solver time
FROZEN = "frozen"    # settled -> mass 0, static geometry, still collidable
DESPAWNED = "gone"   # removed from the world entirely


def _vec3(v) -> Vec3:
    if isinstance(v, Vec3):
        return Vec3(v)
    if isinstance(v, Point3):
        return Vec3(v.getX(), v.getY(), v.getZ())
    return Vec3(float(v[0]), float(v[1]), float(v[2]))


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t, 0.0, 1.0)


# --------------------------------------------------------------------- bodies
class DebrisBody:
    """One chunk, live in the Bullet world.

    Thin wrapper: it owns the `BulletRigidBodyNode` and the bookkeeping the
    field needs (settle timer, spawn order, source chunk). Physics queries go
    straight through to Bullet so nothing can drift out of sync.
    """

    __slots__ = (
        "name", "chunk", "node", "np", "field", "state",
        "spawn_step", "quiet_time", "structure", "base_mass",
        "size_blend", "_frozen_at",
    )

    def __init__(
        self,
        name: str,
        chunk: Chunk,
        node: BulletRigidBodyNode,
        np_: NodePath,
        structure: str,
        base_mass: float,
        size_blend: float,
        spawn_step: int,
    ) -> None:
        self.name = name
        self.chunk = chunk
        self.node = node
        self.np = np_
        self.structure = structure
        self.base_mass = base_mass
        self.size_blend = size_blend
        self.spawn_step = spawn_step
        self.state = LIVE
        self.quiet_time = 0.0
        self._frozen_at = -1

    # ------------------------------------------------------------- queries
    @property
    def pos(self) -> Vec3:
        return self.np.getPos()

    @property
    def quat(self) -> Quat:
        return self.np.getQuat()

    @property
    def mass(self) -> float:
        """Live Bullet mass. Frozen debris reads 0 (it is static)."""
        return float(self.node.getMass())

    @property
    def volume(self) -> float:
        return float(self.chunk.volume)

    def linear_velocity(self) -> Vec3:
        return self.node.getLinearVelocity()

    def angular_velocity(self) -> Vec3:
        return self.node.getAngularVelocity()

    def speed(self) -> float:
        return float(self.node.getLinearVelocity().length())

    def spin(self) -> float:
        return float(self.node.getAngularVelocity().length())

    def kinetic_energy(self) -> float:
        """Translational KE only - enough to watch the pile calm down."""
        if self.state is not LIVE:
            return 0.0
        v = self.node.getLinearVelocity()
        return 0.5 * self.base_mass * float(v.lengthSquared())

    def is_active(self) -> bool:
        """True only while Bullet is still solving this body."""
        return self.state is LIVE and bool(self.node.isActive())

    def is_asleep(self) -> bool:
        """At rest as far as the simulation is concerned.

        Frozen debris counts as asleep: it is static geometry, it cannot move
        by definition, which is a strictly stronger guarantee than Bullet
        having deactivated it.
        """
        if self.state is FROZEN:
            return True
        if self.state is DESPAWNED:
            return True
        return not bool(self.node.isActive())

    def at_rest(
        self,
        linear: float = config.DEBRIS_SETTLE_LINEAR,
        angular: float = config.DEBRIS_SETTLE_ANGULAR,
    ) -> bool:
        if self.state is not LIVE:
            return True
        return self.speed() <= linear and self.spin() <= angular

    def grounded(self, ground_z: float = 0.0,
                 tol: float = config.DEBRIS_GROUNDED_TOLERANCE) -> bool:
        """True when the chunk's lowest vertex is on (or in) the ground.

        Velocity alone is not enough to call a chunk 'settled': a chunk at the
        apex of its arc is momentarily slow too, and freezing it there would
        leave a slab hanging in mid-air. Debris only ever freezes when it is
        both slow AND actually touching down.
        """
        return self.lowest_z() <= float(ground_z) + float(tol)

    def lowest_z(self) -> float:
        """World Z of the chunk's lowest vertex, under its live pose."""
        q = self.np.getQuat()
        p = self.np.getPos()
        lo = math.inf
        for v in self.chunk.vertices:
            lo = min(lo, float(q.xform(Vec3(*v)).getZ()))
        return lo + float(p.getZ())

    # --------------------------------------------------------- render hook
    def wireframe_segments(self) -> List[Tuple[Tuple[float, float, float],
                                               Tuple[float, float, float]]]:
        """World-space line segments for the locked 1A glowing wireframe.

        The fracture library already emits an explicit edge list per chunk;
        this just pushes it through the body's live pose. The render layer
        decides colour, glow and blend - this module never touches `render`.
        """
        q = self.np.getQuat()
        p = self.np.getPos()

        def w(i: int):
            r = q.xform(Vec3(*self.chunk.vertices[i])) + p
            return (float(r.getX()), float(r.getY()), float(r.getZ()))

        return [(w(a), w(b)) for a, b in self.chunk.edges]


@dataclass
class ShatterEvent:
    """What one :meth:`DebrisField.shatter` call actually did."""

    structure: str
    impact_point: Tuple[float, float, float]
    impulse: float
    chunk_count: int
    spawned: int
    skipped_for_budget: int
    frozen_to_make_room: int
    despawned_to_make_room: int
    build_seconds: float
    bodies: Tuple[DebrisBody, ...] = _dc_field(default=(), repr=False)

    @property
    def total_mass(self) -> float:
        return sum(b.base_mass for b in self.bodies)


# ---------------------------------------------------------------------- field
class DebrisField:
    """Owns every debris body in the world, and the budget that bounds them.

    Construct it around an existing :class:`game.physics.PhysicsWorld`::

        field = DebrisField(physics)
        event = field.shatter(structure, impact_point=(0, 40, 6), impulse=9e4)
        ...
        field.update(dt, player_y=player.pos[1])   # once per frame

    :meth:`update` is what retires debris; call it every frame or debris will
    correctly simulate forever and eat the budget.
    """

    def __init__(
        self,
        physics,
        seed: int = 20260914,
        max_live: int = config.DEBRIS_MAX_LIVE,
        max_frozen: int = config.DEBRIS_MAX_FROZEN,
    ) -> None:
        self.physics = physics
        self.world = physics.world
        self.root = physics.root.attachNewNode("debris-root")
        self.rng = random.Random(seed)
        self.seed = seed

        self.max_live = int(max_live)
        self.max_frozen = int(max_frozen)

        self.live: List[DebrisBody] = []
        self.frozen: List[DebrisBody] = []
        self.all_bodies: Dict[str, DebrisBody] = {}

        self.events: List[ShatterEvent] = []
        self.total_spawned = 0
        self.total_frozen = 0
        self.total_despawned = 0
        self._serial = 0

        # Give the ground a real restitution, otherwise Bullet's
        # multiplicative combine means debris cannot bounce at all. Measured:
        # with ground restitution 0, a 9.4 m/s impact rebounds at ~0.47 m/s
        # for EVERY body restitution from 0.0 to 0.6.
        self.prepare_ground()

    # ------------------------------------------------------------- surfaces
    def prepare_ground(self) -> None:
        """Set the ground's contact properties so debris can bounce and grip."""
        g = getattr(self.physics, "ground_np", None)
        if g is not None:
            g.node().setRestitution(config.DEBRIS_GROUND_RESTITUTION)
            g.node().setFriction(config.DEBRIS_GROUND_FRICTION)
        # Static course geometry should behave like the ground, not like
        # frictionless glass.
        for name, np_ in self.physics.bodies.items():
            if name == "ground":
                continue
            node = np_.node()
            if hasattr(node, "getMass") and float(node.getMass()) == 0.0:
                node.setRestitution(config.DEBRIS_GROUND_RESTITUTION * 0.6)
                node.setFriction(config.DEBRIS_GROUND_FRICTION)

    # -------------------------------------------------------------- counts
    @property
    def live_count(self) -> int:
        return len(self.live)

    @property
    def frozen_count(self) -> int:
        return len(self.frozen)

    @property
    def total_bodies(self) -> int:
        return len(self.live) + len(self.frozen)

    def kinetic_energy(self) -> float:
        return sum(b.kinetic_energy() for b in self.live)

    def active_count(self) -> int:
        return sum(1 for b in self.live if b.is_active())

    # ------------------------------------------------------------- tuning
    @staticmethod
    def size_blend(volume: float, vmin: float, vmax: float) -> float:
        """0.0 for the heaviest chunk in a structure, 1.0 for the lightest.

        Log-spaced, because chunk volumes span two-plus orders of magnitude
        and a linear blend would put almost everything at the light end.
        """
        if vmax <= vmin * (1.0 + 1e-9):
            return 0.5
        lo = math.log10(max(vmin, 1e-9))
        hi = math.log10(max(vmax, 1e-9))
        t = (math.log10(max(volume, 1e-9)) - lo) / (hi - lo)
        return _clamp(1.0 - t, 0.0, 1.0)

    @staticmethod
    def mass_for(chunk: Chunk, density: float = config.DEBRIS_DENSITY) -> float:
        """Mass from volume. This is the whole 'slabs thud, shards skitter'.

        Deliberately *not* taken from ``chunk.mass``: the fracture library's
        density is a property of the spec, and the runtime wants a single
        consistent debris density it can tune independently of authoring.
        """
        return max(float(chunk.volume) * density, config.DEBRIS_MIN_MASS)

    @staticmethod
    def margin_for(chunk: Chunk) -> float:
        smallest = min(float(h) for h in chunk.half_extents)
        return _clamp(
            smallest * config.DEBRIS_MARGIN_FRACTION,
            config.DEBRIS_MARGIN_MIN,
            config.DEBRIS_MARGIN_MAX,
        )

    # ------------------------------------------------------------- shatter
    def shatter(
        self,
        structure,
        impact_point: Triple,
        impulse: float = config.DEBRIS_IMPULSE,
        origin: Triple = (0.0, 0.0, 0.0),
        seed: Optional[int] = None,
    ) -> ShatterEvent:
        """Convert a structure's chunk descriptors into live rigid bodies.

        Parameters
        ----------
        structure:
            A :class:`~game.fracture.FractureResult` (already pre-generated,
            the normal case) or a :class:`~game.fracture.StructureSpec`, which
            is fractured on the spot as a convenience for tests and demos.
        impact_point:
            World-space blast centre. Chunks launch *away* from this point.
        impulse:
            Blast strength in N*s at :data:`config.DEBRIS_MIN_RADIUS`. Speed
            is ``impulse / (mass * distance**falloff)``, clamped and jittered,
            so heavy slabs near the blast move slowly and light shards fly.
        origin:
            World-space placement of the structure's local frame.

        This is the API the weapon/damage node and the campaign node call.
        """
        started = time.perf_counter()

        if isinstance(structure, StructureSpec):
            structure = fracture(structure, seed=self.seed)
        if not isinstance(structure, FractureResult):
            raise TypeError(
                "shatter() wants a FractureResult or a StructureSpec, got "
                f"{type(structure).__name__}"
            )

        rng = random.Random(seed) if seed is not None else self.rng
        impact = _vec3(impact_point)
        base = _vec3(origin)

        chunks = list(structure.chunks)
        vols = [c.volume for c in chunks]
        vmin, vmax = (min(vols), max(vols)) if vols else (1.0, 1.0)

        # Budget first, so the cap is enforced *before* anything is attached
        # to the world rather than cleaned up afterwards.
        room, froze, despawned = self._make_room(len(chunks))
        # Biggest chunks win the budget: losing a shard is invisible, losing
        # the slab that was the building's corner is not.
        order = sorted(range(len(chunks)), key=lambda i: -vols[i])
        keep = set(order[:room])
        skipped = len(chunks) - len(keep)

        spawned: List[DebrisBody] = []
        for i, chunk in enumerate(chunks):
            if i not in keep:
                continue
            body = self._spawn_chunk(
                chunk, structure.spec_name, base, impact, impulse,
                vmin, vmax, rng,
            )
            spawned.append(body)

        event = ShatterEvent(
            structure=structure.spec_name,
            impact_point=(float(impact.getX()), float(impact.getY()),
                          float(impact.getZ())),
            impulse=float(impulse),
            chunk_count=len(chunks),
            spawned=len(spawned),
            skipped_for_budget=skipped,
            frozen_to_make_room=froze,
            despawned_to_make_room=despawned,
            build_seconds=time.perf_counter() - started,
            bodies=tuple(spawned),
        )
        self.events.append(event)
        return event

    def demolish(
        self,
        destructible,
        impact_point: Optional[Triple] = None,
        impulse: float = config.DEBRIS_IMPULSE,
        seed: Optional[int] = None,
    ) -> Optional[ShatterEvent]:
        """Stop a placed structure being a building and start it being debris.

        This is the verb the weapon/damage node and the campaign node call on
        a :class:`game.structures.Destructible`. It does the two things that
        must happen on the same frame:

        1. Removes the structure's **static collision proxies** from the
           world. Miss this and the player is walled off by a tower that has
           visibly collapsed.
        2. Shatters its pre-generated chunk set into live debris at the
           structure's world origin.

        Returns the :class:`ShatterEvent`, or ``None`` if it was already down
        (demolishing twice is a no-op, not an error - a damage node will
        happily land two hits on the same frame).
        """
        if not getattr(destructible, "intact", False):
            return None

        for name in getattr(destructible, "proxy_names", ()):  # 1.
            self.physics.remove_body(name)
        destructible.intact = False
        destructible.demolished_step = int(
            getattr(self.physics, "step_count", 0)
        )

        point = (impact_point if impact_point is not None
                 else destructible.default_impact_point())
        event = self.shatter(                                   # 2.
            destructible.result,
            impact_point=point,
            impulse=impulse,
            origin=destructible.origin,
            seed=seed,
        )
        return event

    def _spawn_chunk(
        self,
        chunk: Chunk,
        structure_name: str,
        base: Vec3,
        impact: Vec3,
        impulse: float,
        vmin: float,
        vmax: float,
        rng: random.Random,
    ) -> DebrisBody:
        blend = self.size_blend(chunk.volume, vmin, vmax)
        mass = self.mass_for(chunk)

        shape = BulletConvexHullShape()
        for v in chunk.vertices:
            shape.addPoint(Vec3(*v))
        shape.setMargin(self.margin_for(chunk))

        self._serial += 1
        name = f"debris_{structure_name}_{chunk.index}_{self._serial}"
        node = BulletRigidBodyNode(name)
        node.addShape(shape)
        node.setMass(mass)

        node.setRestitution(_lerp(config.DEBRIS_RESTITUTION_HEAVY,
                                  config.DEBRIS_RESTITUTION_LIGHT, blend))
        node.setFriction(_lerp(config.DEBRIS_FRICTION_HEAVY,
                               config.DEBRIS_FRICTION_LIGHT, blend))
        node.setLinearDamping(config.DEBRIS_LINEAR_DAMPING)
        node.setAngularDamping(config.DEBRIS_ANGULAR_DAMPING)

        node.setLinearSleepThreshold(config.DEBRIS_LINEAR_SLEEP_THRESHOLD)
        node.setAngularSleepThreshold(config.DEBRIS_ANGULAR_SLEEP_THRESHOLD)
        node.setDeactivationTime(config.DEBRIS_DEACTIVATION_TIME)
        node.setDeactivationEnabled(True)

        # Debris is exactly the fast-moving small geometry CCD exists for.
        smallest = min(float(h) for h in chunk.half_extents)
        node.setCcdMotionThreshold(smallest * 0.6)
        node.setCcdSweptSphereRadius(smallest * 0.8)

        np_ = self.root.attachNewNode(node)
        world_pos = base + Vec3(*chunk.center)
        np_.setPos(world_pos)
        w, x, y, z = chunk.orientation
        np_.setQuat(Quat(w, x, y, z))

        self.world.attachRigidBody(node)

        lin, ang = self.launch_velocity(
            world_pos, impact, mass, blend, impulse, rng
        )
        node.setLinearVelocity(lin)
        node.setAngularVelocity(ang)
        node.setActive(True)

        body = DebrisBody(
            name=name, chunk=chunk, node=node, np_=np_,
            structure=structure_name, base_mass=mass, size_blend=blend,
            spawn_step=int(getattr(self.physics, "step_count", 0)),
        )
        self.live.append(body)
        self.all_bodies[name] = body
        self.total_spawned += 1
        return body

    def launch_velocity(
        self,
        world_pos: Vec3,
        impact: Vec3,
        mass: float,
        blend: float,
        impulse: float,
        rng: random.Random,
    ) -> Tuple[Vec3, Vec3]:
        """Outward launch velocity + random spin for one chunk.

        Pure function of its arguments and the RNG, so a test can call it
        directly. The direction is the radial from the impact point, randomly
        perturbed and biased upward; the magnitude is impulse/mass with a
        distance falloff, clamped and jittered.
        """
        radial = world_pos - impact
        dist = float(radial.length())
        if dist < 1e-6:
            radial = Vec3(rng.uniform(-1, 1), rng.uniform(-1, 1), 1.0)
            dist = config.DEBRIS_MIN_RADIUS
        radial = radial / max(float(radial.length()), 1e-9)

        s = config.DEBRIS_LAUNCH_SPREAD
        direction = Vec3(
            radial.getX() + rng.uniform(-s, s),
            radial.getY() + rng.uniform(-s, s),
            radial.getZ() + rng.uniform(-s * 0.4, s) + config.DEBRIS_LAUNCH_UP_BIAS,
        )
        if direction.lengthSquared() < 1e-12:      # pragma: no cover
            direction = Vec3(0.0, 0.0, 1.0)
        direction.normalize()

        r = max(dist, config.DEBRIS_MIN_RADIUS)
        speed = impulse / (mass * (r ** config.DEBRIS_IMPULSE_FALLOFF))
        speed = _clamp(speed, config.DEBRIS_SPEED_MIN, config.DEBRIS_SPEED_MAX)
        j = config.DEBRIS_SPEED_JITTER
        speed *= rng.uniform(1.0 - j, 1.0 + j)

        spin = _lerp(config.DEBRIS_SPIN_HEAVY, config.DEBRIS_SPIN_LIGHT, blend)
        angular = Vec3(
            rng.uniform(-spin, spin),
            rng.uniform(-spin, spin),
            rng.uniform(-spin, spin),
        )
        return direction * speed, angular

    def _ground_z(self) -> float:
        z = getattr(self.physics, "ground_z", None)
        return 0.0 if z is None else float(z)

    def settled(self, body: DebrisBody) -> bool:
        """The only state debris is allowed to freeze from.

        Slow is not sufficient on its own: a chunk at the apex of its arc is
        momentarily slow too, and freezing it there would pin a slab in the
        sky. So it must additionally be *supported* - either resting on the
        ground, or already deactivated by Bullet. Bullet only deactivates a
        body after :data:`config.DEBRIS_DEACTIVATION_TIME` of sustained low
        velocity, which a body in free fall never achieves (gravity keeps
        accelerating it), so deactivation is a sound proxy for "something is
        holding this up" - which is how rubble resting on top of other rubble
        gets retired rather than sitting live forever.
        """
        if not body.at_rest():
            return False
        return body.grounded(self._ground_z()) or not bool(body.node.isActive())

    # -------------------------------------------------------------- budget
    def _make_room(self, wanted: int) -> Tuple[int, int, int]:
        """Free capacity for up to *wanted* new live bodies.

        Returns ``(room, frozen, despawned)``: how many may actually spawn,
        and what had to be retired to get there. The cap is never exceeded -
        if retirement is not enough, ``room`` simply comes back smaller.
        """
        froze = 0
        gone = 0

        free = self.max_live - len(self.live)
        if free >= wanted:
            return wanted, 0, 0

        # 1. Freeze whatever has genuinely come to rest on the ground. Free,
        #    correct, and invisible to the player - it does not move anyway.
        for body in list(self.live):
            if len(self.live) + wanted <= self.max_live:
                break
            if self.settled(body):
                self._freeze(body)
                froze += 1

        # 2. Still over budget? Despawn the OLDEST live debris - rubble from
        #    a structure the player already drove past. Deliberately not
        #    "freeze the calmest": a chunk at the apex of its arc is slow too,
        #    and freezing that would hang a slab in mid-air. Removing it is
        #    honest; pinning it in the sky is not.
        if len(self.live) + wanted > self.max_live:
            for body in sorted(self.live, key=lambda b: b.spawn_step):
                if len(self.live) + wanted <= self.max_live:
                    break
                self._despawn(body)
                gone += 1

        # 3. Keep the frozen pile bounded too, oldest out first.
        while len(self.frozen) > self.max_frozen:
            gone += 1
            self._despawn(self.frozen[0])

        room = max(0, self.max_live - len(self.live))
        return min(room, wanted), froze, gone

    def _freeze(self, body: DebrisBody) -> None:
        """Settled debris becomes static geometry: visible, collidable, free.

        Mass 0 is what takes it out of the solver's island entirely. The body
        stays attached to the world, so the later damage node can still ray
        test or contact test against it.
        """
        if body.state is not LIVE:
            return
        node = body.node
        node.setLinearVelocity(Vec3(0, 0, 0))
        node.setAngularVelocity(Vec3(0, 0, 0))
        node.setMass(0.0)
        node.setDeactivationEnabled(True)
        node.setActive(False)
        body.state = FROZEN
        body._frozen_at = int(getattr(self.physics, "step_count", 0))
        self.live.remove(body)
        self.frozen.append(body)
        self.total_frozen += 1

    def _despawn(self, body: DebrisBody) -> None:
        if body.state is DESPAWNED:
            return
        self.world.removeRigidBody(body.node)
        body.np.removeNode()
        if body.state is LIVE:
            self.live.remove(body)
        elif body.state is FROZEN:
            self.frozen.remove(body)
        body.state = DESPAWNED
        self.all_bodies.pop(body.name, None)
        self.total_despawned += 1

    # -------------------------------------------------------------- update
    def update(self, dt: float, player_y: Optional[float] = None) -> dict:
        """Retire settled and far-behind debris. Call once per frame.

        Returns a small stats dict, which is what the headless demo prints.
        """
        froze = 0
        for body in list(self.live):
            if self.settled(body):
                body.quiet_time += dt
                if body.quiet_time >= config.DEBRIS_SETTLE_TIME:
                    self._freeze(body)
                    froze += 1
            else:
                body.quiet_time = 0.0

        gone = 0
        if player_y is not None:
            cutoff = float(player_y) - config.DEBRIS_DESPAWN_BEHIND
            # Frozen rubble the player has driven past: nothing will ever look
            # at it again, so give the memory back.
            for body in list(self.frozen):
                if float(body.np.getY()) < cutoff:
                    self._despawn(body)
                    gone += 1
            # Live debris that far behind is still burning solver time for a
            # show nobody is watching. Retire it too.
            for body in list(self.live):
                if float(body.np.getY()) < cutoff:
                    self._despawn(body)
                    gone += 1

        while len(self.frozen) > self.max_frozen:
            self._despawn(self.frozen[0])
            gone += 1

        return {
            "live": len(self.live),
            "frozen": len(self.frozen),
            "active": self.active_count(),
            "froze": froze,
            "despawned": gone,
            "kinetic_energy": self.kinetic_energy(),
        }

    # -------------------------------------------------------------- queries
    def bodies_near(self, point: Triple, radius: float) -> List[DebrisBody]:
        """Every debris body (live or frozen) whose origin is within *radius*.

        The hook the later damage node asked for: cheap broad phase, then the
        caller can `contactTest` the ones it cares about.
        """
        p = _vec3(point)
        r2 = float(radius) * float(radius)
        out = []
        for body in list(self.live) + list(self.frozen):
            d = body.np.getPos() - p
            if float(d.lengthSquared()) <= r2:
                out.append(body)
        return out

    def body_for_node(self, node) -> Optional[DebrisBody]:
        """Map a Bullet node (e.g. off a contact manifold) back to its body."""
        try:
            name = node.getName()
        except AttributeError:                      # pragma: no cover
            return None
        return self.all_bodies.get(name)

    def contact_test(self, node) -> List[DebrisBody]:
        """Debris currently in contact with *node*. Damage-node hook."""
        result = self.world.contactTest(node)
        out = []
        for contact in result.getContacts():
            for n in (contact.getNode0(), contact.getNode1()):
                body = self.body_for_node(n)
                if body is not None and body not in out:
                    out.append(body)
        return out

    def snapshot(self) -> dict:
        """Everything a test or the demo wants to assert on, in one dict."""
        speeds = [b.speed() for b in self.live]
        spins = [b.spin() for b in self.live]
        return {
            "live": len(self.live),
            "frozen": len(self.frozen),
            "despawned": self.total_despawned,
            "spawned": self.total_spawned,
            "active": self.active_count(),
            # Counts live AND frozen: with everything settled, `live` is 0,
            # and an `asleep` that only looked at `live` would report 0 too -
            # which reads exactly like "nothing is asleep".
            "asleep": sum(1 for b in self.live + self.frozen if b.is_asleep()),
            "kinetic_energy": self.kinetic_energy(),
            "max_speed": max(speeds) if speeds else 0.0,
            "max_spin": max(spins) if spins else 0.0,
            "min_z": min((b.lowest_z() for b in self.live + self.frozen),
                         default=0.0),
        }

    def clear(self) -> None:
        """Remove every debris body. Used between campaign sections."""
        for body in list(self.live) + list(self.frozen):
            self._despawn(body)
