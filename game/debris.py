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

3. **Bullet's sleep works, but it is slow, and the freeze is what makes rest
   cheap.** Re-measured properly (see `tests/test_settling_is_real.py`, which
   runs with the freeze *disabled*): pure Bullet does bring a 150-chunk
   cluster collapse all the way to rest on its own - peak ~10 m/s at t=2 s,
   decaying to exactly zero with every body deactivated by t~24 s. So the
   tuning is genuinely convergent and the settling is real physics, not a
   trick.

   What it is not, is *prompt* or *cheap*: for the ~20 s between "visually
   stopped" and "Bullet finally deactivates", a couple of hundred bodies stay
   in the solver's islands, and a fresh collapse nearby wakes them straight
   back up. So :class:`DebrisField` runs a stricter settle detector on top,
   and *freezes* a body that has been quiet for
   :data:`config.DEBRIS_SETTLE_TIME` **and** is actually touching down: mass
   goes to 0 and it becomes static geometry. Still there, still visible, still
   collidable by the later damage node - it just costs the solver nothing.

   An earlier draft of this docstring claimed Bullet "never" sleeps a dense
   pile and that ~250 of 260 bodies stay active indefinitely. That was wrong:
   measured, 129 of 260 deactivate unaided, and with freezing off the whole
   pile reaches exact zero. The freeze is a performance and promptness
   measure, not a cover-up, and the tests now prove the difference.

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
        "spawn_step", "spawn_time", "quiet_time", "structure", "base_mass",
        "size_blend", "_frozen_at", "chunk_index", "_volume",
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
        spawn_time: float = 0.0,
    ) -> None:
        self.name = name
        self.chunk = chunk
        #: The source chunk's index and volume, cached as plain numbers so
        #: they outlive :meth:`release` - history records legitimately want
        #: "which chunk was this, how big was it" long after the body is gone.
        self.chunk_index = int(chunk.index)
        self._volume = float(chunk.volume)
        self.node = node
        self.np = np_
        self.structure = structure
        self.base_mass = base_mass
        self.size_blend = size_blend
        self.spawn_step = spawn_step
        #: Field clock reading (simulated seconds) when this body spawned.
        self.spawn_time = float(spawn_time)
        self.state = LIVE
        self.quiet_time = 0.0
        self._frozen_at = -1

    # ------------------------------------------------------------- teardown
    def release(self) -> None:
        """Drop the Bullet handles once this body is gone from the world.

        Called by :meth:`DebrisField._despawn`, which has already detached the
        rigid body and removed the NodePath. Without this, *any* surviving
        Python reference to the body keeps its C++ ``BulletRigidBodyNode`` and
        its ``BulletConvexHullShape`` alive - and surviving references are
        normal, because :class:`ShatterEvent` and
        :class:`~game.damage.DamageReport` are history records that hold the
        bodies they describe.

        That is a real leak, not a theoretical one. Measured on this stack
        over a 60-structure demolition soak: RSS climbed 92 -> 157 MB at a
        dead-linear 1.08 MB per structure, with the *plateau ratio* of late to
        early growth rate at 0.99 - i.e. not converging at all - while the
        live body count sat flat at the cap the whole time. The budget was
        perfectly honest about what it was simulating; the corpses were the
        problem. Diagnosis: 6528 of 6864 DebrisBody objects alive were in
        state DESPAWNED, and every single one of them was reachable from the
        retained event/report history.

        Nulling the handles here means the history keeps what history is for -
        names, masses, volumes, chunk indices, which structure - and owns none
        of the simulation.

        The source :class:`~game.fracture.Chunk` goes with them, for the same
        reason and with its own measurement. A chunk descriptor is its vertex,
        edge and face lists - the single largest thing a body points at - and
        a despawned body is the only thing still pointing at it once
        :meth:`~game.structures.Destructible.release_chunks` has run. Keeping
        it cost a further 0.56 MB per structure, dead linear (measured: 92 ->
        122 MB over 48 structures with the Bullet handles already released).
        Every number the history actually reads - :attr:`volume`,
        :attr:`chunk_index`, :attr:`base_mass`, :attr:`size_blend` - is cached
        as a plain float or int at construction, so dropping the descriptor
        costs the history nothing.

        Together the two releases take the soak from +1.08 MB per structure
        with a 0.99 plateau ratio (no convergence) to a genuine plateau.

        Idempotent. After this, pose, velocity and geometry queries raise
        :class:`ReferenceError` rather than touching freed memory - see
        :meth:`_live_node`.
        """
        self.node = None
        self.np = None
        self.chunk = None

    def _live_node(self):
        """The Bullet node, or a clear error if this body has been released."""
        if self.node is None:
            raise ReferenceError(
                f"debris body {self.name!r} was despawned and released; its "
                f"Bullet handles are gone. History records keep a body's "
                f"identity, not its physics."
            )
        return self.node

    # ------------------------------------------------------------- queries
    @property
    def pos(self) -> Vec3:
        if self.np is None:
            self._live_node()          # raises ReferenceError with context
        return self.np.getPos()

    @property
    def quat(self) -> Quat:
        if self.np is None:
            self._live_node()
        return self.np.getQuat()

    @property
    def mass(self) -> float:
        """Live Bullet mass. Frozen debris reads 0 (it is static)."""
        return float(self._live_node().getMass())

    @property
    def volume(self) -> float:
        """The source chunk's volume. Survives :meth:`release`."""
        return self._volume

    def linear_velocity(self) -> Vec3:
        return self._live_node().getLinearVelocity()

    def angular_velocity(self) -> Vec3:
        return self._live_node().getAngularVelocity()

    def speed(self) -> float:
        return float(self._live_node().getLinearVelocity().length())

    def spin(self) -> float:
        return float(self._live_node().getAngularVelocity().length())

    def kinetic_energy(self) -> float:
        """Translational KE only - enough to watch the pile calm down."""
        if self.state is not LIVE:
            return 0.0
        v = self._live_node().getLinearVelocity()
        return 0.5 * self.base_mass * float(v.lengthSquared())

    def is_active(self) -> bool:
        """True only while Bullet is still solving this body."""
        return (self.state is LIVE and self.node is not None
                and bool(self.node.isActive()))

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
        return not bool(self._live_node().isActive())

    def age(self, now: float) -> float:
        """Simulated seconds this body has existed, given the field clock."""
        return max(0.0, float(now) - self.spawn_time)

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
        if self.np is None or self.chunk is None:
            self._live_node()
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
        if self.np is None or self.chunk is None:
            self._live_node()
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
        #: Simulated seconds this field has been stepped, advanced by
        #: :meth:`update`. Debris age is measured against it, so age-based
        #: reaping is independent of wall clock and of frame rate.
        self.clock = 0.0
        #: Last known player position, set by :meth:`set_player` (which
        #: :meth:`update` calls for you). The budget needs it: eviction is
        #: only allowed to take debris the player is not looking at.
        self._player_pos: Optional[Vec3] = None
        #: Counts of what the budget has actually done, for the tests and the
        #: demo to assert on.
        self.total_evicted = 0
        self.total_floor_despawned = 0
        self.total_behind_despawned = 0

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

    @staticmethod
    def is_stepped(body: "DebrisBody") -> bool:
        """True when the solver is still doing work for this body.

        This is the quantity :data:`config.DEBRIS_MAX_LIVE` bounds. A frozen
        body is mass-0 static geometry - present, visible, collidable, and
        *not* stepped.
        """
        return body.state is LIVE

    def stepped_count(self) -> int:
        """Number of debris bodies currently in the dynamic simulation."""
        return sum(1 for b in self.live if self.is_stepped(b))

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
        # A ShatterEvent holds its bodies. An unbounded history therefore
        # pins every body ever spawned, and _despawn() frees nothing but a
        # dict entry - the cap stays honest while memory climbs forever.
        # Measured: unbounded, a 14-structure soak kept all 2258 spawned
        # bodies resident; bounded, 297 (== what is actually in the world).
        excess = len(self.events) - config.DEBRIS_EVENT_HISTORY
        if excess > 0:
            del self.events[:excess]
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
            spawn_time=self.clock,
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

    def is_supported(self, body: DebrisBody) -> bool:
        """True when something is provably holding this body up.

        Slow is not sufficient to call debris settled: a chunk at the apex of
        its arc is momentarily slow too, and freezing it there would pin a
        slab in the sky. So a body must also be *supported*, which is any of:

        * its lowest vertex is on the ground (:meth:`DebrisBody.grounded`) -
          the cheap, obvious case;
        * Bullet has deactivated it;
        * it has stayed below the settle thresholds for a full
          :data:`config.DEBRIS_SETTLE_TIME`. **Sustained quiet is itself the
          proof of support**: nothing in free flight can stay slow. Gravity
          takes 0.028 s to accelerate a released body past
          :data:`config.DEBRIS_SETTLE_LINEAR` (0.28 m/s), and 0.7 s of free
          fall reaches 6.9 m/s - 24x the threshold. A body that has been quiet
          for the whole dwell window is therefore resting on *something*, even
          if that something is other rubble rather than the ground.

        Why the dwell term is needed at all
        -----------------------------------
        The original test was ``grounded or not isActive()``, taking Bullet's
        deactivation as the proxy for "something is holding this up". That
        proxy is broken by the player:
        :class:`~panda3d.bullet.BulletCharacterControllerNode` is a kinematic
        body that is *always* active, so every contact island it touches is
        kept awake. A rubble pile the player is standing in or against never
        deactivates; debris resting on that pile is then at rest, not
        grounded, and permanently "unsettled" - so it never freezes and burns
        solver time forever.

        Measured, before this fix: 3 structures demolished, then 40 s of
        simulated quiet, and 119 of 371 debris bodies stayed live and active
        indefinitely, all of them at rest and none of them grounded.
        Teleporting the player away, or removing the character from the world,
        dropped the same run to 0 live / 371 frozen - which is what identified
        the cause. With the dwell term the run settles to 0 active with the
        player left exactly where it stood.

        A contact-manifold sweep was tried first and rejected: it is correct,
        but `BulletWorld.getManifold()` leaks in these bindings (measured ~16
        MB per 1000 steps over 536 manifolds, growing without bound), which
        would trade a solver leak for a memory leak. The dwell test needs no
        Bullet query at all.
        """
        if body.grounded(self._ground_z()):
            return True
        if not bool(body.node.isActive()):
            return True
        return body.quiet_time >= config.DEBRIS_SETTLE_TIME

    def settled(self, body: DebrisBody) -> bool:
        """The only state debris is allowed to freeze from: at rest AND held up.

        See :meth:`is_supported` for why "held up" cannot simply be "Bullet
        deactivated it".
        """
        if not body.at_rest():
            return False
        return self.is_supported(body)

    # ---------------------------------------------------- player awareness
    def set_player(
        self,
        pos: Optional[Triple] = None,
        y: Optional[float] = None,
    ) -> Optional[Vec3]:
        """Tell the field where the player is. Returns the stored position.

        Either a full ``(x, y, z)`` or just *y*. The course is a corridor run
        down +Y, so when only *y* is known the previously-known x/z (or the
        centreline) stands in for the rest - good enough for the only thing
        this is used for, which is deciding whether debris is near enough and
        far enough forward to be off-limits to eviction.
        """
        if pos is not None:
            self._player_pos = _vec3(pos)
        elif y is not None:
            old = self._player_pos
            self._player_pos = Vec3(
                float(old.getX()) if old is not None else 0.0,
                float(y),
                float(old.getZ()) if old is not None else 0.0,
            )
        return self._player_pos

    @property
    def player_pos(self) -> Optional[Vec3]:
        return self._player_pos

    @property
    def player_y(self) -> Optional[float]:
        return None if self._player_pos is None else float(self._player_pos.getY())

    def player_distance(self, body: DebrisBody) -> Optional[float]:
        """Horizontal (XY) distance player -> body, or None if unknown.

        Horizontal on purpose: a slab 20 m up and 2 m away is very much the
        player's problem, and a vertical term would rank it as distant.
        """
        if self._player_pos is None:
            return None
        d = body.np.getPos() - self._player_pos
        return math.hypot(float(d.getX()), float(d.getY()))

    def behind_distance(self, body: DebrisBody) -> float:
        """How far *behind* the player this body is, in metres (+Y forward).

        Negative means it is in front of the player. 0.0 when the player
        position is unknown, which makes the eviction order fall back to pure
        age rather than inventing a geometry it does not have.
        """
        if self._player_pos is None:
            return 0.0
        return float(self._player_pos.getY()) - float(body.np.getY())

    def in_flight(self, body: DebrisBody) -> bool:
        """Still moving under its own momentum: the gameplay threat."""
        return body.state is LIVE and not body.at_rest()

    # ------------------------------------------------------------ eviction
    def eviction_protected(self, body: DebrisBody) -> bool:
        """True when this body must NEVER be evicted to free budget.

        Two ways to earn protection, both within
        :data:`config.DEBRIS_PROTECT_RADIUS` of the player:

        * it is **in flight** - that is the chunk arcing toward the player,
          the whole point of the destruction; deleting it mid-air is a lie;
        * it is **in front of** the player - visible, so removing it would
          pop geometry out of the view the player is pointed at.

        Rubble settled behind the player, or anything at all beyond the
        protection radius, is fair game.
        """
        if body.state is DESPAWNED:
            return False
        dist = self.player_distance(body)
        if dist is None or dist > config.DEBRIS_PROTECT_RADIUS:
            return False
        if self.in_flight(body):
            return True
        return self.behind_distance(body) < 0.0

    def eviction_rank(self, body: DebrisBody) -> Tuple:
        """Deterministic sort key: the *best* eviction candidate sorts first.

        Order of preference, least interesting first:

        1. settled/frozen before anything still moving;
        2. farther behind the player before nearer;
        3. older before newer;
        4. name, so the choice is total and reproducible.
        """
        moving = 0 if (body.state is FROZEN or not self.in_flight(body)) else 1
        return (
            moving,
            -self.behind_distance(body),
            body.spawn_time,
            body.spawn_step,
            body.name,
        )

    def eviction_candidates(
        self,
        pool: Optional[Iterable[DebrisBody]] = None,
    ) -> List[DebrisBody]:
        """Evictable bodies, best candidate first. Protected ones excluded.

        Pure with respect to the field: it reads state and returns an order,
        it never removes anything. That is what lets a test assert the policy
        without a window and without stepping a world.
        """
        bodies = list(self.live) if pool is None else list(pool)
        return sorted(
            (b for b in bodies
             if b.state is not DESPAWNED and not self.eviction_protected(b)),
            key=self.eviction_rank,
        )

    def evict_for_budget(self, needed: int) -> int:
        """Despawn up to *needed* live bodies, least interesting first.

        Returns how many were actually evicted, which can be fewer than
        *needed* if everything left is protected. The caller's job is then to
        spawn less - not to break protection.
        """
        gone = 0
        for body in self.eviction_candidates():
            if gone >= int(needed):
                break
            self._despawn(body)
            self.total_evicted += 1
            gone += 1
        return gone

    def despawn_out_of_world(
        self,
        floor_z: Optional[float] = None,
    ) -> int:
        """Despawn debris that has fallen out of the world. Returns the count."""
        limit = (config.DEBRIS_WORLD_FLOOR_Z if floor_z is None
                 else float(floor_z))
        gone = 0
        for body in list(self.live) + list(self.frozen):
            if float(body.np.getZ()) < limit:
                self._despawn(body)
                self.total_floor_despawned += 1
                gone += 1
        return gone

    def despawn_behind_player(
        self,
        player_y: Optional[float] = None,
        behind: Optional[float] = None,
    ) -> int:
        """Despawn debris the player has driven far enough past. Count out.

        Uses :data:`config.DEBRIS_DESPAWN_BEHIND`, which is well past the
        protection radius, so this can never remove something in view.
        """
        if player_y is None:
            player_y = self.player_y
        if player_y is None:
            return 0
        cutoff = float(player_y) - (config.DEBRIS_DESPAWN_BEHIND
                                    if behind is None else float(behind))
        gone = 0
        for body in list(self.live) + list(self.frozen):
            if float(body.np.getY()) < cutoff:
                self._despawn(body)
                self.total_behind_despawned += 1
                gone += 1
        return gone

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

        # 2. Still over budget? Evict, by the policy in
        #    :meth:`eviction_rank`: settled before moving, farthest behind
        #    the player before nearest, oldest before newest - and never a
        #    body :meth:`eviction_protected` covers (in flight near the
        #    player, or anywhere in front of them within the protection
        #    radius). Deliberately not "freeze the calmest": a chunk at the
        #    apex of its arc is slow too, and freezing that would hang a slab
        #    in mid-air. Removing rubble behind the player is honest; pinning
        #    a slab in the sky, or vanishing one from the player's view, is
        #    not. If every remaining body is protected we evict nothing and
        #    `room` simply comes back smaller.
        over = (len(self.live) + wanted) - self.max_live
        if over > 0:
            gone += self.evict_for_budget(over)

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
        # The body is out of the world; drop its Bullet handles so the
        # retained event/report history cannot pin the C++ rigid body and
        # collision shape. See DebrisBody.release for the measurement.
        body.release()

    # -------------------------------------------------------------- update
    def reap(
        self,
        max_age: Optional[float] = None,
        behind_y: Optional[float] = None,
        cap: Optional[int] = None,
    ) -> int:
        """Despawn debris by age, by distance behind the player, or by cap.

        The lifecycle mechanism, exposed but deliberately **untuned**: every
        criterion is off unless the caller passes it. Choosing the actual
        budget, and wiring this to the game loop, is the next node's job - this
        module only guarantees the mechanism exists, is exact, and that
        :attr:`live_count` / :attr:`total_bodies` reflect it immediately.

        Parameters
        ----------
        max_age:
            Despawn any body older than this many *simulated* seconds, measured
            against :attr:`clock` (which :meth:`update` advances). Frozen and
            live debris alike.
        behind_y:
            Despawn any body whose world Y is below this cutoff - debris the
            player has driven past and will never look at again.
        cap:
            Hard ceiling on total bodies. The oldest are despawned first until
            :attr:`total_bodies` is at or below it.

        Returns the number of bodies despawned.
        """
        gone = 0

        if max_age is not None:
            limit = float(max_age)
            for body in list(self.live) + list(self.frozen):
                if body.age(self.clock) > limit:
                    self._despawn(body)
                    gone += 1

        if behind_y is not None:
            cutoff = float(behind_y)
            for body in list(self.live) + list(self.frozen):
                if float(body.np.getY()) < cutoff:
                    self._despawn(body)
                    gone += 1

        if cap is not None:
            ceiling = max(0, int(cap))
            while self.total_bodies > ceiling:
                # Oldest first: newest debris is the debris being watched.
                oldest = min(list(self.live) + list(self.frozen),
                             key=lambda b: (b.spawn_time, b.spawn_step))
                self._despawn(oldest)
                gone += 1

        return gone

    def update(
        self,
        dt: float,
        player_y: Optional[float] = None,
        max_age: Optional[float] = None,
        player_pos: Optional[Triple] = None,
    ) -> dict:
        """Retire settled and far-behind debris. Call once per frame.

        Advances :attr:`clock` by *dt*, so debris age tracks simulated time.
        Pass *max_age* to also reap by age; it is ``None`` (off) by default
        because picking that number is the budget node's call, not this one's.

        Returns a small stats dict, which is what the headless demo prints.
        """
        self.clock += float(dt)
        if player_pos is not None or player_y is not None:
            self.set_player(pos=player_pos, y=player_y)
        froze = 0
        for body in list(self.live):
            # The timer runs on `at_rest` alone, NOT on `settled`: the dwell
            # is one of the things `is_supported` reads, so making it
            # conditional on support would be circular and a body resting on
            # rubble could never accumulate the credit that proves it is
            # resting on rubble.
            if body.at_rest():
                body.quiet_time += dt
            else:
                body.quiet_time = 0.0
            if body.quiet_time >= config.DEBRIS_SETTLE_TIME and self.settled(body):
                self._freeze(body)
                froze += 1

        gone = 0
        if max_age is not None:
            gone += self.reap(max_age=max_age)

        # Fallen out of the world entirely: off the edge of the ground, or
        # through a gap. Unreachable and unseeable, so it goes. Checked before
        # the behind-player sweep because a body can be both.
        gone += self.despawn_out_of_world()

        # Driven past by DEBRIS_DESPAWN_BEHIND metres - frozen rubble nothing
        # will look at again, and live debris still burning solver time for a
        # show nobody is watching. Both retired.
        gone += self.despawn_behind_player()

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
            "clock": self.clock,
            "oldest_age": max((b.age(self.clock)
                               for b in self.live + self.frozen), default=0.0),
            # The budget, and what enforcing it has cost so far.
            "stepped": self.stepped_count(),
            "max_live": self.max_live,
            "evicted": self.total_evicted,
            "floor_despawned": self.total_floor_despawned,
            "behind_despawned": self.total_behind_despawned,
        }

    def clear(self) -> None:
        """Remove every debris body. Used between campaign sections."""
        for body in list(self.live) + list(self.frozen):
            self._despawn(body)
