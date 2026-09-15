"""The simulation layer.

`PhysicsWorld` owns a Bullet world and a detached NodePath tree. It holds no
reference to ShowBase, `render`, the loader or any window, which is what makes
the whole game headless-testable: a test can build a world, step it and assert
on numbers without a display ever existing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from panda3d.bullet import (
    BulletBoxShape,
    BulletPlaneShape,
    BulletRigidBodyNode,
    BulletWorld,
)
from panda3d.core import NodePath, TransformState, Vec3

from . import config

Triple = Sequence[float]


@dataclass(frozen=True)
class BoxSpec:
    """Authored description of a box body: the render layer mirrors these."""

    name: str
    pos: tuple
    half_extents: tuple
    static: bool = True


def reclaim_interned_states() -> int:
    """Hand Panda's interned-pose cache back. Returns states reclaimed.

    Every time Bullet writes a body's transform, Panda interns a new
    :class:`TransformState` in a global table and keeps it there. In the
    windowed game ``GraphicsEngine::render_frame`` calls
    ``TransformState::garbage_collect()`` once per rendered frame, so the table
    stays small and nobody notices. **Headless there is no rendered frame**,
    so nothing ever collects it.

    Measured on this stack (a 14-structure demolition soak, 2258 debris bodies
    spawned): with no reclaim the table reached 679,844 states and RSS climbed
    92 -> 358 MB and was still rising linearly; calling this every step holds
    the table at ~2,000 states and RSS at 92 -> 115 MB. So this is a real leak
    in any long headless run, not a tuning preference.

    It is called per step rather than batched on purpose. The collector is
    incremental - its cost is proportional to the garbage waiting - so per-step
    it costs 0.093 ms mean / 0.257 ms max (measured), while deferring it to
    once a second turns the same total work into a 4.8 ms spike inside one
    frame. Small and constant beats rare and lumpy when there is a frame
    budget.
    """
    return int(TransformState.garbageCollect())


class PhysicsWorld:
    """A fixed-timestep Bullet world with no rendering dependencies."""

    def __init__(
        self,
        gravity_z: float = config.GRAVITY_Z,
        fixed_dt: float = config.FIXED_DT,
    ) -> None:
        # Bullet reads its solver settings at world-construction time, so this
        # must happen before the BulletWorld below exists. Idempotent.
        config.ensure_bullet_config()

        self.world = BulletWorld()
        self.world.setGravity(Vec3(0.0, 0.0, gravity_z))
        self.gravity_z = gravity_z
        self.fixed_dt = fixed_dt

        #: Detached scene root. The render layer may reparent this under
        #: `render`; nothing in the sim requires that it ever happens.
        self.root = NodePath("sim-root")

        self.bodies: Dict[str, NodePath] = {}
        self.box_specs: List[BoxSpec] = []
        self.ground_np: NodePath | None = None
        self.ground_z: float | None = None

        self.step_count = 0
        self.sim_time = 0.0
        #: How many interned TransformStates stepping has handed back. See
        #: :func:`reclaim_interned_states` - without this the sim leaks ~1 MB
        #: of pose cache per simulated second of falling debris.
        self.states_reclaimed = 0
        self._accumulator = 0.0
        self._lag_dropped = 0.0

    # ------------------------------------------------------------ authoring
    def add_ground_plane(self, z: float = 0.0, friction: float = 1.0) -> NodePath:
        """An infinite static plane at `z`, normal +Z."""
        node = BulletRigidBodyNode("ground")
        node.addShape(BulletPlaneShape(Vec3(0.0, 0.0, 1.0), z))
        node.setFriction(friction)
        np_ = self.root.attachNewNode(node)
        self.world.attachRigidBody(node)
        self.ground_np = np_
        self.bodies["ground"] = np_
        self.ground_z = z
        return np_

    def add_static_box(
        self,
        name: str,
        pos: Triple,
        half_extents: Triple,
        friction: float = 0.8,
    ) -> NodePath:
        """Immovable box. Mass stays 0, so Bullet treats it as static."""
        return self._add_box(name, pos, half_extents, mass=0.0, friction=friction)

    def add_dynamic_box(
        self,
        name: str,
        pos: Triple,
        half_extents: Triple,
        mass: float = 1.0,
        friction: float = 0.6,
    ) -> NodePath:
        """Falling/pushable box. Used by tests and by later gameplay nodes."""
        return self._add_box(name, pos, half_extents, mass=mass, friction=friction)

    def _add_box(
        self,
        name: str,
        pos: Triple,
        half_extents: Triple,
        mass: float,
        friction: float,
    ) -> NodePath:
        if name in self.bodies:
            raise ValueError(f"duplicate body name: {name!r}")
        node = BulletRigidBodyNode(name)
        node.addShape(BulletBoxShape(Vec3(*half_extents)))
        node.setMass(mass)
        node.setFriction(friction)
        if mass > 0.0:
            # Keep dynamic bodies awake: deactivation makes step-count based
            # assertions (and later destruction chains) unpredictable.
            node.setDeactivationEnabled(False)
            # Continuous collision detection. Measured on this stack: without
            # it, a body moving at 200 m/s passes straight through the ground
            # plane (ends up at z=+106 or below the floor entirely); with it,
            # it rests correctly every time. Fast debris from later
            # destruction nodes would otherwise escape the world.
            smallest = min(float(v) for v in half_extents)
            node.setCcdMotionThreshold(smallest * 0.5)
            node.setCcdSweptSphereRadius(smallest * 0.7)
        np_ = self.root.attachNewNode(node)
        np_.setPos(*pos)
        self.world.attachRigidBody(node)

        self.bodies[name] = np_
        self.box_specs.append(
            BoxSpec(name, tuple(float(v) for v in pos),
                    tuple(float(v) for v in half_extents), static=mass == 0.0)
        )
        return np_

    # -------------------------------------------------------------- stepping
    def step_fixed(self, steps: int = 1) -> int:
        """Advance exactly `steps` substeps of `fixed_dt`. Deterministic."""
        for _ in range(steps):
            # max_substeps=1 with a matching substep size forbids Bullet from
            # inventing its own interpolation: one call, one fixed tick.
            self.world.doPhysics(self.fixed_dt, 1, self.fixed_dt)
            self.step_count += 1
            self.sim_time += self.fixed_dt
            self.states_reclaimed += reclaim_interned_states()
        return steps

    def advance(self, dt: float, max_steps: int = config.MAX_STEPS_PER_FRAME) -> int:
        """Consume a wall-clock frame delta, running whole fixed steps only.

        This is the decoupling point: render framerate affects *how many*
        steps happen per frame, never the size of a step.
        """
        self._accumulator += dt
        taken = 0
        while self._accumulator >= self.fixed_dt and taken < max_steps:
            self.step_fixed(1)
            self._accumulator -= self.fixed_dt
            taken += 1
        if taken >= max_steps and self._accumulator > self.fixed_dt:
            # Too far behind to catch up: drop the backlog rather than stall.
            self._lag_dropped += self._accumulator
            self._accumulator = 0.0
        return taken

    @property
    def pending_time(self) -> float:
        """Un-simulated remainder currently held in the accumulator."""
        return self._accumulator

    # ------------------------------------------------------------- teardown
    def remove_body(self, name: str) -> bool:
        """Detach a body from the world and the scene graph.

        Used when a destructible stops being a building: its static collision
        proxy has to leave the world on the same frame its debris arrives, or
        the player is blocked by a tower that visibly no longer exists.
        Returns False if there was no such body.
        """
        np_ = self.bodies.pop(name, None)
        if np_ is None:
            return False
        self.world.removeRigidBody(np_.node())
        np_.removeNode()
        self.box_specs = [s for s in self.box_specs if s.name != name]
        if self.ground_np is not None and name == "ground":
            self.ground_np = None
        return True

    def body(self, name: str) -> NodePath:
        return self.bodies[name]

    def lowest_z(self, name: str) -> float:
        return float(self.bodies[name].getZ())
