"""First-person ground character.

Two deliberately separate pieces:

  * `InputState` - a plain dataclass of intent (which keys are down, how much
    the mouse moved). No Panda types, trivially constructible in a test.
  * `Player`     - a Bullet character controller that consumes `InputState`.
    It never reads the keyboard or the mouse itself and never touches a
    camera, so identical code runs headless and windowed.

The character is a `BulletCharacterControllerNode`: a kinematic capsule that
is swept against the world, so it is subject to gravity and cannot tunnel
through the ground or a static wall. There is no fly or noclip path - vertical
motion only ever comes from Bullet.
"""

from __future__ import annotations

from dataclasses import dataclass

from panda3d.bullet import (
    BulletCapsuleShape,
    BulletCharacterControllerNode,
    ZUp,
)
from panda3d.core import Vec3

from . import config
from .physics import PhysicsWorld


@dataclass
class InputState:
    """Movement intent for one frame."""

    forward: bool = False
    backward: bool = False
    left: bool = False
    right: bool = False
    sprint: bool = False
    jump: bool = False
    #: Mouse delta in pixels since the previous frame.
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0

    def clear_mouse(self) -> None:
        self.mouse_dx = 0.0
        self.mouse_dy = 0.0


class Player:
    """Ground-based FPS capsule driven by `InputState`."""

    def __init__(
        self,
        physics: PhysicsWorld,
        spawn=config.SPAWN_POS,
        radius: float = config.PLAYER_RADIUS,
        height: float = config.PLAYER_HEIGHT,
    ) -> None:
        self.physics = physics
        self.radius = radius
        self.height = height
        #: Distance from capsule centre to its flat bottom.
        self.half_total_height = height * 0.5 + radius

        shape = BulletCapsuleShape(radius, height, ZUp)
        self.node = BulletCharacterControllerNode(
            shape, config.PLAYER_STEP_HEIGHT, "player"
        )
        self.np = physics.root.attachNewNode(self.node)
        self.np.setPos(*spawn)
        physics.world.attachCharacter(self.node)

        self.heading = 0.0
        self.pitch = 0.0
        #: Set by the render layer; the sim ignores it entirely.
        self.camera_np = None

    # ----------------------------------------------------------------- look
    def apply_look(self, mouse_dx: float, mouse_dy: float) -> None:
        """Yaw the body, pitch only the view. Pitch is clamped."""
        self.heading -= mouse_dx * config.MOUSE_SENSITIVITY
        self.heading = (self.heading + 180.0) % 360.0 - 180.0
        self.pitch -= mouse_dy * config.MOUSE_SENSITIVITY
        self.pitch = max(-config.PITCH_LIMIT, min(config.PITCH_LIMIT, self.pitch))
        self.np.setH(self.heading)

    # ------------------------------------------------------------- movement
    def desired_velocity(self, state: InputState) -> Vec3:
        """Local-space (right, forward, up) velocity implied by `state`.

        Forward-biased on purpose: you accelerate down the course far faster
        than you can back up or strafe, which is what gives the 'driving
        forward' feel the design asks for.
        """
        forward = 0.0
        if state.forward:
            forward += config.FORWARD_SPEED
        if state.backward:
            forward -= config.BACK_SPEED

        strafe = 0.0
        if state.right:
            strafe += config.STRAFE_SPEED
        if state.left:
            strafe -= config.STRAFE_SPEED

        if state.sprint and forward > 0.0:
            forward *= config.SPRINT_MULTIPLIER
            strafe *= config.SPRINT_MULTIPLIER

        # Panda/Bullet local axes: +X right, +Y forward, +Z up.
        return Vec3(strafe, forward, 0.0)

    def apply_input(self, state: InputState) -> Vec3:
        """Feed one frame of intent into the character controller."""
        self.apply_look(state.mouse_dx, state.mouse_dy)
        vel = self.desired_velocity(state)
        # is_local=True -> Bullet rotates the vector by the node's heading,
        # so 'forward' always means 'where you are looking'.
        self.node.setLinearMovement(vel, True)
        self.node.setAngularMovement(0.0)
        if state.jump and self.node.isOnGround():
            self.node.doJump()
        return vel

    # ---------------------------------------------------------------- query
    @property
    def pos(self) -> Vec3:
        return self.np.getPos()

    @property
    def eye_pos(self) -> Vec3:
        """World-space eye point, derived independently of any camera."""
        return self.np.getPos() + Vec3(0.0, 0.0, config.EYE_HEIGHT)

    @property
    def feet_z(self) -> float:
        """Z of the capsule's lowest point."""
        return float(self.np.getZ()) - self.half_total_height

    def on_ground(self) -> bool:
        return bool(self.node.isOnGround())

    def attach_camera(self, camera_np) -> None:
        """Render-layer hook: parent a camera at eye height. Optional."""
        self.camera_np = camera_np
        camera_np.reparentTo(self.np)
        camera_np.setPos(0.0, 0.0, config.EYE_HEIGHT)
        camera_np.setHpr(0.0, 0.0, 0.0)

    def sync_camera(self) -> None:
        """Push pitch onto the camera. Heading already lives on the body."""
        if self.camera_np is not None:
            self.camera_np.setP(self.pitch)
