"""The render/host layer.

`TumbleApp` is a ShowBase that owns a `PhysicsWorld`, a `Player` and - only
when a window actually exists - visuals, lights, camera and input bindings.

In headless mode (`window-type none`) Panda creates no window and therefore no
default camera. Every render-only step below is guarded, so the exact same
class boots, steps and shuts down cleanly on a server with no X display. That
is what the smoke test exercises.
"""

from __future__ import annotations

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight,
    CardMaker,
    DirectionalLight,
    Vec3,
    Vec4,
    WindowProperties,
)

from . import config, structures
from .debris import DebrisField
from .physics import PhysicsWorld
from .player import InputState, Player
from .world import build_course

KEY_BINDINGS = {
    "w": "forward",
    "s": "backward",
    "a": "left",
    "d": "right",
    "shift": "sprint",
    "space": "jump",
}


class TumbleApp(ShowBase):
    def __init__(self, headless: bool = False, offscreen: bool = False) -> None:
        """headless: no graphics pipe at all. offscreen: render to a buffer
        with no visible window (used by tests to exercise the render path)."""
        config.bootstrap_display(headless, offscreen)
        super().__init__()

        #: True only when no window/graphics pipe came up.
        self.headless = headless or self.win is None
        self.offscreen = offscreen

        # ---- sim (identical in both modes) --------------------------------
        self.physics = PhysicsWorld()
        build_course(self.physics)

        # Destructibles are fractured ONCE, here at load time (~220 ms for
        # all four archetypes, measured) - never on the frame one comes down.
        self.destructibles = structures.build_destructibles(self.physics)
        self.debris = DebrisField(self.physics)

        self.player = Player(self.physics)
        self.input_state = InputState()

        self.frames_run = 0
        self.mouse_centered = False
        self.demolitions = 0

        # ---- render (guarded) --------------------------------------------
        if not self.headless:
            self.physics.root.reparentTo(self.render)
            self._build_visuals()
            self._build_destructible_visuals()
            self._setup_lights()
            self._setup_camera()
            self._bind_keys()
            self.accept("escape", self.user_exit)
            # Placeholder trigger until the weapon node lands: blow up the
            # nearest structure still standing.
            self.accept("f", self.demolish_nearest)

        self.taskMgr.add(self._update, "tumble-update")

    # ------------------------------------------------------------- visuals
    def _build_visuals(self) -> None:
        """Mirror every authored collision box with a visible box."""
        box = self.loader.loadModel("models/box.egg")
        # models/box.egg spans (0,0,0)-(1,1,1); recentre so scaling is about
        # the body's own origin and visual matches the collision half-extents.
        for i, spec in enumerate(self.physics.box_specs):
            np_ = self.physics.body(spec.name)
            vis = box.copyTo(np_)
            hx, hy, hz = spec.half_extents
            vis.setScale(hx * 2.0, hy * 2.0, hz * 2.0)
            vis.setPos(-hx, -hy, -hz)
            shade = 0.45 + 0.1 * (i % 4)
            vis.setColor(shade, shade * 0.92, shade * 0.85, 1.0)

        # Visible ground card (the collision ground is an infinite plane).
        cm = CardMaker("ground-card")
        span = config.COURSE_LENGTH
        cm.setFrame(-span, span, -span, span)
        ground = self.render.attachNewNode(cm.generate())
        ground.setP(-90.0)
        ground.setZ(0.01)
        ground.setColor(0.22, 0.26, 0.22, 1.0)

        # Locked 1A: black void. Everything readable is edge-lit.
        self.setBackgroundColor(0.0, 0.0, 0.0, 1.0)

    def _build_destructible_visuals(self) -> None:
        """Intact structures as neon silhouettes (locked 1A treatment)."""
        from .render_debris import make_structure_wireframe

        self._structure_visuals = {}
        for d in self.destructibles:
            wire = make_structure_wireframe(d)
            wire.reparentTo(self.render)
            self._structure_visuals[d.name] = wire

    def _swap_to_debris_visuals(self, destructible, event) -> None:
        """Silhouette off, glowing chunk outlines on."""
        from .render_debris import attach_debris_visuals

        wire = getattr(self, "_structure_visuals", {}).pop(
            destructible.name, None
        )
        if wire is not None and not wire.isEmpty():
            wire.removeNode()
        attach_debris_visuals(event.bodies)

    def _setup_lights(self) -> None:
        amb = AmbientLight("ambient")
        amb.setColor(Vec4(0.45, 0.47, 0.52, 1.0))
        self.render.setLight(self.render.attachNewNode(amb))

        sun = DirectionalLight("sun")
        sun.setColor(Vec4(0.85, 0.82, 0.75, 1.0))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(-35.0, -55.0, 0.0)
        self.render.setLight(sun_np)

    def _setup_camera(self) -> None:
        self.disableMouse()
        self.camLens.setFov(90.0)
        self.camLens.setNear(0.1)
        self.camLens.setFar(600.0)
        self.player.attach_camera(self.camera)

        # Offscreen buffers have no cursor or pointer to capture.
        if not self.offscreen and hasattr(self.win, "requestProperties"):
            props = WindowProperties()
            props.setCursorHidden(True)
            props.setMouseMode(WindowProperties.M_relative)
            self.win.requestProperties(props)

    def _bind_keys(self) -> None:
        for key, field in KEY_BINDINGS.items():
            self.accept(key, self._set_key, [field, True])
            self.accept(f"{key}-up", self._set_key, [field, False])

    def _set_key(self, field: str, value: bool) -> None:
        setattr(self.input_state, field, value)

    # ---------------------------------------------------------------- input
    def _poll_mouse(self) -> None:
        """Relative mouse mode: read the delta, then recentre the pointer."""
        if self.headless or self.offscreen:
            return
        if not self.mouseWatcherNode.hasMouse():
            return
        md = self.win.getPointer(0)
        cx = self.win.getXSize() // 2
        cy = self.win.getYSize() // 2
        if self.mouse_centered:
            self.input_state.mouse_dx = md.getX() - cx
            self.input_state.mouse_dy = md.getY() - cy
        if self.win.movePointer(0, cx, cy):
            self.mouse_centered = True

    # ---------------------------------------------------------------- frame
    def _update(self, task):
        dt = min(self.clock.getDt(), 0.25)
        self.step_frame(dt)
        return task.cont

    def step_frame(self, dt: float) -> int:
        """One host frame: input -> intent -> fixed-step sim -> camera.

        Returns the number of physics substeps actually taken. Callable
        directly from a test with no task manager involved.
        """
        self._poll_mouse()
        self.player.apply_input(self.input_state)
        self.input_state.clear_mouse()
        steps = self.physics.advance(dt)
        # Retire settled and far-behind debris. Skipping this is how the
        # body budget gets eaten: the debris itself would simulate forever,
        # quite correctly, and never leave.
        self.debris.update(steps * self.physics.fixed_dt,
                           player_y=self.player.pos[1])
        self.player.sync_camera()
        self.frames_run += 1
        return steps

    # --------------------------------------------------------- destruction
    def demolish(self, destructible, impact_point=None,
                 impulse: float = config.DEBRIS_IMPULSE):
        """Bring one structure down: proxies out, debris in, visuals swapped.

        The single call the weapon/damage node and the campaign node make.
        Returns the ShatterEvent, or None if it was already rubble.
        """
        event = self.debris.demolish(
            destructible, impact_point=impact_point, impulse=impulse
        )
        if event is None:
            return None
        self.demolitions += 1
        if not self.headless:
            self._swap_to_debris_visuals(destructible, event)
        return event

    def demolish_nearest(self, impulse: float = config.DEBRIS_IMPULSE):
        """Demolish whichever structure is still standing and closest."""
        target = structures.nearest_intact(self.destructibles, self.player.pos)
        if target is None:
            return None
        return self.demolish(target, impulse=impulse)


def run(headless: bool = False) -> TumbleApp:
    app = TumbleApp(headless=headless)
    app.run()
    return app
