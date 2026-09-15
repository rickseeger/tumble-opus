"""The locked 1A glowing-wireframe debris treatment, verified in code.

I have no vision tool, so nothing here claims the debris *looks* good - that
is Rick's call at playtest. What is mechanically checkable, and checked:

* the wireframe geometry exists, with one line segment per chunk edge from the
  fracture library's own edge list;
* it carries the additive-blend / unlit / no-depth-write state that is what
  makes edges glow and accumulate rather than merely be coloured;
* it is parented to the rigid body, so Bullet's transform carries it and
  there is no per-frame Python cost for a few hundred tumbling pieces;
* rendered offscreen, debris actually puts lit pixels on a black frame, and a
  frame with debris differs measurably from the same frame without it.

The last one is the honest stand-in for looking at the screen.
"""

import math

import pytest

from game import config, fracture, structures
from game.debris import DebrisField
from game.physics import PhysicsWorld
from game.render_debris import (
    DEBRIS_COLORS,
    DEBRIS_GLOW_PASSES,
    attach_debris_visuals,
    debris_color,
    make_chunk_wireframe,
    make_structure_wireframe,
)

DT = config.FIXED_DT


def make_field():
    w = PhysicsWorld()
    w.add_ground_plane(0.0)
    return w, DebrisField(w)


# ============================================================ geometry
def test_chunk_wireframe_has_one_segment_per_chunk_edge():
    """Every edge the fracture library emitted gets drawn, in every pass.

    LineSegs packs runs of connected segments into GeomLinestrips, so the raw
    primitive vertex count is not 2 per segment - it has to be decomposed into
    GeomLines first. Counting the packed form instead is how you get a
    plausible-looking number that is not the segment count at all.
    """
    from panda3d.core import GeomNode

    result = fracture.fracture(
        fracture.with_budget(fracture.cluster_spec(), 40), seed=3
    )
    chunk = result.chunks[0]
    root = make_chunk_wireframe(chunk)

    assert not root.isEmpty()
    assert len(chunk.edges) >= 12, "chunk has implausibly few edges"

    passes = root.getChildren()
    assert len(passes) == len(DEBRIS_GLOW_PASSES), (
        f"{len(passes)} draw passes, expected {len(DEBRIS_GLOW_PASSES)}"
    )
    for p in passes:
        gn = p.node()
        assert isinstance(gn, GeomNode)
        segments = 0
        for i in range(gn.getNumGeoms()):
            lines = gn.getGeom(i).decompose()
            for j in range(lines.getNumPrimitives()):
                segments += lines.getPrimitive(j).getNumVertices() // 2
        assert segments == len(chunk.edges), (
            f"pass {p.getName()} drew {segments} segments for "
            f"{len(chunk.edges)} chunk edges"
        )


def test_wireframe_bounds_match_the_chunk_it_draws():
    """The drawn geometry is the chunk, not a stand-in box.

    Note what `Chunk.half_extents` actually is: ``max(|coord|)`` per axis
    measured from the chunk's *centroid*, which for an irregular convex shard
    is NOT half the bounding-box span (the centroid is off-centre). So the
    honest reference is the chunk's own vertex bounding box.
    """
    result = fracture.fracture(
        fracture.with_budget(fracture.tower_spec(), 30), seed=3
    )
    checked = 0
    for chunk in result.chunks[:8]:
        root = make_chunk_wireframe(chunk)
        lo, hi = root.getTightBounds()
        for axis in range(3):
            coords = [v[axis] for v in chunk.vertices]
            assert float(lo[axis]) == pytest.approx(min(coords), abs=1e-4)
            assert float(hi[axis]) == pytest.approx(max(coords), abs=1e-4)
            # And it never exceeds the half-extent envelope the chunk reports.
            assert max(abs(c) for c in coords) == pytest.approx(
                chunk.half_extents[axis], abs=1e-6
            )
        checked += 1
    assert checked == 8


def test_glow_state_is_additive_unlit_and_does_not_write_depth():
    """This render state is what makes it read as a glowing vector display."""
    from panda3d.core import ColorBlendAttrib, DepthWriteAttrib, LightAttrib

    result = fracture.fracture(fracture.unit_cube_spec(), seed=1) if hasattr(
        fracture, "unit_cube_spec") else fracture.fracture(
        fracture.with_budget(fracture.cluster_spec(), 10), seed=1)
    np_ = make_chunk_wireframe(result.chunks[0])
    state = np_.getState()

    blend = state.getAttrib(ColorBlendAttrib.getClassSlot())
    assert blend is not None, "no ColorBlendAttrib - debris will not glow"
    assert blend.getMode() == ColorBlendAttrib.M_add, (
        "blend is not additive, so overlapping edges will not accumulate"
    )

    depth = state.getAttrib(DepthWriteAttrib.getClassSlot())
    assert depth is not None and depth.getMode() == DepthWriteAttrib.M_off, (
        "depth write is on, so debris glow will occlude debris glow behind it"
    )

    light = state.getAttrib(LightAttrib.getClassSlot())
    assert light is not None and light.hasAllOff(), (
        "geometry is lit; a neon wireframe must be self-coloured"
    )


def test_colour_is_banded_by_chunk_size():
    """Heavy slabs read cool, small shards read hot. Hierarchy in a collapse."""
    cold = debris_color(0.0)
    hot = debris_color(1.0)
    assert cold != hot, "every chunk is the same colour"
    assert cold == DEBRIS_COLORS[0]
    assert hot == DEBRIS_COLORS[-1]
    # Monotone-ish: red rises toward the light end, green/blue does not.
    assert hot[0] > cold[0]
    assert len({debris_color(t / 8.0) for t in range(9)}) == len(DEBRIS_COLORS)


def test_glow_is_a_halo_pass_plus_a_core_pass_not_an_overbright_colour():
    """Vertex colours are clamped, so >1 brightness is a no-op.

    Measured on this stack: the same chunk rendered at colour scale 1.0, 1.85
    and 3.0 produced byte-identical frames (mean brightness 0.030404 for all
    three). An over-bright multiplier does nothing at all. The glow has to
    come from geometry - a thick dim halo under a thin bright core, additively
    blended. This pins that structure so nobody "optimises" it back into a
    multiplier that silently does nothing.
    """
    from panda3d.core import GeomVertexReader

    result = fracture.fracture(
        fracture.with_budget(fracture.cluster_spec(), 10), seed=1
    )
    root = make_chunk_wireframe(result.chunks[0], size_blend=1.0)

    assert len(DEBRIS_GLOW_PASSES) >= 2, "a single pass cannot glow"
    thicknesses = [t for t, _ in DEBRIS_GLOW_PASSES]
    alphas = [a for _, a in DEBRIS_GLOW_PASSES]
    # Halo first: thicker and dimmer than the core it sits under.
    assert thicknesses[0] > thicknesses[-1] * 2.0, (
        f"the halo pass ({thicknesses[0]}px) is not meaningfully thicker "
        f"than the core ({thicknesses[-1]}px)"
    )
    assert alphas[0] < alphas[-1], "the halo is not dimmer than the core"
    assert alphas[-1] == pytest.approx(1.0), "the core pass is not full bright"

    # And no channel is over-bright, because that would be a lie.
    peak = 0.0
    for p in root.getChildren():
        gn = p.node()
        for i in range(gn.getNumGeoms()):
            vdata = gn.getGeom(i).getVertexData()
            if not vdata.hasColumn("color"):
                continue
            reader = GeomVertexReader(vdata, "color")
            while not reader.isAtEnd():
                c = reader.getData4f()
                peak = max(peak, float(c[0]), float(c[1]), float(c[2]))
    assert peak <= 1.0 + 1e-6, (
        f"peak vertex colour is {peak:.3f} - over-bright colour is clamped by "
        f"Panda, so this is a no-op pretending to be a glow"
    )


# ============================================================ attachment
def test_visuals_are_parented_to_the_bodies_so_bullet_drives_them():
    w, field = make_field()
    result = fracture.fracture(
        fracture.with_budget(fracture.cluster_spec(), 30), seed=4
    )
    event = field.shatter(result, impact_point=(0.0, 0.0, 3.0))

    n = attach_debris_visuals(event.bodies)
    assert n == event.spawned

    for body in event.bodies:
        assert body.np.getNumChildren() == 1, (
            f"{body.name} has {body.np.getNumChildren()} visuals, expected 1"
        )
        wire = body.np.getChild(0)
        assert wire.getParent() == body.np

    # Now step: the visual's WORLD transform must move, while its LOCAL
    # transform stays untouched. That is the proof that Bullet's body
    # transform is carrying it and no Python sync is needed.
    body = event.bodies[0]
    wire = body.np.getChild(0)
    local_before = wire.getTransform()
    world_before = wire.getPos(w.root)
    for _ in range(int(1.0 / DT)):
        w.step_fixed(1)
        field.update(DT)
    assert wire.getTransform() == local_before, (
        "the visual's local transform changed - something is syncing per frame"
    )
    assert (wire.getPos(w.root) - world_before).length() > 0.05, (
        "the visual did not move with its body"
    )


def test_structure_silhouette_covers_every_source_block():
    d = structures.generate("arch", (0.0, 20.0, 0.0), seed=9)
    np_ = make_structure_wireframe(d)
    assert not np_.isEmpty()
    lo, hi = np_.getTightBounds(np_.getParent() or np_)
    slo, shi = d.spec.bounds
    for axis in range(3):
        assert float(hi[axis]) - float(lo[axis]) == pytest.approx(
            shi[axis] - slo[axis], abs=1e-3
        ), f"silhouette does not span the structure on axis {axis}"
    assert tuple(float(v) for v in np_.getPos()) == pytest.approx(d.origin)


# ============================================================ real pixels
@pytest.mark.render
class TestOffscreenDebrisPixels:
    """Render real frames offscreen and assert on the pixels.

    Skipped where there is no GL pipe, so the core suite still runs anywhere.
    """

    @staticmethod
    def _boot():
        from game.app import TumbleApp

        try:
            app = TumbleApp(offscreen=True)
        except Exception as exc:                    # pragma: no cover
            pytest.skip(f"no offscreen GL pipe available: {exc}")
        if app.win is None:                         # pragma: no cover
            app.destroy()
            pytest.skip("offscreen buffer could not be created")
        return app

    @staticmethod
    def _sample(app):
        """Mean brightness and lit-pixel count of the current frame."""
        from panda3d.core import PNMImage

        for _ in range(3):
            app.graphicsEngine.renderFrame()
        img = PNMImage()
        if not app.win.getScreenshot(img):          # pragma: no cover
            pytest.skip("could not read back the offscreen buffer")
        w, h = img.getXSize(), img.getYSize()
        total = 0.0
        lit = 0
        n = 0
        for y in range(0, h, 2):
            for x in range(0, w, 2):
                px = img.getXel(x, y)
                v = (px[0] + px[1] + px[2]) / 3.0
                total += v
                if v > 0.30:
                    lit += 1
                n += 1
        return total / n, lit, n

    def test_debris_puts_glowing_pixels_on_the_frame(self):
        """A frame of fresh debris must be measurably brighter than the same
        view with the debris removed. This is the mechanical stand-in for
        looking at the screen."""
        app = self._boot()
        try:
            d = min(app.destructibles, key=lambda x: x.world_bounds()[0][1])
            lo, hi = d.world_bounds()
            cx = (lo[0] + hi[0]) * 0.5

            # Stand back and look straight at the structure.
            app.player.np.setPos(cx, lo[1] - 30.0, 2.0)
            app.player.heading = 0.0
            app.player.np.setH(0.0)
            app.player.pitch = 0.0
            app.player.sync_camera()

            intact_mean, intact_lit, n = self._sample(app)

            event = app.demolish(d)
            assert event is not None and event.spawned > 50
            # Let it break apart so debris fills the view.
            for _ in range(40):
                app.step_frame(1.0 / 60.0)
            app.player.sync_camera()
            debris_mean, debris_lit, _ = self._sample(app)

            # Now hide only the debris and re-measure the identical view.
            app.debris.root.hide()
            hidden_mean, hidden_lit, _ = self._sample(app)
            app.debris.root.show()
            shown_mean, shown_lit, _ = self._sample(app)

            assert debris_lit > 0, "no lit pixels at all with debris in view"
            assert shown_lit > hidden_lit, (
                f"hiding the debris did not darken the frame "
                f"({shown_lit} lit vs {hidden_lit} hidden of {n} samples) - "
                f"the wireframes are not rendering"
            )
            assert shown_mean > hidden_mean, (
                f"debris contributed no brightness "
                f"({shown_mean:.5f} vs {hidden_mean:.5f})"
            )
            assert intact_lit > 0, "the intact structure rendered nothing"
        finally:
            app.destroy()

    def test_every_live_chunk_gets_its_wireframe_on_demolition(self):
        app = self._boot()
        try:
            d = min(app.destructibles, key=lambda x: x.world_bounds()[0][1])
            event = app.demolish(d)
            assert event is not None
            for body in event.bodies:
                assert body.np.getNumChildren() == 1, (
                    f"{body.name} was spawned with no wireframe"
                )
            # And the intact silhouette was taken down.
            assert d.name not in app._structure_visuals
        finally:
            app.destroy()

    def test_the_frame_renders_for_many_frames_after_a_demolition(self):
        """Render the collapse for real - no crash, no leak, no stall."""
        app = self._boot()
        try:
            app.demolish_nearest()
            for _ in range(120):
                app.step_frame(1.0 / 60.0)
                app.graphicsEngine.renderFrame()
            snap = app.debris.snapshot()
            assert snap["live"] + snap["frozen"] > 0
            assert not any(
                math.isnan(float(v)) for v in (snap["max_speed"],
                                               snap["max_spin"],
                                               snap["min_z"])
            ), "NaN leaked into the debris state"
        finally:
            app.destroy()

    def test_the_halo_pass_measurably_brightens_the_frame(self, capsys):
        """The glow claim, measured in pixels rather than asserted.

        Isolation matters here. The course already fills roughly half the
        frame with lit pixels, so a raw lit-pixel count is dominated by the
        background and a real glow difference vanishes into it (measured:
        39271 core-only vs 39352 halo+core out of 76800 - indistinguishable).
        So this differences each frame against an identical frame with the
        wireframe removed, and counts only the pixels the wireframe itself
        brightened.

        This is what the over-bright colour multiplier could never do: Panda
        clamps vertex colour, so scale 1.0 / 1.85 / 3.0 rendered byte-identical
        frames. The halo pass is the only thing that actually glows, and this
        is the measurement that proves it.
        """
        from panda3d.core import PNMImage

        app = self._boot()
        try:
            app.setBackgroundColor(0.0, 0.0, 0.0, 1.0)
            result = fracture.fracture(
                fracture.with_budget(fracture.cluster_spec(), 40), seed=3
            )
            chunk = result.chunks[0]

            def grab():
                for _ in range(3):
                    app.graphicsEngine.renderFrame()
                img = PNMImage()
                if not app.win.getScreenshot(img):     # pragma: no cover
                    pytest.skip("could not read back the offscreen buffer")
                out = []
                for y in range(img.getYSize()):
                    for x in range(img.getXSize()):
                        px = img.getXel(x, y)
                        out.append((px[0] + px[1] + px[2]) / 3.0)
                return out

            # Park the chunk right in front of the camera.
            def place(passes):
                wire = make_chunk_wireframe(chunk, size_blend=1.0,
                                            passes=passes, name="probe-wire")
                wire.reparentTo(app.camera)
                wire.setPos(0.0, 6.0, 0.0)
                return wire

            baseline = grab()

            def contribution(passes):
                wire = place(passes)
                frame = grab()
                wire.removeNode()
                brightened = 0
                added = 0.0
                for a, b in zip(baseline, frame):
                    d = b - a
                    if d > 0.05:
                        brightened += 1
                    if d > 0.0:
                        added += d
                return brightened, added

            core_px, core_sum = contribution(DEBRIS_GLOW_PASSES[-1:])
            glow_px, glow_sum = contribution(DEBRIS_GLOW_PASSES)

            with capsys.disabled():
                print()
                print(f"  [1A glow] core only : {core_px} px brightened, "
                      f"total added light {core_sum:.2f}")
                print(f"  [1A glow] halo+core : {glow_px} px brightened, "
                      f"total added light {glow_sum:.2f}")
                print(f"  [1A glow] halo gain : "
                      f"{glow_px / max(core_px, 1):.2f}x pixels, "
                      f"{glow_sum / max(core_sum, 1e-9):.2f}x light")

            assert core_px > 50, (
                f"the core pass only brightened {core_px} pixels - the "
                f"wireframe is barely rendering at all"
            )
            assert glow_px > core_px * 1.3, (
                f"the halo pass added almost nothing: {core_px} -> {glow_px} "
                f"brightened pixels. It is not producing a glow."
            )
            assert glow_sum > core_sum * 1.3, (
                f"halo+core added {glow_sum:.2f} of light vs core-only "
                f"{core_sum:.2f} - no measurable bloom"
            )
        finally:
            app.destroy()
