"""The locked 1A treatment: glowing wireframe debris.

Black void, neon edge-lit geometry, debris as bright glowing outlines. This
module is the *only* place in the destruction stack that touches Panda's
geometry classes, and it is imported lazily by the render layer - so the sim
and the whole test suite never load it.

How it is drawn
---------------
Each chunk's edge list (already emitted by :mod:`game.fracture`) becomes a
``LineSegs`` node parented to the chunk's own rigid-body NodePath. That is the
important trick: because the geometry is a *child of the body*, Bullet's own
transform drives it and there is nothing to sync per frame - no per-chunk
Python work at all while 260 pieces tumble. Glow comes from an additive blend
with depth-write off and lighting off, so overlapping edges pile up into
brighter lines the way a vector display does.

Whether it *looks* right is Rick's call at playtest. What this module can
promise, and what :mod:`tests.test_render_debris` asserts, is that the
geometry exists, has the right edge count, carries the additive/unlit render
state, and rides the body's live pose.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from panda3d.core import (
    ColorBlendAttrib,
    LineSegs,
    NodePath,
    Vec4,
)

#: Neon palette, indexed by size band. Big slabs read cooler and dimmer, small
#: shards read hotter and brighter, so a collapse has visual hierarchy.
DEBRIS_COLORS = (
    (0.30, 0.95, 1.00),   # heaviest: cold cyan
    (0.45, 0.80, 1.00),
    (0.75, 0.70, 1.00),
    (1.00, 0.55, 0.95),   # lightest: hot magenta
)

#: Edge thickness in pixels. Thin enough to read as a vector line.
DEBRIS_LINE_THICKNESS = 1.6
#: Extra brightness multiplier. >1 saturates under additive blending, which is
#: what makes the lines glow rather than merely being coloured.
DEBRIS_GLOW = 1.85


def debris_color(size_blend: float) -> Tuple[float, float, float]:
    """Pick a neon colour for a chunk. ``size_blend`` is 0 heavy .. 1 light."""
    n = len(DEBRIS_COLORS)
    i = int(max(0.0, min(1.0, float(size_blend))) * (n - 1) + 0.5)
    return DEBRIS_COLORS[i]


def apply_glow_state(np_: NodePath) -> NodePath:
    """Make *np_* render as an unlit, additively-blended glowing line set."""
    np_.setLightOff(1)
    np_.setTextureOff(1)
    np_.setTransparency(False)
    # Additive: overlapping edges accumulate into brighter lines. Depth write
    # off so debris glow never occludes debris glow behind it.
    np_.setAttrib(ColorBlendAttrib.make(
        ColorBlendAttrib.M_add,
        ColorBlendAttrib.O_incoming_alpha,
        ColorBlendAttrib.O_one,
    ))
    np_.setDepthWrite(False)
    np_.setBin("fixed", 20)
    return np_


def make_chunk_wireframe(
    chunk,
    size_blend: float = 0.5,
    thickness: float = DEBRIS_LINE_THICKNESS,
    glow: float = DEBRIS_GLOW,
    name: Optional[str] = None,
) -> NodePath:
    """Build the glowing wireframe for one chunk, in the chunk's local frame.

    Local, not world: parent the result to the chunk's rigid body and Bullet's
    transform carries it for free.
    """
    r, g, b = debris_color(size_blend)
    segs = LineSegs(name or "chunk-wire")
    segs.setThickness(float(thickness))
    segs.setColor(Vec4(r * glow, g * glow, b * glow, 1.0))
    verts = chunk.vertices
    for a, bi in chunk.edges:
        segs.moveTo(*verts[a])
        segs.drawTo(*verts[bi])
    np_ = NodePath(segs.create())
    apply_glow_state(np_)
    return np_


def attach_debris_visuals(bodies: Sequence) -> int:
    """Give every body in *bodies* its glowing wireframe. Returns the count.

    Parented to the body's own NodePath, so there is no per-frame sync.
    """
    n = 0
    for body in bodies:
        wire = make_chunk_wireframe(
            body.chunk, body.size_blend, name=f"{body.name}__wire"
        )
        wire.reparentTo(body.np)
        n += 1
    return n


def make_structure_wireframe(destructible, name: Optional[str] = None) -> NodePath:
    """The intact structure's neon silhouette: its source blocks' edges.

    Drawn from the spec's blocks rather than the chunk set, because an intact
    building should read as a building, not as a pile of seams.
    """
    from .fracture import box_faces

    segs = LineSegs(name or f"{destructible.name}-wire")
    segs.setThickness(2.2)
    segs.setColor(Vec4(0.35 * 1.6, 0.95 * 1.6, 1.0 * 1.6, 1.0))
    for block in destructible.spec.blocks:
        drawn = set()
        for face in box_faces(block.center, block.half_extents):
            m = len(face)
            for i in range(m):
                a, b = face[i], face[(i + 1) % m]
                key = tuple(sorted((tuple(round(c, 6) for c in a),
                                    tuple(round(c, 6) for c in b))))
                if key in drawn:
                    continue
                drawn.add(key)
                segs.moveTo(*a)
                segs.drawTo(*b)
    np_ = NodePath(segs.create())
    apply_glow_state(np_)
    np_.setPos(*destructible.origin)
    return np_
