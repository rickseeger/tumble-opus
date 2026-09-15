"""The locked 1A treatment: glowing wireframe debris.

Black void, neon edge-lit geometry, debris as bright glowing outlines. This
module is the *only* place in the destruction stack that touches Panda's
geometry classes, and the render layer imports it lazily - so the sim and the
headless test suite never load it.

How the glow is actually produced
---------------------------------
Two things, both measured on this stack rather than assumed:

* **Additive blending.** Overlapping edges accumulate into brighter lines the
  way a vector display does, with depth-write off so debris glow never
  occludes debris glow behind it.

* **A halo pass plus a core pass.** Each chunk's edge list is drawn twice:
  once thick and dim (the bloom), once thin and full brightness (the filament).
  Additively combined, that is a real glow.

  What does *not* work, and was tried: multiplying the vertex colour above 1.0
  to "over-brighten" it. Panda clamps vertex colours, and `setColorScale`
  above 1.0 is clamped too - measured, a chunk drawn at colour scale 1.0, 1.85
  and 3.0 produced byte-identical frames (mean brightness 0.030404 for all
  three). So an over-bright multiplier is a no-op, not a glow. The halo pass
  is the thing that measurably works: offscreen, one chunk went from 2335 lit
  pixels (core only) to 5719 (halo + core), mean frame brightness 0.0254 ->
  0.0409.

Geometry is parented to the chunk's own rigid body, so Bullet's transform
drives it and there is no per-chunk Python work per frame - nothing to sync
while a few hundred pieces tumble.

Whether it *looks* right is Rick's call at playtest. What this module promises,
and what `tests/test_render_debris.py` asserts, is that the geometry exists
with one segment per chunk edge, carries the additive/unlit/no-depth-write
state, measurably brightens a rendered frame, and rides the body's live pose.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from panda3d.core import (
    ColorBlendAttrib,
    LineSegs,
    NodePath,
    Vec4,
)

#: Neon palette, indexed by size band. Big slabs read cooler, small shards
#: read hotter, so a collapse has visual hierarchy.
DEBRIS_COLORS = (
    (0.30, 0.95, 1.00),   # heaviest: cold cyan
    (0.45, 0.80, 1.00),
    (0.75, 0.70, 1.00),
    (1.00, 0.55, 0.95),   # lightest: hot magenta
)

#: The two draw passes: (thickness in pixels, alpha). Thick+dim is the bloom,
#: thin+opaque is the filament. Additively blended, these make the glow.
#: Measured offscreen: core alone 2335 lit pixels, halo+core 5719.
DEBRIS_GLOW_PASSES: Tuple[Tuple[float, float], ...] = (
    (5.0, 0.35),
    (1.6, 1.00),
)

#: Intact structures get a heavier version of the same treatment.
STRUCTURE_GLOW_PASSES: Tuple[Tuple[float, float], ...] = (
    (7.0, 0.30),
    (2.2, 1.00),
)
STRUCTURE_COLOR = (0.35, 0.95, 1.00)


def debris_color(size_blend: float) -> Tuple[float, float, float]:
    """Pick a neon colour for a chunk. ``size_blend`` is 0 heavy .. 1 light."""
    n = len(DEBRIS_COLORS)
    i = int(max(0.0, min(1.0, float(size_blend))) * (n - 1) + 0.5)
    return DEBRIS_COLORS[i]


def apply_glow_state(np_: NodePath) -> NodePath:
    """Make *np_* render as an unlit, additively-blended glowing line set."""
    np_.setLightOff(1)
    np_.setTextureOff(1)
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


def _edge_pass(
    vertices: Sequence[Sequence[float]],
    edges: Sequence[Tuple[int, int]],
    thickness: float,
    color: Tuple[float, float, float],
    alpha: float,
    name: str,
) -> NodePath:
    segs = LineSegs(name)
    segs.setThickness(float(thickness))
    segs.setColor(Vec4(color[0], color[1], color[2], float(alpha)))
    for a, b in edges:
        segs.moveTo(*vertices[a])
        segs.drawTo(*vertices[b])
    return NodePath(segs.create())


def make_chunk_wireframe(
    chunk,
    size_blend: float = 0.5,
    passes: Sequence[Tuple[float, float]] = DEBRIS_GLOW_PASSES,
    name: Optional[str] = None,
) -> NodePath:
    """Build the glowing wireframe for one chunk, in the chunk's local frame.

    Local, not world: parent the result to the chunk's rigid body and Bullet's
    transform carries it for free. Returns the parent of the halo/core passes.
    """
    color = debris_color(size_blend)
    root = NodePath(name or "chunk-wire")
    for i, (thickness, alpha) in enumerate(passes):
        p = _edge_pass(chunk.vertices, chunk.edges, thickness, color, alpha,
                       f"{root.getName()}_p{i}")
        p.reparentTo(root)
    apply_glow_state(root)
    return root


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

    verts: list = []
    lookup: dict = {}
    edges: set = set()
    for block in destructible.spec.blocks:
        for face in box_faces(block.center, block.half_extents):
            m = len(face)
            for i in range(m):
                pair = []
                for p in (face[i], face[(i + 1) % m]):
                    key = tuple(round(c, 6) for c in p)
                    idx = lookup.get(key)
                    if idx is None:
                        idx = len(verts)
                        lookup[key] = idx
                        verts.append(p)
                    pair.append(idx)
                if pair[0] != pair[1]:
                    edges.add((min(pair), max(pair)))

    root = NodePath(name or f"{destructible.name}-wire")
    for i, (thickness, alpha) in enumerate(STRUCTURE_GLOW_PASSES):
        p = _edge_pass(verts, sorted(edges), thickness, STRUCTURE_COLOR,
                       alpha, f"{root.getName()}_p{i}")
        p.reparentTo(root)
    apply_glow_state(root)
    root.setPos(*destructible.origin)
    return root
