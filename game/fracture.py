"""Fracture generation for Tumble - pure data, no engine imports.

This module turns a *structure spec* (a handful of axis-aligned blocks that
describe a tower, a slab building, an arch or a cluster of debris) plus an
integer seed into a deterministic list of convex chunk descriptors that tile
the structure's volume.

Design notes
------------
* **No Panda3D, no Bullet, no rendering, no audio.** Standard library only.
  The runtime layer (node 10) consumes :class:`Chunk` records and builds
  whatever convex shapes / wireframe geometry it needs.
* **Real convex polyhedra, not a voxel grid.** Each block is carved by
  recursive half-space splitting with *jittered, oblique* planes, so chunks
  come out as boxes, wedges and shards with genuinely varied aspect ratios.
  Because every cut is a plane through a convex piece, the results are exactly
  space-filling: no overlap, and volume is conserved to floating-point noise.
* **Explicit vertices AND edges.** The locked art direction is glowing
  wireframe, so every chunk exposes ``vertices``, ``edges`` and ``faces`` -
  not just a bounding box.
* **Deterministic.** Randomness comes from :class:`random.Random` seeded by a
  blake2b digest of ``(spec.name, seed, block index)``, so results do not
  depend on ``PYTHONHASHSEED`` or dict ordering.
* **Budgeted and fast.** ``spec.max_chunks`` is a hard ceiling; generation is
  intended to run once at load time, not at shatter time.

Coordinate convention: everything is in the structure's *local* space, Z up,
with the structure's base sitting on ``z = 0``. ``Chunk.center`` is the chunk's
centroid in that space; ``Chunk.vertices`` are relative to that centroid.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Face = List[Vec3]

#: Absolute tolerance used when classifying a point against a split plane.
PLANE_EPS = 1e-9
#: Quantisation used to fuse coincident vertices when emitting a chunk.
WELD_DECIMALS = 9
#: Floats are rounded to this many decimals when serialising, so that the
#: "byte identical" determinism check is not hostage to repr noise.
SERIALISE_DECIMALS = 12


# --------------------------------------------------------------- vector maths
def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _length(a: Vec3) -> float:
    return math.sqrt(_dot(a, a))


def _unit(a: Vec3) -> Vec3:
    n = _length(a)
    if n <= 1e-15:
        return (0.0, 0.0, 1.0)
    return (a[0] / n, a[1] / n, a[2] / n)


# ------------------------------------------------------- polyhedron internals
# A polyhedron is a list of faces; a face is a list of points wound
# counter-clockwise when viewed from *outside* the solid.

def _newell_normal(face: Sequence[Vec3]) -> Vec3:
    nx = ny = nz = 0.0
    m = len(face)
    for i in range(m):
        a = face[i]
        b = face[(i + 1) % m]
        nx += (a[1] - b[1]) * (a[2] + b[2])
        ny += (a[2] - b[2]) * (a[0] + b[0])
        nz += (a[0] - b[0]) * (a[1] + b[1])
    return (nx, ny, nz)


def _face_area(face: Sequence[Vec3]) -> float:
    return 0.5 * _length(_newell_normal(face))


def _oriented(face: Face, outward: Vec3) -> Face:
    """Return ``face`` wound so that its Newell normal points along *outward*."""
    if _dot(_newell_normal(face), outward) < 0.0:
        return list(reversed(face))
    return face


def box_faces(center: Vec3, half: Vec3) -> List[Face]:
    """The six correctly-wound faces of an axis-aligned box."""
    cx, cy, cz = center
    hx, hy, hz = half

    def c(sx: int, sy: int, sz: int) -> Vec3:
        return (cx + sx * hx, cy + sy * hy, cz + sz * hz)

    quads = [
        ([c(+1, -1, -1), c(+1, +1, -1), c(+1, +1, +1), c(+1, -1, +1)], (1.0, 0.0, 0.0)),
        ([c(-1, -1, -1), c(-1, +1, -1), c(-1, +1, +1), c(-1, -1, +1)], (-1.0, 0.0, 0.0)),
        ([c(-1, +1, -1), c(+1, +1, -1), c(+1, +1, +1), c(-1, +1, +1)], (0.0, 1.0, 0.0)),
        ([c(-1, -1, -1), c(+1, -1, -1), c(+1, -1, +1), c(-1, -1, +1)], (0.0, -1.0, 0.0)),
        ([c(-1, -1, +1), c(+1, -1, +1), c(+1, +1, +1), c(-1, +1, +1)], (0.0, 0.0, 1.0)),
        ([c(-1, -1, -1), c(+1, -1, -1), c(+1, +1, -1), c(-1, +1, -1)], (0.0, 0.0, -1.0)),
    ]
    return [_oriented(q, n) for q, n in quads]


def poly_volume(faces: Sequence[Sequence[Vec3]]) -> float:
    """Signed volume via the divergence theorem (positive for outward winding)."""
    total = 0.0
    for f in faces:
        a = f[0]
        for i in range(1, len(f) - 1):
            b = f[i]
            c = f[i + 1]
            total += (
                a[0] * (b[1] * c[2] - b[2] * c[1])
                - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0])
            )
    return total / 6.0


def poly_centroid_volume(faces: Sequence[Sequence[Vec3]]) -> Tuple[Vec3, float]:
    """Centroid and volume of a closed, outward-wound polyhedron."""
    vol6 = 0.0
    cx = cy = cz = 0.0
    for f in faces:
        a = f[0]
        for i in range(1, len(f) - 1):
            b = f[i]
            c = f[i + 1]
            w = (
                a[0] * (b[1] * c[2] - b[2] * c[1])
                - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0])
            )
            vol6 += w
            cx += w * (a[0] + b[0] + c[0]) * 0.25
            cy += w * (a[1] + b[1] + c[1]) * 0.25
            cz += w * (a[2] + b[2] + c[2]) * 0.25
    if abs(vol6) < 1e-18:
        pts = [p for f in faces for p in f]
        n = float(len(pts)) or 1.0
        return (
            (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n,
             sum(p[2] for p in pts) / n),
            0.0,
        )
    return ((cx / vol6, cy / vol6, cz / vol6), vol6 / 6.0)


def poly_bounds(faces: Sequence[Sequence[Vec3]]) -> Tuple[Vec3, Vec3]:
    xs = [p[0] for f in faces for p in f]
    ys = [p[1] for f in faces for p in f]
    zs = [p[2] for f in faces for p in f]
    return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))


def _dedup_ring(pts: Face) -> Face:
    """Drop consecutive duplicates (including the wrap-around pair)."""
    out: Face = []
    for p in pts:
        if not out or _length(_sub(p, out[-1])) > 1e-12:
            out.append(p)
    while len(out) >= 2 and _length(_sub(out[0], out[-1])) <= 1e-12:
        out.pop()
    return out


def _cap_polygon(pts: Sequence[Vec3], normal: Vec3) -> Optional[Face]:
    """Order the cut points into a convex polygon whose normal is *normal*."""
    seen: Dict[Tuple[int, int, int], Vec3] = {}
    for p in pts:
        key = (round(p[0], WELD_DECIMALS), round(p[1], WELD_DECIMALS),
               round(p[2], WELD_DECIMALS))
        ikey = (int(key[0] * 1e9), int(key[1] * 1e9), int(key[2] * 1e9))
        if ikey not in seen:
            seen[ikey] = p
    uniq = list(seen.values())
    if len(uniq) < 3:
        return None
    n = float(len(uniq))
    c = (sum(p[0] for p in uniq) / n, sum(p[1] for p in uniq) / n,
         sum(p[2] for p in uniq) / n)
    # Build an in-plane right-handed basis (u, v) with u x v == normal.
    helper = (0.0, 0.0, 1.0) if abs(normal[2]) < 0.9 else (1.0, 0.0, 0.0)
    u = _unit(_cross(helper, normal))
    v = _cross(normal, u)
    ordered = sorted(
        uniq,
        key=lambda p: math.atan2(_dot(_sub(p, c), v), _dot(_sub(p, c), u)),
    )
    ordered = _dedup_ring(ordered)
    if len(ordered) < 3 or _face_area(ordered) <= 1e-12:
        return None
    return _oriented(ordered, normal)


def clip_halfspace(
    faces: Sequence[Face], normal: Vec3, offset: float, eps: float = PLANE_EPS
) -> Optional[List[Face]]:
    """Clip a convex polyhedron to ``dot(normal, x) <= offset``.

    Returns the new face list, the original list when nothing is cut away, or
    ``None`` when the whole solid lies outside the half-space.
    """
    nx, ny, nz = normal
    new_faces: List[Face] = []
    cap_pts: List[Vec3] = []
    any_out = False
    any_in = False

    for face in faces:
        m = len(face)
        ds: List[float] = []
        for p in face:
            t = nx * p[0] + ny * p[1] + nz * p[2] - offset
            if -eps < t < eps:
                t = 0.0
            ds.append(t)
            if t > 0.0:
                any_out = True
            elif t < 0.0:
                any_in = True
        out: Face = []
        for i in range(m):
            a = face[i]
            da = ds[i]
            j = (i + 1) % m
            b = face[j]
            db = ds[j]
            if da <= 0.0:
                out.append(a)
            if (da < 0.0 < db) or (db < 0.0 < da):
                t = da / (da - db)
                out.append((a[0] + (b[0] - a[0]) * t,
                            a[1] + (b[1] - a[1]) * t,
                            a[2] + (b[2] - a[2]) * t))
        out = _dedup_ring(out)
        if len(out) >= 3 and _face_area(out) > 1e-13:
            new_faces.append(out)
            for p in out:
                if abs(nx * p[0] + ny * p[1] + nz * p[2] - offset) <= eps * 16.0:
                    cap_pts.append(p)

    if not any_out:
        return [list(f) for f in faces]
    if not any_in:
        return None

    cap = _cap_polygon(cap_pts, normal)
    if cap is not None:
        new_faces.append(cap)
    if len(new_faces) < 4:
        return None
    return new_faces


# ------------------------------------------------------------------ spec data
@dataclass(frozen=True)
class Block:
    """One axis-aligned solid piece of a structure."""

    name: str
    center: Vec3
    half_extents: Vec3

    @property
    def volume(self) -> float:
        hx, hy, hz = self.half_extents
        return 8.0 * hx * hy * hz

    @property
    def bounds(self) -> Tuple[Vec3, Vec3]:
        c, h = self.center, self.half_extents
        return ((c[0] - h[0], c[1] - h[1], c[2] - h[2]),
                (c[0] + h[0], c[1] + h[1], c[2] + h[2]))

    def overlap_volume(self, other: "Block") -> float:
        """Volume of the intersection of this block with *other* (0 if none).

        Two source blocks that interpenetrate would double-count their shared
        region in the structure volume *and* hand the same space to two
        different chunks, so this is how the spec validator finds bad specs.
        """
        alo, ahi = self.bounds
        blo, bhi = other.bounds
        v = 1.0
        for i in range(3):
            d = min(ahi[i], bhi[i]) - max(alo[i], blo[i])
            if d <= 0.0:
                return 0.0
            v *= d
        return v


@dataclass(frozen=True)
class StructureSpec:
    """Everything the fracturer needs to know about one destructible object."""

    name: str
    kind: str
    blocks: Tuple[Block, ...]
    #: kg/m^3. Concrete-ish by default; mass = volume * density.
    density: float = 2400.0
    #: Hard ceiling on the number of chunks produced. Never exceeded.
    max_chunks: int = 320
    #: Chunks smaller than this are never created. ``None`` derives a sane
    #: floor from the structure volume and the chunk budget.
    min_chunk_volume: Optional[float] = None
    #: Where the blast comes from; drives each chunk's launch hint.
    #: ``None`` means "centre of the structure's footprint, at its base".
    fracture_origin: Optional[Vec3] = None
    #: How far split planes may tilt away from an axis (0 = axis aligned).
    plane_jitter: float = 0.40
    #: Probability a cut is taken across the piece's *shortest* axis, which
    #: is what produces long thin slabs and shards rather than cubes.
    shard_bias: float = 0.38
    #: Probability a cut lands well off-centre, which spreads the size range.
    uneven_bias: float = 0.45

    @property
    def volume(self) -> float:
        return sum(b.volume for b in self.blocks)

    @property
    def bounds(self) -> Tuple[Vec3, Vec3]:
        lo = [math.inf] * 3
        hi = [-math.inf] * 3
        for b in self.blocks:
            blo, bhi = b.bounds
            for i in range(3):
                lo[i] = min(lo[i], blo[i])
                hi[i] = max(hi[i], bhi[i])
        return (tuple(lo), tuple(hi))  # type: ignore[return-value]

    def resolved_fracture_origin(self) -> Vec3:
        if self.fracture_origin is not None:
            return self.fracture_origin
        lo, hi = self.bounds
        return ((lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, lo[2])

    def resolved_min_chunk_volume(self) -> float:
        if self.min_chunk_volume is not None:
            return self.min_chunk_volume
        return self.volume / (max(self.max_chunks, 1) * 30.0)

    def overlapping_block_pairs(self, eps: float = 1e-9) -> List[Tuple[str, str, float]]:
        """Source blocks that interpenetrate, as ``(name_a, name_b, volume)``.

        A well-formed spec has none: blocks may touch face-to-face, but any
        shared *volume* means :attr:`volume` double-counts and the chunks cut
        from those blocks will occupy the same space.
        """
        bad: List[Tuple[str, str, float]] = []
        for i in range(len(self.blocks)):
            for j in range(i + 1, len(self.blocks)):
                a, b = self.blocks[i], self.blocks[j]
                v = a.overlap_volume(b)
                if v > eps:
                    bad.append((a.name, b.name, v))
        return bad

    def validate(self) -> None:
        """Raise :class:`ValueError` if this spec cannot be fractured cleanly."""
        if not self.blocks:
            raise ValueError(f"spec {self.name!r} has no blocks")
        for b in self.blocks:
            if min(b.half_extents) <= 0.0:
                raise ValueError(
                    f"spec {self.name!r} block {b.name!r} has a non-positive "
                    f"half extent: {b.half_extents}"
                )
        names = [b.name for b in self.blocks]
        if len(set(names)) != len(names):
            raise ValueError(f"spec {self.name!r} has duplicate block names: {names}")
        bad = self.overlapping_block_pairs()
        if bad:
            detail = ", ".join(f"{a}<->{b} ({v:.4f} m^3)" for a, b, v in bad)
            raise ValueError(
                f"spec {self.name!r} has interpenetrating source blocks: {detail}. "
                "Overlapping blocks double-count structure volume and produce "
                "chunks that occupy the same space."
            )
        if self.max_chunks < len(self.blocks):
            raise ValueError(
                f"max_chunks={self.max_chunks} is below the block count "
                f"{len(self.blocks)} for spec {self.name!r}"
            )


# ------------------------------------------------------------------- archetypes
def tower_spec(max_chunks: int = 380) -> StructureSpec:
    """A huge slender tower: small footprint, very tall."""
    return StructureSpec(
        name="tall_tower",
        kind="tower",
        blocks=(Block("shaft", (0.0, 0.0, 30.0), (4.0, 4.0, 30.0)),),
        max_chunks=max_chunks,
    )


def slab_spec(max_chunks: int = 320) -> StructureSpec:
    """A wide, squat slab building: big footprint, low height."""
    return StructureSpec(
        name="wide_slab",
        kind="slab",
        blocks=(Block("slab", (0.0, 0.0, 4.5), (18.0, 11.0, 4.5)),),
        max_chunks=max_chunks,
    )


def arch_spec(max_chunks: int = 260) -> StructureSpec:
    """Two legs and a lintel - a structure with a hole in it."""
    return StructureSpec(
        name="arch",
        kind="arch",
        blocks=(
            Block("leg_left", (-10.0, 0.0, 9.0), (2.5, 3.0, 9.0)),
            Block("leg_right", (10.0, 0.0, 9.0), (2.5, 3.0, 9.0)),
            Block("lintel", (0.0, 0.0, 20.5), (12.5, 3.0, 2.5)),
        ),
        max_chunks=max_chunks,
    )


def cluster_spec(max_chunks: int = 240) -> StructureSpec:
    """A scattered cluster of differently sized blocks."""
    return StructureSpec(
        name="block_cluster",
        kind="cluster",
        blocks=(
            Block("c0", (-7.0, -5.0, 3.0), (3.0, 3.0, 3.0)),
            Block("c1", (6.0, -6.0, 2.0), (4.5, 2.0, 2.0)),
            Block("c2", (0.0, 4.0, 5.0), (2.0, 5.0, 5.0)),
            Block("c3", (-9.0, 7.0, 1.5), (1.5, 1.5, 1.5)),
            Block("c4", (9.0, 6.0, 4.0), (2.5, 4.0, 4.0)),
            Block("c5", (1.0, -10.5, 1.0), (6.0, 1.5, 1.0)),
        ),
        max_chunks=max_chunks,
    )


#: Every archetype the library ships with, by short name.
ARCHETYPES = {
    "tower": tower_spec,
    "slab": slab_spec,
    "arch": arch_spec,
    "cluster": cluster_spec,
}


def default_specs() -> List[StructureSpec]:
    """One spec per archetype, in a stable order."""
    return [ARCHETYPES[k]() for k in ("tower", "slab", "arch", "cluster")]


# ----------------------------------------------------------------- chunk data
@dataclass(frozen=True)
class Chunk:
    """One convex piece of a fractured structure. Pure data.

    ``vertices``/``edges``/``faces`` are expressed in the chunk's own local
    frame, centred on its centroid, so a physics body can be spawned at
    ``center`` with ``orientation`` and the same mesh reused for the glowing
    wireframe. ``orientation`` is a unit quaternion ``(w, x, y, z)``; the
    generator emits chunks already aligned to the structure frame, so it is the
    identity here - the *shape* carries the asymmetry, and the runtime is free
    to overwrite the orientation once the body starts tumbling.
    """

    index: int
    source_block: str
    center: Vec3
    orientation: Tuple[float, float, float, float]
    vertices: Tuple[Vec3, ...]
    edges: Tuple[Tuple[int, int], ...]
    faces: Tuple[Tuple[int, ...], ...]
    half_extents: Vec3
    volume: float
    mass: float
    launch_dir: Vec3

    # -- derived -----------------------------------------------------------
    @property
    def extents(self) -> Vec3:
        """Full width/depth/height of the chunk's local bounding box."""
        h = self.half_extents
        return (h[0] * 2.0, h[1] * 2.0, h[2] * 2.0)

    @property
    def aspect_ratio(self) -> float:
        """Longest bounding-box side over the shortest (1.0 == cube)."""
        e = sorted(self.extents)
        if e[0] <= 1e-12:
            return math.inf
        return e[2] / e[0]

    def world_vertices(self) -> Tuple[Vec3, ...]:
        """Local vertices rotated by ``orientation`` and moved to ``center``."""
        w, x, y, z = self.orientation
        out = []
        for v in self.vertices:
            # Standard quaternion rotation, then translate.
            t = _scale(_cross((x, y, z), v), 2.0)
            r = _add(v, _add(_scale(t, w), _cross((x, y, z), t)))
            out.append(_add(r, self.center))
        return tuple(out)

    def world_faces(self) -> List[List[Vec3]]:
        wv = self.world_vertices()
        return [[wv[i] for i in f] for f in self.faces]

    def contains_point(self, p: Vec3, eps: float = 1e-7) -> bool:
        """True if *p* is inside (or within *eps* of) this convex chunk."""
        for face in self.world_faces():
            n = _unit(_newell_normal(face))
            if _dot(n, _sub(p, face[0])) > eps:
                return False
        return True

    def to_dict(self) -> dict:
        r = SERIALISE_DECIMALS
        return {
            "index": self.index,
            "source_block": self.source_block,
            "center": [round(c, r) for c in self.center],
            "orientation": [round(c, r) for c in self.orientation],
            "vertices": [[round(c, r) for c in v] for v in self.vertices],
            "edges": [list(e) for e in self.edges],
            "faces": [list(f) for f in self.faces],
            "half_extents": [round(c, r) for c in self.half_extents],
            "volume": round(self.volume, r),
            "mass": round(self.mass, r),
            "launch_dir": [round(c, r) for c in self.launch_dir],
        }


@dataclass(frozen=True)
class FractureResult:
    """The full pre-generated chunk set for one structure."""

    spec_name: str
    spec_kind: str
    seed: int
    chunks: Tuple[Chunk, ...]
    source_volume: float
    chunk_volume: float
    bounds: Tuple[Vec3, Vec3]
    generation_seconds: float

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def volume_error(self) -> float:
        """Relative error between summed chunk volume and source volume."""
        if self.source_volume <= 0.0:
            return 0.0
        return abs(self.chunk_volume - self.source_volume) / self.source_volume

    @property
    def total_mass(self) -> float:
        return sum(c.mass for c in self.chunks)

    def volumes(self) -> List[float]:
        return sorted(c.volume for c in self.chunks)

    def serialize(self) -> bytes:
        """Canonical bytes for this chunk set.

        Deliberately excludes wall-clock timing so that two runs of the same
        ``(spec, seed)`` really are byte-identical.
        """
        payload = {
            "spec_name": self.spec_name,
            "spec_kind": self.spec_kind,
            "seed": self.seed,
            "source_volume": round(self.source_volume, SERIALISE_DECIMALS),
            "chunks": [c.to_dict() for c in self.chunks],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


# ----------------------------------------------------------------- generation
def _rng(spec_name: str, seed: int, block_index: int) -> random.Random:
    """A stable RNG: independent of PYTHONHASHSEED and of dict ordering."""
    key = f"{spec_name}|{seed}|{block_index}".encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _allocate_budget(volumes: Sequence[float], budget: int) -> List[int]:
    """Split *budget* across blocks by volume, giving every block at least one."""
    n = len(volumes)
    if n == 0:
        return []
    budget = max(budget, n)
    total = sum(volumes)
    if total <= 0.0:
        base = [budget // n] * n
        for i in range(budget - sum(base)):
            base[i] += 1
        return base
    raw = [budget * v / total for v in volumes]
    alloc = [max(1, int(math.floor(r))) for r in raw]
    over = sum(alloc) - budget
    # Give back the leftovers to the blocks with the biggest fractional part.
    order = sorted(range(n), key=lambda i: (raw[i] - math.floor(raw[i])), reverse=True)
    if over < 0:
        i = 0
        while over < 0:
            alloc[order[i % n]] += 1
            over += 1
            i += 1
    elif over > 0:
        i = 0
        rev = list(reversed(order))
        while over > 0:
            j = rev[i % n]
            if alloc[j] > 1:
                alloc[j] -= 1
                over -= 1
            i += 1
            if i > 100000:  # pragma: no cover - defensive
                break
    return alloc


def _pick_plane(
    faces: Sequence[Face], rng: random.Random, spec: StructureSpec
) -> Tuple[Vec3, float]:
    """Choose a jittered split plane for one piece.

    The axis choice is what drives shape variety: cutting across the *shortest*
    axis yields flat slabs and shards; cutting across the longest yields
    chunkier blocks. The offset choice is what drives *size* variety: most cuts
    land near the middle, a healthy minority land well off-centre.
    """
    lo, hi = poly_bounds(faces)
    ext = [hi[i] - lo[i] for i in range(3)]
    order = sorted(range(3), key=lambda i: ext[i])  # shortest .. longest
    r = rng.random()
    if r < spec.shard_bias:
        axis = order[0]
    elif r < spec.shard_bias + 0.22:
        axis = order[1]
    else:
        axis = order[2]

    base = [0.0, 0.0, 0.0]
    base[axis] = 1.0
    j = spec.plane_jitter
    normal = _unit((
        base[0] + rng.uniform(-j, j),
        base[1] + rng.uniform(-j, j),
        base[2] + rng.uniform(-j, j),
    ))

    projections = [_dot(normal, p) for f in faces for p in f]
    plo, phi = min(projections), max(projections)
    if rng.random() < spec.uneven_bias:
        t = rng.uniform(0.12, 0.32)
        if rng.random() < 0.5:
            t = 1.0 - t
    else:
        t = rng.uniform(0.38, 0.62)
    return normal, plo + t * (phi - plo)


def _split_block(
    block: Block, target: int, rng: random.Random, spec: StructureSpec
) -> List[List[Face]]:
    """Recursively carve one block into at most *target* convex pieces."""
    min_vol = spec.resolved_min_chunk_volume()
    pieces: List[List[Face]] = [box_faces(block.center, block.half_extents)]
    vols: List[float] = [poly_volume(pieces[0])]
    dead: set = set()  # indices that resisted every plane we tried

    while len(pieces) < target:
        live = [i for i in range(len(pieces))
                if i not in dead and vols[i] >= 2.0 * min_vol]
        if not live:
            break
        # Volume-weighted choice (not "always the biggest"): this is what
        # leaves a few large slabs standing next to a swarm of small shards.
        weights = [vols[i] for i in live]
        pick = rng.choices(live, weights=weights, k=1)[0]

        src = pieces[pick]
        placed = False
        for _ in range(10):
            normal, offset = _pick_plane(src, rng, spec)
            lower = clip_halfspace(src, normal, offset)
            upper = clip_halfspace(src, (-normal[0], -normal[1], -normal[2]), -offset)
            if lower is None or upper is None:
                continue
            vlo = poly_volume(lower)
            vhi = poly_volume(upper)
            if vlo < min_vol or vhi < min_vol:
                continue
            pieces[pick] = lower
            vols[pick] = vlo
            pieces.append(upper)
            vols.append(vhi)
            placed = True
            break
        if not placed:
            dead.add(pick)
    return pieces


def _emit_chunk(
    index: int, block_name: str, faces: Sequence[Face], spec: StructureSpec, origin: Vec3
) -> Optional[Chunk]:
    """Weld a raw polyhedron into an immutable :class:`Chunk` descriptor."""
    center, volume = poly_centroid_volume(faces)
    if volume <= 0.0:
        return None

    verts: List[Vec3] = []
    lookup: Dict[Tuple[int, int, int], int] = {}
    index_faces: List[Tuple[int, ...]] = []
    scale = 10 ** WELD_DECIMALS
    for f in faces:
        idxs: List[int] = []
        for p in f:
            key = (int(round(p[0] * scale)), int(round(p[1] * scale)),
                   int(round(p[2] * scale)))
            i = lookup.get(key)
            if i is None:
                i = len(verts)
                lookup[key] = i
                verts.append(_sub(p, center))
            if not idxs or idxs[-1] != i:
                idxs.append(i)
        while len(idxs) >= 2 and idxs[0] == idxs[-1]:
            idxs.pop()
        if len(idxs) >= 3:
            index_faces.append(tuple(idxs))

    if len(verts) < 4 or len(index_faces) < 4:
        return None

    edge_set = set()
    for f in index_faces:
        m = len(f)
        for i in range(m):
            a, b = f[i], f[(i + 1) % m]
            edge_set.add((a, b) if a < b else (b, a))
    edges = tuple(sorted(edge_set))

    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    half = (
        max(max(xs), -min(xs)),
        max(max(ys), -min(ys)),
        max(max(zs), -min(zs)),
    )

    radial = _sub(center, origin)
    launch = _unit(radial) if _length(radial) > 1e-9 else (0.0, 0.0, 1.0)

    return Chunk(
        index=index,
        source_block=block_name,
        center=center,
        orientation=(1.0, 0.0, 0.0, 0.0),
        vertices=tuple(verts),
        edges=edges,
        faces=tuple(index_faces),
        half_extents=half,
        volume=volume,
        mass=volume * spec.density,
        launch_dir=launch,
    )


def fracture(spec: StructureSpec, seed: int) -> FractureResult:
    """Pre-generate the full chunk set for *spec* under *seed*.

    Deterministic: the same ``(spec, seed)`` always produces a byte-identical
    result (see :meth:`FractureResult.serialize`). Never returns more than
    ``spec.max_chunks`` chunks.
    """
    spec.validate()

    started = time.perf_counter()
    origin = spec.resolved_fracture_origin()
    budgets = _allocate_budget([b.volume for b in spec.blocks], spec.max_chunks)

    chunks: List[Chunk] = []
    for bi, (block, budget) in enumerate(zip(spec.blocks, budgets)):
        rng = _rng(spec.name, seed, bi)
        for faces in _split_block(block, budget, rng, spec):
            if len(chunks) >= spec.max_chunks:
                break
            chunk = _emit_chunk(len(chunks), block.name, faces, spec, origin)
            if chunk is not None:
                chunks.append(chunk)

    elapsed = time.perf_counter() - started
    return FractureResult(
        spec_name=spec.name,
        spec_kind=spec.kind,
        seed=seed,
        chunks=tuple(chunks),
        source_volume=spec.volume,
        chunk_volume=sum(c.volume for c in chunks),
        bounds=spec.bounds,
        generation_seconds=elapsed,
    )


def with_budget(spec: StructureSpec, max_chunks: int) -> StructureSpec:
    """A copy of *spec* with a different chunk budget (specs are frozen)."""
    return replace(spec, max_chunks=max_chunks)


def convex_pair_overlap(a: "Chunk", b: "Chunk", eps: float = 1e-6) -> bool:
    """Exact-ish separating-axis test for two convex chunks.

    Monte-Carlo point sampling can miss a thin overlap entirely (that is how an
    interpenetrating source block slipped through validation once). This walks
    every face normal of both solids plus every edge-edge cross product and
    reports ``False`` as soon as one axis separates them, so a clean result is
    a real proof of disjointness rather than a statistical hope.

    Solids that merely *touch* (shared face, shared edge) are not overlapping:
    an axis whose projections meet within *eps* counts as separating.
    """
    fa = a.world_faces()
    fb = b.world_faces()
    va = a.world_vertices()
    vb = b.world_vertices()

    axes: List[Vec3] = []
    for faces in (fa, fb):
        for f in faces:
            n = _newell_normal(f)
            if _length(n) > 1e-12:
                axes.append(_unit(n))

    def _edges(faces):
        out = []
        for f in faces:
            m = len(f)
            for i in range(m):
                d = _sub(f[(i + 1) % m], f[i])
                if _length(d) > 1e-12:
                    out.append(_unit(d))
        return out

    ea, eb = _edges(fa), _edges(fb)
    for da in ea:
        for db in eb:
            c = _cross(da, db)
            if _length(c) > 1e-9:
                axes.append(_unit(c))

    for ax in axes:
        pa = [_dot(ax, p) for p in va]
        pb = [_dot(ax, p) for p in vb]
        if min(pa) >= max(pb) - eps or min(pb) >= max(pa) - eps:
            return False  # separating axis found -> disjoint
    return True


def size_bands(result: FractureResult, bands: int = 3) -> List[int]:
    """Count chunks per logarithmic size band, smallest band first."""
    vols = result.volumes()
    if not vols:
        return [0] * bands
    lo = math.log10(max(vols[0], 1e-12))
    hi = math.log10(max(vols[-1], 1e-12))
    if hi - lo < 1e-12:
        return [len(vols)] + [0] * (bands - 1)
    counts = [0] * bands
    width = (hi - lo) / bands
    for v in vols:
        b = int((math.log10(max(v, 1e-12)) - lo) / width)
        counts[min(b, bands - 1)] += 1
    return counts
