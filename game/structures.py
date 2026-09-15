"""Destructible structure placement: the bridge from authoring to demolition.

A :class:`Destructible` is one structure standing on the course. While it is
intact it is a handful of *static proxy* boxes - solid, cheap, and something
the player can bump into. The moment it is demolished the proxies are removed
from the Bullet world and its **pre-generated** chunk set (produced once at
load time by :mod:`game.fracture`) becomes live debris via
:meth:`game.debris.DebrisField.demolish`.

The split matters for two reasons:

* **Cost.** Fracturing all four archetypes takes ~220 ms on this machine
  (measured). That is fine once at load; it is not fine on the frame a tower
  comes down. So generation happens here, at build time, and the runtime only
  ever converts descriptors into bodies.
* **Honesty of collision.** An intact tower should not be 380 sleeping rigid
  bodies pretending to be a building. A few static boxes are the truthful
  representation right up until the moment it stops being a building.

Nothing in this module imports the render graph or ShowBase.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from typing import List, Optional, Sequence, Tuple

from . import fracture
from .fracture import FractureResult, StructureSpec

Triple = Tuple[float, float, float]

#: The destructible line-up down the course: (archetype, x, y, seed).
#: Spread along +Y so the player meets them one at a time, which is also what
#: lets the retirement policy despawn the rubble behind them.
PLACEMENTS: Tuple[Tuple[str, float, float, int], ...] = (
    ("cluster", -8.0, 30.0, 101),
    ("arch", 0.0, 62.0, 102),
    ("slab", 6.0, 96.0, 103),
    ("tower", -6.0, 134.0, 104),
    ("cluster", 9.0, 168.0, 105),
)


@dataclass
class Destructible:
    """One placed structure, with its chunk set already generated."""

    name: str
    archetype: str
    spec: StructureSpec
    origin: Triple
    result: FractureResult
    #: Names of the static proxy bodies in the :class:`PhysicsWorld`.
    proxy_names: Tuple[str, ...] = ()
    intact: bool = True
    #: Set once it has been demolished, for the campaign node's bookkeeping.
    demolished_step: int = -1
    #: Remembered chunk count, so the figure survives `release_chunks()`.
    _chunk_count: int = -1

    def __post_init__(self) -> None:
        if self._chunk_count < 0 and self.result is not None:
            self._chunk_count = len(self.result.chunks)

    # ------------------------------------------------------------- geometry
    @property
    def chunk_count(self) -> int:
        """How many chunks were generated. Survives :meth:`release_chunks`."""
        if self.result is None:
            return self._chunk_count
        return len(self.result.chunks)

    @property
    def mass(self) -> float:
        return 0.0 if self.result is None else self.result.total_mass

    def world_bounds(self) -> Tuple[Triple, Triple]:
        lo, hi = self.spec.bounds
        o = self.origin
        return (
            (lo[0] + o[0], lo[1] + o[1], lo[2] + o[2]),
            (hi[0] + o[0], hi[1] + o[1], hi[2] + o[2]),
        )

    def default_impact_point(self) -> Triple:
        """Where a blast lands if the caller does not say: the fracture
        origin the library already chose, lifted into world space."""
        fo = self.spec.resolved_fracture_origin()
        o = self.origin
        return (fo[0] + o[0], fo[1] + o[1], fo[2] + o[2])

    def release_chunks(self) -> int:
        """Drop the pre-generated chunk descriptors once they are spent.

        A structure's :class:`~game.fracture.FractureResult` is pure
        load-time data - a few hundred convex hulls with their vertices, edges
        and inertia - and it is only ever read once, on the frame the
        structure shatters. After that it is dead weight: the chunks that
        matter are now :class:`~game.debris.DebrisBody` objects, and the ones
        the budget declined to spawn are never revisited.

        Measured on this stack: authored chunk descriptors cost **1.22 MB per
        structure** (a 240-380 chunk archetype). That is invisible for the
        shipping course of five, but an endless campaign - or the demolition
        soak - authors structures forever, and holding every spent descriptor
        set is a straight linear leak: 60 structures retained is +72.3 MB and
        still climbing, versus +2.3 MB when they are released as they are
        spent.

        Only legal on rubble: a structure still standing needs its chunks to
        shatter into. Returns the number of chunk descriptors released (0 if
        it is still intact or already released), and is idempotent.
        """
        if self.intact or self.result is None:
            return 0
        released = len(self.result.chunks)
        self.result = None
        return released

    @property
    def chunks_released(self) -> bool:
        return self.result is None

    def contains_y(self, y: float, pad: float = 0.0) -> bool:
        lo, hi = self.world_bounds()
        return lo[1] - pad <= float(y) <= hi[1] + pad


def generate(archetype: str, origin: Triple, seed: int,
             name: Optional[str] = None,
             max_chunks: Optional[int] = None) -> Destructible:
    """Pre-generate one destructible. Deterministic in ``seed``."""
    spec = fracture.ARCHETYPES[archetype]()
    if max_chunks is not None:
        spec = fracture.with_budget(spec, max_chunks)
    return Destructible(
        name=name or f"{archetype}_{int(origin[1])}",
        archetype=archetype,
        spec=spec,
        origin=(float(origin[0]), float(origin[1]), float(origin[2])),
        result=fracture.fracture(spec, seed=seed),
    )


def attach_proxies(physics, d: Destructible, friction: float = 0.9) -> Destructible:
    """Give *d* its static collision proxy: one box per source block."""
    names: List[str] = []
    for block in d.spec.blocks:
        bname = f"{d.name}__{block.name}"
        pos = (
            block.center[0] + d.origin[0],
            block.center[1] + d.origin[1],
            block.center[2] + d.origin[2],
        )
        physics.add_static_box(bname, pos, block.half_extents, friction=friction)
        names.append(bname)
    d.proxy_names = tuple(names)
    return d


def build_destructibles(
    physics,
    placements: Sequence[Tuple[str, float, float, int]] = PLACEMENTS,
    max_chunks: Optional[int] = None,
) -> List[Destructible]:
    """Pre-generate and place every destructible on the course.

    Runs the (load-time) fracture pass and attaches the static proxies. The
    returned list is what the campaign node iterates over.
    """
    out: List[Destructible] = []
    for archetype, x, y, seed in placements:
        d = generate(archetype, (x, y, 0.0), seed, max_chunks=max_chunks)
        attach_proxies(physics, d)
        out.append(d)
    return out


def total_chunks(destructibles: Sequence[Destructible]) -> int:
    return sum(d.chunk_count for d in destructibles)


def nearest_intact(destructibles: Sequence[Destructible],
                   point: Triple) -> Optional[Destructible]:
    """The closest structure still standing, by centre distance."""
    best = None
    best_d2 = float("inf")
    for d in destructibles:
        if not d.intact:
            continue
        lo, hi = d.world_bounds()
        c = ((lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5)
        d2 = sum((c[i] - float(point[i])) ** 2 for i in range(3))
        if d2 < best_d2:
            best_d2 = d2
            best = d
    return best
