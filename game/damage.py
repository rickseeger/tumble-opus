"""Accumulated structural damage, and the threshold that turns a building
into debris.

This is the missing middle of the destruction chain. Before this module the
game had two halves that never met:

* :mod:`game.fracture` / :mod:`game.debris` knew *how* to turn a structure's
  pre-generated chunk descriptors into independently simulated rigid bodies
  (node 10/11), and
* :class:`game.app.TumbleApp` had a ``demolish()`` verb and a debug key bound
  to it.

Nothing in the running game ever *decided* that a structure should come down.
Destruction was something a test, a demo or a keypress asked for directly.

:class:`DamageSystem` is that decision. It holds one
:class:`StructureDamage` record per placed structure, accepts damage events in
world space, and when a structure's accumulated damage reaches its integrity
threshold it fires the game's real destruction path - the same
:meth:`game.debris.DebrisField.demolish` that pulls the static collision
proxies out of the Bullet world and spawns the chunk set in their place.

Deliberate properties
---------------------
**Damage is only ever caused by an explicit call.** Nothing in here watches
contacts. That is not an omission: debris raining onto a neighbouring tower
must not demolish it by accident, and a collapsing structure must not chain
into the rest of the course. The later weapon node calls :meth:`apply_damage`;
until then, the only callers are the debug strike and the tests.

**Destruction is resolved, not performed, inside the damage call.** A damage
event appends the structure to :attr:`pending`, and :meth:`resolve` does the
actual demolition. :meth:`game.app.TumbleApp.step_frame` calls ``resolve()``
before it advances physics, so anything damaged during a frame is rubble in
that same frame, before a single substep runs. ``apply_damage`` resolves
immediately by default too, so a caller outside the loop gets the same
guarantee. The split exists so damage can safely be applied from inside a
physics callback, where mutating the world mid-step is illegal.

**No budgeting lives here.** How much debris the world may hold is
:mod:`game.debris`'s business and the budget node's; this module only decides
*when* a structure stops being a building.

No Panda3D imports beyond plain vector maths, no ShowBase, no render graph -
so it is headless-testable exactly like the rest of the sim.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from . import config

Triple = Tuple[float, float, float]


def integrity_for(destructible) -> float:
    """How much damage a structure absorbs before it comes down.

    Proportional to its volume, with a floor, so a 60 m tower is genuinely
    harder to bring down than a low cluster and the number stays meaningful
    if the placement list changes. Volume is taken from the structure's spec,
    which is the same volume the fracture library tiled.
    """
    volume = float(getattr(destructible.spec, "volume", 0.0))
    return max(
        volume * config.STRUCTURE_INTEGRITY_PER_M3,
        config.STRUCTURE_INTEGRITY_MIN,
    )


def _aabb_distance(point: Triple, lo: Triple, hi: Triple) -> float:
    """Distance from *point* to an axis-aligned box; 0 when inside it.

    Distance to the *volume*, not to the centre: a blast landing against the
    face of a 60 m tower is a direct hit, and measuring to the centroid would
    score it as a 30 m near miss.
    """
    d2 = 0.0
    for i in range(3):
        p = float(point[i])
        if p < lo[i]:
            d2 += (lo[i] - p) ** 2
        elif p > hi[i]:
            d2 += (p - hi[i]) ** 2
    return d2 ** 0.5


@dataclass
class StructureDamage:
    """The running damage tally for one placed structure."""

    name: str
    threshold: float
    accumulated: float = 0.0
    hits: int = 0
    destroyed: bool = False
    #: ``physics.step_count`` at the moment the threshold was crossed.
    destroyed_step: int = -1

    @property
    def remaining(self) -> float:
        return max(0.0, self.threshold - self.accumulated)

    @property
    def fraction(self) -> float:
        """0.0 untouched .. 1.0 at (or past) the destruction threshold."""
        if self.threshold <= 0.0:
            return 1.0
        return min(1.0, self.accumulated / self.threshold)

    @property
    def at_threshold(self) -> bool:
        return self.accumulated >= self.threshold


@dataclass
class DamageReport:
    """What one :meth:`DamageSystem.apply_damage` call actually did."""

    point: Triple
    amount: float
    radius: float
    #: ``{structure name: damage actually dealt}`` - falloff already applied.
    damaged: Dict[str, float] = _dc_field(default_factory=dict)
    #: Structures whose accumulated damage reached the threshold on this call.
    destroyed: Tuple[str, ...] = ()
    #: Debris bodies inside the blast radius. The hook the later
    #: "debris hurts the player" node reads; proof here that freshly spawned
    #: debris is immediately visible to the damage system's own query path.
    debris_in_radius: int = 0
    #: :class:`game.debris.ShatterEvent` per destroyed structure, in order.
    events: Tuple[object, ...] = _dc_field(default=(), repr=False)

    @property
    def any_destroyed(self) -> bool:
        return bool(self.destroyed)


class DamageSystem:
    """Accumulated damage for every placed structure, and the trigger.

    Construct it around the live destructible list and debris field::

        damage = DamageSystem(app.destructibles, app.debris,
                              physics=app.physics, on_destroy=app.demolish)
        damage.apply_damage((0.0, 30.0, 4.0), 500.0, radius=14.0)

    ``on_destroy(destructible, impact_point=...)`` is the game's real
    destruction path. In the running game that is
    :meth:`game.app.TumbleApp.demolish`, which removes the collision proxies,
    spawns the debris bodies and swaps the visuals.
    """

    def __init__(
        self,
        destructibles: Sequence,
        debris,
        physics=None,
        on_destroy: Optional[Callable] = None,
    ) -> None:
        self.destructibles = list(destructibles)
        #: Where the killing blow landed, per structure awaiting resolve.
        self._impact_points: Dict[str, Optional[Triple]] = {}
        self.debris = debris
        self.physics = physics if physics is not None else getattr(
            debris, "physics", None
        )
        self.on_destroy = on_destroy

        self.states: Dict[str, StructureDamage] = {
            d.name: StructureDamage(name=d.name, threshold=integrity_for(d))
            for d in self.destructibles
        }
        #: Structures that have crossed the threshold and are awaiting
        #: :meth:`resolve`. Never more than one entry per structure.
        self.pending: List = []
        self.total_damage_dealt = 0.0
        self.total_destroyed = 0
        self.reports: List[DamageReport] = []

    # ------------------------------------------------------------- lookups
    def state(self, destructible) -> StructureDamage:
        name = getattr(destructible, "name", destructible)
        return self.states[name]

    def by_name(self, name: str):
        for d in self.destructibles:
            if d.name == name:
                return d
        return None

    @property
    def intact(self) -> List:
        return [d for d in self.destructibles if d.intact]

    def threshold(self, destructible) -> float:
        return self.state(destructible).threshold

    # -------------------------------------------------------------- damage
    def damage_structure(
        self,
        destructible,
        amount: float,
        impact_point: Optional[Triple] = None,
        resolve: bool = True,
    ) -> StructureDamage:
        """Put *amount* of damage on one structure, ignoring falloff.

        The direct form, for a contact hit or a scripted event. Returns the
        structure's damage record.
        """
        st = self.state(destructible)
        if st.destroyed or not destructible.intact:
            return st
        amount = max(0.0, float(amount))
        if amount <= 0.0:
            return st
        st.accumulated += amount
        st.hits += 1
        self.total_damage_dealt += amount
        if st.at_threshold:
            self._enqueue(destructible, impact_point)
            if resolve:
                self.resolve()
        return st

    def apply_damage(
        self,
        point: Triple,
        amount: float,
        radius: float = config.DAMAGE_DEFAULT_RADIUS,
        resolve: bool = True,
    ) -> DamageReport:
        """Land a blast of *amount* damage at *point* with *radius* falloff.

        Every intact structure whose volume lies within *radius* of the point
        takes damage scaled by :data:`config.DAMAGE_FALLOFF_AT_EDGE` at the
        rim and full strength at zero distance. Structures that reach their
        threshold are demolished through the game's real destruction path
        (immediately, unless ``resolve=False``).
        """
        point = (float(point[0]), float(point[1]), float(point[2]))
        amount = max(0.0, float(amount))
        radius = max(1e-6, float(radius))

        report = DamageReport(point=point, amount=amount, radius=radius)

        for d in list(self.destructibles):
            st = self.states[d.name]
            if st.destroyed or not d.intact:
                continue
            lo, hi = d.world_bounds()
            dist = _aabb_distance(point, lo, hi)
            if dist > radius:
                continue
            t = dist / radius
            scale = 1.0 + (config.DAMAGE_FALLOFF_AT_EDGE - 1.0) * t
            dealt = amount * scale
            if dealt <= 0.0:
                continue
            report.damaged[d.name] = dealt
            self.damage_structure(d, dealt, impact_point=point, resolve=False)

        if self.debris is not None:
            report.debris_in_radius = len(self.debris_near(point, radius))

        pending_names = tuple(d.name for d in self.pending)
        if resolve and self.pending:
            events = self.resolve()
            report.destroyed = pending_names
            report.events = tuple(events)

        self.reports.append(report)
        return report

    # ------------------------------------------------------------- resolve
    def _enqueue(self, destructible, impact_point: Optional[Triple]) -> None:
        if destructible in self.pending:
            return
        self.pending.append(destructible)
        # Remember where the killing blow landed, so the chunks launch away
        # from the actual hit rather than the structure's default origin.
        self._impact_points[destructible.name] = impact_point

    def resolve(self) -> List:
        """Demolish everything that has reached its threshold.

        Called by the frame loop before it advances physics, so a structure
        damaged during a frame is already rubble when that frame simulates.
        Returns the :class:`~game.debris.ShatterEvent` for each demolition.
        """
        events = []
        while self.pending:
            d = self.pending.pop(0)
            st = self.states[d.name]
            if st.destroyed:
                continue
            point = self._impact_points.pop(d.name, None)
            # Marked *before* dispatch: the host's demolish() calls
            # note_destroyed(), and this is what keeps that from
            # double-counting the same structure.
            st.destroyed = True
            st.destroyed_step = int(
                getattr(self.physics, "step_count", 0)
                if self.physics is not None else 0
            )
            self.total_destroyed += 1
            event = self._destroy(d, point)
            if event is not None:
                events.append(event)
        return events

    def _destroy(self, destructible, impact_point: Optional[Triple]):
        if self.on_destroy is not None:
            return self.on_destroy(destructible, impact_point=impact_point)
        # No host app wired in (unit tests): go straight at the debris field,
        # which is exactly what the host's demolish() does minus the visuals.
        return self.debris.demolish(destructible, impact_point=impact_point)

    def note_destroyed(self, destructible) -> None:
        """Record that a structure came down by some path other than damage.

        The debug strike, the campaign script and the tests may call
        :meth:`game.app.TumbleApp.demolish` directly. Without this the tally
        would still show the structure as standing at 0 % damage, and a later
        blast would try to demolish rubble. Idempotent.
        """
        st = self.states.get(getattr(destructible, "name", destructible))
        if st is None or st.destroyed:
            return
        st.destroyed = True
        st.accumulated = max(st.accumulated, st.threshold)
        st.destroyed_step = int(
            getattr(self.physics, "step_count", 0)
            if self.physics is not None else 0
        )
        self.total_destroyed += 1
        if destructible in self.pending:
            self.pending.remove(destructible)
        self._impact_points.pop(getattr(destructible, "name", ""), None)

    # ----------------------------------------------------- debris querying
    def debris_near(self, point: Triple, radius: float) -> List:
        """Debris bodies within *radius* of *point*.

        The damage system's own view of the debris field. A later node makes
        these hurt the player; this node's contract is only that freshly
        spawned debris shows up here the instant it exists.
        """
        if self.debris is None:
            return []
        return self.debris.bodies_near(point, radius)

    def debris_in_contact(self, node) -> List:
        """Debris currently touching a Bullet node (player capsule, etc.)."""
        if self.debris is None:
            return []
        return self.debris.contact_test(node)

    # --------------------------------------------------------------- stats
    def snapshot(self) -> dict:
        return {
            "structures": len(self.destructibles),
            "intact": sum(1 for d in self.destructibles if d.intact),
            "destroyed": self.total_destroyed,
            "pending": len(self.pending),
            "damage_dealt": self.total_damage_dealt,
            "worst_fraction": max(
                (s.fraction for s in self.states.values()), default=0.0
            ),
        }
