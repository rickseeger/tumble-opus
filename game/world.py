"""Course authoring: the ground plane and the placeholder structures.

`build_course` is pure sim - it populates a `PhysicsWorld` and nothing else.
The render layer later walks `physics.box_specs` to draw matching visuals, so
collision and visuals cannot drift out of sync.
"""

from __future__ import annotations

from typing import List

from . import config
from .physics import BoxSpec, PhysicsWorld

#: Placeholder static structures: a few simple box towers straight down the
#: course, plus low walls that fence the lane. Later nodes replace these with
#: fracturable geometry; for now they exist so there is something solid to
#: bump into and something for the tests to prove is impassable.
TOWERS = (
    # (name,        x,     y,     levels)
    ("tower_a",   -6.0,   15.0,  3),
    ("tower_b",    6.0,   28.0,  2),
    ("tower_c",    0.0,   45.0,  4),
    ("tower_d",   -9.0,   62.0,  2),
    ("tower_e",    8.0,   80.0,  3),
)

BLOCK_HALF = (1.5, 1.5, 1.0)


def build_course(physics: PhysicsWorld) -> List[BoxSpec]:
    """Ground plane + placeholder structures. Returns every box authored."""
    physics.add_ground_plane(0.0)

    before = len(physics.box_specs)

    for name, x, y, levels in TOWERS:
        for level in range(levels):
            z = BLOCK_HALF[2] + level * (BLOCK_HALF[2] * 2.0)
            physics.add_static_box(
                f"{name}_{level}", (x, y, z), BLOCK_HALF
            )

    # Lane boundary walls, so the player is funnelled forward down the course.
    wall_half = (0.5, config.COURSE_LENGTH * 0.5, 1.5)
    mid_y = config.COURSE_LENGTH * 0.5 - 20.0
    physics.add_static_box(
        "wall_left", (-config.COURSE_HALF_WIDTH, mid_y, wall_half[2]), wall_half
    )
    physics.add_static_box(
        "wall_right", (config.COURSE_HALF_WIDTH, mid_y, wall_half[2]), wall_half
    )

    return physics.box_specs[before:]


def block_count() -> int:
    """How many tower blocks `build_course` will author (excludes walls)."""
    return sum(levels for _, _, _, levels in TOWERS)
