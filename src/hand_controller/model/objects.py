"""Objects the hand is asked to hold, and whether a grasp can reach them.

Each object carries the five properties the controller has to cope with: the
width the grasp must close on, the mass that has to be held against gravity, the
contact stiffness that decides how much finger travel one newton costs, the
friction coefficient against the fingertip that decides how much force is
needed, and the normal force above which the object is damaged.

The controller is given none of these. It starts from a fixed conservative grip
force and raises it only in response to slip, as Romano et al. (2011) do, so the
object properties act on the loop only through the physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from hand_controller.model.anatomy import HandModel
from hand_controller.model.grasps import GraspDefinition, span_range

__all__ = [
    "OBJECT_NAMES",
    "OBJECT_SET",
    "Feasibility",
    "GraspObject",
    "feasibility",
    "grasp_object",
]


class Feasibility(StrEnum):
    """Whether a grasp can enclose an object at all."""

    FEASIBLE = "feasible"
    TOO_LARGE = "too_large"
    TOO_SMALL = "too_small"


@dataclass(frozen=True, slots=True)
class GraspObject:
    """One object, with the mechanical properties that decide whether it is held."""

    name: str
    description: str
    width: float
    mass: float
    stiffness: float
    friction: float
    crush_force: float
    damping: float = 0.6

    def __post_init__(self) -> None:
        if self.width <= 0.0:
            raise ValueError(f"{self.name}: width must be positive, got {self.width}")
        if self.mass <= 0.0:
            raise ValueError(f"{self.name}: mass must be positive, got {self.mass}")
        if not 0.0 < self.friction <= 2.0:
            raise ValueError(f"{self.name}: friction must lie in (0, 2], got {self.friction}")


_OBJECTS: Final[tuple[GraspObject, ...]] = (
    GraspObject(
        name="drinking_glass",
        description="Straight sided glass holding water, rigid and slippery when wet.",
        width=0.066,
        mass=0.420,
        stiffness=5.0e7,
        friction=0.45,
        crush_force=120.0,
    ),
    GraspObject(
        name="plastic_bottle",
        description="Half filled 500 ml drinks bottle, thin walled and semi rigid.",
        width=0.062,
        mass=0.520,
        stiffness=6.0e4,
        friction=0.40,
        crush_force=35.0,
    ),
    GraspObject(
        name="foam_cup",
        description="Empty expanded foam cup, very deformable and almost weightless.",
        width=0.062,
        mass=0.012,
        stiffness=3.0e3,
        friction=0.70,
        crush_force=2.0,
    ),
    GraspObject(
        name="paper_cup_full",
        description="Thin walled paper cup filled with water, deformable, heavy and easily"
        " crushed.",
        width=0.065,
        mass=0.300,
        stiffness=8.0e3,
        friction=0.35,
        crush_force=4.0,
    ),
    GraspObject(
        name="apple",
        description="Large apple, firm skin over softer flesh.",
        width=0.065,
        mass=0.240,
        stiffness=2.0e5,
        friction=0.32,
        crush_force=45.0,
    ),
    GraspObject(
        name="hardback_book",
        description="Slim hardback book held by its edge, stiff and heavy for a pad grasp.",
        width=0.030,
        mass=0.450,
        stiffness=1.0e6,
        friction=0.40,
        crush_force=60.0,
    ),
    GraspObject(
        name="battery_cell",
        description="D size cell, smooth steel case, heavy for a two digit pinch.",
        width=0.033,
        mass=0.140,
        stiffness=1.0e7,
        friction=0.30,
        crush_force=60.0,
    ),
    GraspObject(
        name="pen",
        description="Ballpoint pen barrel, held in a precision grasp.",
        width=0.011,
        mass=0.012,
        stiffness=1.0e7,
        friction=0.35,
        crush_force=60.0,
    ),
    GraspObject(
        name="door_key",
        description="Flat brass key, held between the thumb and the side of the index.",
        width=0.004,
        mass=0.010,
        stiffness=5.0e7,
        friction=0.30,
        crush_force=100.0,
    ),
    GraspObject(
        name="steel_ball",
        description="Solid polished steel sphere, 62 mm across, heavy and low friction.",
        width=0.062,
        mass=0.980,
        stiffness=1.0e8,
        friction=0.10,
        crush_force=500.0,
    ),
)

OBJECT_SET: Final[tuple[GraspObject, ...]] = _OBJECTS
OBJECT_NAMES: Final[tuple[str, ...]] = tuple(item.name for item in _OBJECTS)


def grasp_object(name: str) -> GraspObject:
    """Look up one object by name."""
    for item in _OBJECTS:
        if item.name == name:
            return item
    raise KeyError(f"unknown object: {name!r}; known objects are {', '.join(OBJECT_NAMES)}")


def feasibility(
    hand: HandModel, definition: GraspDefinition, item: GraspObject
) -> Feasibility:
    """Report whether ``definition`` can enclose ``item`` at all.

    The grasp sweeps its span from the open value down to the closed value. An
    object wider than the open span cannot be surrounded, and one narrower than
    the closed span cannot be reached even with the hand fully shut. Both are
    ordinary failures of a fixed set of prosthetic grasp postures.
    """
    closed_span, open_span = span_range(hand, definition)
    if item.width > open_span:
        return Feasibility.TOO_LARGE
    if item.width < closed_span:
        return Feasibility.TOO_SMALL
    return Feasibility.FEASIBLE
