"""Whether a grasp may be closed on an object at all, decided before the reach.

The loop is deliberately blind. It starts from a light nominal force and raises
it only on tactile evidence, so the way it discovers that an object is too heavy
or too slippery is by dropping it. That is the right behaviour for a controller
and the wrong answer for a hand that is about to reach: the steel ball of the
evaluation set needs 19.221 N per contact against a 15.0 N safety limit and was
never holdable, and the correct response to it is to refuse it rather than to
close on it and find out.

The screen here answers that from the object properties and the contact
expectation of the grasp, with nothing simulated. Three quantities decide it: the
span range of the grasp, which fixes whether the object can be enclosed at all;
the safety limit of the force loop, which is what the actuator may deliver; and
the crush limit of the object, which is what the object will take. The last two
are both ceilings on the force one contact may carry, and the requirement is
compared against the lower of them.

Two things it is not. It is statics, so a grasp it admits is one the loop could
hold if it found the force; how long the loop takes to get there and how far the
object slides on the way are questions only a trace answers. And it ranks by
force alone, so it knows the width of an object but nothing about its shape, and
will prefer the grasp that shares the load over the most contacts among those
that fit. Choosing the grasp that suits the shape stays with the user, and
``EVALUATION_SET`` records those choices.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from hand_controller.algorithm.force import ForceConfig
from hand_controller.model.anatomy import HandModel
from hand_controller.model.contact import required_grip_force
from hand_controller.model.grasps import GRASP_TAXONOMY, GraspDefinition
from hand_controller.model.objects import Feasibility, GraspObject, feasibility

__all__ = [
    "Admissibility",
    "GraspScreen",
    "UnholdableObjectError",
    "screen_grasp",
    "screen_taxonomy",
    "select_grasp",
]


class Admissibility(StrEnum):
    """Whether a grasp may be closed on an object, and why not when it may not."""

    ADMISSIBLE = "admissible"
    TOO_LARGE = "too_large"
    TOO_SMALL = "too_small"
    WOULD_CRUSH = "would_crush"
    EXCEEDS_LIMIT = "exceeds_limit"


class UnholdableObjectError(ValueError):
    """Raised when no grasp of the taxonomy may be closed on an object."""


@dataclass(frozen=True, slots=True)
class GraspScreen:
    """The verdict on one grasp and object pair, with the forces behind it."""

    object_name: str
    grasp_name: str
    verdict: Admissibility
    contacts: int
    required_force: float
    force_ceiling: float
    margin: float

    @property
    def admissible(self) -> bool:
        """True when the grasp encloses the object and the force it needs fits."""
        return self.verdict is Admissibility.ADMISSIBLE


def _verdict(
    enclosure: Feasibility, required: float, ceiling: float, crush_force: float
) -> Admissibility:
    """Decide one verdict from the enclosure and the two force ceilings."""
    if enclosure is Feasibility.TOO_LARGE:
        return Admissibility.TOO_LARGE
    if enclosure is Feasibility.TOO_SMALL:
        return Admissibility.TOO_SMALL
    # The crush limit is tested first because it is the refusal nothing argues
    # with. An object that only passes the safety limit would be held by a
    # stronger hand; one that breaks below the force it needs is held by none.
    if required > crush_force:
        return Admissibility.WOULD_CRUSH
    if required > ceiling:
        return Admissibility.EXCEEDS_LIMIT
    return Admissibility.ADMISSIBLE


def screen_grasp(
    hand: HandModel,
    definition: GraspDefinition,
    item: GraspObject,
    force: ForceConfig | None = None,
    safety_factor: float = 1.0,
) -> GraspScreen:
    """Screen one grasp against one object.

    ``safety_factor`` multiplies the force the object needs, which is how a
    caller asks for headroom above the point at which the object starts to
    slide rather than for the force that exactly balances its weight.
    """
    if safety_factor < 1.0:
        raise ValueError(f"safety factor must be at least 1, got {safety_factor}")
    settings = force if force is not None else ForceConfig()
    contacts = definition.load_bearing_contacts
    required = required_grip_force(item.mass, item.friction, contacts, safety_factor)
    ceiling = min(settings.safety_limit, item.crush_force)
    return GraspScreen(
        object_name=item.name,
        grasp_name=definition.name,
        verdict=_verdict(
            feasibility(hand, definition, item), required, settings.safety_limit, item.crush_force
        ),
        contacts=contacts,
        required_force=required,
        force_ceiling=ceiling,
        margin=ceiling / required,
    )


def screen_taxonomy(
    hand: HandModel,
    item: GraspObject,
    force: ForceConfig | None = None,
    safety_factor: float = 1.0,
) -> tuple[GraspScreen, ...]:
    """Screen every grasp of the taxonomy against one object, in taxonomy order."""
    return tuple(
        screen_grasp(hand, definition, item, force, safety_factor)
        for definition in GRASP_TAXONOMY
    )


def select_grasp(
    hand: HandModel,
    item: GraspObject,
    force: ForceConfig | None = None,
    safety_factor: float = 1.0,
) -> GraspScreen:
    """Return the admissible grasp with the most force headroom.

    Ties are settled by taxonomy order. Raises ``UnholdableObjectError`` when
    no grasp is admissible, naming the one that came closest, because a refusal
    a user cannot interpret is no better than a drop.
    """
    screens = screen_taxonomy(hand, item, force, safety_factor)
    admitted = [screen for screen in screens if screen.admissible]
    if admitted:
        return max(admitted, key=lambda screen: screen.margin)

    enclosing = [
        screen
        for screen in screens
        if screen.verdict not in (Admissibility.TOO_LARGE, Admissibility.TOO_SMALL)
    ]
    if not enclosing:
        raise UnholdableObjectError(
            f"no grasp of the taxonomy encloses {item.name} at {item.width * 1000.0:.1f} mm"
        )
    closest = max(enclosing, key=lambda screen: screen.margin)
    raise UnholdableObjectError(
        f"no grasp of the taxonomy holds {item.name}: the closest is {closest.grasp_name}, "
        f"which needs {closest.required_force:.3f} N per contact against a ceiling of "
        f"{closest.force_ceiling:.3f} N"
    )
