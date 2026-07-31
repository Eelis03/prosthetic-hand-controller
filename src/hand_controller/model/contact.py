"""Compliant contact between a fingertip and a grasped object.

The fingertip pad and the object are treated as two Hertzian springs in series.
A single contact between elastic bodies obeys ``f = k d**n`` with ``n = 3/2``
for curved surfaces (Hertz; see Johnson, 1985, chapter 4). Two such springs in
series carry the same force, so their indentations add:

    d = (f / k_pad)**(1/n) + (f / k_object)**(1/n)

which inverts in closed form to ``f = k_eff d**n`` with

    k_eff = (k_pad**(-1/n) + k_object**(-1/n))**(-n)

Energy loss during the approach is added as the velocity proportional term of
Hunt and Crossley (1975), ``f = k_eff d**n (1 + lambda d_dot)``, which keeps the
force continuous at the instant of touchdown instead of stepping like a
Kelvin-Voigt damper.

The consequence that matters for control is that ``k_eff`` is dominated by the
softer of the two bodies. A rigid object is limited by the pad, a deformable one
by itself, and the same commanded opening therefore produces very different
forces on the two.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

__all__ = [
    "GRAVITY",
    "HERTZ_EXPONENT",
    "FingertipPad",
    "contact_force",
    "default_pad",
    "detect_contact",
    "effective_stiffness",
    "equilibrium_force",
    "friction_capacity",
    "indentation_for_force",
    "penetration",
    "required_grip_force",
]

GRAVITY: Final[float] = 9.80665
"""Standard gravitational acceleration, m/s^2."""

HERTZ_EXPONENT: Final[float] = 1.5
"""Exponent of the Hertzian force to indentation law for curved contacts."""


@dataclass(frozen=True, slots=True)
class FingertipPad:
    """The elastic pad on a fingertip.

    ``stiffness`` is the Hertzian coefficient in N/m^n for the exponent ``n``,
    so it is not a spring rate and cannot be compared with one directly. The
    default corresponds to a silicone prosthetic fingertip that indents about
    one millimetre under six newtons.
    """

    stiffness: float
    exponent: float = HERTZ_EXPONENT

    def __post_init__(self) -> None:
        if self.stiffness <= 0.0:
            raise ValueError(f"pad stiffness must be positive, got {self.stiffness}")
        if self.exponent <= 0.0:
            raise ValueError(f"pad exponent must be positive, got {self.exponent}")


def default_pad() -> FingertipPad:
    """Return the reference silicone fingertip pad."""
    return FingertipPad(stiffness=2.0e5, exponent=HERTZ_EXPONENT)


def effective_stiffness(pad: FingertipPad, object_stiffness: float) -> float:
    """Return the series stiffness of the pad and the object, in N/m^n."""
    if object_stiffness <= 0.0:
        raise ValueError(f"object stiffness must be positive, got {object_stiffness}")
    exponent = pad.exponent
    compliance = pad.stiffness ** (-1.0 / exponent) + object_stiffness ** (-1.0 / exponent)
    return float(compliance ** (-exponent))


def penetration(span: float, width: float) -> float:
    """Return the indentation at one contact, in metres.

    The grasp closes on the object from two opposing sides, so the excess of the
    object width over the grasp span is shared equally between them. The value is
    negative when the hand is still open around the object, which is what makes
    ``detect_contact`` a strict sign test.
    """
    return 0.5 * (width - span)


def detect_contact(indentation: float) -> bool:
    """True when and only when the indentation is strictly positive."""
    return indentation > 0.0


def contact_force(
    indentation: float,
    indentation_rate: float,
    stiffness: float,
    exponent: float = HERTZ_EXPONENT,
    damping: float = 0.0,
) -> float:
    """Return the Hunt-Crossley normal force at one contact, in newtons.

    The force is zero whenever the indentation is not positive, and is floored at
    zero during retraction so that the damping term can never pull the object
    into the finger.
    """
    if not detect_contact(indentation):
        return 0.0
    elastic = stiffness * float(indentation**exponent)
    force = elastic * (1.0 + damping * indentation_rate)
    return max(force, 0.0)


def equilibrium_force(
    span: float,
    width: float,
    pad: FingertipPad,
    object_stiffness: float,
) -> float:
    """Return the static contact force at a held grasp span, in newtons.

    Equilibrium means the indentation rate is zero, so only the elastic term of
    the contact law survives. This is the quantity that separates a rigid object
    from a deformable one: at one commanded span the two reach different forces.
    """
    indentation = penetration(span, width)
    stiffness = effective_stiffness(pad, object_stiffness)
    return contact_force(indentation, 0.0, stiffness, pad.exponent, damping=0.0)


def indentation_for_force(force: float, stiffness: float, exponent: float) -> float:
    """Invert the elastic contact law, returning the indentation in metres."""
    if force <= 0.0:
        return 0.0
    return float((force / stiffness) ** (1.0 / exponent))


def friction_capacity(grip_force: float, friction: float, contacts: int) -> float:
    """Return the tangential load the grasp can carry, in newtons.

    Each expected contact of the grasp presses with the normal force
    ``grip_force`` and contributes ``friction * grip_force`` of Coulomb
    capacity, so the taxonomy's contact expectation sets the multiplier. This is
    the simplification that lets a five contact wrap hold more than a two contact
    pinch at the same commanded force.
    """
    if contacts < 1:
        raise ValueError(f"a grasp needs at least one contact, got {contacts}")
    return max(grip_force, 0.0) * friction * contacts


def required_grip_force(
    mass: float, friction: float, contacts: int, safety_factor: float = 1.0
) -> float:
    """Return the normal force per contact needed to hold ``mass`` against gravity."""
    if friction <= 0.0:
        raise ValueError(f"friction coefficient must be positive, got {friction}")
    if contacts < 1:
        raise ValueError(f"a grasp needs at least one contact, got {contacts}")
    return safety_factor * mass * GRAVITY / (friction * contacts)
