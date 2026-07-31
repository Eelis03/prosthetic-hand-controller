"""Geometry and joint limits of the hand model.

Link lengths follow the anthropometric hand data of Buchholz, Armstrong and
Goldstein (1992). Joint ranges are the flexion arcs commonly quoted for the
metacarpophalangeal, proximal interphalangeal and distal interphalangeal joints,
truncated to the arcs that a multi-articulating prosthetic hand actually
provides (Belter et al., 2013).

Frame convention, right hand, origin at the centre of the wrist:

* ``x`` points distally, from the wrist towards the fingertips,
* ``y`` points radially, towards the thumb side,
* ``z`` points dorsally, out of the back of the hand.

Flexion therefore moves a fingertip towards negative ``z``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Final

__all__ = [
    "FINGERS",
    "Digit",
    "FingerGeometry",
    "HandModel",
    "JointCoupling",
    "JointRange",
    "ThumbGeometry",
    "default_hand",
]


class Digit(IntEnum):
    """Index of one digit of the hand, ordered thumb first."""

    THUMB = 0
    INDEX = 1
    MIDDLE = 2
    RING = 3
    LITTLE = 4


FINGERS: Final[tuple[Digit, ...]] = (Digit.INDEX, Digit.MIDDLE, Digit.RING, Digit.LITTLE)
"""The four non-thumb digits, in radial to ulnar order."""


@dataclass(frozen=True, slots=True)
class JointRange:
    """Closed interval of admissible angles for one joint, in radians."""

    lower: float
    upper: float

    def contains(self, value: float, *, tolerance: float = 0.0) -> bool:
        """True when ``value`` lies inside the range, widened by ``tolerance``."""
        return self.lower - tolerance <= value <= self.upper + tolerance

    def clamp(self, value: float) -> float:
        """Return ``value`` restricted to the range."""
        return min(max(value, self.lower), self.upper)


@dataclass(frozen=True, slots=True)
class JointCoupling:
    """Fixed ratios that drive the two distal joints of a digit from its base joint.

    An underactuated prosthetic finger carries one actuator and transmits its
    motion to three joints through a linkage or a tendon, so the joint angles are
    not independent (Birglen et al., 2008). For a finger the base joint is the
    metacarpophalangeal joint, the middle joint is the proximal interphalangeal
    joint, and the distal joint is the distal interphalangeal joint. For the
    thumb they are the carpometacarpal flexion, metacarpophalangeal and
    interphalangeal joints.

    ``distal_per_middle`` defaults to the two thirds relation between the distal
    and proximal interphalangeal joints used in hand models since Rijpkema and
    Girard (1991).
    """

    middle_per_base: float
    distal_per_middle: float

    def angles(self, base: float) -> tuple[float, float, float]:
        """Return the base, middle and distal joint angles driven by ``base``."""
        middle = self.middle_per_base * base
        distal = self.distal_per_middle * middle
        return base, middle, distal


@dataclass(frozen=True, slots=True)
class FingerGeometry:
    """Link lengths, base position, joint limits and coupling of one finger."""

    name: str
    base: tuple[float, float, float]
    proximal_length: float
    middle_length: float
    distal_length: float
    pad_radius: float
    proximal_radius: float
    mcp_range: JointRange
    pip_range: JointRange
    dip_range: JointRange
    abduction_range: JointRange
    coupling: JointCoupling

    @property
    def chain_length(self) -> float:
        """Total length of the three phalanges."""
        return self.proximal_length + self.middle_length + self.distal_length

    @property
    def link_lengths(self) -> tuple[float, float, float]:
        """The three phalanx lengths, proximal to distal."""
        return (self.proximal_length, self.middle_length, self.distal_length)

    @property
    def joint_ranges(self) -> tuple[JointRange, JointRange, JointRange]:
        """The three flexion ranges, proximal to distal."""
        return (self.mcp_range, self.pip_range, self.dip_range)


@dataclass(frozen=True, slots=True)
class ThumbGeometry:
    """Link lengths, base pose, joint limits and coupling of the thumb.

    ``radial_splay`` is the fixed rotation of the thumb chain about the dorsal
    axis, which carries the thumb away from the index finger in the plane of the
    palm. ``opposition`` is the free rotation of the thumb flexion plane about
    the thumb's own long axis: at zero the thumb is adducted and flexes across
    the palm, and at ninety degrees it is fully abducted and flexes towards the
    palmar side, where its pad faces the finger pads.
    """

    name: str
    base: tuple[float, float, float]
    metacarpal_length: float
    proximal_length: float
    distal_length: float
    pad_radius: float
    cmc_range: JointRange
    mcp_range: JointRange
    ip_range: JointRange
    opposition_range: JointRange
    radial_splay: float
    coupling: JointCoupling

    @property
    def chain_length(self) -> float:
        """Total length of the metacarpal and the two phalanges."""
        return self.metacarpal_length + self.proximal_length + self.distal_length

    @property
    def link_lengths(self) -> tuple[float, float, float]:
        """The three segment lengths, proximal to distal."""
        return (self.metacarpal_length, self.proximal_length, self.distal_length)

    @property
    def joint_ranges(self) -> tuple[JointRange, JointRange, JointRange]:
        """The three flexion ranges, proximal to distal."""
        return (self.cmc_range, self.mcp_range, self.ip_range)


@dataclass(frozen=True, slots=True)
class HandModel:
    """A complete hand: four fingers in radial to ulnar order plus a thumb."""

    fingers: tuple[FingerGeometry, FingerGeometry, FingerGeometry, FingerGeometry]
    thumb: ThumbGeometry

    def finger(self, digit: Digit) -> FingerGeometry:
        """Return the geometry of one non-thumb digit."""
        if digit is Digit.THUMB:
            raise ValueError("the thumb is not one of the four fingers")
        return self.fingers[int(digit) - 1]

    @property
    def digit_count(self) -> int:
        """Number of digits, thumb included."""
        return len(self.fingers) + 1


_FINGER_MCP: Final[JointRange] = JointRange(0.0, 1.5708)
_FINGER_PIP: Final[JointRange] = JointRange(0.0, 1.7453)
_FINGER_DIP: Final[JointRange] = JointRange(0.0, 1.3963)
_FINGER_ABDUCTION: Final[JointRange] = JointRange(-0.2618, 0.2618)

# One actuator per finger drives the proximal interphalangeal joint at the same
# rate as the metacarpophalangeal joint, and the distal interphalangeal joint at
# two thirds of that rate.
_FINGER_COUPLING: Final[JointCoupling] = JointCoupling(
    middle_per_base=1.0, distal_per_middle=2.0 / 3.0
)
_THUMB_COUPLING: Final[JointCoupling] = JointCoupling(
    middle_per_base=0.8, distal_per_middle=2.0 / 3.0
)


def _finger(
    name: str,
    base: tuple[float, float, float],
    lengths: tuple[float, float, float],
) -> FingerGeometry:
    return FingerGeometry(
        name=name,
        base=base,
        proximal_length=lengths[0],
        middle_length=lengths[1],
        distal_length=lengths[2],
        pad_radius=0.0090,
        proximal_radius=0.0100,
        mcp_range=_FINGER_MCP,
        pip_range=_FINGER_PIP,
        dip_range=_FINGER_DIP,
        abduction_range=_FINGER_ABDUCTION,
        coupling=_FINGER_COUPLING,
    )


def default_hand() -> HandModel:
    """Return the reference hand, sized from Buchholz et al. (1992) male means.

    Pad radii are the half thickness of the soft covering: nine millimetres at a
    fingertip, ten at the thumb tip, and ten across the proximal phalanx of the
    index, which is the surface a lateral grasp presses against. They convert the
    joint centre distances produced by forward kinematics into the surface
    separation an object actually occupies.
    """
    index = _finger("index", (0.0900, 0.0250, 0.0), (0.0398, 0.0224, 0.0158))
    middle = _finger("middle", (0.0950, 0.0050, 0.0), (0.0446, 0.0263, 0.0174))
    ring = _finger("ring", (0.0900, -0.0150, 0.0), (0.0414, 0.0257, 0.0173))
    little = _finger("little", (0.0830, -0.0350, 0.0), (0.0327, 0.0187, 0.0158))
    thumb = ThumbGeometry(
        name="thumb",
        base=(0.0200, 0.0400, -0.0100),
        metacarpal_length=0.0462,
        proximal_length=0.0316,
        distal_length=0.0211,
        pad_radius=0.0100,
        cmc_range=JointRange(0.0, 1.0472),
        mcp_range=JointRange(0.0, 0.9599),
        ip_range=JointRange(0.0, 1.3963),
        opposition_range=JointRange(0.0, 1.5708),
        radial_splay=0.3000,
        coupling=_THUMB_COUPLING,
    )
    return HandModel(fingers=(index, middle, ring, little), thumb=thumb)
