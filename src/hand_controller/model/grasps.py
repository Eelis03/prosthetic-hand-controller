"""The grasp taxonomy, held as data, and the geometry it implies.

The six grasps below are taken from the GRASP taxonomy of Feix, Romero,
Schmiedmayer, Dollar and Kragic (2016), which classifies thirty three human
grasps by opposition type, thumb position and the power to precision axis of
Cutkosky (1989). The subset chosen here is the set that multi-articulating
prosthetic hands actually offer (Belter et al., 2013): it spans palm, pad and
side opposition, both thumb positions, and all three power classes.

Grasps with no opposition, such as the Fixed Hook that carries a bag handle in
the crook of the fingers, are deliberately left out. They hold a load without
squeezing it, so a grip force controller has nothing to regulate and the span
between opposing surfaces is not defined.

Each entry records the taxonomy number and name, the classification, the open
and closed joint postures, and the digits expected to contact the object. The
contact expectation is what turns a posture into a mechanical prediction: it
fixes how many surfaces share the tangential load.

The thumb angles of each closed posture are the values that bring the opposing
surfaces of this particular hand together. They were selected by sweeping the
thumb over its two degrees of freedom and taking the configuration that closes
the span monotonically and leaves the pads in contact; the taxonomy fixes which
surfaces oppose, and the geometry fixes the angles that achieve it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from hand_controller.model.anatomy import FINGERS, Digit, HandModel
from hand_controller.model.kinematics import (
    HandConfiguration,
    digit_points,
    fingertip,
    limit_violations,
)

__all__ = [
    "GRASP_NAMES",
    "GRASP_TAXONOMY",
    "SPAN_SOLVER_TOLERANCE",
    "GraspClass",
    "GraspDefinition",
    "OppositionType",
    "ThumbPosition",
    "closure_at_span",
    "grasp",
    "opposition_span",
    "reachability",
    "span_profile",
    "span_range",
    "taxonomy_digits",
]

SPAN_SOLVER_TOLERANCE: Final[float] = 1e-12
"""Absolute tolerance requested from the closure solver, in units of closure."""


class OppositionType(StrEnum):
    """Direction in which the hand applies force, as defined by Feix et al. (2016)."""

    PALM = "palm"
    PAD = "pad"
    SIDE = "side"
    NONE = "none"


class ThumbPosition(StrEnum):
    """Whether the thumb is carried away from or against the palm."""

    ABDUCTED = "abducted"
    ADDUCTED = "adducted"


class GraspClass(StrEnum):
    """Power, intermediate or precision, following Cutkosky (1989)."""

    POWER = "power"
    INTERMEDIATE = "intermediate"
    PRECISION = "precision"


def _radians(values: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    converted = tuple(math.radians(value) for value in values)
    return (converted[0], converted[1], converted[2], converted[3])


def _posture(
    flexion: tuple[float, float, float, float],
    abduction: tuple[float, float, float, float],
    thumb_flexion: float,
    thumb_opposition: float,
) -> HandConfiguration:
    """Build a configuration from angles written in degrees."""
    return HandConfiguration(
        finger_flexion=_radians(flexion),
        finger_abduction=_radians(abduction),
        thumb_flexion=math.radians(thumb_flexion),
        thumb_opposition=math.radians(thumb_opposition),
    )


@dataclass(frozen=True, slots=True)
class GraspDefinition:
    """One grasp of the taxonomy, with its postures and contact expectation."""

    name: str
    taxonomy_index: int
    taxonomy_name: str
    opposition: OppositionType
    thumb: ThumbPosition
    grasp_class: GraspClass
    open_posture: HandConfiguration
    closed_posture: HandConfiguration
    contact_digits: tuple[Digit, ...]
    palm_contact: bool
    description: str

    @property
    def load_bearing_contacts(self) -> int:
        """Number of surfaces expected to share the tangential load."""
        return len(self.contact_digits) + int(self.palm_contact)

    @property
    def contact_fingers(self) -> tuple[Digit, ...]:
        """The expected contacting digits other than the thumb."""
        return tuple(digit for digit in self.contact_digits if digit is not Digit.THUMB)

    def posture(self, closure: float) -> HandConfiguration:
        """Return the posture at ``closure``, zero fully open and one fully closed."""
        return self.open_posture.blend(self.closed_posture, closure)


_TAXONOMY: Final[tuple[GraspDefinition, ...]] = (
    GraspDefinition(
        name="medium_wrap",
        taxonomy_index=3,
        taxonomy_name="Medium Wrap",
        opposition=OppositionType.PALM,
        thumb=ThumbPosition.ABDUCTED,
        grasp_class=GraspClass.POWER,
        open_posture=_posture((5.0, 5.0, 5.0, 5.0), (5.0, 2.0, -2.0, -5.0), 0.0, 32.0),
        closed_posture=_posture((78.0, 80.0, 78.0, 74.0), (5.0, 2.0, -2.0, -5.0), 26.0, 32.0),
        contact_digits=(Digit.THUMB, Digit.INDEX, Digit.MIDDLE, Digit.RING, Digit.LITTLE),
        palm_contact=True,
        description="Cylinder wrapped by all four fingers against the palm, thumb opposed.",
    ),
    GraspDefinition(
        name="power_sphere",
        taxonomy_index=11,
        taxonomy_name="Power Sphere",
        opposition=OppositionType.PALM,
        thumb=ThumbPosition.ABDUCTED,
        grasp_class=GraspClass.POWER,
        open_posture=_posture((5.0, 5.0, 5.0, 5.0), (14.0, 5.0, -5.0, -14.0), 0.0, 42.0),
        closed_posture=_posture((58.0, 60.0, 58.0, 55.0), (14.0, 5.0, -5.0, -14.0), 39.0, 42.0),
        contact_digits=(Digit.THUMB, Digit.INDEX, Digit.MIDDLE, Digit.RING, Digit.LITTLE),
        palm_contact=False,
        description="Spread fingers curled around a sphere, thumb abducted and opposed.",
    ),
    GraspDefinition(
        name="prismatic_four",
        taxonomy_index=6,
        taxonomy_name="Prismatic Four Finger",
        opposition=OppositionType.PAD,
        thumb=ThumbPosition.ABDUCTED,
        grasp_class=GraspClass.PRECISION,
        open_posture=_posture((5.0, 5.0, 5.0, 5.0), (2.0, 1.0, -1.0, -2.0), 0.0, 42.0),
        closed_posture=_posture((50.0, 48.0, 46.0, 44.0), (2.0, 1.0, -1.0, -2.0), 40.0, 42.0),
        contact_digits=(Digit.THUMB, Digit.INDEX, Digit.MIDDLE, Digit.RING, Digit.LITTLE),
        palm_contact=False,
        description="Thumb opposed to all four fingertips on a flat object held clear of the palm.",
    ),
    GraspDefinition(
        name="palmar_pinch",
        taxonomy_index=9,
        taxonomy_name="Palmar Pinch",
        opposition=OppositionType.PAD,
        thumb=ThumbPosition.ABDUCTED,
        grasp_class=GraspClass.PRECISION,
        open_posture=_posture((5.0, 12.0, 45.0, 50.0), (0.0, 0.0, 0.0, 0.0), 0.0, 56.0),
        closed_posture=_posture((52.0, 12.0, 45.0, 50.0), (0.0, 0.0, 0.0, 0.0), 28.0, 56.0),
        contact_digits=(Digit.THUMB, Digit.INDEX),
        palm_contact=False,
        description="Two digit pad to pad pinch, the ulnar fingers curled clear.",
    ),
    GraspDefinition(
        name="tripod",
        taxonomy_index=14,
        taxonomy_name="Tripod",
        opposition=OppositionType.PAD,
        thumb=ThumbPosition.ABDUCTED,
        grasp_class=GraspClass.PRECISION,
        open_posture=_posture((5.0, 5.0, 48.0, 52.0), (6.0, -2.0, 0.0, 0.0), 0.0, 50.0),
        closed_posture=_posture((48.0, 46.0, 48.0, 52.0), (6.0, -2.0, 0.0, 0.0), 36.0, 50.0),
        contact_digits=(Digit.THUMB, Digit.INDEX, Digit.MIDDLE),
        palm_contact=False,
        description="Thumb opposed to the index and middle pads, as when holding a pen.",
    ),
    GraspDefinition(
        name="lateral",
        taxonomy_index=16,
        taxonomy_name="Lateral",
        opposition=OppositionType.SIDE,
        thumb=ThumbPosition.ADDUCTED,
        grasp_class=GraspClass.INTERMEDIATE,
        open_posture=_posture((60.0, 62.0, 62.0, 60.0), (0.0, 0.0, 0.0, 0.0), 0.0, 10.0),
        closed_posture=_posture((60.0, 62.0, 62.0, 60.0), (0.0, 0.0, 0.0, 0.0), 18.0, 10.0),
        contact_digits=(Digit.THUMB, Digit.INDEX),
        palm_contact=False,
        description="Thumb pad pressed onto the radial side of the index, as when turning a key.",
    ),
)

GRASP_TAXONOMY: Final[tuple[GraspDefinition, ...]] = _TAXONOMY
GRASP_NAMES: Final[tuple[str, ...]] = tuple(entry.name for entry in _TAXONOMY)


def grasp(name: str) -> GraspDefinition:
    """Look up one grasp by name."""
    for entry in _TAXONOMY:
        if entry.name == name:
            return entry
    raise KeyError(f"unknown grasp: {name!r}; known grasps are {', '.join(GRASP_NAMES)}")


def _finger_centroid(
    hand: HandModel, config: HandConfiguration, digits: tuple[Digit, ...]
) -> NDArray[np.float64]:
    tips = np.array([fingertip(hand, config, digit) for digit in digits], dtype=np.float64)
    centroid: NDArray[np.float64] = tips.mean(axis=0)
    return centroid


def opposition_span(hand: HandModel, definition: GraspDefinition, closure: float) -> float:
    """Return the width of the opening the grasp closes on, in metres.

    The span is the separation of the two *surfaces* that oppose each other. The
    taxonomy fixes which surfaces those are through the opposition type:

    * palm and pad opposition, between the thumb pad and the pads of the
      contacting fingertips, taken at their centroid;
    * side opposition, between the thumb pad and the radial face of the index
      proximal phalanx.

    Forward kinematics returns joint centres, so the pad radii of the two
    surfaces are subtracted from the centre to centre distance. The result is
    negative when the digits would overlap, which is what a hand closed on
    nothing does.

    An object of width ``w`` is enclosed by the grasp once the span falls below
    ``w``, and the difference is what the compliant contact model turns into a
    force.
    """
    if definition.opposition is OppositionType.NONE:
        raise ValueError(
            f"{definition.name} has no opposition, so its span is undefined; "
            "the force controller does not model grasps that carry load without a squeeze"
        )

    config = definition.posture(closure)
    thumb_tip = fingertip(hand, config, Digit.THUMB)
    if definition.opposition is OppositionType.SIDE:
        index = digit_points(hand, config, Digit.INDEX)
        target = 0.5 * (index[0] + index[1])
        clearance = hand.thumb.pad_radius + hand.finger(Digit.INDEX).proximal_radius
    else:
        target = _finger_centroid(hand, config, definition.contact_fingers)
        clearance = hand.thumb.pad_radius + hand.finger(Digit.INDEX).pad_radius
    return float(np.linalg.norm(thumb_tip - target)) - clearance


def span_range(hand: HandModel, definition: GraspDefinition) -> tuple[float, float]:
    """Return the closed and open spans of a grasp, in metres."""
    return (
        opposition_span(hand, definition, 1.0),
        opposition_span(hand, definition, 0.0),
    )


def span_profile(
    hand: HandModel, definition: GraspDefinition, samples: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return closure values and the span at each of them."""
    if samples < 2:
        raise ValueError(f"samples must be at least 2, got {samples}")
    closures = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    spans = np.array(
        [opposition_span(hand, definition, float(value)) for value in closures],
        dtype=np.float64,
    )
    return closures, spans


def closure_at_span(hand: HandModel, definition: GraspDefinition, span: float) -> float:
    """Return the closure at which the grasp span equals ``span``.

    The span is a nonlinear function of closure with no closed form inverse, so
    the value is bracketed on [0, 1] and refined by Brent's method to
    ``SPAN_SOLVER_TOLERANCE``. Raises ``ValueError`` when ``span`` lies outside
    the range the grasp can reach.
    """
    closed_span, open_span = span_range(hand, definition)
    if not closed_span <= span <= open_span:
        raise ValueError(
            f"span {span:.6f} m is outside the reach of {definition.name}: "
            f"[{closed_span:.6f}, {open_span:.6f}] m"
        )

    def residual(closure: float) -> float:
        return opposition_span(hand, definition, closure) - span

    root = brentq(residual, 0.0, 1.0, xtol=SPAN_SOLVER_TOLERANCE)
    return float(root)


def reachability(hand: HandModel, definition: GraspDefinition) -> tuple[str, ...]:
    """Return every joint limit violation over the closure sweep of a grasp.

    Both postures and the interpolation between them are checked, because a
    linear blend of two admissible postures is admissible only when every joint
    range is an interval, which the sweep verifies rather than assumes.
    """
    messages: list[str] = []
    for closure in np.linspace(0.0, 1.0, 21, dtype=np.float64):
        config = definition.posture(float(closure))
        for message in limit_violations(hand, config):
            messages.append(f"{definition.name} at closure {closure:.2f}: {message}")
    return tuple(messages)


def taxonomy_digits(definition: GraspDefinition) -> tuple[Digit, ...]:
    """Return the contacting digits of a grasp in canonical order."""
    order = (Digit.THUMB, *FINGERS)
    return tuple(digit for digit in order if digit in definition.contact_digits)
