"""Forward kinematics of the coupled, underactuated hand.

Every digit is a serial chain of three segments. A finger flexes in a plane
that is the palm's sagittal plane rotated about the dorsal axis by its abduction
angle. The thumb flexes in a plane that is first rotated about its own long axis
by the opposition angle and then carried radially by the fixed splay of the
metacarpal, so that opposition sweeps the thumb pad from the plane of the palm
towards the finger pads.

All functions here are pure: they take a hand and a configuration and return
positions or angles. Nothing in this module performs input or output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hand_controller.model.anatomy import FINGERS, Digit, HandModel

__all__ = [
    "HandConfiguration",
    "digit_points",
    "fingertip",
    "fingertip_positions",
    "joint_angles",
    "limit_violations",
    "within_limits",
]


@dataclass(frozen=True, slots=True)
class HandConfiguration:
    """The commanded pose of the hand.

    ``finger_flexion`` and ``finger_abduction`` are ordered index, middle, ring,
    little. Flexion drives all three joints of a finger through its coupling, so
    the ten numbers here expand to fifteen joint angles.
    """

    finger_flexion: tuple[float, float, float, float]
    finger_abduction: tuple[float, float, float, float]
    thumb_flexion: float
    thumb_opposition: float

    def blend(self, other: HandConfiguration, fraction: float) -> HandConfiguration:
        """Linearly interpolate towards ``other``, at ``fraction`` in [0, 1]."""
        alpha = min(max(fraction, 0.0), 1.0)

        def mix(first: float, second: float) -> float:
            return first + alpha * (second - first)

        flexion = tuple(
            mix(a, b) for a, b in zip(self.finger_flexion, other.finger_flexion, strict=True)
        )
        abduction = tuple(
            mix(a, b) for a, b in zip(self.finger_abduction, other.finger_abduction, strict=True)
        )
        return HandConfiguration(
            finger_flexion=(flexion[0], flexion[1], flexion[2], flexion[3]),
            finger_abduction=(abduction[0], abduction[1], abduction[2], abduction[3]),
            thumb_flexion=mix(self.thumb_flexion, other.thumb_flexion),
            thumb_opposition=mix(self.thumb_opposition, other.thumb_opposition),
        )


def joint_angles(hand: HandModel, config: HandConfiguration, digit: Digit) -> tuple[float, ...]:
    """Return the three flexion angles of ``digit``, expanded through its coupling."""
    if digit is Digit.THUMB:
        return hand.thumb.coupling.angles(config.thumb_flexion)
    geometry = hand.finger(digit)
    return geometry.coupling.angles(config.finger_flexion[int(digit) - 1])


def _rotation_z(angle: float) -> NDArray[np.float64]:
    cos, sin = math.cos(angle), math.sin(angle)
    return np.array(
        [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _rotation_x(angle: float) -> NDArray[np.float64]:
    cos, sin = math.cos(angle), math.sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]],
        dtype=np.float64,
    )


def _chain(
    base: tuple[float, float, float],
    orientation: NDArray[np.float64],
    lengths: tuple[float, float, float],
    angles: tuple[float, ...],
    flexion_axis: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Accumulate a three segment planar chain and map it into the palm frame.

    ``flexion_axis`` is the unit vector, in the local chain frame, that a segment
    moves towards as its cumulative flexion angle grows. The chain extends along
    the local ``x`` axis at zero flexion.
    """
    points = np.empty((4, 3), dtype=np.float64)
    points[0] = np.asarray(base, dtype=np.float64)
    forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    total = 0.0
    for index, (length, angle) in enumerate(zip(lengths, angles, strict=True)):
        total += angle
        local = math.cos(total) * forward + math.sin(total) * flexion_axis
        points[index + 1] = points[index] + length * (orientation @ local)
    return points


def digit_points(hand: HandModel, config: HandConfiguration, digit: Digit) -> NDArray[np.float64]:
    """Return the four joint centres of ``digit``, base first and fingertip last.

    For a finger the rows are the metacarpophalangeal, proximal interphalangeal
    and distal interphalangeal joint centres followed by the fingertip. For the
    thumb they are the carpometacarpal, metacarpophalangeal and interphalangeal
    joint centres followed by the thumb tip.
    """
    angles = joint_angles(hand, config, digit)
    if digit is Digit.THUMB:
        thumb = hand.thumb
        orientation = _rotation_z(thumb.radial_splay) @ _rotation_x(config.thumb_opposition)
        # At zero opposition the thumb flexes across the palm, towards the
        # ulnar side; at ninety degrees it flexes towards the palmar side.
        axis = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        return _chain(thumb.base, orientation, thumb.link_lengths, angles, axis)

    geometry = hand.finger(digit)
    abduction = config.finger_abduction[int(digit) - 1]
    orientation = _rotation_z(abduction)
    axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return _chain(geometry.base, orientation, geometry.link_lengths, angles, axis)


def fingertip(hand: HandModel, config: HandConfiguration, digit: Digit) -> NDArray[np.float64]:
    """Return the fingertip position of ``digit`` in the palm frame."""
    tip: NDArray[np.float64] = digit_points(hand, config, digit)[3]
    return tip


def fingertip_positions(hand: HandModel, config: HandConfiguration) -> NDArray[np.float64]:
    """Return a five by three array of fingertip positions, thumb in row zero."""
    tips = np.empty((5, 3), dtype=np.float64)
    for digit in Digit:
        tips[int(digit)] = fingertip(hand, config, digit)
    return tips


def limit_violations(hand: HandModel, config: HandConfiguration) -> tuple[str, ...]:
    """Return one message per joint angle that lies outside its range.

    A tolerance of one microradian absorbs the rounding of angles that were
    written down in degrees and converted once.
    """
    tolerance = 1e-6
    messages: list[str] = []

    for digit in FINGERS:
        geometry = hand.finger(digit)
        angles = joint_angles(hand, config, digit)
        names = ("mcp", "pip", "dip")
        for name, angle, allowed in zip(names, angles, geometry.joint_ranges, strict=True):
            if not allowed.contains(angle, tolerance=tolerance):
                messages.append(
                    f"{geometry.name} {name} is {angle:.6f} rad, "
                    f"outside [{allowed.lower:.6f}, {allowed.upper:.6f}]"
                )
        abduction = config.finger_abduction[int(digit) - 1]
        if not geometry.abduction_range.contains(abduction, tolerance=tolerance):
            messages.append(
                f"{geometry.name} abduction is {abduction:.6f} rad, "
                f"outside [{geometry.abduction_range.lower:.6f}, "
                f"{geometry.abduction_range.upper:.6f}]"
            )

    thumb = hand.thumb
    thumb_angles = joint_angles(hand, config, Digit.THUMB)
    for name, angle, allowed in zip(
        ("cmc", "mcp", "ip"), thumb_angles, thumb.joint_ranges, strict=True
    ):
        if not allowed.contains(angle, tolerance=tolerance):
            messages.append(
                f"thumb {name} is {angle:.6f} rad, "
                f"outside [{allowed.lower:.6f}, {allowed.upper:.6f}]"
            )
    if not thumb.opposition_range.contains(config.thumb_opposition, tolerance=tolerance):
        messages.append(
            f"thumb opposition is {config.thumb_opposition:.6f} rad, "
            f"outside [{thumb.opposition_range.lower:.6f}, {thumb.opposition_range.upper:.6f}]"
        )
    return tuple(messages)


def within_limits(hand: HandModel, config: HandConfiguration) -> bool:
    """True when every joint angle implied by ``config`` lies inside its range."""
    return not limit_violations(hand, config)
