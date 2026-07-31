"""Property and invariant tests for the coupled hand kinematics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hand_controller.model import (
    FINGERS,
    Digit,
    HandConfiguration,
    JointCoupling,
    default_hand,
    digit_points,
    fingertip,
    fingertip_positions,
    joint_angles,
    limit_violations,
    within_limits,
)

HAND = default_hand()


def _configuration(
    flexion: float = 0.0,
    abduction: float = 0.0,
    thumb_flexion: float = 0.0,
    thumb_opposition: float = 0.0,
) -> HandConfiguration:
    return HandConfiguration(
        finger_flexion=(flexion, flexion, flexion, flexion),
        finger_abduction=(abduction, abduction, abduction, abduction),
        thumb_flexion=thumb_flexion,
        thumb_opposition=thumb_opposition,
    )


def test_extended_finger_lies_along_the_distal_axis() -> None:
    """With every joint at zero the finger is a straight line down the palm x axis."""
    config = _configuration()
    geometry = HAND.finger(Digit.INDEX)
    expected = np.array(geometry.base, dtype=np.float64) + np.array(
        [geometry.chain_length, 0.0, 0.0]
    )
    np.testing.assert_allclose(fingertip(HAND, config, Digit.INDEX), expected, atol=1e-15)


def test_forward_kinematics_matches_a_hand_computed_configuration() -> None:
    """Check the index fingertip against a chain summed by hand.

    At a metacarpophalangeal angle of ninety degrees the coupling gives a
    proximal interphalangeal angle of ninety degrees and a distal
    interphalangeal angle of sixty degrees, so the cumulative angles are 90, 180
    and 240 degrees. The tolerance is 1e-15 m, which is the rounding of a sum of
    three terms of order 0.1 m in double precision.
    """
    config = _configuration(flexion=math.pi / 2.0)
    geometry = HAND.finger(Digit.INDEX)
    lengths = geometry.link_lengths
    cumulative = (math.pi / 2.0, math.pi, math.pi + math.pi / 3.0)
    expected = np.array(geometry.base, dtype=np.float64)
    for length, angle in zip(lengths, cumulative, strict=True):
        expected = expected + length * np.array([math.cos(angle), 0.0, -math.sin(angle)])
    np.testing.assert_allclose(fingertip(HAND, config, Digit.INDEX), expected, atol=1e-15)


@pytest.mark.parametrize("digit", FINGERS)
@pytest.mark.parametrize("flexion", [0.0, 0.2, 0.7, 1.1, 1.5])
def test_segment_lengths_are_preserved(digit: Digit, flexion: float) -> None:
    """Forward kinematics moves joints without stretching any phalanx."""
    config = _configuration(flexion=flexion, abduction=0.1)
    points = digit_points(HAND, config, digit)
    measured = np.linalg.norm(np.diff(points, axis=0), axis=1)
    expected = np.array(HAND.finger(digit).link_lengths)
    np.testing.assert_allclose(measured, expected, atol=1e-15)


@pytest.mark.parametrize("flexion", [0.0, 0.4, 0.9, 1.4])
def test_thumb_segment_lengths_are_preserved(flexion: float) -> None:
    """The same invariant holds for the thumb at any opposition."""
    config = _configuration(thumb_flexion=flexion, thumb_opposition=0.8)
    points = digit_points(HAND, config, Digit.THUMB)
    measured = np.linalg.norm(np.diff(points, axis=0), axis=1)
    np.testing.assert_allclose(measured, np.array(HAND.thumb.link_lengths), atol=1e-15)


@pytest.mark.parametrize("abduction", [-0.25, -0.1, 0.0, 0.1, 0.25])
def test_abduction_rotates_the_finger_plane_about_the_dorsal_axis(abduction: float) -> None:
    """Abduction is a pure rotation of the chain about the base, in the palm plane."""
    flat = fingertip(HAND, _configuration(flexion=0.8), Digit.MIDDLE)
    spread = fingertip(HAND, _configuration(flexion=0.8, abduction=abduction), Digit.MIDDLE)
    base = np.array(HAND.finger(Digit.MIDDLE).base, dtype=np.float64)
    cos, sin = math.cos(abduction), math.sin(abduction)
    rotation = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(spread - base, rotation @ (flat - base), atol=1e-15)


@pytest.mark.parametrize("flexion", np.linspace(0.0, 1.5, 31))
@pytest.mark.parametrize("digit", FINGERS)
def test_finger_coupling_holds_the_declared_ratio_exactly(digit: Digit, flexion: float) -> None:
    """The two distal joints are exactly the declared multiples of the base joint."""
    config = _configuration(flexion=float(flexion))
    coupling = HAND.finger(digit).coupling
    base, middle, distal = joint_angles(HAND, config, digit)
    assert base == float(flexion)
    assert middle == coupling.middle_per_base * base
    assert distal == coupling.distal_per_middle * middle


def test_the_declared_coupling_is_the_two_thirds_relation() -> None:
    """The distal interphalangeal joint follows two thirds of the proximal one."""
    for digit in FINGERS:
        assert HAND.finger(digit).coupling.distal_per_middle == 2.0 / 3.0
    assert HAND.thumb.coupling.distal_per_middle == 2.0 / 3.0


def test_coupling_expands_one_command_into_three_angles() -> None:
    """A coupling is a pure function of its base angle."""
    coupling = JointCoupling(middle_per_base=1.2, distal_per_middle=0.5)
    assert coupling.angles(0.5) == (0.5, 0.6, 0.3)


def test_thumb_opposition_carries_the_tip_towards_the_palm() -> None:
    """Opposition rotates the thumb flexion plane from across the palm to palmar."""
    across = fingertip(HAND, _configuration(thumb_flexion=0.7, thumb_opposition=0.0), Digit.THUMB)
    palmar = fingertip(
        HAND, _configuration(thumb_flexion=0.7, thumb_opposition=math.pi / 2.0), Digit.THUMB
    )
    assert palmar[2] < across[2]
    assert palmar[1] > across[1]


def test_fingertip_positions_are_ordered_thumb_first() -> None:
    """The array indexing follows the digit enumeration."""
    config = _configuration(flexion=0.5, thumb_flexion=0.4, thumb_opposition=0.9)
    tips = fingertip_positions(HAND, config)
    assert tips.shape == (5, 3)
    for digit in Digit:
        np.testing.assert_array_equal(tips[int(digit)], fingertip(HAND, config, digit))


def test_limits_accept_the_neutral_posture() -> None:
    assert within_limits(HAND, _configuration())


def test_limits_reject_hyperextension() -> None:
    """A negative flexion angle is outside every finger range and is named."""
    messages = limit_violations(HAND, _configuration(flexion=-0.2))
    assert len(messages) == 12
    assert any("index mcp" in message for message in messages)


def test_limits_reject_excess_abduction() -> None:
    messages = limit_violations(HAND, _configuration(abduction=1.0))
    assert len(messages) == 4
    assert all("abduction" in message for message in messages)


def test_limits_reject_excess_thumb_opposition() -> None:
    messages = limit_violations(HAND, _configuration(thumb_opposition=2.0))
    assert len(messages) == 1
    assert "thumb opposition" in messages[0]


def test_blend_interpolates_and_clamps() -> None:
    start = _configuration(flexion=0.0, thumb_opposition=0.2)
    end = _configuration(flexion=1.0, thumb_opposition=0.8)
    middle = start.blend(end, 0.25)
    assert middle.finger_flexion == (0.25, 0.25, 0.25, 0.25)
    assert middle.thumb_opposition == pytest.approx(0.35)
    assert start.blend(end, -3.0) == start
    assert start.blend(end, 4.0) == end


def test_the_thumb_is_not_one_of_the_fingers() -> None:
    with pytest.raises(ValueError, match="thumb is not one of the four fingers"):
        HAND.finger(Digit.THUMB)
