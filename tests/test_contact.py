"""Property and invariant tests for the compliant contact model."""

from __future__ import annotations

import numpy as np
import pytest

from hand_controller.model import (
    GRAVITY,
    HERTZ_EXPONENT,
    FingertipPad,
    contact_force,
    default_pad,
    detect_contact,
    effective_stiffness,
    equilibrium_force,
    friction_capacity,
    indentation_for_force,
    penetration,
    required_grip_force,
)

PAD = default_pad()
RIGID = 5.0e7
DEFORMABLE = 3.0e3


@pytest.mark.parametrize("indentation", np.linspace(-0.01, 0.01, 2001))
def test_contact_fires_when_and_only_when_indentation_is_positive(indentation: float) -> None:
    """The detector is a strict sign test, checked densely across the sign change."""
    assert detect_contact(float(indentation)) == (indentation > 0.0)


def test_contact_does_not_fire_at_exactly_zero() -> None:
    """Touching without indenting is not contact."""
    assert detect_contact(0.0) is False
    assert detect_contact(5e-324) is True
    assert detect_contact(-5e-324) is False


@pytest.mark.parametrize("span", np.linspace(0.0, 0.12, 61))
def test_penetration_is_half_the_excess_width(span: float) -> None:
    width = 0.060
    assert penetration(float(span), width) == pytest.approx(0.5 * (width - span), abs=1e-18)


@pytest.mark.parametrize("indentation", np.linspace(-0.005, 0.0, 51))
def test_no_force_without_contact(indentation: float) -> None:
    stiffness = effective_stiffness(PAD, RIGID)
    assert contact_force(float(indentation), 1.0, stiffness) == 0.0


def test_contact_force_follows_the_hertzian_law() -> None:
    """A single elastic contact obeys f = k d^n with n = 3/2."""
    stiffness = 1.0e5
    for indentation in (1e-4, 5e-4, 2e-3):
        expected = stiffness * indentation**HERTZ_EXPONENT
        assert contact_force(indentation, 0.0, stiffness) == pytest.approx(expected, rel=1e-12)


def test_damping_cannot_pull_the_object_into_the_finger() -> None:
    """The Hunt-Crossley term is floored at zero during fast retraction."""
    stiffness = effective_stiffness(PAD, RIGID)
    assert contact_force(1e-3, -1e6, stiffness, damping=0.6) == 0.0
    assert contact_force(1e-3, 0.0, stiffness, damping=0.6) > 0.0


def test_series_stiffness_is_dominated_by_the_softer_body() -> None:
    """A rigid object leaves the pad in charge, a soft one takes charge itself."""
    rigid = effective_stiffness(PAD, RIGID)
    soft = effective_stiffness(PAD, DEFORMABLE)
    assert rigid == pytest.approx(PAD.stiffness, rel=0.05)
    assert soft == pytest.approx(DEFORMABLE, rel=0.15)
    assert soft < rigid
    assert rigid < PAD.stiffness
    assert soft < DEFORMABLE


def test_series_stiffness_of_two_equal_springs() -> None:
    """Two identical Hertzian contacts in series halve the indentation exponent scale."""
    pad = FingertipPad(stiffness=1.0e5, exponent=1.5)
    combined = effective_stiffness(pad, 1.0e5)
    assert combined == pytest.approx(1.0e5 * 2.0 ** (-1.5), rel=1e-12)


def test_a_deformable_object_reaches_a_different_equilibrium_than_a_rigid_one() -> None:
    """The same commanded span produces very different forces on the two bodies.

    This is the property that shows the compliance model is in the loop rather
    than decorative: with the span held two millimetres inside a 65 mm object,
    the rigid body carries more than an order of magnitude more force.
    """
    width = 0.065
    span = width - 0.004
    rigid = equilibrium_force(span, width, PAD, RIGID)
    soft = equilibrium_force(span, width, PAD, DEFORMABLE)
    assert rigid > 10.0 * soft
    assert soft > 0.0


@pytest.mark.parametrize("force", [0.5, 1.2, 2.6, 8.0])
def test_the_same_force_costs_more_travel_on_a_deformable_object(force: float) -> None:
    rigid = indentation_for_force(force, effective_stiffness(PAD, RIGID), PAD.exponent)
    soft = indentation_for_force(force, effective_stiffness(PAD, DEFORMABLE), PAD.exponent)
    assert soft > 5.0 * rigid


def test_indentation_inverts_the_elastic_law() -> None:
    stiffness = effective_stiffness(PAD, DEFORMABLE)
    for force in (0.1, 1.0, 5.0):
        indentation = indentation_for_force(force, stiffness, PAD.exponent)
        assert contact_force(indentation, 0.0, stiffness) == pytest.approx(force, rel=1e-12)
    assert indentation_for_force(-1.0, stiffness, PAD.exponent) == 0.0


def test_equilibrium_force_is_zero_when_the_hand_is_still_open() -> None:
    assert equilibrium_force(0.070, 0.060, PAD, RIGID) == 0.0


@pytest.mark.parametrize("contacts", [2, 3, 5, 6])
def test_friction_capacity_scales_with_the_contact_expectation(contacts: int) -> None:
    assert friction_capacity(2.0, 0.4, contacts) == pytest.approx(2.0 * 0.4 * contacts)


def test_friction_capacity_is_never_negative() -> None:
    assert friction_capacity(-5.0, 0.4, 2) == 0.0


def test_required_grip_force_matches_the_hand_computation() -> None:
    """A 0.42 kg object on six contacts at a friction coefficient of 0.45."""
    expected = 0.420 * GRAVITY / (0.45 * 6)
    assert required_grip_force(0.420, 0.45, 6) == pytest.approx(expected, rel=1e-15)


def test_required_force_and_capacity_are_consistent() -> None:
    """At the required force the capacity is exactly the weight."""
    mass, friction, contacts = 0.30, 0.35, 6
    required = required_grip_force(mass, friction, contacts)
    assert friction_capacity(required, friction, contacts) == pytest.approx(mass * GRAVITY)


@pytest.mark.parametrize(
    ("callable_", "arguments"),
    [
        (effective_stiffness, (PAD, 0.0)),
        (friction_capacity, (1.0, 0.4, 0)),
        (required_grip_force, (1.0, 0.0, 2)),
        (required_grip_force, (1.0, 0.4, 0)),
    ],
)
def test_invalid_arguments_are_rejected(callable_: object, arguments: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        callable_(*arguments)  # type: ignore[operator]


def test_pad_validates_itself() -> None:
    with pytest.raises(ValueError, match="stiffness must be positive"):
        FingertipPad(stiffness=0.0)
    with pytest.raises(ValueError, match="exponent must be positive"):
        FingertipPad(stiffness=1.0, exponent=0.0)
