"""Property and invariant tests for grip force regulation.

Convergence is checked against the real plant, that is the grasp geometry and
the compliant contact model, rather than against a linear stand in, because the
point of the loop is that it works across a plant gain that varies by two orders
of magnitude between a glass and a foam cup.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hand_controller.algorithm import ForceConfig, ProportionalForceRegulator, SlipResponseConfig
from hand_controller.algorithm.protocols import GripForceRegulator
from hand_controller.model import (
    closure_at_span,
    contact_force,
    default_hand,
    default_pad,
    effective_stiffness,
    grasp,
    opposition_span,
    penetration,
)

HAND = default_hand()
GRASP = grasp("medium_wrap")
PAD = default_pad()
WIDTH = 0.065
DT = 0.001
RIGID = 5.0e7
DEFORMABLE = 3.0e3


def _hold(
    object_stiffness: float,
    target: float,
    config: ForceConfig | None = None,
    steps: int = 3000,
) -> tuple[float, float, float]:
    """Close on an object until the loop settles, returning force, command and closure."""
    settings = config if config is not None else ForceConfig()
    regulator = ProportionalForceRegulator(settings)
    stiffness = effective_stiffness(PAD, object_stiffness)
    closure = closure_at_span(HAND, GRASP, WIDTH)
    regulator.demand(target)
    force = 0.0
    previous = penetration(opposition_span(HAND, GRASP, closure), WIDTH)
    for _ in range(steps):
        rate = regulator.update(force, DT)
        closure = min(max(closure + rate * DT, 0.0), 1.0)
        indentation = penetration(opposition_span(HAND, GRASP, closure), WIDTH)
        force = contact_force(
            indentation, (indentation - previous) / DT, stiffness, PAD.exponent, 0.6
        )
        previous = indentation
    return force, regulator.commanded_force, closure


def test_the_regulator_satisfies_the_protocol() -> None:
    regulator: GripForceRegulator = ProportionalForceRegulator(ForceConfig())
    regulator.demand(1.0)
    assert regulator.target_force == 1.0
    assert regulator.update(0.0, DT) > 0.0
    regulator.reset()
    assert regulator.commanded_force == 0.0


@pytest.mark.parametrize("target", [0.6, 1.2, 2.6, 6.0])
def test_grip_force_converges_with_no_steady_state_error_on_a_rigid_object(
    target: float,
) -> None:
    """The measured force reaches the commanded force and stays there.

    The tolerance is the controller's own convergence band, ``ForceConfig.tolerance``,
    which is the scale at which the loop declares the grip established. The
    second assertion shows the realised error is two orders of magnitude inside
    that band, which is what a proportional law on an integrating plant gives:
    the closure is the integrator, so no integral term is needed and none is used.
    """
    config = ForceConfig()
    force, command, _ = _hold(RIGID, target, config)
    assert command == pytest.approx(target, abs=config.tolerance)
    assert abs(force - command) <= config.tolerance
    assert abs(force - command) <= 0.01 * config.tolerance


@pytest.mark.parametrize("target", [0.6, 1.2, 2.6])
def test_grip_force_converges_on_a_deformable_object_as_well(target: float) -> None:
    """The online plant gain estimate keeps the loop working across both extremes."""
    config = ForceConfig()
    force, command, _ = _hold(DEFORMABLE, target, config)
    assert abs(force - command) <= config.tolerance


def test_a_deformable_object_needs_far_more_travel_for_the_same_force() -> None:
    """Same command, same loop, very different closure: the compliance is in the loop."""
    contact_closure = closure_at_span(HAND, GRASP, WIDTH)
    _, _, rigid_closure = _hold(RIGID, 2.6)
    _, _, soft_closure = _hold(DEFORMABLE, 2.6)
    rigid_travel = rigid_closure - contact_closure
    soft_travel = soft_closure - contact_closure
    assert soft_travel > 5.0 * rigid_travel


def test_the_loop_does_not_overshoot_the_ramped_demand() -> None:
    """A proportional law on an integrating plant approaches its demand from below."""
    config = ForceConfig()
    regulator = ProportionalForceRegulator(config)
    stiffness = effective_stiffness(PAD, RIGID)
    closure = closure_at_span(HAND, GRASP, WIDTH)
    regulator.demand(2.6)
    force = 0.0
    previous = penetration(opposition_span(HAND, GRASP, closure), WIDTH)
    peak = 0.0
    for _ in range(2000):
        rate = regulator.update(force, DT)
        closure = min(max(closure + rate * DT, 0.0), 1.0)
        indentation = penetration(opposition_span(HAND, GRASP, closure), WIDTH)
        force = contact_force(
            indentation, (indentation - previous) / DT, stiffness, PAD.exponent, 0.6
        )
        previous = indentation
        peak = max(peak, force)
    assert peak <= 2.6 + config.tolerance


@pytest.mark.parametrize(
    "demand", [1e3, 1e9, 1e300, math.inf, -math.inf, math.nan, -1.0, -1e9, 0.0]
)
def test_the_commanded_force_never_leaves_the_safety_interval(demand: float) -> None:
    """Adversarial demands are clamped, not passed through."""
    config = ForceConfig()
    regulator = ProportionalForceRegulator(config)
    regulator.demand(demand)
    assert 0.0 <= regulator.target_force <= config.safety_limit
    for _ in range(500):
        regulator.update(0.0, DT)
        assert 0.0 <= regulator.commanded_force <= config.safety_limit


def test_the_commanded_force_stays_bounded_under_a_random_adversarial_sequence() -> None:
    """A thousand random demands and measurements cannot push the command out of range."""
    config = ForceConfig()
    regulator = ProportionalForceRegulator(config)
    generator = np.random.default_rng(20260731)
    extremes = np.array([math.inf, -math.inf, math.nan, 1e300, -1e300])
    for step in range(1000):
        if step % 7 == 0:
            demand = float(extremes[step % extremes.size])
        else:
            demand = float(generator.normal(0.0, 1e6))
        regulator.demand(demand)
        measured = float(generator.normal(0.0, 1e3))
        rate = regulator.update(measured, DT)
        assert 0.0 <= regulator.commanded_force <= config.safety_limit
        assert 0.0 <= regulator.target_force <= config.safety_limit
        assert abs(rate) <= config.closure_rate_limit


def test_the_slip_response_ladder_is_clamped_at_the_safety_limit() -> None:
    """Repeated slip responses saturate rather than run away."""
    config = ForceConfig()
    response = SlipResponseConfig()
    regulator = ProportionalForceRegulator(config)
    regulator.demand(config.nominal_force)
    values = []
    for _ in range(30):
        regulator.demand(response.raised(regulator.target_force))
        values.append(regulator.target_force)
    assert values[0] == pytest.approx(
        config.nominal_force * response.multiplier + response.increment
    )
    assert max(values) == config.safety_limit
    assert values[-1] == config.safety_limit


def test_the_slip_response_step_is_the_declared_rule() -> None:
    response = SlipResponseConfig(multiplier=2.0, increment=0.2)
    assert response.raised(1.2) == pytest.approx(2.6)


def test_the_regulator_rejects_a_non_positive_step() -> None:
    regulator = ProportionalForceRegulator(ForceConfig())
    with pytest.raises(ValueError, match="dt must be positive"):
        regulator.update(1.0, 0.0)


def test_a_non_finite_measurement_is_treated_as_zero() -> None:
    regulator = ProportionalForceRegulator(ForceConfig())
    regulator.demand(1.0)
    assert math.isfinite(regulator.update(math.nan, DT))


def test_the_stiffness_estimate_stays_inside_its_bounds() -> None:
    config = ForceConfig()
    regulator = ProportionalForceRegulator(config)
    regulator.demand(5.0)
    generator = np.random.default_rng(7)
    for _ in range(500):
        regulator.update(float(generator.uniform(0.0, 10.0)), DT)
        assert config.minimum_stiffness <= regulator.stiffness_estimate
        assert regulator.stiffness_estimate <= config.maximum_stiffness


def test_the_estimate_adapts_towards_the_true_plant_gain() -> None:
    """A soft object drives the estimate far below its conservative start."""
    config = ForceConfig()
    regulator = ProportionalForceRegulator(config)
    stiffness = effective_stiffness(PAD, DEFORMABLE)
    closure = closure_at_span(HAND, GRASP, WIDTH)
    regulator.demand(2.0)
    force = 0.0
    previous = penetration(opposition_span(HAND, GRASP, closure), WIDTH)
    for _ in range(2000):
        rate = regulator.update(force, DT)
        closure = min(max(closure + rate * DT, 0.0), 1.0)
        indentation = penetration(opposition_span(HAND, GRASP, closure), WIDTH)
        force = contact_force(
            indentation, (indentation - previous) / DT, stiffness, PAD.exponent, 0.6
        )
        previous = indentation
    assert regulator.stiffness_estimate < 0.5 * config.initial_stiffness


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("safety_limit", 0.0),
        ("nominal_force", 0.0),
        ("nominal_force", 100.0),
        ("ramp_rate", 0.0),
        ("proportional_gain", 0.0),
        ("tolerance", 0.0),
        ("minimum_stiffness", 0.0),
    ],
)
def test_force_configuration_rejects_impossible_parameters(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        ForceConfig(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"), [("multiplier", 0.5), ("increment", -0.1), ("refractory_time", -0.1)]
)
def test_slip_response_configuration_rejects_impossible_parameters(
    field: str, value: float
) -> None:
    with pytest.raises(ValueError):
        SlipResponseConfig(**{field: value})
