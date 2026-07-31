"""Property and invariant tests for the proportional control law and its latency."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hand_controller.algorithm import (
    EnvelopeFilter,
    ProportionalConfig,
    ProportionalControlLaw,
    command_latency,
    control_law,
    saturation_difference,
)
from hand_controller.algorithm.protocols import ControlLaw

CONFIG = ProportionalConfig()
DT = 0.001
# The characteristic is sampled densely rather than at a handful of points, so
# that a discontinuity anywhere in the range would be caught.
DIFFERENCES = np.linspace(-1.0, 1.0, 4001)
COMMANDS = np.array([control_law(CONFIG, float(value)) for value in DIFFERENCES])


def test_the_law_satisfies_the_control_law_protocol() -> None:
    law: ControlLaw = ProportionalControlLaw(CONFIG)
    assert law.command(0.5) == control_law(CONFIG, 0.5)


def test_the_command_is_exactly_zero_inside_the_dead_zone() -> None:
    """Nothing inside the dead zone commands motion, checked at 4001 points."""
    inside = np.abs(DIFFERENCES) <= CONFIG.deadzone
    assert inside.sum() > 100
    assert np.count_nonzero(COMMANDS[inside]) == 0


def test_the_command_is_non_zero_immediately_outside_the_dead_zone() -> None:
    outside = np.abs(DIFFERENCES) > CONFIG.deadzone
    assert np.all(np.abs(COMMANDS[outside]) > 0.0)


def test_the_characteristic_is_monotone_non_decreasing_everywhere() -> None:
    assert bool(np.all(np.diff(COMMANDS) >= 0.0))


def test_the_characteristic_is_strictly_increasing_between_the_dead_zone_and_saturation() -> None:
    """Between the two limits every increment of activation increases the command."""
    upper = saturation_difference(CONFIG)
    band = (CONFIG.deadzone < DIFFERENCES) & (upper > DIFFERENCES)
    assert band.sum() > 500
    assert bool(np.all(np.diff(COMMANDS[band]) > 0.0))


def test_the_command_saturates_at_the_configured_limit() -> None:
    assert float(np.abs(COMMANDS).max()) == pytest.approx(CONFIG.saturation, abs=0.0)
    above = np.abs(DIFFERENCES) >= saturation_difference(CONFIG)
    np.testing.assert_allclose(
        np.abs(COMMANDS[above]), CONFIG.saturation, atol=0.0, rtol=0.0
    )


def test_saturation_difference_is_the_first_saturating_input() -> None:
    edge = saturation_difference(CONFIG)
    assert control_law(CONFIG, edge) == pytest.approx(CONFIG.saturation)
    assert control_law(CONFIG, edge - 1e-9) < CONFIG.saturation


def test_the_law_is_odd_about_zero() -> None:
    """Closing and opening are mirror images, so the interface has no bias."""
    mirrored = np.array([control_law(CONFIG, float(-value)) for value in DIFFERENCES])
    np.testing.assert_allclose(mirrored, -COMMANDS, atol=0.0, rtol=0.0)


def test_positive_difference_closes_and_negative_opens() -> None:
    assert control_law(CONFIG, 0.5) > 0.0
    assert control_law(CONFIG, -0.5) < 0.0


@pytest.mark.parametrize("difference", [math.nan, math.inf, -math.inf])
def test_a_broken_channel_commands_nothing(difference: float) -> None:
    assert control_law(CONFIG, difference) == 0.0


@pytest.mark.parametrize("difference", [-1e9, -5.0, 5.0, 1e9])
def test_out_of_range_inputs_are_clipped_not_amplified(difference: float) -> None:
    assert abs(control_law(CONFIG, difference)) == CONFIG.saturation


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("deadzone", -0.1),
        ("deadzone", 1.0),
        ("gain", 0.0),
        ("saturation", 0.0),
        ("envelope_window", 0.0),
        ("slew_limit", 0.0),
    ],
)
def test_configuration_rejects_impossible_parameters(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        ProportionalConfig(**{field: value})


def test_envelope_filter_window_and_group_delay() -> None:
    envelope = EnvelopeFilter(0.050, DT)
    assert envelope.length == 50
    assert envelope.group_delay == pytest.approx(24.5)


def test_envelope_filter_reaches_the_input_after_one_window() -> None:
    envelope = EnvelopeFilter(0.050, DT)
    for _ in range(envelope.length):
        opening, closing = envelope.update(0.2, 0.8)
    assert opening == pytest.approx(0.2, abs=1e-12)
    assert closing == pytest.approx(0.8, abs=1e-12)


def test_envelope_filter_clips_and_sanitises_its_input() -> None:
    envelope = EnvelopeFilter(0.005, DT)
    for _ in range(envelope.length):
        opening, closing = envelope.update(math.nan, 5.0)
    assert opening == 0.0
    assert closing == pytest.approx(1.0)


def test_envelope_filter_resets() -> None:
    envelope = EnvelopeFilter(0.010, DT)
    for _ in range(20):
        envelope.update(0.6, 0.6)
    envelope.reset()
    opening, closing = envelope.update(0.0, 0.0)
    assert opening == 0.0
    assert closing == 0.0


@pytest.mark.parametrize(("window", "dt"), [(0.0, DT), (0.05, 0.0)])
def test_envelope_filter_rejects_impossible_geometry(window: float, dt: float) -> None:
    with pytest.raises(ValueError):
        EnvelopeFilter(window, dt)


def test_command_latency_meets_the_stated_budget() -> None:
    """The measured 90 percent rise time is inside the budget from Farrell and Weir (2007).

    The tolerance is one control period, because the measurement is the index of
    a sample and is therefore quantised to ``DT``.
    """
    latency = command_latency(CONFIG, DT)
    assert latency <= CONFIG.latency_budget + DT
    assert latency == pytest.approx(round(latency / DT) * DT, abs=1e-12)


def test_command_latency_is_at_least_the_smoothing_delay() -> None:
    """No configuration can respond faster than its own window allows."""
    latency = command_latency(CONFIG, DT)
    assert latency >= 0.5 * CONFIG.envelope_window


def test_a_shorter_window_lowers_the_latency() -> None:
    fast = ProportionalConfig(envelope_window=0.020)
    assert command_latency(fast, DT) < command_latency(CONFIG, DT)


def test_command_latency_rejects_a_level_inside_the_dead_zone() -> None:
    with pytest.raises(ValueError, match="does not command any motion"):
        command_latency(CONFIG, DT, close_level=0.05)


def test_command_latency_reports_a_command_that_never_arrives() -> None:
    slow = ProportionalConfig(envelope_window=0.5)
    with pytest.raises(RuntimeError, match="did not reach"):
        command_latency(slow, DT, horizon=0.05)
