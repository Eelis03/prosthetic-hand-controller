"""Property and invariant tests for slip detection and the response to it.

The last two tests measure the effect of the slip response rather than asserting
it: the same object, the same disturbance and the same controller are run twice,
once with the response enabled and once without, and the difference in how far
the object slid is the result.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hand_controller.algorithm import BandPassSlipDetector, SlipDetectorConfig
from hand_controller.algorithm.protocols import SlipDetector
from hand_controller.analysis import summarise
from hand_controller.pipeline import reference_trial, simulate, without_slip_response

CONFIG = SlipDetectorConfig()
DT = 0.001


def _feed(
    detector: BandPassSlipDetector, amplitude: float, frequency: float, duration: float
) -> tuple[bool, float]:
    detected = False
    energy = 0.0
    for step in range(round(duration / DT)):
        sample = amplitude * math.sin(2.0 * math.pi * frequency * step * DT)
        reading = detector.update(sample, DT)
        detected = detected or reading.detected
        energy = max(energy, reading.energy)
    return detected, energy


def test_the_detector_satisfies_the_protocol() -> None:
    detector: SlipDetector = BandPassSlipDetector(CONFIG)
    reading = detector.update(0.0, DT)
    assert reading.detected is False
    detector.reset()


def test_a_grip_force_applied_smoothly_is_not_slip() -> None:
    """A load that arrives the way the controller applies one carries no band energy."""
    detector = BandPassSlipDetector(CONFIG)
    rise = 0.200
    detected = False
    peak = 0.0
    for step in range(round(1.0 / DT)):
        time = step * DT
        fraction = min(time / rise, 1.0)
        force = 3.0 * 0.5 * (1.0 - math.cos(math.pi * fraction))
        reading = detector.update(force, DT)
        detected = detected or reading.detected
        peak = max(peak, reading.energy)
    assert not detected
    assert peak < CONFIG.on_threshold


def test_the_force_profile_of_a_held_object_is_never_read_as_slip() -> None:
    """Replay a real grip force history, contact transient included, into a fresh detector.

    This is the false trigger case for the force loop: if the detector answered
    yes to the loop's own force changes, every response would raise the force
    again and the grip would climb to the safety limit.
    """
    trace = simulate(reference_trial("foam_cup"))
    detector = BandPassSlipDetector(CONFIG)
    peak = 0.0
    for value in trace.grip_force:
        reading = detector.update(float(value), DT)
        assert not reading.detected
        peak = max(peak, reading.energy)
    assert peak < CONFIG.on_threshold


def test_a_vibration_inside_the_band_is_slip() -> None:
    detector = BandPassSlipDetector(CONFIG)
    detected, energy = _feed(detector, amplitude=1.0, frequency=60.0, duration=0.2)
    assert detected
    assert energy > CONFIG.on_threshold


def test_a_vibration_below_the_band_is_rejected() -> None:
    detector = BandPassSlipDetector(CONFIG)
    detected, _ = _feed(detector, amplitude=1.0, frequency=2.0, duration=0.5)
    assert not detected


def test_the_latch_releases_below_the_lower_threshold() -> None:
    """Hysteresis, so that a marginal signal cannot chatter the grip force up."""
    detector = BandPassSlipDetector(CONFIG)
    detected, _ = _feed(detector, amplitude=1.0, frequency=60.0, duration=0.2)
    assert detected
    released = False
    for _ in range(round(0.3 / DT)):
        released = not detector.update(0.0, DT).detected
    assert released


def test_the_detector_resets() -> None:
    detector = BandPassSlipDetector(CONFIG)
    _feed(detector, amplitude=1.0, frequency=60.0, duration=0.2)
    detector.reset()
    assert detector.envelope == 0.0
    assert detector.update(0.0, DT).detected is False


def test_a_non_finite_sample_is_treated_as_zero() -> None:
    detector = BandPassSlipDetector(CONFIG)
    reading = detector.update(math.nan, DT)
    assert math.isfinite(reading.energy)


def test_the_detector_rejects_a_non_positive_step() -> None:
    detector = BandPassSlipDetector(CONFIG)
    with pytest.raises(ValueError, match="dt must be positive"):
        detector.update(0.0, 0.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_rate", 0.0),
        ("low_cut", 0.0),
        ("high_cut", 10.0),
        ("order", 0),
        ("envelope_time_constant", 0.0),
        ("off_threshold", 0.0),
        ("confirm_time", -0.1),
    ],
)
def test_configuration_rejects_impossible_parameters(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        SlipDetectorConfig(**{field: value})


def test_the_band_must_fit_below_the_nyquist_frequency() -> None:
    with pytest.raises(ValueError, match="sample_rate / 2"):
        SlipDetectorConfig(low_cut=30.0, high_cut=600.0, sample_rate=1000.0)


@pytest.mark.parametrize(
    "object_name", ["drinking_glass", "plastic_bottle", "paper_cup_full", "battery_cell"]
)
def test_injected_slip_is_detected_and_the_force_increase_stops_it(object_name: str) -> None:
    """Measure, do not assume: the response must arrest the slide.

    The object starts to slip the instant it is lifted, because the nominal grip
    force is below what the friction coefficient requires. The assertions record
    that the detector fires, that the commanded force rises above the required
    value, and that the sliding stops.
    """
    trace = simulate(reference_trial(object_name))
    metrics = summarise(trace)
    lift = round(trace.config.lift_time / trace.config.dt)

    assert metrics.peak_slip_speed > 0.0
    assert bool(trace.slip_detected[lift:].any())
    assert metrics.slip_recovery_time is not None
    assert metrics.final_command > metrics.required_force
    assert trace.slip_velocity[-1] == 0.0
    assert metrics.total_slip < trace.config.plant.drop_distance
    assert metrics.success


@pytest.mark.parametrize(
    "object_name", ["drinking_glass", "plastic_bottle", "paper_cup_full", "battery_cell"]
)
def test_the_same_object_is_dropped_without_the_slip_response(object_name: str) -> None:
    """The measurement that gives the slip response its value."""
    config = reference_trial(object_name)
    with_response = summarise(simulate(config))
    without = summarise(simulate(without_slip_response(config)))

    assert with_response.success
    assert not without.success
    assert without.total_slip > 100.0 * with_response.total_slip
    assert without.final_command == pytest.approx(
        config.controller.force.nominal_force, abs=config.controller.force.tolerance
    )


def test_detection_is_fast_enough_to_matter() -> None:
    """The detector fires within one envelope time constant of the slip starting.

    The tolerance is the detector's own envelope time constant, which is the
    scale on which the band limited energy can rise at all.
    """
    trace = simulate(reference_trial("plastic_bottle"))
    threshold = trace.config.plant.slip_speed_threshold
    moving = np.nonzero(trace.slip_velocity > threshold)[0]
    detected = np.nonzero(trace.slip_detected)[0]
    assert moving.size > 0
    assert detected.size > 0
    lag = (int(detected[0]) - int(moving[0])) * trace.config.dt
    assert 0.0 < lag <= CONFIG.envelope_time_constant + CONFIG.confirm_time
