"""Tests of the closed loop simulation and the trace it produces."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from hand_controller.model import GRAVITY, Feasibility, detect_contact, grasp, grasp_object
from hand_controller.pipeline import (
    EVALUATION_SET,
    GraspPhase,
    GraspTrace,
    TactileConfig,
    TrialConfig,
    mode_switch_trial,
    reference_trial,
    simulate,
    tactile_signal,
)


@pytest.fixture(scope="module")
def traces(evaluation_traces: tuple[GraspTrace, ...]) -> tuple[GraspTrace, ...]:
    """The shared evaluation set, named locally for readability."""
    return evaluation_traces


def test_the_evaluation_set_covers_every_object_once(traces: tuple[GraspTrace, ...]) -> None:
    names = [trace.item.name for trace in traces]
    assert names == [name for name, _ in EVALUATION_SET]
    assert len(set(names)) == len(names)


def test_every_trace_has_one_row_per_control_period(traces: tuple[GraspTrace, ...]) -> None:
    """Every column is the same length, and that length is the trial that ran.

    A trial that keeps its object runs to the configured duration. One that
    loses it ends on the sample the object left, which is shorter.
    """
    for trace in traces:
        expected = len(trace.time)
        assert expected == trace.config.steps or trace.released
        assert expected <= trace.config.steps
        for name in (
            "emg_open",
            "command_velocity",
            "closure",
            "span",
            "grip_force",
            "commanded_force",
            "slip_velocity",
            "phase",
            "mode",
        ):
            assert getattr(trace, name).shape == (expected,)


def test_contact_flags_match_the_sign_of_the_indentation(
    traces: tuple[GraspTrace, ...],
) -> None:
    """The recorded contact flag is exactly the strict sign test, at every sample."""
    for trace in traces:
        expected = np.array([detect_contact(float(value)) for value in trace.indentation])
        np.testing.assert_array_equal(trace.contact, expected)
        np.testing.assert_array_equal(trace.contact, trace.indentation > 0.0)


def test_no_force_is_recorded_without_contact(traces: tuple[GraspTrace, ...]) -> None:
    for trace in traces:
        assert float(trace.grip_force[~trace.contact].max(initial=0.0)) == 0.0


def test_the_commanded_force_never_exceeds_the_safety_limit(
    traces: tuple[GraspTrace, ...],
) -> None:
    """The clamp holds for every object, including the one that needs more than the limit."""
    for trace in traces:
        limit = trace.config.controller.force.safety_limit
        assert float(trace.commanded_force.max()) <= limit
        assert float(trace.commanded_force.min()) >= 0.0
        assert float(trace.demanded_force.max()) <= limit


def test_closure_stays_inside_its_travel(traces: tuple[GraspTrace, ...]) -> None:
    for trace in traces:
        assert float(trace.closure.min()) >= 0.0
        assert float(trace.closure.max()) <= 1.0


def test_the_phase_never_runs_backwards(traces: tuple[GraspTrace, ...]) -> None:
    for trace in traces:
        assert bool(np.all(np.diff(trace.phase) >= 0))
        assert trace.phase[0] == int(GraspPhase.REACH)


def test_the_load_appears_only_after_the_lift(traces: tuple[GraspTrace, ...]) -> None:
    for trace in traces:
        lift = round(trace.config.lift_time / trace.config.dt)
        assert float(trace.tangential_load[:lift].max()) == 0.0
        assert float(trace.tangential_load[lift:].min()) > 0.0


def test_slip_displacement_never_decreases(traces: tuple[GraspTrace, ...]) -> None:
    for trace in traces:
        assert bool(np.all(np.diff(trace.slip_displacement) >= 0.0))
        assert bool(np.all(trace.slip_velocity >= 0.0))


def test_a_trial_ends_on_the_sample_the_object_leaves_the_hand(
    traces: tuple[GraspTrace, ...],
) -> None:
    """The drop distance is a departure, not a marker passed on the way through.

    Exactly one recorded sample may lie beyond the drop distance, and it must be
    the last one. Anything else means the model went on simulating an object
    that was no longer in the hand.
    """
    for trace in traces:
        drop = trace.config.plant.drop_distance
        beyond = trace.slip_displacement > drop
        assert bool(beyond.any()) == trace.released
        assert len(trace) <= trace.config.steps
        if trace.released:
            assert int(np.count_nonzero(beyond)) == 1
            assert bool(beyond[-1])
        else:
            assert len(trace) == trace.config.steps
            assert float(trace.slip_displacement[-1]) <= drop


def test_no_object_slides_further_or_faster_than_it_can(
    traces: tuple[GraspTrace, ...],
) -> None:
    """Two bounds on the slide, neither with a fitted constant in it.

    The object leaves the hand at the drop distance, so it can never be recorded
    further than one control period of travel past it. It starts at rest and its
    acceleration is the weight less the friction capacity divided by the mass, so
    it can never exceed gravity, and its speed after sliding a distance is at
    most the free fall speed over that distance.

    The first bound is the one that has teeth here. The model used to report a
    slide of 8.50 m at 6636.33 mm/s for the object it dropped, four hundred
    times the drop distance, because it went on sliding an object that had left
    the hand two seconds earlier.
    """
    for trace in traces:
        distance = float(trace.slip_displacement[-1])
        peak = float(trace.slip_velocity.max())
        assert distance <= trace.config.plant.drop_distance + peak * trace.config.dt
        assert peak <= math.sqrt(2.0 * GRAVITY * distance)


def test_the_deformable_object_needs_more_closure_than_the_rigid_one() -> None:
    """The compliance shows up in the loop, not only in the static model."""
    rigid = simulate(reference_trial("drinking_glass"))
    soft = simulate(reference_trial("paper_cup_full"))
    assert float(soft.indentation[-1]) > 5.0 * float(rigid.indentation[-1])
    assert float(soft.closure[-1]) > float(rigid.closure[-1])


def test_the_grasp_holds_after_the_user_relaxes() -> None:
    """Shared control: the hand keeps the object once the contraction has decayed."""
    trace = simulate(reference_trial("drinking_glass"))
    tail = slice(-200, None)
    assert float(trace.smoothed_close[tail].max()) < 0.2
    assert bool(trace.contact[tail].all())
    assert float(trace.grip_force[-1]) > 1.0


def test_an_infeasible_object_is_reported_before_it_is_simulated() -> None:
    """A pen is far below the closed span of a medium wrap, so the wrap cannot reach it."""
    config = reference_trial("pen", grasp_name="medium_wrap")
    trace = simulate(config)
    assert trace.feasible is Feasibility.TOO_SMALL


def test_the_trial_can_be_driven_by_a_different_hand_and_object() -> None:
    config = reference_trial("drinking_glass")
    item = dataclasses.replace(grasp_object("drinking_glass"), mass=0.100)
    trace = simulate(config, definition=grasp("medium_wrap"), item=item)
    assert trace.item.mass == 0.100


def test_the_mode_switch_trial_selects_grasps_without_closing_the_hand() -> None:
    trace = simulate(mode_switch_trial())
    assert int(trace.mode.max()) > 0
    assert float(np.abs(trace.command_velocity).max()) == 0.0
    assert float(trace.closure.max()) == 0.0


def test_the_tactile_signal_is_the_force_plus_slip_vibration() -> None:
    config = TactileConfig(slip_gain=100.0, slip_frequency=50.0, noise_std=0.0)
    assert tactile_signal(config, 2.0, 0.0, 0.3, 0.0) == pytest.approx(2.0)
    quarter = 1.0 / (4.0 * config.slip_frequency)
    assert tactile_signal(config, 0.0, 0.01, quarter, 0.0) == pytest.approx(1.0)


def test_a_noiseless_sensor_produces_no_noise() -> None:
    config = reference_trial("foam_cup")
    plant = dataclasses.replace(config.plant, tactile=TactileConfig(noise_std=0.0))
    trace = simulate(dataclasses.replace(config, plant=plant))
    np.testing.assert_allclose(trace.tactile, trace.grip_force, atol=1e-12)


@pytest.mark.parametrize(
    ("field", "value"),
    [("dt", 0.0), ("duration", 0.0005), ("lift_time", -1.0), ("lift_time", 10.0)],
    ids=["zero step", "single step", "negative lift", "lift beyond the trial"],
)
def test_the_trial_configuration_rejects_impossible_timing(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        TrialConfig(grasp="tripod", object_name="pen", **{field: value})


@pytest.mark.parametrize(
    ("field", "value"), [("slip_gain", -1.0), ("slip_frequency", 0.0), ("noise_std", -1.0)]
)
def test_the_tactile_configuration_rejects_impossible_parameters(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        TactileConfig(**{field: value})


def test_a_shorter_trial_produces_a_shorter_trace() -> None:
    trace = simulate(reference_trial("pen", duration=1.5, lift_time=1.0))
    assert len(trace) == 1500
