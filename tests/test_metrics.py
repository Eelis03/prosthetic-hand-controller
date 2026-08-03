"""Tests of the evaluation metrics and the text reports built from them."""

from __future__ import annotations

import numpy as np
import pytest

from hand_controller.analysis import (
    FailureMode,
    GraspMetrics,
    force_overshoot,
    format_evaluation,
    format_feasibility,
    format_set_summary,
    format_slip_comparison,
    format_taxonomy,
    format_trial,
    slip_episodes,
    steady_state_error,
    success_rate,
    summarise,
    summarise_set,
)
from hand_controller.model import default_hand, required_grip_force
from hand_controller.pipeline import GraspTrace, simulate, without_slip_response
from hand_controller.pipeline.scenarios import reference_trial

HAND = default_hand()


@pytest.fixture(scope="module")
def traces(evaluation_traces: tuple[GraspTrace, ...]) -> tuple[GraspTrace, ...]:
    """The shared evaluation set, named locally for readability."""
    return evaluation_traces


@pytest.fixture(scope="module")
def metrics(evaluation_metrics: tuple[GraspMetrics, ...]) -> tuple[GraspMetrics, ...]:
    """The shared metrics, named locally for readability."""
    return evaluation_metrics


def test_nine_of_the_ten_objects_are_held(metrics: tuple[GraspMetrics, ...]) -> None:
    assert len(metrics) == 10
    assert success_rate(metrics) == pytest.approx(0.9)
    assert sum(1 for entry in metrics if entry.success) == 9


def test_the_failure_is_the_steel_ball_and_it_is_gone_before_the_ladder_climbs(
    traces: tuple[GraspTrace, ...], metrics: tuple[GraspMetrics, ...]
) -> None:
    """The one object the controller cannot hold, and the reason it cannot.

    The ball needs more force per contact than the safety limit allows, so it
    was never holdable. It is also gone before the demand ladder gets anywhere
    near that limit: one slip response is granted, the next is barred by the
    refractory interval, and the ball has left the hand first.
    """
    failures = [entry for entry in metrics if not entry.success]
    assert [entry.object_name for entry in failures] == ["steel_ball"]
    failure = failures[0]
    trace = next(item for item in traces if item.item.name == "steel_ball")
    limit = trace.config.controller.force.safety_limit
    refractory = trace.config.controller.slip_response.refractory_time

    assert failure.failure is FailureMode.DROPPED
    assert failure.required_force > limit
    assert failure.slip_events == 1
    assert failure.drop_time is not None
    assert failure.drop_time < refractory
    assert failure.final_command < limit
    assert failure.required_force > failure.final_command
    assert not failure.force_saturated
    assert trace.released


def test_the_required_force_matches_the_closed_form(
    traces: tuple[GraspTrace, ...], metrics: tuple[GraspMetrics, ...]
) -> None:
    for trace, entry in zip(traces, metrics, strict=True):
        expected = required_grip_force(
            trace.item.mass, trace.item.friction, trace.grasp.load_bearing_contacts
        )
        assert entry.required_force == pytest.approx(expected, rel=1e-15)


def test_no_held_object_is_crushed(metrics: tuple[GraspMetrics, ...]) -> None:
    for entry in metrics:
        if entry.success:
            assert entry.peak_force < entry.crush_force


def test_timing_metrics_are_quantised_to_the_time_step(
    traces: tuple[GraspTrace, ...], metrics: tuple[GraspMetrics, ...]
) -> None:
    """Every time reported is an integer number of control periods.

    The tolerance is a millionth of the time step: the values are products of an
    integer index and ``dt``, so any residue larger than floating point rounding
    would mean the metric had been computed some other way.
    """
    for trace, entry in zip(traces, metrics, strict=True):
        dt = trace.config.dt
        for value in (entry.time_to_contact, entry.time_to_grip):
            assert value == pytest.approx(round(value / dt) * dt, abs=1e-6 * dt)
        if entry.slip_recovery_time is not None:
            recovery = entry.slip_recovery_time
            assert recovery == pytest.approx(round(recovery / dt) * dt, abs=1e-6 * dt)


def test_the_loop_leaves_no_steady_state_error(
    traces: tuple[GraspTrace, ...], metrics: tuple[GraspMetrics, ...]
) -> None:
    """The tolerance is the controller's own convergence band.

    A steady state is only defined for a trial that reached one. The trial that
    loses its object ends on the sample the object left, in the middle of the
    rise that followed the slip response, and is asserted on separately below.
    """
    for trace, entry in zip(traces, metrics, strict=True):
        assert steady_state_error(trace) == entry.steady_state_error
        if not trace.released:
            assert entry.steady_state_error <= trace.config.controller.force.tolerance


def test_the_trial_that_loses_its_object_ends_while_the_force_is_still_rising(
    traces: tuple[GraspTrace, ...], metrics: tuple[GraspMetrics, ...]
) -> None:
    """The one trial with a steady state error has not settled, it has stopped.

    The force is still chasing a demand that was raised 60 ms earlier when the
    object leaves, which is exactly why its final error is not evidence about
    the control law.
    """
    for trace, entry in zip(traces, metrics, strict=True):
        if not trace.released:
            continue
        assert entry.steady_state_error > trace.config.controller.force.tolerance
        assert entry.final_force < entry.final_command
        assert float(trace.demanded_force[-1]) > trace.config.controller.force.nominal_force


def test_the_load_ramp_does_not_overshoot(
    traces: tuple[GraspTrace, ...], metrics: tuple[GraspMetrics, ...]
) -> None:
    """The tolerance is the convergence band expressed as a fraction of the nominal force.

    Anything smaller than that is inside the band in which the controller does
    not distinguish the force from its demand, so it is not an overshoot.
    """
    for trace, entry in zip(traces, metrics, strict=True):
        config = trace.config.controller.force
        assert entry.force_overshoot <= config.tolerance / config.nominal_force
        assert force_overshoot(trace) == entry.force_overshoot


def test_slip_episodes_bracket_the_moving_samples() -> None:
    trace = simulate(reference_trial("plastic_bottle"))
    episodes = slip_episodes(trace)
    assert len(episodes) == 1
    start, end = episodes[0]
    assert end is not None
    threshold = trace.config.plant.slip_speed_threshold
    assert trace.slip_velocity[start] > threshold
    assert trace.slip_velocity[end] <= threshold
    assert bool(np.all(trace.slip_velocity[start:end] > threshold))


def test_an_object_that_never_slips_has_no_episodes() -> None:
    trace = simulate(reference_trial("foam_cup"))
    assert slip_episodes(trace) == ()
    assert summarise(trace).slip_recovery_time is None


def test_the_set_summary_agrees_with_the_individual_metrics(
    metrics: tuple[GraspMetrics, ...],
) -> None:
    summary = summarise_set(metrics)
    assert summary.trials == len(metrics)
    assert summary.successes == sum(1 for entry in metrics if entry.success)
    assert summary.success_rate == success_rate(metrics)
    assert summary.dropped_trials == sum(1 for entry in metrics if entry.drop_time is not None)
    assert summary.slipping_trials == sum(1 for entry in metrics if entry.peak_slip_speed > 0.0)
    assert summary.max_slip_when_held < 0.020


def test_summaries_reject_an_empty_set() -> None:
    with pytest.raises(ValueError, match="no trials"):
        success_rate(())
    with pytest.raises(ValueError, match="no trials"):
        summarise_set(())


def test_the_reports_render_every_row(
    traces: tuple[GraspTrace, ...], metrics: tuple[GraspMetrics, ...]
) -> None:
    table = format_evaluation(metrics)
    assert len(table.splitlines()) == len(metrics) + 2
    for entry in metrics:
        assert entry.object_name in table

    summary = format_set_summary(summarise_set(metrics))
    assert "success rate" in summary

    detail = format_trial(traces[0], metrics[0])
    assert "outcome" in detail

    taxonomy = format_taxonomy(HAND)
    assert "Medium Wrap" in taxonomy

    feasible = format_feasibility(HAND)
    assert "too_large" in feasible


def test_the_slip_comparison_reports_both_runs() -> None:
    config = reference_trial("drinking_glass")
    on = summarise(simulate(config))
    off = summarise(simulate(without_slip_response(config)))
    table = format_slip_comparison((on,), (off,))
    assert "drinking_glass" in table
    assert "objects held" in table
