"""Regression tests pinning recorded behaviour to a stored reference.

What is pinned, and what is not, was decided by measurement rather than by
argument. Every candidate quantity was recomputed with one object property
perturbed by a relative 1e-12 and by a relative 1e-9, which are both far below
any physically meaningful change, across five properties and every trial of the
evaluation set. The movement that produced is the reproducibility scale of that
quantity, and it separated the candidates cleanly into two groups.

Quantities that did not move at all, or moved only by double precision rounding,
and are therefore pinned:

* every discrete verdict, that is feasibility, success, failure mode, the slip
  detection count, whether the object slid at all, whether recovery occurred,
  whether the object was held, and whether the commanded force saturated: no
  change in any of the 100 perturbed runs. Note that ``slipped`` is a predicate
  over ``peak_slip_speed``, which is itself unstable, but the peak speeds are 26
  to 84 mm/s while the predicate compares against zero, so the verdict is nowhere
  near its own boundary even though the number is;
* ``time_to_contact`` and ``time_to_grip``: no change at all. Both are crossings
  of a signal moving quickly through its threshold. The indentation sweeps
  through zero at 0.035 mm to 0.043 mm per control period and the grip force
  ramps through its convergence band at 60 N per second, so a last bit
  difference cannot move either crossing into a different sample;
* ``drop_time``: no change. The object passes the drop distance at 0.59 m/s,
  which is the same steep crossing;
* ``final_command``: no change, because the demand ladder is a fixed sequence of
  arithmetic once the number of responses is fixed;
* ``final_force`` and ``peak_force`` of a trial that kept its object: at most
  7.0e-14 N, which is rounding;
* ``contact_stiffness`` and ``required_force``: closed form.

Quantities that moved far more than their own quantisation, and are therefore
bounded rather than pinned:

* ``slip_recovery_time``: up to 19 control periods, and not monotonically in the
  size of the perturbation;
* ``total_slip``: up to 2.08 mm, which is 25 times the distance the object
  covers in one control period at its peak sliding speed;
* ``peak_slip_speed``: up to 43.8 percent;
* ``final_force`` and ``peak_force`` of the trial that loses its object: up to
  2.71e-2 N, which is 3.9e11 times the movement the same two quantities show on
  a trial that settles. The trial ends on the sample the object leaves the hand,
  60 ms into the rise that follows the slip response, so both are readings taken
  during the transient rather than after it. They are bounded by the demand the
  loop was chasing at the time.

The reason is the same in every case. All of them are downstream of the finite
difference the grip force regulator uses to estimate its plant gain. That
estimate is a ratio of two small differences, so it is ill conditioned by
construction, and in the first few control periods after a slip response it
amplifies a 1e-12 change in the geometry into a seven percent change in the grip
force. The force settles to the same value either way, which is why the settled
quantities above are exact, but the deceleration of a sliding object differs
throughout the arrest, the instant it comes to rest moves by many samples, and a
force read before the transient has died is read at a different point on a
different curve. Quantisation bounds a readout, not a crossing, and this
crossing is reached tangentially.

The bounds are taken from configuration constants or from other pinned
quantities rather than from the recorded values, so that they cannot drift
towards whatever a run happens to produce. ``RECOVERY_BOUND`` is the slip
response refractory interval, the time within which one response is intended to
have done its work; the largest recorded recovery is 96 ms and the largest
measured movement is 19 ms, so the bound leaves 35 ms of headroom. ``SLIP_BOUND``
is half the drop distance at which the object is declared lost; the largest
recorded slide is 4.58 mm and the largest measured movement is 2.08 mm, so the
bound leaves 3.34 mm of headroom. The forces of the lost trial are bounded by
``final_command``, which is pinned and exact. None of the three is a widened
tolerance: ``test_the_slip_bounds_have_teeth`` shows that the first two are
violated as soon as the slip response is switched off, and the third fails as
soon as the loop delivers the force it was asked for, which is what every trial
that keeps its object does.

The remaining tolerances are derived from the measurement in the same way. The
natural scales are the control period for anything timed, the controller's
convergence band for anything in newtons, and the solver tolerance for anything
bracketed by a root finder.

Running this module as a script rewrites the reference file, which should only
be done after a change of behaviour has been reviewed.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from hand_controller.algorithm import SlipResponseConfig
from hand_controller.analysis import GraspMetrics, summarise, summarise_set
from hand_controller.model import (
    GRASP_NAMES,
    GRASP_TAXONOMY,
    closure_at_span,
    default_hand,
    grasp_object,
    span_range,
)
from hand_controller.model.grasps import SPAN_SOLVER_TOLERANCE
from hand_controller.pipeline import (
    EVALUATION_SET,
    GraspTrace,
    PlantConfig,
    reference_trial,
    run_evaluation,
    simulate,
)

REFERENCE_PATH = Path(__file__).resolve().parent / "data" / "reference_run.json"

# One control period. The event times that are pinned all cross their threshold
# steeply, and none of them moved at all under the perturbation study, so the
# period is the coarsest resolution at which two runs can be said to differ.
TIME_TOLERANCE = 1.0e-3
# The band inside which the grip force controller declares the grip established.
# The settled forces moved by at most 1.7e-13 N, twelve orders of magnitude
# inside it.
FORCE_TOLERANCE = 0.05
# A thousand times the bracket width Brent's method is asked to reach, so the
# comparison does not sit on the solver's own boundary.
CLOSURE_TOLERANCE = 1000.0 * SPAN_SOLVER_TOLERANCE
# Forward kinematics is a fixed sequence of arithmetic on a few numbers, so a
# span is reproducible to the rounding of that sum.
SPAN_TOLERANCE = 1.0e-12
# Closed form quantities are compared relatively.
CLOSED_FORM_TOLERANCE = 1.0e-12

# Bounds, not tolerances. Both come from configuration constants, and both are
# checked for regression power by test_the_slip_bounds_have_teeth.
RECOVERY_BOUND = SlipResponseConfig().refractory_time
SLIP_BOUND = 0.5 * PlantConfig().drop_distance


def _build_reference(traces: tuple[GraspTrace, ...] | None = None) -> dict[str, Any]:
    """Compute the reference record from the current implementation."""
    hand = default_hand()
    traces = traces if traces is not None else run_evaluation()
    metrics = tuple(summarise(trace) for trace in traces)
    summary = summarise_set(metrics)

    geometry = {}
    for definition in GRASP_TAXONOMY:
        closed, opened = span_range(hand, definition)
        midpoint = 0.5 * (closed + opened)
        geometry[definition.name] = {
            "closed_span": closed,
            "open_span": opened,
            "closure_at_midpoint": closure_at_span(hand, definition, midpoint),
            "load_bearing_contacts": definition.load_bearing_contacts,
        }

    # slip_recovery_time, total_slip and peak_slip_speed are deliberately absent.
    # The perturbation study recorded in the module docstring shows they are not
    # reproducible, and recording a number that is never asserted on would only
    # invite someone to start asserting on it.
    trials = {}
    for trace, entry in zip(traces, metrics, strict=True):
        trials[entry.object_name] = {
            "grasp": entry.grasp_name,
            "feasible": entry.feasible.value,
            "success": entry.success,
            "failure": entry.failure.value,
            "contact_stiffness": trace.contact_stiffness,
            "required_force": entry.required_force,
            "time_to_contact": entry.time_to_contact,
            "time_to_grip": entry.time_to_grip,
            "final_command": entry.final_command,
            "final_force": entry.final_force,
            "peak_force": entry.peak_force,
            "slip_events": entry.slip_events,
            "slip_detected": entry.slip_events > 0,
            "slipped": entry.peak_slip_speed > 0.0,
            "recovered": entry.slip_recovery_time is not None,
            "held": entry.drop_time is None,
            "drop_time": entry.drop_time,
            "force_saturated": entry.force_saturated,
        }

    return {
        "description": "Evaluation set at the reference configuration, for regression testing.",
        "evaluation_set": [list(pair) for pair in EVALUATION_SET],
        "grasp_names": list(GRASP_NAMES),
        "geometry": geometry,
        "trials": trials,
        "summary": {
            "trials": summary.trials,
            "successes": summary.successes,
            "success_rate": summary.success_rate,
            "slipping_trials": summary.slipping_trials,
            "dropped_trials": summary.dropped_trials,
        },
    }


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    """Load the stored reference record."""
    if not REFERENCE_PATH.exists():
        raise AssertionError(
            f"reference file is missing: {REFERENCE_PATH}. "
            "Regenerate it with 'uv run python tests/test_regression.py'."
        )
    with REFERENCE_PATH.open(encoding="utf-8") as handle:
        loaded: dict[str, Any] = json.load(handle)
    return loaded


@pytest.fixture(scope="module")
def recorded(evaluation_traces: tuple[GraspTrace, ...]) -> dict[str, Any]:
    """Recompute the record from the current implementation."""
    return _build_reference(evaluation_traces)


def test_the_evaluation_set_has_not_drifted(
    reference: dict[str, Any], recorded: dict[str, Any]
) -> None:
    assert recorded["evaluation_set"] == reference["evaluation_set"]
    assert recorded["grasp_names"] == reference["grasp_names"]


@pytest.mark.parametrize("name", GRASP_NAMES)
def test_grasp_geometry_matches_the_reference(
    reference: dict[str, Any], recorded: dict[str, Any], name: str
) -> None:
    """Spans are pure forward kinematics, and the closure is a converged root."""
    stored = reference["geometry"][name]
    current = recorded["geometry"][name]
    assert current["load_bearing_contacts"] == stored["load_bearing_contacts"]
    assert current["closed_span"] == pytest.approx(stored["closed_span"], abs=SPAN_TOLERANCE)
    assert current["open_span"] == pytest.approx(stored["open_span"], abs=SPAN_TOLERANCE)
    assert current["closure_at_midpoint"] == pytest.approx(
        stored["closure_at_midpoint"], abs=CLOSURE_TOLERANCE
    )


@pytest.mark.parametrize("name", [pair[0] for pair in EVALUATION_SET])
def test_trial_verdicts_match_the_reference(
    reference: dict[str, Any], recorded: dict[str, Any], name: str
) -> None:
    """Verdicts and counts are exact: they are decisions, not measurements.

    None of these changed in any of the 100 perturbed runs of the study
    described in the module docstring, which is why they are compared with no
    tolerance at all.
    """
    stored = reference["trials"][name]
    current = recorded["trials"][name]
    for key in (
        "grasp",
        "feasible",
        "success",
        "failure",
        "slip_events",
        "slip_detected",
        "slipped",
        "recovered",
        "held",
        "force_saturated",
    ):
        assert current[key] == stored[key], key


@pytest.mark.parametrize("name", [pair[0] for pair in EVALUATION_SET])
def test_trial_forces_match_the_reference(
    reference: dict[str, Any], recorded: dict[str, Any], name: str
) -> None:
    """Settled forces are compared inside the controller's convergence band."""
    stored = reference["trials"][name]
    current = recorded["trials"][name]
    assert current["contact_stiffness"] == pytest.approx(
        stored["contact_stiffness"], rel=CLOSED_FORM_TOLERANCE
    )
    assert current["required_force"] == pytest.approx(
        stored["required_force"], rel=CLOSED_FORM_TOLERANCE
    )
    assert current["final_command"] == pytest.approx(stored["final_command"], abs=FORCE_TOLERANCE)
    if stored["held"]:
        for key in ("final_force", "peak_force"):
            assert current[key] == pytest.approx(stored[key], abs=FORCE_TOLERANCE), key
    else:
        # A trial that lost its object stopped in the middle of a rise, so its
        # forces are transient readings. What is asserted is the claim they
        # exist to support: the loop had not yet delivered the force it was
        # asked for when the object left.
        assert 0.0 < current["peak_force"] < current["final_command"]
        assert current["final_force"] <= current["peak_force"]


@pytest.mark.parametrize("name", [pair[0] for pair in EVALUATION_SET])
def test_trial_event_times_match_the_reference(
    reference: dict[str, Any], recorded: dict[str, Any], name: str
) -> None:
    """Pin only the event times whose crossings are steep.

    Each of these three crosses its threshold far faster than one control period
    of numerical noise, and none of them moved at all in the perturbation study,
    so the control period is a safe tolerance. The arrest of a sliding object is
    not in this list, for the reason set out in the module docstring.
    """
    stored = reference["trials"][name]
    current = recorded["trials"][name]
    for key in ("time_to_contact", "time_to_grip", "drop_time"):
        if stored[key] is None:
            assert current[key] is None, key
        else:
            assert current[key] == pytest.approx(stored[key], abs=TIME_TOLERANCE), key


@pytest.mark.parametrize("name", [pair[0] for pair in EVALUATION_SET])
def test_the_slip_response_stays_inside_its_budget(
    metrics_by_object: dict[str, GraspMetrics], name: str
) -> None:
    """Bound the two quantities the perturbation study showed are not reproducible.

    ``slip_recovery_time`` moves by up to 19 control periods and ``total_slip``
    by up to 2.08 mm under a perturbation of 1e-12, so neither is pinned. What is
    asserted instead is the claim the numbers exist to support: that a detected
    slip is arrested within the refractory interval of one response, and that the
    object slides less than half the distance at which it would be declared lost.
    Both limits come from configuration constants, not from a recorded run.
    """
    metrics = metrics_by_object[name]
    if metrics.slip_recovery_time is not None:
        assert 0.0 < metrics.slip_recovery_time <= RECOVERY_BOUND
    if metrics.drop_time is None:
        assert metrics.total_slip <= SLIP_BOUND


def _perturbed_metrics(relative: float) -> dict[str, GraspMetrics]:
    """Recompute the evaluation set with every object width nudged by ``relative``."""
    results: dict[str, GraspMetrics] = {}
    for object_name, grasp_name in EVALUATION_SET:
        item = grasp_object(object_name)
        nudged = dataclasses.replace(item, width=item.width * (1.0 + relative))
        trace = simulate(reference_trial(object_name, grasp_name), item=nudged)
        results[object_name] = summarise(trace)
    return results


@pytest.mark.parametrize("relative", [1e-12, 1e-9])
def test_the_pinned_fields_survive_a_negligible_perturbation(
    metrics_by_object: dict[str, GraspMetrics], relative: float
) -> None:
    """Every pinned field must be insensitive to a change far below any real one.

    This is the reproducibility contract stated as a test. A difference in the
    order a floating point sum is reduced, which is what separates one machine
    from another, is far smaller than 1e-12 of an object width. Anything that
    survives this is safe to pin, and anything that does not has to be bounded
    instead. Both bounded quantities are checked here against their bounds rather
    than against the unperturbed value.
    """
    perturbed = _perturbed_metrics(relative)
    for name, baseline in metrics_by_object.items():
        other = perturbed[name]
        assert other.feasible == baseline.feasible, name
        assert other.success == baseline.success, name
        assert other.failure == baseline.failure, name
        assert other.slip_events == baseline.slip_events, name
        assert other.force_saturated == baseline.force_saturated, name
        assert (other.slip_recovery_time is None) == (baseline.slip_recovery_time is None), name
        assert (other.drop_time is None) == (baseline.drop_time is None), name
        assert other.time_to_contact == pytest.approx(
            baseline.time_to_contact, abs=TIME_TOLERANCE
        ), name
        assert other.time_to_grip == pytest.approx(baseline.time_to_grip, abs=TIME_TOLERANCE), name
        assert other.final_command == pytest.approx(baseline.final_command, abs=FORCE_TOLERANCE), (
            name
        )
        if other.drop_time is not None and baseline.drop_time is not None:
            assert other.drop_time == pytest.approx(baseline.drop_time, abs=TIME_TOLERANCE), name
        if other.slip_recovery_time is not None:
            assert 0.0 < other.slip_recovery_time <= RECOVERY_BOUND, name
        if other.drop_time is None:
            assert other.peak_force == pytest.approx(baseline.peak_force, abs=FORCE_TOLERANCE), name
            assert other.total_slip <= SLIP_BOUND, name
        else:
            assert 0.0 < other.peak_force < other.final_command, name


def test_the_slip_bounds_have_teeth() -> None:
    """A bound that nothing can violate is a widened tolerance in disguise.

    Switching off the slip response leaves the rest of the controller untouched
    and makes both bounds fail on every object that needs more than the nominal
    grip force, which is what gives them regression power.
    """
    traces = run_evaluation(slip_response=False)
    metrics = [summarise(trace) for trace in traces]
    affected = [entry for entry in metrics if entry.peak_slip_speed > 0.0]
    assert len(affected) >= 6
    for entry in affected:
        assert entry.slip_recovery_time is None
        assert entry.total_slip > SLIP_BOUND


def test_the_summary_matches_the_reference(
    reference: dict[str, Any], recorded: dict[str, Any]
) -> None:
    """Success rate and counts are exact."""
    assert recorded["summary"] == reference["summary"]


def main() -> int:
    """Write the reference file from the current implementation."""
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REFERENCE_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_build_reference(), handle, indent=1, sort_keys=True)
        handle.write("\n")
    print(f"wrote {REFERENCE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
