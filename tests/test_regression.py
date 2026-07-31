"""Regression tests pinning recorded behaviour to a stored reference.

What is pinned, and what is not, is a deliberate choice.

Pinned: converged geometry, verdicts, counts, event times and settled forces.
Every one of these is either a closed form quantity, an integer, or a value the
loop has stopped changing.

Not pinned: the raw late run state of a contact simulation. A value taken from
an iterative solve that has not converged is not reproducible on another
machine, because the order in which a linear algebra kernel reduces a sum
differs between builds and the difference grows with the number of iterations.
Pinning such a value produces a test that passes where it was recorded and fails
everywhere else.

Every tolerance below is derived from the measurement rather than from the error
that happened to be observed. The natural scales are the simulation time step
for anything timed, the controller's convergence band for anything in newtons,
the solver tolerance for anything bracketed by a root finder, and, for the slip
displacement, the distance the object covers in one time step at its peak speed.

Running this module as a script rewrites the reference file, which should only
be done after a change of behaviour has been reviewed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hand_controller.analysis import summarise, summarise_set
from hand_controller.model import (
    GRASP_NAMES,
    GRASP_TAXONOMY,
    closure_at_span,
    default_hand,
    span_range,
)
from hand_controller.model.grasps import SPAN_SOLVER_TOLERANCE
from hand_controller.pipeline import EVALUATION_SET, run_evaluation

REFERENCE_PATH = Path(__file__).resolve().parent / "data" / "reference_run.json"

# One control period. Every recorded time is an integer multiple of it, so this
# is the coarsest resolution at which two runs can be said to differ.
TIME_TOLERANCE = 1.0e-3
# The band inside which the grip force controller declares the grip established.
FORCE_TOLERANCE = 0.05
# A thousand times the bracket width Brent's method is asked to reach, so the
# comparison does not sit on the solver's own boundary.
CLOSURE_TOLERANCE = 1000.0 * SPAN_SOLVER_TOLERANCE
# Forward kinematics is a fixed sequence of arithmetic on a few numbers, so a
# span is reproducible to the rounding of that sum.
SPAN_TOLERANCE = 1.0e-12
# Closed form quantities are compared relatively.
CLOSED_FORM_TOLERANCE = 1.0e-12


def _build_reference() -> dict[str, Any]:
    """Compute the reference record from the current implementation."""
    hand = default_hand()
    traces = run_evaluation()
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
            "slip_recovery_time": entry.slip_recovery_time,
            "peak_slip_speed": entry.peak_slip_speed,
            "total_slip": entry.total_slip if entry.drop_time is None else None,
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
def recorded() -> dict[str, Any]:
    """Recompute the record from the current implementation."""
    return _build_reference()


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
    """Verdicts and counts are exact: they are decisions, not measurements."""
    stored = reference["trials"][name]
    current = recorded["trials"][name]
    for key in ("grasp", "feasible", "success", "failure", "slip_events", "force_saturated"):
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
    for key in ("final_command", "final_force", "peak_force"):
        assert current[key] == pytest.approx(stored[key], abs=FORCE_TOLERANCE), key


@pytest.mark.parametrize("name", [pair[0] for pair in EVALUATION_SET])
def test_trial_event_times_match_the_reference(
    reference: dict[str, Any], recorded: dict[str, Any], name: str
) -> None:
    """Event times are quantised to the control period, so that is the tolerance."""
    stored = reference["trials"][name]
    current = recorded["trials"][name]
    for key in ("time_to_contact", "time_to_grip", "slip_recovery_time", "drop_time"):
        if stored[key] is None:
            assert current[key] is None, key
        else:
            assert current[key] == pytest.approx(stored[key], abs=TIME_TOLERANCE), key


@pytest.mark.parametrize("name", [pair[0] for pair in EVALUATION_SET])
def test_slip_displacement_matches_the_reference(
    reference: dict[str, Any], recorded: dict[str, Any], name: str
) -> None:
    """Compare the settled slide with the distance covered in one control period.

    The displacement stops changing once the object is arrested, so it is a
    converged quantity, but the instant of arrest is quantised to the control
    period. One period at the peak sliding speed is therefore the smallest
    difference the measurement can resolve, and it is the tolerance. Objects that
    passed the drop distance record ``None``, because their displacement
    afterwards is the free fall of an object that has already left the hand.
    """
    stored = reference["trials"][name]
    current = recorded["trials"][name]
    if stored["total_slip"] is None:
        assert current["total_slip"] is None
        return
    tolerance = max(stored["peak_slip_speed"] * TIME_TOLERANCE, 1e-12)
    assert current["total_slip"] == pytest.approx(stored["total_slip"], abs=tolerance)


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
