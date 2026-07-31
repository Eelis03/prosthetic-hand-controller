"""Metrics extracted from a grasp trace.

Every quantity here is derived from the recorded trace alone, so the same
definitions apply to a run with the slip response on and a run with it off. The
timing quantities are quantised to the simulation time step, which is the
tolerance any comparison against them has to use.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from hand_controller.model.contact import required_grip_force
from hand_controller.model.objects import Feasibility
from hand_controller.pipeline.simulation import GraspPhase, GraspTrace

__all__ = [
    "FailureMode",
    "GraspMetrics",
    "SetSummary",
    "force_overshoot",
    "slip_episodes",
    "steady_state_error",
    "success_rate",
    "summarise",
    "summarise_set",
]


class FailureMode(StrEnum):
    """Why a grasp trial did not succeed."""

    NONE = "none"
    INFEASIBLE = "infeasible"
    NO_CONTACT = "no_contact"
    DROPPED = "dropped"
    STILL_SLIPPING = "still_slipping"
    CRUSHED = "crushed"


@dataclass(frozen=True, slots=True)
class GraspMetrics:
    """The outcome of one trial."""

    object_name: str
    grasp_name: str
    feasible: Feasibility
    success: bool
    failure: FailureMode
    time_to_contact: float
    time_to_grip: float
    peak_force: float
    final_force: float
    final_command: float
    required_force: float
    crush_force: float
    force_overshoot: float
    steady_state_error: float
    slip_events: int
    total_slip: float
    peak_slip_speed: float
    slip_recovery_time: float | None
    drop_time: float | None
    force_saturated: bool


def _first_true(flags: NDArray[np.bool_]) -> int | None:
    found = np.nonzero(flags)[0]
    return int(found[0]) if found.size else None


def slip_episodes(
    trace: GraspTrace,
) -> tuple[tuple[int, int | None], ...]:
    """Return the start and end index of every interval in which the object slid.

    An episode starts at the first sample whose slip speed exceeds the plant's
    threshold and ends at the first later sample that falls back below it. An
    episode that never ends has ``None`` for its end index.
    """
    threshold = trace.config.plant.slip_speed_threshold
    moving = trace.slip_velocity > threshold
    episodes: list[tuple[int, int | None]] = []
    index = 0
    size = int(moving.size)
    while index < size:
        if not moving[index]:
            index += 1
            continue
        start = index
        while index < size and moving[index]:
            index += 1
        episodes.append((start, index if index < size else None))
    return tuple(episodes)


def force_overshoot(trace: GraspTrace) -> float:
    """Return the fractional overshoot of the initial load ramp.

    The window runs from first contact to the earlier of the lift and the first
    slip response, so it measures how well the loop settles on the nominal force
    before any disturbance arrives. It is zero when the force never exceeds the
    nominal value.
    """
    nominal = trace.config.controller.force.nominal_force
    contact = _first_true(trace.contact)
    if contact is None:
        return 0.0
    lift_index = round(trace.config.lift_time / trace.config.dt)
    response = _first_true(trace.slip_detected)
    end = min(lift_index, response if response is not None else lift_index)
    if end <= contact:
        return 0.0
    peak = float(trace.grip_force[contact:end].max())
    return max(0.0, (peak - nominal) / nominal)


def steady_state_error(trace: GraspTrace, window: float = 0.200) -> float:
    """Return the largest gap between the measured and commanded force at the end.

    The window is the last ``window`` seconds of the trial, by which time the
    demand has stopped moving in every scenario in the evaluation set. The value
    is the evidence that the proportional law leaves no offset: the plant from
    closure rate to force contains an integrator, so no integral term is needed
    and none is used.
    """
    samples = max(1, round(window / trace.config.dt))
    tail = slice(-samples, None)
    return float(np.abs(trace.grip_force[tail] - trace.commanded_force[tail]).max())


def summarise(trace: GraspTrace) -> GraspMetrics:
    """Reduce one trace to the numbers the evaluation table reports."""
    dt = trace.config.dt
    item = trace.item
    contacts = trace.grasp.load_bearing_contacts
    required = required_grip_force(item.mass, item.friction, contacts)

    onset = _first_true(trace.command_velocity > 0.0)
    contact = _first_true(trace.contact)
    time_to_contact = float("nan")
    if onset is not None and contact is not None:
        time_to_contact = (contact - onset) * dt

    force_config = trace.config.controller.force
    settled = _first_true(
        (trace.phase != int(GraspPhase.REACH))
        & (np.abs(trace.grip_force - force_config.nominal_force) <= force_config.tolerance)
    )
    time_to_grip = float("nan")
    if onset is not None and settled is not None:
        time_to_grip = (settled - onset) * dt

    episodes = slip_episodes(trace)
    recovery: float | None = None
    if episodes:
        start, end = episodes[0]
        recovery = (end - start) * dt if end is not None else None

    departed = _first_true(trace.slip_displacement > trace.config.plant.drop_distance)
    drop_time = None if departed is None else float(trace.time[departed]) - trace.config.lift_time

    events = int(np.count_nonzero(np.diff(trace.slip_detected.astype(np.int8)) > 0))
    total_slip = float(trace.slip_displacement[-1])
    peak_force = float(trace.grip_force.max())
    limit = trace.config.controller.force.safety_limit
    saturated = bool(trace.commanded_force.max() >= limit - 1e-12)

    failure = FailureMode.NONE
    if trace.feasible is not Feasibility.FEASIBLE:
        failure = FailureMode.INFEASIBLE
    elif contact is None:
        failure = FailureMode.NO_CONTACT
    elif total_slip > trace.config.plant.drop_distance:
        failure = FailureMode.DROPPED
    elif peak_force > item.crush_force:
        failure = FailureMode.CRUSHED
    elif trace.slip_velocity[-1] > trace.config.plant.slip_speed_threshold:
        failure = FailureMode.STILL_SLIPPING
    elif not bool(trace.contact[-1]):
        failure = FailureMode.NO_CONTACT

    return GraspMetrics(
        object_name=item.name,
        grasp_name=trace.grasp.name,
        feasible=trace.feasible,
        success=failure is FailureMode.NONE,
        failure=failure,
        time_to_contact=time_to_contact,
        time_to_grip=time_to_grip,
        peak_force=peak_force,
        final_force=float(trace.grip_force[-1]),
        final_command=float(trace.commanded_force[-1]),
        required_force=required,
        crush_force=item.crush_force,
        force_overshoot=force_overshoot(trace),
        steady_state_error=steady_state_error(trace),
        slip_events=events,
        total_slip=total_slip,
        peak_slip_speed=float(trace.slip_velocity.max()),
        slip_recovery_time=recovery,
        drop_time=drop_time,
        force_saturated=saturated,
    )


@dataclass(frozen=True, slots=True)
class SetSummary:
    """Aggregate outcome over a set of trials."""

    trials: int
    successes: int
    success_rate: float
    mean_time_to_contact: float
    mean_force_overshoot: float
    slipping_trials: int
    dropped_trials: int
    mean_slip_recovery: float
    max_slip_when_held: float


def success_rate(metrics: tuple[GraspMetrics, ...]) -> float:
    """Fraction of trials that succeeded."""
    if not metrics:
        raise ValueError("no trials to summarise")
    return sum(1 for entry in metrics if entry.success) / len(metrics)


def summarise_set(metrics: tuple[GraspMetrics, ...]) -> SetSummary:
    """Aggregate a set of trials into one record."""
    if not metrics:
        raise ValueError("no trials to summarise")
    recoveries = [
        entry.slip_recovery_time for entry in metrics if entry.slip_recovery_time is not None
    ]
    held = [entry.total_slip for entry in metrics if entry.drop_time is None]
    return SetSummary(
        trials=len(metrics),
        successes=sum(1 for entry in metrics if entry.success),
        success_rate=success_rate(metrics),
        mean_time_to_contact=float(np.mean([entry.time_to_contact for entry in metrics])),
        mean_force_overshoot=float(np.mean([entry.force_overshoot for entry in metrics])),
        slipping_trials=sum(1 for entry in metrics if entry.peak_slip_speed > 0.0),
        dropped_trials=sum(1 for entry in metrics if entry.drop_time is not None),
        mean_slip_recovery=float(np.mean(recoveries)) if recoveries else float("nan"),
        max_slip_when_held=max(held) if held else float("nan"),
    )
