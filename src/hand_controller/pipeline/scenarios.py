"""Named configurations shared by the examples, the tests and the regression file.

The evaluation set pairs each object with the grasp of the taxonomy that suits
its shape, which is the choice a user makes before reaching. The two extra
scenarios build a trial with the slip response disabled and a trial in which a
sequence of co-contractions steps through the grasps.
"""

from __future__ import annotations

import dataclasses
from typing import Final

from hand_controller.algorithm.force import SlipResponseConfig
from hand_controller.pipeline.emg import EmgProfile, co_contraction_profile, reaching_profile
from hand_controller.pipeline.simulation import (
    ControllerConfig,
    GraspTrace,
    TrialConfig,
    simulate,
)

__all__ = [
    "EVALUATION_SET",
    "evaluation_pairs",
    "grasp_for_object",
    "mode_switch_trial",
    "reference_trial",
    "run_evaluation",
    "without_slip_response",
]

EVALUATION_SET: Final[tuple[tuple[str, str], ...]] = (
    ("drinking_glass", "medium_wrap"),
    ("plastic_bottle", "medium_wrap"),
    ("foam_cup", "medium_wrap"),
    ("paper_cup_full", "medium_wrap"),
    ("apple", "power_sphere"),
    ("hardback_book", "prismatic_four"),
    ("battery_cell", "palmar_pinch"),
    ("pen", "tripod"),
    ("door_key", "lateral"),
    ("steel_ball", "power_sphere"),
)
"""Each object with the taxonomy grasp used to hold it."""


def evaluation_pairs() -> tuple[tuple[str, str], ...]:
    """Return the object and grasp pairs of the evaluation set."""
    return EVALUATION_SET


def grasp_for_object(name: str) -> str:
    """Return the grasp the evaluation set uses for one object."""
    for object_name, grasp_name in EVALUATION_SET:
        if object_name == name:
            return grasp_name
    raise KeyError(f"object {name!r} is not in the evaluation set")


def reference_trial(
    object_name: str = "drinking_glass",
    grasp_name: str | None = None,
    duration: float = 3.000,
    dt: float = 0.001,
    lift_time: float = 1.000,
    emg: EmgProfile | None = None,
) -> TrialConfig:
    """Build the standard trial for one object of the evaluation set."""
    selected = grasp_name if grasp_name is not None else grasp_for_object(object_name)
    profile = emg if emg is not None else reaching_profile(relax_time=lift_time, duration=duration)
    return TrialConfig(
        grasp=selected,
        object_name=object_name,
        duration=duration,
        dt=dt,
        lift_time=lift_time,
        emg=profile,
    )


def without_slip_response(config: TrialConfig) -> TrialConfig:
    """Return the same trial with the slip response switched off."""
    controller = dataclasses.replace(
        config.controller,
        slip_response=dataclasses.replace(config.controller.slip_response, enabled=False),
    )
    return dataclasses.replace(config, controller=controller)


def mode_switch_trial(
    burst_times: tuple[float, ...] = (0.30, 0.90, 1.50),
    duration: float = 2.000,
    dt: float = 0.001,
) -> TrialConfig:
    """A trial driven only by co-contractions, used to exercise grasp selection."""
    return TrialConfig(
        grasp="medium_wrap",
        object_name="drinking_glass",
        duration=duration,
        dt=dt,
        lift_time=duration,
        emg=co_contraction_profile(burst_times),
        controller=ControllerConfig(slip_response=SlipResponseConfig(enabled=False)),
    )


def run_evaluation(
    duration: float = 3.000,
    dt: float = 0.001,
    lift_time: float = 1.000,
    slip_response: bool = True,
) -> tuple[GraspTrace, ...]:
    """Simulate every object of the evaluation set and return the traces."""
    traces: list[GraspTrace] = []
    for object_name, grasp_name in EVALUATION_SET:
        config = reference_trial(
            object_name, grasp_name, duration=duration, dt=dt, lift_time=lift_time
        )
        if not slip_response:
            config = without_slip_response(config)
        traces.append(simulate(config))
    return tuple(traces)
