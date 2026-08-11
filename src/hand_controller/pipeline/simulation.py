"""The closed loop grasp simulation and the trace it records.

One control period does, in order:

1. read the two activation envelopes,
2. run the two site front end, which smooths them, applies the proportional map
   and decides whether a co-contraction has selected another grasp,
3. choose the closure rate: the user's command before contact, the grip force
   regulator afterwards,
4. adopt a newly selected grasp if the hand is still open, then integrate
   closure, evaluate forward kinematics, and take the span between the opposing
   surfaces of the grasp,
5. turn the overlap of the object into a normal force through the compliant
   contact model,
6. compare the friction capacity with the tangential load and advance the slip
   state of the object,
7. synthesise the tactile signal and run the slip detector,
8. raise the force demand if slip was reported.

The loop knows nothing about the object beyond the force it measures and the
tactile signal it receives. Everything else, the mass, the friction coefficient
and the stiffness, acts only through the physics.

An object that slides past the drop distance has left the hand, and the trial
ends on that sample. The trace is therefore as long as the trial lasted rather
than as long as it was configured for, ``released`` says which of the two
happened, and every recorded quantity describes an object that was still between
the fingers. An earlier version kept integrating the slide after the object was
gone, which reported a failed trial as a slide of metres at six metres per
second and let the force loop go on climbing its demand ladder against an object
that was already on the floor.

Simplifications recorded here and in the design notes: the grasp is reduced to
one span and one normal force rather than a wrench on a rigid body, and the
object slides along a single tangential axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from hand_controller.algorithm.force import (
    ForceConfig,
    ProportionalForceRegulator,
    SlipResponseConfig,
)
from hand_controller.algorithm.modeswitch import CoContractionConfig, MyoelectricController
from hand_controller.algorithm.proportional import ProportionalConfig
from hand_controller.algorithm.protocols import GripForceRegulator, SlipDetector
from hand_controller.algorithm.slip import BandPassSlipDetector, SlipDetectorConfig
from hand_controller.model.anatomy import HandModel, default_hand
from hand_controller.model.contact import (
    GRAVITY,
    FingertipPad,
    contact_force,
    default_pad,
    detect_contact,
    effective_stiffness,
    friction_capacity,
    penetration,
)
from hand_controller.model.grasps import (
    GRASP_TAXONOMY,
    GraspDefinition,
    grasp,
    opposition_span,
)
from hand_controller.model.objects import Feasibility, GraspObject, feasibility, grasp_object
from hand_controller.pipeline.emg import EmgProfile, reaching_profile

__all__ = [
    "ControllerConfig",
    "GraspPhase",
    "GraspTrace",
    "PlantConfig",
    "TactileConfig",
    "TrialConfig",
    "simulate",
    "tactile_signal",
]


class GraspPhase(IntEnum):
    """Stage of the grasp, following the structure of Romano et al. (2011)."""

    REACH = 0
    LOAD = 1
    HOLD = 2


@dataclass(frozen=True, slots=True)
class TactileConfig:
    """The simulated tactile sensor.

    The sensor reports the normal force plus a band limited vibration whose
    amplitude is proportional to the sliding speed, which is the signature Howe
    and Cutkosky (1993) use for slip, plus white sensor noise.
    """

    slip_gain: float = 200.0
    slip_frequency: float = 60.0
    noise_std: float = 0.005
    seed: int = 4711

    def __post_init__(self) -> None:
        if self.slip_gain < 0.0:
            raise ValueError(f"slip gain must not be negative, got {self.slip_gain}")
        if self.slip_frequency <= 0.0:
            raise ValueError(f"slip frequency must be positive, got {self.slip_frequency}")
        if self.noise_std < 0.0:
            raise ValueError(f"noise standard deviation must not be negative, got {self.noise_std}")


def tactile_signal(
    config: TactileConfig, normal_force: float, slip_velocity: float, time: float, noise: float
) -> float:
    """Return one tactile sample."""
    phase = 2.0 * math.pi * config.slip_frequency * time
    vibration = config.slip_gain * slip_velocity * math.sin(phase)
    return normal_force + vibration + noise


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    """Every parameter of the controller, gathered in one place."""

    proportional: ProportionalConfig = field(default_factory=ProportionalConfig)
    co_contraction: CoContractionConfig = field(default_factory=CoContractionConfig)
    force: ForceConfig = field(default_factory=ForceConfig)
    slip_response: SlipResponseConfig = field(default_factory=SlipResponseConfig)
    slip_detector: SlipDetectorConfig = field(default_factory=SlipDetectorConfig)


@dataclass(frozen=True, slots=True)
class PlantConfig:
    """Every parameter of the simulated hand and sensor."""

    pad: FingertipPad = field(default_factory=default_pad)
    tactile: TactileConfig = field(default_factory=TactileConfig)
    drop_distance: float = 0.020
    """Slide at which the object has left the hand and the trial ends."""
    slip_speed_threshold: float = 1.0e-4


@dataclass(frozen=True, slots=True)
class TrialConfig:
    """One grasp trial: which grasp, which object, and how it is driven."""

    grasp: str
    object_name: str
    duration: float = 3.000
    dt: float = 0.001
    lift_time: float = 1.000
    """Time at which the object takes up its own weight. Equal to ``duration`` means never."""
    emg: EmgProfile = field(default_factory=reaching_profile)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    plant: PlantConfig = field(default_factory=PlantConfig)

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.duration <= self.dt:
            raise ValueError(f"duration must exceed one step, got {self.duration}")
        if not 0.0 <= self.lift_time <= self.duration:
            raise ValueError(f"lift time must lie in [0, {self.duration}], got {self.lift_time}")

    @property
    def steps(self) -> int:
        """Number of control periods in the trial."""
        return round(self.duration / self.dt)


@dataclass(frozen=True, slots=True)
class GraspTrace:
    """Everything one trial recorded, one row per control period.

    A trace runs to the configured duration unless the object left the hand
    first, in which case ``released`` is true and the last row is the sample on
    which the slide passed the drop distance.
    """

    config: TrialConfig
    grasp: GraspDefinition
    item: GraspObject
    feasible: Feasibility
    released: bool
    contact_stiffness: float
    time: NDArray[np.float64]
    emg_open: NDArray[np.float64]
    emg_close: NDArray[np.float64]
    smoothed_open: NDArray[np.float64]
    smoothed_close: NDArray[np.float64]
    command_velocity: NDArray[np.float64]
    closure: NDArray[np.float64]
    span: NDArray[np.float64]
    indentation: NDArray[np.float64]
    contact: NDArray[np.bool_]
    grip_force: NDArray[np.float64]
    commanded_force: NDArray[np.float64]
    demanded_force: NDArray[np.float64]
    friction_capacity: NDArray[np.float64]
    tangential_load: NDArray[np.float64]
    tactile: NDArray[np.float64]
    slip_energy: NDArray[np.float64]
    slip_detected: NDArray[np.bool_]
    slip_velocity: NDArray[np.float64]
    slip_displacement: NDArray[np.float64]
    phase: NDArray[np.int8]
    mode: NDArray[np.int16]

    def __len__(self) -> int:
        return int(self.time.size)


def _make_regulator(config: ForceConfig) -> GripForceRegulator:
    return ProportionalForceRegulator(config)


def _make_detector(config: SlipDetectorConfig) -> SlipDetector:
    return BandPassSlipDetector(config)


def simulate(
    config: TrialConfig,
    hand: HandModel | None = None,
    definition: GraspDefinition | None = None,
    item: GraspObject | None = None,
) -> GraspTrace:
    """Run one grasp trial and return its trace."""
    hand = hand if hand is not None else default_hand()
    definition = definition if definition is not None else grasp(config.grasp)
    item = item if item is not None else grasp_object(config.object_name)

    steps = config.steps
    dt = config.dt
    times = np.arange(steps, dtype=np.float64) * dt
    emg_open, emg_close = config.emg.envelopes(times)

    sensor = config.plant.tactile
    noise = (
        np.random.default_rng(sensor.seed).normal(0.0, sensor.noise_std, size=steps)
        if sensor.noise_std > 0.0
        else np.zeros(steps, dtype=np.float64)
    )

    # Grasp selection is part of the loop. The front end cycles through the
    # taxonomy on a confirmed co-contraction, and a new selection takes effect
    # only while the hand is still fully open, which is when a user of a
    # multi-articulating hand chooses a grasp.
    catalogue = GRASP_TAXONOMY
    initial_mode = next(
        (index for index, entry in enumerate(catalogue) if entry.name == definition.name), 0
    )
    front_end = MyoelectricController(
        config.controller.proportional,
        config.controller.co_contraction,
        dt,
        mode_count=len(catalogue),
        initial_mode=initial_mode,
    )
    regulator = _make_regulator(config.controller.force)
    detector = _make_detector(config.controller.slip_detector)

    stiffness = effective_stiffness(config.plant.pad, item.stiffness)
    contacts = definition.load_bearing_contacts

    trace: dict[str, NDArray[np.float64]] = {
        name: np.zeros(steps, dtype=np.float64)
        for name in (
            "smoothed_open",
            "smoothed_close",
            "command_velocity",
            "closure",
            "span",
            "indentation",
            "grip_force",
            "commanded_force",
            "demanded_force",
            "capacity",
            "load",
            "tactile",
            "slip_energy",
            "slip_velocity",
            "slip_displacement",
        )
    }
    contact_flags = np.zeros(steps, dtype=np.bool_)
    slip_flags = np.zeros(steps, dtype=np.bool_)
    phases = np.zeros(steps, dtype=np.int8)
    modes = np.zeros(steps, dtype=np.int16)

    closure = 0.0
    force = 0.0
    previous_indentation = penetration(opposition_span(hand, definition, 0.0), item.width)
    slip_velocity = 0.0
    slip_displacement = 0.0
    phase = GraspPhase.REACH
    refractory = 0.0
    response = config.controller.slip_response
    released = False
    recorded = steps

    for step in range(steps):
        time = float(times[step])
        output = front_end.update(float(emg_open[step]), float(emg_close[step]), dt)

        if output.switched and phase is GraspPhase.REACH and closure <= 0.0:
            definition = catalogue[output.mode]
            contacts = definition.load_bearing_contacts
            previous_indentation = penetration(opposition_span(hand, definition, 0.0), item.width)

        rate = output.velocity if phase is GraspPhase.REACH else regulator.update(force, dt)
        closure = min(max(closure + rate * dt, 0.0), 1.0)

        span = opposition_span(hand, definition, closure)
        indentation = penetration(span, item.width)
        indentation_rate = (indentation - previous_indentation) / dt
        previous_indentation = indentation
        force = contact_force(
            indentation, indentation_rate, stiffness, config.plant.pad.exponent, item.damping
        )

        if phase is GraspPhase.REACH and force >= config.controller.force.contact_force_threshold:
            phase = GraspPhase.LOAD
            regulator.demand(config.controller.force.nominal_force)
        elif (
            phase is GraspPhase.LOAD
            and abs(regulator.commanded_force - config.controller.force.nominal_force)
            <= config.controller.force.tolerance
        ):
            phase = GraspPhase.HOLD

        load = item.mass * GRAVITY if time >= config.lift_time else 0.0
        capacity = friction_capacity(force, item.friction, contacts)
        if slip_velocity > 0.0 or load > capacity:
            acceleration = (load - capacity) / item.mass
            slip_velocity = max(0.0, slip_velocity + acceleration * dt)
            slip_displacement += slip_velocity * dt

        sample = tactile_signal(sensor, force, slip_velocity, time, float(noise[step]))
        reading = detector.update(sample, dt)

        # A further increase is granted only once the previous one has actually
        # been delivered. Stacking demands the hand has not yet reached would let
        # the ladder climb on transport delay rather than on evidence of slip.
        refractory = max(0.0, refractory - dt)
        delivered = force + config.controller.force.tolerance >= regulator.target_force
        if (
            response.enabled
            and reading.detected
            and refractory <= 0.0
            and delivered
            and phase is not GraspPhase.REACH
        ):
            regulator.demand(response.raised(regulator.target_force))
            refractory = response.refractory_time

        trace["smoothed_open"][step] = output.smoothed_open
        trace["smoothed_close"][step] = output.smoothed_close
        trace["command_velocity"][step] = output.velocity
        trace["closure"][step] = closure
        trace["span"][step] = span
        trace["indentation"][step] = indentation
        trace["grip_force"][step] = force
        trace["commanded_force"][step] = regulator.commanded_force
        trace["demanded_force"][step] = regulator.target_force
        trace["capacity"][step] = capacity
        trace["load"][step] = load
        trace["tactile"][step] = sample
        trace["slip_energy"][step] = reading.energy
        trace["slip_velocity"][step] = slip_velocity
        trace["slip_displacement"][step] = slip_displacement
        contact_flags[step] = detect_contact(indentation)
        slip_flags[step] = reading.detected
        phases[step] = int(phase)
        modes[step] = output.mode

        # The object has passed the drop distance, so it is out of the hand.
        # This sample, the one on which it left, is recorded and the trial ends:
        # there is no object left to press on, to slide, or to feel.
        if slip_displacement > config.plant.drop_distance:
            released = True
            recorded = step + 1
            break

    end = slice(0, recorded)
    return GraspTrace(
        config=config,
        grasp=definition,
        item=item,
        feasible=feasibility(hand, definition, item),
        released=released,
        contact_stiffness=stiffness,
        time=times[end],
        emg_open=emg_open[end],
        emg_close=emg_close[end],
        smoothed_open=trace["smoothed_open"][end],
        smoothed_close=trace["smoothed_close"][end],
        command_velocity=trace["command_velocity"][end],
        closure=trace["closure"][end],
        span=trace["span"][end],
        indentation=trace["indentation"][end],
        contact=contact_flags[end],
        grip_force=trace["grip_force"][end],
        commanded_force=trace["commanded_force"][end],
        demanded_force=trace["demanded_force"][end],
        friction_capacity=trace["capacity"][end],
        tangential_load=trace["load"][end],
        tactile=trace["tactile"][end],
        slip_energy=trace["slip_energy"][end],
        slip_detected=slip_flags[end],
        slip_velocity=trace["slip_velocity"][end],
        slip_displacement=trace["slip_displacement"][end],
        phase=phases[end],
        mode=modes[end],
    )
