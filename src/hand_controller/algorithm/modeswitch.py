"""Grasp selection by co-contraction on a two site interface.

A two site interface has one degree of freedom, so selecting between grasps
needs a signal that is not part of the proportional command. The standard one is
co-contraction: the user tenses flexor and extensor together, which produces a
pattern the differential map ignores because the two envelopes cancel (Fougner
et al., 2012; Scheme and Englehart, 2011).

Recognising it needs three conditions, and the third is the one that matters:

1. both envelopes are above an activation threshold,
2. the envelopes are balanced, that is their difference is small,
3. both hold for a confirmation time.

Condition 2 is what separates a genuine co-contraction from a strong single site
contraction. Without it, any hard closing effort with a little crosstalk on the
opening channel would change grasp in the middle of a task. That failure mode is
tested explicitly.

While a co-contraction is being confirmed, and for a refractory interval after a
switch, the proportional command is gated to zero so that the residual imbalance
between the two channels cannot drive the hand.

Latency budget. The switch is a discrete event rather than a tracking command,
so it can afford more delay than the proportional path; the budget adopted here
is 250 ms, spent on the 50 ms envelope window, the 150 ms confirmation hold and
one control period. ``mode_switch_latency`` measures the realised figure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from hand_controller.algorithm.proportional import (
    EnvelopeFilter,
    ProportionalConfig,
    control_law,
)

__all__ = [
    "CoContractionConfig",
    "ModeSwitchResult",
    "ModeSwitcher",
    "MyoelectricController",
    "MyoelectricOutput",
    "SwitchState",
    "is_co_contraction",
    "mode_switch_latency",
]


class SwitchState(StrEnum):
    """State of the co-contraction recogniser."""

    IDLE = "idle"
    ARMING = "arming"
    REFRACTORY = "refractory"


@dataclass(frozen=True, slots=True)
class CoContractionConfig:
    """Thresholds and timings of the co-contraction recogniser."""

    activation_threshold: float = 0.40
    balance_tolerance: float = 0.15
    hold_time: float = 0.150
    refractory_time: float = 0.400
    latency_budget: float = 0.250

    def __post_init__(self) -> None:
        if not 0.0 < self.activation_threshold <= 1.0:
            raise ValueError(
                f"activation threshold must lie in (0, 1], got {self.activation_threshold}"
            )
        if self.balance_tolerance < 0.0:
            raise ValueError(
                f"balance tolerance must not be negative, got {self.balance_tolerance}"
            )
        if self.hold_time <= 0.0:
            raise ValueError(f"hold time must be positive, got {self.hold_time}")
        if self.refractory_time < 0.0:
            raise ValueError(f"refractory time must not be negative, got {self.refractory_time}")


def is_co_contraction(
    config: CoContractionConfig, open_envelope: float, close_envelope: float
) -> bool:
    """True when both envelopes are active and balanced.

    Both conditions are needed. A strong single site contraction satisfies the
    activation threshold on one channel only, and a strong contraction with
    crosstalk can satisfy it on both while leaving the envelopes far apart.
    Neither is a co-contraction.
    """
    if not (math.isfinite(open_envelope) and math.isfinite(close_envelope)):
        return False
    if open_envelope < config.activation_threshold:
        return False
    if close_envelope < config.activation_threshold:
        return False
    return abs(close_envelope - open_envelope) <= config.balance_tolerance


@dataclass(frozen=True, slots=True)
class ModeSwitchResult:
    """What the recogniser concluded from one sample."""

    mode: int
    switched: bool
    state: SwitchState
    gated: bool


class ModeSwitcher:
    """Cycles through the available grasps on a confirmed co-contraction."""

    def __init__(self, config: CoContractionConfig, mode_count: int, initial_mode: int = 0) -> None:
        if mode_count < 1:
            raise ValueError(f"mode_count must be at least 1, got {mode_count}")
        if not 0 <= initial_mode < mode_count:
            raise ValueError(f"initial_mode must lie in [0, {mode_count}), got {initial_mode}")
        self._config = config
        self._mode_count = mode_count
        self._initial_mode = initial_mode
        self._mode = initial_mode
        self._held = 0.0
        self._refractory = 0.0

    @property
    def config(self) -> CoContractionConfig:
        """The configuration in force."""
        return self._config

    @property
    def mode(self) -> int:
        """The grasp index currently selected."""
        return self._mode

    def reset(self) -> None:
        """Return to the initial grasp and clear both timers."""
        self._mode = self._initial_mode
        self._held = 0.0
        self._refractory = 0.0

    def update(self, open_envelope: float, close_envelope: float, dt: float) -> ModeSwitchResult:
        """Consume one smoothed envelope pair and report the selected grasp."""
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        if self._refractory > 0.0:
            self._refractory = max(0.0, self._refractory - dt)
            self._held = 0.0
            state = SwitchState.REFRACTORY if self._refractory > 0.0 else SwitchState.IDLE
            return ModeSwitchResult(self._mode, False, state, gated=True)

        if not is_co_contraction(self._config, open_envelope, close_envelope):
            self._held = 0.0
            return ModeSwitchResult(self._mode, False, SwitchState.IDLE, gated=False)

        self._held += dt
        if self._held + 1e-12 < self._config.hold_time:
            return ModeSwitchResult(self._mode, False, SwitchState.ARMING, gated=True)

        self._mode = (self._mode + 1) % self._mode_count
        self._held = 0.0
        self._refractory = self._config.refractory_time
        state = SwitchState.REFRACTORY if self._refractory > 0.0 else SwitchState.IDLE
        return ModeSwitchResult(self._mode, True, state, gated=True)


@dataclass(frozen=True, slots=True)
class MyoelectricOutput:
    """Everything the two site front end produces in one control period."""

    velocity: float
    mode: int
    switched: bool
    state: SwitchState
    smoothed_open: float
    smoothed_close: float


class MyoelectricController:
    """The complete two site front end: smoothing, proportional map, mode switching."""

    def __init__(
        self,
        proportional: ProportionalConfig,
        co_contraction: CoContractionConfig,
        dt: float,
        mode_count: int = 1,
        initial_mode: int = 0,
    ) -> None:
        self._proportional = proportional
        self._filter = EnvelopeFilter(proportional.envelope_window, dt)
        self._switcher = ModeSwitcher(co_contraction, mode_count, initial_mode)
        self._velocity = 0.0

    @property
    def proportional(self) -> ProportionalConfig:
        """The proportional configuration in force."""
        return self._proportional

    @property
    def mode(self) -> int:
        """The grasp index currently selected."""
        return self._switcher.mode

    def reset(self) -> None:
        """Clear the smoother, the recogniser and the slew limiter."""
        self._filter.reset()
        self._switcher.reset()
        self._velocity = 0.0

    def update(self, open_envelope: float, close_envelope: float, dt: float) -> MyoelectricOutput:
        """Consume one raw envelope pair and return the command and the selected grasp."""
        opening, closing = self._filter.update(open_envelope, close_envelope)
        switch = self._switcher.update(opening, closing, dt)
        desired = 0.0 if switch.gated else control_law(self._proportional, closing - opening)
        limit = self._proportional.slew_limit * dt
        self._velocity += min(max(desired - self._velocity, -limit), limit)
        return MyoelectricOutput(
            velocity=self._velocity,
            mode=switch.mode,
            switched=switch.switched,
            state=switch.state,
            smoothed_open=opening,
            smoothed_close=closing,
        )


def mode_switch_latency(
    proportional: ProportionalConfig,
    co_contraction: CoContractionConfig,
    dt: float,
    *,
    level: float = 0.70,
    horizon: float = 2.0,
) -> float:
    """Measure the time from the onset of a co-contraction to the grasp changing.

    Both envelopes step to ``level`` at the first sample and the returned value
    is the time at which the switch fires. It includes the smoothing delay, which
    is why the measurement is made end to end rather than added up from parts.
    """
    controller = MyoelectricController(proportional, co_contraction, dt, mode_count=2)
    steps = round(horizon / dt)
    for step in range(1, steps + 1):
        output = controller.update(level, level, dt)
        if output.switched:
            return step * dt
    raise RuntimeError(f"no mode switch occurred within {horizon} s at level {level}")
