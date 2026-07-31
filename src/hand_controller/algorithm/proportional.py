"""Two site proportional myoelectric control.

A conventional two site system places one electrode over a flexor and one over
an extensor, rectifies and smooths each channel into an activation envelope, and
maps the difference of the two envelopes to a velocity command for a single
degree of freedom (Fougner et al., 2012; Parker et al., 2006). The map has three
parts and this module implements exactly those three:

* a dead zone, so that resting activity and electrode noise command nothing,
* a gain, so that stronger contraction closes faster,
* a saturation, so that the command cannot exceed the actuator's rate.

The dead zone is applied with rescaling: the surviving activation is stretched
back over the full range, which keeps the command continuous at the edge of the
dead zone instead of stepping to ``deadzone * gain``.

Latency budget. Farrell and Weir (2007) measured the controller delay that users
of a myoelectric prosthesis tolerate and found the optimum between 100 ms and
125 ms, with performance degrading above it. The budget adopted here is
therefore 100 ms from a change in the envelope to 90 percent of the resulting
command, and it is spent on:

* the envelope smoothing window, 50 ms,
* the command slew limit, which needs 42 ms to sweep the whole command range at
  the configured limit,
* one control period, 1 ms.

The slew limit overlaps the smoothing window rather than adding to it, because
the smoothed envelope already ramps. ``command_latency`` measures the realised
figure instead of assuming it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "EnvelopeFilter",
    "ProportionalConfig",
    "ProportionalControlLaw",
    "command_latency",
    "control_law",
    "saturation_difference",
]


@dataclass(frozen=True, slots=True)
class ProportionalConfig:
    """Parameters of the two site proportional map.

    ``gain`` and ``saturation`` are in units of closure fraction per second: a
    saturation of 1.25 closes the hand from fully open to fully closed in
    0.80 s, which is within the 0.5 s to 1.5 s range that Belter et al. (2013)
    report for commercial multi-articulating hands.
    """

    deadzone: float = 0.10
    gain: float = 2.20
    saturation: float = 1.25
    envelope_window: float = 0.050
    slew_limit: float = 30.0
    latency_budget: float = 0.100

    def __post_init__(self) -> None:
        if not 0.0 <= self.deadzone < 1.0:
            raise ValueError(f"deadzone must lie in [0, 1), got {self.deadzone}")
        if self.gain <= 0.0:
            raise ValueError(f"gain must be positive, got {self.gain}")
        if self.saturation <= 0.0:
            raise ValueError(f"saturation must be positive, got {self.saturation}")
        if self.envelope_window <= 0.0:
            raise ValueError(f"envelope window must be positive, got {self.envelope_window}")
        if self.slew_limit <= 0.0:
            raise ValueError(f"slew limit must be positive, got {self.slew_limit}")


def control_law(config: ProportionalConfig, difference: float) -> float:
    """Map an activation difference to a closure rate command.

    ``difference`` is the closing envelope minus the opening envelope, both in
    [0, 1], so it lies in [-1, 1]. Positive commands close the hand. Values that
    are not finite command nothing, which is the safe response to a broken
    electrode channel.
    """
    if not math.isfinite(difference):
        return 0.0
    clipped = min(max(difference, -1.0), 1.0)
    magnitude = abs(clipped)
    if magnitude <= config.deadzone:
        return 0.0
    scaled = (magnitude - config.deadzone) / (1.0 - config.deadzone)
    command = math.copysign(config.gain * scaled, clipped)
    return min(max(command, -config.saturation), config.saturation)


def saturation_difference(config: ProportionalConfig) -> float:
    """Return the smallest activation difference that saturates the command."""
    return config.deadzone + (1.0 - config.deadzone) * config.saturation / config.gain


@dataclass(frozen=True, slots=True)
class ProportionalControlLaw:
    """The memoryless part of the controller, as a ``ControlLaw``."""

    config: ProportionalConfig

    def command(self, difference: float) -> float:
        """Map an activation difference to a closure rate command."""
        return control_law(self.config, difference)


class EnvelopeFilter:
    """Moving average smoother for the pair of activation envelopes.

    A rectangular window is used rather than an exponential one so that the
    contribution of the latency budget is exactly the window length rather than
    a settling time that has to be quoted with a tolerance.
    """

    def __init__(self, window: float, dt: float) -> None:
        if window <= 0.0:
            raise ValueError(f"window must be positive, got {window}")
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        self._length = max(1, round(window / dt))
        self._buffer: NDArray[np.float64] = np.zeros((2, self._length), dtype=np.float64)
        self._cursor = 0

    @property
    def length(self) -> int:
        """Number of samples in the window."""
        return self._length

    @property
    def group_delay(self) -> float:
        """Delay of the rectangular window in samples, as a sample count."""
        return 0.5 * (self._length - 1)

    def reset(self) -> None:
        """Empty the window."""
        self._buffer.fill(0.0)
        self._cursor = 0

    def update(self, open_envelope: float, close_envelope: float) -> tuple[float, float]:
        """Push one sample pair and return the smoothed pair."""
        opening = open_envelope if math.isfinite(open_envelope) else 0.0
        closing = close_envelope if math.isfinite(close_envelope) else 0.0
        self._buffer[0, self._cursor] = min(max(opening, 0.0), 1.0)
        self._buffer[1, self._cursor] = min(max(closing, 0.0), 1.0)
        self._cursor = (self._cursor + 1) % self._length
        means = self._buffer.mean(axis=1)
        return float(means[0]), float(means[1])


def command_latency(
    config: ProportionalConfig,
    dt: float,
    *,
    close_level: float = 0.50,
    fraction: float = 0.90,
    horizon: float = 1.0,
) -> float:
    """Measure the time from a step in the closing envelope to ``fraction`` of the command.

    The filter starts empty, the closing envelope steps to ``close_level`` at the
    first sample, and the returned value is the first time at which the slew
    limited command reaches ``fraction`` of its steady state. ``close_level``
    defaults to a value that does not saturate the command, because a saturating
    step would reach its target early and understate the delay.
    """
    steady = control_law(config, close_level)
    if steady <= 0.0:
        raise ValueError(f"close_level {close_level} does not command any motion")
    target = fraction * steady

    envelope = EnvelopeFilter(config.envelope_window, dt)
    command = 0.0
    steps = round(horizon / dt)
    for step in range(1, steps + 1):
        _, closing = envelope.update(0.0, close_level)
        desired = control_law(config, closing)
        limit = config.slew_limit * dt
        command += min(max(desired - command, -limit), limit)
        if command >= target:
            return step * dt
    raise RuntimeError(
        f"the command did not reach {fraction:.0%} of {steady:.4f} within {horizon} s"
    )
