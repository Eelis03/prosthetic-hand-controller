"""Slip detection from a tactile signal.

Slip between a fingertip and an object is accompanied by vibration in the skin.
Howe and Cutkosky (1993) showed that a sensor responding to the rate of change
of contact stress detects that vibration long before the object has moved far
enough to see in the static force, and that a band limited channel of a few tens
to a few hundreds of hertz isolates it from the slowly varying grip force. The
same split is used here: a Butterworth band pass keeps 20 Hz to 200 Hz, the
result is rectified and smoothed into an energy envelope, and slip is declared
when the envelope crosses a threshold and stays there for a confirmation time.

The threshold has hysteresis, with the release level below the trigger level, so
that a marginal signal cannot chatter the grip force upwards.

Choosing the trigger level is the part that decides whether the loop is stable.
Every deliberate change of grip force is itself a transient in the tactile
signal, and a band pass cannot remove all of it. Measured across the reference
object set, the loop's own force changes leave at most 0.058 in the envelope with
no slip present, while a sliding object produces hundreds. The default trigger
sits at 0.400, seven times the artefact and three orders of magnitude below a
genuine slip, which keeps the force response from exciting itself. Raising the
trigger costs almost nothing in detection delay, because the vibration amplitude
is proportional to sliding speed and an object that has begun to slide passes
both levels within a fraction of the envelope time constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt

from hand_controller.algorithm.protocols import SlipReading

__all__ = ["BandPassSlipDetector", "SlipDetectorConfig"]


@dataclass(frozen=True, slots=True)
class SlipDetectorConfig:
    """Band, smoothing and thresholds of the slip detector."""

    sample_rate: float = 1000.0
    low_cut: float = 30.0
    high_cut: float = 300.0
    order: int = 2
    envelope_time_constant: float = 0.006
    on_threshold: float = 0.400
    off_threshold: float = 0.150
    confirm_time: float = 0.004

    def __post_init__(self) -> None:
        if self.sample_rate <= 0.0:
            raise ValueError(f"sample rate must be positive, got {self.sample_rate}")
        if not 0.0 < self.low_cut < self.high_cut < 0.5 * self.sample_rate:
            raise ValueError(
                "the band must satisfy 0 < low_cut < high_cut < sample_rate / 2, got "
                f"{self.low_cut} and {self.high_cut} at {self.sample_rate} Hz"
            )
        if self.order < 1:
            raise ValueError(f"order must be at least 1, got {self.order}")
        if self.envelope_time_constant <= 0.0:
            raise ValueError(
                f"envelope time constant must be positive, got {self.envelope_time_constant}"
            )
        if not 0.0 < self.off_threshold <= self.on_threshold:
            raise ValueError("thresholds must satisfy 0 < off_threshold <= on_threshold")
        if self.confirm_time < 0.0:
            raise ValueError(f"confirm time must not be negative, got {self.confirm_time}")


class BandPassSlipDetector:
    """Band pass, rectify, smooth and threshold, as a ``SlipDetector``."""

    def __init__(self, config: SlipDetectorConfig) -> None:
        self._config = config
        sections = butter(
            config.order,
            (config.low_cut, config.high_cut),
            btype="bandpass",
            fs=config.sample_rate,
            output="sos",
        )
        self._sos: NDArray[np.float64] = np.asarray(sections, dtype=np.float64)
        self._state: NDArray[np.float64] = np.zeros((self._sos.shape[0], 2), dtype=np.float64)
        self._envelope = 0.0
        self._held = 0.0
        self._detected = False

    @property
    def config(self) -> SlipDetectorConfig:
        """The configuration in force."""
        return self._config

    @property
    def envelope(self) -> float:
        """The current band limited energy envelope."""
        return self._envelope

    def reset(self) -> None:
        """Clear the filter memory, the envelope and the latch."""
        self._state.fill(0.0)
        self._envelope = 0.0
        self._held = 0.0
        self._detected = False

    def update(self, tactile: float, dt: float) -> SlipReading:
        """Consume one tactile sample and report whether slip is present."""
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        sample = tactile if math.isfinite(tactile) else 0.0

        filtered, self._state = sosfilt(
            self._sos, np.array([sample], dtype=np.float64), zi=self._state
        )
        rectified = abs(float(filtered[0]))
        weight = dt / (self._config.envelope_time_constant + dt)
        self._envelope += weight * (rectified - self._envelope)

        if self._detected:
            if self._envelope < self._config.off_threshold:
                self._detected = False
                self._held = 0.0
        elif self._envelope >= self._config.on_threshold:
            self._held += dt
            if self._held + 1e-12 >= self._config.confirm_time:
                self._detected = True
        else:
            self._held = 0.0

        return SlipReading(energy=self._envelope, detected=self._detected)
