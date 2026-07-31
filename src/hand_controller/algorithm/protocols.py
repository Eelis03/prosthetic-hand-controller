"""Interfaces the closed loop depends on.

The simulation talks to a control law, a grip force regulator and a slip
detector only through these protocols, so any of the three can be replaced
without touching the loop. Nothing here imports the model or the pipeline, and
nothing here performs input or output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = ["ControlLaw", "GripForceRegulator", "SlipDetector", "SlipReading"]


@dataclass(frozen=True, slots=True)
class SlipReading:
    """One sample of the slip detector output."""

    energy: float
    detected: bool


@runtime_checkable
class ControlLaw(Protocol):
    """A memoryless map from a two site activation difference to a velocity."""

    def command(self, difference: float) -> float:
        """Return the commanded closure rate for an activation difference."""
        ...


@runtime_checkable
class GripForceRegulator(Protocol):
    """A regulator that drives the measured grip force towards a demand."""

    @property
    def commanded_force(self) -> float:
        """The force the regulator is currently asking for, after ramping."""
        ...

    @property
    def target_force(self) -> float:
        """The force the ramp is heading towards."""
        ...

    def demand(self, force: float) -> None:
        """Set the force the regulator should ramp towards."""
        ...

    def update(self, measured_force: float, dt: float) -> float:
        """Return the closure rate that reduces the force error."""
        ...

    def reset(self) -> None:
        """Return the regulator to its initial state."""
        ...


@runtime_checkable
class SlipDetector(Protocol):
    """A detector that reports slip from a tactile signal."""

    def update(self, tactile: float, dt: float) -> SlipReading:
        """Consume one tactile sample and report whether slip is present."""
        ...

    def reset(self) -> None:
        """Return the detector to its initial state."""
        ...
