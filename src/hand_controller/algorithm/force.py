"""Grip force regulation and the response to slip.

The grasp is run in the phases Romano et al. (2011) use for a tactile robotic
grasp controller: the hand closes under the user's command until contact is
detected, the demanded force is ramped to a nominal value, and from then on the
force is held and raised only when slip is reported. The controller is told
nothing about the object. It does not know the mass, the friction coefficient or
the stiffness, so the nominal force is deliberately light and the slip response
is what makes a heavy or slippery object holdable. That is the same division of
labour as in the human precision grip, where the grip force is set just above
the slip ratio and raised within about seventy milliseconds of a slip signal
from the tactile afferents (Johansson and Westling, 1984, 1987).

Control law. The actuator is commanded in closure rate, and the force depends on
closure through a static compliant contact. The plant from rate to force
therefore already contains an integrator, so a proportional law has no steady
state error and an integral term would only add windup. What the loop does need
is a gain that does not change by two orders of magnitude between a glass and a
foam cup. The local sensitivity of force to closure is estimated online from the
increments the loop itself produces, and the proportional gain is divided by it,
which leaves a loop whose closed form response is ``f_dot = k_p e`` for both.

Safety. Every demand passes through ``_clamp``, which rejects values that are not
finite and restricts the rest to ``[0, safety_limit]``. The limit is 15 N per
contact by default, at the low end of the 15 N to 35 N pinch forces reported for
commercial multi-articulating hands by Belter et al. (2013). No path in this
module can produce a commanded force outside that interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["ForceConfig", "ProportionalForceRegulator", "SlipResponseConfig"]


@dataclass(frozen=True, slots=True)
class ForceConfig:
    """Parameters of the grip force loop.

    ``proportional_gain`` is in reciprocal seconds because it acts on the plant
    after the online stiffness estimate has normalised it: the closed loop force
    error decays with time constant ``1 / proportional_gain``.
    """

    nominal_force: float = 1.20
    contact_force_threshold: float = 0.05
    ramp_rate: float = 60.0
    proportional_gain: float = 120.0
    closure_rate_limit: float = 1.25
    safety_limit: float = 15.0
    tolerance: float = 0.05
    initial_stiffness: float = 400.0
    minimum_stiffness: float = 50.0
    maximum_stiffness: float = 20000.0
    stiffness_time_constant: float = 0.030

    def __post_init__(self) -> None:
        if self.safety_limit <= 0.0:
            raise ValueError(f"safety limit must be positive, got {self.safety_limit}")
        if not 0.0 < self.nominal_force <= self.safety_limit:
            raise ValueError(
                f"nominal force must lie in (0, {self.safety_limit}], got {self.nominal_force}"
            )
        if self.ramp_rate <= 0.0:
            raise ValueError(f"ramp rate must be positive, got {self.ramp_rate}")
        if self.proportional_gain <= 0.0:
            raise ValueError(f"proportional gain must be positive, got {self.proportional_gain}")
        if self.tolerance <= 0.0:
            raise ValueError(f"tolerance must be positive, got {self.tolerance}")
        if not 0.0 < self.minimum_stiffness <= self.maximum_stiffness:
            raise ValueError("stiffness bounds must satisfy 0 < minimum <= maximum")


@dataclass(frozen=True, slots=True)
class SlipResponseConfig:
    """How much the demand rises on each detected slip.

    A multiplicative step with an additive floor reproduces the shape of the
    human grip force upgrade, which scales with the force already applied
    (Johansson and Westling, 1987). The refractory interval stops one slip
    episode from being counted many times while the force is still rising.
    """

    enabled: bool = True
    multiplier: float = 2.00
    increment: float = 0.20
    refractory_time: float = 0.150

    def __post_init__(self) -> None:
        if self.multiplier < 1.0:
            raise ValueError(f"multiplier must be at least 1, got {self.multiplier}")
        if self.increment < 0.0:
            raise ValueError(f"increment must not be negative, got {self.increment}")
        if self.refractory_time < 0.0:
            raise ValueError(f"refractory time must not be negative, got {self.refractory_time}")

    def raised(self, force: float) -> float:
        """Return the demand one slip response step above ``force``."""
        return force * self.multiplier + self.increment


class ProportionalForceRegulator:
    """Grip force regulator with an online plant gain estimate.

    Implements the ``GripForceRegulator`` protocol.
    """

    def __init__(self, config: ForceConfig) -> None:
        self._config = config
        self._demand = 0.0
        self._applied = 0.0
        self._stiffness = config.initial_stiffness
        self._previous_force = 0.0
        self._previous_step = 0.0
        self._started = False

    @property
    def config(self) -> ForceConfig:
        """The configuration in force."""
        return self._config

    @property
    def commanded_force(self) -> float:
        """The ramped demand, always inside ``[0, safety_limit]``."""
        return self._applied

    @property
    def target_force(self) -> float:
        """The demand the ramp is heading towards."""
        return self._demand

    @property
    def stiffness_estimate(self) -> float:
        """Current estimate of force change per unit closure, in newtons."""
        return self._stiffness

    def _clamp(self, force: float) -> float:
        if not math.isfinite(force):
            return 0.0
        return min(max(force, 0.0), self._config.safety_limit)

    def demand(self, force: float) -> None:
        """Set the force the ramp should head towards, clamped to the safety limit."""
        self._demand = self._clamp(force)

    def reset(self) -> None:
        """Return the regulator to its initial state."""
        self._demand = 0.0
        self._applied = 0.0
        self._stiffness = self._config.initial_stiffness
        self._previous_force = 0.0
        self._previous_step = 0.0
        self._started = False

    def _update_stiffness(self, measured_force: float, dt: float) -> None:
        """Refresh the estimate of force change per unit closure.

        The increment used is the one the regulator itself commanded on the
        previous step, which the actuator follows except at the travel limits.
        The estimate is bounded on both sides so that a step taken against a
        limit cannot drive the loop gain anywhere dangerous.
        """
        if abs(self._previous_step) <= 0.0:
            return
        raw = (measured_force - self._previous_force) / self._previous_step
        bounded = min(max(raw, self._config.minimum_stiffness), self._config.maximum_stiffness)
        weight = dt / (self._config.stiffness_time_constant + dt)
        self._stiffness += weight * (bounded - self._stiffness)

    def update(self, measured_force: float, dt: float) -> float:
        """Return the closure rate that drives the measured force to the demand."""
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        measured = measured_force if math.isfinite(measured_force) else 0.0

        step = self._config.ramp_rate * dt
        self._applied += min(max(self._demand - self._applied, -step), step)
        self._applied = self._clamp(self._applied)

        if self._started:
            self._update_stiffness(measured, dt)
        self._started = True

        error = self._applied - measured
        rate = self._config.proportional_gain * error / self._stiffness
        limit = self._config.closure_rate_limit
        rate = min(max(rate, -limit), limit)

        self._previous_force = measured
        self._previous_step = rate * dt
        return rate
