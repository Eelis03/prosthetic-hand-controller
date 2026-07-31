"""Simulated activation envelopes for the two control sites.

No real electromyogram is recorded, filtered or classified anywhere in this
project. What the controller receives is a pair of activation envelopes, which
is the signal a conventional two site prosthesis derives from its electrodes
after rectification and smoothing. This module produces them as trapezoidal
bursts with additive noise, so that a trial can be described by when the user
contracted, how hard, and on which site.

Noise is drawn from a seeded ``numpy`` generator, which reproduces bit for bit
across platforms for a given seed, so an entire trial is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np
from numpy.typing import NDArray

__all__ = ["EmgBurst", "EmgProfile", "EmgSite", "co_contraction_profile", "reaching_profile"]


class EmgSite(StrEnum):
    """Which of the two electrode sites a burst appears on."""

    OPEN = "open"
    CLOSE = "close"


@dataclass(frozen=True, slots=True)
class EmgBurst:
    """One trapezoidal contraction on one site."""

    site: EmgSite
    start: float
    rise: float
    hold: float
    fall: float
    level: float

    def __post_init__(self) -> None:
        if self.rise <= 0.0 or self.fall <= 0.0:
            raise ValueError("rise and fall must be positive")
        if self.hold < 0.0:
            raise ValueError(f"hold must not be negative, got {self.hold}")
        if not 0.0 <= self.level <= 1.0:
            raise ValueError(f"level must lie in [0, 1], got {self.level}")

    def shape(self, times: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the burst on a time grid."""
        elapsed = times - self.start
        rising = np.clip(elapsed / self.rise, 0.0, 1.0)
        falling = np.clip((elapsed - self.rise - self.hold) / self.fall, 0.0, 1.0)
        profile = np.clip(rising - falling, 0.0, 1.0)
        return self.level * profile


@dataclass(frozen=True, slots=True)
class EmgProfile:
    """A complete two site recording described by its bursts."""

    bursts: tuple[EmgBurst, ...]
    rest_level: float = 0.020
    noise_std: float = 0.010
    seed: int = 20260731

    def __post_init__(self) -> None:
        if self.noise_std < 0.0:
            raise ValueError(f"noise standard deviation must not be negative, got {self.noise_std}")
        if not 0.0 <= self.rest_level <= 1.0:
            raise ValueError(f"rest level must lie in [0, 1], got {self.rest_level}")

    def envelopes(
        self, times: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the opening and closing envelopes on ``times``, both in [0, 1]."""
        opening = np.full(times.shape, self.rest_level, dtype=np.float64)
        closing = np.full(times.shape, self.rest_level, dtype=np.float64)
        for burst in self.bursts:
            if burst.site is EmgSite.OPEN:
                opening = opening + burst.shape(times)
            else:
                closing = closing + burst.shape(times)
        if self.noise_std > 0.0:
            generator = np.random.default_rng(self.seed)
            noise = generator.normal(0.0, self.noise_std, size=(2, times.size))
            opening = opening + noise[0]
            closing = closing + noise[1]
        return np.clip(opening, 0.0, 1.0), np.clip(closing, 0.0, 1.0)


_CLOSE_LEVEL: Final[float] = 0.55


def reaching_profile(
    close_start: float = 0.050,
    close_level: float = _CLOSE_LEVEL,
    relax_time: float = 1.000,
    duration: float = 3.000,
) -> EmgProfile:
    """A single closing contraction held until the object is secure, then relaxed.

    The user drives the hand shut, and once the grip force loop has taken over
    the contraction decays. From that point the hand holds the object without any
    myoelectric input, which is the shared control arrangement of Cipriani et al.
    (2008).
    """
    hold = max(relax_time - close_start - 0.200, 0.0)
    return EmgProfile(
        bursts=(
            EmgBurst(
                site=EmgSite.CLOSE,
                start=close_start,
                rise=0.200,
                hold=hold,
                fall=0.300,
                level=close_level,
            ),
            EmgBurst(
                site=EmgSite.CLOSE,
                start=relax_time + 0.100,
                rise=0.200,
                hold=max(duration - relax_time - 0.300, 0.0),
                fall=0.200,
                level=0.080,
            ),
        )
    )


def co_contraction_profile(
    burst_times: tuple[float, ...],
    level: float = 0.700,
    hold: float = 0.250,
) -> EmgProfile:
    """Balanced bursts on both sites at each of ``burst_times``, to change grasp."""
    bursts: list[EmgBurst] = []
    for start in burst_times:
        for site in (EmgSite.OPEN, EmgSite.CLOSE):
            bursts.append(
                EmgBurst(site=site, start=start, rise=0.080, hold=hold, fall=0.080, level=level)
            )
    return EmgProfile(bursts=tuple(bursts))
