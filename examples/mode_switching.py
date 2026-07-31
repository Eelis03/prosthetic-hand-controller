"""Step through the grasp taxonomy with co-contractions, in a closed loop trial."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from hand_controller.algorithm import (
    CoContractionConfig,
    MyoelectricController,
    ProportionalConfig,
)
from hand_controller.model import GRASP_NAMES
from hand_controller.pipeline import EmgProfile, co_contraction_profile, reaching_profile


def build_parser() -> argparse.ArgumentParser:
    """Build the command line interface of this example."""
    parser = argparse.ArgumentParser(
        description="Select grasps by co-contraction and show that a single site cannot."
    )
    parser.add_argument("--dt", type=float, default=0.001, help="control period in seconds")
    parser.add_argument("--duration", type=float, default=2.5, help="trial length in seconds")
    parser.add_argument(
        "--bursts",
        type=float,
        nargs="+",
        default=[0.30, 0.90, 1.50],
        help="times at which the user co-contracts",
    )
    return parser


def _run(
    profile: EmgProfile,
    config: ProportionalConfig,
    switching: CoContractionConfig,
    dt: float,
    duration: float,
    mode_count: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int16],
    list[tuple[float, int]],
]:
    """Drive the front end with one profile and record what it produced."""
    times = np.arange(round(duration / dt), dtype=np.float64) * dt
    opening, closing = profile.envelopes(times)
    controller = MyoelectricController(config, switching, dt, mode_count=mode_count)
    switches: list[tuple[float, int]] = []
    velocity = np.zeros(times.size, dtype=np.float64)
    modes = np.zeros(times.size, dtype=np.int16)
    for index, time in enumerate(times):
        output = controller.update(float(opening[index]), float(closing[index]), dt)
        velocity[index] = output.velocity
        modes[index] = output.mode
        if output.switched:
            switches.append((float(time), output.mode))
    return times, opening, closing, velocity, modes, switches


def main(argv: Sequence[str] | None = None) -> int:
    """Run the example."""
    arguments = build_parser().parse_args(argv)
    config = ProportionalConfig()
    switching = CoContractionConfig()
    mode_count = len(GRASP_NAMES)

    burst_times = tuple(float(value) for value in arguments.bursts)
    profile = co_contraction_profile(burst_times)
    _, _, _, velocity, modes, switches = _run(
        profile, config, switching, arguments.dt, arguments.duration, mode_count
    )

    print("Co-contraction sequence")
    print(f"bursts commanded at {', '.join(f'{value:.2f} s' for value in burst_times)}")
    header = f"{'burst_s':>9}{'switch_s':>10}{'lag_ms':>9}{'grasp':>18}"
    print(header)
    print("-" * len(header))
    for burst, (moment, mode) in zip(burst_times, switches, strict=True):
        print(f"{burst:>9.2f}{moment:>10.3f}{(moment - burst) * 1e3:>9.1f}{GRASP_NAMES[mode]:>18}")
    print(f"switches                {len(switches)}")
    print(f"peak commanded velocity {np.abs(velocity).max():.4f} closure per second")
    print(f"final grasp             {GRASP_NAMES[int(modes[-1])]}")

    print()
    print("The same front end driven by a strong single site contraction")
    single = reaching_profile(
        close_start=0.100,
        close_level=0.95,
        relax_time=arguments.duration,
        duration=arguments.duration,
    )
    _, opening, closing, velocity, modes, switches = _run(
        single, config, switching, arguments.dt, arguments.duration, mode_count
    )
    print(f"peak closing envelope   {closing.max():.3f}")
    print(f"peak opening envelope   {opening.max():.3f}")
    print(f"switches                {len(switches)}")
    print(f"peak commanded velocity {np.abs(velocity).max():.4f} closure per second")
    print(f"final grasp             {GRASP_NAMES[int(modes[-1])]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
