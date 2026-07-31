"""Characterise the proportional control law, its latency and its mode switching."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from hand_controller.algorithm import (
    CoContractionConfig,
    MyoelectricController,
    ProportionalConfig,
    command_latency,
    control_law,
    is_co_contraction,
    mode_switch_latency,
    saturation_difference,
)
from hand_controller.analysis.figures import plot_control_characteristic


def build_parser() -> argparse.ArgumentParser:
    """Build the command line interface of this example."""
    parser = argparse.ArgumentParser(
        description="Characterise the two site proportional control law and mode switching."
    )
    parser.add_argument("--dt", type=float, default=0.001, help="control period in seconds")
    parser.add_argument("--samples", type=int, default=4001, help="points on the characteristic")
    parser.add_argument(
        "--timing-steps", type=int, default=20000, help="control periods timed for throughput"
    )
    parser.add_argument("--figure-dir", type=Path, default=None, help="directory for figures")
    parser.add_argument("--no-figures", action="store_true", help="skip figure writing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the example."""
    arguments = build_parser().parse_args(argv)
    config = ProportionalConfig()
    switching = CoContractionConfig()

    differences = np.linspace(-1.0, 1.0, arguments.samples)
    commands = np.array([control_law(config, float(value)) for value in differences])
    inside = np.abs(differences) <= config.deadzone
    outside = np.abs(differences) > config.deadzone

    print("Proportional control law")
    print(f"dead zone           {config.deadzone:.3f} of full activation")
    print(f"gain                {config.gain:.3f} closure per second per unit activation")
    print(f"saturation          {config.saturation:.3f} closure per second")
    print(f"saturating at       {saturation_difference(config):.4f} activation difference")
    print(f"points sampled      {arguments.samples}")
    print(f"maximum command inside the dead zone {np.abs(commands[inside]).max():.3e}")
    print(f"minimum command outside it           {np.abs(commands[outside]).min():.3e}")
    print(f"monotone non decreasing              {bool(np.all(np.diff(commands) >= 0.0))}")
    print(f"peak command                         {np.abs(commands).max():.4f}")

    print()
    print("Latency")
    proportional = command_latency(config, arguments.dt)
    switch = mode_switch_latency(config, switching, arguments.dt)
    print(f"proportional path   {proportional * 1e3:.1f} ms of a {config.latency_budget * 1e3:.0f}"
          f" ms budget, met: {proportional <= config.latency_budget}")
    print(f"mode switch         {switch * 1e3:.1f} ms of a {switching.latency_budget * 1e3:.0f}"
          f" ms budget, met: {switch <= switching.latency_budget}")

    controller = MyoelectricController(config, switching, arguments.dt, mode_count=6)
    start = time.perf_counter()
    for _ in range(arguments.timing_steps):
        controller.update(0.05, 0.55, arguments.dt)
    elapsed = time.perf_counter() - start
    per_step = elapsed / arguments.timing_steps
    print(f"compute per period  {per_step * 1e6:.1f} us against a {arguments.dt * 1e6:.0f} us"
          f" period, real time factor {arguments.dt / per_step:.0f}")

    print()
    print("Co-contraction recognition, one row per input pattern")
    header = f"{'pattern':<34}{'open':>7}{'close':>7}{'recognised':>12}"
    print(header)
    print("-" * len(header))
    patterns = (
        ("rest", 0.02, 0.02),
        ("strong close only", 0.05, 0.90),
        ("strong close with crosstalk", 0.45, 0.90),
        ("strong open only", 0.90, 0.05),
        ("balanced co-contraction", 0.70, 0.70),
        ("co-contraction, slight imbalance", 0.60, 0.70),
        ("weak balanced effort", 0.20, 0.20),
    )
    for name, opening, closing in patterns:
        print(
            f"{name:<34}{opening:>7.2f}{closing:>7.2f}"
            f"{is_co_contraction(switching, opening, closing)!s:>12}"
        )

    print()
    print("Sustained patterns driven through the full front end for 0.5 s")
    header = f"{'pattern':<34}{'switches':>10}{'final velocity':>16}"
    print(header)
    print("-" * len(header))
    for name, opening, closing in patterns:
        controller = MyoelectricController(config, switching, arguments.dt, mode_count=6)
        switches = 0
        velocity = 0.0
        for _ in range(round(0.5 / arguments.dt)):
            output = controller.update(opening, closing, arguments.dt)
            switches += int(output.switched)
            velocity = output.velocity
        print(f"{name:<34}{switches:>10}{velocity:>16.4f}")

    if not arguments.no_figures:
        directory = arguments.figure_dir if arguments.figure_dir is not None else Path("figures")
        written = plot_control_characteristic(config, directory / "control_characteristic.png")
        print()
        print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
