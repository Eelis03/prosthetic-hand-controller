"""Compare the grip force loop on a rigid object and on a deformable one."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from hand_controller.analysis import format_trial, summarise
from hand_controller.analysis.figures import plot_force_profile
from hand_controller.model import (
    default_hand,
    default_pad,
    effective_stiffness,
    equilibrium_force,
    grasp,
    grasp_object,
    indentation_for_force,
    opposition_span,
)
from hand_controller.pipeline import reference_trial, simulate


def build_parser() -> argparse.ArgumentParser:
    """Build the command line interface of this example."""
    parser = argparse.ArgumentParser(
        description="Run the grip force loop on a rigid object and a deformable one."
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        default=["drinking_glass", "paper_cup_full", "foam_cup"],
        help="objects to run, in order",
    )
    parser.add_argument("--duration", type=float, default=3.0, help="trial length in seconds")
    parser.add_argument("--dt", type=float, default=0.001, help="control period in seconds")
    parser.add_argument("--figure-dir", type=Path, default=None, help="directory for figures")
    parser.add_argument("--no-figures", action="store_true", help="skip figure writing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the example."""
    arguments = build_parser().parse_args(argv)
    hand = default_hand()
    pad = default_pad()

    print("Static equilibrium at one commanded span, before any control")
    print("The grasp is medium_wrap and the span is held 2.0 mm inside a 65 mm object.")
    header = f"{'stiffness_N/m^1.5':>18}{'k_eff':>12}{'force_N':>10}{'travel_for_2N_mm':>18}"
    print(header)
    print("-" * len(header))
    width = 0.065
    indentation = 0.002
    for stiffness, label in ((5.0e7, "rigid glass"), (8.0e3, "wet paper"), (3.0e3, "foam")):
        effective = effective_stiffness(pad, stiffness)
        force = equilibrium_force(width - 2.0 * indentation, width, pad, stiffness)
        travel = indentation_for_force(2.0, effective, pad.exponent)
        print(f"{stiffness:>18.3g}{effective:>12.4g}{force:>10.3f}{travel * 1e3:>18.3f}   {label}")

    traces = []
    for name in arguments.objects:
        config = reference_trial(name, duration=arguments.duration, dt=arguments.dt)
        trace = simulate(config)
        metrics = summarise(trace)
        traces.append((trace, metrics))
        print()
        print(format_trial(trace, metrics))

    print()
    print("Finger travel each object needed between contact and the held force")
    header = f"{'object':<16}{'k_eff':>12}{'indent_mm':>11}{'closure':>9}{'force_N':>9}"
    print(header)
    print("-" * len(header))
    for trace, metrics in traces:
        contact = np.nonzero(trace.contact)[0]
        start = int(contact[0]) if contact.size else 0
        print(
            f"{metrics.object_name:<16}{trace.contact_stiffness:>12.4g}"
            f"{trace.indentation[-1] * 1e3:>11.3f}"
            f"{trace.closure[-1] - trace.closure[start]:>9.4f}{metrics.final_force:>9.3f}"
        )

    definition = grasp("medium_wrap")
    item = grasp_object("foam_cup")
    print()
    print(
        f"Span of medium_wrap at closure 0.5: "
        f"{opposition_span(hand, definition, 0.5) * 1e3:.3f} mm, "
        f"{item.name} width {item.width * 1e3:.1f} mm"
    )

    if not arguments.no_figures:
        directory = arguments.figure_dir if arguments.figure_dir is not None else Path("figures")
        for trace, metrics in traces:
            written = plot_force_profile(trace, directory / f"force_{metrics.object_name}.png")
            print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
