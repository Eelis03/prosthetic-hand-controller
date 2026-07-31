"""Evaluate the controller over the whole object set and report the failures."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from hand_controller.analysis import (
    format_evaluation,
    format_set_summary,
    format_trial,
    summarise,
    summarise_set,
)
from hand_controller.analysis.figures import plot_force_profile
from hand_controller.model import required_grip_force
from hand_controller.pipeline import run_evaluation


def build_parser() -> argparse.ArgumentParser:
    """Build the command line interface of this example."""
    parser = argparse.ArgumentParser(
        description="Run every object of the evaluation set and report the outcome."
    )
    parser.add_argument("--duration", type=float, default=3.0, help="trial length in seconds")
    parser.add_argument("--dt", type=float, default=0.001, help="control period in seconds")
    parser.add_argument("--lift-time", type=float, default=1.0, help="time the object is lifted")
    parser.add_argument("--figure-dir", type=Path, default=None, help="directory for figures")
    parser.add_argument("--no-figures", action="store_true", help="skip figure writing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the example."""
    arguments = build_parser().parse_args(argv)
    traces = run_evaluation(
        duration=arguments.duration, dt=arguments.dt, lift_time=arguments.lift_time
    )
    metrics = tuple(summarise(trace) for trace in traces)

    print("Grasp evaluation over the object set")
    print(format_evaluation(metrics))
    print()
    print(format_set_summary(summarise_set(metrics)))

    failures = [entry for entry in metrics if not entry.success]
    print()
    if not failures:
        print("No failures.")
    for entry in failures:
        trace = next(item for item in traces if item.item.name == entry.object_name)
        print(f"Failure case: {entry.object_name}")
        print(format_trial(trace, entry))
        limit = trace.config.controller.force.safety_limit
        required = required_grip_force(
            trace.item.mass, trace.item.friction, trace.grasp.load_bearing_contacts
        )
        print(
            f"explanation         holding this object needs {required:.2f} N per contact, "
            f"which is {required / limit:.2f} times the {limit:.1f} N safety limit"
        )
        print()

    if not arguments.no_figures:
        directory = arguments.figure_dir if arguments.figure_dir is not None else Path("figures")
        for entry in failures:
            trace = next(item for item in traces if item.item.name == entry.object_name)
            written = plot_force_profile(trace, directory / f"failure_{entry.object_name}.png")
            print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
