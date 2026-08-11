"""Regenerate the two figures the README embeds, into a tracked directory.

This is the one command that rewrites `docs/figures`. The figures are snapshots
of a run, not test fixtures: Matplotlib output is not byte reproducible across
platforms or versions, so nothing compares them, and the size budget is checked
here rather than in CI.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from hand_controller.analysis.figures import plot_grasp_postures, plot_slip_recovery
from hand_controller.model import default_hand, grasp
from hand_controller.pipeline import reference_trial, simulate, without_slip_response

BUDGET_BYTES = 250 * 1024
"""Total size the published figures must stay inside."""


def build_parser() -> argparse.ArgumentParser:
    """Build the command line interface of this example."""
    parser = argparse.ArgumentParser(
        description="Write the figures the README embeds and report their size."
    )
    parser.add_argument(
        "--figure-dir", type=Path, default=Path("docs/figures"), help="directory for figures"
    )
    parser.add_argument("--object", default="plastic_bottle", help="object of the slip figure")
    parser.add_argument("--duration", type=float, default=1.4, help="trial length in seconds")
    parser.add_argument("--dt", type=float, default=0.001, help="control period in seconds")
    parser.add_argument(
        "--widths",
        type=float,
        nargs="+",
        default=[0.070, 0.050, 0.030],
        help="object widths of the posture figure, in metres",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the example."""
    arguments = build_parser().parse_args(argv)
    directory: Path = arguments.figure_dir

    config = reference_trial(arguments.object, duration=arguments.duration, dt=arguments.dt)
    written = [
        plot_slip_recovery(
            simulate(config),
            simulate(without_slip_response(config)),
            directory / "slip_recovery.png",
        ),
        plot_grasp_postures(
            default_hand(),
            grasp("medium_wrap"),
            tuple(arguments.widths),
            directory / "grasp_postures.png",
        ),
    ]

    total = 0
    for path in written:
        size = path.stat().st_size
        total += size
        print(f"wrote {path} ({size} bytes)")
    print(f"total {total} bytes of a {BUDGET_BYTES} byte budget")
    if total > BUDGET_BYTES:
        print("over budget: reduce the resolution or the canvas size")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
