"""Report the hand model: link lengths, joint coupling and fingertip positions.

This script contains no computation of its own. It parses arguments and calls
the library.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from hand_controller.analysis.figures import plot_span_profiles
from hand_controller.model import (
    FINGERS,
    GRASP_TAXONOMY,
    Digit,
    default_hand,
    digit_points,
    fingertip_positions,
    grasp,
    joint_angles,
    limit_violations,
    opposition_span,
    reachability,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command line interface of this example."""
    parser = argparse.ArgumentParser(description="Report the hand kinematic model.")
    parser.add_argument(
        "--grasp", default="palmar_pinch", help="grasp whose fingertip positions are printed"
    )
    parser.add_argument(
        "--samples", type=int, default=201, help="closure samples used for the span figure"
    )
    parser.add_argument("--figure-dir", type=Path, default=None, help="directory for figures")
    parser.add_argument("--no-figures", action="store_true", help="skip figure writing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the example."""
    arguments = build_parser().parse_args(argv)
    hand = default_hand()

    print("Segment lengths, millimetres")
    header = f"{'digit':<10}{'proximal':>10}{'middle':>10}{'distal':>10}{'total':>10}"
    print(header)
    print("-" * len(header))
    for digit in FINGERS:
        geometry = hand.finger(digit)
        lengths = geometry.link_lengths
        print(
            f"{geometry.name:<10}{lengths[0] * 1e3:>10.1f}{lengths[1] * 1e3:>10.1f}"
            f"{lengths[2] * 1e3:>10.1f}{geometry.chain_length * 1e3:>10.1f}"
        )
    thumb = hand.thumb
    print(
        f"{thumb.name:<10}{thumb.metacarpal_length * 1e3:>10.1f}"
        f"{thumb.proximal_length * 1e3:>10.1f}{thumb.distal_length * 1e3:>10.1f}"
        f"{thumb.chain_length * 1e3:>10.1f}"
    )

    coupling = hand.finger(Digit.INDEX).coupling
    print()
    print(
        f"Finger coupling: pip = {coupling.middle_per_base:.4f} x mcp, "
        f"dip = {coupling.distal_per_middle:.4f} x pip"
    )
    thumb_coupling = hand.thumb.coupling
    print(
        f"Thumb coupling:  mcp = {thumb_coupling.middle_per_base:.4f} x cmc, "
        f"ip = {thumb_coupling.distal_per_middle:.4f} x mcp"
    )

    definition = grasp(arguments.grasp)
    print()
    print(f"Joint angles of the closed {definition.name} posture, degrees")
    closed = definition.closed_posture
    header = f"{'digit':<10}{'base':>10}{'middle':>10}{'distal':>10}"
    print(header)
    print("-" * len(header))
    for digit in (Digit.THUMB, *FINGERS):
        angles = joint_angles(hand, closed, digit)
        name = "thumb" if digit is Digit.THUMB else hand.finger(digit).name
        print(
            f"{name:<10}{math.degrees(angles[0]):>10.2f}"
            f"{math.degrees(angles[1]):>10.2f}{math.degrees(angles[2]):>10.2f}"
        )
    print(f"limit violations in this posture: {len(limit_violations(hand, closed))}")

    print()
    print(f"Fingertip positions of the closed {definition.name} posture, millimetres")
    tips = fingertip_positions(hand, closed) * 1e3
    header = f"{'digit':<10}{'x':>10}{'y':>10}{'z':>10}"
    print(header)
    print("-" * len(header))
    for digit in (Digit.THUMB, *FINGERS):
        name = "thumb" if digit is Digit.THUMB else hand.finger(digit).name
        row = tips[int(digit)]
        print(f"{name:<10}{row[0]:>10.2f}{row[1]:>10.2f}{row[2]:>10.2f}")

    index_chain = digit_points(hand, closed, Digit.INDEX)
    measured = np.linalg.norm(np.diff(index_chain, axis=0), axis=1)
    expected = np.array(hand.finger(Digit.INDEX).link_lengths)
    error = float(np.abs(measured - expected).max())
    print()
    print(f"Index segment length error from forward kinematics: {error:.3e} m")

    print()
    print("Span of each grasp and joint limit check over the whole closure sweep")
    header = f"{'grasp':<16}{'closed_mm':>11}{'open_mm':>10}{'violations':>12}{'monotone':>10}"
    print(header)
    print("-" * len(header))
    for entry in GRASP_TAXONOMY:
        spans = np.array(
            [
                opposition_span(hand, entry, float(value))
                for value in np.linspace(0.0, 1.0, arguments.samples)
            ]
        )
        monotone = bool(np.all(np.diff(spans) < 0.0))
        print(
            f"{entry.name:<16}{spans[-1] * 1e3:>11.2f}{spans[0] * 1e3:>10.2f}"
            f"{len(reachability(hand, entry)):>12}{monotone!s:>10}"
        )

    if not arguments.no_figures:
        directory = arguments.figure_dir if arguments.figure_dir is not None else Path("figures")
        written = plot_span_profiles(hand, directory / "grasp_spans.png", arguments.samples)
        print()
        print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
