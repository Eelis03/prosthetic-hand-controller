"""Print the grasp taxonomy, the object set, and which grasp can hold what."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from hand_controller.analysis import format_feasibility, format_taxonomy
from hand_controller.model import (
    OBJECT_SET,
    default_hand,
    default_pad,
    effective_stiffness,
    grasp,
    required_grip_force,
)
from hand_controller.pipeline import EVALUATION_SET


def build_parser() -> argparse.ArgumentParser:
    """Build the command line interface of this example."""
    parser = argparse.ArgumentParser(
        description="Print the grasp taxonomy and the object properties it is tested on."
    )
    parser.add_argument(
        "--no-feasibility", action="store_true", help="skip the grasp by object feasibility table"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the example."""
    arguments = build_parser().parse_args(argv)
    hand = default_hand()
    pad = default_pad()

    print("Grasp taxonomy, numbered as in Feix et al. (2016)")
    print(format_taxonomy(hand))

    print()
    print("Object set")
    header = (
        f"{'object':<16}{'width_mm':>9}{'mass_kg':>9}{'stiff_N/m^1.5':>15}"
        f"{'k_eff':>12}{'friction':>9}{'crush_N':>9}"
    )
    print(header)
    print("-" * len(header))
    for item in OBJECT_SET:
        print(
            f"{item.name:<16}{item.width * 1e3:>9.1f}{item.mass:>9.3f}"
            f"{item.stiffness:>15.3g}{effective_stiffness(pad, item.stiffness):>12.4g}"
            f"{item.friction:>9.2f}{item.crush_force:>9.1f}"
        )

    print()
    print("Grip force each object needs, given the grasp the evaluation set uses")
    header = f"{'object':<16}{'grasp':<16}{'contacts':>9}{'weight_N':>10}{'required_N':>12}"
    print(header)
    print("-" * len(header))
    for object_name, grasp_name in EVALUATION_SET:
        item = next(entry for entry in OBJECT_SET if entry.name == object_name)
        definition = grasp(grasp_name)
        contacts = definition.load_bearing_contacts
        required = required_grip_force(item.mass, item.friction, contacts)
        print(
            f"{object_name:<16}{grasp_name:<16}{contacts:>9}"
            f"{item.mass * 9.80665:>10.3f}{required:>12.3f}"
        )

    if not arguments.no_feasibility:
        print()
        print("Which grasp can enclose which object")
        print(format_feasibility(hand))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
