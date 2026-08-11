"""Measure what the slip response is worth, by running the object set with and without it."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from hand_controller.analysis import format_slip_comparison, summarise
from hand_controller.analysis.figures import plot_slip_comparison
from hand_controller.pipeline import (
    reference_trial,
    run_evaluation,
    simulate,
    without_slip_response,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command line interface of this example."""
    parser = argparse.ArgumentParser(
        description="Measure the effect of the slip response on the whole object set."
    )
    parser.add_argument("--duration", type=float, default=3.0, help="trial length in seconds")
    parser.add_argument("--dt", type=float, default=0.001, help="control period in seconds")
    parser.add_argument("--lift-time", type=float, default=1.0, help="time the object is lifted")
    parser.add_argument(
        "--detail", default="plastic_bottle", help="object whose timeline is printed"
    )
    parser.add_argument("--figure-dir", type=Path, default=None, help="directory for figures")
    parser.add_argument("--no-figures", action="store_true", help="skip figure writing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the example."""
    arguments = build_parser().parse_args(argv)

    enabled = run_evaluation(
        duration=arguments.duration, dt=arguments.dt, lift_time=arguments.lift_time
    )
    disabled = run_evaluation(
        duration=arguments.duration,
        dt=arguments.dt,
        lift_time=arguments.lift_time,
        slip_response=False,
    )
    with_metrics = tuple(summarise(trace) for trace in enabled)
    without_metrics = tuple(summarise(trace) for trace in disabled)

    print("Slip response on against off, same objects and same disturbance")
    print(format_slip_comparison(with_metrics, without_metrics))

    print()
    print("Slip response timing, objects that actually slipped")
    header = (
        f"{'object':<16}{'onset_s':>9}{'detect_s':>10}{'detect_lag_ms':>15}"
        f"{'recovery_ms':>13}{'slip_mm':>9}{'peak_v_mm_s':>13}"
    )
    print(header)
    print("-" * len(header))
    threshold = enabled[0].config.plant.slip_speed_threshold
    for trace, metrics in zip(enabled, with_metrics, strict=True):
        moving = np.nonzero(trace.slip_velocity > threshold)[0]
        if moving.size == 0:
            continue
        detected = np.nonzero(trace.slip_detected)[0]
        onset = float(trace.time[moving[0]])
        detect = float(trace.time[detected[0]]) if detected.size else float("nan")
        recovery = (
            "n/a"
            if metrics.slip_recovery_time is None
            else f"{metrics.slip_recovery_time * 1e3:.0f}"
        )
        print(
            f"{metrics.object_name:<16}{onset:>9.3f}{detect:>10.3f}"
            f"{(detect - onset) * 1e3:>15.1f}{recovery:>13}"
            f"{metrics.total_slip * 1e3:>9.2f}{metrics.peak_slip_speed * 1e3:>13.2f}"
        )

    detail = reference_trial(
        arguments.detail,
        duration=arguments.duration,
        dt=arguments.dt,
        lift_time=arguments.lift_time,
    )
    on_trace = simulate(detail)
    off_trace = simulate(without_slip_response(detail))
    on_metrics = summarise(on_trace)
    off_metrics = summarise(off_trace)

    print()
    print(f"Detail for {arguments.detail}")
    print(f"required grip force       {on_metrics.required_force:.3f} N")
    print(f"nominal grip force        {detail.controller.force.nominal_force:.3f} N")
    print(f"commanded after response  {on_metrics.final_command:.3f} N")
    off_drop = "n/a" if off_metrics.drop_time is None else f"{off_metrics.drop_time * 1e3:.0f} ms"
    print(f"slip with response on     {on_metrics.total_slip * 1e3:.2f} mm")
    print(f"drop time, response off   {off_drop} after the lift")
    print(
        f"outcome on                {'success' if on_metrics.success else on_metrics.failure.value}"
    )
    print(
        f"outcome off               "
        f"{'success' if off_metrics.success else off_metrics.failure.value}"
    )

    if not arguments.no_figures:
        directory = arguments.figure_dir if arguments.figure_dir is not None else Path("figures")
        written = plot_slip_comparison(
            on_trace, off_trace, directory / f"slip_{arguments.detail}.png"
        )
        print()
        print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
