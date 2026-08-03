"""Matplotlib figures. This is the only module in the package that imports Matplotlib.

Two of these figures are published in the README and are therefore tracked in the
repository, so their size on disk is part of their design. Both are drawn at
``PUBLISHED_DPI`` on a canvas narrow enough to be legible at the width GitHub
renders, which keeps the pair inside a 250 kB budget without any post processing
and without a compression dependency. The rest are working figures written to an
untracked directory, where size does not matter and the default dpi is higher.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from hand_controller.algorithm.proportional import (
    ProportionalConfig,
    control_law,
)
from hand_controller.analysis.metrics import slip_episodes
from hand_controller.model.anatomy import FINGERS, Digit, HandModel
from hand_controller.model.contact import required_grip_force
from hand_controller.model.grasps import (
    GRASP_TAXONOMY,
    GraspDefinition,
    closure_at_span,
    span_profile,
)
from hand_controller.model.kinematics import digit_points
from hand_controller.pipeline.simulation import GraspTrace

__all__ = [
    "PUBLISHED_DPI",
    "plot_control_characteristic",
    "plot_force_profile",
    "plot_grasp_postures",
    "plot_slip_comparison",
    "plot_slip_recovery",
    "plot_span_profiles",
]

PUBLISHED_DPI: Final[int] = 100
"""Resolution of the figures tracked in the repository."""

_ON: Final[str] = "#1f4e79"
_OFF: Final[str] = "#a33"
_RULE: Final[str] = "#555555"


def _save(figure: Figure, path: Path, dpi: int = 130) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_control_characteristic(config: ProportionalConfig, path: Path) -> Path:
    """Draw the dead zone, gain and saturation of the proportional map."""
    differences = np.linspace(-1.0, 1.0, 2001)
    commands = np.array([control_law(config, float(value)) for value in differences])
    figure, axes = plt.subplots(figsize=(6.4, 4.0))
    axes.plot(differences, commands, color="#1f4e79", linewidth=1.8)
    axes.axvspan(-config.deadzone, config.deadzone, color="#cccccc", alpha=0.6, label="dead zone")
    axes.axhline(config.saturation, color="#a33", linestyle="--", linewidth=1.0)
    axes.axhline(
        -config.saturation, color="#a33", linestyle="--", linewidth=1.0, label="saturation"
    )
    axes.set_xlabel("closing envelope minus opening envelope")
    axes.set_ylabel("commanded closure rate, 1/s")
    axes.set_title("Two site proportional control law")
    axes.grid(alpha=0.3)
    axes.legend(loc="upper left", fontsize=8)
    return _save(figure, path)


def plot_span_profiles(hand: HandModel, path: Path, samples: int = 201) -> Path:
    """Draw the opposition span of every grasp against closure."""
    figure, axes = plt.subplots(figsize=(6.4, 4.2))
    for definition in GRASP_TAXONOMY:
        closures, spans = span_profile(hand, definition, samples)
        axes.plot(closures, spans * 1000.0, linewidth=1.5, label=definition.name)
    axes.set_xlabel("closure, 0 open to 1 closed")
    axes.set_ylabel("opposition span, mm")
    axes.set_title("Span of each grasp through its closing trajectory")
    axes.grid(alpha=0.3)
    axes.legend(fontsize=8)
    return _save(figure, path)


def plot_force_profile(trace: GraspTrace, path: Path) -> Path:
    """Draw grip force, commanded force, slip and the tactile signal of one trial."""
    figure, axes = plt.subplots(3, 1, figsize=(6.8, 7.2), sharex=True)

    axes[0].plot(trace.time, trace.commanded_force, color="#999999", linewidth=1.2,
                 label="commanded")
    axes[0].plot(trace.time, trace.grip_force, color="#1f4e79", linewidth=1.6, label="measured")
    axes[0].plot(trace.time, trace.tangential_load, color="#a33", linewidth=1.0,
                 linestyle="--", label="tangential load")
    axes[0].plot(trace.time, trace.friction_capacity, color="#2a7", linewidth=1.0,
                 label="friction capacity")
    axes[0].set_ylabel("force, N")
    axes[0].set_title(f"{trace.item.name} held in a {trace.grasp.name} grasp")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")

    axes[1].plot(trace.time, trace.span * 1000.0, color="#1f4e79", linewidth=1.4, label="span")
    axes[1].axhline(trace.item.width * 1000.0, color="#a33", linestyle="--", linewidth=1.0,
                    label="object width")
    axes[1].set_ylabel("span, mm")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="upper right")

    axes[2].plot(trace.time, trace.slip_displacement * 1000.0, color="#1f4e79", linewidth=1.4,
                 label="slip displacement")
    axes[2].plot(trace.time, trace.slip_detected.astype(float), color="#a33", linewidth=1.0,
                 label="slip detected")
    axes[2].set_xlabel("time, s")
    axes[2].set_ylabel("mm, and detector flag")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8, loc="upper left")

    return _save(figure, path)


def plot_slip_recovery(
    with_response: GraspTrace,
    without_response: GraspTrace,
    path: Path,
    lead: float = 0.040,
    window: float = 0.160,
    dpi: int = PUBLISHED_DPI,
) -> Path:
    """Draw the arrest of a slide beside the same slide left alone.

    The window is the few hundred milliseconds around the lift, because that is
    where everything happens: the object starts to move, the detector fires, the
    demand doubles, and the slide either stops or runs on until the object is out
    of the hand. A three second axis would compress all of it into one pixel.
    """
    lift = with_response.config.lift_time
    start, end = lift - lead, lift + window
    required = required_grip_force(
        with_response.item.mass,
        with_response.item.friction,
        with_response.grasp.load_bearing_contacts,
    )

    def clip(trace: GraspTrace) -> slice:
        inside = np.nonzero((trace.time >= start) & (trace.time <= end))[0]
        return slice(int(inside[0]), int(inside[-1]) + 1) if inside.size else slice(0, 0)

    on, off = clip(with_response), clip(without_response)
    figure, axes = plt.subplots(2, 1, figsize=(6.0, 4.2), sharex=True)

    axes[0].plot(
        with_response.time[on], with_response.grip_force[on], color=_ON, linewidth=1.7,
        label="slip response on",
    )
    axes[0].plot(
        without_response.time[off], without_response.grip_force[off], color=_OFF,
        linewidth=1.5, linestyle="--", label="slip response off",
    )
    axes[0].axhline(
        required, color=_RULE, linewidth=1.0, linestyle=":",
        label=f"force needed to hold, {required:.2f} N",
    )
    axes[0].set_ylabel("grip force, N")
    axes[0].set_title(
        f"{with_response.item.name} slipping at the lift, "
        f"{with_response.grasp.name} grasp"
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="center right", framealpha=0.92)

    axes[1].plot(
        with_response.time[on], with_response.slip_displacement[on] * 1e3, color=_ON,
        linewidth=1.7,
    )
    axes[1].plot(
        without_response.time[off], without_response.slip_displacement[off] * 1e3,
        color=_OFF, linewidth=1.5, linestyle="--",
    )
    axes[1].axhline(
        with_response.config.plant.drop_distance * 1e3, color=_RULE, linewidth=1.0,
        linestyle=":",
    )

    def label(text: str, index: int, trace: GraspTrace, where: tuple[float, float]) -> None:
        axes[1].annotate(
            text,
            xy=(trace.time[index], trace.slip_displacement[index] * 1e3),
            xytext=where, textcoords="axes fraction", fontsize=8,
            arrowprops={"arrowstyle": "->", "color": _RULE, "linewidth": 0.9},
        )

    episodes = slip_episodes(with_response)
    detected = np.nonzero(with_response.slip_detected)[0]
    dt = with_response.config.dt
    if episodes and detected.size:
        began, stopped = episodes[0]
        first = int(detected[0])
        axes[1].plot(
            with_response.time[first], with_response.slip_displacement[first] * 1e3,
            marker="o", color=_ON, markersize=4,
        )
        label(
            f"detected {(first - began) * dt * 1e3:.0f} ms after the slide starts",
            first, with_response, (0.02, 0.42),
        )
        if stopped is not None:
            axes[1].plot(
                with_response.time[stopped], with_response.slip_displacement[stopped] * 1e3,
                marker="o", color=_ON, markersize=4,
            )
            label(
                f"held after {(stopped - began) * dt * 1e3:.0f} ms "
                f"and {with_response.slip_displacement[stopped] * 1e3:.2f} mm",
                stopped, with_response, (0.56, 0.06),
            )
    if without_response.released:
        last = len(without_response) - 1
        axes[1].plot(
            without_response.time[last], without_response.slip_displacement[last] * 1e3,
            marker="x", color=_OFF, markersize=7,
        )
        label(
            f"out of the hand {(without_response.time[last] - lift) * 1e3:.0f} ms after the lift",
            last, without_response, (0.10, 0.80),
        )

    axes[1].set_xlabel("time, s")
    axes[1].set_ylabel("slide, mm")
    axes[1].grid(alpha=0.3)
    figure.align_ylabels(axes)
    return _save(figure, path, dpi=dpi)


def plot_grasp_postures(
    hand: HandModel,
    definition: GraspDefinition,
    widths: Sequence[float],
    path: Path,
    dpi: int = PUBLISHED_DPI,
) -> Path:
    """Draw the hand of one grasp at the closure that meets each object width.

    One commanded number sets fifteen joint angles through the finger coupling,
    and the object decides where along that trajectory the hand stops. The
    panels are the same grasp and the same coupling, drawn at the closures the
    span solver returns for each width.

    The view is the sagittal projection, which is the plane the fingers flex in.
    The thumb is drawn in it as well, and is foreshortened there, because
    opposition carries the thumb flexion plane out of the sagittal plane; its
    tip position is correct but its bend is not visible from this side.
    """
    count = len(widths)
    if count == 0:
        raise ValueError("at least one width is needed")
    figure, axes = plt.subplots(1, count, figsize=(2.30 * count, 2.25))
    panels = np.atleast_1d(np.asarray(axes, dtype=object))

    for index, (axis, width) in enumerate(zip(panels, widths, strict=True)):
        closure = closure_at_span(hand, definition, width)
        config = definition.posture(closure)

        axis.plot(
            [0.0, 92.0], [0.0, 0.0], color="#c8c8c8", linewidth=9.0, solid_capstyle="round",
            zorder=1,
        )
        thumb_tip = digit_points(hand, config, Digit.THUMB)[3]
        tips = np.array(
            [digit_points(hand, config, digit)[3] for digit in definition.contact_fingers],
            dtype=np.float64,
        )
        centre = 0.5 * (thumb_tip + tips.mean(axis=0)) * 1e3
        axis.add_patch(
            Circle(
                (float(centre[0]), float(centre[2])), 0.5 * width * 1e3,
                facecolor="#f2dda8", edgecolor="#8a7330", linewidth=1.0, zorder=0,
            )
        )
        for digit in (Digit.THUMB, *FINGERS):
            points = digit_points(hand, config, digit) * 1e3
            colour = _OFF if digit is Digit.THUMB else _ON
            axis.plot(
                points[:, 0], points[:, 2], color=colour, linewidth=2.0,
                solid_capstyle="round", zorder=2,
            )
            axis.plot(
                points[:, 0], points[:, 2], linestyle="none", marker="o", color="white",
                markersize=3.4, markeredgecolor=colour, markeredgewidth=1.0, zorder=3,
            )

        axis.set_title(f"{width * 1e3:.0f} mm, closure {closure:.2f}", fontsize=9)
        axis.set_aspect("equal")
        axis.set_xlim(-12.0, 196.0)
        axis.set_ylim(-92.0, 26.0)
        axis.tick_params(labelsize=7)
        axis.set_xlabel("distal, mm", fontsize=8)
        if index:
            axis.tick_params(labelleft=False)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)

    panels[0].set_ylabel("palmar, mm", fontsize=8)
    figure.suptitle(
        f"{definition.taxonomy_name} seen from the side, at the closure that meets "
        "each width, thumb in red",
        fontsize=10,
    )
    figure.tight_layout()
    return _save(figure, path, dpi=dpi)


def plot_slip_comparison(
    with_response: GraspTrace, without_response: GraspTrace, path: Path
) -> Path:
    """Draw the same object with and without the slip response."""
    figure, axes = plt.subplots(2, 1, figsize=(6.8, 5.4), sharex=True)

    axes[0].plot(with_response.time, with_response.grip_force, color="#1f4e79", linewidth=1.6,
                 label="slip response on")
    axes[0].plot(without_response.time, without_response.grip_force, color="#a33", linewidth=1.6,
                 label="slip response off")
    axes[0].set_ylabel("grip force, N")
    axes[0].set_title(f"Slip response on {with_response.item.name}")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")

    axes[1].plot(with_response.time, with_response.slip_displacement * 1000.0,
                 color="#1f4e79", linewidth=1.6, label="slip response on")
    axes[1].plot(without_response.time, without_response.slip_displacement * 1000.0,
                 color="#a33", linewidth=1.6, label="slip response off")
    axes[1].axhline(with_response.config.plant.drop_distance * 1000.0, color="#555",
                    linestyle="--", linewidth=1.0, label="drop distance")
    axes[1].set_xlabel("time, s")
    axes[1].set_ylabel("slip displacement, mm")
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="upper left")

    return _save(figure, path)
