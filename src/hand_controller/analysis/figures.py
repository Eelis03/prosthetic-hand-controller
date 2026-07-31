"""Matplotlib figures. This is the only module in the package that imports Matplotlib."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from hand_controller.algorithm.proportional import (
    ProportionalConfig,
    control_law,
)
from hand_controller.model.anatomy import HandModel
from hand_controller.model.grasps import GRASP_TAXONOMY, span_profile
from hand_controller.pipeline.simulation import GraspTrace

__all__ = [
    "plot_control_characteristic",
    "plot_force_profile",
    "plot_slip_comparison",
    "plot_span_profiles",
]


def _save(figure: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130, bbox_inches="tight")
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
