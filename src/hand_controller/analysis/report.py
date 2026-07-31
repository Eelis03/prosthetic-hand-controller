"""Fixed width text rendering of every result this project reports."""

from __future__ import annotations

from hand_controller.analysis.metrics import GraspMetrics, SetSummary
from hand_controller.model.anatomy import HandModel
from hand_controller.model.grasps import GRASP_TAXONOMY, span_range
from hand_controller.model.objects import OBJECT_SET, Feasibility, feasibility
from hand_controller.pipeline.simulation import GraspTrace

__all__ = [
    "format_evaluation",
    "format_feasibility",
    "format_set_summary",
    "format_slip_comparison",
    "format_taxonomy",
    "format_trial",
]


def _milli(value: float) -> str:
    return f"{value * 1000.0:.2f}"


def format_taxonomy(hand: HandModel) -> str:
    """Render the grasp taxonomy with its classification and span range."""
    header = (
        f"{'grasp':<15}{'no':>3} {'taxonomy name':<22}{'opposition':<12}"
        f"{'thumb':<11}{'class':<13}{'contacts':>9}{'closed_mm':>11}{'open_mm':>9}"
    )
    lines = [header, "-" * len(header)]
    for definition in GRASP_TAXONOMY:
        closed, opened = span_range(hand, definition)
        lines.append(
            f"{definition.name:<15}{definition.taxonomy_index:>3} "
            f"{definition.taxonomy_name:<22}{definition.opposition.value:<12}"
            f"{definition.thumb.value:<11}{definition.grasp_class.value:<13}"
            f"{definition.load_bearing_contacts:>9}"
            f"{closed * 1000.0:>11.1f}{opened * 1000.0:>9.1f}"
        )
    return "\n".join(lines)


def format_feasibility(hand: HandModel) -> str:
    """Render which grasp can enclose which object."""
    names = [definition.name for definition in GRASP_TAXONOMY]
    header = f"{'object':<16}{'width_mm':>9}  " + "".join(f"{name:<16}" for name in names)
    lines = [header, "-" * len(header)]
    for item in OBJECT_SET:
        cells = []
        for definition in GRASP_TAXONOMY:
            verdict = feasibility(hand, definition, item)
            cells.append("yes" if verdict is Feasibility.FEASIBLE else verdict.value)
        lines.append(
            f"{item.name:<16}{item.width * 1000.0:>9.1f}  "
            + "".join(f"{cell:<16}" for cell in cells)
        )
    return "\n".join(lines)


def format_trial(trace: GraspTrace, metrics: GraspMetrics) -> str:
    """Render one trial as a block of labelled numbers."""
    recovery = (
        "n/a" if metrics.slip_recovery_time is None else f"{metrics.slip_recovery_time:.3f} s"
    )
    dropped = "n/a" if metrics.drop_time is None else f"{metrics.drop_time:.3f} s after the lift"
    lines = [
        f"object              {metrics.object_name}",
        f"grasp               {metrics.grasp_name} ({trace.grasp.taxonomy_name})",
        f"object width        {_milli(trace.item.width)} mm",
        f"object mass         {trace.item.mass:.3f} kg",
        f"contact stiffness   {trace.contact_stiffness:.4g} N/m^1.5",
        f"load contacts       {trace.grasp.load_bearing_contacts}",
        f"feasibility         {metrics.feasible.value}",
        f"time to contact     {metrics.time_to_contact:.3f} s",
        f"time to grip        {metrics.time_to_grip:.3f} s",
        f"required force      {metrics.required_force:.3f} N",
        f"commanded force     {metrics.final_command:.3f} N",
        f"peak force          {metrics.peak_force:.3f} N",
        f"final force         {metrics.final_force:.3f} N",
        f"crush limit         {metrics.crush_force:.3f} N",
        f"force overshoot     {metrics.force_overshoot * 100.0:.2f} percent",
        f"steady state error  {metrics.steady_state_error:.3e} N",
        f"slip detections     {metrics.slip_events}",
        f"peak slip speed     {_milli(metrics.peak_slip_speed)} mm/s",
        f"total slip          {_milli(metrics.total_slip)} mm",
        f"slip recovery       {recovery}",
        f"drop distance passed {dropped}",
        f"force saturated     {metrics.force_saturated}",
        f"outcome             {'success' if metrics.success else metrics.failure.value}",
    ]
    return "\n".join(lines)


def format_evaluation(metrics: tuple[GraspMetrics, ...]) -> str:
    """Render the evaluation table over the object set."""
    header = (
        f"{'object':<16}{'grasp':<16}{'t_contact':>10}{'t_grip':>8}{'req_N':>8}"
        f"{'cmd_N':>8}{'peak_N':>8}{'over_%':>8}{'slip_mm':>9}{'recov_ms':>9}"
        f"{'drop_ms':>9}{'outcome':>16}"
    )
    lines = [header, "-" * len(header)]
    for entry in metrics:
        recovery = (
            "n/a" if entry.slip_recovery_time is None else f"{entry.slip_recovery_time * 1e3:.0f}"
        )
        dropped = "n/a" if entry.drop_time is None else f"{entry.drop_time * 1e3:.0f}"
        slip = (
            f"{entry.total_slip * 1000.0:.2f}"
            if entry.drop_time is None
            else "gone"
        )
        outcome = "success" if entry.success else entry.failure.value
        lines.append(
            f"{entry.object_name:<16}{entry.grasp_name:<16}"
            f"{entry.time_to_contact:>10.3f}{entry.time_to_grip:>8.3f}"
            f"{entry.required_force:>8.3f}{entry.final_command:>8.3f}{entry.peak_force:>8.3f}"
            f"{entry.force_overshoot * 100.0:>8.2f}{slip:>9}"
            f"{recovery:>9}{dropped:>9}{outcome:>16}"
        )
    return "\n".join(lines)


def format_set_summary(summary: SetSummary) -> str:
    """Render the aggregate outcome of an evaluation run."""
    return "\n".join(
        [
            f"trials              {summary.trials}",
            f"successes           {summary.successes}",
            f"success rate        {summary.success_rate * 100.0:.1f} percent",
            f"mean time to contact {summary.mean_time_to_contact:.3f} s",
            f"mean force overshoot {summary.mean_force_overshoot * 100.0:.2f} percent",
            f"trials that slipped {summary.slipping_trials}",
            f"trials dropped      {summary.dropped_trials}",
            f"mean slip recovery  {summary.mean_slip_recovery * 1000.0:.1f} ms",
            f"largest slip, held  {summary.max_slip_when_held * 1000.0:.2f} mm",
        ]
    )


def format_slip_comparison(
    with_response: tuple[GraspMetrics, ...], without_response: tuple[GraspMetrics, ...]
) -> str:
    """Render the same object set with and without the slip response."""
    header = (
        f"{'object':<16}{'slip_on_mm':>12}{'recov_on_ms':>13}{'outcome_on':>12}"
        f"{'outcome_off':>13}{'drop_off_ms':>13}"
    )
    lines = [header, "-" * len(header)]
    for on, off in zip(with_response, without_response, strict=True):
        recovery = "n/a" if on.slip_recovery_time is None else f"{on.slip_recovery_time * 1e3:.0f}"
        dropped = "n/a" if off.drop_time is None else f"{off.drop_time * 1e3:.0f}"
        slip = f"{on.total_slip * 1000.0:.2f}" if on.drop_time is None else "gone"
        lines.append(
            f"{on.object_name:<16}{slip:>12}{recovery:>13}"
            f"{('success' if on.success else on.failure.value):>12}"
            f"{('success' if off.success else off.failure.value):>13}"
            f"{dropped:>13}"
        )
    held_on = sum(1 for entry in with_response if entry.success)
    held_off = sum(1 for entry in without_response if entry.success)
    lines.append("-" * len(header))
    lines.append(f"{'objects held':<16}{held_on:>12}{'':>13}{'':>12}{held_off:>13}")
    return "\n".join(lines)
