"""Fixtures shared across the test modules.

The evaluation set is simulated once per session rather than once per module.
Nothing mutates a trace, so sharing is safe, and it removes five repetitions of
the same three second run.
"""

from __future__ import annotations

import pytest

from hand_controller.analysis import GraspMetrics, summarise
from hand_controller.pipeline import GraspTrace, run_evaluation


@pytest.fixture(scope="session")
def evaluation_traces() -> tuple[GraspTrace, ...]:
    """Every trial of the evaluation set at the reference configuration."""
    return run_evaluation()


@pytest.fixture(scope="session")
def evaluation_metrics(
    evaluation_traces: tuple[GraspTrace, ...],
) -> tuple[GraspMetrics, ...]:
    """Metrics for every trial, in the order of the evaluation set."""
    return tuple(summarise(trace) for trace in evaluation_traces)


@pytest.fixture(scope="session")
def metrics_by_object(
    evaluation_metrics: tuple[GraspMetrics, ...],
) -> dict[str, GraspMetrics]:
    """Metrics keyed by object name."""
    return {entry.object_name: entry for entry in evaluation_metrics}
