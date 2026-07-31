"""Property and invariant tests for the grasp taxonomy and the object set."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from hand_controller.model import (
    GRASP_NAMES,
    GRASP_TAXONOMY,
    OBJECT_NAMES,
    OBJECT_SET,
    Digit,
    Feasibility,
    GraspDefinition,
    GraspObject,
    OppositionType,
    closure_at_span,
    default_hand,
    feasibility,
    grasp,
    grasp_object,
    opposition_span,
    reachability,
    span_profile,
    span_range,
    taxonomy_digits,
)
from hand_controller.model.grasps import SPAN_SOLVER_TOLERANCE
from hand_controller.pipeline import EVALUATION_SET, grasp_for_object

HAND = default_hand()
SPAN_SAMPLES = 401


@pytest.mark.parametrize("definition", GRASP_TAXONOMY, ids=GRASP_NAMES)
def test_every_grasp_stays_inside_its_joint_limits(definition: GraspDefinition) -> None:
    """Every grasp of the taxonomy is reachable, not only the one used most."""
    assert reachability(HAND, definition) == ()


@pytest.mark.parametrize("definition", GRASP_TAXONOMY, ids=GRASP_NAMES)
def test_every_grasp_closes_monotonically(definition: GraspDefinition) -> None:
    """The span falls strictly as the grasp closes, so contact happens once."""
    _, spans = span_profile(HAND, definition, SPAN_SAMPLES)
    assert bool(np.all(np.diff(spans) < 0.0))


@pytest.mark.parametrize("definition", GRASP_TAXONOMY, ids=GRASP_NAMES)
def test_span_range_brackets_the_profile(definition: GraspDefinition) -> None:
    closed, opened = span_range(HAND, definition)
    _, spans = span_profile(HAND, definition, SPAN_SAMPLES)
    assert closed == pytest.approx(float(spans[-1]), abs=1e-15)
    assert opened == pytest.approx(float(spans[0]), abs=1e-15)
    assert closed < opened


@pytest.mark.parametrize("definition", GRASP_TAXONOMY, ids=GRASP_NAMES)
def test_every_grasp_opens_wide_enough_to_be_useful(definition: GraspDefinition) -> None:
    """A grasp with no working range could not hold anything."""
    closed, opened = span_range(HAND, definition)
    assert opened - closed > 0.020


def test_taxonomy_entries_are_unique_and_classified() -> None:
    names = [definition.name for definition in GRASP_TAXONOMY]
    indices = [definition.taxonomy_index for definition in GRASP_TAXONOMY]
    assert len(set(names)) == len(names)
    assert len(set(indices)) == len(indices)
    for definition in GRASP_TAXONOMY:
        assert definition.opposition is not OppositionType.NONE
        assert Digit.THUMB in definition.contact_digits
        assert definition.load_bearing_contacts >= 2
        assert definition.description.endswith(".")


def test_the_taxonomy_covers_three_opposition_types_and_both_thumb_positions() -> None:
    oppositions = {definition.opposition for definition in GRASP_TAXONOMY}
    thumbs = {definition.thumb for definition in GRASP_TAXONOMY}
    classes = {definition.grasp_class for definition in GRASP_TAXONOMY}
    assert len(oppositions) == 3
    assert len(thumbs) == 2
    assert len(classes) == 3


@pytest.mark.parametrize("definition", GRASP_TAXONOMY, ids=GRASP_NAMES)
def test_taxonomy_digits_are_ordered_thumb_first(definition: GraspDefinition) -> None:
    ordered = taxonomy_digits(definition)
    assert ordered[0] is Digit.THUMB
    assert set(ordered) == set(definition.contact_digits)
    assert list(ordered) == sorted(ordered)


@pytest.mark.parametrize("definition", GRASP_TAXONOMY, ids=GRASP_NAMES)
@pytest.mark.parametrize("fraction", [0.05, 0.25, 0.5, 0.75, 0.95])
def test_closure_at_span_inverts_the_span(definition: GraspDefinition, fraction: float) -> None:
    """The solver recovers the closure that produced a span.

    The tolerance is a thousand times ``SPAN_SOLVER_TOLERANCE``, the bracket
    width Brent's method is asked to reach. Comparing at the solver tolerance
    itself would put the assertion on its own boundary.
    """
    closure = fraction
    span = opposition_span(HAND, definition, closure)
    recovered = closure_at_span(HAND, definition, span)
    assert recovered == pytest.approx(closure, abs=1000.0 * SPAN_SOLVER_TOLERANCE)


@pytest.mark.parametrize("definition", GRASP_TAXONOMY, ids=GRASP_NAMES)
def test_closure_at_span_rejects_spans_outside_the_range(definition: GraspDefinition) -> None:
    closed, opened = span_range(HAND, definition)
    with pytest.raises(ValueError, match="outside the reach"):
        closure_at_span(HAND, definition, opened + 0.010)
    with pytest.raises(ValueError, match="outside the reach"):
        closure_at_span(HAND, definition, closed - 0.010)


def test_a_grasp_without_opposition_has_no_span() -> None:
    """Grasps that carry load without a squeeze are refused, not guessed at."""
    hook = dataclasses.replace(grasp("medium_wrap"), opposition=OppositionType.NONE, name="hook")
    with pytest.raises(ValueError, match="no opposition"):
        opposition_span(HAND, hook, 0.5)


def test_grasp_lookup_reports_unknown_names() -> None:
    with pytest.raises(KeyError, match="unknown grasp"):
        grasp("power_grip")
    assert grasp("tripod").taxonomy_index == 14


def test_object_lookup_reports_unknown_names() -> None:
    with pytest.raises(KeyError, match="unknown object"):
        grasp_object("banana")
    assert grasp_object("pen").width == 0.011


def test_object_definitions_are_physical() -> None:
    assert len(set(OBJECT_NAMES)) == len(OBJECT_NAMES)
    for item in OBJECT_SET:
        assert item.width > 0.0
        assert item.mass > 0.0
        assert item.stiffness > 0.0
        assert 0.0 < item.friction <= 2.0
        assert item.crush_force > 0.0
        assert item.description.endswith(".")


@pytest.mark.parametrize(
    ("field", "value"),
    [("width", -0.01), ("mass", 0.0), ("friction", 0.0), ("friction", 3.0)],
)
def test_object_rejects_impossible_properties(field: str, value: float) -> None:
    item = grasp_object("pen")
    with pytest.raises(ValueError):
        dataclasses.replace(item, **{field: value})


@pytest.mark.parametrize(("object_name", "grasp_name"), EVALUATION_SET)
def test_the_evaluation_set_is_feasible(object_name: str, grasp_name: str) -> None:
    """Every pairing the evaluation uses can actually be enclosed."""
    item = grasp_object(object_name)
    assert feasibility(HAND, grasp(grasp_name), item) is Feasibility.FEASIBLE
    assert grasp_for_object(object_name) == grasp_name


def test_feasibility_reports_both_failure_directions() -> None:
    definition = grasp("medium_wrap")
    closed, opened = span_range(HAND, definition)
    template = grasp_object("pen")
    too_large = dataclasses.replace(template, name="slab", width=opened + 0.010)
    too_small = dataclasses.replace(template, name="wire", width=max(closed - 0.005, 1e-4))
    assert feasibility(HAND, definition, too_large) is Feasibility.TOO_LARGE
    assert feasibility(HAND, definition, too_small) is Feasibility.TOO_SMALL


def test_feasibility_boundaries_are_inclusive() -> None:
    definition = grasp("tripod")
    closed, opened = span_range(HAND, definition)
    template: GraspObject = grasp_object("pen")
    for width in (closed, opened):
        item = dataclasses.replace(template, width=width)
        assert feasibility(HAND, definition, item) is Feasibility.FEASIBLE


def test_grasp_for_object_reports_unknown_objects() -> None:
    with pytest.raises(KeyError, match="not in the evaluation set"):
        grasp_for_object("banana")


def test_span_profile_needs_at_least_two_samples() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        span_profile(HAND, grasp("tripod"), 1)
