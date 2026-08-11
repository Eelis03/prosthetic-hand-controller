"""Property tests for the screen that decides whether to close on an object at all."""

from __future__ import annotations

import dataclasses

import pytest

from hand_controller.algorithm import ForceConfig
from hand_controller.analysis import GraspMetrics
from hand_controller.model import (
    GRASP_NAMES,
    GRAVITY,
    OBJECT_NAMES,
    default_hand,
    grasp,
    grasp_object,
    span_range,
)
from hand_controller.pipeline import EVALUATION_SET, grasp_for_object
from hand_controller.pipeline.admissibility import (
    Admissibility,
    UnholdableObjectError,
    screen_grasp,
    screen_taxonomy,
    select_grasp,
)

HAND = default_hand()
LIMIT = ForceConfig().safety_limit


@pytest.mark.parametrize(("object_name", "grasp_name"), EVALUATION_SET)
def test_the_screen_agrees_with_the_trial_on_every_object(
    object_name: str, grasp_name: str, metrics_by_object: dict[str, GraspMetrics]
) -> None:
    """The verdict reached before the reach is the one the loop reaches by holding.

    Nine objects are admitted and held, and the steel ball is refused in advance
    and dropped. The screen is statics and the trial is a closed loop simulation,
    so the agreement is a claim about both rather than a restatement of one.
    """
    screen = screen_grasp(HAND, grasp(grasp_name), grasp_object(object_name))
    assert screen.admissible is metrics_by_object[object_name].success


@pytest.mark.parametrize(("object_name", "grasp_name"), EVALUATION_SET)
def test_the_requirement_is_the_force_the_metrics_report(
    object_name: str, grasp_name: str, metrics_by_object: dict[str, GraspMetrics]
) -> None:
    """Screening and reporting must not disagree about the same number."""
    screen = screen_grasp(HAND, grasp(grasp_name), grasp_object(object_name))
    assert screen.required_force == pytest.approx(
        metrics_by_object[object_name].required_force, rel=1e-15
    )


@pytest.mark.parametrize(("object_name", "grasp_name"), EVALUATION_SET)
def test_the_margin_is_the_ceiling_over_the_requirement(object_name: str, grasp_name: str) -> None:
    item = grasp_object(object_name)
    screen = screen_grasp(HAND, grasp(grasp_name), item)
    assert screen.force_ceiling == min(LIMIT, item.crush_force)
    assert screen.margin == pytest.approx(screen.force_ceiling / screen.required_force, rel=1e-15)
    assert (screen.margin >= 1.0) is screen.admissible


@pytest.mark.parametrize("object_name", OBJECT_NAMES)
def test_the_load_one_contact_carries_falls_as_contacts_are_added(object_name: str) -> None:
    """The requirement times the contact count is the weight over the friction.

    This is the taxonomy's contact expectation doing mechanical work: the same
    object needs a third less force per contact in a six contact wrap than in a
    four contact grasp, and a third of what a two digit pinch needs.
    """
    item = grasp_object(object_name)
    expected = item.mass * GRAVITY / item.friction
    for screen in screen_taxonomy(HAND, item):
        assert screen.required_force * screen.contacts == pytest.approx(expected, rel=1e-15)


def test_the_screen_covers_every_grasp_in_taxonomy_order() -> None:
    screens = screen_taxonomy(HAND, grasp_object("apple"))
    assert tuple(screen.grasp_name for screen in screens) == GRASP_NAMES
    assert {screen.object_name for screen in screens} == {"apple"}


def test_no_grasp_of_the_taxonomy_holds_the_steel_ball() -> None:
    """The object the evaluation set drops was never holdable by any of the six.

    A medium wrap shares the load over six contacts rather than the five of the
    power sphere and still needs 16.018 N against the 15.0 N limit, so the
    refusal is a fact about the ball and not about the grasp paired with it.
    """
    ball = grasp_object("steel_ball")
    screens = screen_taxonomy(HAND, ball)
    assert not any(screen.admissible for screen in screens)
    with pytest.raises(UnholdableObjectError, match="closest is medium_wrap"):
        select_grasp(HAND, ball)


def test_the_verdict_names_the_ceiling_that_binds() -> None:
    """A crushable object and a heavy one are refused for different reasons."""
    filled_cup = dataclasses.replace(grasp_object("foam_cup"), name="filled_foam_cup", mass=1.5)
    lead_glass = dataclasses.replace(grasp_object("drinking_glass"), name="lead_glass", mass=6.0)
    wrap = grasp("medium_wrap")
    assert screen_grasp(HAND, wrap, filled_cup).verdict is Admissibility.WOULD_CRUSH
    assert screen_grasp(HAND, wrap, lead_glass).verdict is Admissibility.EXCEEDS_LIMIT


def test_an_object_that_fails_below_its_own_requirement_is_refused_as_crushed() -> None:
    """When both ceilings are passed the crush limit is the one reported.

    A stronger actuator would hold an object that only passes the safety limit.
    Nothing at all holds one that breaks at less force than it needs, so that is
    the refusal worth naming.
    """
    brick = dataclasses.replace(grasp_object("paper_cup_full"), name="paper_brick", mass=5.0)
    screen = screen_grasp(HAND, grasp("medium_wrap"), brick)
    assert screen.required_force > LIMIT
    assert screen.required_force > brick.crush_force
    assert screen.verdict is Admissibility.WOULD_CRUSH


def test_geometry_is_screened_before_force() -> None:
    """An object the grasp cannot reach is refused on that, whatever it weighs."""
    wrap = grasp("medium_wrap")
    closed, opened = span_range(HAND, wrap)
    template = grasp_object("steel_ball")
    boulder = dataclasses.replace(template, name="boulder", width=opened + 0.010)
    grit = dataclasses.replace(template, name="grit", width=max(closed - 0.005, 1e-4))
    assert screen_grasp(HAND, wrap, boulder).verdict is Admissibility.TOO_LARGE
    assert screen_grasp(HAND, wrap, grit).verdict is Admissibility.TOO_SMALL


def test_an_object_no_grasp_encloses_is_refused_on_its_width() -> None:
    beach_ball = dataclasses.replace(grasp_object("apple"), name="beach_ball", width=0.100)
    with pytest.raises(UnholdableObjectError, match="encloses beach_ball"):
        select_grasp(HAND, beach_ball)


def test_a_safety_factor_raises_the_requirement_it_screens_against() -> None:
    """Demanding headroom in advance is the same screen against a larger force."""
    item = grasp_object("paper_cup_full")
    wrap = grasp("medium_wrap")
    plain = screen_grasp(HAND, wrap, item)
    doubled = screen_grasp(HAND, wrap, item, safety_factor=2.0)
    assert plain.admissible
    assert doubled.required_force == pytest.approx(2.0 * plain.required_force, rel=1e-15)
    assert doubled.margin == pytest.approx(0.5 * plain.margin, rel=1e-15)
    assert screen_grasp(HAND, wrap, item, safety_factor=3.0).verdict is Admissibility.WOULD_CRUSH


def test_a_safety_factor_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="safety factor"):
        screen_grasp(HAND, grasp("tripod"), grasp_object("pen"), safety_factor=0.5)


def test_selection_returns_the_admissible_grasp_with_the_most_headroom() -> None:
    item = grasp_object("drinking_glass")
    chosen = select_grasp(HAND, item)
    admitted = [screen for screen in screen_taxonomy(HAND, item) if screen.admissible]
    assert chosen.grasp_name == "medium_wrap"
    assert chosen.contacts == 6
    assert chosen.margin == max(screen.margin for screen in admitted)


def test_selection_is_blind_to_shape() -> None:
    """It ranks by load sharing, so it prefers a wrap where a user would not.

    The evaluation set holds the apple in a power sphere because it is round.
    The screen sees a 65 mm width and a mass, a medium wrap encloses that on six
    contacts rather than five, and so that is what it returns. Choosing the grasp
    that suits the shape stays with the user.
    """
    chosen = select_grasp(HAND, grasp_object("apple"))
    assert chosen.grasp_name == "medium_wrap"
    assert grasp_for_object("apple") == "power_sphere"


def test_a_higher_safety_limit_makes_the_steel_ball_holdable() -> None:
    """The refusal is a statement about the limit rather than about the ball.

    Given 20.0 N per contact a medium wrap needs 16.018 N of it and the ball is
    admitted. The limit is 15.0 N because a hand that squeezes harder than that
    damages what it holds, so this is the screen working rather than a suggestion.
    """
    chosen = select_grasp(HAND, grasp_object("steel_ball"), ForceConfig(safety_limit=20.0))
    assert chosen.grasp_name == "medium_wrap"
    assert chosen.required_force == pytest.approx(16.0175, abs=1e-3)
    assert chosen.verdict is Admissibility.ADMISSIBLE
