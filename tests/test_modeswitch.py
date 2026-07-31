"""Property and invariant tests for co-contraction grasp selection.

The test that matters most here is the false trigger case. A recogniser that
looks only at how hard the user is contracting will change grasp during an
ordinary strong closing effort, which would be worse than having no mode
switching at all.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hand_controller.algorithm import (
    CoContractionConfig,
    ModeSwitcher,
    MyoelectricController,
    ProportionalConfig,
    SwitchState,
    is_co_contraction,
    mode_switch_latency,
)

SWITCHING = CoContractionConfig()
PROPORTIONAL = ProportionalConfig()
DT = 0.001


def _drive(
    switcher: ModeSwitcher, opening: float, closing: float, duration: float
) -> tuple[int, SwitchState]:
    switches = 0
    state = SwitchState.IDLE
    for _ in range(round(duration / DT)):
        result = switcher.update(opening, closing, DT)
        switches += int(result.switched)
        state = result.state
    return switches, state


def test_a_balanced_effort_on_both_sites_is_a_co_contraction() -> None:
    assert is_co_contraction(SWITCHING, 0.70, 0.70)
    assert is_co_contraction(SWITCHING, 0.60, 0.70)


@pytest.mark.parametrize("closing", np.linspace(0.0, 1.0, 201))
def test_a_single_site_contraction_is_never_a_co_contraction(closing: float) -> None:
    """The false trigger case: any closing effort at rest on the other site."""
    assert not is_co_contraction(SWITCHING, 0.02, float(closing))
    assert not is_co_contraction(SWITCHING, float(closing), 0.02)


@pytest.mark.parametrize("crosstalk", np.linspace(0.0, 0.74, 76))
def test_a_strong_contraction_with_crosstalk_is_not_a_co_contraction(crosstalk: float) -> None:
    """Even when both channels are active, unbalanced activity is not a co-contraction."""
    closing = 0.90
    assert not is_co_contraction(SWITCHING, float(crosstalk), closing)


def test_the_balance_condition_is_what_rejects_the_false_trigger() -> None:
    """Removing the balance test would accept a strong single site contraction."""
    unbalanced = CoContractionConfig(balance_tolerance=1.0)
    assert not is_co_contraction(SWITCHING, 0.45, 0.90)
    assert is_co_contraction(unbalanced, 0.45, 0.90)


def test_a_weak_balanced_effort_is_below_the_activation_threshold() -> None:
    assert not is_co_contraction(SWITCHING, 0.20, 0.20)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_a_broken_channel_is_not_a_co_contraction(value: float) -> None:
    assert not is_co_contraction(SWITCHING, value, 0.70)
    assert not is_co_contraction(SWITCHING, 0.70, value)


def test_a_co_contraction_shorter_than_the_hold_time_does_not_switch() -> None:
    switcher = ModeSwitcher(SWITCHING, mode_count=3)
    switches, state = _drive(switcher, 0.70, 0.70, SWITCHING.hold_time - 2 * DT)
    assert switches == 0
    assert state is SwitchState.ARMING


def test_a_sustained_co_contraction_switches_once_per_hold_time() -> None:
    switcher = ModeSwitcher(SWITCHING, mode_count=3)
    switches, _ = _drive(switcher, 0.70, 0.70, SWITCHING.hold_time + DT)
    assert switches == 1
    assert switcher.mode == 1


def test_the_refractory_interval_blocks_an_immediate_second_switch() -> None:
    switcher = ModeSwitcher(SWITCHING, mode_count=3)
    _drive(switcher, 0.70, 0.70, SWITCHING.hold_time + DT)
    switches, state = _drive(switcher, 0.70, 0.70, SWITCHING.refractory_time - 2 * DT)
    assert switches == 0
    assert state is SwitchState.REFRACTORY


def test_grasp_selection_cycles_through_every_mode() -> None:
    switcher = ModeSwitcher(SWITCHING, mode_count=3)
    period = SWITCHING.hold_time + SWITCHING.refractory_time + 2 * DT
    modes = []
    for _ in range(4):
        _drive(switcher, 0.70, 0.70, period)
        modes.append(switcher.mode)
    assert modes == [1, 2, 0, 1]


def test_the_recogniser_resets() -> None:
    switcher = ModeSwitcher(SWITCHING, mode_count=3, initial_mode=2)
    _drive(switcher, 0.70, 0.70, SWITCHING.hold_time + DT)
    assert switcher.mode == 0
    switcher.reset()
    assert switcher.mode == 2


@pytest.mark.parametrize(
    ("mode_count", "initial"), [(0, 0), (3, -1), (3, 3)]
)
def test_the_recogniser_validates_its_mode_range(mode_count: int, initial: int) -> None:
    with pytest.raises(ValueError):
        ModeSwitcher(SWITCHING, mode_count=mode_count, initial_mode=initial)


def test_the_recogniser_rejects_a_non_positive_step() -> None:
    switcher = ModeSwitcher(SWITCHING, mode_count=2)
    with pytest.raises(ValueError, match="dt must be positive"):
        switcher.update(0.7, 0.7, 0.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("activation_threshold", 0.0),
        ("activation_threshold", 1.5),
        ("balance_tolerance", -0.1),
        ("hold_time", 0.0),
        ("refractory_time", -0.1),
    ],
)
def test_configuration_rejects_impossible_parameters(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        CoContractionConfig(**{field: value})


def test_the_command_is_gated_while_a_co_contraction_is_confirmed() -> None:
    """A balanced effort must not creep the hand shut while it is being recognised."""
    controller = MyoelectricController(PROPORTIONAL, SWITCHING, DT, mode_count=3)
    peak = 0.0
    for _ in range(round(0.5 / DT)):
        output = controller.update(0.75, 0.60, DT)
        peak = max(peak, abs(output.velocity))
    assert peak == 0.0


@pytest.mark.parametrize("closing", [0.5, 0.7, 0.9, 1.0])
def test_a_strong_single_site_contraction_drives_without_switching(closing: float) -> None:
    """The whole front end, not just the recogniser, ignores a single site effort."""
    controller = MyoelectricController(PROPORTIONAL, SWITCHING, DT, mode_count=6)
    switches = 0
    velocity = 0.0
    for _ in range(round(1.0 / DT)):
        output = controller.update(0.02, closing, DT)
        switches += int(output.switched)
        velocity = output.velocity
    assert switches == 0
    assert controller.mode == 0
    assert velocity > 0.0


def test_the_front_end_resets() -> None:
    controller = MyoelectricController(PROPORTIONAL, SWITCHING, DT, mode_count=3)
    for _ in range(400):
        controller.update(0.7, 0.7, DT)
    controller.reset()
    assert controller.mode == 0
    assert controller.update(0.0, 0.0, DT).velocity == 0.0


def test_mode_switch_latency_meets_the_stated_budget() -> None:
    """The measured end to end switch delay is inside the 250 ms budget.

    The tolerance is one control period, because the measurement is the index of
    a sample and is therefore quantised to ``DT``.
    """
    latency = mode_switch_latency(PROPORTIONAL, SWITCHING, DT)
    assert latency <= SWITCHING.latency_budget + DT


def test_mode_switch_latency_exceeds_the_confirmation_hold() -> None:
    """The smoothing window is part of the delay, so the hold time alone understates it."""
    latency = mode_switch_latency(PROPORTIONAL, SWITCHING, DT)
    assert latency > SWITCHING.hold_time


def test_mode_switch_latency_reports_a_switch_that_never_arrives() -> None:
    with pytest.raises(RuntimeError, match="no mode switch"):
        mode_switch_latency(PROPORTIONAL, SWITCHING, DT, level=0.1, horizon=0.5)
