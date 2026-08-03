"""Integration tests running every example script under a reduced step count."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

# Every example, with the arguments that shrink it to a fast integration run.
# "{tmp}" is replaced by a temporary directory, so that an example whose whole
# job is to write files can be run without touching the tracked ones.
EXAMPLE_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "force_control": ("--objects", "drinking_glass", "--duration", "1.4", "--no-figures"),
    "grasp_evaluation": ("--duration", "1.4", "--lift-time", "0.9", "--no-figures"),
    "grasp_taxonomy": ("--no-feasibility",),
    "hand_kinematics": ("--samples", "21", "--no-figures"),
    "mode_switching": ("--duration", "1.0", "--bursts", "0.3"),
    "proportional_control": ("--samples", "201", "--timing-steps", "200", "--no-figures"),
    "readme_figures": ("--figure-dir", "{tmp}", "--duration", "1.2"),
    "slip_recovery": ("--duration", "1.4", "--lift-time", "0.9", "--no-figures"),
}


def _example_names() -> tuple[str, ...]:
    return tuple(
        sorted(path.stem for path in EXAMPLES_DIR.glob("*.py") if not path.stem.startswith("_"))
    )


def _load(name: str) -> ModuleType:
    path = EXAMPLES_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"hand_controller_example_{name}", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load example {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_every_example_is_covered() -> None:
    """A new example script must be added to the integration table."""
    assert _example_names() == tuple(sorted(EXAMPLE_ARGUMENTS))


@pytest.mark.parametrize("name", sorted(EXAMPLE_ARGUMENTS))
def test_example_runs_to_completion(
    name: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load(name)
    arguments: Sequence[str] = tuple(
        value.replace("{tmp}", str(tmp_path)) for value in EXAMPLE_ARGUMENTS[name]
    )
    assert module.main(arguments) == 0
    captured = capsys.readouterr()
    assert captured.out.strip()
    assert captured.err == ""


@pytest.mark.parametrize("name", sorted(EXAMPLE_ARGUMENTS))
def test_example_exposes_a_parser(name: str) -> None:
    module = _load(name)
    parser = module.build_parser()
    assert parser.description


def test_example_writes_figures(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The figure writing path is exercised at least once."""
    module = _load("force_control")
    arguments = (
        "--objects", "foam_cup",
        "--duration", "1.4",
        "--figure-dir", str(tmp_path),
    )
    assert module.main(arguments) == 0
    capsys.readouterr()
    assert sorted(path.name for path in tmp_path.glob("*.png")) == ["force_foam_cup.png"]


def test_kinematics_example_writes_its_figure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load("hand_kinematics")
    assert module.main(("--samples", "21", "--figure-dir", str(tmp_path))) == 0
    capsys.readouterr()
    assert sorted(path.name for path in tmp_path.glob("*.png")) == ["grasp_spans.png"]


def test_control_example_writes_its_figure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load("proportional_control")
    arguments = (
        "--samples", "201",
        "--timing-steps", "200",
        "--figure-dir", str(tmp_path),
    )
    assert module.main(arguments) == 0
    capsys.readouterr()
    assert sorted(path.name for path in tmp_path.glob("*.png")) == ["control_characteristic.png"]


def test_the_readme_figure_script_writes_exactly_the_published_pair(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One command regenerates the tracked figures, and reports what they cost."""
    module = _load("readme_figures")
    assert module.main(("--figure-dir", str(tmp_path), "--duration", "1.2")) == 0
    captured = capsys.readouterr()
    assert sorted(path.name for path in tmp_path.glob("*.png")) == [
        "grasp_postures.png",
        "slip_recovery.png",
    ]
    total = sum(path.stat().st_size for path in tmp_path.glob("*.png"))
    assert total <= module.BUDGET_BYTES
    assert f"total {total} bytes" in captured.out


def test_slip_example_writes_its_figure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load("slip_recovery")
    arguments = (
        "--duration", "1.4",
        "--lift-time", "0.9",
        "--detail", "drinking_glass",
        "--figure-dir", str(tmp_path),
    )
    assert module.main(arguments) == 0
    capsys.readouterr()
    assert sorted(path.name for path in tmp_path.glob("*.png")) == ["slip_drinking_glass.png"]
