"""What the repository ships: the typing marker and the published figures.

These are the two things that are correct in the working tree and still wrong for
everybody else if they are left out of the package or out of version control.
"""

from __future__ import annotations

import re
from pathlib import Path

import hand_controller
from hand_controller.analysis.figures import PUBLISHED_DPI

ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = ROOT / "docs" / "figures"
README = ROOT / "README.md"
BUDGET_BYTES = 250 * 1024

_IMAGE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<target>[^)\s]+)\)")


def test_the_package_ships_a_typing_marker() -> None:
    """PEP 561. Without this file the strict typing is invisible to an installer."""
    package = Path(hand_controller.__file__).resolve().parent
    marker = package / "py.typed"
    assert marker.is_file()
    assert marker.parent.name == "hand_controller"
    assert marker.read_bytes() == b""
    assert (ROOT / "src" / "hand_controller" / "py.typed").is_file()


def test_the_published_figures_are_tracked_and_inside_the_budget() -> None:
    """A README figure is only published if it is small enough to be committed."""
    figures = sorted(FIGURE_DIR.glob("*.png"))
    assert [path.name for path in figures] == ["grasp_postures.png", "slip_recovery.png"]
    total = sum(path.stat().st_size for path in figures)
    assert 0 < total <= BUDGET_BYTES
    assert PUBLISHED_DPI <= 130


def test_every_local_image_in_the_readme_exists_and_says_what_it_shows() -> None:
    """Alt text is the caption a reader gets when the image does not load."""
    text = README.read_text(encoding="utf-8")
    local = [
        (match.group("alt"), match.group("target"))
        for match in _IMAGE.finditer(text)
        if not match.group("target").startswith(("http://", "https://"))
    ]
    assert len(local) == len(sorted(FIGURE_DIR.glob("*.png")))
    for alt, target in local:
        assert (ROOT / target).is_file(), target
        assert len(alt.strip()) >= 20, alt
