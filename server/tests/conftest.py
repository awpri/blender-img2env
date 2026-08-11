"""Path bootstrap and shared fixtures.

Two trees have to be importable: the repo root (for `server.*`) and addon/ (for
`photo3d.*`, the dependency-free core the server shares with the add-on).
Importing `photo3d` outside Blender is safe by construction — see
addon/photo3d/__init__.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"

for path in (REPO_ROOT, REPO_ROOT / "addon"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_fixture(name: str) -> dict:
    """Read an `exiftool -j` dump from fixtures/ and return the record."""
    data = json.loads((FIXTURES / name).read_text())
    return data[0] if isinstance(data, list) else data


@pytest.fixture
def img7096() -> dict:
    """The reference photo: 14 mm ultra-wide, portrait, ProRAW, no coordinates."""
    return load_fixture("img7096_exif.json")


@pytest.fixture
def geneva() -> dict:
    """Same camera, but with coordinates, so the ephemeris path runs."""
    return load_fixture("sunny_geneva_exif.json")


@pytest.fixture
def orientation_set() -> dict[int, dict]:
    """One physical pose, four ways of holding the phone."""
    return {o: load_fixture(f"orientation_{o}_exif.json") for o in (1, 3, 6, 8)}
