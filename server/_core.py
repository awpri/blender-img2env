"""Bridge to the add-on's dependency-free core modules.

coords.py and imaging.py live in addon/photo3d/ because the add-on has to be
self-contained when it is zipped into Blender. The server needs bit-identical
maths — the same gravity axis map, the same unprojection, the same sign for
roll — so it imports those modules from the add-on tree instead of keeping its
own copy. A vendored copy would drift, and the drift would present as an
accelerometer bug rather than as the merge conflict it actually is.

addon/photo3d/__init__.py guards its bpy imports, so importing the package
outside Blender loads nothing but stdlib and numpy.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ADDON_ROOT = Path(__file__).resolve().parent.parent / "addon"
if str(_ADDON_ROOT) not in sys.path:
    sys.path.insert(0, str(_ADDON_ROOT))

from photo3d import coords, imaging  # noqa: E402  (path bootstrap must run first)

__all__ = ["coords", "imaging"]
