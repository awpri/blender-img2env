"""Package addon/photo3d into a zip Blender can install.

    python tools/build_addon_zip.py
    Blender > Edit > Preferences > Add-ons > Install from Disk > photo3d.zip

The zip is self-contained, which is the reason coords.py and imaging.py live
inside the add-on rather than in a shared top-level package: whatever Blender
gets has to work with no path setup and no pip.
"""

from __future__ import annotations

import ast
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKAGE = REPO / "addon" / "photo3d"
OUTPUT = REPO / "photo3d.zip"

ALLOWED = set(sys.stdlib_module_names) | {"bpy", "mathutils", "numpy", "bmesh", "gpu"}


def check_dependencies() -> list[str]:
    """Refuse to ship an add-on that would need something installed into
    Blender's Python. Same check as test_addon.py, repeated here so the failure
    happens at build time rather than at enable time on someone else's machine.
    """
    problems = []
    for path in sorted(PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                names.add(node.module.split(".")[0])
        for forbidden in sorted(names - ALLOWED):
            problems.append(f"{path.name} imports {forbidden}")
    return problems


def main() -> int:
    problems = check_dependencies()
    if problems:
        print("refusing to build — the add-on may only import bpy, mathutils, "
              "numpy and stdlib:")
        for problem in problems:
            print(f"  {problem}")
        return 1

    sources = sorted(PACKAGE.glob("*.py"))
    if not sources:
        print(f"nothing to package in {PACKAGE}")
        return 1

    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, Path("photo3d") / path.name)

    size_kb = OUTPUT.stat().st_size / 1024
    print(f"wrote {OUTPUT.relative_to(REPO)} ({size_kb:.0f} KB, {len(sources)} modules)")
    print("install with: Blender > Preferences > Add-ons > Install from Disk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
