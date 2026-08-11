"""Structural checks on the add-on, without Blender.

bpy cannot be imported here, so these parse the source instead. The one that
earns its keep is the dependency allowlist: "never install packages into
Blender's bundled Python" is a constraint that is easy to state, easy to agree
with, and easy to break six months later with one convenient `import requests`.
A test makes it fail immediately instead.

For behaviour that genuinely needs Blender, see tools/blender_smoke_test.py.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ADDON = Path(__file__).resolve().parents[2] / "addon" / "photo3d"
MODULES = sorted(ADDON.glob("*.py"))

#: Everything the add-on is allowed to import. Blender bundles numpy; bpy and
#: mathutils are Blender itself. Anything else means a broken install for the
#: user and, historically, a broken Blender.
ALLOWED = set(sys.stdlib_module_names) | {"bpy", "mathutils", "numpy", "bmesh", "gpu"}

#: bpy-free by contract: the server imports these, and pytest imports them here.
PURE_MODULES = {"coords.py", "imaging.py"}


def top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:                    # relative import, our own package
                continue
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def test_there_are_modules_to_check():
    assert MODULES, "no add-on modules found; the path bootstrap is wrong"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_module_compiles(path):
    compile(path.read_text(), str(path), "exec")


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_only_permitted_dependencies(path):
    """The hard constraint from the handoff, mechanised."""
    forbidden = top_level_imports(path) - ALLOWED
    assert not forbidden, (
        f"{path.name} imports {sorted(forbidden)}, which would have to be "
        "installed into Blender's bundled Python. Move that work to the solver "
        "daemon and reach it over HTTP.")


@pytest.mark.parametrize("name", sorted(PURE_MODULES))
def test_shared_core_modules_do_not_import_bpy(name):
    """coords.py and imaging.py are imported by the server, so a stray bpy
    import in either breaks the daemon rather than the add-on — and it breaks
    it at request time, not at import time."""
    imports = top_level_imports(ADDON / name)
    assert "bpy" not in imports and "mathutils" not in imports


def test_package_imports_outside_blender():
    """__init__.py guards its bpy imports so the server can import coords."""
    from photo3d import coords, imaging
    assert hasattr(coords, "rotation_from_gravity")
    assert hasattr(imaging, "box_blur")


def test_bl_info_is_complete():
    tree = ast.parse((ADDON / "__init__.py").read_text())
    info = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "bl_info":
            info = ast.literal_eval(node.value)
    assert info is not None, "bl_info must be a literal at module level or Blender cannot read it"
    for key in ("name", "author", "version", "blender", "location", "description", "category"):
        assert key in info, f"bl_info is missing {key}"
    assert info["blender"] >= (4, 2, 0), "Cycles-only features assume 4.2 LTS or newer"


def _class_attributes(path: Path) -> dict[str, dict]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found: dict[str, dict] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        attributes = {}
        for statement in node.body:
            if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Constant):
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        attributes[target.id] = statement.value.value
        found[node.name] = attributes
    return found


def test_operator_idnames_share_the_photo3d_namespace():
    for path in MODULES:
        for name, attributes in _class_attributes(path).items():
            idname = attributes.get("bl_idname")
            if idname and name.startswith("PHOTO3D_OT_"):
                assert idname.startswith("photo3d."), f"{name} has bl_idname {idname!r}"


def test_the_ui_is_one_panel_with_sections():
    """The handoff asks for one panel with sections, not two tabs. In Blender
    terms: exactly one root panel, everything else parented to it, one
    bl_category."""
    panels = {name: attributes for name, attributes in _class_attributes(ADDON / "ui.py").items()
              if name.startswith("PHOTO3D_PT_")}
    assert panels, "no panels found"

    roots = [name for name, attributes in panels.items() if "bl_parent_id" not in attributes]
    assert len(roots) == 1, f"expected one root panel, found {roots}"

    for name, attributes in panels.items():
        parent = attributes.get("bl_parent_id")
        if parent:
            assert parent == "PHOTO3D_PT_main", f"{name} parents to {parent!r}"


def test_every_module_registers_and_unregisters():
    """__init__ calls register()/unregister() on each submodule; a missing one
    is an AttributeError at add-on enable time with a useless traceback."""
    for path in MODULES:
        if path.name in PURE_MODULES | {"__init__.py", "client.py"}:
            continue
        source = path.read_text()
        assert "\ndef register():" in source, f"{path.name} has no register()"
        assert "\ndef unregister():" in source, f"{path.name} has no unregister()"


def test_init_registers_every_operator_module():
    source = (ADDON / "__init__.py").read_text()
    for module in ("props", "solve", "proxy", "radiance", "ui"):
        assert module in source, f"__init__.py never imports {module}"
