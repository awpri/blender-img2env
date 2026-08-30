"""Install the packaged add-on into a throwaway Blender profile and check it.

Two things, both of which have actually gone wrong:

  1. A FRESH install registers every operator and panel the source defines. It
     is possible to add a class and forget to add it to CLASSES, and the source
     then looks correct while the installed add-on is missing a button.

  2. An UPGRADE into a running Blender picks up the new code. Blender reloads
     the add-on's __init__.py when it changes on disk but leaves the submodules
     in sys.modules, so a new install reports its new version while still
     running the previous version's panels and operators. That failure is
     invisible from the source tree and maddening from the outside — the
     version number says the upgrade worked.

Run:
    blender --background --python tools/addon_install_test.py

Uses BLENDER_USER_RESOURCES so it never touches a real configuration.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

import bpy

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP = os.path.join(REPO, "photo3d.zip")

#: Buttons a user is told about in the docs. Losing one silently is the failure
#: this list exists to catch.
EXPECTED_OPS = {
    "solve", "solve_metadata", "check_server", "align_view",
    "rebuild_proxy", "add_ground_plane", "drop_test", "crispify",
    "calibrate_exposure", "calibrate_bounce", "bounce_proxy", "toggle_bounce",
    "fetch_shade_mask", "bake_gobo", "load_panorama",
    "edit_proxy", "split_material_region", "segment_proxy",
    "reset_camera", "diagnose", "measure_shadow", "light_from_selection",
    "probe_at_object",
}
EXPECTED_PANELS = {
    "PHOTO3D_PT_main", "PHOTO3D_PT_camera", "PHOTO3D_PT_proxy",
    "PHOTO3D_PT_bounce", "PHOTO3D_PT_sun", "PHOTO3D_PT_gobo",
    "PHOTO3D_PT_materials", "PHOTO3D_PT_panorama", "PHOTO3D_PT_minecraft",
}

MARKER = '''class PHOTO3D_OT_upgrade_marker(bpy.types.Operator):
    """marker planted by the upgrade test"""
    bl_idname = "photo3d.upgrade_marker"
    bl_label = "Marker"

    def execute(self, context):
        return {"FINISHED"}


CLASSES = (PHOTO3D_OT_upgrade_marker, PHOTO3D_OT_calibrate_exposure,'''


def registered():
    panels = {c.__name__ for c in bpy.types.Panel.__subclasses__()
              if getattr(c, "bl_category", None) == "Photo3D"}
    return set(dir(bpy.ops.photo3d)), panels


def main() -> int:
    if not os.path.exists(ZIP):
        print(f"no {ZIP} — run 'make addon' first")
        return 1

    failures = []
    bpy.ops.preferences.addon_install(filepath=ZIP, overwrite=True)
    bpy.ops.preferences.addon_enable(module="photo3d")

    ops, panels = registered()
    missing_ops = EXPECTED_OPS - ops
    missing_panels = EXPECTED_PANELS - panels
    if missing_ops:
        failures.append(f"fresh install is missing operators: {sorted(missing_ops)}")
    if missing_panels:
        failures.append(f"fresh install is missing panels: {sorted(missing_panels)}")
    print(f"  fresh install: {len(ops)} operators, {len(panels)} panels")

    # --- the upgrade path -------------------------------------------------
    addon_dir = os.path.dirname(sys.modules["photo3d"].__file__)
    radiance = os.path.join(addon_dir, "radiance.py")
    source = open(radiance).read()
    if "CLASSES = (PHOTO3D_OT_calibrate_exposure," not in source:
        failures.append("upgrade test could not find its anchor in radiance.py")
    else:
        open(radiance, "w").write(
            source.replace("CLASSES = (PHOTO3D_OT_calibrate_exposure,", MARKER, 1))
        init = os.path.join(addon_dir, "__init__.py")
        stamp = time.time() + 2
        os.utime(init, (stamp, stamp))       # what a real re-install looks like

        bpy.ops.preferences.addon_disable(module="photo3d")
        bpy.ops.preferences.addon_enable(module="photo3d")
        if "upgrade_marker" not in dir(bpy.ops.photo3d):
            failures.append(
                "upgrading in a running Blender kept the OLD submodules — the "
                "add-on would report its new version while running old code")
        else:
            print("  upgrade in place: new code picked up")

    print()
    for problem in failures:
        print(f"  FAIL  {problem}")
    if failures:
        return 1
    print("add-on install checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
