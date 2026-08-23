bl_info = {
    "name": "Photo3D — photo to solved scene",
    "author": "Alexander",
    "version": (0, 13, 0),
    "blender": (4, 2, 0),
    "location": "View3D > N-panel > Photo3D",
    "description": "Solve camera, proxy geometry and image-derived lighting from a single photo.",
    "doc_url": "https://github.com/awpri/blender-img2env",
    "category": "3D View",
}

"""Photo3D add-on.

Imports bpy, mathutils, numpy and stdlib. Nothing else, ever — installing
packages into Blender's bundled Python is the constraint the whole two-process
architecture exists to respect. Anything needing torch, rawpy or exiftool lives
in the solver daemon and is reached over localhost HTTP.

Images cross that boundary as filesystem paths and depth crosses as .npy
float32. Never base64 (a 48 MP plate is ~25 MB), never PNG for depth (the
quantisation destroys metric precision and with it the whole collision story).

The two submodules that contain no bpy calls — coords and imaging — are also
imported by the server, so this file must stay importable outside Blender. The
guard below is what makes that true.
"""

try:
    import bpy  # noqa: F401
    _IN_BLENDER = True
except ModuleNotFoundError:      # imported by the server or by pytest
    _IN_BLENDER = False

#: Every submodule, in dependency order. Reloaded as a group on re-enable.
_SUBMODULE_NAMES = ("coords", "imaging", "props", "solve", "proxy", "materials",
                    "radiance", "ui")


def _reload_stale_submodules():
    """Re-import submodules that Python has cached from an earlier install.

    Installing a new version into a RUNNING Blender otherwise appears to do
    almost nothing. Blender notices this file changed on disk and reloads it —
    so bl_info updates and the version number in the panel goes up — but
    `from . import ui` then returns whatever is already in sys.modules, which
    is the previous version's code. The result is a add-on that reports the new
    version while running the old panels and operators, which is a genuinely
    baffling thing to debug from the outside.

    Reloading here is safe because Blender calls unregister() before it
    re-enables, so no stale classes are still registered when this runs.
    """
    import importlib
    import sys

    for name in _SUBMODULE_NAMES:
        cached = sys.modules.get(f"{__name__}.{name}")
        if cached is not None:
            importlib.reload(cached)


_reload_stale_submodules()

from . import coords, imaging  # noqa: E402,F401  (bpy-free, always safe)

if _IN_BLENDER:
    from . import props, solve, proxy, materials, radiance, ui  # noqa: E402

    _MODULES = (props, solve, proxy, materials, radiance, ui)

    def register():
        for module in _MODULES:
            module.register()

    def unregister():
        for module in reversed(_MODULES):
            module.unregister()
