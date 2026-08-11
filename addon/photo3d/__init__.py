bl_info = {
    "name": "Photo3D — photo to solved scene",
    "author": "Alexander",
    "version": (0, 4, 0),
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

from . import coords, imaging  # noqa: E402,F401  (bpy-free, always safe)

if _IN_BLENDER:
    from . import props, solve, proxy, radiance, ui  # noqa: E402

    _MODULES = (props, solve, proxy, radiance, ui)

    def register():
        for module in _MODULES:
            module.register()

    def unregister():
        for module in reversed(_MODULES):
            module.unregister()
