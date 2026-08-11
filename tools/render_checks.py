"""Actually render, and measure whether the light behaves.

    blender --background --factory-startup --python tools/render_checks.py

Everything in blender_smoke_test.py checks that the scene is wired correctly.
Wiring can be perfect and the picture still wrong, because Cycles decides what
a shadow catcher and a Transparent BSDF mean together, not us. These checks
render small images and compare pixels, which is the only way to know.

Each check renders twice and diffs, so it measures a *difference* rather than
an absolute — no golden images to rot.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "addon"))

import numpy as np  # noqa: E402

import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

FAILURES: list[str] = []
RES = 160
SAMPLES = 24


def check(label):
    def decorator(fn):
        try:
            fn()
            print(f"  ok    {label}")
        except Exception as exc:                                  # noqa: BLE001
            FAILURES.append(label)
            print(f"  FAIL  {label}: {exc}")
            traceback.print_exc()
        return fn
    return decorator


#: Render Result pixels are not readable from a background Blender, so the
#: only way to measure a render is to write it and read it back. EXR keeps the
#: alpha linear and unclamped, which is what the shadow-catcher matte lives in.
_OUT = Path(__file__).resolve().parent.parent / ".render_checks"
_counter = [0]


def render_rgba() -> np.ndarray:
    """Render the current scene and return (H, W, 4) float32."""
    scene = bpy.context.scene
    scene.render.resolution_x = scene.render.resolution_y = RES
    scene.render.resolution_percentage = 100
    scene.cycles.samples = SAMPLES
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "32"

    _OUT.mkdir(exist_ok=True)
    _counter[0] += 1
    path = _OUT / f"check_{_counter[0]:03d}.exr"
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    if not path.exists():
        raise RuntimeError(f"render produced no file at {path}")

    image = bpy.data.images.load(str(path), check_existing=False)
    buf = np.empty(image.size[0] * image.size[1] * 4, dtype=np.float32)
    image.pixels.foreach_get(buf)
    out = buf.reshape(image.size[1], image.size[0], 4)[::-1]     # Blender is bottom-up
    bpy.data.images.remove(image)
    return out


def build_scene(with_shadow_catcher_material: bool):
    """A minimal solved-scene stand-in: level ground, sun overhead-ish, camera
    looking along +Y from 1.6 m, exactly as a real solve produces."""
    from photo3d import proxy as proxy_mod

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    scene.cycles.use_denoising = False

    cam_data = bpy.data.cameras.new("Photo3D_Cam")
    cam_data.lens = 24.0
    cam_data.sensor_width = 36.0
    camera = bpy.data.objects.new("Photo3D_Cam", cam_data)
    scene.collection.objects.link(camera)
    # look horizontally along +Y from 1.6 m, tipped down 20 degrees
    camera.rotation_euler = (np.radians(70.0), 0.0, 0.0)
    camera.location = (0.0, -4.0, 1.6)
    scene.camera = camera

    sun_data = bpy.data.lights.new("Photo3D_Sun", type="SUN")
    sun_data.energy = 4.0
    sun_data.angle = np.radians(0.545)
    sun = bpy.data.objects.new("Photo3D_Sun", sun_data)
    scene.collection.objects.link(sun)
    sun.rotation_euler = (np.radians(35.0), 0.0, np.radians(20.0))

    plate = bpy.data.images.new("plate", 64, 64)
    pixels = np.zeros((64, 64, 4), dtype=np.float32)
    pixels[..., 0] = 0.8          # a strongly red plate, so reflections are obvious
    pixels[..., 1] = 0.1
    pixels[..., 2] = 0.1
    pixels[..., 3] = 1.0
    plate.pixels.foreach_set(pixels.ravel())

    bpy.ops.mesh.primitive_plane_add(size=40.0, location=(0.0, 0.0, 0.0))
    ground = bpy.context.active_object
    ground.name = "Photo3D_Proxy"
    if with_shadow_catcher_material:
        ground.data.materials.append(proxy_mod.make_proxy_material(plate))
    else:
        mat = bpy.data.materials.new("plain")
        mat.use_nodes = True
        ground.data.materials.append(mat)
    ground.is_shadow_catcher = True
    return scene, ground, plate


def add_caster():
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 1.0))
    cube = bpy.context.active_object
    cube.name = "Caster"
    return cube


def main() -> int:
    import photo3d
    photo3d.register()          # the compositor check needs scene.photo3d
    print(f"Blender {bpy.app.version_string}  ({RES}x{RES}, {SAMPLES} samples)\n")

    def shadow_pixels(ground) -> tuple[int, float]:
        """Count pixels that are shadow rather than caster.

        Render the caster with the ground hidden to get its own silhouette,
        then again with the ground catching. Anything opaque in the second and
        not in the first is shadow. Sampling a fixed band instead would depend
        on guessing where the shadow lands, which is how the first version of
        this check managed to report a false failure.
        """
        ground.hide_render = True
        add_caster()
        caster_only = render_rgba()
        ground.hide_render = False
        together = render_rgba()

        caster = caster_only[..., 3] > 0.01
        shadow = (together[..., 3] > 0.01) & ~caster
        strength = float(together[..., 3][shadow].mean()) if shadow.any() else 0.0
        return int(shadow.sum()), strength

    @check("CG object casts a visible shadow onto the proxy")
    def _():
        """The thing the whole shadow-catcher setup exists for.

        This is what caught the Is-Camera-Ray/Transparent mix silently
        cancelling the shadow catcher: a camera ray that passes through the
        proxy never hits it, so Cycles has no surface to darken.
        """
        _, ground, _ = build_scene(with_shadow_catcher_material=True)
        count, strength = shadow_pixels(ground)
        print(f"        {count} shadow pixels, mean alpha {strength:.3f}")
        assert count > 50, ("no shadow reached the ground — check that the proxy "
                            "material is not transparent to camera rays")

    @check("shadow survives the bounce-proxy ray-visibility split")
    def _():
        """Build Bounce Proxy turns the shadow proxy's diffuse/glossy visibility
        off. If that also killed the shadow the two features would be
        incompatible in practice, however good they look separately."""
        _, ground, _ = build_scene(with_shadow_catcher_material=True)
        ground.visible_diffuse = False
        ground.visible_glossy = False
        ground.visible_transmission = False
        count, strength = shadow_pixels(ground)
        print(f"        {count} shadow pixels, mean alpha {strength:.3f}")
        assert count > 50, "the bounce split silences shadows on the shadow proxy"

    @check("shadow composites over the plate through Alpha Over")
    def _():
        """A shadow in the alpha channel is not a shadow the audience sees. The
        compositor has to turn it into darkened plate.

        The mask is derived from the uncomposited renders first: once Alpha
        Over has laid the opaque plate down, alpha is 1.0 everywhere and can no
        longer tell shadow from anything else.
        """
        from photo3d import solve as solve_mod

        scene, ground, plate = build_scene(with_shadow_catcher_material=True)
        ground.hide_render = True
        add_caster()
        caster_only = render_rgba()
        ground.hide_render = False
        together = render_rgba()
        caster = caster_only[..., 3] > 0.01
        shadow = (together[..., 3] > 0.5) & ~caster
        assert shadow.sum() > 50, "no shadow region to composite"

        solve_mod.setup_render(scene, plate, scene.photo3d)
        composited = render_rgba()

        open_plate = (~caster) & (~shadow) & (composited[..., 0] > 0.01)
        shaded = float(composited[..., 0][shadow].mean())
        lit = float(composited[..., 0][open_plate].mean())
        print(f"        composited plate red: lit {lit:.3f}, shadowed {shaded:.3f}")
        assert shaded < lit * 0.95, "the shadow did not darken the composited plate"

    @check("chrome sphere reflects the plate, not grey")
    def _():
        """M4's other half: Window coordinates are what make a polished block
        reflect the actual ground it stands on."""
        build_scene(with_shadow_catcher_material=True)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.9, location=(0.0, 0.0, 0.9))
        sphere = bpy.context.active_object
        mat = bpy.data.materials.new("chrome")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Metallic"].default_value = 1.0
        bsdf.inputs["Roughness"].default_value = 0.05
        sphere.data.materials.append(mat)

        image = render_rgba()
        # the sphere sits in the middle of frame; the plate is strongly red
        centre = image[int(RES * 0.35):int(RES * 0.62), int(RES * 0.35):int(RES * 0.65)]
        lit = centre[centre[..., 3] > 0.5]
        assert lit.size, "the sphere did not render"
        redness = float(lit[..., 0].mean() - lit[..., 2].mean())
        print(f"        sphere red-minus-blue {redness:+.4f}")
        assert redness > 0.01, ("the chrome sphere is not picking up the red plate — "
                                "check the Window coordinate link")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all render checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
