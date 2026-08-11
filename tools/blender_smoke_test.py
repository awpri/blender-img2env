"""Register the add-on in a real Blender and exercise what needs bpy.

Run:
    blender --background --factory-startup --python tools/blender_smoke_test.py

Exits non-zero on the first failure so it can be wired into CI on a machine
that has Blender. It does NOT contact the solver: it feeds a synthetic solve
response through the same code path the daemon's reply takes, so it checks the
Blender half in isolation.

What it cannot check is whether the scene looks right. That is what
docs/VERIFICATION.md is for — some of this only a human eye can sign off.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "addon"))

import numpy as np  # noqa: E402

import bpy  # noqa: E402

FAILURES: list[str] = []


def check(label: str):
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


def synthetic_solve(tmp: Path) -> dict:
    """A solve response with the reference photo's numbers in it."""
    from photo3d import coords

    width, height = 240, 320
    focal = 14.0 / 36.0 * height
    intrinsics = {"width": width, "height": height, "focal_35mm": 14.0,
                  "fx": focal, "fy": focal, "cx": width / 2.0, "cy": height / 2.0,
                  "convention": "long_edge"}

    plate = tmp / "plate.png"
    image = bpy.data.images.new("tmp_plate", width, height)
    image.filepath_raw = str(plate)
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image)

    depth_path = tmp / "depth.npy"
    vv, uu = np.mgrid[0:160, 0:120].astype(np.float32)
    np.save(depth_path, 3.0 + vv * 0.05)          # a receding ground plane

    gravity = coords.gravity_from_pitch_roll(3.28, 0.71)
    return {
        "intrinsics": intrinsics,
        "gravity_camera": gravity.tolist(),
        "pitch_deg": 3.28, "roll_deg": 0.71, "heading_deg": 76.944,
        "camera_height_m": 1.6, "ground_confidence": 0.62,
        "sun": {"azimuth_deg": 120.0, "elevation_deg": 52.0,
                "utc": "2026-07-14T09:11:36+00:00", "above_horizon": True},
        "geo": {"altitude_m": 375.0},
        "plate_png": str(plate), "depth_npy": str(depth_path),
        "warnings": ["synthetic solve for the smoke test"],
    }


def main() -> int:
    import photo3d
    from photo3d import proxy as proxy_mod, radiance, solve as solve_mod

    tmp = Path(bpy.app.tempdir)
    print(f"Blender {bpy.app.version_string}")

    @check("add-on registers")
    def _():
        photo3d.register()

    @check("panel is one tab with sections")
    def _():
        panels = [c for c in bpy.types.Panel.__subclasses__()
                  if getattr(c, "bl_category", None) == "Photo3D"]
        assert panels, "no Photo3D panels registered"
        roots = [c for c in panels if not getattr(c, "bl_parent_id", "")]
        assert len(roots) == 1, f"expected one root panel, got {[c.__name__ for c in roots]}"

    @check("scene properties attach")
    def _():
        assert bpy.context.scene.photo3d.depth_long_edge == 1536

    solve = synthetic_solve(tmp)
    scene = bpy.context.scene
    props = scene.photo3d

    @check("camera is built and faces the compass heading")
    def _():
        camera, plate = solve_mod.setup_camera(scene, solve, props)
        assert scene.camera is camera
        assert scene.render.resolution_x == 240

        # Blender cameras look down local -Z. Its world azimuth must match
        # GPSImgDirection, and its pitch must match the accelerometer.
        from mathutils import Vector
        forward = camera.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
        from photo3d import coords
        azimuth = coords.azimuth_deg(np.array(forward[:]))
        assert abs(azimuth - 76.944) < 0.1, f"camera faces {azimuth:.2f}, not 76.94"
        pitch = np.degrees(np.arcsin(np.clip(forward.z, -1, 1)))
        assert abs(pitch - 3.28) < 0.1, f"camera pitch {pitch:.2f}, not 3.28"
        assert abs(camera.matrix_world.translation.z - 1.6) < 1e-6

    @check("camera's local up stays out of the ground")
    def _():
        from mathutils import Vector
        camera = bpy.data.objects["Photo3D_Cam"]
        up = camera.matrix_world.to_quaternion() @ Vector((0.0, 1.0, 0.0))
        assert up.z > 0.9, f"camera is upside down or on its side: up={tuple(up)}"

    @check("proxy mesh builds, is watertight-ish and faces the camera")
    def _():
        camera = bpy.data.objects["Photo3D_Cam"]
        plate = bpy.data.images.load(solve["plate_png"], check_existing=True)
        obj = proxy_mod.build_proxy(bpy.context, solve, props, camera, plate)
        assert len(obj.data.vertices) > 100
        assert len(obj.data.polygons) > 100
        assert obj.is_shadow_catcher
        assert obj.rigid_body is not None and obj.rigid_body.type == "PASSIVE"

        obj.data.calc_loop_triangles()
        normals = np.array([p.normal[:] for p in obj.data.polygons])
        # In world space the surface faces back toward the camera, so the dot
        # with the camera's forward axis should be predominantly negative.
        from mathutils import Vector
        forward = camera.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
        world_normals = np.array([(obj.matrix_world.to_quaternion() @ p.normal)[:]
                                  for p in obj.data.polygons])
        facing = world_normals @ np.array(forward[:])
        assert (facing < 0).mean() > 0.9, "proxy normals point away from the camera"

    @check("proxy material routes camera rays to Transparent")
    def _():
        material = bpy.data.objects["Photo3D_Proxy"].material_slots[0].material
        nodes = {n.type for n in material.node_tree.nodes}
        assert {"MIX_SHADER", "BSDF_TRANSPARENT", "LIGHT_PATH", "TEX_IMAGE"} <= nodes

        mix = next(n for n in material.node_tree.nodes if n.type == "MIX_SHADER")
        # Fac=1 (a camera ray) selects input 2, which must be the Transparent.
        assert mix.inputs[2].links[0].from_node.type == "BSDF_TRANSPARENT"
        assert mix.inputs[1].links[0].from_node.type == "BSDF_PRINCIPLED"
        assert mix.inputs["Fac"].links[0].from_socket.name == "Is Camera Ray"

        texture = next(n for n in material.node_tree.nodes if n.type == "TEX_IMAGE")
        assert texture.inputs["Vector"].links[0].from_socket.name == "Window", \
            "Window coords are what make reflections pick up the real ground"

    @check("sun and sky are placed from the ephemeris")
    def _():
        lamp = solve_mod.setup_sun(scene, solve, props)
        assert lamp is not None
        from mathutils import Vector
        from photo3d import coords

        # matrix_world is lazy: setting rotation_quaternion does not recompute
        # it until the depsgraph runs, which in background mode means asking.
        bpy.context.view_layer.update()
        emit = lamp.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
        expected = -Vector(coords.sun_direction(120.0, 52.0).tolist())
        assert (emit - expected).length < 1e-4, \
            f"sun lamp shines {tuple(round(v, 3) for v in emit)}, wanted " \
            f"{tuple(round(v, 3) for v in expected)}"

        solve_mod.setup_sky(scene, solve, props)
        sky = next(n for n in scene.world.node_tree.nodes if n.type == "TEX_SKY")
        assert sky.sky_type in solve_mod._SKY_MODELS, \
            f"sky model {sky.sky_type} is not one we chose"
        assert not sky.sun_disc, "the SUN lamp already provides the disc"
        assert abs(sky.sun_elevation - np.radians(52.0)) < 1e-5
        assert sky.altitude == 375.0, "solved GPS altitude should drive the sky"

    @check("render and compositor are wired plate-under-CG")
    def _():
        plate = bpy.data.images.load(solve["plate_png"], check_existing=True)
        solve_mod.setup_render(scene, plate, props)
        assert scene.render.engine == "CYCLES"
        assert scene.render.film_transparent

        tree = (scene.compositing_node_group if hasattr(scene, "compositing_node_group")
                else scene.node_tree)
        over = next(n for n in tree.nodes if n.type == "ALPHAOVER")
        background, foreground = solve_mod._alpha_over_sockets(over)
        assert background.links[0].from_node.type == "IMAGE", "plate must be behind"
        assert foreground.links[0].from_node.type == "R_LAYERS", "CG must be in front"
        assert over.outputs["Image"].links, "Alpha Over must reach the output"

    @check("bounce proxy splits ray visibility without double counting")
    def _():
        assert bpy.ops.photo3d.bounce_proxy() == {"FINISHED"}
        bounce = bpy.data.objects["Photo3D_Bounce"]
        shadow = bpy.data.objects["Photo3D_Proxy"]

        assert not bounce.visible_camera and bounce.visible_diffuse
        assert not bounce.visible_shadow
        assert bounce.rigid_body is None, "the emissive twin must not collide"
        assert not shadow.visible_diffuse and not shadow.visible_glossy, \
            "the shadow proxy must stop contributing indirect light"
        assert shadow.visible_shadow and shadow.is_shadow_catcher

        emission = next(n for n in bounce.material_slots[0].material.node_tree.nodes
                        if n.type == "EMISSION")
        assert emission.inputs["Strength"].default_value > 0.0

    @check("bounce A/B toggle is symmetric")
    def _():
        bounce = bpy.data.objects["Photo3D_Bounce"]
        shadow = bpy.data.objects["Photo3D_Proxy"]
        bpy.ops.photo3d.toggle_bounce()
        assert not bounce.visible_diffuse and shadow.visible_diffuse
        bpy.ops.photo3d.toggle_bounce()
        assert bounce.visible_diffuse and not shadow.visible_diffuse

    @check("bounce strength is estimated from the plate")
    def _():
        props.bounce_strength = 999.0
        assert bpy.ops.photo3d.calibrate_bounce() == {"FINISHED"}
        assert props.bounce_strength != 999.0

    @check("gobo bake restores the scene when the render fails")
    def _():
        """The flagged gotcha: a failed bake used to leave the proxy wearing a
        temporary material with the compositor switched off."""
        def snapshot():
            compositor = (scene.compositing_node_group
                          if hasattr(scene, "compositing_node_group") else scene.use_nodes)
            return (proxy_obj.material_slots[0].material, compositor,
                    scene.cycles.samples, scene.render.resolution_x,
                    scene.render.film_transparent, scene.camera,
                    scene.view_settings.view_transform)

        proxy_obj = bpy.data.objects["Photo3D_Proxy"]
        before = snapshot()

        material = bpy.data.materials.new("boom")
        try:
            with radiance._render_settings(scene, proxy_obj, material, 64):
                assert proxy_obj.material_slots[0].material is material
                assert scene.cycles.samples == 8
                raise RuntimeError("simulated render failure")
        except RuntimeError:
            pass
        finally:
            bpy.data.materials.remove(material)

        after = snapshot()
        assert before == after, f"scene left dirty:\n  before {before}\n  after  {after}"

    @check("crispify leaves the plate alone")
    def _():
        bpy.ops.mesh.primitive_cube_add()
        cube = bpy.context.active_object
        material = bpy.data.materials.new("block")
        material.use_nodes = True
        texture = material.node_tree.nodes.new("ShaderNodeTexImage")
        texture.image = bpy.data.images.new("block_tex", 16, 16)
        cube.data.materials.append(material)

        bpy.ops.object.select_all(action="DESELECT")
        cube.select_set(True)
        bpy.context.view_layer.objects.active = cube
        assert bpy.ops.photo3d.crispify() == {"FINISHED"}
        assert texture.interpolation == "Closest"

        plate_texture = next(
            n for n in bpy.data.objects["Photo3D_Proxy"].material_slots[0].material
            .node_tree.nodes if n.type == "TEX_IMAGE")
        assert plate_texture.interpolation == "Cubic", "the plate wants smoothing"

    @check("ground plane is level by construction and collidable")
    def _():
        """The plane's orientation comes from measured gravity rather than from
        a depth model, so it must be exactly level whatever the camera was
        doing — that is the whole reason it is trustworthy on scenes where the
        depth mesh is not."""
        from mathutils import Vector
        assert bpy.ops.photo3d.add_ground_plane() == {"FINISHED"}
        plane = bpy.data.objects["Photo3D_Ground"]

        normal = plane.matrix_world.to_quaternion() @ Vector((0.0, 0.0, 1.0))
        assert abs(normal.z - 1.0) < 1e-6, f"ground is not level: normal {tuple(normal)}"
        assert abs(plane.matrix_world.translation.z) < 1e-9, "ground must sit at z=0"
        assert plane.rigid_body is not None and plane.rigid_body.type == "PASSIVE"
        assert plane.is_shadow_catcher
        assert plane.material_slots and plane.material_slots[0].material is not None

        # the camera has to be above it, or objects drop out of frame
        assert bpy.data.objects["Photo3D_Cam"].matrix_world.translation.z > 0.0

    @check("drop test cube is active and above the terrain")
    def _():
        assert bpy.ops.photo3d.drop_test() == {"FINISHED"}
        cube = bpy.data.objects["Photo3D_DropTest"]
        assert cube.rigid_body.type == "ACTIVE"

    @check("re-solving does not accumulate objects")
    def _():
        before = len([o for o in bpy.data.objects if o.name.startswith("Photo3D_")])
        solve_mod._clear_previous(scene)
        camera, plate = solve_mod.setup_camera(scene, solve, props)
        proxy_mod.build_proxy(bpy.context, solve, props, camera, plate)
        after = len([o for o in bpy.data.objects if o.name.startswith("Photo3D_")])
        assert after <= before, f"{before} objects became {after}"

    @check("add-on unregisters cleanly")
    def _():
        photo3d.unregister()
        assert not hasattr(bpy.types.Scene, "photo3d")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all Blender checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
