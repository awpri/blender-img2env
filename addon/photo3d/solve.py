"""Camera, lighting, render setup, and the operator that drives the solve.

The camera pose is not eyeballed and not fitted: gravity gives pitch and roll,
the compass gives yaw, metric depth gives height. All of the arithmetic lives
in coords.py, where it is unit-tested with synthetic inputs; this module only
converts the result into Blender objects.
"""

from __future__ import annotations

from math import degrees, radians

import bpy
from mathutils import Matrix, Vector

from . import client, coords, proxy


# ---------------------------------------------------------------------------
# camera
# ---------------------------------------------------------------------------

def camera_matrix_from_solve(solve: dict, props) -> Matrix:
    """World matrix for the solved camera.

    Falls back to a level, user-pitched camera when there is no accelerometer
    vector, because a photo from another camera should still get a usable
    scene rather than an error.
    """
    gravity = solve.get("gravity_camera")
    if gravity is None:
        gravity = coords.gravity_from_pitch_roll(props.fallback_pitch, 0.0)

    heading = solve.get("heading_deg") if props.use_heading else None
    matrix = coords.camera_matrix(
        gravity, heading, float(solve.get("camera_height_m") or props.fallback_height),
        fallback_heading_deg=props.fallback_heading)
    return Matrix([list(row) for row in matrix])


def setup_camera(scene, solve: dict, props):
    intrinsics = solve["intrinsics"]

    data = bpy.data.cameras.new("Photo3D_Cam")
    # AUTO applies sensor_width to the LONGER render dimension, which is exactly
    # the long-edge convention the intrinsics were built with. Setting
    # HORIZONTAL/VERTICAL by hand re-derives the same thing and gets it wrong
    # for portrait frames.
    data.sensor_fit = "AUTO"
    data.sensor_width = 36.0
    data.lens = float(intrinsics["focal_35mm"])
    if intrinsics.get("convention") == "diagonal":
        # The diagonal basis is not expressible as a sensor width alone, so
        # convert the solved pixel focal back into Blender's terms.
        data.lens = 36.0 * intrinsics["fx"] / max(intrinsics["width"], intrinsics["height"])
    data.clip_start = 0.05
    data.clip_end = max(1000.0, props.max_depth * 10.0)

    camera = bpy.data.objects.new("Photo3D_Cam", data)
    scene.collection.objects.link(camera)
    camera.matrix_world = camera_matrix_from_solve(solve, props)
    scene.camera = camera

    plate = bpy.data.images.load(solve["plate_png"], check_existing=True)
    data.show_background_images = True
    background = data.background_images.new()
    background.image = plate
    background.alpha = 1.0
    background.display_depth = "BACK"
    background.frame_method = "FIT"

    scene.render.resolution_x = int(intrinsics["width"])
    scene.render.resolution_y = int(intrinsics["height"])
    scene.render.resolution_percentage = int(props.render_percent)
    return camera, plate


# ---------------------------------------------------------------------------
# lighting
# ---------------------------------------------------------------------------

def setup_sun(scene, solve: dict, props):
    """Direct sun from the solar ephemeris.

    This is one of the two things the ephemeris is good for — direction and
    hardness of the key. Everything else (green bounce off grass, warm bounce
    off stone, the entirety of any interior) comes from the image itself, via
    the bounce proxy in radiance.py.
    """
    sun_info = solve.get("sun")
    if not sun_info or sun_info["elevation_deg"] < -3.0:
        return None

    azimuth = float(sun_info["azimuth_deg"])
    elevation = float(sun_info["elevation_deg"])
    to_sun = Vector(coords.sun_direction(azimuth, elevation).tolist())

    data = bpy.data.lights.new("Photo3D_Sun", type="SUN")
    data.angle = radians(0.545)              # the real angular size of the solar disc
    data.energy = props.sun_strength
    lamp = bpy.data.objects.new("Photo3D_Sun", data)
    scene.collection.objects.link(lamp)

    lamp.location = to_sun * 100.0
    lamp.rotation_mode = "QUATERNION"
    # A sun lamp emits along its local -Z, so point that at the scene.
    lamp.rotation_quaternion = Vector((0.0, 0.0, -1.0)).rotation_difference(-to_sun)
    return lamp


#: Physical sky models, best first. Blender 4.x called the atmospheric model
#: NISHITA; 5.0 split it into the two scattering variants it was always built
#: from and exposed them by name. Preetham and Hosek-Wilkie are analytic
#: fallbacks — usable, but they are fits to clear-sky data rather than a
#: scattering integral, so they respond less well to the solved altitude.
_SKY_MODELS = ("MULTIPLE_SCATTERING", "NISHITA", "SINGLE_SCATTERING", "HOSEK_WILKIE")


def _set_physical_sky(sky_node):
    available = sky_node.bl_rna.properties["sky_type"].enum_items.keys()
    for model in _SKY_MODELS:
        if model in available:
            sky_node.sky_type = model
            return model
    return sky_node.sky_type


def setup_sky(scene, solve: dict, props):
    sun_info = solve.get("sun")
    world = scene.world or bpy.data.worlds.new("Photo3D_World")
    scene.world = world
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()

    output = tree.nodes.new("ShaderNodeOutputWorld"); output.location = (400, 0)
    background = tree.nodes.new("ShaderNodeBackground"); background.location = (200, 0)
    background.inputs["Strength"].default_value = props.sky_strength
    tree.links.new(background.outputs["Background"], output.inputs["Surface"])

    if not sun_info:
        # No ephemeris: a flat grey ambient is honest about knowing nothing,
        # and the bounce proxy will do the real work anyway.
        background.inputs["Color"].default_value = (0.5, 0.55, 0.6, 1.0)
        return world

    sky = tree.nodes.new("ShaderNodeTexSky"); sky.location = (-100, 0)
    _set_physical_sky(sky)
    sky.sun_elevation = radians(float(sun_info["elevation_deg"]))
    # Nishita's zero-rotation reference is not true north, and the sign runs the
    # other way from a compass bearing. Calibrate the residual once against a
    # real hard shadow and leave sky_rotation_offset alone afterwards.
    sky.sun_rotation = -radians(float(sun_info["azimuth_deg"])) + radians(props.sky_rotation_offset)
    sky.sun_disc = False                     # the SUN lamp already provides it
    sky.altitude = max(0.0, float((solve.get("geo") or {}).get("altitude_m") or 0.0))
    tree.links.new(sky.outputs["Color"], background.inputs["Color"])
    return world


# ---------------------------------------------------------------------------
# render + compositor
# ---------------------------------------------------------------------------

def _compositor_tree(scene):
    """Get an empty compositor node tree, and its output node.

    Blender 5.0 rebuilt the scene compositor: `Scene.node_tree` became
    `Scene.compositing_node_group`, a real node-group datablock, and
    CompositorNodeComposite was removed in favour of a Group Output fed by a
    declared interface socket. Both spellings are supported here because 4.2 is
    the stated LTS floor and 5.x is what is actually installed.

    Returns (tree, output_node, output_socket_name).
    """
    if hasattr(scene, "compositing_node_group"):                  # Blender 5.x
        tree = scene.compositing_node_group
        if tree is None:
            tree = bpy.data.node_groups.new("Photo3D_Compositor", "CompositorNodeTree")
            scene.compositing_node_group = tree
        tree.nodes.clear()
        if not any(item.name == "Image" for item in tree.interface.items_tree):
            tree.interface.new_socket(name="Image", in_out="OUTPUT",
                                      socket_type="NodeSocketColor")
        output = tree.nodes.new("NodeGroupOutput")
        return tree, output, "Image"

    scene.use_nodes = True                                        # Blender 4.x
    tree = scene.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("CompositorNodeComposite")
    return tree, output, "Image"


def _alpha_over_sockets(node):
    """(background, foreground) sockets, whatever this version calls them.

    4.x exposes them as two identically named "Image" sockets at indices 1 and
    2; 5.x renamed them to Background and Foreground. Getting the order wrong
    puts the photograph on top of the CG, which renders as an untouched plate
    and reads as "nothing rendered".
    """
    names = {socket.name for socket in node.inputs}
    if {"Background", "Foreground"} <= names:
        return node.inputs["Background"], node.inputs["Foreground"]
    return node.inputs[1], node.inputs[2]


def _set_scale_to_render_size(scale_node) -> bool:
    """Make a compositor Scale node fit the render, on 4.x or 5.x.

    This node is not cosmetic. An Image node emits at the IMAGE's resolution,
    not the render's, so without it the compositor pastes an 8064 px plate 1:1
    into a half-size render and you get a centre crop — a 2x zoom into the
    photograph, with the CG apparently missing. It looks correct only at
    Render % = 100, which is exactly the setting nobody uses while iterating,
    so the bug hides until final output.

    Blender 5 moved the mode from a `space` enum property onto a menu input
    socket whose values are title-case strings.
    """
    if hasattr(scale_node, "space"):                     # Blender 4.x
        scale_node.space = "RENDER_SIZE"
        return True
    socket = scale_node.inputs.get("Type")               # Blender 5.x
    if socket is not None:
        try:
            socket.default_value = "Render Size"
        except TypeError:
            return False
        frame = scale_node.inputs.get("Frame Type")
        if frame is not None:
            # The plate and the render share an aspect ratio by construction,
            # so Stretch and Fit agree; Stretch is the safer of the two if a
            # rounding difference ever makes them disagree by a pixel.
            frame.default_value = "Stretch"
        return True
    return False


def setup_render(scene, plate_image, props):
    scene.render.engine = "CYCLES"
    try:
        scene.cycles.device = "GPU"
    except (AttributeError, TypeError):
        pass
    scene.render.film_transparent = True
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.samples = 512               # 512+ is also the fix for Minecraft
    scene.cycles.use_denoising = True        # texture shimmer at distance
    scene.view_settings.view_transform = props.view_transform

    tree, output, output_socket = _compositor_tree(scene)
    output.location = (700, 0)
    layers = tree.nodes.new("CompositorNodeRLayers"); layers.location = (0, 200)
    plate = tree.nodes.new("CompositorNodeImage"); plate.location = (-260, -200)
    plate.image = plate_image
    over = tree.nodes.new("CompositorNodeAlphaOver"); over.location = (460, 0)

    scale = tree.nodes.new("CompositorNodeScale"); scale.location = (60, -200)
    _set_scale_to_render_size(scale)
    tree.links.new(plate.outputs["Image"], scale.inputs["Image"])

    background, foreground = _alpha_over_sockets(over)
    tree.links.new(scale.outputs["Image"], background)
    tree.links.new(layers.outputs["Image"], foreground)
    tree.links.new(over.outputs["Image"], output.inputs[output_socket])

    viewer = tree.nodes.new("CompositorNodeViewer"); viewer.location = (560, -180)
    tree.links.new(over.outputs["Image"], viewer.inputs["Image"])


def ensure_rigidbody_world(context):
    if context.scene.rigidbody_world is None:
        with context.temp_override(scene=context.scene):
            bpy.ops.rigidbody.world_add()


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------

def _clear_previous(scene):
    """Remove the objects a previous solve made, so re-solving is idempotent."""
    for name in ("Photo3D_Cam", "Photo3D_Proxy", "Photo3D_Bounce", "Photo3D_Sun",
                 "Photo3D_Gobo", "Photo3D_GoboCam", "Photo3D_Ground"):
        existing = bpy.data.objects.get(name)
        if existing is not None:
            bpy.data.objects.remove(existing, do_unlink=True)


class PHOTO3D_OT_check_server(bpy.types.Operator):
    """Ask the daemon whether it is alive and what it has loaded"""
    bl_idname = "photo3d.check_server"
    bl_label = "Check Solver"

    def execute(self, context):
        try:
            info = client.health()
        except (client.SolverUnreachable, client.SolverRefused) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        resident = ", ".join(info.get("models_resident") or []) or "none yet"
        self.report({"INFO"}, f"solver {info.get('version')} on {info.get('device')}; "
                              f"models resident: {resident}")
        return {"FINISHED"}


class PHOTO3D_OT_solve(bpy.types.Operator):
    """Solve the photo and build the scene: camera, proxy, lighting, compositor"""
    bl_idname = "photo3d.solve"
    bl_label = "Solve Photo"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene.photo3d.image_path)

    def execute(self, context):
        scene = context.scene
        props = scene.photo3d

        payload = {
            "image_path": bpy.path.abspath(props.image_path),
            "refine": props.refine,
            "depth_long_edge": props.depth_long_edge,
            "assumed_eye_height": props.fallback_height,
            "far_clamp_m": props.max_depth,
            "focal_convention": props.focal_convention,
            "depth_scale": props.depth_scale,
        }
        if props.focal_override > 0.0:
            payload["focal_override"] = props.focal_override

        try:
            solve = client.solve(payload)
        except (client.SolverUnreachable, client.SolverRefused) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        _clear_previous(scene)
        camera, plate = setup_camera(scene, solve, props)

        mesh = None
        if solve.get("depth_npy"):
            try:
                mesh = proxy.build_proxy(context, solve, props, camera, plate)
            except Exception as exc:                              # noqa: BLE001
                self.report({"ERROR"}, f"camera solved but the proxy failed: {exc}")

        setup_sun(scene, solve, props)
        setup_sky(scene, solve, props)
        setup_render(scene, plate, props)
        if mesh is not None:
            ensure_rigidbody_world(context)

        self._record(props, solve)
        for warning in solve.get("warnings", []):
            self.report({"WARNING"}, warning)
        self.report({"INFO"}, props.solved_summary)
        return {"FINISHED"}

    @staticmethod
    def _record(props, solve: dict):
        """Keep the measurements so the panel can show them. A scene you cannot
        interrogate is a scene you cannot calibrate."""
        props.solved = True
        props.solved_focal = float(solve["intrinsics"]["focal_35mm"])
        props.solved_pitch = float(solve.get("pitch_deg") or 0.0)
        props.solved_roll = float(solve.get("roll_deg") or 0.0)
        props.solved_heading = float(solve.get("heading_deg") or 0.0)
        props.solved_height = float(solve.get("camera_height_m") or 0.0)
        props.solved_confidence = float(solve.get("ground_confidence") or 0.0)
        props.solved_focal = float(solve["intrinsics"]["focal_35mm"])
        props.solved_sun_azimuth = float((solve.get("sun") or {}).get("azimuth_deg") or 0.0)
        props.solved_sun_elevation = float((solve.get("sun") or {}).get("elevation_deg") or 0.0)
        props.solved_orientation_source = (
            "accelerometer" if solve.get("gravity_camera") else "fallback pitch")
        props.solved_warnings = " | ".join(solve.get("warnings", []))
        props.plate_png = solve.get("plate_png") or ""
        props.lighting_exr = solve.get("lighting_exr") or ""
        props.depth_npy = solve.get("depth_npy") or ""
        props.solved_summary = (
            f"{solve['intrinsics']['focal_35mm']:.0f}mm eq, "
            f"pitch {props.solved_pitch:+.1f}deg, roll {props.solved_roll:+.1f}deg, "
            f"h={props.solved_height:.2f}m, orientation from "
            f"{props.solved_orientation_source}")


class PHOTO3D_OT_solve_metadata_only(bpy.types.Operator):
    """Solve the camera from EXIF alone — no depth model, no proxy mesh"""
    bl_idname = "photo3d.solve_metadata"
    bl_label = "Solve Camera Only"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene.photo3d.image_path)

    def execute(self, context):
        scene = context.scene
        props = scene.photo3d
        try:
            solve = client.solve({"image_path": bpy.path.abspath(props.image_path),
                                  "skip_depth": True,
                                  "focal_convention": props.focal_convention},
                                 timeout=60.0)
        except (client.SolverUnreachable, client.SolverRefused) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        _clear_previous(scene)
        _, plate = setup_camera(scene, solve, props)
        setup_sun(scene, solve, props)
        setup_sky(scene, solve, props)
        setup_render(scene, plate, props)
        PHOTO3D_OT_solve._record(props, solve)
        for warning in solve.get("warnings", []):
            self.report({"WARNING"}, warning)
        self.report({"INFO"}, props.solved_summary)
        return {"FINISHED"}


class PHOTO3D_OT_align_view(bpy.types.Operator):
    """Look through the solved camera — the fastest way to judge the solve"""
    bl_idname = "photo3d.align_view"
    bl_label = "Look Through Camera"

    def execute(self, context):
        camera = bpy.data.objects.get("Photo3D_Cam")
        if camera is None:
            self.report({"ERROR"}, "no Photo3D_Cam in the scene; solve first")
            return {"CANCELLED"}
        context.scene.camera = camera
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.region_3d.view_perspective = "CAMERA"
        return {"FINISHED"}


CLASSES = (PHOTO3D_OT_check_server, PHOTO3D_OT_solve, PHOTO3D_OT_solve_metadata_only,
           PHOTO3D_OT_align_view)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
