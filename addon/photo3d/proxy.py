"""Depth map -> proxy geometry, the proxy shader, and the collider.

Monocular depth gives a bas-relief, not a scene: every surface facing away from
the camera is missing. What that buys and what it costs:

  occlusion            works well — the visible surface is what needs to occlude
  ground collision     works well — a metrically-scaled floor
  shadows onto plate   works well where the geometry is roughly right
  passing behind things works, but there is no back of the boulder to bounce off
  shadow onto a tree   unreliable; the tree is a flat-backed shell

The array maths lives in coords.proxy_arrays() and is unit-tested. This module
is the Blender half: making a mesh object out of it, giving it the shader that
is invisible to the eye but present to every other ray, and making it collide.
"""

from __future__ import annotations

import numpy as np

import bpy
from mathutils import Vector

from . import coords


def _fill_mesh(mesh, verts: np.ndarray, quads: np.ndarray):
    """Load vertex and face arrays into a mesh.

    from_pydata is the guaranteed-portable route but it walks Python lists;
    at the default 640-vertex long edge that is ~400k quads and it is slow
    enough to feel like a hang. The foreach_set path does the same work through
    the buffer protocol. Blender has moved the polygon storage internals more
    than once, so it is attempted first and quietly falls back.
    """
    n_verts, n_faces = len(verts), len(quads)
    try:
        mesh.vertices.add(n_verts)
        mesh.vertices.foreach_set("co", verts.astype(np.float32).ravel())
        mesh.loops.add(n_faces * 4)
        mesh.loops.foreach_set("vertex_index", quads.astype(np.int32).ravel())
        mesh.polygons.add(n_faces)
        mesh.polygons.foreach_set("loop_start",
                                  np.arange(0, n_faces * 4, 4, dtype=np.int32))
        mesh.update(calc_edges=True)
    except Exception:                                             # noqa: BLE001
        mesh.clear_geometry()
        mesh.from_pydata(verts.tolist(), [], quads.tolist())
        mesh.update(calc_edges=True)
    mesh.validate(verbose=False)


def build_proxy_mesh(solve: dict, props, camera):
    depth = np.load(solve["depth_npy"]).astype(np.float32)
    verts, quads = coords.proxy_arrays(
        depth, solve["intrinsics"],
        mesh_long_edge=props.mesh_long_edge,
        edge_threshold=props.edge_threshold,
        max_depth=props.max_depth)

    mesh = bpy.data.meshes.new("Photo3D_Proxy")
    _fill_mesh(mesh, verts, quads)

    obj = bpy.data.objects.new("Photo3D_Proxy", mesh)
    bpy.context.scene.collection.objects.link(obj)
    # Vertices come out of proxy_arrays in Blender CAMERA space, so parking the
    # object on the camera's matrix aligns it exactly, by construction, with no
    # fitting step that could drift.
    obj.matrix_world = camera.matrix_world
    return obj


def make_proxy_material(plate_image):
    """The plate, reprojected onto the proxy from the camera's own viewpoint.

        Texture Coordinate --[Window]--> Image Texture (plate, Cubic)
                                              |
                                              v Color
                                       Principled BSDF --> Output
                                        Roughness 0.55

    Window coordinates are the whole trick. They are screen-space, so the plate
    reprojects onto the proxy from exactly the render camera's viewpoint, which
    is what makes a polished Minecraft block reflect the actual gravel it is
    standing on — real colour, real texture — instead of a grey approximation.

    WHY THERE IS NO Is-Camera-Ray/Transparent MIX HERE ANY MORE
        The original design routed camera rays to a Transparent BSDF so the eye
        would see through the proxy to the plate. That is exactly what Cycles'
        shadow catcher already does, and doing both breaks it: a camera ray
        that passes straight through never hits the catcher, so Cycles has no
        surface on which to compute the shadow ratio, and CG objects cast no
        shadow at all.

        Measured, same scene, only the material differing:
            Is Camera Ray -> Transparent :   0 shadow pixels
            plain Principled             : 263 shadow pixels, mean alpha 0.79

        So the object is made invisible-but-present by `is_shadow_catcher` plus
        `film_transparent`, not by the shader. Glossy and diffuse rays still hit
        the photo-textured surface exactly as before — the chrome-sphere check
        in tools/render_checks.py holds either way.

    Cycles only. EEVEE Next will not do screen-space reflections off geometry
    it is not rendering.
    """
    mat = bpy.data.materials.new("Photo3D_ProxyMat")
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()

    output = tree.nodes.new("ShaderNodeOutputMaterial"); output.location = (400, 0)
    bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (140, 0)
    texture = tree.nodes.new("ShaderNodeTexImage"); texture.location = (-200, 0)
    coord = tree.nodes.new("ShaderNodeTexCoord"); coord.location = (-440, 0)

    texture.image = plate_image
    texture.interpolation = "Cubic"          # the PLATE wants smoothing
    texture.extension = "EXTEND"
    bsdf.inputs["Roughness"].default_value = 0.55
    bsdf.inputs["Metallic"].default_value = 0.0

    tree.links.new(coord.outputs["Window"], texture.inputs["Vector"])
    tree.links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
    tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return mat


def make_shadow_catcher(obj, props):
    """Matte to the camera, solid to everything else."""
    obj.is_shadow_catcher = True
    obj.visible_camera = True
    obj.visible_diffuse = True
    obj.visible_glossy = True
    obj.visible_transmission = True
    # See props._update_proxy_blocks_light: off by default, because the plate
    # already contains the scene's own shadows and a jagged mesh self-shadows.
    obj.visible_shadow = props.proxy_blocks_light
    obj.display_type = "WIRE"

    if props.collision_smooth > 0.0:
        modifier = obj.modifiers.new("Photo3D_Smooth", "SMOOTH")
        modifier.factor = props.collision_smooth
        modifier.iterations = 2


def add_passive_collider(context, obj):
    """Static collider, so blocks and mobs land on the real terrain.

    Rigid bodies can only be created through an operator, and the operator
    reads the active object out of context — which is not reliably this one
    when the add-on is driven from a panel. temp_override makes it explicit.
    """
    with context.temp_override(object=obj, active_object=obj, selected_objects=[obj]):
        bpy.ops.rigidbody.object_add(type="PASSIVE")
    obj.rigid_body.collision_shape = "MESH"
    obj.rigid_body.mesh_source = "FINAL"
    obj.rigid_body.friction = 0.9
    obj.rigid_body.restitution = 0.05


def build_proxy(context, solve: dict, props, camera, plate_image):
    """Everything at once: mesh, shader, ray visibility, collider."""
    obj = build_proxy_mesh(solve, props, camera)
    obj.data.materials.append(make_proxy_material(plate_image))
    make_shadow_catcher(obj, props)
    add_passive_collider(context, obj)
    return obj


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------

class PHOTO3D_OT_rebuild_proxy(bpy.types.Operator):
    """Rebuild the proxy mesh from the cached depth, without re-running depth"""
    bl_idname = "photo3d.rebuild_proxy"
    bl_label = "Rebuild Proxy"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        props = context.scene.photo3d
        return bool(props.depth_npy) and bpy.data.objects.get("Photo3D_Cam") is not None

    def execute(self, context):
        props = context.scene.photo3d
        camera = bpy.data.objects.get("Photo3D_Cam")
        plate = bpy.data.images.get(bpy.path.basename(props.plate_png))
        if plate is None and props.plate_png:
            plate = bpy.data.images.load(props.plate_png, check_existing=True)
        if plate is None:
            self.report({"ERROR"}, "no plate image; re-solve")
            return {"CANCELLED"}

        for name in ("Photo3D_Proxy", "Photo3D_Bounce"):
            existing = bpy.data.objects.get(name)
            if existing is not None:
                bpy.data.objects.remove(existing, do_unlink=True)

        # The cached solve response is not kept, so rebuild the slice of it the
        # mesh builder needs from the scene's own record.
        camera_data = camera.data
        long_edge = max(context.scene.render.resolution_x, context.scene.render.resolution_y)
        f_px = camera_data.lens / camera_data.sensor_width * long_edge
        stand_in = {
            "depth_npy": props.depth_npy,
            "intrinsics": {
                "width": context.scene.render.resolution_x,
                "height": context.scene.render.resolution_y,
                "fx": f_px, "fy": f_px,
                "cx": context.scene.render.resolution_x / 2.0,
                "cy": context.scene.render.resolution_y / 2.0,
            },
        }
        obj = build_proxy(context, stand_in, props, camera, plate)
        self.report({"INFO"}, f"proxy rebuilt: {len(obj.data.vertices)} verts, "
                              f"{len(obj.data.polygons)} faces")
        return {"FINISHED"}


class PHOTO3D_OT_add_ground_plane(bpy.types.Operator):
    """A true level ground plane from measured gravity — no depth model involved"""
    bl_idname = "photo3d.add_ground_plane"
    bl_label = "Add Ground Plane"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bpy.data.objects.get("Photo3D_Cam") is not None

    def execute(self, context):
        # Why this exists, and why it is exact where the depth mesh is not:
        #
        # The camera rotation was built so that world +Z is true up, because the
        # accelerometer measured gravity. So the plane z = 0 is genuinely level
        # — not level according to a depth model, level according to hardware.
        # The ONLY uncertain number is how far below the lens it sits, and that
        # is one value the user can drag.
        #
        # On scenes where monocular metric depth falls apart (ultra-wide, huge
        # depth range) this is the reliable way to place and land objects: the
        # depth mesh keeps doing occlusion and shadow-catching, where relative
        # ordering is all that matters, and the plane does the standing-on.
        props = context.scene.photo3d
        existing = bpy.data.objects.get("Photo3D_Ground")
        if existing is not None:
            bpy.data.objects.remove(existing, do_unlink=True)

        bpy.ops.mesh.primitive_plane_add(size=props.ground_plane_size,
                                         location=(0.0, 0.0, 0.0))
        plane = context.active_object
        plane.name = "Photo3D_Ground"

        plate = None
        proxy_obj = bpy.data.objects.get("Photo3D_Proxy")
        if proxy_obj is not None:
            for slot in proxy_obj.material_slots:
                if slot.material:
                    plane.data.materials.append(slot.material)
                    plate = slot.material
                    break
        if plate is None and props.plate_png:
            image = bpy.data.images.load(props.plate_png, check_existing=True)
            plane.data.materials.append(make_proxy_material(image))

        plane.is_shadow_catcher = True
        plane.visible_camera = True
        plane.visible_shadow = True
        plane.display_type = "WIRE"
        add_passive_collider(context, plane)

        camera = bpy.data.objects.get("Photo3D_Cam")
        height = camera.matrix_world.translation.z if camera else 0.0
        self.report({"INFO"},
                    f"level ground at z=0, camera {height:.2f} m above it. Move the "
                    "plane in Z to set the scale — its orientation is measured, "
                    "only the height is a choice")
        return {"FINISHED"}


class PHOTO3D_OT_crispify(bpy.types.Operator):
    """Force every image texture on the selected objects to nearest-neighbour"""
    bl_idname = "photo3d.crispify"
    bl_label = "Crispify Textures"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        # Cycles does no mipmapping in Closest mode, so a 16x16 block texture
        # stays 16 hard pixels at any scale. The trade is minification aliasing
        # on distant blocks — fix that with sample count, not by reintroducing
        # filtering.
        count = 0
        skipped = 0
        for obj in context.selected_objects:
            for slot in obj.material_slots:
                material = slot.material
                if not material or not material.use_nodes:
                    continue
                # Our own materials carry the photographic plate, which wants
                # smoothing rather than hard steps. Identify them by material
                # name: matching on the image path instead compares two empty
                # strings on a freshly generated texture and skips everything.
                is_ours = material.name.startswith("Photo3D_")
                for node in material.node_tree.nodes:
                    if node.type != "TEX_IMAGE" or node.image is None:
                        continue
                    if is_ours:
                        skipped += 1
                        continue
                    node.interpolation = "Closest"
                    node.extension = "EXTEND"
                    count += 1
        message = f"{count} texture node(s) set to Closest"
        if skipped:
            message += f"; left {skipped} plate texture(s) on Cubic"
        self.report({"INFO"}, message)
        return {"FINISHED"}


class PHOTO3D_OT_drop_test(bpy.types.Operator):
    """Drop a 1m cube from 3m above the camera's ground — the M3 sanity check"""
    bl_idname = "photo3d.drop_test"
    bl_label = "Add Drop Test Cube"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if bpy.data.objects.get("Photo3D_Proxy") is None:
            self.report({"ERROR"}, "no proxy to land on; solve first")
            return {"CANCELLED"}

        # Drop it a few metres in front of the lens rather than at the origin,
        # so it lands somewhere the camera can actually see it.
        camera = bpy.data.objects.get("Photo3D_Cam")
        location = (0.0, 0.0, 3.0)
        if camera is not None:
            forward = camera.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
            ahead = camera.matrix_world.translation + forward * 4.0
            location = (ahead.x, ahead.y, ahead.z + 3.0)

        bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
        cube = context.active_object
        cube.name = "Photo3D_DropTest"
        with context.temp_override(object=cube, active_object=cube, selected_objects=[cube]):
            bpy.ops.rigidbody.object_add(type="ACTIVE")
        self.report({"INFO"}, "cube added — play the timeline; it should land on "
                              "the terrain at plausible scale")
        return {"FINISHED"}


CLASSES = (PHOTO3D_OT_rebuild_proxy, PHOTO3D_OT_add_ground_plane, PHOTO3D_OT_crispify,
           PHOTO3D_OT_drop_test)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
