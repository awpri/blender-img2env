"""Give parts of the proxy real materials.

The depth mesh arrives as one surface wearing one shader, which is right for
ground and rock and wrong for everything that is not opaque and diffuse. A
glass shelter modelled as matte grey will not transmit a CG object standing
behind it, and a puddle will not reflect one standing beside it.

Estimating that automatically from a single photograph is not a solved problem
— glass in particular is defined by what is *behind* it, which monocular depth
has no access to. So the region is chosen by hand here, which is both reliable
and immediate: you know which surface is glass.

The split is the reusable part. A segmentation model would produce exactly the
same thing — a set of faces to peel off and re-shade — so when SAM 2 lands it
feeds this code rather than replacing it.
"""

from __future__ import annotations

import numpy as np

import bpy
from bpy.props import EnumProperty
from mathutils import Vector

#: (id, label, tooltip). The presets are starting points, not answers; every
#: one of them is a surface you will want to tune once you see it rendered.
PRESETS = [
    ("GLASS", "Glass", "Transmissive, IOR 1.45. Shelters, windows, screens"),
    ("WATER", "Water", "Transmissive, IOR 1.33, slightly rough. Puddles, lakes"),
    ("METAL", "Metal", "Fully metallic, low roughness. Rails, poles, vehicles"),
    ("MIRROR", "Polished", "Sharp specular. Shopfronts, still water"),
    ("MATTE", "Matte", "Plain diffuse. Undo a region back to ordinary ground"),
]


def build_material(kind: str, plate_image=None):
    """A Principled BSDF configured for `kind`.

    Where a plate is supplied it drives base colour through Window coordinates,
    the same reprojection the proxy uses, so a tinted surface keeps the real
    colour of whatever it is made of.
    """
    mat = bpy.data.materials.new(f"Photo3D_{kind.title()}")
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()

    output = tree.nodes.new("ShaderNodeOutputMaterial"); output.location = (400, 0)
    bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (140, 0)
    tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    def set_input(name, value):
        if name in bsdf.inputs:
            bsdf.inputs[name].default_value = value

    if kind == "GLASS":
        set_input("Roughness", 0.0)
        set_input("IOR", 1.45)
        set_input("Transmission Weight", 1.0)
        set_input("Base Color", (1.0, 1.0, 1.0, 1.0))
        mat.use_backface_culling = False
    elif kind == "WATER":
        set_input("Roughness", 0.05)
        set_input("IOR", 1.33)
        set_input("Transmission Weight", 1.0)
        set_input("Base Color", (0.85, 0.93, 0.95, 1.0))
    elif kind == "METAL":
        set_input("Metallic", 1.0)
        set_input("Roughness", 0.25)
    elif kind == "MIRROR":
        set_input("Metallic", 1.0)
        set_input("Roughness", 0.02)
    else:                                                # MATTE
        set_input("Roughness", 0.55)
        set_input("Metallic", 0.0)

    if plate_image is not None and kind in {"METAL", "MATTE"}:
        # Metals and matte surfaces take their colour from the photograph;
        # glass and water are defined by what passes through them, so tinting
        # them with the plate would double-count the scene behind.
        texture = tree.nodes.new("ShaderNodeTexImage"); texture.location = (-200, 0)
        coord = tree.nodes.new("ShaderNodeTexCoord"); coord.location = (-440, 0)
        texture.image = plate_image
        texture.interpolation = "Cubic"
        texture.extension = "EXTEND"
        tree.links.new(coord.outputs["Window"], texture.inputs["Vector"])
        tree.links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def configure_visibility(obj, kind: str):
    """Ray visibility for a re-shaded region.

    Every region stays a camera-invisible shadow catcher, including the glass.
    That is deliberate and it is the opposite of the obvious choice.

    The real glass is ALREADY IN THE PHOTOGRAPH. Making the region visible to
    camera rays renders a second sheet of glass on top of the first, refracting
    a scene that is itself mostly the plate projected onto matte geometry — so
    the frame fills with smeared reflections of the station, which is exactly
    what it did.

    What marking it as glass still buys you is every OTHER ray: a CG object
    beside it gets a real specular reflection off it, and light transmits
    through it rather than being blocked.

    The cost, stated plainly: a CG object placed BEHIND the glass is seen
    directly rather than refracted through it, because the camera never hits
    the glass. Single-photograph compositing cannot have both — the plate
    already fixed what that surface looks like from this viewpoint.
    """
    transmissive = kind in {"GLASS", "WATER"}
    obj.is_shadow_catcher = True
    obj.visible_camera = True          # the shadow catcher makes it a matte
    obj.visible_diffuse = True
    obj.visible_glossy = True
    obj.visible_transmission = True
    # Glass does not block the sun, so it must not cast a shadow. Left on, a
    # window or shelter throws a hard silhouette across the platform that is
    # not in the photograph and never was.
    obj.visible_shadow = not transmissive
    obj.display_type = "WIRE"


class PHOTO3D_OT_split_material_region(bpy.types.Operator):
    """Split the selected faces off the proxy and give them a real material"""
    bl_idname = "photo3d.split_material_region"
    bl_label = "Assign Material to Selection"
    bl_options = {"REGISTER", "UNDO"}

    kind: EnumProperty(name="Material", items=PRESETS, default="GLASS")

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (obj is not None and obj.type == "MESH"
                and obj.name.startswith("Photo3D_") and context.mode == "EDIT_MESH")

    def execute(self, context):
        from . import radiance

        source = context.active_object
        before = set(bpy.data.objects.keys())

        # separate() acts on the selection and needs to run in edit mode; it
        # leaves the new object selected but not active, hence the diff.
        bpy.ops.mesh.separate(type="SELECTED")
        bpy.ops.object.mode_set(mode="OBJECT")

        new_names = [n for n in bpy.data.objects.keys() if n not in before]
        if not new_names:
            self.report({"ERROR"}, "nothing was selected, so nothing was split off")
            return {"CANCELLED"}

        region = bpy.data.objects[new_names[0]]
        index = sum(1 for o in bpy.data.objects if o.name.startswith("Photo3D_Region"))
        region.name = region.data.name = f"Photo3D_Region_{index:02d}_{self.kind.title()}"

        plate = radiance.find_plate(source)
        region.data.materials.clear()
        region.data.materials.append(build_material(self.kind, plate))
        configure_visibility(region, self.kind)

        context.view_layer.objects.active = region
        self.report({"INFO"},
                    f"{region.name}: {len(region.data.polygons)} faces. "
                    + ("CG beside it now gets a real reflection off it; the camera "
                       "still sees the photograph, because the glass is already in it"
                       if self.kind in {"GLASS", "WATER"} else
                       "It still catches shadows like the rest of the proxy"))
        return {"FINISHED"}


class PHOTO3D_OT_select_proxy_for_editing(bpy.types.Operator):
    """Enter face-select mode on the proxy, ready to mark a region"""
    bl_idname = "photo3d.edit_proxy"
    bl_label = "Edit Proxy Faces"

    @classmethod
    def poll(cls, context):
        return bpy.data.objects.get("Photo3D_Proxy") is not None

    def execute(self, context):
        proxy_obj = bpy.data.objects["Photo3D_Proxy"]
        # The proxy renders as a wireframe, which cannot be face-picked; and it
        # is usually hidden behind its own shadow-catcher invisibility, so put
        # it into a state where the region is actually clickable.
        proxy_obj.display_type = "TEXTURED"
        # mode_set polls for an active object, and there may not be one — a
        # freshly solved scene leaves nothing selected.
        if context.object is not None and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        proxy_obj.select_set(True)
        context.view_layer.objects.active = proxy_obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="FACE")
        bpy.ops.mesh.select_all(action="DESELECT")
        self.report({"INFO"}, "select the faces covering the surface, then pick a "
                              "material below. Hover and press L to grab a connected "
                              "patch, or box-select over the region")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# segmentation-driven splitting
# ---------------------------------------------------------------------------

def face_labels(obj, labels, intrinsics) -> np.ndarray:
    """Which label each face of `obj` falls in, by projecting its centroid.

    The proxy's local coordinates ARE Blender camera space — it was parked on
    the camera's matrix at build time — so projecting needs no world transform
    and no view matrix, just the intrinsics the solve already produced. That
    also means it keeps working after the object has been split, which the
    face-index bookkeeping otherwise would not.

    Blender camera space is x right, y up, -z forward; OpenCV is x right,
    y down, +z forward. Hence the sign flips below.
    """
    count = len(obj.data.polygons)
    centres = np.empty(count * 3, dtype=np.float32)
    obj.data.polygons.foreach_get("center", centres)
    centres = centres.reshape(-1, 3)

    forward = -centres[:, 2]
    valid = forward > 1e-6
    forward = np.where(valid, forward, 1.0)

    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    u = (cx + fx * centres[:, 0] / forward) / intrinsics["width"]
    v = (cy - fy * centres[:, 1] / forward) / intrinsics["height"]

    height, width = labels.shape
    cols = np.clip((u * width).astype(np.int32), 0, width - 1)
    rows = np.clip((v * height).astype(np.int32), 0, height - 1)
    out = labels[rows, cols]
    out[~valid] = 0
    return out


class PHOTO3D_OT_segment_proxy(bpy.types.Operator):
    """Split the proxy into separate objects using SAM 2 masks"""
    bl_idname = "photo3d.segment_proxy"
    bl_label = "Segment Into Regions"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return (bpy.data.objects.get("Photo3D_Proxy") is not None
                and bool(context.scene.photo3d.image_path))

    def execute(self, context):
        from . import client

        props = context.scene.photo3d
        try:
            result = client.segment({
                "image_path": bpy.path.abspath(props.image_path),
                "long_edge": props.segment_long_edge,
                "max_regions": props.segment_max_regions,
                "min_area_fraction": props.segment_min_area,
            })
        except (client.SolverUnreachable, client.SolverRefused) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        labels = np.load(result["labels_npy"]).astype(np.int32)
        proxy_obj = bpy.data.objects["Photo3D_Proxy"]
        scene = context.scene
        long_edge = max(scene.render.resolution_x, scene.render.resolution_y)
        camera = bpy.data.objects.get("Photo3D_Cam")
        f_px = (camera.data.lens / camera.data.sensor_width * long_edge) if camera else long_edge
        intrinsics = {"fx": f_px, "fy": f_px,
                      "cx": scene.render.resolution_x / 2.0,
                      "cy": scene.render.resolution_y / 2.0,
                      "width": scene.render.resolution_x,
                      "height": scene.render.resolution_y}

        wanted = sorted({int(r["label"]) for r in result["regions"]})
        if context.object is not None and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        created = 0
        for label in wanted:
            # Re-derive membership against the CURRENT mesh each time: every
            # separate() renumbers the faces that are left behind, so indices
            # computed once up front would drift after the first split.
            current = face_labels(proxy_obj, labels, intrinsics)
            selected = current == label
            if selected.sum() < 8:
                continue

            # Clear vertex and edge flags before setting faces. A freshly built
            # mesh has every vertex flagged selected, and entering Edit Mode
            # flushes vertex selection upward — so setting only the polygon
            # flags selects the entire mesh and the first region swallows
            # everything.
            mesh = proxy_obj.data
            mesh.vertices.foreach_set("select", np.zeros(len(mesh.vertices), np.int8))
            mesh.edges.foreach_set("select", np.zeros(len(mesh.edges), np.int8))
            mesh.polygons.foreach_set("select", selected.astype(np.int8))
            before = set(bpy.data.objects.keys())
            bpy.ops.object.select_all(action="DESELECT")
            proxy_obj.select_set(True)
            context.view_layer.objects.active = proxy_obj
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_mode(type="FACE")
            try:
                bpy.ops.mesh.separate(type="SELECTED")
            finally:
                bpy.ops.object.mode_set(mode="OBJECT")

            new_names = [n for n in bpy.data.objects.keys() if n not in before]
            if not new_names:
                continue
            region = bpy.data.objects[new_names[0]]
            region.name = region.data.name = f"Photo3D_Seg_{label:02d}"
            # They inherit the proxy's material and shadow-catcher behaviour, so
            # the scene renders identically until a material is assigned. The
            # split is the useful part; what each surface is made of is the
            # user's call, because SAM 2 is class-agnostic.
            region.is_shadow_catcher = proxy_obj.is_shadow_catcher
            region.display_type = "WIRE"
            created += 1

        context.view_layer.objects.active = proxy_obj
        self.report({"INFO"},
                    f"{created} regions split off ({result['note']}). Select one and "
                    "give it a material below; the preview is at "
                    + result["preview_png"])
        return {"FINISHED"}


def _selected_faces(obj):
    """Indices of selected polygons, read outside edit mode.

    Edit-mode selection lives in the BMesh and is only flushed back to the
    Mesh on leaving edit mode, so anything reading polygon.select has to step
    out first or it sees the state from the last time you did.
    """
    was_edit = obj.mode == "EDIT"
    if was_edit:
        bpy.ops.object.mode_set(mode="OBJECT")
    selected = [i for i, poly in enumerate(obj.data.polygons) if poly.select]
    if was_edit:
        bpy.ops.object.mode_set(mode="EDIT")
    return selected


class PHOTO3D_OT_light_from_selection(bpy.types.Operator):
    """Turn the selected faces into an area light of that size and colour"""
    bl_idname = "photo3d.light_from_selection"
    bl_label = "Make Light From Selection"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (obj is not None and obj.type == "MESH"
                and obj.name.startswith("Photo3D_"))

    def execute(self, context):
        """Build a light where a bright surface is in the photograph.

        The bounce proxy already turns the whole plate into an emitter, but it
        emits what the plate SAYS, and a display-referred plate clips a sunlit
        window to white — so the one surface most worth emitting is the one
        whose real brightness the photograph could not record. Marking it by
        hand and giving it a real wattage is not a workaround for that, it is
        the only way to put back a value the file never held.

        Position, orientation and size come from the faces. Colour comes from
        the plate, normalised so the hue survives but the (clipped) magnitude
        does not dictate the power — that is yours to set.
        """
        from . import radiance

        obj = context.active_object
        faces = _selected_faces(obj)
        if not faces:
            self.report({"ERROR"}, "select the faces covering the window or lamp first")
            return {"CANCELLED"}

        mesh = obj.data
        centres, normals, corners = [], [], []
        for index in faces:
            poly = mesh.polygons[index]
            centres.append(obj.matrix_world @ poly.center)
            normals.append((obj.matrix_world.to_quaternion() @ poly.normal).normalized())
            corners.extend(obj.matrix_world @ mesh.vertices[v].co for v in poly.vertices)

        centre = sum(centres, Vector()) / len(centres)
        normal = sum(normals, Vector()) / len(normals)
        if normal.length < 1e-6:
            normal = Vector((0.0, 0.0, -1.0))
        normal.normalize()

        # Size the light by the spread of its corners in the plane it lies in.
        right = normal.cross(Vector((0.0, 0.0, 1.0)))
        if right.length < 1e-4:
            right = Vector((1.0, 0.0, 0.0))
        right.normalize()
        up = normal.cross(right).normalized()
        offsets = [c - centre for c in corners]
        width = max((abs(o.dot(right)) for o in offsets), default=0.5) * 2.0
        height = max((abs(o.dot(up)) for o in offsets), default=0.5) * 2.0

        colour = (1.0, 1.0, 1.0)
        plate = radiance.find_plate(obj) or radiance.find_plate(
            bpy.data.objects.get("Photo3D_Proxy"))
        if plate is not None:
            pixels = radiance.image_to_array(plate, 256)
            scene = context.scene
            long_edge = max(scene.render.resolution_x, scene.render.resolution_y)
            camera = bpy.data.objects.get("Photo3D_Cam")
            f_px = (camera.data.lens / camera.data.sensor_width * long_edge) if camera else long_edge
            local = obj.matrix_world.inverted() @ centre
            forward = -local.z
            if forward > 1e-6:
                u = (scene.render.resolution_x / 2.0 + f_px * local.x / forward) / scene.render.resolution_x
                v = (scene.render.resolution_y / 2.0 - f_px * local.y / forward) / scene.render.resolution_y
                row = int(np.clip(v * pixels.shape[0], 0, pixels.shape[0] - 1))
                col = int(np.clip(u * pixels.shape[1], 0, pixels.shape[1] - 1))
                sample = pixels[max(0, row - 2):row + 3, max(0, col - 2):col + 3]
                if sample.size:
                    mean = sample.reshape(-1, 3).mean(axis=0)
                    peak = float(mean.max())
                    # Normalise: keep the hue, drop the magnitude. A clipped
                    # window reads as flat white, and its power is the one thing
                    # the photograph genuinely cannot tell us.
                    if peak > 1e-4:
                        colour = tuple(float(c / peak) for c in mean)

        existing = sum(1 for o in bpy.data.objects if o.name.startswith("Photo3D_Light"))
        data = bpy.data.lights.new(f"Photo3D_Light_{existing:02d}", type="AREA")
        data.shape = "RECTANGLE"
        data.size = max(0.05, width)
        data.size_y = max(0.05, height)
        data.color = colour
        # Radiance x area x pi is the honest starting point for a Lambertian
        # emitter, assuming the surface was about as bright as diffuse white.
        data.energy = max(1.0, width * height * np.pi * 30.0)

        light = bpy.data.objects.new(data.name, data)
        context.scene.collection.objects.link(light)
        light.location = centre + normal * 0.02      # just off the surface
        light.rotation_mode = "QUATERNION"
        # An area light emits along its local -Z.
        light.rotation_quaternion = Vector((0.0, 0.0, -1.0)).rotation_difference(normal)

        self.report({"INFO"},
                    f"{data.name}: {width:.2f} x {height:.2f} m, colour "
                    f"({colour[0]:.2f}, {colour[1]:.2f}, {colour[2]:.2f}), "
                    f"{data.energy:.0f} W. Aim and set the power by eye — the plate "
                    "clipped this surface, so its real brightness is not in the file")
        return {"FINISHED"}


CLASSES = (PHOTO3D_OT_split_material_region, PHOTO3D_OT_select_proxy_for_editing,
           PHOTO3D_OT_segment_proxy, PHOTO3D_OT_light_from_selection)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
