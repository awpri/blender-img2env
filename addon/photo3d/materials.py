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

import bpy
from bpy.props import EnumProperty

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

    Transmissive regions have to be visible to the camera, or a CG object
    behind them is seen directly rather than through them — which is the whole
    reason for marking the glass in the first place. They also stop being
    shadow catchers, because a shadow catcher is a matte and a matte cannot
    refract.
    """
    transmissive = kind in {"GLASS", "WATER"}
    obj.is_shadow_catcher = not transmissive
    obj.visible_camera = True
    obj.visible_diffuse = True
    obj.visible_glossy = True
    obj.visible_transmission = True
    obj.visible_shadow = True
    obj.display_type = "TEXTURED" if transmissive else "WIRE"


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
                    + ("It is camera-visible and refracts, so CG behind it is seen "
                       "through it" if self.kind in {"GLASS", "WATER"} else
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


CLASSES = (PHOTO3D_OT_split_material_region, PHOTO3D_OT_select_proxy_for_editing)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
