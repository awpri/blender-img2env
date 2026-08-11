bl_info = {
    "name": "Photo3D Radiance — image-derived lighting",
    "blender": (4, 2, 0),
    "location": "View3D > N-panel > Photo3D",
    "description": "Bounce proxy, sun gobo, and panorama HDRI. Requires photo3d_blender_addon.",
    "category": "3D View",
}

"""
Solar ephemeris gives you ONE thing: the direction and colour of the direct sun.
Everything else in a real photograph — green bounce off grass, warm bounce off a
stone wall, dappled shade under a larch, the entirety of any interior — has to
come from the image itself.

Three mechanisms here, in descending order of importance:

  1. BOUNCE PROXY  the plate-textured depth mesh becomes an emitter, so the real
                   scene lights the CG objects. This is the big one, it is
                   ground-truth-accurate at close range, and it is the only one
                   of the three that works for interiors.
  2. SUN GOBO      the shadow pattern already visible on the ground is baked
                   into a cucoloris in front of the sun, so CG objects fall into
                   the same dappled shade the real ground is sitting in.
  3. PANORAMA      an outpainted 360 environment fills in sky and off-frame
                   surroundings for distant/specular response.
"""

import os
from math import radians

import bpy
import numpy as np
from bpy.props import FloatProperty, IntProperty, StringProperty, BoolProperty
from mathutils import Vector


# ---------------------------------------------------------------------------
# 1. Bounce proxy — the photograph as a light source
# ---------------------------------------------------------------------------

def make_bounce_material(plate_image, strength, saturation):
    """
    Emission driven by the plate, reprojected through Window coords exactly as
    the shadow proxy is. A CG block one metre above grass receives real green
    bounce, with the real spatial falloff, because the emitting geometry IS the
    grass at its real distance.

    Colour handling: the plate is display-referred sRGB, so its values top out
    at 1.0 while real sunlit grass is many times brighter than that. `strength`
    is the scene-referred multiplier. See calibrate() below.
    """
    mat = bpy.data.materials.new("Photo3D_BounceMat")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial");  out.location = (600, 0)
    emit = nt.nodes.new("ShaderNodeEmission");       emit.location = (400, 0)
    hsv = nt.nodes.new("ShaderNodeHueSaturation");   hsv.location = (180, 0)
    tex = nt.nodes.new("ShaderNodeTexImage");        tex.location = (-100, 0)
    coord = nt.nodes.new("ShaderNodeTexCoord");      coord.location = (-340, 0)

    tex.image = plate_image
    tex.interpolation = "Cubic"
    tex.extension = "EXTEND"
    hsv.inputs["Saturation"].default_value = saturation
    emit.inputs["Strength"].default_value = strength

    nt.links.new(coord.outputs["Window"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], hsv.inputs["Color"])
    nt.links.new(hsv.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


class PHOTO3D_OT_bounce_proxy(bpy.types.Operator):
    """Duplicate the shadow proxy into a light-emitting twin"""
    bl_idname = "photo3d.bounce_proxy"
    bl_label = "Build Bounce Proxy"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        p = context.scene.photo3d_rad
        src = bpy.data.objects.get("Photo3D_Proxy")
        if src is None:
            self.report({"ERROR"}, "Solve a photo first — no Photo3D_Proxy in scene.")
            return {"CANCELLED"}

        old = bpy.data.objects.get("Photo3D_Bounce")
        if old:
            bpy.data.objects.remove(old, do_unlink=True)

        plate = None
        for slot in src.material_slots:
            for n in (slot.material.node_tree.nodes if slot.material else []):
                if n.type == "TEX_IMAGE" and n.image:
                    plate = n.image
        if plate is None:
            self.report({"ERROR"}, "Could not find the plate image on the proxy.")
            return {"CANCELLED"}

        bounce = src.copy()
        bounce.data = src.data.copy()
        bounce.name = bounce.data.name = "Photo3D_Bounce"
        context.scene.collection.objects.link(bounce)
        bounce.matrix_world = src.matrix_world

        bounce.data.materials.clear()
        bounce.data.materials.append(make_bounce_material(plate, p.bounce_strength,
                                                          p.bounce_saturation))

        # emits only; never seen, never casts, never collides
        bounce.is_shadow_catcher = False
        bounce.visible_camera = False
        bounce.visible_shadow = False
        bounce.visible_diffuse = True
        bounce.visible_glossy = True
        bounce.visible_transmission = True
        bounce.display_type = "BOUNDS"
        if bounce.rigid_body:
            context.view_layer.objects.active = bounce
            bpy.ops.rigidbody.object_remove()

        # the original stops contributing indirect light so nothing double-counts
        src.visible_diffuse = False
        src.visible_glossy = False
        src.visible_transmission = False

        self.report({"INFO"}, "Bounce proxy built; shadow proxy demoted to matte only.")
        return {"FINISHED"}


class PHOTO3D_OT_calibrate_bounce(bpy.types.Operator):
    """
    Estimate emission strength from the plate instead of guessing.

    Assumes the ground has a plausible diffuse albedo (grass ~0.18, snow ~0.75,
    asphalt ~0.10) and solves for the multiplier that makes the emitted radiance
    consistent with the sun+sky already in the scene.
    """
    bl_idname = "photo3d.calibrate_bounce"
    bl_label = "Estimate Strength"

    def execute(self, context):
        p = context.scene.photo3d_rad
        proxy = bpy.data.objects.get("Photo3D_Proxy")
        plate = None
        for slot in (proxy.material_slots if proxy else []):
            for n in (slot.material.node_tree.nodes if slot.material else []):
                if n.type == "TEX_IMAGE" and n.image:
                    plate = n.image
        if plate is None:
            self.report({"ERROR"}, "No plate found.")
            return {"CANCELLED"}

        img = plate.copy()
        img.scale(256, int(256 * plate.size[1] / plate.size[0]))
        px = np.array(img.pixels[:], dtype=np.float32).reshape(-1, 4)[:, :3]
        bpy.data.images.remove(img)

        lower = px[: len(px) // 2]                      # bottom half == ground
        median_lin = float(np.median(lower.mean(axis=1)))

        sun = bpy.data.lights.get("Photo3D_Sun")
        irradiance = (sun.energy if sun else 3.0) + p.sky_irradiance_guess
        target = irradiance * p.assumed_albedo / 3.14159
        p.bounce_strength = max(0.05, target / max(median_lin, 1e-4))

        bounce = bpy.data.objects.get("Photo3D_Bounce")
        if bounce and bounce.material_slots:
            for n in bounce.material_slots[0].material.node_tree.nodes:
                if n.type == "EMISSION":
                    n.inputs["Strength"].default_value = p.bounce_strength

        self.report({"INFO"}, f"strength ≈ {p.bounce_strength:.2f} "
                              f"(plate median {median_lin:.3f})")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# 2. Sun gobo — steal the dappled shade that is already in the photo
# ---------------------------------------------------------------------------

def extract_shadow_mask(plate, downsample, blur_px, contrast):
    """
    Local luminance divided by a heavily blurred copy of itself. Albedo varies
    slowly, illumination under foliage varies fast, so the ratio isolates
    shadow. Only valid over roughly uniform ground (grass, gravel, snow, road) —
    it will read a dark rock as shade, which is why the mask gets clamped.
    """
    img = plate.copy()
    w = downsample
    h = max(1, int(w * plate.size[1] / plate.size[0]))
    img.scale(w, h)
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    bpy.data.images.remove(img)

    lum = px[..., :3] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    k = max(3, int(blur_px) | 1)
    pad = k // 2
    padded = np.pad(lum, pad, mode="edge")
    kern = np.ones(k, dtype=np.float32) / k
    tmp = np.apply_along_axis(lambda m: np.convolve(m, kern, mode="valid"), 1, padded)
    base = np.apply_along_axis(lambda m: np.convolve(m, kern, mode="valid"), 0, tmp)

    ratio = lum / np.clip(base, 1e-4, None)
    mask = np.clip((ratio - 1.0) * contrast + 1.0, 0.15, 1.0)
    return mask


class PHOTO3D_OT_bake_gobo(bpy.types.Operator):
    """
    Project the extracted shade mask onto the ground proxy, then re-render it
    from an orthographic camera aligned with the sun. That gives the pattern in
    SUN space, which is the only space a cucoloris is meaningful in.
    """
    bl_idname = "photo3d.bake_gobo"
    bl_label = "Bake Sun Gobo"
    bl_options = {"REGISTER"}

    def execute(self, context):
        scene = context.scene
        p = scene.photo3d_rad
        proxy = bpy.data.objects.get("Photo3D_Proxy")
        sun = bpy.data.objects.get("Photo3D_Sun")
        if proxy is None or sun is None:
            self.report({"ERROR"}, "Need a solved scene with Photo3D_Proxy and Photo3D_Sun.")
            return {"CANCELLED"}

        plate = None
        for slot in proxy.material_slots:
            for n in (slot.material.node_tree.nodes if slot.material else []):
                if n.type == "TEX_IMAGE" and n.image:
                    plate = n.image
        mask = extract_shadow_mask(plate, p.mask_res, p.mask_blur, p.mask_contrast)
        h, w = mask.shape

        mask_img = bpy.data.images.new("Photo3D_ShadeMask", w, h, float_buffer=True)
        rgba = np.ones((h, w, 4), dtype=np.float32)
        rgba[..., :3] = mask[..., None]
        mask_img.pixels = rgba.ravel()

        # --- temporary emission material carrying the mask, Window-projected ---
        tmp_mat = bpy.data.materials.new("Photo3D_TmpMask")
        tmp_mat.use_nodes = True
        nt = tmp_mat.node_tree
        nt.nodes.clear()
        o = nt.nodes.new("ShaderNodeOutputMaterial")
        e = nt.nodes.new("ShaderNodeEmission")
        t = nt.nodes.new("ShaderNodeTexImage")
        c = nt.nodes.new("ShaderNodeTexCoord")
        t.image = mask_img
        t.extension = "EXTEND"
        t.image.colorspace_settings.name = "Non-Color"
        nt.links.new(c.outputs["Window"], t.inputs["Vector"])
        nt.links.new(t.outputs["Color"], e.inputs["Color"])
        nt.links.new(e.outputs["Emission"], o.inputs["Surface"])

        saved_mats = [s.material for s in proxy.material_slots]
        saved_flags = (proxy.is_shadow_catcher, proxy.visible_camera)
        for s in proxy.material_slots:
            s.material = tmp_mat
        proxy.is_shadow_catcher = False
        proxy.visible_camera = True

        # --- orthographic camera looking down the sun direction ---
        sun_dir = (sun.matrix_world.to_quaternion() @ Vector((0, 0, -1))).normalized()
        cam_data = bpy.data.cameras.new("Photo3D_GoboCam")
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = p.gobo_size
        gcam = bpy.data.objects.new("Photo3D_GoboCam", cam_data)
        scene.collection.objects.link(gcam)
        centre = proxy.matrix_world @ Vector(
            np.array([v.co for v in proxy.data.vertices]).mean(axis=0).tolist()
        )
        gcam.location = centre - sun_dir * p.gobo_distance
        gcam.rotation_mode = "QUATERNION"
        gcam.rotation_quaternion = Vector((0, 0, -1)).rotation_difference(sun_dir)

        saved = (scene.camera, scene.render.filepath, scene.render.resolution_x,
                 scene.render.resolution_y, scene.render.resolution_percentage,
                 scene.render.film_transparent, scene.use_nodes, scene.cycles.samples)
        out_path = os.path.join(bpy.app.tempdir, "photo3d_gobo.png")
        scene.camera = gcam
        scene.render.filepath = out_path
        scene.render.resolution_x = scene.render.resolution_y = p.gobo_res
        scene.render.resolution_percentage = 100
        scene.render.film_transparent = False
        scene.use_nodes = False
        scene.cycles.samples = 8
        bpy.ops.render.render(write_still=True)

        (scene.camera, scene.render.filepath, scene.render.resolution_x,
         scene.render.resolution_y, scene.render.resolution_percentage,
         scene.render.film_transparent, scene.use_nodes, scene.cycles.samples) = saved
        for s, m in zip(proxy.material_slots, saved_mats):
            s.material = m
        proxy.is_shadow_catcher, proxy.visible_camera = saved_flags
        bpy.data.objects.remove(gcam, do_unlink=True)

        gobo_img = bpy.data.images.load(out_path, check_existing=False)
        gobo_img.colorspace_settings.name = "Non-Color"
        self._build_gobo_plane(context, sun, sun_dir, centre, gobo_img, p)
        self.report({"INFO"}, "Gobo baked. Tweak Mask Contrast if shade reads too hard.")
        return {"FINISHED"}

    @staticmethod
    def _build_gobo_plane(context, sun, sun_dir, centre, gobo_img, p):
        old = bpy.data.objects.get("Photo3D_Gobo")
        if old:
            bpy.data.objects.remove(old, do_unlink=True)

        bpy.ops.mesh.primitive_plane_add(size=p.gobo_size)
        plane = context.active_object
        plane.name = "Photo3D_Gobo"
        plane.location = centre - sun_dir * (p.gobo_distance * 0.5)
        plane.rotation_mode = "QUATERNION"
        plane.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(-sun_dir)

        mat = bpy.data.materials.new("Photo3D_GoboMat")
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        o = nt.nodes.new("ShaderNodeOutputMaterial")
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        tex = nt.nodes.new("ShaderNodeTexImage")
        crd = nt.nodes.new("ShaderNodeTexCoord")
        tex.image = gobo_img
        tex.extension = "EXTEND"
        # Generated coords on the plane == the ortho frame we rendered from
        nt.links.new(crd.outputs["Generated"], tex.inputs["Vector"])
        nt.links.new(tex.outputs["Color"], tr.inputs["Color"])
        nt.links.new(tr.outputs["BSDF"], o.inputs["Surface"])
        plane.data.materials.append(mat)

        # exists only to tint shadow rays
        plane.visible_camera = False
        plane.visible_diffuse = False
        plane.visible_glossy = False
        plane.visible_transmission = False
        plane.visible_volume_scatter = False
        plane.visible_shadow = True
        plane.display_type = "WIRE"


# ---------------------------------------------------------------------------
# 3. Panorama HDRI for sky and off-frame surroundings
# ---------------------------------------------------------------------------

class PHOTO3D_OT_load_panorama(bpy.types.Operator):
    """Swap the Nishita sky for an outpainted equirectangular environment"""
    bl_idname = "photo3d.load_panorama"
    bl_label = "Load Panorama HDRI"

    def execute(self, context):
        p = context.scene.photo3d_rad
        path = bpy.path.abspath(p.panorama_path)
        if not os.path.exists(path):
            self.report({"ERROR"}, "Panorama file not found.")
            return {"CANCELLED"}

        world = context.scene.world
        world.use_nodes = True
        nt = world.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputWorld");  out.location = (500, 0)
        bg = nt.nodes.new("ShaderNodeBackground");    bg.location = (300, 0)
        env = nt.nodes.new("ShaderNodeTexEnvironment"); env.location = (40, 0)
        mp = nt.nodes.new("ShaderNodeMapping");       mp.location = (-200, 0)
        crd = nt.nodes.new("ShaderNodeTexCoord");     crd.location = (-420, 0)

        env.image = bpy.data.images.load(path, check_existing=True)
        bg.inputs["Strength"].default_value = p.panorama_strength
        # align the panorama's centre with the camera's compass heading
        mp.inputs["Rotation"].default_value[2] = radians(p.panorama_rotation)
        nt.links.new(crd.outputs["Generated"], mp.inputs["Vector"])
        nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
        nt.links.new(env.outputs["Color"], bg.inputs["Color"])
        nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# properties + UI
# ---------------------------------------------------------------------------

class Photo3DRadProps(bpy.types.PropertyGroup):
    bounce_strength: FloatProperty(name="Bounce strength", default=3.0, min=0.0, soft_max=50.0)
    bounce_saturation: FloatProperty(name="Bounce saturation", default=1.15, min=0.0, max=3.0)
    assumed_albedo: FloatProperty(name="Ground albedo", default=0.18, min=0.01, max=0.95,
                                  description="grass .18  asphalt .10  gravel .25  snow .75")
    sky_irradiance_guess: FloatProperty(name="Sky irradiance", default=1.0, min=0.0)

    mask_res: IntProperty(name="Mask res", default=512, min=128, max=2048)
    mask_blur: IntProperty(name="Mask blur (px)", default=41, min=5, max=201)
    mask_contrast: FloatProperty(name="Mask contrast", default=1.6, min=0.1, max=6.0)
    gobo_res: IntProperty(name="Gobo res", default=1024, min=256, max=4096)
    gobo_size: FloatProperty(name="Gobo size (m)", default=30.0, min=1.0)
    gobo_distance: FloatProperty(name="Gobo distance (m)", default=20.0, min=1.0)

    panorama_path: StringProperty(name="Panorama", subtype="FILE_PATH")
    panorama_strength: FloatProperty(name="Panorama strength", default=1.0, min=0.0)
    panorama_rotation: FloatProperty(name="Panorama rotation", default=0.0, min=-360, max=360)


class PHOTO3D_PT_radiance(bpy.types.Panel):
    bl_label = "Radiance"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Photo3D"

    def draw(self, context):
        p = context.scene.photo3d_rad
        lay = self.layout

        box = lay.box(); box.label(text="Bounce (works indoors too)", icon="LIGHT_AREA")
        box.prop(p, "bounce_strength")
        box.prop(p, "bounce_saturation")
        row = box.row(align=True)
        row.prop(p, "assumed_albedo")
        row.operator("photo3d.calibrate_bounce", text="Estimate")
        box.operator("photo3d.bounce_proxy", icon="OUTLINER_OB_LIGHT")

        box = lay.box(); box.label(text="Sun gobo (dappled shade)", icon="MOD_MASK")
        box.prop(p, "mask_blur")
        box.prop(p, "mask_contrast")
        box.prop(p, "gobo_size")
        box.operator("photo3d.bake_gobo", icon="RENDER_STILL")

        box = lay.box(); box.label(text="Panorama", icon="WORLD")
        box.prop(p, "panorama_path")
        box.prop(p, "panorama_strength")
        box.prop(p, "panorama_rotation")
        box.operator("photo3d.load_panorama")


CLASSES = (Photo3DRadProps, PHOTO3D_OT_bounce_proxy, PHOTO3D_OT_calibrate_bounce,
           PHOTO3D_OT_bake_gobo, PHOTO3D_OT_load_panorama, PHOTO3D_PT_radiance)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.photo3d_rad = bpy.props.PointerProperty(type=Photo3DRadProps)


def unregister():
    del bpy.types.Scene.photo3d_rad
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
