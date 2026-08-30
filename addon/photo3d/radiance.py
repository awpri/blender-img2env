"""Image-derived lighting: bounce proxy, sun gobo, panorama.

Solar ephemeris gives exactly one thing — the direction and colour of the
direct sun. Everything else in a real photograph (green bounce off grass, warm
bounce off a stone wall, dappled shade under a larch, the entirety of any
interior) has to come from the image itself.

Three mechanisms, in descending order of importance:

  1. BOUNCE PROXY  the plate-textured depth mesh becomes an emitter, so the
                   real scene lights the CG. Ground-truth-accurate at close
                   range, and the only one of the three that works indoors.
  2. SUN GOBO      the shadow pattern already printed on the ground is baked
                   into a cucoloris in front of the sun, so CG objects fall
                   into the same dappled shade the real ground sits in.
  3. PANORAMA      an outpainted 360 environment fills in sky and off-frame
                   surroundings for distant and specular response.
"""

from __future__ import annotations

import contextlib
import os
from math import radians

import numpy as np

import bpy
from mathutils import Vector

from . import client, imaging


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def find_plate(obj) -> bpy.types.Image | None:
    for slot in (obj.material_slots if obj else []):
        if not slot.material or not slot.material.use_nodes:
            continue
        for node in slot.material.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image is not None:
                return node.image
    return None


def emitter_image(props, fallback):
    """The image the bounce proxy should emit from.

    The linear EXR when there is one, the display PNG otherwise. This is the
    whole payoff of the DNG path: a display-referred plate clips everything
    brighter than diffuse white, and those clipped regions carry most of the
    light energy in the scene, so a bounce driven by the PNG systematically
    under-reads the sky and renders flat, grey CG. The EXR keeps the
    highlights — measured at 2.89 against a median of 0.21 on IMG_7263.

    Returns (image, is_linear) so the caller can tell the user which it got.
    """
    path = bpy.path.abspath(props.lighting_exr) if props.lighting_exr else ""
    if path and os.path.exists(path):
        image = bpy.data.images.load(path, check_existing=True)
        # EXR is scene-linear by definition; say so rather than trusting the
        # colour management guess, which reads the extension and can be wrong
        # if the file was renamed.
        image.colorspace_settings.name = "Linear Rec.709"
        return image, True
    return fallback, False


def image_to_array(image: bpy.types.Image, long_edge: int) -> np.ndarray:
    """Downsampled RGB float array from a Blender image.

    foreach_get rather than pixels[:] — the slice builds a Python list of
    floats, which at plate resolution is tens of millions of objects and long
    enough to look like a hang.
    """
    copy = image.copy()
    try:
        width = min(long_edge, image.size[0])
        height = max(1, int(round(width * image.size[1] / max(1, image.size[0]))))
        copy.scale(width, height)
        buffer = np.empty(width * height * 4, dtype=np.float32)
        copy.pixels.foreach_get(buffer)
    finally:
        bpy.data.images.remove(copy)
    # Blender stores images bottom-up; flip so row 0 is the top of the frame,
    # which is what "the lower half is the ground" assumes.
    return buffer.reshape(height, width, 4)[::-1, :, :3]


# ---------------------------------------------------------------------------
# 1. bounce proxy — the photograph as an emitter
# ---------------------------------------------------------------------------

def make_bounce_material(plate_image, strength: float, saturation: float):
    """Emission driven by the plate, reprojected through the same Window coords
    as the shadow proxy.

    A CG block one metre above grass receives real green bounce with the real
    spatial falloff and the real solid angle, because the emitting geometry IS
    the grass, at its measured distance. A red barn to camera-left throws warm
    light on the left face of a block. None of that needs an estimator.

    The plate is display-referred, so its values top out at 1.0 while real
    sunlit grass is many times brighter. `strength` is the scene-referred
    multiplier — see the Estimate operator, which solves for it instead of
    leaving it as a slider to guess at.
    """
    mat = bpy.data.materials.new("Photo3D_BounceMat")
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()

    output = tree.nodes.new("ShaderNodeOutputMaterial"); output.location = (600, 0)
    emission = tree.nodes.new("ShaderNodeEmission"); emission.location = (400, 0)
    hsv = tree.nodes.new("ShaderNodeHueSaturation"); hsv.location = (180, 0)
    texture = tree.nodes.new("ShaderNodeTexImage"); texture.location = (-100, 0)
    coord = tree.nodes.new("ShaderNodeTexCoord"); coord.location = (-340, 0)

    texture.image = plate_image
    texture.interpolation = "Cubic"
    texture.extension = "EXTEND"
    hsv.inputs["Saturation"].default_value = saturation
    emission.inputs["Strength"].default_value = strength

    tree.links.new(coord.outputs["Window"], texture.inputs["Vector"])
    tree.links.new(texture.outputs["Color"], hsv.inputs["Color"])
    tree.links.new(hsv.outputs["Color"], emission.inputs["Color"])
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def set_ray_visibility(obj, camera=False, diffuse=False, glossy=False,
                       transmission=False, shadow=False, shadow_catcher=False):
    obj.visible_camera = camera
    obj.visible_diffuse = diffuse
    obj.visible_glossy = glossy
    obj.visible_transmission = transmission
    obj.visible_shadow = shadow
    obj.is_shadow_catcher = shadow_catcher


class PHOTO3D_OT_bounce_proxy(bpy.types.Operator):
    """Duplicate the shadow proxy into a light-emitting twin"""
    bl_idname = "photo3d.bounce_proxy"
    bl_label = "Build Bounce Proxy"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.photo3d
        source = bpy.data.objects.get("Photo3D_Proxy")
        if source is None:
            self.report({"ERROR"}, "solve a photo first — no Photo3D_Proxy in the scene")
            return {"CANCELLED"}

        plate = find_plate(source)
        if plate is None:
            self.report({"ERROR"}, "could not find the plate image on the proxy")
            return {"CANCELLED"}

        existing = bpy.data.objects.get("Photo3D_Bounce")
        if existing is not None:
            bpy.data.objects.remove(existing, do_unlink=True)

        emitter, is_linear = emitter_image(props, plate)
        bounce = source.copy()
        bounce.data = source.data.copy()
        bounce.name = bounce.data.name = "Photo3D_Bounce"
        context.scene.collection.objects.link(bounce)
        bounce.matrix_world = source.matrix_world
        bounce.data.materials.clear()
        bounce.data.materials.append(
            make_bounce_material(emitter, props.bounce_strength, props.bounce_saturation))

        # The split that stops anything being counted twice:
        #
        #                  shadow proxy          bounce proxy
        #   camera         matte (catcher)       invisible
        #   diffuse/glossy OFF                   ON, emissive
        #   shadow         on                    off
        #   collision      yes                   no
        #   catcher        yes                   yes  <- see below
        #
        # The bounce proxy must ALSO be a shadow catcher, which reads as a
        # contradiction — it is invisible to the camera, so it can never catch
        # anything. The flag is not doing its usual job here; it is telling
        # Cycles which side of the shadow-catcher ratio this object belongs on.
        #
        # That pass is a multiplier: for each camera ray reaching a catcher it
        # divides the light arriving WITH the CG objects present by the light
        # arriving WITHOUT them. Anything Cycles counts as CG appears in the
        # numerator only. The bounce proxy is solid to diffuse and glossy rays,
        # so counted as CG it occludes sky from the shadow proxy standing
        # directly behind it and substitutes dimmer plate-derived emission. The
        # ratio falls below 1, and the compositor dutifully darkens a
        # photograph that was already that dark — the same double-counting that
        # this whole file is arranged to avoid, arriving by a back route.
        #
        # Marking it a catcher puts it on the non-CG side, so it is present in
        # both legs and cancels. Cycles keeps shadow catchers visible to
        # indirect rays, so it goes on lighting CG exactly as before. Measured
        # on a solved station scene, white sphere masked by its own alpha:
        #
        #                          plate pixels darkened   sphere brightness
        #   no bounce proxy                 0.99 %              0.5128
        #   bounce, plain                  14.43 %              0.5345
        #   bounce, catcher                 0.86 %              0.5348
        #
        # Both columns matter. Light linking was tried first and fails both:
        # excluding the catchers as receivers stops the emission without
        # stopping the occlusion, which took the darkening to 80 %.
        set_ray_visibility(bounce, camera=False, diffuse=True, glossy=True,
                           transmission=True, shadow=False, shadow_catcher=True)
        bounce.display_type = "BOUNDS"
        if bounce.rigid_body is not None:
            with context.temp_override(object=bounce, active_object=bounce,
                                       selected_objects=[bounce]):
                bpy.ops.rigidbody.object_remove()

        source.visible_diffuse = False
        source.visible_glossy = False
        source.visible_transmission = False

        origin = ("linear EXR lighting plate, highlights intact"
                  if is_linear else
                  "display-referred plate — highlights are clipped, so the sky "
                  "will under-read. Shoot ProRAW for the lighting plate")
        self.report({"INFO"}, f"bounce proxy built from the {origin}. It lights CG "
                              "without re-darkening the photograph")
        return {"FINISHED"}


class PHOTO3D_OT_toggle_bounce(bpy.types.Operator):
    """A/B the bounce proxy — the M5 double-counting check"""
    bl_idname = "photo3d.toggle_bounce"
    bl_label = "Toggle Bounce"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        bounce = bpy.data.objects.get("Photo3D_Bounce")
        source = bpy.data.objects.get("Photo3D_Proxy")
        if bounce is None:
            self.report({"ERROR"}, "no bounce proxy to toggle")
            return {"CANCELLED"}

        # Off means the bounce stops emitting AND the shadow proxy takes its
        # indirect duties back, so the two states are genuinely comparable
        # rather than one of them simply being darker.
        turning_on = not bounce.visible_diffuse
        set_ray_visibility(bounce, camera=False, diffuse=turning_on, glossy=turning_on,
                           transmission=turning_on, shadow=False, shadow_catcher=True)
        if source is not None:
            source.visible_diffuse = not turning_on
            source.visible_glossy = not turning_on
            source.visible_transmission = not turning_on

        self.report({"INFO"}, f"bounce proxy {'ON' if turning_on else 'OFF'} — render a "
                              "white sphere in both states and compare")
        return {"FINISHED"}


def measure_ground_irradiance(context) -> float:
    """Render a white probe on the ground and read the irradiance off it.

    The previous version multiplied the sky strength by a guessed constant for
    "how much irradiance a Nishita sky delivers". That guess was wrong enough
    that the answer depended almost entirely on whatever sky strength you had
    dialled in, which is precisely the knob it was supposed to be solving.

    So measure instead. A white Lambertian surface under irradiance E has
    radiance E/pi, so one tiny render of a matte white card gives E exactly —
    no sky model, no sun-angle term, and correct for whatever combination of
    lamps, sky and bounce happens to be in the scene.

    The proxy is left visible on purpose: it occludes, so a probe under a
    station canopy measures the shaded irradiance, which is what the plate's
    ground brightness there actually corresponds to.
    """
    scene = context.scene
    probe = None
    camera = None
    material = None
    saved = {
        "camera": scene.camera,
        "filepath": scene.render.filepath,
        "resolution_x": scene.render.resolution_x,
        "resolution_y": scene.render.resolution_y,
        "percentage": scene.render.resolution_percentage,
        "film_transparent": scene.render.film_transparent,
        "samples": scene.cycles.samples,
        "view_transform": scene.view_settings.view_transform,
        "look": scene.view_settings.look,
        "exposure": scene.view_settings.exposure,
        "gamma": scene.view_settings.gamma,
        # Leaking these left every later render writing EXR — including the
        # gobo bake, which then could not find the PNG it went looking for.
        "file_format": scene.render.image_settings.file_format,
        "color_depth": scene.render.image_settings.color_depth,
        "color_mode": scene.render.image_settings.color_mode,
    }
    has_group = hasattr(scene, "compositing_node_group")
    saved["compositor"] = scene.compositing_node_group if has_group else scene.use_nodes

    try:
        material = bpy.data.materials.new("Photo3D_Probe")
        material.use_nodes = True
        tree = material.node_tree
        tree.nodes.clear()
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
        diffuse.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        tree.links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])

        mesh = bpy.data.meshes.new("Photo3D_Probe")
        size = 0.25
        mesh.from_pydata([(-size, -size, 0.0), (size, -size, 0.0),
                          (size, size, 0.0), (-size, size, 0.0)], [], [(0, 1, 2, 3)])
        mesh.update()
        probe = bpy.data.objects.new("Photo3D_Probe", mesh)
        probe.data.materials.append(material)
        # A hair above the ground plane so it is not co-planar with it.
        probe.location = (0.0, 0.0, 0.002)
        scene.collection.objects.link(probe)

        camera_data = bpy.data.cameras.new("Photo3D_ProbeCam")
        camera_data.type = "ORTHO"
        camera_data.ortho_scale = size
        camera = bpy.data.objects.new("Photo3D_ProbeCam", camera_data)
        scene.collection.objects.link(camera)
        camera.location = (0.0, 0.0, 0.35)          # looking straight down
        camera.rotation_euler = (0.0, 0.0, 0.0)

        scene.camera = camera
        scene.render.resolution_x = scene.render.resolution_y = 32
        scene.render.resolution_percentage = 100
        scene.render.film_transparent = False
        scene.cycles.samples = 24
        # The probe is a measurement, so no view transform may touch it.
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
        if has_group:
            scene.compositing_node_group = None
        else:
            scene.use_nodes = False

        import os
        path = os.path.join(bpy.app.tempdir, "photo3d_probe.exr")
        scene.render.filepath = path
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_depth = "32"
        bpy.ops.render.render(write_still=True)

        image = bpy.data.images.load(path, check_existing=False)
        buffer = np.empty(image.size[0] * image.size[1] * 4, dtype=np.float32)
        image.pixels.foreach_get(buffer)
        bpy.data.images.remove(image)
        radiance = float(np.median(
            imaging.luminance(buffer.reshape(-1, 4)[:, :3].reshape(1, -1, 3))))
        return radiance * float(np.pi)
    finally:
        scene.camera = saved["camera"]
        scene.render.filepath = saved["filepath"]
        scene.render.resolution_x = saved["resolution_x"]
        scene.render.resolution_y = saved["resolution_y"]
        scene.render.resolution_percentage = saved["percentage"]
        scene.render.film_transparent = saved["film_transparent"]
        scene.cycles.samples = saved["samples"]
        scene.view_settings.view_transform = saved["view_transform"]
        scene.view_settings.look = saved["look"]
        scene.view_settings.exposure = saved["exposure"]
        scene.view_settings.gamma = saved["gamma"]
        scene.render.image_settings.file_format = saved["file_format"]
        scene.render.image_settings.color_depth = saved["color_depth"]
        scene.render.image_settings.color_mode = saved["color_mode"]
        if has_group:
            scene.compositing_node_group = saved["compositor"]
        else:
            scene.use_nodes = saved["compositor"]
        for datablock, collection in ((probe, bpy.data.objects),
                                      (camera, bpy.data.objects),
                                      (material, bpy.data.materials)):
            if datablock is not None:
                collection.remove(datablock, do_unlink=True)


class PHOTO3D_OT_calibrate_exposure(bpy.types.Operator):
    """Match CG brightness to the photograph, by measuring not guessing"""
    bl_idname = "photo3d.calibrate_exposure"
    bl_label = "Match Exposure to Plate"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        """Scale the lights so a surface of the assumed albedo renders as
        bright as the same surface looks in the photograph.

        radiance = irradiance * albedo / pi, so the plate's ground brightness
        says what irradiance the scene needs, a probe render says what it
        currently has, and the ratio is the correction. The pi cancels.
        """
        props = context.scene.photo3d
        proxy_obj = bpy.data.objects.get("Photo3D_Proxy")
        plate = find_plate(proxy_obj)
        if plate is None:
            self.report({"ERROR"}, "no plate found; solve first")
            return {"CANCELLED"}

        pixels = image_to_array(plate, 256)
        ground = pixels[pixels.shape[0] // 2:]          # lower half is the ground
        plate_luminance = float(np.median(imaging.luminance(ground)))
        if plate_luminance <= 1e-5:
            self.report({"ERROR"}, "the plate's ground reads as black; cannot calibrate")
            return {"CANCELLED"}

        sun = bpy.data.lights.get("Photo3D_Sun")
        try:
            total = measure_ground_irradiance(context)
            if sun is not None and sun.energy > 0.0:
                held = sun.energy
                try:
                    sun.energy = 0.0
                    sky_only = measure_ground_irradiance(context)
                finally:
                    sun.energy = held
            else:
                sky_only = total
        except Exception as exc:                                  # noqa: BLE001
            self.report({"ERROR"}, f"probe render failed: {exc}")
            return {"CANCELLED"}
        if total <= 1e-6:
            self.report({"ERROR"}, "the scene delivers no light at ground level — "
                                   "is there a sun or a sky?")
            return {"CANCELLED"}

        # Two probes give the per-unit response of each light separately, which
        # is what makes the SPLIT solvable and not just the total. Measured on
        # the station plate: a physical sky delivers ~42 irradiance per unit of
        # Background strength, so the 1.0 default was pouring four times more
        # ambient into the scene than the sun was putting in. Light from every
        # direction at once casts no shadow, which is why objects looked lit but
        # cast nothing — the geometry was never the problem.
        sun_contribution = max(0.0, total - sky_only)
        sun_per_unit = (sun_contribution / sun.energy) if (sun and sun.energy > 0) else 0.0
        sky_per_unit = (sky_only / props.sky_strength) if props.sky_strength > 0 else 0.0

        needed = plate_luminance * float(np.pi) / max(props.assumed_albedo, 1e-3)
        # Cached so Direct sun share can re-split instantly afterwards.
        props.measured_sun_per_unit = sun_per_unit
        props.measured_sky_per_unit = sky_per_unit
        props.measured_needed = needed
        share = float(np.clip(props.sun_share, 0.0, 1.0))
        if sun_per_unit <= 1e-9:
            share = 0.0                       # no usable sun; put it all in the sky

        if share > 0.0 and sun_per_unit > 1e-9:
            props.sun_strength = needed * share / sun_per_unit
        elif sun is not None:
            props.sun_strength = 0.0
        if sky_per_unit > 1e-9:
            props.sky_strength = needed * (1.0 - share) / sky_per_unit

        self.report({"INFO"},
                    f"ground needs {needed:.1f} (plate {plate_luminance:.3f}, albedo "
                    f"{props.assumed_albedo:.2f}). Measured {sun_per_unit:.2f} per unit "
                    f"of sun and {sky_per_unit:.1f} per unit of sky -> sun "
                    f"{props.sun_strength:.2f}, sky {props.sky_strength:.4f} at a "
                    f"{share:.0%} direct share")
        return {"FINISHED"}


class PHOTO3D_OT_calibrate_bounce(bpy.types.Operator):
    """Solve the emission multiplier from the plate instead of guessing it"""
    bl_idname = "photo3d.calibrate_bounce"
    bl_label = "Estimate Strength"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.photo3d
        proxy_obj = bpy.data.objects.get("Photo3D_Proxy")
        plate = find_plate(proxy_obj)
        if plate is None:
            self.report({"ERROR"}, "no plate found; solve first")
            return {"CANCELLED"}

        # Calibrate against whatever the bounce actually emits from, or the
        # number will be right for an image the emitter is not using.
        emitter, is_linear = emitter_image(props, plate)
        pixels = image_to_array(emitter, 256)
        # Blender hands back scene-linear values already, so no sRGB decode.
        sun = bpy.data.lights.get("Photo3D_Sun")
        irradiance = (sun.energy if sun else 3.0) + props.sky_irradiance_guess
        props.bounce_strength = imaging.bounce_strength(
            pixels, irradiance, props.assumed_albedo)

        bounce = bpy.data.objects.get("Photo3D_Bounce")
        if bounce is not None and bounce.material_slots:
            for node in bounce.material_slots[0].material.node_tree.nodes:
                if node.type == "EMISSION":
                    node.inputs["Strength"].default_value = props.bounce_strength

        self.report({"INFO"}, f"strength = {props.bounce_strength:.2f} from the "
                              f"{'linear' if is_linear else 'display-referred'} plate "
                              f"(irradiance {irradiance:.1f}, albedo {props.assumed_albedo:.2f})")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# 2. sun gobo — steal the dapple that is already in the photo
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _render_settings(scene, proxy_obj, temp_material, gobo_resolution):
    """Swap the scene into gobo-bake state and put it back, whatever happens.

    The bake mutates render settings and materials on a live scene. The
    original version left the scene broken when a render failed — proxy still
    wearing a temporary emission material, compositor still off, samples still
    at 8 — and the damage looked like a solve bug hours later. try/finally is
    the whole fix.
    """
    # Blender 5 replaced Scene.node_tree with a compositing node group, so
    # "switch the compositor off for this render" is spelled two ways.
    has_node_group = hasattr(scene, "compositing_node_group")
    saved = {
        "camera": scene.camera,
        "filepath": scene.render.filepath,
        "resolution_x": scene.render.resolution_x,
        "resolution_y": scene.render.resolution_y,
        "resolution_percentage": scene.render.resolution_percentage,
        "film_transparent": scene.render.film_transparent,
        "compositor": scene.compositing_node_group if has_node_group else scene.use_nodes,
        "samples": scene.cycles.samples,
        "view_transform": scene.view_settings.view_transform,
        "file_format": scene.render.image_settings.file_format,
        "color_mode": scene.render.image_settings.color_mode,
    }
    saved_materials = [slot.material for slot in proxy_obj.material_slots]
    saved_flags = (proxy_obj.is_shadow_catcher, proxy_obj.visible_camera,
                   proxy_obj.visible_diffuse, proxy_obj.visible_glossy,
                   proxy_obj.visible_transmission, proxy_obj.visible_shadow)
    try:
        for slot in proxy_obj.material_slots:
            slot.material = temp_material
        proxy_obj.is_shadow_catcher = False
        proxy_obj.visible_camera = True
        scene.render.resolution_x = scene.render.resolution_y = gobo_resolution
        scene.render.resolution_percentage = 100
        scene.render.film_transparent = False
        if has_node_group:
            scene.compositing_node_group = None
        else:
            scene.use_nodes = False
        scene.cycles.samples = 8
        # The mask is data, not a picture. A view transform would grade it.
        scene.view_settings.view_transform = "Standard"
        # Set the format explicitly: whatever it happened to be decides the
        # extension Blender appends, and the loader below expects a PNG.
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGB"
        yield
    finally:
        scene.camera = saved["camera"]
        scene.render.filepath = saved["filepath"]
        scene.render.resolution_x = saved["resolution_x"]
        scene.render.resolution_y = saved["resolution_y"]
        scene.render.resolution_percentage = saved["resolution_percentage"]
        scene.render.film_transparent = saved["film_transparent"]
        if has_node_group:
            scene.compositing_node_group = saved["compositor"]
        else:
            scene.use_nodes = saved["compositor"]
        scene.cycles.samples = saved["samples"]
        scene.view_settings.view_transform = saved["view_transform"]
        scene.render.image_settings.file_format = saved["file_format"]
        scene.render.image_settings.color_mode = saved["color_mode"]
        for slot, material in zip(proxy_obj.material_slots, saved_materials):
            slot.material = material
        (proxy_obj.is_shadow_catcher, proxy_obj.visible_camera,
         proxy_obj.visible_diffuse, proxy_obj.visible_glossy,
         proxy_obj.visible_transmission, proxy_obj.visible_shadow) = saved_flags


def _mask_image(mask: np.ndarray) -> bpy.types.Image:
    height, width = mask.shape
    stale = bpy.data.images.get("Photo3D_ShadeMask")
    if stale is not None:            # otherwise repeat bakes leave .001, .002, ...
        bpy.data.images.remove(stale)
    image = bpy.data.images.new("Photo3D_ShadeMask", width, height, float_buffer=True)
    rgba = np.ones((height, width, 4), dtype=np.float32)
    rgba[..., :3] = mask[..., None]
    image.pixels.foreach_set(rgba[::-1].ravel())     # back to Blender's bottom-up order
    image.colorspace_settings.name = "Non-Color"
    return image


class PHOTO3D_OT_fetch_shade_mask(bpy.types.Operator):
    """Ask the daemon for a shade mask via intrinsic decomposition"""
    bl_idname = "photo3d.fetch_shade_mask"
    bl_label = "Extract Shade Mask"

    @classmethod
    def poll(cls, context):
        return bool(context.scene.photo3d.image_path)

    def execute(self, context):
        props = context.scene.photo3d
        try:
            result = client.shadow_mask({
                "image_path": bpy.path.abspath(props.image_path),
                "blur_px": props.mask_blur,
                "contrast": props.mask_contrast,
                "floor": props.mask_floor,
                "prefer_intrinsic": True,
            })
        except (client.SolverUnreachable, client.SolverRefused) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        props.shade_mask_npy = result["npy_path"]
        self.report({"INFO"}, f"{result['method']} mask, "
                              f"{result['shadow_fraction'] * 100:.0f}% in shade — "
                              f"{result['note']}")
        return {"FINISHED"}


class PHOTO3D_OT_bake_gobo(bpy.types.Operator):
    """Bake the plate's shade pattern into a cucoloris in front of the sun"""
    bl_idname = "photo3d.bake_gobo"
    bl_label = "Bake Sun Gobo"
    bl_options = {"REGISTER"}

    def execute(self, context):
        scene = context.scene
        props = scene.photo3d
        proxy_obj = bpy.data.objects.get("Photo3D_Proxy")
        sun = bpy.data.objects.get("Photo3D_Sun")
        if proxy_obj is None or sun is None:
            self.report({"ERROR"}, "need a solved scene with Photo3D_Proxy and Photo3D_Sun")
            return {"CANCELLED"}

        mask = self._load_mask(props, proxy_obj)
        if mask is None:
            self.report({"ERROR"}, "no plate to extract a mask from")
            return {"CANCELLED"}
        if float(np.mean(mask < 0.75)) < 0.005:
            self.report({"WARNING"}, "the mask is almost entirely lit — either this "
                                     "shot has no dapple, or Mask blur is smaller "
                                     "than the shadows you are trying to steal")

        mask_image = _mask_image(mask)
        temp_material = self._mask_material(mask_image)
        out_path = os.path.join(bpy.app.tempdir, "photo3d_gobo.png")

        # Look down the sun direction with an ortho camera: that converts the
        # pattern from camera space into SUN space, which is the only space in
        # which a cucoloris means anything.
        sun_direction = (sun.matrix_world.to_quaternion() @ Vector((0, 0, -1))).normalized()
        centre = self._proxy_centre(proxy_obj)
        gobo_camera = self._ortho_camera(scene, props, centre, sun_direction)

        try:
            with _render_settings(scene, proxy_obj, temp_material, props.gobo_res):
                scene.camera = gobo_camera
                scene.render.filepath = out_path
                bpy.ops.render.render(write_still=True)
        except Exception as exc:                                  # noqa: BLE001
            self.report({"ERROR"}, f"gobo render failed: {exc}")
            return {"CANCELLED"}
        finally:
            bpy.data.objects.remove(gobo_camera, do_unlink=True)
            bpy.data.materials.remove(temp_material)

        if not os.path.exists(out_path):
            self.report({"ERROR"},
                        f"the gobo render produced no file at {out_path}. Check the "
                        "output format in Render Properties is an image format")
            return {"CANCELLED"}
        gobo_image = bpy.data.images.load(out_path, check_existing=False)
        gobo_image.colorspace_settings.name = "Non-Color"
        self._build_gobo_plane(context, sun_direction, centre, gobo_image, props)
        self.report({"INFO"}, "gobo baked — lower Mask contrast if the shade reads too hard")
        return {"FINISHED"}

    # -- pieces ----------------------------------------------------------

    def _load_mask(self, props, proxy_obj):
        """Server mask if one has been fetched, otherwise compute it here."""
        if props.use_server_mask and props.shade_mask_npy:
            path = bpy.path.abspath(props.shade_mask_npy)
            if os.path.exists(path):
                return np.load(path).astype(np.float32)
        plate = find_plate(proxy_obj)
        if plate is None:
            return None
        pixels = image_to_array(plate, props.mask_res)
        return imaging.retinex_shadow_mask(pixels, blur_px=props.mask_blur,
                                           contrast=props.mask_contrast,
                                           floor=props.mask_floor)

    @staticmethod
    def _mask_material(mask_image):
        material = bpy.data.materials.new("Photo3D_TmpMask")
        material.use_nodes = True
        tree = material.node_tree
        tree.nodes.clear()
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        emission = tree.nodes.new("ShaderNodeEmission")
        texture = tree.nodes.new("ShaderNodeTexImage")
        coord = tree.nodes.new("ShaderNodeTexCoord")
        texture.image = mask_image
        texture.extension = "EXTEND"
        tree.links.new(coord.outputs["Window"], texture.inputs["Vector"])
        tree.links.new(texture.outputs["Color"], emission.inputs["Color"])
        tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
        return material

    @staticmethod
    def _proxy_centre(proxy_obj):
        count = len(proxy_obj.data.vertices)
        coords_flat = np.empty(count * 3, dtype=np.float32)
        proxy_obj.data.vertices.foreach_get("co", coords_flat)
        local = Vector(coords_flat.reshape(-1, 3).mean(axis=0).tolist())
        return proxy_obj.matrix_world @ local

    @staticmethod
    def _ortho_camera(scene, props, centre, sun_direction):
        data = bpy.data.cameras.new("Photo3D_GoboCam")
        data.type = "ORTHO"
        data.ortho_scale = props.gobo_size
        camera = bpy.data.objects.new("Photo3D_GoboCam", data)
        scene.collection.objects.link(camera)
        camera.location = centre - sun_direction * props.gobo_distance
        camera.rotation_mode = "QUATERNION"
        camera.rotation_quaternion = Vector((0, 0, -1)).rotation_difference(sun_direction)
        return camera

    @staticmethod
    def _build_gobo_plane(context, sun_direction, centre, gobo_image, props):
        existing = bpy.data.objects.get("Photo3D_Gobo")
        if existing is not None:
            bpy.data.objects.remove(existing, do_unlink=True)

        bpy.ops.mesh.primitive_plane_add(size=props.gobo_size)
        plane = context.active_object
        plane.name = "Photo3D_Gobo"
        plane.location = centre - sun_direction * (props.gobo_distance * 0.5)
        plane.rotation_mode = "QUATERNION"
        plane.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(-sun_direction)

        material = bpy.data.materials.new("Photo3D_GoboMat")
        material.use_nodes = True
        tree = material.node_tree
        tree.nodes.clear()
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
        texture = tree.nodes.new("ShaderNodeTexImage")
        coord = tree.nodes.new("ShaderNodeTexCoord")
        texture.image = gobo_image
        texture.extension = "EXTEND"
        # Generated coords on the plane match the ortho frame it was rendered
        # from, so the pattern lands back where it was measured.
        tree.links.new(coord.outputs["Generated"], texture.inputs["Vector"])
        tree.links.new(texture.outputs["Color"], transparent.inputs["Color"])
        tree.links.new(transparent.outputs["BSDF"], output.inputs["Surface"])
        plane.data.materials.append(material)

        # It exists only to tint shadow rays -- and it must be a shadow catcher,
        # for the same reason the bounce proxy is one.
        #
        # The gobo carries shade extracted FROM THE PHOTOGRAPH. Cycles counts
        # any shadow-casting object that is not a catcher as CG, and the
        # shadow-catcher pass is a ratio of light-with-CG over light-without-CG,
        # so a CG gobo appears in the numerator alone: the compositor then
        # multiplies the plate by the plate's own dapple. Measured on IMG_9920,
        # composite against plate:
        #
        #                        plate darkened   median   white cube
        #   no gobo                    3.27 %      1.000     0.3579
        #   gobo counted as CG        96.43 %      0.165     0.2157
        #   gobo as shadow catcher     4.37 %      1.000     0.2180
        #
        # 96 % of the photograph multiplied down to a sixth of its brightness.
        # Marking it a catcher puts it in both legs of the ratio, where it
        # cancels, while it goes on dappling CG exactly as before -- the cube
        # is unchanged at 0.218, still darkened from the 0.358 it reads with no
        # gobo at all. Both columns matter: a gobo that stopped reaching CG
        # would leave the plate alone and do nothing.
        set_ray_visibility(plane, camera=False, diffuse=False, glossy=False,
                           transmission=False, shadow=True, shadow_catcher=True)
        plane.visible_volume_scatter = False
        plane.display_type = "WIRE"


# ---------------------------------------------------------------------------
# 3. panorama
# ---------------------------------------------------------------------------

class PHOTO3D_OT_load_panorama(bpy.types.Operator):
    """Swap the Nishita sky for an outpainted equirectangular environment"""
    bl_idname = "photo3d.load_panorama"
    bl_label = "Load Panorama HDRI"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.photo3d
        path = bpy.path.abspath(props.panorama_path)
        if not path or not os.path.exists(path):
            self.report({"ERROR"}, "panorama file not found")
            return {"CANCELLED"}

        world = context.scene.world or bpy.data.worlds.new("Photo3D_World")
        context.scene.world = world
        world.use_nodes = True
        tree = world.node_tree
        tree.nodes.clear()
        output = tree.nodes.new("ShaderNodeOutputWorld"); output.location = (500, 0)
        background = tree.nodes.new("ShaderNodeBackground"); background.location = (300, 0)
        environment = tree.nodes.new("ShaderNodeTexEnvironment"); environment.location = (40, 0)
        mapping = tree.nodes.new("ShaderNodeMapping"); mapping.location = (-200, 0)
        coord = tree.nodes.new("ShaderNodeTexCoord"); coord.location = (-420, 0)

        environment.image = bpy.data.images.load(path, check_existing=True)
        background.inputs["Strength"].default_value = props.panorama_strength
        mapping.inputs["Rotation"].default_value[2] = radians(props.panorama_rotation)
        tree.links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
        tree.links.new(mapping.outputs["Vector"], environment.inputs["Vector"])
        tree.links.new(environment.outputs["Color"], background.inputs["Color"])
        tree.links.new(background.outputs["Background"], output.inputs["Surface"])

        # Let the bounce proxy own everything within ~10 m, where it is simply
        # more accurate, and let the panorama handle distance and rim light.
        self.report({"INFO"}, "panorama loaded; keep the bounce proxy for near-field light")
        return {"FINISHED"}


def _shift(mask, dy, dx):
    """Translate a boolean mask, filling vacated space with False."""
    h, w = mask.shape
    out = np.zeros_like(mask)
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
        mask[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out


def _dilate(mask, k):
    """Square dilation, done separably: 2*(2k+1) shifts rather than (2k+1)**2.

    Shifts rather than np.roll — roll wraps, so a probe near a frame edge would
    gather its "surroundings" from the opposite side of the picture.
    """
    for axis in (0, 1):
        grown = np.zeros_like(mask)
        for d in range(-k, k + 1):
            grown |= _shift(mask, d if axis == 0 else 0, 0 if axis == 0 else d)
        mask = grown
    return mask


def _ring_around(mask, inner=2, outer=8):
    """The band of pixels surrounding a mask without touching it.

    The inner gap keeps the sphere's own antialiased rim out of the reading —
    those pixels are part sphere, part photograph, and they drag the plate
    sample toward the very thing it is meant to be compared against.
    """
    return _dilate(mask, outer) & ~_dilate(mask, inner)


class PHOTO3D_OT_probe_at_object(bpy.types.Operator):
    """Render an 18% grey sphere where your object is and measure the light there"""
    bl_idname = "photo3d.probe_at_object"
    bl_label = "Probe Lighting At Object"

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (obj is not None and obj.type == "MESH"
                and not obj.name.startswith("Photo3D_"))

    def execute(self, context):
        """Close the loop between "the numbers are calibrated" and "it matches".

        Match Exposure to Plate solves for the GROUND, under an assumed albedo.
        Then the object gets placed somewhere the ground is not — under a
        canopy, beside a warm window — where the real local radiance can be
        several times different. Nothing else in this pipeline says by how
        much, so the last step is done by eye against a scene whose every other
        quantity was measured.

        This puts a matte 18% grey sphere exactly where the object is, renders
        it through the real pipeline, and prints what it came out as beside the
        photograph immediately around it.

        THE CAVEAT, because the number is misleading without it: the sphere is
        18% grey and the plate around it is whatever that surface happens to be.
        Comparing them is only an exposure match where the surroundings are
        near mid-grey. Two readings are honest regardless:

          * the ratio's CHANGE as you drag a slider, which is albedo-free —
            probe, change one thing, probe again;
          * the colour cast, R:B against the plate's R:B, which is what tells
            you the bounce is not carrying the warm light you can see in the
            photograph.
        """
        scene = context.scene
        target = context.active_object

        corners = [target.matrix_world @ Vector(c) for c in target.bound_box]
        centre = sum(corners, Vector((0.0, 0.0, 0.0))) / 8.0
        span = max(target.dimensions) if max(target.dimensions) > 0 else 1.0
        radius = min(max(span * 0.25, 0.05), 0.5)

        saved = {
            "filepath": scene.render.filepath,
            "percentage": scene.render.resolution_percentage,
            "samples": scene.cycles.samples,
            "file_format": scene.render.image_settings.file_format,
            "color_depth": scene.render.image_settings.color_depth,
            "color_mode": scene.render.image_settings.color_mode,
            "use_compositing": scene.render.use_compositing,
        }
        hidden = {o.name: o.hide_render for o in bpy.data.objects}
        # primitive_uv_sphere_add makes the sphere active, and the sphere is
        # deleted below — which left no active object at all, so the operator
        # could not be run a second time. Probing twice is the whole point:
        # the reading that means anything is the CHANGE between two of them.
        selection = [o for o in context.view_layer.objects if o.select_get()]
        sphere = material = None
        out = os.path.join(bpy.app.tempdir, "photo3d_probe.exr")

        try:
            material = bpy.data.materials.new("Photo3D_ProbeGrey")
            material.use_nodes = True
            tree = material.node_tree
            tree.nodes.clear()
            output = tree.nodes.new("ShaderNodeOutputMaterial")
            diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
            diffuse.inputs["Color"].default_value = (0.18, 0.18, 0.18, 1.0)
            tree.links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])

            bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=centre)
            sphere = context.active_object
            sphere.name = "Photo3D_ProbeSphere"
            sphere.data.materials.append(material)

            # Long edge near 800px: the sphere lands on enough pixels to average
            # without paying for a full-resolution render of a 48 MP plate.
            long_edge = max(scene.render.resolution_x, scene.render.resolution_y)
            scene.render.resolution_percentage = int(
                min(100, max(5, round(800.0 / max(long_edge, 1) * 100))))
            scene.cycles.samples = max(32, min(96, scene.cycles.samples))
            scene.render.image_settings.file_format = "OPEN_EXR"
            scene.render.image_settings.color_depth = "32"

            def render(rgba: bool) -> np.ndarray:
                scene.render.image_settings.color_mode = "RGBA" if rgba else "RGB"
                scene.render.filepath = out
                bpy.ops.render.render(write_still=True)
                image = bpy.data.images.load(out, check_existing=False)
                buf = np.empty(image.size[0] * image.size[1] * 4, dtype=np.float32)
                image.pixels.foreach_get(buf)
                array = buf.reshape(image.size[1], image.size[0], 4).copy()
                bpy.data.images.remove(image)
                return array

            # 1. the plate as it stands, with neither the object nor the sphere:
            #    sampling the ring off a render containing the sphere would read
            #    the sphere's own shadow as if it were the photograph.
            target.hide_render = True
            sphere.hide_render = True
            scene.render.use_compositing = True
            plate = render(False)[..., :3]

            # 2. the sphere standing in for the object
            sphere.hide_render = False
            lit = render(False)[..., :3]

            # 3. its silhouette, from a raw render with everything else gone
            for obj in bpy.data.objects:
                if obj.type == "MESH" and obj is not sphere:
                    obj.hide_render = True
            scene.render.use_compositing = False
            mask = render(True)[..., 3] > 0.5
        except Exception as exc:                                   # noqa: BLE001
            self.report({"ERROR"}, f"probe render failed: {exc}")
            return {"CANCELLED"}
        finally:
            if sphere is not None:
                bpy.data.objects.remove(sphere, do_unlink=True)
            if material is not None:
                bpy.data.materials.remove(material)
            for name, was_hidden in hidden.items():
                obj = bpy.data.objects.get(name)
                if obj is not None:
                    obj.hide_render = was_hidden
            for obj in context.view_layer.objects:
                obj.select_set(obj in selection)
            context.view_layer.objects.active = target
            scene.render.filepath = saved["filepath"]
            scene.render.resolution_percentage = saved["percentage"]
            scene.cycles.samples = saved["samples"]
            scene.render.image_settings.file_format = saved["file_format"]
            scene.render.image_settings.color_depth = saved["color_depth"]
            scene.render.image_settings.color_mode = saved["color_mode"]
            scene.render.use_compositing = saved["use_compositing"]

        if mask.sum() < 20:
            self.report({"ERROR"}, "the probe sphere is not visible from the camera — "
                                   "is the object behind the proxy, or off screen?")
            return {"CANCELLED"}

        ring = _ring_around(mask, inner=2, outer=8) & (plate.mean(axis=2) > 1e-4)
        if ring.sum() < 20:
            self.report({"ERROR"}, "no photograph around the object to compare against")
            return {"CANCELLED"}

        cg = lit[mask].mean(axis=0)
        near = plate[ring].mean(axis=0)
        cg_luma, near_luma = float(cg.mean()), float(near.mean())
        ratio = cg_luma / near_luma if near_luma > 1e-6 else float("inf")

        self.report({"INFO"}, f"18% grey at the object: "
                              f"R {cg[0]:.4f} G {cg[1]:.4f} B {cg[2]:.4f}")
        self.report({"INFO"}, f"photograph around it:   "
                              f"R {near[0]:.4f} G {near[1]:.4f} B {near[2]:.4f}")
        self.report({"INFO"}, f"brightness ratio {ratio:.2f}x  "
                              f"({int(mask.sum())} sphere px, {int(ring.sum())} plate px)")

        # Warm/cool is the reading that survives not knowing the plate's albedo.
        cg_warm = float(cg[0] / cg[2]) if cg[2] > 1e-6 else float("inf")
        near_warm = float(near[0] / near[2]) if near[2] > 1e-6 else float("inf")
        self.report({"INFO"}, f"warmth R:B  CG {cg_warm:.2f}  plate {near_warm:.2f}")

        if ratio < 0.75:
            self.report({"WARNING"}, "CG reads dark for this spot — raise Bounce "
                                     "strength, or add a light for a source the "
                                     "proxy cannot see")
        elif ratio > 1.4:
            self.report({"WARNING"}, "CG reads bright — lower sun or sky, or re-run "
                                     "Match Exposure to Plate")
        if near_warm > cg_warm * 1.25:
            self.report({"WARNING"}, "the photograph here is warmer than your CG. The "
                                     "bounce is not carrying that light: raise Bounce "
                                     "saturation, or select the warm surface and use "
                                     "Make Light From Selection")
        self.report({"INFO"}, "the sphere is 18% grey and the plate is whatever is "
                              "there, so trust the CHANGE between two probes more "
                              "than one absolute ratio")
        return {"FINISHED"}


CLASSES = (PHOTO3D_OT_calibrate_exposure, PHOTO3D_OT_bounce_proxy, PHOTO3D_OT_toggle_bounce,
           PHOTO3D_OT_calibrate_bounce, PHOTO3D_OT_fetch_shade_mask, PHOTO3D_OT_bake_gobo,
           PHOTO3D_OT_load_panorama, PHOTO3D_OT_probe_at_object)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
