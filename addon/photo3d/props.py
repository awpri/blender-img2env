"""Scene properties for the whole add-on.

One PropertyGroup, one panel. The scaffold had two of each because the lighting
work arrived as a second file; there is no reason for a user to meet that
history.

Results of the last solve are kept here too, so the panel can show what was
actually measured — pitch, roll, heading, ground confidence — rather than
leaving the user to trust a scene they cannot interrogate.
"""

from __future__ import annotations

from math import radians

import bpy
from mathutils import Vector
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, IntProperty,
                       StringProperty)


# ---------------------------------------------------------------------------
# live updates
# ---------------------------------------------------------------------------
#
# These properties used to be read only at solve time, which meant dragging the
# sun strength slider did nothing at all until you re-solved — and re-solving
# resets it from the defaults, so it never appeared to work. A light control
# that does not change the light is worse than no control: it teaches you that
# the lighting is broken.
#
# Each setter below pushes the value into the datablock that actually renders.

def _sun_lamp():
    lamp = bpy.data.objects.get("Photo3D_Sun")
    return lamp.data if lamp and lamp.type == "LIGHT" else None


def _world_nodes(context):
    world = context.scene.world
    return world.node_tree if world and world.use_nodes else None


def _update_sun_strength(self, context):
    lamp = _sun_lamp()
    if lamp is not None:
        lamp.energy = self.sun_strength


def _update_sky_strength(self, context):
    tree = _world_nodes(context)
    for node in (tree.nodes if tree else []):
        if node.type == "BACKGROUND":
            node.inputs["Strength"].default_value = self.sky_strength


def _effective_sun_azimuth(props) -> float:
    """Where the sun actually is, after the user's correction."""
    return props.solved_sun_azimuth + props.sun_azimuth_offset


def _update_sky_rotation(self, context):
    """Re-derive the sky's rotation from the solved azimuth plus the offsets.

    Offsets are calibration constants, so they are applied to the measured
    azimuth rather than accumulated onto whatever is already there.
    """
    tree = _world_nodes(context)
    for node in (tree.nodes if tree else []):
        if node.type == "TEX_SKY":
            node.sun_rotation = -radians(_effective_sun_azimuth(self)) + radians(
                self.sky_rotation_offset)


def _update_sun_direction(self, context):
    """Re-aim the sun lamp, and bring the sky round with it.

    The ephemeris knows where the sun was to arcminutes; what it cannot know is
    which way the camera faced. That comes from GPSImgDirection, a magnetometer,
    and a magnetometer inside a steel railway station or beside a car is simply
    wrong — by tens of degrees, not the 5-15 the specification promises. When
    the CG shadows do not run parallel to the real ones in the plate, this is
    the knob: turn it until they do. Everything else about the solve stays put.
    """
    from math import asin, cos, degrees, sin

    lamp = bpy.data.objects.get("Photo3D_Sun")
    if lamp is None:
        return
    azimuth = radians(_effective_sun_azimuth(self))
    elevation = radians(self.solved_sun_elevation)
    to_sun = Vector((sin(azimuth) * cos(elevation),
                     cos(azimuth) * cos(elevation),
                     sin(elevation)))
    lamp.location = to_sun * 100.0
    lamp.rotation_mode = "QUATERNION"
    lamp.rotation_quaternion = Vector((0.0, 0.0, -1.0)).rotation_difference(-to_sun)
    _update_sky_rotation(self, context)


def _bounce_nodes():
    bounce = bpy.data.objects.get("Photo3D_Bounce")
    if bounce is None or not bounce.material_slots:
        return None
    material = bounce.material_slots[0].material
    return material.node_tree if material and material.use_nodes else None


def _update_bounce_strength(self, context):
    tree = _bounce_nodes()
    for node in (tree.nodes if tree else []):
        if node.type == "EMISSION":
            node.inputs["Strength"].default_value = self.bounce_strength


def _update_bounce_saturation(self, context):
    tree = _bounce_nodes()
    for node in (tree.nodes if tree else []):
        if node.type == "HUE_SAT":
            node.inputs["Saturation"].default_value = self.bounce_saturation


def _update_view_transform(self, context):
    context.scene.view_settings.view_transform = self.view_transform


def _update_render_percent(self, context):
    context.scene.render.resolution_percentage = self.render_percent


class Photo3DProps(bpy.types.PropertyGroup):

    # --- input -----------------------------------------------------------
    image_path: StringProperty(
        name="Photo", subtype="FILE_PATH",
        description="ProRAW DNG preferred. A processed JPEG works for geometry "
                    "but its highlights are clipped, so image-derived lighting "
                    "will under-read the sky")
    focal_override: FloatProperty(
        name="Focal (35mm eq)", default=0.0, min=0.0,
        description="0 reads FocalLengthIn35mmFormat from EXIF")
    focal_convention: EnumProperty(
        name="Focal basis", default="long_edge",
        items=[("long_edge", "Long edge (36mm)",
                "35mm-equivalent measured against the 36mm long edge. Matches "
                "Blender's AUTO sensor fit exactly"),
               ("diagonal", "Diagonal (43.3mm)",
                "35mm-equivalent measured against the frame diagonal, the "
                "textbook crop factor. About 4% longer on a 4:3 sensor")],
        description="Apple does not document which it uses. If the backplate "
                    "consistently mismatches at the frame edges, switch this "
                    "rather than nudging the focal length")

    # --- depth -----------------------------------------------------------
    refine: BoolProperty(
        name="Marigold detail pass", default=False,
        description="Fit Marigold's micro-detail onto Depth Pro's metric scale. "
                    "Roughly triples solve time for a mesh the camera never sees")
    depth_long_edge: IntProperty(name="Depth resolution", default=1536, min=512, max=3072)
    mesh_long_edge: IntProperty(
        name="Proxy resolution", default=640, min=128, max=2048,
        description="Vertices along the long edge of the proxy mesh")
    edge_threshold: FloatProperty(
        name="Edge cull", default=0.08, min=0.005, max=1.0,
        description="Disparity spread above which a quad is deleted. Lower it "
                    "if stretched skirts trail off silhouettes, raise it if "
                    "silhouettes develop holes")
    depth_scale: FloatProperty(
        name="Depth scale", default=1.0, min=0.01, soft_max=20.0,
        description="Multiplier on metric depth. Leave at 1.0 for normal "
                    "scenes — at 24mm on a station platform Depth Pro measured "
                    "within 5% at 20m. It is ultra-wide frames with a huge "
                    "depth range that break it: at 14mm on alpine landscapes it "
                    "read 5x low. tools/calibrate_scale.py solves the factor "
                    "from an object of known size")
    max_depth: FloatProperty(
        name="Far clip (m)", default=120.0, min=5.0,
        description="Metric depth degrades badly past ~100m and a 14mm frame "
                    "has a lot of scene out there. Nothing collides with a peak "
                    "4km away")
    collision_smooth: FloatProperty(name="Collider smoothing", default=0.3, min=0.0, max=1.0)
    ground_plane_size: FloatProperty(
        name="Ground plane size (m)", default=200.0, min=1.0,
        description="A level plane at z=0. Its orientation comes from measured "
                    "gravity, so it is exact; only its distance below the lens "
                    "is a choice. Use it to stand objects on when the depth "
                    "mesh is unreliable — big landscapes, cluttered scenes")

    # --- camera ----------------------------------------------------------
    use_heading: BoolProperty(
        name="Yaw from compass", default=True,
        description="GPSImgDirection. Accurate to 5-15 degrees; magnetic "
                    "interference is the limit, not the sensor")
    fallback_pitch: FloatProperty(name="Fallback pitch", default=0.0, min=-89.0, max=89.0)
    fallback_heading: FloatProperty(name="Fallback heading", default=0.0, min=0.0, max=360.0)
    fallback_height: FloatProperty(name="Fallback height (m)", default=1.55, min=0.1)
    render_percent: IntProperty(name="Render %", default=50, min=5, max=100,
                                update=_update_render_percent)

    # --- lighting --------------------------------------------------------
    sun_strength: FloatProperty(name="Sun strength", default=4.0, min=0.0,
                                update=_update_sun_strength)
    sky_strength: FloatProperty(name="Sky strength", default=1.0, min=0.0,
                                update=_update_sky_strength)
    sun_share: FloatProperty(
        name="Direct sun share", default=0.85, min=0.0, max=1.0,
        description="Fraction of the ground's light that comes from the sun "
                    "rather than the sky. THIS is what decides whether shadows "
                    "exist: light from every direction at once casts none. "
                    "0.85 hard sun, 0.6 hazy, 0.2 overcast, 0.0 no sun at all")
    sun_azimuth_offset: FloatProperty(
        name="Sun azimuth offset", default=0.0, min=-180.0, max=180.0,
        update=_update_sun_direction,
        description="Turn the sun until CG shadows run parallel to the real "
                    "ones in the photo. The ephemeris is exact; the COMPASS is "
                    "not, and a magnetometer inside a steel station or beside a "
                    "car can be tens of degrees out")
    sky_rotation_offset: FloatProperty(
        name="Sky rotation offset", default=0.0, min=-360.0, max=360.0,
        update=_update_sky_rotation,
        description="Nishita's zero-rotation reference is not true north. "
                    "Calibrate once against a real hard shadow, then leave it")
    view_transform: EnumProperty(
        name="View transform", default="Standard", update=_update_view_transform,
        items=[("Standard", "Standard (match the plate)",
                "sRGB in, sRGB out. The plate passes through unchanged, which "
                "is what a composite needs"),
               ("AgX", "AgX (regrade everything)",
                "Blender's default. Re-tonemaps the plate as well as the CG, so "
                "the backplate stops matching the original photo")])

    # --- radiance --------------------------------------------------------
    bounce_strength: FloatProperty(name="Bounce strength", default=3.0, min=0.0,
                                   soft_max=50.0, update=_update_bounce_strength)
    bounce_saturation: FloatProperty(name="Bounce saturation", default=1.15, min=0.0,
                                     max=3.0, update=_update_bounce_saturation)
    assumed_albedo: FloatProperty(
        name="Ground albedo", default=0.18, min=0.01, max=0.95,
        description="grass 0.18  asphalt 0.10  gravel 0.25  snow 0.75")
    sky_irradiance_guess: FloatProperty(name="Sky irradiance", default=1.0, min=0.0)

    mask_res: IntProperty(name="Mask resolution", default=512, min=128, max=2048)
    mask_blur: IntProperty(
        name="Mask blur (px)", default=41, min=5, max=501,
        description="Must be comfortably LARGER than the dapple you want to "
                    "steal. The ratio only sees illumination finer than its own "
                    "kernel, so too small a blur returns a nearly blank mask")
    mask_contrast: FloatProperty(name="Mask contrast", default=1.6, min=0.1, max=6.0)
    mask_floor: FloatProperty(
        name="Mask floor", default=0.15, min=0.0, max=0.9,
        description="Darkest the gobo may go. Bounds the damage when dark "
                    "albedo is mistaken for shade")
    use_server_mask: BoolProperty(
        name="Extract on server", default=True,
        description="Use intrinsic decomposition on the daemon, which separates "
                    "reflectance from shading properly. Off falls back to the "
                    "luminance ratio computed here in Blender")
    shade_mask_npy: StringProperty(name="Shade mask", subtype="FILE_PATH")

    gobo_res: IntProperty(name="Gobo resolution", default=1024, min=256, max=4096)
    gobo_size: FloatProperty(name="Gobo size (m)", default=30.0, min=1.0)
    gobo_distance: FloatProperty(name="Gobo distance (m)", default=20.0, min=1.0)

    segment_long_edge: IntProperty(
        name="Segment resolution", default=768, min=256, max=2048,
        description="SAM 2 runs on a downscaled plate. Higher finds smaller "
                    "objects and takes longer")
    segment_max_regions: IntProperty(name="Max regions", default=24, min=1, max=128)
    segment_min_area: FloatProperty(
        name="Min region area", default=0.002, min=0.0001, max=0.5,
        description="Fraction of the frame a mask must cover to be kept")

    panorama_path: StringProperty(name="Panorama", subtype="FILE_PATH")
    panorama_strength: FloatProperty(name="Panorama strength", default=1.0, min=0.0)
    panorama_rotation: FloatProperty(name="Panorama rotation", default=0.0, min=-360.0, max=360.0)

    # --- last solve, for the readout ------------------------------------
    solved: BoolProperty(default=False)
    solved_summary: StringProperty(default="")
    solved_focal: FloatProperty(default=0.0)
    solved_pitch: FloatProperty(default=0.0)
    solved_roll: FloatProperty(default=0.0)
    solved_heading: FloatProperty(default=0.0)
    solved_height: FloatProperty(default=0.0)
    solved_confidence: FloatProperty(default=0.0)
    solved_focal: FloatProperty(default=0.0)
    solved_sun_azimuth: FloatProperty(default=0.0)
    solved_sun_elevation: FloatProperty(default=0.0)
    solved_orientation_source: StringProperty(default="")
    solved_warnings: StringProperty(default="")
    plate_png: StringProperty(default="")
    lighting_exr: StringProperty(default="")
    depth_npy: StringProperty(default="")


CLASSES = (Photo3DProps,)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.photo3d = bpy.props.PointerProperty(type=Photo3DProps)


def unregister():
    del bpy.types.Scene.photo3d
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
