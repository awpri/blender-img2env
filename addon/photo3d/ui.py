"""One panel, with sections.

The scaffold had two panels in two tabs because the lighting work arrived as a
second file. A user does not need to meet that history: bounce lighting is not
a different feature from solving the camera, it is the next step in the same
job. Sub-panels under one parent give sections that collapse, in one tab.

Order follows the order of operations in LIGHTING_ADDENDUM.md — solve, verify
the ground, then bounce, then sun, then the polish nobody needs on every shot.
"""

from __future__ import annotations

import bpy


class Photo3DPanel:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Photo3D"


class PHOTO3D_PT_main(Photo3DPanel, bpy.types.Panel):
    bl_label = "Photo3D"
    bl_idname = "PHOTO3D_PT_main"

    def draw(self, context):
        props = context.scene.photo3d
        layout = self.layout

        column = layout.column(align=True)
        column.prop(props, "image_path")

        row = layout.row(align=True)
        row.scale_y = 1.4
        row.operator("photo3d.solve", icon="CAMERA_DATA")
        layout.operator("photo3d.solve_metadata", icon="DRIVER_TRANSFORM")

        if not props.solved:
            box = layout.box()
            box.label(text="Not solved yet", icon="INFO")
            box.operator("photo3d.check_server", icon="PLUGIN")
            return

        box = layout.box()
        box.label(text="Measured", icon="CHECKMARK")
        grid = box.grid_flow(columns=2, align=True)
        grid.label(text="Pitch")
        grid.label(text=f"{props.solved_pitch:+.2f}deg")
        grid.label(text="Roll")
        grid.label(text=f"{props.solved_roll:+.2f}deg")
        grid.label(text="Heading")
        grid.label(text=f"{props.solved_heading:.1f}deg")
        grid.label(text="Height")
        grid.label(text=f"{props.solved_height:.2f} m")
        grid.label(text="Ground conf.")
        grid.label(text=f"{props.solved_confidence:.0%}")
        box.label(text=f"orientation from {props.solved_orientation_source}")

        if props.solved_confidence < 0.15:
            warning = box.column()
            warning.alert = True
            warning.label(text="Weak ground fit — check scale", icon="ERROR")

        if props.solved_warnings:
            box = layout.box()
            box.label(text="Warnings", icon="ERROR")
            for message in props.solved_warnings.split(" | "):
                if not message:
                    continue
                for line in _wrap(message, 44):
                    box.label(text=line)

        layout.operator("photo3d.align_view", icon="VIEW_CAMERA")


class PHOTO3D_PT_camera(Photo3DPanel, bpy.types.Panel):
    bl_label = "Camera & Lens"
    bl_parent_id = "PHOTO3D_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        props = context.scene.photo3d
        column = self.layout.column(align=True)
        column.prop(props, "focal_override")
        column.prop(props, "focal_convention")
        column.separator()
        column.prop(props, "use_heading")
        column.prop(props, "render_percent")
        column.prop(props, "view_transform")

        box = self.layout.box()
        box.label(text="Fallbacks (no accelerometer)", icon="ERROR")
        fallback = box.column(align=True)
        fallback.prop(props, "fallback_pitch")
        fallback.prop(props, "fallback_heading")
        fallback.prop(props, "fallback_height")


class PHOTO3D_PT_proxy(Photo3DPanel, bpy.types.Panel):
    bl_label = "Proxy Geometry"
    bl_parent_id = "PHOTO3D_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        props = context.scene.photo3d
        column = self.layout.column(align=True)
        column.prop(props, "refine")
        column.prop(props, "depth_long_edge")
        column.prop(props, "mesh_long_edge")
        column.prop(props, "edge_threshold")
        column.prop(props, "max_depth")
        column.prop(props, "collision_smooth")

        box = self.layout.box()
        # Only shout about the scale on the lens where it measured wrong: 14mm
        # alpine frames read 5x low, 24mm on a platform was within 5% at 20m.
        wide = props.solved and 0.0 < props.solved_focal < 20.0
        row = box.row()
        row.alert = wide and props.depth_scale == 1.0
        row.prop(props, "depth_scale")
        if wide and props.depth_scale == 1.0:
            column = box.column(align=True)
            column.scale_y = 0.8
            column.label(text=f"{props.solved_focal:.0f}mm is ultra-wide, where", icon="ERROR")
            column.label(text="depth has measured 5x low. Check")
            column.label(text="against something of known size:")
            column.label(text="tools/calibrate_scale.py")

        self.layout.operator("photo3d.rebuild_proxy", icon="MOD_REMESH")
        self.layout.operator("photo3d.drop_test", icon="PHYSICS")


class PHOTO3D_PT_bounce(Photo3DPanel, bpy.types.Panel):
    bl_label = "Bounce Light"
    bl_parent_id = "PHOTO3D_PT_main"

    def draw(self, context):
        props = context.scene.photo3d
        layout = self.layout
        layout.label(text="The photo as an emitter. Works indoors.", icon="LIGHT_AREA")

        column = layout.column(align=True)
        column.prop(props, "bounce_strength")
        column.prop(props, "bounce_saturation")

        row = layout.row(align=True)
        row.prop(props, "assumed_albedo")
        row.operator("photo3d.calibrate_bounce", text="Estimate")
        layout.prop(props, "sky_irradiance_guess")

        layout.operator("photo3d.bounce_proxy", icon="OUTLINER_OB_LIGHT")
        layout.operator("photo3d.toggle_bounce", icon="ARROW_LEFTRIGHT")


class PHOTO3D_PT_sun(Photo3DPanel, bpy.types.Panel):
    bl_label = "Sun & Sky"
    bl_parent_id = "PHOTO3D_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        props = context.scene.photo3d
        column = self.layout.column(align=True)
        column.prop(props, "sun_strength")
        column.prop(props, "sky_strength")
        column.prop(props, "sky_rotation_offset")

        if props.solved and props.solved_heading == 0.0:
            box = self.layout.box()
            box.label(text="No compass heading — the sun's", icon="ERROR")
            box.label(text="position in the sky is arbitrary.")


class PHOTO3D_PT_gobo(Photo3DPanel, bpy.types.Panel):
    bl_label = "Sun Gobo (dappled shade)"
    bl_parent_id = "PHOTO3D_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        props = context.scene.photo3d
        layout = self.layout
        column = layout.column(align=True)
        column.prop(props, "use_server_mask")
        column.prop(props, "mask_blur")
        column.prop(props, "mask_contrast")
        column.prop(props, "mask_floor")
        column.prop(props, "mask_res")
        layout.operator("photo3d.fetch_shade_mask", icon="SHADING_RENDERED")

        column = layout.column(align=True)
        column.prop(props, "gobo_size")
        column.prop(props, "gobo_distance")
        column.prop(props, "gobo_res")
        layout.operator("photo3d.bake_gobo", icon="RENDER_STILL")


class PHOTO3D_PT_panorama(Photo3DPanel, bpy.types.Panel):
    bl_label = "Panorama"
    bl_parent_id = "PHOTO3D_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        props = context.scene.photo3d
        column = self.layout.column(align=True)
        column.prop(props, "panorama_path")
        column.prop(props, "panorama_strength")
        column.prop(props, "panorama_rotation")
        self.layout.operator("photo3d.load_panorama", icon="WORLD")


class PHOTO3D_PT_minecraft(Photo3DPanel, bpy.types.Panel):
    bl_label = "Minecraft"
    bl_parent_id = "PHOTO3D_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        layout.operator("photo3d.crispify", icon="TEXTURE")
        column = layout.column(align=True)
        column.scale_y = 0.8
        column.label(text="Closest interpolation keeps a 16x16")
        column.label(text="texture at 16 hard pixels, any scale.")
        column.label(text="Distant blocks will shimmer — fix")
        column.label(text="with samples, not with filtering.")
        column.separator()
        column.label(text="Assets: install MCprep.")


def _wrap(text: str, width: int) -> list[str]:
    """Blender labels do not wrap and long warnings are the useful ones."""
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines[:6]


CLASSES = (PHOTO3D_PT_main, PHOTO3D_PT_camera, PHOTO3D_PT_proxy, PHOTO3D_PT_bounce,
           PHOTO3D_PT_sun, PHOTO3D_PT_gobo, PHOTO3D_PT_panorama, PHOTO3D_PT_minecraft)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
