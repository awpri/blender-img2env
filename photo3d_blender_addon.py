bl_info = {
    "name": "Photo3D — Stager-style plate solver",
    "author": "Alexander",
    "version": (0, 3, 0),
    "blender": (4, 2, 0),
    "location": "View3D > N-panel > Photo3D",
    "description": "Solve camera, proxy geometry and lighting from a single photo.",
    "category": "3D View",
}

import json
import os
import urllib.request
import urllib.error
from math import radians, degrees, atan2, sin, cos

import bpy
import numpy as np
from bpy.props import StringProperty, FloatProperty, IntProperty, BoolProperty
from mathutils import Vector, Matrix, Quaternion

ENDPOINT = "http://127.0.0.1:8765"

# Blender camera basis (X right, Y up, -Z forward) -> OpenCV (X right, Y down, +Z forward)
CV_TO_BL = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------

def post_json(route, payload, timeout=600):
    req = urllib.request.Request(
        ENDPOINT + route,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


# ---------------------------------------------------------------------------
# camera solve
# ---------------------------------------------------------------------------

def camera_matrix_from_solve(solve, props):
    """
    Gravity gives us pitch and roll exactly. Compass heading gives yaw.
    Metric depth gives height. Nothing is eyeballed.
    """
    g = solve.get("gravity_cam")
    if g:
        up_cam = (-Vector(g)).normalized()                # world up, in cam coords
        R = up_cam.rotation_difference(Vector((0, 0, 1))).to_matrix()
    else:
        # fallback: level camera, user-set pitch
        R = Matrix.Rotation(radians(90.0 + props.fallback_pitch), 3, "X")

    heading = solve.get("heading_deg")
    if heading is not None and props.use_heading:
        fwd = R @ Vector((0, 0, 1))                       # OpenCV forward, in world
        az_cur = atan2(fwd.x, fwd.y)                      # +Y == true north
        R = Matrix.Rotation(az_cur - radians(float(heading)), 3, "Z") @ R

    M = (R @ CV_TO_BL).to_4x4()
    M.translation = Vector((0.0, 0.0, float(solve["camera_height_m"])))
    return M


def setup_camera(scene, solve, props):
    cam_data = bpy.data.cameras.new("Photo3D_Cam")
    cam_data.sensor_fit = "HORIZONTAL" if solve["intrinsics"]["width"] >= solve["intrinsics"]["height"] else "VERTICAL"
    cam_data.sensor_width = 36.0
    cam_data.sensor_height = 36.0
    cam_data.lens = solve["intrinsics"]["focal_35mm"]

    cam = bpy.data.objects.new("Photo3D_Cam", cam_data)
    scene.collection.objects.link(cam)
    cam.matrix_world = camera_matrix_from_solve(solve, props)
    scene.camera = cam

    # viewport backplate
    img = bpy.data.images.load(solve["plate_png"], check_existing=True)
    cam_data.show_background_images = True
    bg = cam_data.background_images.new()
    bg.image = img
    bg.alpha = 1.0
    bg.display_depth = "BACK"

    scene.render.resolution_x = solve["intrinsics"]["width"]
    scene.render.resolution_y = solve["intrinsics"]["height"]
    scene.render.resolution_percentage = props.render_percent
    return cam, img


# ---------------------------------------------------------------------------
# proxy geometry
# ---------------------------------------------------------------------------

def build_proxy_mesh(solve, props, cam):
    """
    Unproject the depth map into camera space, then cull any quad that
    straddles a depth discontinuity. Without that cull you get the classic
    'melted spaghetti' skirt between foreground and background, which is the
    single biggest thing that makes depth-mesh comps look fake.
    """
    depth = np.load(solve["depth_npy"]).astype(np.float64)
    dh, dw = depth.shape

    step = max(1, int(round(max(dh, dw) / props.mesh_long_edge)))
    d = depth[::step, ::step]
    h, w = d.shape

    K = solve["intrinsics"]
    sx, sy = w / K["width"], h / K["height"]
    fx, fy = K["fx"] * sx, K["fy"] * sy
    cx, cy = K["cx"] * sx, K["cy"] * sy

    vv, uu = np.mgrid[0:h, 0:w].astype(np.float64)
    z = np.clip(d, 0.05, props.max_depth)
    X = (uu - cx) / fx * z
    Y = (vv - cy) / fy * z
    verts_cv = np.stack([X, Y, z], axis=-1).reshape(-1, 3)
    verts = verts_cv * np.array([1.0, -1.0, -1.0])        # -> Blender camera space

    # quad culling on relative disparity gradient
    disp = 1.0 / z
    d00, d10, d01, d11 = disp[:-1, :-1], disp[1:, :-1], disp[:-1, 1:], disp[1:, 1:]
    stack = np.stack([d00, d10, d01, d11])
    spread = stack.max(axis=0) - stack.min(axis=0)
    keep = spread < (props.edge_threshold * stack.mean(axis=0))

    idx = np.arange(h * w).reshape(h, w)
    a, b_, c, e = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    faces = np.stack([a, b_, c, e], axis=-1)[keep].reshape(-1, 4)

    mesh = bpy.data.meshes.new("Photo3D_Proxy")
    mesh.from_pydata(verts.tolist(), [], faces.tolist())
    mesh.validate()
    mesh.update()

    obj = bpy.data.objects.new("Photo3D_Proxy", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.matrix_world = cam.matrix_world          # exact alignment, by construction

    # invisible to camera, but present for shadows, occlusion and reflections
    obj.is_shadow_catcher = True
    obj.visible_diffuse = True
    obj.visible_glossy = True
    obj.visible_transmission = True
    obj.visible_shadow = True
    obj.display_type = "WIRE"

    # physics: static collider so blocks and mobs land on real terrain
    obj.modifiers.new("Smooth", "SMOOTH").factor = props.collision_smooth
    bpy.context.view_layer.objects.active = obj
    bpy.ops.rigidbody.object_add(type="PASSIVE")
    obj.rigid_body.collision_shape = "MESH"
    obj.rigid_body.mesh_source = "FINAL"
    obj.rigid_body.friction = 0.9
    return obj


# ---------------------------------------------------------------------------
# the proxy shader — invisible to the eye, fully present to every other ray
# ---------------------------------------------------------------------------

def make_proxy_material(plate_image):
    mat = bpy.data.materials.new("Photo3D_ProxyMat")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial");      out.location = (600, 0)
    mix = nt.nodes.new("ShaderNodeMixShader");           mix.location = (400, 0)
    transp = nt.nodes.new("ShaderNodeBsdfTransparent");  transp.location = (200, -180)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled");     bsdf.location = (200, 200)
    lp = nt.nodes.new("ShaderNodeLightPath");            lp.location = (200, 420)
    tex = nt.nodes.new("ShaderNodeTexImage");            tex.location = (-140, 200)
    coord = nt.nodes.new("ShaderNodeTexCoord");          coord.location = (-380, 200)

    tex.image = plate_image
    tex.interpolation = "Cubic"            # the PLATE wants smoothing
    tex.extension = "EXTEND"
    # Window coords are screen-space, so the plate reprojects onto the proxy
    # exactly as the real light left it. This is what makes a chrome block
    # reflect the actual gravel underneath it.
    nt.links.new(coord.outputs["Window"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.55
    bsdf.inputs["Metallic"].default_value = 0.0

    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
    nt.links.new(bsdf.outputs["BSDF"], mix.inputs[1])
    nt.links.new(transp.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return mat


# ---------------------------------------------------------------------------
# lighting
# ---------------------------------------------------------------------------

def setup_lighting(scene, solve, props):
    sun_info = solve.get("sun")
    if not sun_info or sun_info["elevation_deg"] < -3.0:
        return None

    az = radians(sun_info["azimuth_deg"])       # clockwise from true north (+Y)
    el = radians(sun_info["elevation_deg"])
    to_sun = Vector((sin(az) * cos(el), cos(az) * cos(el), sin(el)))

    lamp_data = bpy.data.lights.new("Photo3D_Sun", type="SUN")
    lamp_data.angle = radians(0.545)            # true solar disc
    lamp_data.energy = props.sun_strength
    lamp = bpy.data.objects.new("Photo3D_Sun", lamp_data)
    scene.collection.objects.link(lamp)
    lamp.location = to_sun * 100.0
    lamp.rotation_mode = "QUATERNION"
    lamp.rotation_quaternion = Vector((0, 0, -1)).rotation_difference(-to_sun)

    # Nishita sky: physically-derived ambient + a matching horizon for reflections
    world = scene.world or bpy.data.worlds.new("Photo3D_World")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld");   out.location = (400, 0)
    bg = nt.nodes.new("ShaderNodeBackground");     bg.location = (200, 0)
    sky = nt.nodes.new("ShaderNodeTexSky");        sky.location = (-100, 0)
    sky.sky_type = "NISHITA"
    sky.sun_elevation = el
    # Nishita's zero-rotation reference differs from true north; calibrate once
    # against a known shadow direction, then leave sky_rotation_offset alone.
    sky.sun_rotation = -az + radians(props.sky_rotation_offset)
    sky.sun_disc = False                        # the SUN lamp already provides it
    sky.altitude = max(0.0, (solve.get("geo") or {}).get("alt", 0.0) or 0.0)
    bg.inputs["Strength"].default_value = props.sky_strength
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    return lamp


# ---------------------------------------------------------------------------
# render + compositor
# ---------------------------------------------------------------------------

def setup_render(scene, plate_image):
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.render.film_transparent = True
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.samples = 512
    scene.cycles.use_denoising = True
    scene.view_settings.view_transform = "AgX"

    scene.use_nodes = True
    nt = scene.node_tree
    nt.nodes.clear()
    rl = nt.nodes.new("CompositorNodeRLayers");  rl.location = (0, 200)
    plate = nt.nodes.new("CompositorNodeImage"); plate.location = (0, -200)
    plate.image = plate_image
    over = nt.nodes.new("CompositorNodeAlphaOver"); over.location = (320, 0)
    comp = nt.nodes.new("CompositorNodeComposite"); comp.location = (560, 0)
    view = nt.nodes.new("CompositorNodeViewer");    view.location = (560, -180)
    nt.links.new(plate.outputs["Image"], over.inputs[1])
    nt.links.new(rl.outputs["Image"], over.inputs[2])
    nt.links.new(over.outputs["Image"], comp.inputs["Image"])
    nt.links.new(over.outputs["Image"], view.inputs["Image"])


# ---------------------------------------------------------------------------
# minecraft asset helper
# ---------------------------------------------------------------------------

class PHOTO3D_OT_crispify(bpy.types.Operator):
    """Force every image texture on selected objects to nearest-neighbour"""
    bl_idname = "photo3d.crispify"
    bl_label = "Crispify Textures (nearest-neighbour)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        n = 0
        for obj in context.selected_objects:
            for slot in obj.material_slots:
                mat = slot.material
                if not mat or not mat.use_nodes:
                    continue
                for node in mat.node_tree.nodes:
                    if node.type == "TEX_IMAGE":
                        node.interpolation = "Closest"
                        node.extension = "EXTEND"
                        if node.image:
                            node.image.colorspace_settings.name = "sRGB"
                        n += 1
        self.report({"INFO"}, f"{n} texture node(s) set to Closest")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# main operator
# ---------------------------------------------------------------------------

class PHOTO3D_OT_solve(bpy.types.Operator):
    bl_idname = "photo3d.solve"
    bl_label = "Solve Photo"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.photo3d
        payload = {
            "image_path": bpy.path.abspath(props.image_path),
            "refine": props.refine,
            "depth_long_edge": props.depth_long_edge,
            "assumed_eye_height": props.fallback_height,
        }
        if props.focal_override > 0.0:
            payload["focal_override"] = props.focal_override

        try:
            solve = post_json("/solve", payload)
        except urllib.error.URLError as e:
            self.report({"ERROR"}, f"Solver unreachable at {ENDPOINT} — is it running? ({e})")
            return {"CANCELLED"}

        scene = context.scene
        cam, plate = setup_camera(scene, solve, props)
        proxy = build_proxy_mesh(solve, props, cam)
        proxy.data.materials.append(make_proxy_material(plate))
        setup_lighting(scene, solve, props)
        setup_render(scene, plate)

        if not scene.rigidbody_world:
            bpy.ops.rigidbody.world_add()

        src = "gravity vector" if solve.get("gravity_cam") else "fallback pitch"
        self.report({"INFO"},
                    f"Solved: {solve['intrinsics']['focal_35mm']:.0f}mm eq, "
                    f"h={solve['camera_height_m']:.2f}m, orientation from {src}")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# properties + UI
# ---------------------------------------------------------------------------

class Photo3DProps(bpy.types.PropertyGroup):
    image_path: StringProperty(name="Photo", subtype="FILE_PATH")
    focal_override: FloatProperty(name="Focal (35mm eq)", default=0.0, min=0.0,
                                  description="0 = read from EXIF")
    refine: BoolProperty(name="Marigold detail pass", default=False)
    depth_long_edge: IntProperty(name="Depth res", default=1536, min=512, max=3072)
    mesh_long_edge: IntProperty(name="Proxy res", default=640, min=128, max=2048)
    edge_threshold: FloatProperty(name="Edge cull", default=0.08, min=0.005, max=1.0)
    max_depth: FloatProperty(name="Far clip (m)", default=120.0, min=5.0)
    collision_smooth: FloatProperty(name="Collider smooth", default=0.3, min=0.0, max=1.0)
    fallback_pitch: FloatProperty(name="Fallback pitch", default=0.0, min=-89, max=89)
    fallback_height: FloatProperty(name="Fallback height (m)", default=1.55, min=0.1)
    use_heading: BoolProperty(name="Yaw from compass", default=True)
    sun_strength: FloatProperty(name="Sun strength", default=4.0, min=0.0)
    sky_strength: FloatProperty(name="Sky strength", default=1.0, min=0.0)
    sky_rotation_offset: FloatProperty(name="Sky rot offset", default=0.0, min=-360, max=360)
    render_percent: IntProperty(name="Render %", default=50, min=5, max=100)


class PHOTO3D_PT_panel(bpy.types.Panel):
    bl_label = "Photo3D"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Photo3D"

    def draw(self, context):
        p = context.scene.photo3d
        col = self.layout.column(align=True)
        col.prop(p, "image_path")
        col.prop(p, "focal_override")
        col.separator()
        col.prop(p, "refine")
        col.prop(p, "depth_long_edge")
        col.prop(p, "mesh_long_edge")
        col.prop(p, "edge_threshold")
        col.separator()
        col.prop(p, "use_heading")
        col.prop(p, "sun_strength")
        col.prop(p, "sky_strength")
        col.prop(p, "render_percent")
        col.separator()
        col.operator("photo3d.solve", icon="CAMERA_DATA")
        col.operator("photo3d.crispify", icon="TEXTURE")


CLASSES = (Photo3DProps, PHOTO3D_OT_solve, PHOTO3D_OT_crispify, PHOTO3D_PT_panel)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.photo3d = bpy.props.PointerProperty(type=Photo3DProps)


def unregister():
    del bpy.types.Scene.photo3d
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
