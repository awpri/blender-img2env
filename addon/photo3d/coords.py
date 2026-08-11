"""Coordinate math for the Photo3D pipeline. numpy + stdlib only — no bpy.

WHY THIS FILE LIVES IN THE ADD-ON
    The add-on must be self-contained when zipped into Blender, and Blender's
    bundled Python may never grow dependencies. So the add-on owns this module.
    The server imports it (see server/_core.py) rather than vendoring a copy,
    because two implementations of a sign convention will drift apart and the
    drift will not be obvious — it will look like an accelerometer bug.

    Nothing here imports bpy, so all of it is unit-testable outside Blender.
    That is the point: every sign in this file is pinned by a test in
    server/tests/test_coords.py before it is allowed near a viewport.

FRAMES
    world (Blender)      +X east, +Y north, +Z up
    camera (OpenCV)      +x right, +y down, +z forward along the optical axis
    camera (Blender obj) +x right, +y up,  -z forward
    device (CoreMotion)  +X toward the phone's right edge in portrait,
                         +Y toward the phone's top edge,
                         +Z out of the screen toward the user

    Blender's camera object looks down its local -Z with +Y up, so
    R_world_object = R_world_camera @ CV_TO_BL where CV_TO_BL = diag(1, -1, -1).

QUANTITIES
    gravity vectors are unit vectors pointing DOWN, in the frame the name says.
    heading   degrees clockwise from true north, of the optical axis (EXIF
              GPSImgDirection).
    pitch     degrees the optical axis sits above the horizon. + = looking up.
    roll      degrees the world's up direction leans to the RIGHT in the
              displayed image. + = the horizon runs downhill to the right.
              (Equivalently: the camera is banked counter-clockwise as seen
              from behind it.) This sign is a free choice; it is pinned here so
              that the IMG_7096 reference photo reports +0.7 rather than -0.7.
"""

from __future__ import annotations

import math

import numpy as np

# OpenCV camera basis -> Blender camera-object basis.
CV_TO_BL = np.array([[1.0, 0.0, 0.0],
                     [0.0, -1.0, 0.0],
                     [0.0, 0.0, -1.0]])

WORLD_UP = np.array([0.0, 0.0, 1.0])

# Beyond this the optical axis is close enough to the zenith/nadir that
# GPSImgDirection no longer describes where the *axis* points, only where the
# top of the frame faces. See rotation_from_gravity() for the fallback.
_NEAR_VERTICAL = math.cos(math.radians(1.0))  # |sin(pitch)| above this


class CoordinateError(ValueError):
    """Raised when an input cannot describe a physically realisable camera."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def normalize(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise CoordinateError(f"cannot normalize a zero-length vector: {v!r}")
    return v / n


def azimuth_deg(v) -> float:
    """Compass bearing of a world vector: degrees clockwise from north (+Y)."""
    v = np.asarray(v, dtype=np.float64)
    return float(np.degrees(math.atan2(v[0], v[1])) % 360.0)


# ---------------------------------------------------------------------------
# device -> camera, per EXIF orientation
# ---------------------------------------------------------------------------
#
# Apple's MakerNote AccelerationVector is in DEVICE axes, which do not move when
# you rotate the phone. The image, however, is stored in the sensor's own axes
# and then rotated for display according to EXIF Orientation. Intrinsics, depth
# maps and the proxy mesh all live in DISPLAY space, so gravity has to be
# carried into display space too.
#
# Derivation, once, so the next person does not have to redo it:
#
#   The rear camera's native sensor buffer is device-fixed. Holding the phone in
#   landscape with the home button on the right gives Orientation 1 and an
#   upright image, and in that pose device +X points up, device +Y points left,
#   device +Z points backwards. The OpenCV camera axes there are
#       x_cam = right = -Y_dev,  y_cam = down = -X_dev,  z_cam = fwd = -Z_dev
#   which is the Orientation-1 row below. The other three rows are that row
#   composed with the image rotation Orientation asks for: 6 = rotate 90 CW to
#   display, 8 = rotate 90 CCW, 3 = rotate 180.
#
# Cross-check (all four must agree, and they do): a level camera reads
#   O1 g_dev=(-1,0,0)   O3 g_dev=(+1,0,0)   O6 g_dev=(0,-1,0)   O8 g_dev=(0,+1,0)
# and every row maps those to g_cam=(0,1,0), i.e. "down is +y in the image".
# test_coords.py asserts exactly that.

_DEVICE_TO_CAMERA = {
    1: np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),
    3: np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),
    6: np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
    8: np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]),
}

#: Orientations that swap the stored image's width and height for display.
TRANSPOSED_ORIENTATIONS = frozenset({5, 6, 7, 8})


def gravity_device_to_camera(g_device, orientation: int = 6) -> np.ndarray:
    """Rotate a device-frame gravity vector into display-space camera coords.

    orientation is the EXIF Orientation tag. Only the four non-mirrored values
    a rear camera can produce are accepted; 2/4/5/7 flip handedness and would
    silently corrupt the rotation solve, so they raise instead.
    """
    try:
        M = _DEVICE_TO_CAMERA[int(orientation)]
    except KeyError:
        raise CoordinateError(
            f"EXIF Orientation {orientation} is mirrored or unknown; a rear "
            "camera only produces 1, 3, 6 or 8."
        ) from None
    return normalize(M @ normalize(g_device))


def gravity_camera_to_device(g_camera, orientation: int = 6) -> np.ndarray:
    """Inverse of gravity_device_to_camera(). Used to synthesise fixtures."""
    try:
        M = _DEVICE_TO_CAMERA[int(orientation)]
    except KeyError:
        raise CoordinateError(f"unsupported EXIF Orientation {orientation}") from None
    return normalize(M.T @ normalize(g_camera))


# ---------------------------------------------------------------------------
# gravity <-> pitch/roll
# ---------------------------------------------------------------------------

def pitch_roll_from_gravity(g_camera) -> tuple[float, float]:
    """Camera pitch and roll in degrees, from gravity in camera coords.

    With the camera's forward axis f = (0,0,1) and gravity g pointing down:
        g . f = -sin(pitch)                    -> pitch = asin(-g_z)
    and the world-up direction -g projects into the image plane as (-g_x, -g_y);
    the angle that projection leans away from straight-up (0, -1) is the roll.
    """
    g = normalize(g_camera)
    pitch = math.degrees(math.asin(float(np.clip(-g[2], -1.0, 1.0))))
    roll = math.degrees(math.atan2(float(-g[0]), float(g[1])))
    return pitch, roll


def gravity_from_pitch_roll(pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Exact inverse of pitch_roll_from_gravity(). Exists so the round trip is
    testable, and so tests can synthesise a camera pose without a photograph."""
    p, r = math.radians(pitch_deg), math.radians(roll_deg)
    return np.array([-math.sin(r) * math.cos(p),
                     math.cos(r) * math.cos(p),
                     -math.sin(p)])


# ---------------------------------------------------------------------------
# the full rotation
# ---------------------------------------------------------------------------

def _triad(primary, secondary) -> np.ndarray:
    """Orthonormal 3x3 with `primary` as column 0 and `secondary` gram-schmidted
    into column 1. Columns 0 and 1 must not be parallel."""
    e1 = normalize(primary)
    rest = np.asarray(secondary, dtype=np.float64) - np.dot(secondary, e1) * e1
    e2 = normalize(rest)
    return np.column_stack([e1, e2, np.cross(e1, e2)])


def rotation_from_gravity(g_camera, heading_deg: float | None = None,
                          fallback_heading_deg: float = 0.0) -> np.ndarray:
    """Camera-to-world rotation (3x3), from measured gravity and the compass.

    Gravity fixes two of the three degrees of freedom exactly — that is the
    whole premise of this pipeline. The compass fixes the third. The result is
    the unique rotation R with

        R @ g_camera == (0, 0, -1)          gravity points down in world
        azimuth(R @ (0,0,1)) == heading     the optical axis faces the compass

    and it is built by matching orthonormal triads rather than by composing
    Euler angles, because triad matching has no gimbal case and no ordering
    convention to get wrong.

    If the camera is within a degree of pointing straight up or down, the
    optical axis has no meaningful bearing, so the heading is applied to the
    top-of-frame direction instead. That branch keeps the function total; it is
    not expected to fire on a landscape photograph.
    """
    g = normalize(g_camera)
    up_cam = -g
    heading = fallback_heading_deg if heading_deg is None else float(heading_deg)

    pitch_rad = math.asin(float(np.clip(up_cam[2], -1.0, 1.0)))
    if abs(math.sin(pitch_rad)) < _NEAR_VERTICAL:
        h = math.radians(heading)
        forward_world = np.array([math.sin(h) * math.cos(pitch_rad),
                                  math.cos(h) * math.cos(pitch_rad),
                                  math.sin(pitch_rad)])
        cam = _triad(np.array([0.0, 0.0, 1.0]), up_cam)   # forward, then up
        world = _triad(forward_world, WORLD_UP)
    else:
        # Near-vertical: aim the top of the frame at the compass bearing.
        h = math.radians(heading)
        frame_up_world = np.array([math.sin(h), math.cos(h), 0.0])
        cam = _triad(np.array([0.0, -1.0, 0.0]), up_cam)  # image-up, then up
        world = _triad(frame_up_world, WORLD_UP)

    return world @ cam.T


def camera_matrix(g_camera, heading_deg: float | None, height_m: float,
                  east_m: float = 0.0, north_m: float = 0.0,
                  fallback_heading_deg: float = 0.0) -> np.ndarray:
    """4x4 world matrix for a Blender camera object.

    The rotation is the OpenCV camera-to-world rotation post-multiplied by
    CV_TO_BL, because Blender's camera object looks down -Z with +Y up while
    everything upstream of here uses OpenCV's +Z forward, +Y down.
    """
    R = rotation_from_gravity(g_camera, heading_deg, fallback_heading_deg)
    M = np.eye(4)
    M[:3, :3] = R @ CV_TO_BL
    M[:3, 3] = (east_m, north_m, height_m)
    return M


def sun_direction(azimuth_deg_: float, elevation_deg: float) -> np.ndarray:
    """Unit world vector pointing TOWARD the sun. Azimuth is clockwise from
    true north, matching both the solar ephemeris and GPSImgDirection."""
    az, el = math.radians(azimuth_deg_), math.radians(elevation_deg)
    return np.array([math.sin(az) * math.cos(el),
                     math.cos(az) * math.cos(el),
                     math.sin(el)])


# ---------------------------------------------------------------------------
# depth -> geometry
# ---------------------------------------------------------------------------

def scaled_intrinsics(intrinsics: dict, width: int, height: int) -> tuple[float, float, float, float]:
    """Rescale full-resolution intrinsics to a smaller depth/mesh raster.

    Depth inference runs at ~1.5K while the plate is 8K, so every consumer of a
    depth map needs this. Getting it wrong scales the whole scene, which reads
    as "the depth model is bad" rather than "the intrinsics are stale".
    """
    sx = width / float(intrinsics["width"])
    sy = height / float(intrinsics["height"])
    return (intrinsics["fx"] * sx, intrinsics["fy"] * sy,
            intrinsics["cx"] * sx, intrinsics["cy"] * sy)


def unproject(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Depth raster -> (H, W, 3) point cloud in OpenCV camera coords.

    `depth` is z-depth along the optical axis (what Depth Pro returns), not
    ray length. Mixing those up puts a bowl in every flat floor.
    """
    h, w = depth.shape
    vv, uu = np.mgrid[0:h, 0:w].astype(np.float64)
    z = depth.astype(np.float64)
    return np.stack([(uu - cx) / fx * z, (vv - cy) / fy * z, z], axis=-1)


def height_above_camera(points: np.ndarray, g_camera) -> np.ndarray:
    """Signed height of each point relative to the camera, along world up."""
    return points @ (-normalize(g_camera))


def cull_discontinuous_quads(depth: np.ndarray, edge_threshold: float) -> np.ndarray:
    """Boolean (H-1, W-1) mask of quads worth keeping.

    A quad straddling a depth discontinuity produces the rubber sheet that
    stretches from every foreground silhouette back to the horizon, and CG
    shadows crawl up that sheet. Culling on *relative disparity* spread rather
    than on metric depth difference is what makes one threshold work for both a
    rock at 2 m and a ridge at 2 km.
    """
    disp = 1.0 / np.clip(depth, 1e-6, None)
    corners = np.stack([disp[:-1, :-1], disp[1:, :-1], disp[:-1, 1:], disp[1:, 1:]])
    spread = corners.max(axis=0) - corners.min(axis=0)
    finite = np.isfinite(corners).all(axis=0)
    return finite & (spread < edge_threshold * corners.mean(axis=0))


def proxy_arrays(depth: np.ndarray, intrinsics: dict, mesh_long_edge: int = 640,
                 edge_threshold: float = 0.08, max_depth: float = 120.0,
                 min_depth: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Depth raster -> (verts, quads) ready for Blender's from_pydata.

    Vertices come out in BLENDER camera space (x right, y up, -z forward), so
    the caller can drop the mesh straight onto the camera's matrix_world and
    have it line up by construction.

    Winding is (r,c) -> (r+1,c) -> (r+1,c+1) -> (r,c+1), which after the y/z
    flip gives face normals pointing back at the camera. Reverse it and the
    proxy turns into a one-sided mirror that catches no shadows.
    """
    dh, dw = depth.shape
    step = max(1, int(round(max(dh, dw) / float(mesh_long_edge))))
    d = depth[::step, ::step]
    h, w = d.shape

    fx, fy, cx, cy = scaled_intrinsics(intrinsics, w, h)
    z = np.clip(np.nan_to_num(d, nan=max_depth, posinf=max_depth, neginf=min_depth),
                min_depth, max_depth)

    pts = unproject(z, fx, fy, cx, cy).reshape(-1, 3)
    verts = pts * np.array([1.0, -1.0, -1.0])          # OpenCV -> Blender camera

    keep = cull_discontinuous_quads(z, edge_threshold)
    idx = np.arange(h * w).reshape(h, w)
    quads = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]],
                     axis=-1)[keep].reshape(-1, 4)
    return verts, quads


def ground_plane_height(depth: np.ndarray, intrinsics: dict, g_camera,
                        lower_frame_fraction: float = 0.45,
                        near_m: float = 0.5, far_m: float = 60.0,
                        band_m: float = 0.15,
                        min_samples: int = 500) -> tuple[float | None, float]:
    """Camera height above the ground, in metres, plus an inlier confidence.

    Because gravity is measured, the ground plane's *normal* is already known
    and the usual 3-DoF RANSAC collapses to finding one number: the offset.
    That is a 1-D mode-finding problem, which is both faster and far more
    stable than plane fitting on a noisy monocular point cloud.

    Returns (None, 0.0) when there is no gravity or too little usable ground.
    """
    if g_camera is None:
        return None, 0.0

    h, w = depth.shape
    fx, fy, cx, cy = scaled_intrinsics(intrinsics, w, h)
    heights = height_above_camera(unproject(depth, fx, fy, cx, cy), g_camera)

    usable = np.zeros_like(depth, dtype=bool)
    usable[int(h * lower_frame_fraction):, :] = True
    usable &= np.isfinite(depth) & (depth > near_m) & (depth < far_m)
    usable &= np.isfinite(heights)
    if usable.sum() < min_samples:
        return None, 0.0

    vals = heights[usable]
    lo, hi = np.percentile(vals, 1.0), np.percentile(vals, 60.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None, 0.0

    hist, edges = np.histogram(vals, bins=200, range=(float(lo), float(hi)))
    offset = float(edges[int(np.argmax(hist))] + 0.5 * (edges[1] - edges[0]))

    # Two refinement passes: the histogram gets us inside the ground blob, the
    # medians centre us on it without letting a distant slope drag the answer.
    inliers = vals[np.abs(vals - offset) < band_m]
    for _ in range(2):
        if inliers.size < min_samples // 2:
            break
        offset = float(np.median(inliers))
        inliers = vals[np.abs(vals - offset) < band_m]

    confidence = float(inliers.size) / float(usable.sum())
    return -offset, confidence
