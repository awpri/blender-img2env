"""Milestone M3 — depth to proxy mesh, on synthetic depth with a known answer.

The acceptance test for M3 is a cube resting within 15% of ground truth. That
needs a photo, a solver and a physics step. What can be checked here is the
part that would cause such a failure: whether a depth map of a floor at a known
height, seen through known intrinsics, solves back to that height. If this
passes and the cube still lands wrong, the bug is in Blender, not in the maths.
"""

from __future__ import annotations

import numpy as np
import pytest

from photo3d import coords
from server import depth as depth_mod


def synthetic_intrinsics(width=320, height=240, focal_35mm=14.0) -> dict:
    f_px = focal_35mm / 36.0 * max(width, height)
    return {"width": width, "height": height, "fx": f_px, "fy": f_px,
            "cx": width / 2.0, "cy": height / 2.0}


def ground_plane_depth(intrinsics, camera_height, pitch_deg=0.0, roll_deg=0.0,
                       far_m=1e4):
    """Render a z-depth map of an infinite level floor.

    Inverts the pinhole projection analytically, so the only thing under test
    is the recovery, not some other approximation. Pixels whose ray points at
    or above the horizon get `far_m` — that is what the sky looks like to a
    depth model, and it is exactly the input that used to drag a mode-finder
    off the ground.
    """
    g_cam = coords.gravity_from_pitch_roll(pitch_deg, roll_deg)
    up = -g_cam
    h, w = intrinsics["height"], intrinsics["width"]
    fx, fy, cx, cy = (intrinsics["fx"], intrinsics["fy"],
                      intrinsics["cx"], intrinsics["cy"])

    vv, uu = np.mgrid[0:h, 0:w].astype(np.float64)
    rays = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], axis=-1)
    # point = z * ray, and the floor is where point . up == -camera_height
    denominator = rays @ up
    with np.errstate(divide="ignore", invalid="ignore"):
        z = -camera_height / denominator
    z[(denominator >= -1e-9) | ~np.isfinite(z) | (z <= 0)] = far_m
    return np.minimum(z, far_m).astype(np.float32), g_cam


# ---------------------------------------------------------------------------
# unprojection
# ---------------------------------------------------------------------------

def test_unprojection_puts_the_principal_ray_on_the_optical_axis():
    k = synthetic_intrinsics()
    pts = coords.unproject(np.full((240, 320), 7.0), k["fx"], k["fy"], k["cx"], k["cy"])
    np.testing.assert_allclose(pts[120, 160], [0.0, 0.0, 7.0], atol=1e-9)


def test_unprojection_respects_the_field_of_view():
    """A point one focal length off-axis in pixels must sit at 45 degrees.

    Sampled at an integer column, so the expected offset is that column's true
    distance from the principal point rather than a round 1.0.
    """
    k = synthetic_intrinsics()
    pts = coords.unproject(np.ones((240, 320)), k["fx"], k["fy"], k["cx"], k["cy"])
    column = int(k["cx"] + k["fx"])
    edge = pts[120, column]
    assert edge[0] == pytest.approx((column - k["cx"]) / k["fx"])
    assert edge[0] == pytest.approx(1.0, abs=0.01)
    assert edge[2] == pytest.approx(1.0)


def test_intrinsics_rescale_to_the_depth_raster():
    """Depth runs at ~1.5K while the plate is 8K. Forgetting to rescale scales
    the entire reconstruction, which reads as 'the depth model is bad'."""
    k = synthetic_intrinsics(width=6048, height=8064)
    fx, fy, cx, cy = coords.scaled_intrinsics(k, 1152, 1536)
    assert fx == pytest.approx(k["fx"] * 1152 / 6048)
    assert cx == pytest.approx(576.0)
    assert cy == pytest.approx(768.0)


# ---------------------------------------------------------------------------
# the ground-plane solve
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("height", [1.2, 1.55, 1.8, 12.0])
@pytest.mark.parametrize("pitch", [-20.0, -8.0, 0.0, 3.3])
def test_recovers_camera_height_from_a_flat_floor(height, pitch):
    k = synthetic_intrinsics()
    depth, g_cam = ground_plane_depth(k, height, pitch_deg=pitch)
    solved, confidence = coords.ground_plane_height(depth, k, g_cam)
    assert solved == pytest.approx(height, rel=0.02)
    assert confidence > 0.3


def test_roll_does_not_change_the_recovered_height():
    """The plane's normal comes from gravity, so a banked camera must still
    measure the same floor. If this fails, roll is being applied twice."""
    k = synthetic_intrinsics()
    heights = []
    for roll in (-15.0, 0.0, 15.0):
        depth, g_cam = ground_plane_depth(k, 1.6, pitch_deg=-5.0, roll_deg=roll)
        heights.append(coords.ground_plane_height(depth, k, g_cam)[0])
    assert max(heights) - min(heights) < 0.02


def test_survives_clutter_on_the_floor():
    """Boxes standing on the ground must not drag the estimate up. The mode is
    the ground because the ground is what most of the lower frame is."""
    k = synthetic_intrinsics()
    depth, g_cam = ground_plane_depth(k, 1.6, pitch_deg=-10.0)
    depth[150:200, 40:110] *= 0.55       # a nearer object occluding the floor
    depth[190:230, 200:260] *= 0.75
    solved, _ = coords.ground_plane_height(depth, k, g_cam)
    assert solved == pytest.approx(1.6, rel=0.05)


def test_sky_only_frame_reports_no_ground():
    k = synthetic_intrinsics()
    solved, confidence = coords.ground_plane_height(np.full((240, 320), 1e4, np.float32),
                                                    k, [0.0, 1.0, 0.0])
    assert solved is None
    assert confidence == 0.0


def test_no_gravity_means_no_ground_solve():
    k = synthetic_intrinsics()
    assert coords.ground_plane_height(np.ones((240, 320), np.float32), k, None) == (None, 0.0)


def test_solve_ground_falls_back_and_says_so():
    k = synthetic_intrinsics()
    result = depth_mod.solve_ground(np.ones((240, 320), np.float32), k, None,
                                    fallback_height_m=1.55)
    assert result.camera_height_m == 1.55
    assert result.confidence == 0.0
    assert "assumed" in result.source

    depth, g_cam = ground_plane_depth(k, 1.7, pitch_deg=-6.0)
    good = depth_mod.solve_ground(depth, k, g_cam)
    assert good.camera_height_m == pytest.approx(1.7, rel=0.02)
    assert "gravity" in good.source


def test_ground_above_camera_is_rejected_rather_than_returned_negative():
    """A ceiling-dominant frame used to return a negative height, which then
    put the Blender camera underground."""
    k = synthetic_intrinsics()
    depth, g_cam = ground_plane_depth(k, 1.6, pitch_deg=0.0)
    result = depth_mod.solve_ground(depth, k, -np.asarray(g_cam))   # gravity inverted
    assert result.camera_height_m > 0.0
    assert "assumed" in result.source


# ---------------------------------------------------------------------------
# discontinuity culling — the single biggest 'looks fake' fix
# ---------------------------------------------------------------------------

def test_culls_quads_that_straddle_a_depth_cliff():
    depth = np.ones((8, 8), dtype=np.float32) * 2.0
    depth[:, 4:] = 40.0                       # foreground object against a hillside
    keep = coords.cull_discontinuous_quads(depth, edge_threshold=0.08)
    assert keep[:, :3].all(), "flat regions must survive"
    assert keep[:, 4:].all()
    assert not keep[:, 3].any(), "the straddling column is the rubber sheet"


def test_culling_is_scale_free():
    """The same relative step must be culled at 2 m and at 200 m — that is why
    the test is on disparity spread and not on metric difference."""
    near = np.ones((4, 4), np.float32) * 2.0
    near[:, 2:] = 2.0 * 1.5
    far = near * 100.0
    threshold = 0.08
    assert np.array_equal(coords.cull_discontinuous_quads(near, threshold),
                          coords.cull_discontinuous_quads(far, threshold))


def test_a_smooth_gradient_survives_culling():
    depth = np.linspace(2.0, 6.0, 64, dtype=np.float32)[None, :].repeat(64, axis=0)
    assert coords.cull_discontinuous_quads(depth, 0.08).all()


# ---------------------------------------------------------------------------
# the proxy mesh arrays
# ---------------------------------------------------------------------------

def test_proxy_arrays_are_in_blender_camera_space():
    """Blender's camera looks down -Z, so every vertex in front of the lens must
    have a negative z. A positive one means the mesh is behind the camera and
    the viewport looks empty."""
    k = synthetic_intrinsics()
    depth, _ = ground_plane_depth(k, 1.6, pitch_deg=-10.0)
    verts, quads = coords.proxy_arrays(depth, k, mesh_long_edge=64, max_depth=60.0)
    assert (verts[:, 2] < 0).all()
    assert verts.shape[1] == 3 and quads.shape[1] == 4
    assert quads.max() < len(verts)


def test_proxy_face_normals_point_back_at_the_camera():
    """Reverse the winding and the proxy becomes a one-sided surface that
    catches no shadows and reflects nothing. Checked here because it is
    invisible in a wireframe viewport."""
    k = synthetic_intrinsics(width=64, height=64)
    verts, quads = coords.proxy_arrays(np.full((64, 64), 5.0, np.float32), k,
                                       mesh_long_edge=64, max_depth=60.0)
    a, b, c = verts[quads[0, 0]], verts[quads[0, 1]], verts[quads[0, 2]]
    normal = np.cross(b - a, c - b)
    assert normal[2] > 0, "normal must face +Z, back toward the camera at the origin"


def test_proxy_resolution_is_capped_by_mesh_long_edge():
    k = synthetic_intrinsics(width=512, height=512)
    verts, _ = coords.proxy_arrays(np.full((512, 512), 5.0, np.float32), k,
                                   mesh_long_edge=128)
    assert len(verts) == 128 * 128


def test_far_clamp_bounds_the_mesh():
    """14 mm frames run to the horizon and metric depth is meaningless out
    there. Unclamped, a single sky pixel puts a vertex kilometres away and
    every viewport navigation operation becomes useless."""
    k = synthetic_intrinsics()
    depth, _ = ground_plane_depth(k, 1.6, pitch_deg=5.0, far_m=1e4)
    verts, _ = coords.proxy_arrays(depth, k, mesh_long_edge=64, max_depth=120.0)
    assert np.abs(verts[:, 2]).max() <= 120.0 + 1e-6


def test_nan_depth_does_not_reach_the_mesh():
    k = synthetic_intrinsics()
    depth = np.full((64, 64), 5.0, np.float32)
    depth[10:20, 10:20] = np.nan
    depth[30, 30] = np.inf
    verts, _ = coords.proxy_arrays(depth, k, mesh_long_edge=64)
    assert np.isfinite(verts).all()


def test_clamp_far_field_removes_non_finite_values():
    dirty = np.array([[np.nan, np.inf, -np.inf], [0.0, 5.0, 5000.0]], dtype=np.float32)
    clean = depth_mod.clamp_far_field(dirty, far_m=120.0, near_m=0.05)
    assert np.isfinite(clean).all()
    assert clean.max() <= 120.0 and clean.min() >= 0.05


def test_depth_statistics_ignores_non_finite():
    stats = depth_mod.depth_statistics(np.array([1.0, 2.0, np.nan, 100.0], np.float32))
    assert stats["min_m"] == 1.0
    assert stats["max_m"] == 100.0
    assert depth_mod.depth_statistics(np.array([np.nan], np.float32))["min_m"] is None


# ---------------------------------------------------------------------------
# Marigold's affine fit
# ---------------------------------------------------------------------------

def test_relative_depth_fits_onto_metric_scale():
    """Marigold has no scale and no offset. The fit has to recover both, and it
    only works because it is done in disparity space."""
    metric = np.linspace(2.0, 50.0, 4096).reshape(64, 64).astype(np.float64)
    relative = 1.0 / metric * 3.7 - 0.42          # an arbitrary affine disparity
    fitted = depth_mod.fit_relative_to_metric(relative, metric)
    np.testing.assert_allclose(fitted, metric, rtol=1e-3)


def test_affine_fit_declines_when_there_is_nothing_to_fit_to():
    metric = np.full((8, 8), 300.0)               # everything beyond the valid band
    np.testing.assert_allclose(depth_mod.fit_relative_to_metric(np.ones((8, 8)), metric),
                               metric)
