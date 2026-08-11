"""Against the actual files in testphotos/, when they are there.

Everything else in the suite runs on fixtures so CI stays hermetic. This module
is the one that touches real pixels and real decoders, and it skips cleanly on
a machine that does not have the photos — which includes CI.

    IMG_7096.HEIC   14mm ultra-wide, portrait, pitched up 3.3 degrees
    IMG_7263.DNG    same lens, near-level, ProRAW (DNG 1.7, JPEG XL)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from server import exif as exif_mod
from server import raw as raw_mod

PHOTOS = Path(__file__).resolve().parents[2] / "testphotos"
HEIC = PHOTOS / "IMG_7096.HEIC"
DNG = PHOTOS / "IMG_7263.DNG"

pytestmark = pytest.mark.skipif(
    not (HEIC.exists() and DNG.exists()),
    reason="testphotos/ not present (they are gitignored — photos are large and personal)")


@pytest.fixture(scope="module")
def heic_state():
    return exif_mod.camera_state_from_path(HEIC)


@pytest.fixture(scope="module")
def dng_state():
    return exif_mod.camera_state_from_path(DNG)


# ---------------------------------------------------------------------------
# the solve, on real metadata read by a real exiftool
# ---------------------------------------------------------------------------

def test_heic_solves_to_the_documented_pose(heic_state):
    """The M1 acceptance criterion, against the file rather than a fixture."""
    assert heic_state.orientation.pitch_deg == pytest.approx(3.3, abs=0.5)
    assert heic_state.orientation.roll_deg == pytest.approx(0.7, abs=0.5)
    assert heic_state.geo.heading_deg == pytest.approx(76.9, abs=0.1)
    assert heic_state.intrinsics.focal_35mm == 14.0
    assert (heic_state.intrinsics.width, heic_state.intrinsics.height) == (6048, 8064)


def test_neither_photo_produces_a_warning(heic_state, dng_state):
    """Unmodified originals should solve clean. A warning here means a tag this
    pipeline depends on is missing or ambiguous on a real file."""
    assert heic_state.warnings == []
    assert dng_state.warnings == []


def test_dng_is_a_near_level_camera(dng_state):
    """IMG_7263 is close to the level-floor calibration frame the handoff asks
    for: if the phone really was upright, pitch and roll must be near zero."""
    assert abs(dng_state.orientation.pitch_deg) < 2.0
    assert abs(dng_state.orientation.roll_deg) < 2.0
    assert dng_state.orientation.is_static


def test_both_photos_agree_on_the_lens(heic_state, dng_state):
    """Same ultra-wide module, so the intrinsics must come out identical even
    though one file is HEIC and the other is a DNG with a different tag
    layout."""
    assert heic_state.intrinsics.fx == pytest.approx(dng_state.intrinsics.fx)
    assert heic_state.intrinsics.width == dng_state.intrinsics.width


def test_semantic_mask_subifd_does_not_win_the_size_lookup(dng_state):
    """The DNG carries EXIF:SubIFD:ImageWidth = 8064 (the raw) and
    EXIF:SubIFD1:ImageWidth = 2016 (a semantic mask). A prefix match on
    "EXIF:SubIFD" hits both, and picking the mask would scale the whole solve
    by a third."""
    assert exif_mod.stored_image_size(exif_mod.read_exif(DNG)) == (8064, 6048)


def test_sun_is_solved_for_both(heic_state, dng_state):
    """Bernese Oberland, 14 July, late morning: high and to the south-east."""
    for state in (heic_state, dng_state):
        assert state.sun is not None and state.sun.above_horizon
        assert 40.0 < state.sun.elevation_deg < 70.0
        assert 90.0 < state.sun.azimuth_deg < 200.0

    # The DNG was shot 1h32m later, so its sun must be higher and further west.
    assert dng_state.sun.elevation_deg > heic_state.sun.elevation_deg
    assert dng_state.sun.azimuth_deg > heic_state.sun.azimuth_deg


def test_camera_rotation_is_buildable_from_the_real_solve(heic_state):
    from photo3d import coords

    matrix = coords.camera_matrix(heic_state.orientation.gravity_camera,
                                  heic_state.geo.heading_deg, 1.6)
    rotation = matrix[:3, :3]
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
    # Blender's camera looks down local -Z; that has to face the compass.
    forward = rotation @ np.array([0.0, 0.0, -1.0])
    assert coords.azimuth_deg(forward) == pytest.approx(76.944, abs=0.01)


# ---------------------------------------------------------------------------
# plates, on real pixels
# ---------------------------------------------------------------------------

def test_heic_opens_and_is_display_oriented(tmp_path):
    """Blender cannot open HEIC, so it has to become a PNG — and the PNG has to
    come out portrait, matching the intrinsics. If it does not, the
    Window-projected reprojection lands rotated."""
    pytest.importorskip("pillow_heif")
    plates = raw_mod.make_plates(HEIC, tmp_path, want_lighting_plate=False)
    assert (plates.width, plates.height) == (6048, 8064)
    assert Path(plates.display_png).is_file()


def test_dng_needs_a_decoder_libraw_does_not_have():
    """LibRaw 0.22 returns "Unsupported file format" on DNG 1.7 with JPEG XL.
    Pinned so that a future rawpy release quietly fixing it is noticed rather
    than assumed."""
    rawpy = pytest.importorskip("rawpy")
    with pytest.raises(Exception) as caught:
        raw_mod._develop_linear_rawpy(DNG)
    assert "nsupported" in str(caught.value) or "not RAW" in str(caught.value)


def test_linear_develop_has_real_highlight_headroom():
    """The entire reason for the DNG path. A display-referred plate clips
    everything above diffuse white, and those clipped regions carry most of the
    scene's light energy."""
    pytest.importorskip("Quartz", reason="pip install pyobjc-framework-Quartz")
    linear = raw_mod.develop_linear(DNG, long_edge=1024)

    assert linear.dtype == np.float32
    assert linear.max() > 1.5, "no headroom means the tone curve was applied"
    assert linear.max() / max(float(np.median(linear)), 1e-9) > 4.0
    assert (linear > 1.0).mean() > 0.0005, "some pixels must exceed diffuse white"


def test_both_plates_are_written_for_a_raw_source(tmp_path):
    pytest.importorskip("Quartz")
    plates = raw_mod.make_plates(DNG, tmp_path, exif=exif_mod.read_exif(DNG))

    assert plates.source_is_raw
    assert (plates.width, plates.height) == (6048, 8064), "display-oriented, portrait"
    assert Path(plates.display_png).is_file()
    if plates.lighting_exr is None:
        pytest.skip(f"no EXR backend installed: {plates.note}")
    assert Path(plates.lighting_exr).is_file()
    assert "peaks at" in plates.note


def test_this_dng_carries_no_warp_opcodes():
    """Apple's ProRAW here is PhotometricInterpretation "Linear Raw" — already
    demosaiced and geometrically corrected — so there is no WarpRectilinear to
    apply and no distortion to reintroduce. See docs/DECISIONS.md section 1;
    this pins the finding so the claim is not taken on trust."""
    result = raw_mod.warp_opcodes(exif_mod.read_exif(DNG))
    assert not result.present
    assert result.names == []
