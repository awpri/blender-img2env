"""Milestone M1 — EXIF solve, no ML at all.

Acceptance, straight from the handoff: on the reference photo the solve returns
pitch 3.3 +/- 0.5, roll 0.7 +/- 0.5, heading 76.9, focal 14 mm.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from server import exif as exif_mod
from server.exif import Photo3DError


# ---------------------------------------------------------------------------
# M1 acceptance
# ---------------------------------------------------------------------------

def test_m1_acceptance(img7096):
    state = exif_mod.camera_state_from_exif(img7096)

    assert state.intrinsics.focal_35mm == pytest.approx(14.0)
    assert state.orientation is not None, "AccelerationVector must be parsed"
    assert state.orientation.pitch_deg == pytest.approx(3.3, abs=0.5)
    assert state.orientation.roll_deg == pytest.approx(0.7, abs=0.5)
    assert state.geo.heading_deg == pytest.approx(76.9, abs=0.1)
    assert state.geo.heading_is_true


def test_reference_photo_is_portrait_in_display_space(img7096):
    """Stored 8064x6048 with Orientation 6 means a 6048x8064 image on screen.
    Depth, the proxy mesh and the Window-projected plate all live in display
    space, so the intrinsics have to as well."""
    k = exif_mod.intrinsics_from_exif(img7096)
    assert (k.width, k.height) == (6048, 8064)
    assert k.cx == pytest.approx(3024.0)
    assert k.cy == pytest.approx(4032.0)


def test_thumbnail_ifd_does_not_win_the_size_lookup(img7096):
    """A DNG's IFD0 describes the embedded thumbnail. Picking it up gives a
    256 px wide 'photo' and a focal length 30x too short."""
    assert exif_mod.stored_image_size(img7096) == (8064, 6048)


def test_focal_in_pixels(img7096):
    k = exif_mod.intrinsics_from_exif(img7096)
    assert k.fx == pytest.approx(14.0 / 36.0 * 8064)
    assert k.fx == k.fy, "square pixels; no anamorphic phones yet"
    # 14 mm ultra-wide. The frame is portrait, so the 104-degree long-edge
    # field is the VERTICAL one; across the frame it is only 88. That much
    # depth range in one shot is where monocular depth is weakest, which is
    # what the far clamp in coords.proxy_arrays() is for.
    assert k.vertical_fov_deg == pytest.approx(104.25, abs=0.5)
    assert k.horizontal_fov_deg == pytest.approx(87.9, abs=0.5)


def test_focal_conventions_differ_by_a_few_percent(img7096):
    """Documented ambiguity, not a bug — see docs/DECISIONS.md. The test exists
    so the gap stays visible and nobody 'fixes' it by nudging focal_override."""
    long_edge = exif_mod.intrinsics_from_exif(img7096, convention="long_edge")
    diagonal = exif_mod.intrinsics_from_exif(img7096, convention="diagonal")
    ratio = diagonal.fx / long_edge.fx
    assert 1.02 < ratio < 1.06
    assert long_edge.width == diagonal.width


def test_focal_override_wins(img7096):
    k = exif_mod.intrinsics_from_exif(img7096, focal_override=24.0)
    assert k.focal_35mm == pytest.approx(24.0)


def test_missing_focal_is_an_actionable_error(img7096):
    stripped = {k: v for k, v in img7096.items() if "FocalLengthIn35mm" not in k}
    with pytest.raises(Photo3DError, match="focal_override"):
        exif_mod.intrinsics_from_exif(stripped)


def test_unknown_focal_convention_is_refused(img7096):
    with pytest.raises(Photo3DError, match="focal convention"):
        exif_mod.intrinsics_from_exif(img7096, convention="diagonalish")


# ---------------------------------------------------------------------------
# the four-orientation calibration set
# ---------------------------------------------------------------------------

def test_all_four_orientations_agree(orientation_set):
    """The handoff's calibration requirement: the same scene shot in all four
    orientations has to solve to the same camera. Disagreement here means the
    device-axis map is wrong for at least one branch, which is exactly the bug
    that would otherwise only show up as a scene tipped 90 degrees."""
    solved = {o: exif_mod.orientation_from_gravity(e) for o, e in orientation_set.items()}
    assert all(s is not None for s in solved.values())

    pitches = [s.pitch_deg for s in solved.values()]
    rolls = [s.roll_deg for s in solved.values()]
    assert max(pitches) - min(pitches) < 0.25
    assert max(rolls) - min(rolls) < 0.25
    assert pitches[0] == pytest.approx(3.28, abs=0.1)


def test_orientation_is_read_per_file(orientation_set):
    for expected, record in orientation_set.items():
        assert exif_mod.orientation_from_gravity(record).exif_orientation == expected


def test_landscape_orientations_are_not_transposed(orientation_set):
    portrait = exif_mod.intrinsics_from_exif(orientation_set[6])
    landscape = exif_mod.intrinsics_from_exif(orientation_set[1])
    assert (portrait.width, portrait.height) == (6048, 8064)
    assert (landscape.width, landscape.height) == (8064, 6048)
    assert portrait.fx == pytest.approx(landscape.fx), "same lens, same pixels"


# ---------------------------------------------------------------------------
# accelerometer edge cases
# ---------------------------------------------------------------------------

def test_missing_accelerometer_is_a_warning_not_a_crash(img7096):
    stripped = {k: v for k, v in img7096.items() if "AccelerationVector" not in k}
    state = exif_mod.camera_state_from_exif(stripped)
    assert state.orientation is None
    assert any("AccelerationVector" in w for w in state.warnings)
    assert state.intrinsics.focal_35mm == 14.0, "the rest of the solve survives"


def test_motion_during_capture_is_flagged(img7096):
    moving = dict(img7096)
    moving["MakerNotes:Apple:AccelerationVector"] = "-0.01226 -1.42264 0.05630"
    state = exif_mod.camera_state_from_exif(moving)
    assert not state.orientation.is_static
    assert any("not 1.000" in w for w in state.warnings)


def test_reference_capture_counts_as_static(img7096):
    state = exif_mod.camera_state_from_exif(img7096)
    assert state.orientation.magnitude_g == pytest.approx(0.984, abs=0.002)
    assert state.orientation.is_static


def test_acceleration_vector_accepts_a_json_list(img7096):
    as_list = dict(img7096)
    as_list["MakerNotes:Apple:AccelerationVector"] = [-0.01226, -0.98264, 0.05630]
    assert exif_mod.orientation_from_gravity(as_list).pitch_deg == pytest.approx(3.28, abs=0.01)


# ---------------------------------------------------------------------------
# GPS, time, sun
# ---------------------------------------------------------------------------

def test_stripped_coordinates_produce_the_specific_warning(img7096):
    """The reference photo has GPSLatitudeRef with no GPSLatitude. That exact
    shape means 'Photos stripped the location on export', and the message has
    to say so, because the fix is in the export dialog, not in this code."""
    geo = exif_mod.geo_from_exif(img7096)
    assert geo.latitude is None
    assert any("Export Unmodified Original" in w for w in geo.warnings)
    assert exif_mod.sun_from_geo(geo) is None


def test_gps_timestamp_is_read_as_utc(img7096):
    geo = exif_mod.geo_from_exif(img7096)
    assert geo.utc == datetime(2026, 7, 14, 9, 11, 36, tzinfo=timezone.utc)


def test_local_time_plus_offset_falls_back_to_the_same_utc(img7096):
    """11:11:36 +02:00 is 09:11:36 UTC. Subtracting when it should add puts the
    sun four hours away, which is a completely different lighting setup."""
    no_gps_time = {k: v for k, v in img7096.items()
                   if "GPSDateStamp" not in k and "GPSTimeStamp" not in k
                   and "GPSDateTime" not in k}
    geo = exif_mod.geo_from_exif(no_gps_time)
    assert geo.utc == datetime(2026, 7, 14, 9, 11, 36, tzinfo=timezone.utc)


def test_timezone_free_timestamp_warns(img7096):
    naive = {k: v for k, v in img7096.items()
             if "GPS" not in k and "OffsetTime" not in k}
    geo = exif_mod.geo_from_exif(naive)
    assert any("assumed UTC" in w for w in geo.warnings)


def test_magnetic_heading_is_flagged(img7096):
    magnetic = dict(img7096)
    magnetic["EXIF:GPS:GPSImgDirectionRef"] = "M"
    geo = exif_mod.geo_from_exif(magnetic)
    assert not geo.heading_is_true
    assert any("MAGNETIC" in w for w in geo.warnings)


def test_sun_is_solved_when_coordinates_survive(geneva):
    """Geneva, 21 June 2026, 10:00 UTC (midsummer, late morning). The sun should
    be well up and to the east-southeast."""
    state = exif_mod.camera_state_from_exif(geneva)
    assert state.sun is not None
    assert state.sun.above_horizon
    assert 45.0 < state.sun.elevation_deg < 60.0
    assert 100.0 < state.sun.azimuth_deg < 145.0


def test_full_state_serialises(img7096):
    """The daemon returns this over HTTP, so it has to be JSON-shaped."""
    import json
    payload = exif_mod.camera_state_from_exif(img7096).as_dict()
    json.dumps(payload)
    assert payload["pitch_deg"] == pytest.approx(3.28, abs=0.05)
    assert payload["heading_deg"] == pytest.approx(76.944)
    assert payload["sun"] is None


# ---------------------------------------------------------------------------
# the tag lookup itself
# ---------------------------------------------------------------------------

def test_find_prefers_the_requested_group():
    record = {"EXIF:IFD0:ImageWidth": 256, "EXIF:SubIFD:ImageWidth": 8064}
    assert exif_mod.find(record, "ImageWidth", prefer=("EXIF:SubIFD",)) == 8064
    assert exif_mod.find(record, "ImageWidth", prefer=("EXIF:IFD0",)) == 256
    assert exif_mod.find(record, "NotATag") is None


def test_find_ignores_group_prefixes_when_matching_names():
    assert exif_mod.find({"MakerNotes:Apple:HDRHeadroom": 1.01}, "HDRHeadroom") == 1.01


@pytest.mark.parametrize("value,expected", [
    ("-0.01226 -0.98264 0.05630", [-0.01226, -0.98264, 0.05630]),
    ("-0.01226, -0.98264, 0.05630", [-0.01226, -0.98264, 0.05630]),
    ([1, 2, 3], [1.0, 2.0, 3.0]),
    (7, [7.0]),
    ("not a number", None),
    (None, None),
])
def test_float_list_parsing(value, expected):
    assert exif_mod._floats(value) == expected


def test_horizontal_and_vertical_fov_are_consistent(img7096):
    k = exif_mod.intrinsics_from_exif(img7096)
    expected_v = math.degrees(2 * math.atan(k.height / (2 * k.fy)))
    assert k.vertical_fov_deg == pytest.approx(expected_v)
    assert k.vertical_fov_deg > k.horizontal_fov_deg, "portrait frame"
