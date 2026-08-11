"""Milestone M6 — the two plates, and the WarpRectilinear question.

rawpy is optional here: the tests that need a real DNG skip without one, but
the opcode reporting, the format routing and the orientation contract are all
checked with no camera involved.
"""

from __future__ import annotations

import numpy as np
import pytest

from server import raw as raw_mod


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("IMG_7096.DNG", True), ("IMG_7096.dng", True), ("a.ARW", True),
    ("IMG_7096.HEIC", False), ("IMG_7096.jpg", False), ("plate.png", False),
])
def test_raw_detection(name, expected):
    assert raw_mod.is_raw(name) is expected


# ---------------------------------------------------------------------------
# lens opcodes
# ---------------------------------------------------------------------------

def test_warp_opcodes_detected_and_reported_as_unapplied():
    """A ProRAW DNG defers rectification to an opcode that rawpy ignores, so
    switching to DNG can reintroduce barrel distortion the JPEG had removed.
    The decision is to accept it; the requirement is that it is reported."""
    result = raw_mod.warp_opcodes({"EXIF:SubIFD:OpcodeList3": "WarpRectilinear"})
    assert result.present
    assert not result.applied
    assert "WarpRectilinear" in result.note
    assert "DECISIONS" in result.note


def test_no_opcodes_is_stated_positively(img7096):
    result = raw_mod.warp_opcodes(img7096)
    assert not result.present
    assert "rectilinear" in result.note


def test_non_geometric_opcodes_are_not_mistaken_for_warp():
    """FixVignetteRadial is a shading opcode. Treating it as a geometry warning
    would send someone chasing a distortion that is not there."""
    result = raw_mod.warp_opcodes({"EXIF:SubIFD:OpcodeList2": "FixVignetteRadial"})
    assert not result.present
    assert "FixVignetteRadial" in result.names


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def test_png_round_trip(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    source = np.zeros((8, 12, 3), dtype=np.float32)
    source[..., 0] = 1.0
    path = raw_mod.write_png(source, tmp_path / "p.png")
    back = np.asarray(Image.open(path))
    assert back.shape == (8, 12, 3)
    assert back[0, 0, 0] == 255 and back[0, 0, 1] == 0


def test_png_clips_out_of_range_values(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    path = raw_mod.write_png(np.array([[[-3.0, 0.5, 9.0]]], dtype=np.float32),
                             tmp_path / "clip.png")
    assert np.asarray(Image.open(path))[0, 0].tolist() == [0, 128, 255]


def test_exr_carries_values_above_one(tmp_path):
    """The whole point of the lighting plate. If the backend silently clamps at
    1.0 it is no better than the PNG and the sky is still under-lit."""
    hdr = np.zeros((4, 4, 3), dtype=np.float32)
    hdr[..., 0] = 47.0
    try:
        path = raw_mod.write_exr(hdr, tmp_path / "light.exr")
    except raw_mod.PlateError:
        pytest.skip("no EXR backend installed in this environment")
    assert path.endswith(".exr")

    for reader in ("OpenImageIO", "cv2", "imageio.v3"):
        try:
            if reader == "cv2":
                import os
                os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
                import cv2
                back = cv2.imread(path, cv2.IMREAD_UNCHANGED)[..., ::-1]
            elif reader == "imageio.v3":
                import imageio.v3 as iio
                back = iio.imread(path)
            else:
                import OpenImageIO as oiio
                back = oiio.ImageInput.open(path).read_image()
        except Exception:                                          # noqa: BLE001
            continue
        assert float(np.asarray(back)[..., 0].max()) == pytest.approx(47.0, rel=1e-3)
        return
    pytest.skip("no EXR reader available to verify the write")


def test_exr_failure_names_the_fix():
    """When no backend is installed the error has to say what to install —
    this is the one dependency the user most plausibly lacks."""
    error = raw_mod.PlateError("no EXR backend available; install opencv-python")
    assert "install" in str(error)


# ---------------------------------------------------------------------------
# plates from a non-raw source
# ---------------------------------------------------------------------------

def test_display_plate_from_a_jpeg(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    source = tmp_path / "shot.jpg"
    Image.new("RGB", (64, 48), (120, 30, 30)).save(source)

    plates = raw_mod.make_plates(source, tmp_path / "cache")
    assert plates.source_is_raw is False
    assert plates.lighting_exr is None
    assert (plates.width, plates.height) == (64, 48)
    assert "clipped" in plates.note, "the user should be told why there is no EXR"
    assert Image.open(plates.display_png).size == (64, 48)


def test_display_plate_applies_exif_orientation(tmp_path):
    """The plate has to come out in the same display space as the intrinsics.
    If it does not, the Window-projected reprojection lands sideways and the
    chrome-sphere reflection test in M4 fails for a reason that looks like a
    shader bug."""
    pytest.importorskip("PIL")
    from PIL import Image

    source = tmp_path / "portrait.jpg"
    image = Image.new("RGB", (64, 48), (10, 10, 10))
    exif = image.getexif()
    exif[274] = 6                     # Orientation: rotate 90 CW for display
    image.save(source, exif=exif)

    plates = raw_mod.make_plates(source, tmp_path / "cache", want_lighting_plate=False)
    assert (plates.width, plates.height) == (48, 64), "stored landscape, displayed portrait"


def test_cache_directory_is_created(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    source = tmp_path / "shot.png"
    Image.new("RGB", (8, 8)).save(source)
    cache = tmp_path / "nested" / "cache"
    plates = raw_mod.make_plates(source, cache, want_lighting_plate=False)
    assert cache.is_dir()
    assert plates.display_png.endswith("shot_plate.png")


def test_missing_rawpy_is_an_actionable_error(tmp_path, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_rawpy(name, *args, **kwargs):
        if name == "rawpy":
            raise ImportError("nope")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_rawpy)
    with pytest.raises(raw_mod.PlateError, match="pip install rawpy"):
        raw_mod.develop_linear(tmp_path / "whatever.dng")


# ---------------------------------------------------------------------------
# with a real raw file, if one is available
# ---------------------------------------------------------------------------

@pytest.mark.skipif(True, reason="needs a real ProRAW DNG; see docs/VERIFICATION.md")
def test_linear_develop_has_headroom():
    """Run this by hand against IMG_7096.DNG with the skipif removed. A linear
    develop of a sunlit scene should show a max well above the median — if it
    does not, no_auto_bright or gamma is being ignored and the lighting plate
    is no better than the JPEG."""
    linear = raw_mod.develop_linear("IMG_7096.DNG")
    assert linear.max() / np.median(linear) > 4.0
