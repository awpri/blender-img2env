"""Milestone M7 — shadow extraction.

compphoto/Intrinsic is a heavy optional dependency, so what is tested here is
the contract around it: that the fallback engages cleanly, that the caller can
always tell which method produced the mask, and that a mask lands on disk in a
form the add-on can consume.
"""

from __future__ import annotations

import numpy as np
import pytest

from server import shading as shading_mod


@pytest.fixture
def dappled_ground():
    """Uniform grass with a fine bright/dark pattern — larch dapple, roughly."""
    rng = np.random.default_rng(7)
    ground = np.full((128, 128, 3), 0.45, dtype=np.float32)
    ground += rng.normal(0.0, 0.01, ground.shape).astype(np.float32)
    for cy, cx in [(30, 40), (70, 90), (100, 25), (55, 60)]:
        ground[cy - 6:cy + 6, cx - 6:cx + 6] *= 0.35
    return np.clip(ground, 0.0, 1.0)


def test_falls_back_to_retinex_and_says_why(dappled_ground):
    """The intrinsic model is almost certainly not installed in CI. That must
    produce a usable mask plus an explanation, not an exception."""
    mask, method, note = shading_mod.shadow_mask(dappled_ground, blur_px=61,
                                                 prefer_intrinsic=True)
    assert method in {"intrinsic", "retinex"}
    assert mask.shape == dappled_ground.shape[:2]
    if method == "retinex":
        assert "fell back" in note
        assert "dark albedo" in note, "the caller needs the caveat, not just the fact"


def test_explicit_retinex_skips_the_model_entirely(dappled_ground):
    mask, method, note = shading_mod.shadow_mask(dappled_ground, blur_px=61,
                                                 prefer_intrinsic=False)
    assert method == "retinex"
    assert "explicitly" in note
    assert mask.min() >= 0.15


def test_mask_finds_the_dapple(dappled_ground):
    mask, _, _ = shading_mod.shadow_mask(dappled_ground, blur_px=61,
                                         prefer_intrinsic=False)
    assert mask[30, 40] < 0.75, "a dappled patch should read as shade"
    assert mask[10, 110] > 0.9, "open ground should read as lit"


def test_mask_is_bounded(dappled_ground):
    mask, _, _ = shading_mod.shadow_mask(dappled_ground, contrast=6.0, floor=0.15,
                                         prefer_intrinsic=False)
    assert mask.min() >= 0.15 and mask.max() <= 1.0


def test_written_mask_is_float32_npy(tmp_path, dappled_ground):
    """.npy, not PNG. The gobo is a light modifier; 8-bit banding in a mask
    becomes visible stepping in a soft shadow edge."""
    pytest.importorskip("PIL")
    result = shading_mod.write_shadow_mask(dappled_ground, tmp_path, "IMG_7096",
                                           blur_px=61, prefer_intrinsic=False)

    loaded = np.load(result.npy_path)
    assert loaded.dtype == np.float32
    assert loaded.shape == dappled_ground.shape[:2]
    assert result.shape == loaded.shape
    assert result.npy_path.endswith("IMG_7096_shademask.npy")


def test_preview_png_is_written_for_human_inspection(tmp_path, dappled_ground):
    pytest.importorskip("PIL")
    from PIL import Image

    result = shading_mod.write_shadow_mask(dappled_ground, tmp_path, "IMG_7096",
                                           blur_px=61, prefer_intrinsic=False)
    assert Image.open(result.preview_png).size == (128, 128)


def test_shadow_fraction_is_reported(tmp_path, dappled_ground):
    """Baking a gobo from a mask that found nothing wastes a render, so the
    number is returned and the panel shows it."""
    pytest.importorskip("PIL")
    result = shading_mod.write_shadow_mask(dappled_ground, tmp_path, "IMG_7096",
                                           blur_px=61, prefer_intrinsic=False)
    assert 0.0 < result.shadow_fraction < 0.5

    flat = np.full((64, 64, 3), 0.4, dtype=np.float32)
    empty = shading_mod.write_shadow_mask(flat, tmp_path, "flat", prefer_intrinsic=False)
    assert empty.shadow_fraction == pytest.approx(0.0, abs=1e-6)


def test_intrinsic_shading_failure_is_caught_not_raised(monkeypatch, dappled_ground):
    def explode(_):
        raise RuntimeError("weights not downloaded")

    monkeypatch.setattr(shading_mod, "intrinsic_shading", explode)
    mask, method, note = shading_mod.shadow_mask(dappled_ground, prefer_intrinsic=True)
    assert method == "retinex"
    assert "weights not downloaded" in note
    assert np.isfinite(mask).all()
