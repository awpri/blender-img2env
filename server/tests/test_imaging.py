"""The shared image maths — blur, masks, bounce calibration.

These run in Blender too (imaging.py is bpy-free and ships with the add-on), so
a regression here breaks the gobo bake as well as the server.
"""

from __future__ import annotations

import numpy as np
import pytest

from photo3d import imaging


# ---------------------------------------------------------------------------
# box blur
# ---------------------------------------------------------------------------

def test_blur_preserves_a_constant_image():
    """Replicate padding, not zero padding. Zero padding darkens the border,
    which puts a bright frame around every shadow mask."""
    flat = np.full((32, 48), 0.42, dtype=np.float32)
    np.testing.assert_allclose(imaging.box_blur(flat, 7), flat, atol=1e-6)


def test_blur_conserves_the_mean():
    rng = np.random.default_rng(0)
    img = rng.random((64, 64)).astype(np.float32)
    assert imaging.box_blur(img, 5).mean() == pytest.approx(img.mean(), abs=0.02)


def test_blur_matches_a_naive_convolution():
    """The summed-area version is ~200x faster; it has to give the same answer."""
    rng = np.random.default_rng(1)
    img = rng.random((24, 24)).astype(np.float32)
    radius = 3
    padded = np.pad(img, radius, mode="edge")
    window = 2 * radius + 1
    naive = np.empty_like(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            naive[y, x] = padded[y:y + window, x:x + window].mean()
    np.testing.assert_allclose(imaging.box_blur(img, radius), naive, atol=1e-5)


def test_blur_radius_zero_is_a_no_op():
    img = np.arange(16, dtype=np.float32).reshape(4, 4)
    np.testing.assert_allclose(imaging.box_blur(img, 0), img)


def test_blur_radius_larger_than_the_image_does_not_crash():
    img = np.arange(9, dtype=np.float32).reshape(3, 3)
    assert imaging.box_blur(img, 50).shape == (3, 3)


def test_blur_smooths_an_impulse_symmetrically():
    impulse = np.zeros((21, 21), dtype=np.float32)
    impulse[10, 10] = 1.0
    blurred = imaging.box_blur(impulse, 3)
    assert blurred[10, 10] == pytest.approx(1.0 / 49.0, rel=1e-4)
    assert blurred[10, 7] == pytest.approx(blurred[10, 13], rel=1e-5)
    assert blurred[7, 10] == pytest.approx(blurred[13, 10], rel=1e-5)


# ---------------------------------------------------------------------------
# the shadow mask
# ---------------------------------------------------------------------------

def test_uniform_ground_has_no_shadow():
    ground = np.full((64, 64, 3), 0.35, dtype=np.float32)
    mask = imaging.retinex_shadow_mask(ground, blur_px=21)
    np.testing.assert_allclose(mask, 1.0, atol=1e-4)


def test_a_dark_patch_reads_as_shade():
    ground = np.full((64, 64, 3), 0.5, dtype=np.float32)
    ground[28:36, 28:36] = 0.15
    mask = imaging.retinex_shadow_mask(ground, blur_px=61)
    assert mask[32, 32] < 0.6
    assert mask[5, 5] > 0.9


def test_shadows_wider_than_the_blur_are_invisible_to_the_ratio():
    """A real and load-bearing limitation, not a bug.

    The ratio detects illumination structure only at scales SMALLER than the
    blur. In the middle of a shadow wider than the kernel, the blurred
    reference has already fallen to the local luminance, the ratio returns to
    1, and the mask reads 'fully lit'.

    Practical consequence: mask_blur has to be comfortably larger than the
    dapple you are trying to steal. Set it too small and the gobo comes back
    blank, which looks like the extraction failing rather than like a knob
    being in the wrong place. It is why the panel's tooltip says so and why
    intrinsic decomposition, which has no such scale, is preferred when it is
    installed.
    """
    ground = np.full((96, 96, 3), 0.5, dtype=np.float32)
    ground[24:72, 24:72] = 0.15
    mask = imaging.retinex_shadow_mask(ground, blur_px=15)
    assert mask[48, 48] == pytest.approx(1.0, abs=1e-3), "centre reads as lit"
    assert mask[24, 48] < 0.9, "the edge of the shadow is still found"


def test_mask_never_goes_fully_black():
    """The clamp is what makes the failure mode 'slightly too dark' rather than
    'a hole punched in the sunlight'. A dark rock reads as shade and there is
    no fixing that from one photo, so bound the damage instead."""
    ground = np.full((32, 32, 3), 0.5, dtype=np.float32)
    ground[10:20, 10:20] = 0.0
    mask = imaging.retinex_shadow_mask(ground, blur_px=11, contrast=6.0, floor=0.15)
    assert mask.min() >= 0.15
    assert mask.max() <= 1.0


def test_contrast_deepens_the_mask():
    ground = np.full((48, 48, 3), 0.5, dtype=np.float32)
    ground[20:28, 20:28] = 0.25
    soft = imaging.retinex_shadow_mask(ground, blur_px=15, contrast=0.5, floor=0.0)
    hard = imaging.retinex_shadow_mask(ground, blur_px=15, contrast=3.0, floor=0.0)
    assert hard[24, 24] < soft[24, 24]


def test_shading_normalisation_ignores_a_single_specular_pixel():
    """A high percentile rather than the max as the 'fully lit' reference —
    otherwise one glint off a wet rock drags the entire mask dark."""
    shading = np.full((32, 32), 0.5, dtype=np.float32)
    shading[0, 0] = 500.0
    assert imaging.normalize_shading(shading)[16, 16] == pytest.approx(1.0)


def test_shading_normalisation_handles_a_black_layer():
    assert imaging.normalize_shading(np.zeros((8, 8), np.float32)).min() == 1.0


def test_shading_normalisation_accepts_three_channels():
    assert imaging.normalize_shading(np.full((8, 8, 3), 0.4, np.float32)).shape == (8, 8)


# ---------------------------------------------------------------------------
# colour
# ---------------------------------------------------------------------------

def test_srgb_round_trip():
    values = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    np.testing.assert_allclose(imaging.linear_to_srgb(imaging.srgb_to_linear(values)),
                               values, atol=1e-5)


def test_srgb_midpoint_is_the_familiar_number():
    """0.5 sRGB is ~0.214 linear. Getting this wrong makes every bounce
    calibration off by a factor of two."""
    assert float(imaging.srgb_to_linear(np.float32(0.5))) == pytest.approx(0.2140, abs=1e-3)


def test_luminance_weights_sum_to_one():
    assert imaging.LUMA_709.sum() == pytest.approx(1.0)
    assert imaging.luminance(np.ones((4, 4, 3), np.float32)) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# bounce strength — the number that stops it being a slider you guess at
# ---------------------------------------------------------------------------

def test_bounce_strength_inverts_the_lambertian_relation():
    """radiance = irradiance * albedo / pi. Feed in a plate whose median is
    already the right radiance and the multiplier must come back as 1."""
    irradiance, albedo = 4.0, 0.18
    radiance = irradiance * albedo / np.pi
    plate = np.full((32, 32, 3), radiance, dtype=np.float32)
    assert imaging.bounce_strength(plate, irradiance, albedo) == pytest.approx(1.0, rel=1e-4)


def test_darker_ground_needs_more_gain():
    bright = np.full((32, 32, 3), 0.5, dtype=np.float32)
    dark = np.full((32, 32, 3), 0.05, dtype=np.float32)
    assert imaging.bounce_strength(dark, 4.0, 0.18) > imaging.bounce_strength(bright, 4.0, 0.18)


def test_bounce_strength_reads_the_lower_half():
    """The ground is at the bottom of the frame; the sky at the top would
    otherwise dominate the median and under-drive the bounce."""
    plate = np.zeros((32, 32, 3), dtype=np.float32)
    plate[:16] = 1.0          # blown-out sky
    plate[16:] = 0.1          # ground
    assert imaging.bounce_strength(plate, 4.0, 0.18) > \
           imaging.bounce_strength(plate, 4.0, 0.18, lower_half_only=False)


def test_bounce_strength_survives_a_black_plate():
    assert np.isfinite(imaging.bounce_strength(np.zeros((8, 8, 3), np.float32), 4.0, 0.18))


def test_resize_nearest_samples_pixel_centres():
    """Centre sampling, not corner sampling: output pixel i reads input row
    floor((i + 0.5) * h / height). Corner sampling would bias every downsampled
    statistic toward the top-left of the frame."""
    img = np.arange(64, dtype=np.float32).reshape(8, 8)
    small = imaging.resize_nearest(img, 4, 4)
    assert small.shape == (4, 4)
    assert small[0, 0] == img[1, 1]
    assert small[-1, -1] == img[7, 7]


def test_resize_nearest_is_the_identity_at_the_same_size():
    img = np.arange(36, dtype=np.float32).reshape(6, 6)
    np.testing.assert_array_equal(imaging.resize_nearest(img, 6, 6), img)
