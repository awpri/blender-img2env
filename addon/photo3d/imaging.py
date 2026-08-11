"""Image maths shared by the add-on and the server. numpy + stdlib only.

Same rationale as coords.py: the add-on has to be able to do this without the
solver running, the server has to do it at full resolution, and neither may
grow a dependency. Everything here is pure array work, so it is tested outside
Blender.

Nothing here does file I/O. Blender hands us pixels from bpy.types.Image, the
server hands us pixels from PIL or rawpy, and this module does not care which.
"""

from __future__ import annotations

import numpy as np

#: Rec.709 luminance weights. Correct for sRGB primaries, which is what both a
#: display-referred plate and a linear rawpy sRGB develop are in.
LUMA_709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3+) -> (H, W). Extra channels (alpha) are ignored."""
    return np.asarray(rgb, dtype=np.float32)[..., :3] @ LUMA_709


def box_blur(img: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur via summed-area tables. O(n) in the pixel count and
    independent of the radius.

    The obvious np.convolve/apply_along_axis version is ~200x slower at the
    radii this pipeline uses (41 px at 512 wide, and more at plate resolution),
    slow enough that it stalls Blender's UI thread during a gobo bake. Edges
    are handled by replicate padding, which keeps the shadow mask from
    developing a bright frame.
    """
    radius = int(max(0, radius))
    if radius == 0:
        return np.asarray(img, dtype=np.float32).copy()

    out = np.asarray(img, dtype=np.float64)
    for axis in (0, 1):
        out = np.moveaxis(out, axis, 0)
        n = out.shape[0]
        r = min(radius, n - 1)
        padded = np.concatenate(
            [np.repeat(out[:1], r, axis=0), out, np.repeat(out[-1:], r, axis=0)], axis=0)
        cumulative = np.cumsum(padded, axis=0)
        cumulative = np.concatenate([np.zeros_like(cumulative[:1]), cumulative], axis=0)
        window = 2 * r + 1
        out = (cumulative[window:window + n] - cumulative[:n]) / window
        out = np.moveaxis(out, 0, axis)
    return out.astype(np.float32)


def retinex_shadow_mask(rgb: np.ndarray, blur_px: int = 41, contrast: float = 1.6,
                        floor: float = 0.15) -> np.ndarray:
    """Shadow mask from local luminance divided by a heavily blurred copy.

    The assumption is that albedo varies slowly across a surface while
    illumination under foliage varies fast, so the ratio isolates shade. That
    assumption is only true over roughly uniform ground — grass, gravel, snow,
    tarmac. Over mixed terrain a dark rock reads as shadow, which is why the
    output is clamped at `floor`: the worst case is then a slightly-too-dark
    patch of gobo rather than a black hole punched in the sunlight.

    `blur_px` must be comfortably LARGER than the shadow features you want.
    The ratio only sees illumination structure finer than its own kernel: in
    the middle of a shadow wider than the blur, the blurred reference has
    already fallen to the local luminance and the mask reads as fully lit. Too
    small a blur therefore returns a nearly blank mask, which looks like the
    extraction failing rather than like a knob in the wrong place.

    This is the fallback. shading.py's intrinsic decomposition separates
    reflectance from shading properly and does not make the uniform-albedo
    assumption at all; prefer it when the model is installed.
    """
    lum = luminance(rgb)
    base = box_blur(lum, max(1, int(blur_px) // 2))
    ratio = lum / np.clip(base, 1e-4, None)
    return np.clip((ratio - 1.0) * float(contrast) + 1.0, float(floor), 1.0)


def normalize_shading(shading: np.ndarray, percentile: float = 95.0,
                      floor: float = 0.15) -> np.ndarray:
    """Scale a shading layer so 'fully lit' sits at 1.0.

    Intrinsic decomposition returns shading in arbitrary units. A high
    percentile rather than the max is used as the lit reference so a single
    specular pixel cannot drag the whole mask dark.
    """
    s = np.asarray(shading, dtype=np.float32)
    if s.ndim == 3:
        s = luminance(s)
    reference = float(np.nanpercentile(s, percentile))
    if not np.isfinite(reference) or reference <= 1e-6:
        return np.ones_like(s)
    return np.clip(s / reference, float(floor), 1.0)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Display-referred sRGB -> linear. Needed before any mask arithmetic that
    claims to be about light rather than about pixel values."""
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.clip(x, 0, None) ** (1 / 2.4) - 0.055)


def resize_nearest(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Nearest-neighbour resample. Crude on purpose — it has no dependencies and
    it is only ever used to shrink a plate for statistics, never for anything
    the eye sees."""
    h, w = img.shape[:2]
    rows = np.clip((np.arange(height) + 0.5) * h / height, 0, h - 1).astype(int)
    cols = np.clip((np.arange(width) + 0.5) * w / width, 0, w - 1).astype(int)
    return img[rows][:, cols]


def bounce_strength(plate_linear: np.ndarray, irradiance: float, albedo: float,
                    lower_half_only: bool = True) -> float:
    """Emission multiplier that makes the bounce proxy radiometrically honest.

    A display-referred plate tops out at 1.0 while real sunlit grass is many
    times brighter than that, so the emissive twin needs a scene-referred
    multiplier. For a Lambertian surface, radiance = irradiance * albedo / pi.
    Solve for the factor that takes the plate's median ground value to that
    radiance and you have a number derived from the scene rather than dialled
    in by eye.

    `irradiance` is the sun + sky irradiance already present in the scene, in
    the same units as the sun lamp's strength.
    """
    px = np.asarray(plate_linear, dtype=np.float32)
    if lower_half_only:
        px = px[px.shape[0] // 2:]
    median = float(np.median(luminance(px)))
    target = float(irradiance) * float(albedo) / np.pi
    return float(max(0.05, target / max(median, 1e-4)))
