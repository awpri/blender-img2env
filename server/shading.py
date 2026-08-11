"""Intrinsic decomposition -> shadow mask, for the sun gobo.

Light through larch branches cannot be reconstructed from one photo: the
canopy geometry is not recoverable and never will be. But the shadow pattern it
casts is already printed on the ground in the plate, so the move is to steal it
rather than simulate it — a cucoloris, which is a real film-lighting technique.

Two ways to get the mask, in descending order of correctness:

  1. INTRINSIC DECOMPOSITION (compphoto/Intrinsic). Separates an image into
     reflectance and shading. The shading layer *is* the mask, with no
     assumption about albedo being uniform, so it survives mixed terrain.
  2. RETINEX RATIO (imaging.retinex_shadow_mask). Local luminance over blurred
     luminance. Assumes albedo varies slowly and illumination under foliage
     varies fast. Cheap, dependency-free, and it reads a dark rock as shade.

The fallback is not a placeholder — it is genuinely better on uniform ground
(grass, gravel, snow, tarmac) because it has no model to be wrong. Both outputs
are clamped, so the worst case is a slightly-too-dark patch of gobo rather than
a black hole punched in the sunlight.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._core import imaging

_MODELS: dict[str, object] = {}


@dataclass
class ShadowMask:
    npy_path: str
    preview_png: str
    method: str
    shape: tuple[int, int]
    shadow_fraction: float
    note: str


def get_intrinsic_model():
    """compphoto/Intrinsic, kept resident like every other model here."""
    if "intrinsic" not in _MODELS:
        from chrislib.general import uninvert  # noqa: F401  (part of the same install)
        from intrinsic.pipeline import load_models
        _MODELS["intrinsic"] = load_models("v2")
    return _MODELS["intrinsic"]


def intrinsic_shading(rgb: np.ndarray) -> np.ndarray:
    """Shading layer from intrinsic decomposition, as a single channel."""
    from intrinsic.pipeline import run_pipeline
    result = run_pipeline(get_intrinsic_model(), np.asarray(rgb, dtype=np.float32))
    for key in ("shading", "hr_shd", "inv_shading", "shd"):
        if key in result:
            layer = np.asarray(result[key], dtype=np.float32)
            if key.startswith("inv"):
                layer = 1.0 / np.clip(layer, 1e-4, None)
            return layer.squeeze()
    raise RuntimeError(f"intrinsic pipeline returned no shading layer: {sorted(result)}")


def shadow_mask(rgb: np.ndarray, blur_px: int = 41, contrast: float = 1.6,
                floor: float = 0.15, prefer_intrinsic: bool = True) -> tuple[np.ndarray, str, str]:
    """(mask, method, note). Mask is 1.0 in full light, `floor` in deep shade."""
    if prefer_intrinsic:
        try:
            shading = intrinsic_shading(rgb)
            return (imaging.normalize_shading(shading, floor=floor),
                    "intrinsic",
                    "shading layer from intrinsic decomposition; albedo is "
                    "separated properly, so mixed terrain is handled")
        except Exception as exc:                                  # noqa: BLE001
            note = (f"intrinsic decomposition unavailable ({type(exc).__name__}: {exc}); "
                    "fell back to the retinex ratio, which reads dark albedo as shade")
    else:
        note = "retinex ratio requested explicitly"
    return (imaging.retinex_shadow_mask(rgb, blur_px=blur_px, contrast=contrast,
                                        floor=floor),
            "retinex", note)


def write_shadow_mask(rgb: np.ndarray, cache_dir: str | Path, stem: str,
                      blur_px: int = 41, contrast: float = 1.6, floor: float = 0.15,
                      prefer_intrinsic: bool = True) -> ShadowMask:
    """Compute the mask and put it on disk for the add-on to pick up.

    .npy for the add-on (float32, no quantisation) and a PNG next to it purely
    so a human can look at the mask before trusting a gobo baked from it.
    """
    from .raw import write_png

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    mask, method, note = shadow_mask(rgb, blur_px, contrast, floor, prefer_intrinsic)

    npy_path = cache_dir / f"{stem}_shademask.npy"
    np.save(npy_path, mask.astype(np.float32))
    preview = write_png(np.repeat(mask[..., None], 3, axis=2),
                        cache_dir / f"{stem}_shademask.png")

    return ShadowMask(
        npy_path=str(npy_path), preview_png=preview, method=method,
        shape=(int(mask.shape[0]), int(mask.shape[1])),
        shadow_fraction=float(np.mean(mask < 0.75)),
        note=note)
