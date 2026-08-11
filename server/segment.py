"""SAM 2 segmentation, for splitting the proxy into real surfaces.

Depth alone gives soft, wobbly silhouettes at exactly the edges the eye
scrutinises, and one undifferentiated surface where the scene actually has
glass, water, metal and ground. Segmentation fixes both: each mask becomes its
own object, so it can take its own material and its own ray visibility, and its
outline follows an image edge rather than a depth gradient.

SAM 2 is class-agnostic — it finds *things*, not *kinds of thing*. It cannot
tell you which region is glass, and no monocular method can, because glass is
defined by what is behind it. So this produces the regions and the user says
what they are made of, using the same material presets that already exist.

Weights come from HuggingFace via transformers and are cached under
~/.cache/huggingface. The tiny model is the default: on an M4 Pro it loads in
about 5 s and segments a 500 px frame in about 12 s, and mask quality is
dominated by image resolution rather than model size for this purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

_MODELS: dict[str, object] = {}

DEFAULT_MODEL = "facebook/sam2.1-hiera-tiny"


def get_generator(model_id: str = DEFAULT_MODEL):
    """Resident automatic-mask generator. Same rule as every other model here:
    load once, keep it in unified memory."""
    if model_id not in _MODELS:
        try:
            from transformers import pipeline
        except ImportError:
            raise RuntimeError("transformers is not installed: "
                               "pip install transformers accelerate") from None
        from .depth import torch_device
        _MODELS[model_id] = pipeline("mask-generation", model=model_id,
                                     device=str(torch_device()))
    return _MODELS[model_id]


@dataclass
class Region:
    label: int
    area_fraction: float
    centroid_xy: list[float]          # normalised, 0-1, origin top-left
    bbox: list[float]                 # normalised x0, y0, x1, y1


@dataclass
class Segmentation:
    labels_npy: str
    preview_png: str
    width: int
    height: int
    regions: list[dict]
    model: str
    note: str


def masks_to_labels(masks, min_area_fraction: float = 0.002,
                    max_regions: int = 24) -> tuple[np.ndarray, list[Region]]:
    """Overlapping boolean masks -> one int32 label image, 0 meaning unassigned.

    A label image rather than N masks because the add-on has to answer "which
    region is this face in?" for a hundred thousand faces, and one lookup beats
    N membership tests.

    Painted largest first so smaller masks land on top: SAM 2 happily returns a
    mask for a whole platform and another for a bag sitting on it, and the bag
    is the one worth having as its own object.
    """
    arrays = [np.asarray(m, dtype=bool) for m in masks]
    arrays = [m for m in arrays if m.mean() >= min_area_fraction]
    arrays.sort(key=lambda m: m.mean(), reverse=True)
    arrays = arrays[:max_regions]

    if not arrays:
        return np.zeros((1, 1), dtype=np.int32), []

    height, width = arrays[0].shape
    labels = np.zeros((height, width), dtype=np.int32)
    regions: list[Region] = []

    for index, mask in enumerate(arrays, start=1):
        labels[mask] = index

    # Measure AFTER painting, so the numbers describe what actually survived
    # rather than what the mask claimed before it was overwritten.
    for index in range(1, len(arrays) + 1):
        present = labels == index
        area = float(present.mean())
        if area <= 0.0:
            continue
        rows, cols = np.nonzero(present)
        regions.append(Region(
            label=index,
            area_fraction=area,
            centroid_xy=[float(cols.mean() / width), float(rows.mean() / height)],
            bbox=[float(cols.min() / width), float(rows.min() / height),
                  float(cols.max() / width), float(rows.max() / height)]))
    return labels, regions


def _preview(labels: np.ndarray) -> np.ndarray:
    """False-colour the label map so a human can see what was found."""
    rng = np.random.default_rng(0)
    palette = rng.uniform(0.15, 1.0, size=(int(labels.max()) + 1, 3))
    palette[0] = 0.05
    return palette[labels].astype(np.float32)


def segment(rgb: np.ndarray, cache_dir: str | Path, stem: str,
            model_id: str = DEFAULT_MODEL, points_per_batch: int = 32,
            min_area_fraction: float = 0.002,
            max_regions: int = 24) -> Segmentation:
    from PIL import Image

    from .raw import write_png

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    image = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    generator = get_generator(model_id)
    output = generator(image, points_per_batch=points_per_batch)
    labels, regions = masks_to_labels(output["masks"], min_area_fraction, max_regions)

    labels_path = cache_dir / f"{stem}_labels.npy"
    np.save(labels_path, labels.astype(np.int32))
    preview = write_png(_preview(labels), cache_dir / f"{stem}_labels.png")

    return Segmentation(
        labels_npy=str(labels_path), preview_png=preview,
        width=int(labels.shape[1]), height=int(labels.shape[0]),
        regions=[asdict(r) for r in regions], model=model_id,
        note=f"{len(regions)} regions kept of {len(output['masks'])} masks. "
             "SAM 2 is class-agnostic, so it found the surfaces but not what "
             "they are made of — assign materials per region.")
