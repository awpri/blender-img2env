"""Metric depth: Depth Pro as the backbone, Marigold as optional garnish.

Marigold produces affine-invariant depth — beautiful, detailed, and with no
absolute scale and no absolute offset. You cannot get a camera height or a
metre-accurate collision surface out of it alone; a Minecraft block dropped
onto a Marigold mesh lands at an arbitrary size. Depth Pro outputs real metres
and is native to Apple Silicon, so it is the backbone. `refine` fits Marigold
to Depth Pro's scale in disparity space, where the relationship is genuinely
affine.

Models stay resident in this module's registry. A cold reload per photo is a
bug, not a slow path: the first solve pays ~15 s of weight loading and every
one after it should be 1-3 s.

Torch is imported lazily. Tests exercise the array maths — unprojection, the
ground-plane solve, the far clamp — without it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from ._core import coords

_MODELS: dict[str, object] = {}

#: Depth Pro degrades badly past ~100 m, and a 14 mm frame has a lot of scene
#: out there. Nothing in a composite collides with a peak 4 km away, so the far
#: field is clamped rather than trusted. Raise it only if something you care
#: about is genuinely that far off and you have checked the numbers.
DEFAULT_FAR_CLAMP_M = 120.0


def torch_device():
    """MPS on Apple Silicon, CUDA if someone points this at a desktop GPU.

    The daemon is deliberately device-agnostic: the whole thing runs over
    localhost HTTP, so pointing ENDPOINT at another machine is a config change,
    not a rewrite.
    """
    import torch
    if os.environ.get("PHOTO3D_DEVICE"):
        return torch.device(os.environ["PHOTO3D_DEVICE"])
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def loaded_models() -> list[str]:
    return sorted(_MODELS)


# ---------------------------------------------------------------------------
# Depth Pro
# ---------------------------------------------------------------------------

#: Where the 1.9 GB checkpoint might be. Depth Pro's default config points at
#: "./checkpoints/depth_pro.pt" — relative to the *current working directory* —
#: so a daemon started from the repo root looks in the repo root and finds
#: nothing, while get_pretrained_models.sh has put the file next to the cloned
#: source. That mismatch is the first thing anyone hits, and the resulting
#: error does not say what is wrong, so resolve it here instead.
CHECKPOINT_CANDIDATES = (
    "~/src/ml-depth-pro/checkpoints/depth_pro.pt",
    "./checkpoints/depth_pro.pt",
    "~/.cache/photo3d/checkpoints/depth_pro.pt",
    "~/ml-depth-pro/checkpoints/depth_pro.pt",
)


def find_checkpoint() -> str | None:
    """Locate depth_pro.pt, honouring PHOTO3D_DEPTH_PRO_CHECKPOINT first."""
    override = os.environ.get("PHOTO3D_DEPTH_PRO_CHECKPOINT")
    if override:
        path = os.path.expanduser(override)
        return path if os.path.exists(path) else None
    for candidate in CHECKPOINT_CANDIDATES:
        path = os.path.expanduser(candidate)
        if os.path.exists(path):
            return path
    return None


def get_depth_pro():
    if "depth_pro" not in _MODELS:
        try:
            import depth_pro
        except ImportError:
            raise RuntimeError(
                "Depth Pro is not installed. In the solver venv:\n"
                "  pip install git+https://github.com/apple/ml-depth-pro.git") from None

        checkpoint = find_checkpoint()
        if checkpoint is None:
            searched = "\n  ".join(os.path.expanduser(c) for c in CHECKPOINT_CANDIDATES)
            raise RuntimeError(
                "Depth Pro is installed but its weights are missing. Fetch them:\n"
                "  git clone https://github.com/apple/ml-depth-pro.git ~/src/ml-depth-pro\n"
                "  cd ~/src/ml-depth-pro && source get_pretrained_models.sh\n\n"
                f"Looked in:\n  {searched}\n\n"
                "Set PHOTO3D_DEPTH_PRO_CHECKPOINT to point somewhere else.")

        # Copy the default config and redirect it at the checkpoint we found,
        # rather than depending on the daemon's working directory.
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
        import dataclasses
        config = dataclasses.replace(DEFAULT_MONODEPTH_CONFIG_DICT,
                                     checkpoint_uri=checkpoint)
        model, transform = depth_pro.create_model_and_transforms(
            config=config, device=torch_device())
        model.eval()
        _MODELS["depth_pro"] = (model, transform)
    return _MODELS["depth_pro"]


def run_depth_pro(image, f_px: float) -> np.ndarray:
    """PIL image -> metric depth in metres, as float32.

    f_px must be the focal length in pixels *of the image actually passed in*,
    not of the full-resolution plate. Handing it the full-res value tells the
    model the lens is several times longer than it is, and the metric scale
    comes back wrong by that factor — which looks exactly like the depth model
    being bad rather than like a bookkeeping error.
    """
    import torch
    model, transform = get_depth_pro()
    tensor = transform(image.convert("RGB"))
    with torch.no_grad():
        out = model.infer(tensor, f_px=torch.tensor(float(f_px)))
    return out["depth"].squeeze().float().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# optional Marigold refinement
# ---------------------------------------------------------------------------

def get_marigold():
    if "marigold" not in _MODELS:
        import torch
        from diffusers import MarigoldDepthPipeline
        _MODELS["marigold"] = MarigoldDepthPipeline.from_pretrained(
            "prs-eth/marigold-depth-v1-1", torch_dtype=torch.float16).to(torch_device())
    return _MODELS["marigold"]


def fit_relative_to_metric(relative: np.ndarray, metric: np.ndarray,
                           near_m: float = 0.1, far_m: float = 200.0) -> np.ndarray:
    """Least-squares fit of affine-invariant depth onto metric depth.

    Done in DISPARITY space (1/z), because that is where the relationship
    between an affine-invariant prediction and true depth is actually affine.
    Fitting in depth space instead produces a good match in the foreground and
    a wildly wrong horizon.
    """
    metric = np.asarray(metric, dtype=np.float64)
    relative = np.asarray(relative, dtype=np.float64)
    disparity = 1.0 / np.clip(metric, 1e-3, None)

    valid = (np.isfinite(disparity) & np.isfinite(relative)
             & (metric > near_m) & (metric < far_m))
    if valid.sum() < 100:
        return metric.astype(np.float32)

    design = np.stack([relative[valid], np.ones(int(valid.sum()))], axis=1)
    scale, offset = np.linalg.lstsq(design, disparity[valid], rcond=None)[0]
    return (1.0 / np.clip(scale * relative + offset, 1e-3, None)).astype(np.float32)


def refine_with_marigold(image, metric: np.ndarray, steps: int = 4) -> np.ndarray:
    """Marigold's micro-detail at Depth Pro's scale.

    Roughly triples solve time for a mesh that is invisible to the camera by
    design, so leave it off until you hit a shot where silhouette detail
    actually matters.
    """
    from PIL import Image as PILImage
    pipe = get_marigold()
    relative = np.asarray(pipe(image, num_inference_steps=steps,
                               ensemble_size=1).prediction).squeeze().astype(np.float64)
    if relative.shape != metric.shape:
        relative = np.asarray(PILImage.fromarray(relative).resize(
            (metric.shape[1], metric.shape[0]), PILImage.BILINEAR))
    return fit_relative_to_metric(relative, metric)


# ---------------------------------------------------------------------------
# post-processing (no torch)
# ---------------------------------------------------------------------------

def clamp_far_field(depth: np.ndarray, far_m: float = DEFAULT_FAR_CLAMP_M,
                    near_m: float = 0.05) -> np.ndarray:
    """Clamp and de-NaN. Keeps a single inf from turning the proxy mesh into a
    vertex at 1e38, which makes every viewport navigation operation useless."""
    d = np.nan_to_num(np.asarray(depth, dtype=np.float32),
                      nan=far_m, posinf=far_m, neginf=near_m)
    return np.clip(d, near_m, far_m)


@dataclass
class GroundSolve:
    camera_height_m: float | None
    confidence: float
    source: str


def solve_ground(depth: np.ndarray, intrinsics: dict, g_camera,
                 fallback_height_m: float = 1.55) -> GroundSolve:
    """Camera height above ground.

    Gravity already fixes the ground plane's normal, so this is a 1-D mode
    search rather than a plane fit — see coords.ground_plane_height(). When
    there is no gravity, or too little ground in frame, fall back to an assumed
    eye height and say so, because a wrong-but-plausible height is worse than
    an admitted guess.
    """
    if g_camera is None:
        return GroundSolve(fallback_height_m, 0.0, "assumed eye height (no gravity)")

    height, confidence = coords.ground_plane_height(depth, intrinsics, g_camera)
    if height is None:
        return GroundSolve(fallback_height_m, 0.0, "assumed eye height (no ground found)")
    if height <= 0.0:
        # The mode landed above the camera: usually an interior shot where the
        # lower frame is a table, or a camera pointed up at a ceiling.
        return GroundSolve(fallback_height_m, 0.0, "assumed eye height (ground above camera)")
    return GroundSolve(float(height), float(confidence), "gravity-constrained plane fit")


def depth_statistics(depth: np.ndarray) -> dict:
    """Numbers worth putting in front of the user before they trust a mesh."""
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return {"min_m": None, "median_m": None, "p99_m": None, "max_m": None}
    return {
        "min_m": float(np.min(finite)),
        "median_m": float(np.median(finite)),
        "p99_m": float(np.percentile(finite, 99)),
        "max_m": float(np.max(finite)),
    }
