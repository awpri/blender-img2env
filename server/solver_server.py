"""Persistent local solve daemon.

Runs OUTSIDE Blender in its own venv, keeps models resident in unified memory,
and talks to the add-on over localhost HTTP. One round trip per photo:

    POST /solve  {"image_path": "/abs/path/IMG_7096.DNG"}
    ->  intrinsics, gravity, pitch/roll/heading, camera height, sun angles,
        and paths to a metric depth .npy and the plates.

Two rules the transport depends on, both from the handoff:

  * Images cross the boundary as FILESYSTEM PATHS, never base64. A 48 MP plate
    is ~25 MB and base64 adds a third on top plus a JSON parse at both ends.
  * Depth crosses as .npy float32, never PNG. PNG quantisation destroys metric
    precision and with it the whole collision story.

Run:
    uvicorn server.solver_server:app --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import os
import time
import traceback
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__, depth as depth_mod, exif as exif_mod, raw as raw_mod
from .exif import Photo3DError

CACHE = Path(os.environ.get("PHOTO3D_CACHE", Path.home() / ".cache" / "photo3d"))
CACHE.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Photo3D solver", version=__version__)


# ---------------------------------------------------------------------------
# request / response
# ---------------------------------------------------------------------------

class SolveRequest(BaseModel):
    image_path: str
    focal_override: float | None = Field(None, description="35mm-equivalent mm; None reads EXIF")
    focal_convention: str = "long_edge"
    refine: bool = Field(False, description="add the Marigold detail pass (~3x slower)")
    depth_long_edge: int = Field(1536, ge=256, le=4096)
    assumed_eye_height: float = Field(1.55, gt=0.0)
    far_clamp_m: float = Field(depth_mod.DEFAULT_FAR_CLAMP_M, gt=1.0)
    depth_scale: float = Field(
        1.0, gt=0.0,
        description="Multiplier on metric depth. Depth Pro saturates on wide "
                    "scenes with a big depth range — on both reference photos a "
                    "person at 11-14 m read a fifth of their real height. "
                    "Calibrate once per lens with tools/calibrate_scale.py.")
    want_lighting_plate: bool = True
    skip_depth: bool = Field(False, description="metadata-only solve (milestone M1)")


class SegmentRequest(BaseModel):
    image_path: str
    long_edge: int = Field(768, ge=256, le=2048,
                           description="SAM 2 runs on a downscaled plate; masks are "
                                       "returned in this raster's coordinates")
    points_per_batch: int = Field(32, ge=4, le=256)
    min_area_fraction: float = Field(0.002, gt=0.0, lt=1.0)
    max_regions: int = Field(24, ge=1, le=128)
    model: str | None = None


class ShadowMaskRequest(BaseModel):
    image_path: str
    blur_px: int = Field(41, ge=3, le=501)
    contrast: float = Field(1.6, gt=0.0, le=6.0)
    floor: float = Field(0.15, ge=0.0, lt=1.0)
    prefer_intrinsic: bool = True
    long_edge: int = Field(1536, ge=256, le=8192)


def _resolve(image_path: str) -> Path:
    path = Path(os.path.expanduser(image_path)).resolve()
    if not path.exists():
        raise HTTPException(404, f"not found: {path}")
    if not path.is_file():
        raise HTTPException(400, f"not a file: {path}")
    return path


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Cheap enough for the add-on to poll before it bothers the user.

    Reports the checkpoint too: "is torch alive" and "can I actually run depth"
    are different questions, and the second one is the one that fails first.
    """
    try:
        device = str(depth_mod.torch_device())
    except Exception:                                             # noqa: BLE001
        device = "torch unavailable (metadata-only solves will still work)"
    checkpoint = depth_mod.find_checkpoint()
    return {"ok": True, "version": __version__, "device": device,
            "models_resident": depth_mod.loaded_models(),
            "depth_pro_checkpoint": checkpoint or "NOT FOUND — run get_pretrained_models.sh",
            "depth_available": checkpoint is not None,
            "cache": str(CACHE)}


@app.post("/solve")
def solve(req: SolveRequest):
    """The one endpoint that matters."""
    started = time.perf_counter()
    path = _resolve(req.image_path)

    try:
        exif = exif_mod.read_exif(path)
        state = exif_mod.camera_state_from_exif(exif, req.focal_override,
                                                req.focal_convention)
    except Photo3DError as exc:
        raise HTTPException(422, str(exc)) from None

    intrinsics = state.intrinsics
    warnings = list(state.warnings)

    # --- plates ----------------------------------------------------------
    try:
        plates = raw_mod.make_plates(path, CACHE, exif, req.want_lighting_plate)
    except raw_mod.PlateError as exc:
        raise HTTPException(422, f"could not prepare plates: {exc}") from None
    if plates.note:
        warnings.append(plates.note)

    # EXIF and the developed pixels can disagree — a DNG's SubIFD carries the
    # full sensor readout while rawpy crops to the active area. The pixels are
    # the truth, because that is what the depth map and the plate are in.
    if (plates.width, plates.height) != (intrinsics.width, intrinsics.height):
        warnings.append(
            f"EXIF said {intrinsics.width}x{intrinsics.height} but the developed "
            f"plate is {plates.width}x{plates.height}; intrinsics rescaled to the pixels")
        scale = plates.width / float(intrinsics.width)
        intrinsics = exif_mod.Intrinsics(
            width=plates.width, height=plates.height, focal_35mm=intrinsics.focal_35mm,
            fx=intrinsics.fx * scale, fy=intrinsics.fy * scale,
            cx=plates.width / 2.0, cy=plates.height / 2.0,
            convention=intrinsics.convention)

    response = {
        "intrinsics": intrinsics.__dict__,
        "gravity_camera": state.orientation.gravity_camera if state.orientation else None,
        "pitch_deg": state.orientation.pitch_deg if state.orientation else None,
        "roll_deg": state.orientation.roll_deg if state.orientation else None,
        "heading_deg": state.geo.heading_deg,
        "sun": state.sun.__dict__ if state.sun else None,
        "geo": state.geo.as_dict(),
        "plate_png": plates.display_png,
        "lighting_exr": plates.lighting_exr,
        "depth_npy": None,
        "depth_shape": None,
        "depth_stats": None,
        "camera_height_m": req.assumed_eye_height,
        "ground_confidence": 0.0,
        "ground_source": "not solved (skip_depth)",
        "warnings": warnings,
    }

    if req.skip_depth:
        response["solve_seconds"] = time.perf_counter() - started
        return response

    # --- depth -----------------------------------------------------------
    from PIL import Image

    image = raw_mod.load_display_image(plates.display_png)
    scale = req.depth_long_edge / float(max(image.size))
    if scale < 1.0:
        small = image.resize((max(1, round(image.width * scale)),
                              max(1, round(image.height * scale))), Image.LANCZOS)
    else:
        small = image

    # The focal length must be that of the image actually handed to the model.
    f_px_small = intrinsics.fx * (small.width / float(intrinsics.width))

    try:
        depth = depth_mod.run_depth_pro(small, f_px_small)
        if req.refine:
            depth = depth_mod.refine_with_marigold(small, depth)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from None
    except Exception as exc:                                      # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(500, f"depth inference failed: {exc}") from None

    # Scale before clamping: the clamp is a metric distance, so applying it to
    # unscaled depth would cut the scene at the wrong place.
    if req.depth_scale != 1.0:
        depth = depth * float(req.depth_scale)
    depth = depth_mod.clamp_far_field(depth, req.far_clamp_m)
    ground = depth_mod.solve_ground(depth, intrinsics.__dict__,
                                    state.orientation.gravity_camera if state.orientation else None,
                                    req.assumed_eye_height)

    depth_path = CACHE / f"{path.stem}_depth.npy"
    np.save(depth_path, depth.astype(np.float32))

    response.update({
        "depth_npy": str(depth_path),
        "depth_shape": [int(depth.shape[0]), int(depth.shape[1])],
        "depth_stats": depth_mod.depth_statistics(depth),
        "depth_scale": req.depth_scale,
        "camera_height_m": ground.camera_height_m,
        "ground_confidence": ground.confidence,
        "ground_source": ground.source,
        "solve_seconds": time.perf_counter() - started,
    })
    if ground.confidence < 0.15:
        response["warnings"].append(
            f"ground plane confidence is only {ground.confidence:.2f} — the camera "
            "height is weakly supported, so check the scale before trusting a "
            "physics drop.")
    # Only flag the scale on the lens where it actually went wrong. Measured:
    # at 14 mm on alpine landscapes Depth Pro under-read by 5x, while at 24 mm
    # on a station platform it was within 5% at 20 m. Warning on every solve
    # would train the user to ignore the warnings that matter.
    if req.depth_scale == 1.0 and intrinsics.focal_35mm < 20.0:
        response["warnings"].append(
            f"{intrinsics.focal_35mm:.0f}mm is ultra-wide, where Depth Pro's "
            "metric scale has measured 5x low on scenes with a large depth "
            "range. Check it against something of known size before trusting a "
            "physics drop — tools/calibrate_scale.py does it for you.")
    return response


@app.post("/shadow_mask")
def shadow_mask(req: ShadowMaskRequest):
    """Shade mask for the sun gobo. Separate endpoint because it is a different
    question from 'where was the camera' and is only worth asking on shots that
    actually have dappled shade."""
    from PIL import Image

    from . import shading as shading_mod

    path = _resolve(req.image_path)
    image = raw_mod.load_display_image(path)
    scale = req.long_edge / float(max(image.size))
    if scale < 1.0:
        image = image.resize((max(1, round(image.width * scale)),
                              max(1, round(image.height * scale))), Image.LANCZOS)

    rgb = np.asarray(image, dtype=np.float32) / 255.0
    try:
        result = shading_mod.write_shadow_mask(
            rgb, CACHE, path.stem, req.blur_px, req.contrast, req.floor,
            req.prefer_intrinsic)
    except Exception as exc:                                      # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(500, f"shadow mask failed: {exc}") from None
    return result.__dict__


@app.post("/segment")
def segment(req: SegmentRequest):
    """SAM 2 masks for splitting the proxy into separate surfaces.

    Returns a label image, not a list of masks: the add-on has to answer "which
    region is this face in?" for a hundred thousand faces, and one lookup beats
    N membership tests.
    """
    from PIL import Image

    from . import segment as segment_mod

    path = _resolve(req.image_path)
    image = raw_mod.load_display_image(path)
    scale = req.long_edge / float(max(image.size))
    if scale < 1.0:
        image = image.resize((max(1, round(image.width * scale)),
                              max(1, round(image.height * scale))), Image.LANCZOS)

    rgb = np.asarray(image, dtype=np.float32) / 255.0
    try:
        result = segment_mod.segment(
            rgb, CACHE, path.stem,
            model_id=req.model or segment_mod.DEFAULT_MODEL,
            points_per_batch=req.points_per_batch,
            min_area_fraction=req.min_area_fraction,
            max_regions=req.max_regions)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from None
    except Exception as exc:                                      # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(500, f"segmentation failed: {exc}") from None
    return result.__dict__


def main():
    import uvicorn
    uvicorn.run(app, host=os.environ.get("PHOTO3D_HOST", "127.0.0.1"),
                port=int(os.environ.get("PHOTO3D_PORT", 8765)))


if __name__ == "__main__":
    main()
