"""
solver_server.py — persistent local solve daemon for the Photo→3D pipeline.

Runs OUTSIDE Blender in its own venv (Blender's bundled Python must never get
torch installed into it). Keeps models resident in unified memory so the second
and subsequent photos solve in ~1-3 s instead of reloading weights every time.

One endpoint, one round trip:
    POST /solve  {"image_path": "/abs/path/IMG_1234.HEIC", ...}
    ->  everything Blender needs: intrinsics, camera rotation, camera height,
        sun angles, and a path to a metric depth .npy

Install:
    python3.11 -m venv ~/.venvs/photo3d && source ~/.venvs/photo3d/bin/activate
    pip install fastapi uvicorn torch torchvision pillow numpy pillow-heif astral
    pip install git+https://github.com/apple/ml-depth-pro.git
    brew install exiftool
Run:
    uvicorn solver_server:app --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CACHE = os.path.expanduser("~/.cache/photo3d")
os.makedirs(CACHE, exist_ok=True)

app = FastAPI()
_models = {}


# ---------------------------------------------------------------------------
# 1. EXIF — the single highest-value, zero-cost source of camera state
# ---------------------------------------------------------------------------

def read_exif(path: str) -> dict:
    """exiftool -j gives us Apple MakerNote fields PIL/piexif silently drop."""
    out = subprocess.run(
        ["exiftool", "-j", "-n", "-G0:1", path],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)[0]


def _find(exif: dict, *suffixes):
    """exiftool group-prefixes keys (e.g. 'Apple:AccelerationVector')."""
    for key, val in exif.items():
        bare = key.split(":")[-1]
        if bare in suffixes:
            return val
    return None


@dataclass
class Intrinsics:
    width: int
    height: int
    focal_35mm: float      # equivalent focal length, 36 mm sensor basis
    fx: float              # pixels
    fy: float
    cx: float
    cy: float


def intrinsics_from_exif(exif: dict, focal_override: float | None) -> Intrinsics:
    w = int(_find(exif, "ImageWidth", "ExifImageWidth"))
    h = int(_find(exif, "ImageHeight", "ExifImageHeight"))
    f35 = focal_override or _find(exif, "FocalLengthIn35mmFormat", "FocalLengthIn35mmFilm")
    if f35 is None:
        raise HTTPException(422, "No 35mm-equivalent focal length in EXIF; pass focal_override.")
    f35 = float(f35)
    # 35mm-equivalent is defined against the LONG edge of a 36x24 frame.
    long_edge = max(w, h)
    f_px = f35 / 36.0 * long_edge
    return Intrinsics(w, h, f35, f_px, f_px, w / 2.0, h / 2.0)


def gravity_from_exif(exif: dict):
    """
    Apple MakerNote 0x0008 'AccelerationVector': XYZ acceleration in g.
    At rest this IS the gravity vector in device coordinates — which means
    pitch and roll are MEASURED, not estimated. No vanishing lines needed.

    Sign/axis convention differs across iOS versions and orientation, so
    calibrate ONCE: photograph a level floor holding the phone vertically,
    and confirm the returned pitch is ~0 and roll ~0.
    """
    vec = _find(exif, "AccelerationVector")
    if vec is None:
        return None
    if isinstance(vec, str):
        vec = [float(x) for x in vec.replace(",", " ").split()]
    g = np.array(vec, dtype=np.float64)

    orientation = int(_find(exif, "Orientation") or 1)
    # Map device axes -> OpenCV camera axes (x right, y down, z forward).
    # Portrait (Orientation 6 on iPhone rear camera) is the common case.
    g_cam = np.array([g[0], -g[1], -g[2]])
    if orientation in (6,):          # rotate 90 CW
        g_cam = np.array([g_cam[1], -g_cam[0], g_cam[2]])
    elif orientation in (8,):        # rotate 90 CCW
        g_cam = np.array([-g_cam[1], g_cam[0], g_cam[2]])
    elif orientation in (3,):        # 180
        g_cam = np.array([-g_cam[0], -g_cam[1], g_cam[2]])

    n = np.linalg.norm(g_cam)
    if n < 1e-6:
        return None
    return (g_cam / n).tolist()


def geo_from_exif(exif: dict):
    lat, lon = _find(exif, "GPSLatitude"), _find(exif, "GPSLongitude")
    if lat is None or lon is None:
        return None
    return {
        "lat": float(lat),
        "lon": float(lon),
        "alt": float(_find(exif, "GPSAltitude") or 0.0),
        "datetime": _find(exif, "DateTimeOriginal"),
        "offset": _find(exif, "OffsetTimeOriginal"),
        "heading": _find(exif, "GPSImgDirection"),
        "heading_ref": _find(exif, "GPSImgDirectionRef"),
    }


def sun_from_geo(geo: dict):
    """
    Physically correct sun position from where and when the shutter fired.
    This beats any image-based light estimator, and it is free.
    """
    if not geo or not geo.get("datetime"):
        return None
    from astral.sun import azimuth, elevation
    from astral import Observer

    dt = datetime.strptime(geo["datetime"][:19], "%Y:%m:%d %H:%M:%S")
    off = geo.get("offset")
    if off:
        sign = 1 if off[0] == "+" else -1
        hh, mm = int(off[1:3]), int(off[4:6])
        dt = dt.replace(tzinfo=timezone.utc) - sign * __import__("datetime").timedelta(hours=hh, minutes=mm)
    else:
        dt = dt.replace(tzinfo=timezone.utc)

    obs = Observer(latitude=geo["lat"], longitude=geo["lon"], elevation=geo["alt"])
    return {
        "azimuth_deg": float(azimuth(obs, dt)),      # clockwise from true north
        "elevation_deg": float(elevation(obs, dt)),  # degrees above horizon
        "utc": dt.isoformat(),
    }


# ---------------------------------------------------------------------------
# 2. Depth — metric backbone + optional high-detail refinement
# ---------------------------------------------------------------------------

def get_depth_pro():
    """Apple Depth Pro: metric depth, sharp boundaries, native Apple Silicon."""
    if "depth_pro" not in _models:
        import depth_pro
        model, transform = depth_pro.create_model_and_transforms(device=DEVICE)
        model.eval()
        _models["depth_pro"] = (model, transform)
    return _models["depth_pro"]


def run_depth_pro(img: Image.Image, f_px: float) -> np.ndarray:
    model, transform = get_depth_pro()
    x = transform(img.convert("RGB"))
    with torch.no_grad():
        out = model.infer(x, f_px=torch.tensor(f_px))
    return out["depth"].squeeze().float().cpu().numpy()  # metres


def refine_with_marigold(img: Image.Image, metric: np.ndarray) -> np.ndarray:
    """
    Marigold gives affine-invariant (relative) depth — gorgeous micro-detail,
    no scale. Fit it to Depth Pro's metric scale in DISPARITY space, which is
    where the relationship is genuinely affine.
    """
    if "marigold" not in _models:
        from diffusers import MarigoldDepthPipeline
        pipe = MarigoldDepthPipeline.from_pretrained(
            "prs-eth/marigold-depth-v1-1", torch_dtype=torch.float16
        ).to(DEVICE)
        _models["marigold"] = pipe
    pipe = _models["marigold"]

    rel = pipe(img, num_inference_steps=4, ensemble_size=1).prediction
    rel = np.asarray(rel).squeeze().astype(np.float64)
    if rel.shape != metric.shape:
        rel = np.asarray(Image.fromarray(rel).resize(metric.shape[::-1], Image.BILINEAR))

    d_metric = 1.0 / np.clip(metric, 1e-3, None)
    valid = np.isfinite(d_metric) & (metric > 0.1) & (metric < 200.0)
    A = np.stack([rel[valid], np.ones(valid.sum())], axis=1)
    a, b = np.linalg.lstsq(A, d_metric[valid], rcond=None)[0]
    return 1.0 / np.clip(a * rel + b, 1e-3, None)


# ---------------------------------------------------------------------------
# 3. Ground plane — gravity turns a hard RANSAC into a 1-D histogram problem
# ---------------------------------------------------------------------------

def solve_camera_height(depth: np.ndarray, K: Intrinsics, g_cam: list | None):
    """
    Because gravity is known, the ground plane's NORMAL is already known.
    All that remains is its offset — the mode of the point cloud's height
    distribution in the lower half of the frame.
    """
    if g_cam is None:
        return None, None

    h, w = depth.shape
    sy, sx = h / K.height, w / K.width
    fx, fy, cx, cy = K.fx * sx, K.fy * sy, K.cx * sx, K.cy * sy

    vv, uu = np.mgrid[0:h, 0:w]
    z = depth
    X = (uu - cx) / fx * z
    Y = (vv - cy) / fy * z
    pts = np.stack([X, Y, z], axis=-1)

    up = -np.array(g_cam)                       # world up, in camera coords
    height_above_cam = pts @ up                 # signed, camera at 0

    lower = np.zeros_like(depth, dtype=bool)
    lower[int(h * 0.45):, :] = True
    m = lower & np.isfinite(z) & (z > 0.5) & (z < 60.0)
    if m.sum() < 500:
        return None, None

    vals = height_above_cam[m]
    hist, edges = np.histogram(vals, bins=200,
                               range=(np.percentile(vals, 1), np.percentile(vals, 60)))
    peak = edges[int(np.argmax(hist))] + (edges[1] - edges[0]) * 0.5
    inliers = vals[np.abs(vals - peak) < 0.15]
    ground_offset = float(np.median(inliers)) if inliers.size > 200 else float(peak)
    return -ground_offset, float(inliers.size / max(m.sum(), 1))


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class SolveRequest(BaseModel):
    image_path: str
    focal_override: float | None = None      # 35mm-equivalent mm
    refine: bool = False                     # add Marigold detail pass
    depth_long_edge: int = 1536              # inference resolution
    assumed_eye_height: float = 1.55         # fallback when gravity is absent


@app.get("/health")
def health():
    return {"ok": True, "device": str(DEVICE), "loaded": list(_models)}


@app.post("/solve")
def solve(req: SolveRequest):
    path = os.path.expanduser(req.image_path)
    if not os.path.exists(path):
        raise HTTPException(404, f"not found: {path}")

    exif = read_exif(path)
    K = intrinsics_from_exif(exif, req.focal_override)
    g_cam = gravity_from_exif(exif)
    geo = geo_from_exif(exif)
    sun = sun_from_geo(geo) if geo else None

    img = Image.open(path).convert("RGB")
    scale = req.depth_long_edge / max(img.size)
    if scale < 1.0:
        img_small = img.resize((round(img.width * scale), round(img.height * scale)),
                               Image.LANCZOS)
    else:
        img_small = img

    f_px_small = K.fx * (img_small.width / K.width)
    depth = run_depth_pro(img_small, f_px_small)
    if req.refine:
        depth = refine_with_marigold(img_small, depth)

    cam_height, ground_conf = solve_camera_height(depth, K, g_cam)
    if cam_height is None:
        cam_height, ground_conf = req.assumed_eye_height, 0.0

    stem = os.path.splitext(os.path.basename(path))[0]
    depth_path = os.path.join(CACHE, f"{stem}_depth.npy")
    np.save(depth_path, depth.astype(np.float32))

    # Blender can't open HEIC — hand it a PNG plate at full resolution.
    plate_path = os.path.join(CACHE, f"{stem}_plate.png")
    if not os.path.exists(plate_path):
        img.save(plate_path)

    return {
        "intrinsics": asdict(K),
        "gravity_cam": g_cam,
        "camera_height_m": cam_height,
        "ground_confidence": ground_conf,
        "heading_deg": (float(geo["heading"]) if geo and geo.get("heading") else None),
        "sun": sun,
        "geo": geo,
        "depth_npy": depth_path,
        "depth_shape": list(depth.shape),
        "plate_png": plate_path,
        "depth_range_m": [float(np.nanmin(depth)), float(np.nanpercentile(depth, 99))],
    }
