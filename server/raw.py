"""DNG -> linear EXR lighting plate, and the display plate beside it.

The reason to use DNG here is not resolution. It is HIGHLIGHT HEADROOM.

The reference JPEG reports HDRHeadroom 1.01 — essentially none. Every pixel
brighter than diffuse white (sky, a specular glint off a lake, a window in an
interior) is clipped to 1.0, and those clipped regions carry most of the light
energy in the scene. A bounce proxy driven by a clipped plate systematically
underestimates the sky and renders flat, grey CG. ProRAW carries 12-14 stops of
linear scene radiance with the tone curve not yet applied, which is the
difference between measuring the sky's brightness and guessing at it.

So: two plates, two jobs. The beauty plate stays Apple-graded and
display-referred, because it is what the audience sees. The lighting plate
stays linear and unclipped, because it is what Cycles integrates.

Both come out in DISPLAY orientation, matching the intrinsics in exif.py.
rawpy honours the DNG's flip by default and PIL gets exif_transpose applied
explicitly; a plate that disagrees with the intrinsics puts the whole
Window-projected reprojection half a frame out.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RAW_SUFFIXES = {".dng", ".arw", ".cr2", ".cr3", ".nef", ".raf", ".orf", ".rw2"}


class PlateError(RuntimeError):
    pass


def is_raw(path: str | Path) -> bool:
    return Path(path).suffix.lower() in RAW_SUFFIXES


# ---------------------------------------------------------------------------
# lens opcodes — the 14 mm question
# ---------------------------------------------------------------------------

@dataclass
class WarpOpcodes:
    present: bool
    names: list[str]
    applied: bool
    note: str


def warp_opcodes(exif: dict) -> WarpOpcodes:
    """Report whether the DNG carries geometric-correction opcodes.

    Apple's ISP rectifies the processed JPEG, but a ProRAW DNG defers that to a
    WarpRectilinear opcode, and LibRaw/rawpy do NOT apply it. So switching to
    DNG for the lighting benefits above can reintroduce barrel distortion that
    the JPEG had already removed — on a 14 mm ultra-wide that is several pixels
    at the frame corners.

    DECISION: this pipeline does not apply them. See docs/DECISIONS.md. The
    lighting plate is integrated over solid angle, where a few pixels of edge
    distortion is irrelevant, and the display plate for a DNG source comes out
    of the same undistorted develop, so plate and CG at least agree with each
    other. If a long straight edge near a frame border bothers you, correct it
    once in the compositor's Lens Distortion node — it is a fixed lens, so one
    k1 solved once is good forever.

    This function exists so the situation is reported rather than discovered.
    """
    names: list[str] = []
    for key, value in exif.items():
        if "OpcodeList" in key.split(":")[-1] and value:
            names.extend(str(value).replace(",", " ").split())
    warp = [n for n in names if "Warp" in n or "Rectilinear" in n]
    if not warp:
        return WarpOpcodes(False, names, False,
                           "no geometric opcodes; the develop is already rectilinear")
    return WarpOpcodes(
        True, warp, False,
        "DNG carries " + ", ".join(sorted(set(warp))) + " and rawpy does not apply it. "
        "Expect barrel distortion at the frame edges that the processed JPEG "
        "does not have. Accepted by design — see docs/DECISIONS.md.")


# ---------------------------------------------------------------------------
# developing
# ---------------------------------------------------------------------------

#: Long edge the linear plate is developed at. Lighting is low-frequency —
#: a bounce proxy does not care about pixel-level detail — and a full 48 MP
#: float32 RGBA buffer is ~780 MB, which is a lot of unified memory to spend on
#: something that will be blurred by the first diffuse bounce anyway.
LIGHTING_LONG_EDGE = 2048


def _develop_linear_coreimage(path: str | Path, long_edge: int) -> np.ndarray:
    """Apple's own RAW decoder, via Core Image. macOS only.

    This exists because LibRaw 0.22 cannot open an iPhone 17 Pro ProRAW file at
    all: DNG 1.7 with JPEG XL compression returns "Unsupported file format or
    not RAW file". Core Image handles it natively, offline, with no extra
    download — it is the same decoder Photos uses.

    boostAmount=0 disables Apple's tone/boost curve, which is the whole point:
    the output stays linear and keeps the highlights a display-referred plate
    throws away. Measured on IMG_7263.DNG: max 2.89 against a median of 0.21,
    with 0.5% of pixels above 1.0.
    """
    import objc
    from Foundation import NSURL
    from Quartz import (CGColorSpaceCreateWithName, CIContext, CIFilter,
                        CIRAWFilter, kCGColorSpaceExtendedLinearSRGB, kCIFormatRGBAf)

    raw_filter = CIRAWFilter.filterWithImageURL_(NSURL.fileURLWithPath_(str(path)))
    if raw_filter is None:
        raise PlateError(f"Core Image could not open {path} as RAW")

    raw_filter.setBoostAmount_(0.0)                 # no tone curve; stay linear
    for setter, value in (("setGamutMappingEnabled_", False),
                          ("setExtendedDynamicRangeAmount_", 2.0)):
        if hasattr(raw_filter, setter):
            getattr(raw_filter, setter)(value)      # keep values above 1.0

    image = raw_filter.outputImage()
    if image is None:
        raise PlateError(f"Core Image produced no image for {path}")

    extent = image.extent()
    scale = min(1.0, long_edge / max(extent.size.width, extent.size.height))
    if scale < 1.0:
        resize = CIFilter.filterWithName_("CILanczosScaleTransform")
        resize.setValue_forKey_(image, "inputImage")
        resize.setValue_forKey_(scale, "inputScale")
        image = resize.outputImage()
        extent = image.extent()

    width, height = int(extent.size.width), int(extent.size.height)
    row_bytes = width * 4 * 4                        # RGBA, float32
    buffer = bytearray(row_bytes * height)
    context = CIContext.contextWithOptions_(None)
    context.render_toBitmap_rowBytes_bounds_format_colorSpace_(
        image, buffer, row_bytes, extent, kCIFormatRGBAf,
        CGColorSpaceCreateWithName(kCGColorSpaceExtendedLinearSRGB))

    del objc                                          # imported only to fail early
    return np.frombuffer(bytes(buffer), dtype=np.float32).reshape(height, width, 4)[..., :3].copy()


def _develop_linear_rawpy(path: str | Path, half_size: bool = False) -> np.ndarray:
    """LibRaw via rawpy. Broad camera support, but see the note above about
    DNG 1.7.

    gamma=(1,1) and no_auto_bright are the whole point: any tone curve or
    exposure stretch here would put the pipeline back where the JPEG was.
    """
    import rawpy

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            gamma=(1, 1),
            no_auto_bright=True,
            output_bps=16,
            output_color=rawpy.ColorSpace.sRGB,
            use_camera_wb=True,
            half_size=half_size,
        )
    return rgb.astype(np.float32) / 65535.0


def develop_linear(path: str | Path, long_edge: int = LIGHTING_LONG_EDGE) -> np.ndarray:
    """RAW -> linear scene-referred float32 RGB.

    Core Image first on macOS, because the target camera is an iPhone and
    Apple's decoder is the only one that reads its ProRAW; rawpy second,
    because it is the portable one. Both return display-oriented pixels, so
    the plate agrees with the intrinsics in exif.py.
    """
    attempts = []

    if sys.platform == "darwin":
        try:
            return _develop_linear_coreimage(path, long_edge)
        except ImportError as exc:
            attempts.append(f"Core Image: {exc} (pip install pyobjc-framework-Quartz)")
        except Exception as exc:                                  # noqa: BLE001
            attempts.append(f"Core Image: {type(exc).__name__}: {exc}")

    try:
        return _develop_linear_rawpy(path)
    except ImportError:
        attempts.append("rawpy: not installed (pip install rawpy)")
    except Exception as exc:                                      # noqa: BLE001
        attempts.append(f"rawpy/LibRaw: {exc}")

    raise PlateError(
        f"no RAW decoder could read {Path(path).name}. Tried:\n  "
        + "\n  ".join(attempts)
        + "\n\nAn iPhone 17 Pro ProRAW is DNG 1.7 with JPEG XL compression, "
        "which LibRaw 0.22 does not support. On macOS install "
        "pyobjc-framework-Quartz to use Apple's own decoder. Otherwise convert "
        "with the free Adobe DNG Converter first.")


def develop_display(path: str | Path) -> np.ndarray:
    """RAW -> display-referred RGB for the visible backplate.

    Apple's rendering is wanted here, not avoided: the beauty plate should look
    the way the photograph looks. Only the lighting plate needs to be linear.
    """
    if sys.platform == "darwin":
        try:
            from Foundation import NSURL
            from Quartz import (CIContext, CIRAWFilter,
                                CGColorSpaceCreateWithName, kCGColorSpaceSRGB,
                                kCIFormatRGBA8)

            raw_filter = CIRAWFilter.filterWithImageURL_(NSURL.fileURLWithPath_(str(path)))
            if raw_filter is not None and raw_filter.outputImage() is not None:
                image = raw_filter.outputImage()
                extent = image.extent()
                width, height = int(extent.size.width), int(extent.size.height)
                row_bytes = width * 4
                buffer = bytearray(row_bytes * height)
                CIContext.contextWithOptions_(None).render_toBitmap_rowBytes_bounds_format_colorSpace_(
                    image, buffer, row_bytes, extent, kCIFormatRGBA8,
                    CGColorSpaceCreateWithName(kCGColorSpaceSRGB))
                return np.frombuffer(bytes(buffer), dtype=np.uint8).reshape(
                    height, width, 4)[..., :3].copy()
        except Exception:                                          # noqa: BLE001
            pass                                                   # fall through to rawpy

    try:
        import rawpy
    except ImportError:
        raise PlateError("no RAW decoder for the display plate: install "
                         "pyobjc-framework-Quartz (macOS) or rawpy") from None

    with rawpy.imread(str(path)) as raw:
        return raw.postprocess(output_bps=8, use_camera_wb=True,
                               output_color=rawpy.ColorSpace.sRGB)


def load_display_image(path: str | Path):
    """Open any non-raw source as a PIL image, with EXIF orientation applied."""
    from PIL import Image, ImageOps
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def _write_exr_coreimage(data: np.ndarray, path: str) -> str:
    """EXR via Core Image. macOS 14+, and it needs nothing beyond the pyobjc
    bridge the RAW decoder already uses.

    Worth preferring over OpenCV specifically: Apple's Depth Pro pins numpy<2
    while opencv-python 5 requires numpy>=2, so installing both leaves cv2
    unimportable and takes the EXR writer down with it. This backend has no
    such conflict.

    Core Image's origin is bottom-left while a numpy image raster is top-down,
    so the rows are flipped on the way in. Getting that wrong puts the sky at
    the bottom of the lighting plate, which through Window coordinates lights
    the scene upside down.
    """
    from Foundation import NSData, NSURL
    from Quartz import (CGColorSpaceCreateWithName, CGSizeMake, CIContext, CIImage,
                        kCGColorSpaceExtendedLinearSRGB, kCIFormatRGBAf)

    height, width = data.shape[:2]
    rgba = np.ones((height, width, 4), dtype=np.float32)
    rgba[..., :3] = data[..., :3]
    rgba = np.ascontiguousarray(rgba[::-1])          # top-down -> bottom-up

    image = CIImage.imageWithBitmapData_bytesPerRow_size_format_colorSpace_(
        NSData.dataWithBytes_length_(rgba.tobytes(), rgba.nbytes),
        width * 16, CGSizeMake(width, height), kCIFormatRGBAf,
        CGColorSpaceCreateWithName(kCGColorSpaceExtendedLinearSRGB))
    if image is None:
        raise PlateError("Core Image could not wrap the pixel buffer")

    ok, error = CIContext.contextWithOptions_(None).\
        writeOpenEXRRepresentationOfImage_toURL_options_error_(
            image, NSURL.fileURLWithPath_(path), {}, None)
    if not ok:
        raise PlateError(f"Core Image EXR write failed: {error}")
    return path


def write_exr(image: np.ndarray, path: str | Path) -> str:
    """Write float32 RGB as a 32-bit EXR, using whichever backend is available.

    Blender reads EXR natively and treats it as linear, which is the only
    format in this pipeline that can carry a value above 1.0 — the entire
    reason for the DNG path.
    """
    path = str(path)
    data = np.ascontiguousarray(np.asarray(image, dtype=np.float32))
    errors = []

    if sys.platform == "darwin":
        try:
            return _write_exr_coreimage(data, path)
        except ImportError as exc:
            errors.append(f"Core Image: {exc} (pip install pyobjc-framework-Quartz)")
        except Exception as exc:                                  # noqa: BLE001
            errors.append(f"Core Image: {type(exc).__name__}: {exc}")

    try:
        import OpenImageIO as oiio
        spec = oiio.ImageSpec(data.shape[1], data.shape[0], data.shape[2], "float")
        out = oiio.ImageOutput.create(path)
        if out and out.open(path, spec):
            out.write_image(data)
            out.close()
            return path
        errors.append(f"OpenImageIO could not open {path}")
    except ImportError as exc:
        errors.append(f"OpenImageIO: {exc}")

    try:
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")   # must precede the import
        import cv2
        if cv2.imwrite(path, data[..., ::-1]):                   # cv2 wants BGR
            return path
        errors.append("cv2.imwrite returned False (OpenEXR support not compiled in?)")
    except ImportError as exc:
        errors.append(f"cv2: {exc}")

    try:
        import imageio.v3 as iio
        iio.imwrite(path, data)
        return path
    except Exception as exc:                                      # noqa: BLE001
        errors.append(f"imageio: {exc}")

    raise PlateError(
        "no EXR backend available; the linear lighting plate cannot be written.\n"
        "On macOS:  pip install pyobjc-framework-Quartz   (no numpy constraint)\n"
        "Otherwise: pip install 'opencv-python<5'         (opencv 5 needs numpy>=2, "
        "which conflicts with Depth Pro's numpy<2)\n"
        "Tried:\n  " + "\n  ".join(errors))


def write_png(image: np.ndarray, path: str | Path) -> str:
    from PIL import Image
    data = np.asarray(image)
    if data.dtype != np.uint8:
        data = (np.clip(data, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(data).save(str(path))
    return str(path)


# ---------------------------------------------------------------------------
# the pair of plates
# ---------------------------------------------------------------------------

@dataclass
class Plates:
    display_png: str
    lighting_exr: str | None
    source_is_raw: bool
    width: int
    height: int
    note: str


def make_plates(path: str | Path, cache_dir: str | Path, exif: dict | None = None,
                want_lighting_plate: bool = True) -> Plates:
    """Produce the display plate, and the linear lighting plate when possible.

    Blender cannot open HEIC or DNG, so the display plate is always written out
    as PNG at full resolution. The lighting plate is only meaningful from a raw
    source: developing a clipped JPEG into EXR would produce a linear file with
    the highlights still missing, which is worse than not having one because it
    looks like it should work.
    """
    path = Path(path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    display_path = cache_dir / f"{stem}_plate.png"
    notes = []

    if is_raw(path):
        display = develop_display(path)
        write_png(display, display_path)
        height, width = display.shape[:2]

        lighting_path = None
        if want_lighting_plate:
            linear = develop_linear(path)
            try:
                lighting_path = write_exr(linear, cache_dir / f"{stem}_lighting.exr")
                notes.append(f"linear plate peaks at {float(linear.max()):.3f}")
            except PlateError as exc:
                notes.append(str(exc))
        if exif is not None:
            opcodes = warp_opcodes(exif)
            if opcodes.present:
                notes.append(opcodes.note)
        return Plates(str(display_path), lighting_path, True, width, height,
                      "; ".join(notes))

    image = load_display_image(path)
    image.save(display_path)
    if want_lighting_plate:
        notes.append(
            "source is display-referred, so there is no lighting plate. Its "
            "highlights are already clipped; shoot ProRAW if the CG looks flat.")
    return Plates(str(display_path), None, False, image.width, image.height,
                  "; ".join(notes))
