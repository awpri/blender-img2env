"""EXIF -> intrinsics, gravity, GPS, sun. numpy + stdlib + exiftool.

The premise of the whole pipeline is that an iPhone already recorded the camera
state at the moment of capture, so none of it has to be inferred from the image.
This module is where that claim is cashed in.

It deliberately imports nothing from fastapi: everything here has to be
callable from a test with a JSON fixture and no server running, which is what
makes milestone M1 checkable without a photograph.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ._core import coords
from .solar import solar_position


class Photo3DError(RuntimeError):
    """Anything the caller could fix by supplying a different photo or flag."""


# ---------------------------------------------------------------------------
# reading exiftool output
# ---------------------------------------------------------------------------
#
# exiftool is the only thing that reads Apple's MakerNote. PIL and piexif drop
# it silently, which is worse than failing.

EXIFTOOL_ARGS = ("-j", "-n", "-G0:1")


def read_exif(path: str | Path) -> dict:
    """Run exiftool and return its JSON for one file.

    -n keeps numbers numeric instead of pretty-printing them ("76.944" rather
    than "76.9 deg"), and -G0:1 prefixes every key with its group so that the
    four different ImageWidth tags in a DNG stay distinguishable.
    """
    try:
        out = subprocess.run(["exiftool", *EXIFTOOL_ARGS, str(path)],
                             capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise Photo3DError(
            "exiftool not found. brew install exiftool — it is the only thing "
            "that reads Apple's MakerNote gravity vector.") from None
    except subprocess.CalledProcessError as exc:
        raise Photo3DError(f"exiftool failed on {path}: {exc.stderr.strip()}") from None
    records = json.loads(out.stdout)
    if not records:
        raise Photo3DError(f"exiftool returned no metadata for {path}")
    return records[0]


def load_exif_json(path: str | Path) -> dict:
    """Load a saved `exiftool -j -n -G0:1` dump. Fixtures use this."""
    data = json.loads(Path(path).read_text())
    return data[0] if isinstance(data, list) else data


def find(exif: dict, *names: str, prefer: tuple[str, ...] = ()):
    """Look a tag up by its bare name across all groups.

    `prefer` is an ordered list of group prefixes to try first. It matters more
    than it looks: in a DNG, IFD0 describes the embedded thumbnail while the
    real image is in a SubIFD, so an unqualified ImageWidth can be 256 px.
    """
    matches: list[tuple[str, object]] = []
    for key, value in exif.items():
        if key.split(":")[-1] in names:
            matches.append((key, value))
    if not matches:
        return None
    for group in prefer:
        for key, value in matches:
            if key.startswith(group):
                return value
    return matches[0][1]


def _floats(value) -> list[float] | None:
    """exiftool returns list-valued tags as either a list or a spaced string."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
        try:
            return [float(p) for p in parts]
        except ValueError:
            return None
    return [float(value)]


# ---------------------------------------------------------------------------
# intrinsics
# ---------------------------------------------------------------------------

#: How to turn a 35mm-equivalent focal length into pixels.
#:
#: "long_edge" assumes the equivalence is against the 36 mm long edge of a 35mm
#: frame. "diagonal" assumes it is against the 43.27 mm diagonal, which is the
#: textbook definition of a crop factor.
#:
#: On a 4:3 phone sensor these differ by about 4%, which on a 14 mm ultra-wide
#: is several degrees of field of view — enough to see a backplate drift away
#: from the CG at the frame edges. Apple does not document which it uses. The
#: default is long_edge because it matches Blender's sensor_fit='AUTO' with a
#: 36 mm sensor exactly; if the M2 backplate check shows a consistent scale
#: error, switch it, do not nudge the focal length. See docs/DECISIONS.md.
FOCAL_CONVENTIONS = ("long_edge", "diagonal")

FRAME_LONG_EDGE_MM = 36.0
FRAME_DIAGONAL_MM = math.hypot(36.0, 24.0)


@dataclass
class Intrinsics:
    """Pinhole intrinsics in DISPLAY pixel space (EXIF Orientation applied)."""
    width: int
    height: int
    focal_35mm: float
    fx: float
    fy: float
    cx: float
    cy: float
    convention: str = "long_edge"

    @property
    def horizontal_fov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.width / (2.0 * self.fx)))

    @property
    def vertical_fov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.height / (2.0 * self.fy)))


def orientation_from_exif(exif: dict) -> int:
    value = find(exif, "Orientation", prefer=("EXIF:IFD0", "EXIF", "MakerNotes"))
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def stored_image_size(exif: dict) -> tuple[int, int]:
    """(width, height) of the stored pixel array, before orientation."""
    size = _floats(find(exif, "ImageSize", prefer=("Composite",)))
    if size and len(size) == 2:
        return int(size[0]), int(size[1])
    w = find(exif, "ImageWidth", "ExifImageWidth",
             prefer=("File", "EXIF:SubIFD", "EXIF:ExifIFD", "EXIF:IFD0"))
    h = find(exif, "ImageHeight", "ExifImageHeight",
             prefer=("File", "EXIF:SubIFD", "EXIF:ExifIFD", "EXIF:IFD0"))
    if w is None or h is None:
        raise Photo3DError("no image dimensions in EXIF")
    return int(w), int(h)


def focal_pixels(focal_35mm: float, width: int, height: int,
                 convention: str = "long_edge") -> float:
    if convention == "long_edge":
        return focal_35mm / FRAME_LONG_EDGE_MM * max(width, height)
    if convention == "diagonal":
        return focal_35mm / FRAME_DIAGONAL_MM * math.hypot(width, height)
    raise Photo3DError(f"unknown focal convention {convention!r}; "
                       f"expected one of {FOCAL_CONVENTIONS}")


def intrinsics_from_exif(exif: dict, focal_override: float | None = None,
                         convention: str = "long_edge") -> Intrinsics:
    """Intrinsics in display space.

    Orientation 6 and 8 store the image sideways, so width and height are
    swapped here. Every downstream consumer — depth, the proxy mesh, the
    Window-projected plate — works in display space, so this is the only place
    that is allowed to know the difference.
    """
    stored_w, stored_h = stored_image_size(exif)
    orientation = orientation_from_exif(exif)
    if orientation in coords.TRANSPOSED_ORIENTATIONS:
        width, height = stored_h, stored_w
    else:
        width, height = stored_w, stored_h

    f35 = focal_override or find(exif, "FocalLengthIn35mmFormat", "FocalLengthIn35mmFilm",
                                 prefer=("EXIF:ExifIFD", "Composite", "EXIF"))
    if f35 is None:
        raise Photo3DError(
            "no 35mm-equivalent focal length in EXIF; pass focal_override")
    f35 = float(f35)

    f_px = focal_pixels(f35, width, height, convention)
    return Intrinsics(width=width, height=height, focal_35mm=f35,
                      fx=f_px, fy=f_px, cx=width / 2.0, cy=height / 2.0,
                      convention=convention)


# ---------------------------------------------------------------------------
# gravity
# ---------------------------------------------------------------------------

@dataclass
class Orientation3D:
    gravity_device: list[float]
    gravity_camera: list[float]
    pitch_deg: float
    roll_deg: float
    exif_orientation: int
    magnitude_g: float

    @property
    def is_static(self) -> bool:
        """A magnitude far from 1 g means the phone was accelerating, so the
        vector is not purely gravity and pitch/roll are correspondingly off."""
        return 0.93 <= self.magnitude_g <= 1.07


def orientation_from_gravity(exif: dict) -> Orientation3D | None:
    """Pitch and roll, MEASURED by the accelerometer, not inferred from the image.

    Apple MakerNote 0x0008 AccelerationVector is the gravity vector in device
    coordinates at capture. Returns None when the tag is absent — a JPEG
    re-exported by some tools loses the MakerNote, and that is a caller
    decision, not an error.
    """
    raw = _floats(find(exif, "AccelerationVector", prefer=("MakerNotes:Apple", "MakerNotes")))
    if not raw or len(raw) != 3:
        return None
    magnitude = math.sqrt(sum(v * v for v in raw))
    if magnitude < 1e-6:
        return None

    exif_orientation = orientation_from_exif(exif)
    g_cam = coords.gravity_device_to_camera(raw, exif_orientation)
    pitch, roll = coords.pitch_roll_from_gravity(g_cam)
    return Orientation3D(gravity_device=[v / magnitude for v in raw],
                         gravity_camera=g_cam.tolist(),
                         pitch_deg=pitch, roll_deg=roll,
                         exif_orientation=exif_orientation,
                         magnitude_g=magnitude)


# ---------------------------------------------------------------------------
# GPS, time and the sun
# ---------------------------------------------------------------------------

@dataclass
class Geo:
    latitude: float | None
    longitude: float | None
    altitude_m: float
    utc: datetime | None
    heading_deg: float | None
    heading_is_true: bool
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["utc"] = self.utc.isoformat() if self.utc else None
        return d


def _capture_utc(exif: dict) -> datetime | None:
    """UTC of the shutter.

    GPSDateStamp/GPSTimeStamp are already UTC and carry no timezone ambiguity,
    so they win. DateTimeOriginal is local wall-clock and is only usable when
    OffsetTimeOriginal is there to say which local.
    """
    gps_dt = find(exif, "GPSDateTime", prefer=("Composite",))
    if isinstance(gps_dt, str):
        cleaned = gps_dt.strip().rstrip("Z")
        try:
            return datetime.strptime(cleaned[:19], "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    date = find(exif, "GPSDateStamp", prefer=("EXIF:GPS",))
    time_ = find(exif, "GPSTimeStamp", prefer=("EXIF:GPS",))
    if isinstance(date, str) and isinstance(time_, str):
        try:
            return datetime.strptime(f"{date[:10]} {time_[:8]}",
                                     "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    local = find(exif, "DateTimeOriginal", "CreateDate", prefer=("EXIF:ExifIFD", "EXIF"))
    if not isinstance(local, str):
        return None
    try:
        naive = datetime.strptime(local[:19], "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None

    offset = find(exif, "OffsetTimeOriginal", "OffsetTime", prefer=("EXIF:ExifIFD", "EXIF"))
    if isinstance(offset, str) and len(offset) >= 6 and offset[0] in "+-":
        sign = 1 if offset[0] == "+" else -1
        delta = timedelta(hours=int(offset[1:3]), minutes=int(offset[4:6]))
        return (naive - sign * delta).replace(tzinfo=timezone.utc)
    # No offset: assume the wall clock was UTC and let the caller see the warning.
    return naive.replace(tzinfo=timezone.utc)


def geo_from_exif(exif: dict) -> Geo:
    warnings: list[str] = []

    lat = find(exif, "GPSLatitude", prefer=("Composite", "EXIF:GPS"))
    lon = find(exif, "GPSLongitude", prefer=("Composite", "EXIF:GPS"))
    lat = float(lat) if lat is not None else None
    lon = float(lon) if lon is not None else None
    if lat is None or lon is None:
        # The classic failure: Photos' default JPEG export keeps GPSLatitudeRef
        # ("N") and drops the coordinate it refers to. No coordinates, no sun.
        if find(exif, "GPSLatitudeRef", "GPSLongitudeRef") is not None:
            warnings.append(
                "GPS Ref fields present but coordinates missing — the export "
                "stripped location. Re-export with Photos > File > Export > "
                "Export Unmodified Original. Without this there is no sun.")
        else:
            warnings.append("no GPS coordinates; solar ephemeris unavailable")

    utc = _capture_utc(exif)
    if utc is None:
        warnings.append("no usable capture timestamp; solar ephemeris unavailable")
    elif find(exif, "GPSDateTime", "GPSDateStamp") is None and \
            find(exif, "OffsetTimeOriginal", "OffsetTime") is None:
        warnings.append("capture time has no timezone; assumed UTC, so the sun "
                        "may be hours out. Check OffsetTimeOriginal.")

    heading = find(exif, "GPSImgDirection", prefer=("EXIF:GPS", "Composite"))
    heading = float(heading) if heading is not None else None
    heading_ref = find(exif, "GPSImgDirectionRef", prefer=("EXIF:GPS",))
    heading_is_true = str(heading_ref).upper().startswith("T") if heading_ref else True
    if heading is not None and not heading_is_true:
        warnings.append("compass heading is MAGNETIC, not true north. No offline "
                        "declination model here, so yaw is off by the local "
                        "declination until you correct it by hand.")

    altitude = find(exif, "GPSAltitude", prefer=("Composite", "EXIF:GPS"))
    try:
        altitude_m = float(altitude)
    except (TypeError, ValueError):
        altitude_m = 0.0

    return Geo(latitude=lat, longitude=lon, altitude_m=altitude_m, utc=utc,
               heading_deg=heading, heading_is_true=heading_is_true, warnings=warnings)


@dataclass
class Sun:
    azimuth_deg: float
    elevation_deg: float
    utc: str
    above_horizon: bool


def sun_from_geo(geo: Geo) -> Sun | None:
    """Physically correct sun position from where and when the shutter fired.

    This beats every image-based light estimator and costs nothing, but it is
    only half the lighting story — see LIGHTING_ADDENDUM.md. It gives the
    direction and hardness of the direct sun; the bounce proxy gives everything
    else.
    """
    if geo.latitude is None or geo.longitude is None or geo.utc is None:
        return None
    azimuth, elevation = solar_position(geo.utc, geo.latitude, geo.longitude)
    return Sun(azimuth_deg=azimuth, elevation_deg=elevation,
               utc=geo.utc.isoformat(), above_horizon=elevation > -0.833)


# ---------------------------------------------------------------------------
# everything at once
# ---------------------------------------------------------------------------

@dataclass
class CameraState:
    """Complete camera solve from metadata alone. No pixels were consulted."""
    intrinsics: Intrinsics
    orientation: Orientation3D | None
    geo: Geo
    sun: Sun | None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "intrinsics": asdict(self.intrinsics),
            "orientation": asdict(self.orientation) if self.orientation else None,
            "gravity_camera": self.orientation.gravity_camera if self.orientation else None,
            "pitch_deg": self.orientation.pitch_deg if self.orientation else None,
            "roll_deg": self.orientation.roll_deg if self.orientation else None,
            "heading_deg": self.geo.heading_deg,
            "geo": self.geo.as_dict(),
            "sun": asdict(self.sun) if self.sun else None,
            "warnings": self.warnings,
        }


def camera_state_from_exif(exif: dict, focal_override: float | None = None,
                           convention: str = "long_edge") -> CameraState:
    intrinsics = intrinsics_from_exif(exif, focal_override, convention)
    orientation = orientation_from_gravity(exif)
    geo = geo_from_exif(exif)
    sun = sun_from_geo(geo)

    warnings = list(geo.warnings)
    if orientation is None:
        warnings.append(
            "no AccelerationVector in MakerNote — pitch and roll are unknown. "
            "Either the file is not an unmodified Apple original, or it came "
            "from another camera. Fall back to horizon-line detection or a "
            "hand-set pitch.")
    elif not orientation.is_static:
        warnings.append(
            f"accelerometer read {orientation.magnitude_g:.3f} g, not 1.000 — "
            "the phone was moving, so pitch and roll carry that error.")
    if geo.heading_deg is None:
        warnings.append("no GPSImgDirection; yaw is unconstrained and the sun "
                        "will land in an arbitrary part of the sky.")
    return CameraState(intrinsics=intrinsics, orientation=orientation, geo=geo,
                       sun=sun, warnings=warnings)


def camera_state_from_path(path: str | Path, focal_override: float | None = None,
                           convention: str = "long_edge") -> CameraState:
    return camera_state_from_exif(read_exif(path), focal_override, convention)
