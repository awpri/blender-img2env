"""Solar position from latitude, longitude and UTC time. stdlib only.

This is the NOAA solar position algorithm. It is here rather than behind
`astral` for two reasons: it removes a dependency from the one part of the
pipeline that has to work offline forever, and it is short enough to unit-test
against published values, which a third-party import is not.

Accuracy is a fraction of a degree over the years this pipeline will see —
several orders of magnitude better than the compass heading it gets combined
with, so it is never the limiting factor.

Angles out:
    azimuth    degrees clockwise from true north, matching GPSImgDirection.
    elevation  degrees above the horizon, refraction-corrected by default.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone


def julian_day(dt_utc: datetime) -> float:
    """Julian Day Number including the fractional day, from a UTC datetime."""
    if dt_utc.tzinfo is None:
        raise ValueError("solar position needs a timezone-aware UTC datetime")
    dt = dt_utc.astimezone(timezone.utc)

    year, month = dt.year, dt.month
    if month <= 2:                       # Jan/Feb count as months 13/14 of the previous year
        year -= 1
        month += 12
    a = year // 100
    b = 2 - a + a // 4                   # Gregorian correction
    day_fraction = (dt.hour + dt.minute / 60.0 + (dt.second + dt.microsecond / 1e6) / 3600.0) / 24.0
    return (math.floor(365.25 * (year + 4716))
            + math.floor(30.6001 * (month + 1))
            + dt.day + day_fraction + b - 1524.5)


def _refraction_deg(elevation_deg: float) -> float:
    """Atmospheric refraction lift, in degrees. Saemundsson's formula.

    Matters only near the horizon, where it is worth about half a degree — the
    difference between the sun having set and not, which decides whether the
    pipeline places a sun lamp at all.
    """
    if elevation_deg > 85.0:
        return 0.0
    e = math.radians(elevation_deg)
    if elevation_deg > 5.0:
        r = 58.1 / math.tan(e) - 0.07 / math.tan(e) ** 3 + 0.000086 / math.tan(e) ** 5
    elif elevation_deg > -0.575:
        r = 1735.0 + elevation_deg * (-518.2 + elevation_deg * (
            103.4 + elevation_deg * (-12.79 + elevation_deg * 0.711)))
    else:
        r = -20.774 / math.tan(e)
    return r / 3600.0


def solar_position(dt_utc: datetime, latitude: float, longitude: float,
                   refraction: bool = True) -> tuple[float, float]:
    """(azimuth, elevation) in degrees. Longitude is positive east."""
    jd = julian_day(dt_utc)
    t = (jd - 2451545.0) / 36525.0                       # Julian centuries since J2000.0

    mean_long = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360.0
    mean_anom = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    eccentricity = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)

    m = math.radians(mean_anom)
    centre = (math.sin(m) * (1.914602 - t * (0.004817 + 0.000014 * t))
              + math.sin(2 * m) * (0.019993 - 0.000101 * t)
              + math.sin(3 * m) * 0.000289)
    true_long = mean_long + centre
    apparent_long = true_long - 0.00569 - 0.00478 * math.sin(math.radians(125.04 - 1934.136 * t))

    mean_obliquity = 23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0
    obliquity = mean_obliquity + 0.00256 * math.cos(math.radians(125.04 - 1934.136 * t))

    declination = math.asin(math.sin(math.radians(obliquity)) * math.sin(math.radians(apparent_long)))

    # Equation of time: the gap between clock noon and the sun actually being
    # due south, worth up to ~16 minutes. Skipping it misplaces the sun by up
    # to 4 degrees, which is visible in a cast shadow.
    y = math.tan(math.radians(obliquity) / 2.0) ** 2
    l0 = math.radians(mean_long)
    eq_time = 4.0 * math.degrees(
        y * math.sin(2 * l0)
        - 2.0 * eccentricity * math.sin(m)
        + 4.0 * eccentricity * y * math.sin(m) * math.cos(2 * l0)
        - 0.5 * y * y * math.sin(4 * l0)
        - 1.25 * eccentricity * eccentricity * math.sin(2 * m))

    dt = dt_utc.astimezone(timezone.utc)
    minutes_utc = dt.hour * 60.0 + dt.minute + dt.second / 60.0 + dt.microsecond / 6e7
    true_solar_time = (minutes_utc + eq_time + 4.0 * longitude) % 1440.0
    hour_angle = math.radians(true_solar_time / 4.0 - 180.0)

    lat = math.radians(latitude)
    sin_elev = (math.sin(lat) * math.sin(declination)
                + math.cos(lat) * math.cos(declination) * math.cos(hour_angle))
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, sin_elev))))

    # atan2 form rather than the NOAA spreadsheet's acos-with-a-branch: no
    # quadrant special case to get wrong, and it stays continuous through noon.
    azimuth = math.degrees(math.atan2(
        -math.cos(declination) * math.sin(hour_angle),
        math.sin(declination) * math.cos(lat)
        - math.cos(declination) * math.sin(lat) * math.cos(hour_angle))) % 360.0

    if refraction:
        elevation += _refraction_deg(elevation)
    return azimuth, elevation
