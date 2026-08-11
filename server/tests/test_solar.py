"""Solar position, checked against physics rather than against itself.

Most of these pin facts about the sky that are true regardless of which
algorithm computes them, so they would still catch an error if the whole NOAA
implementation were replaced. The astral cross-check at the bottom is the
independent second opinion, and it skips cleanly when astral is absent because
the point of implementing this in stdlib was to not need it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from server.solar import julian_day, solar_position

UTC = timezone.utc


def _max_elevation_over_day(date, lat, lon, step_minutes=2):
    """Peak elevation and the azimuth at that moment — local solar noon, found
    by search so no equation-of-time assumption sneaks into the test."""
    best = (-90.0, 0.0)
    t = datetime(date.year, date.month, date.day, tzinfo=UTC)
    for _ in range(24 * 60 // step_minutes):
        azimuth, elevation = solar_position(t, lat, lon)
        if elevation > best[0]:
            best = (elevation, azimuth)
        t += timedelta(minutes=step_minutes)
    return best


def test_julian_day_epoch():
    """J2000.0 is exactly 2451545.0 by definition."""
    assert julian_day(datetime(2000, 1, 1, 12, 0, 0, tzinfo=UTC)) == pytest.approx(2451545.0)
    assert julian_day(datetime(2000, 1, 2, 0, 0, 0, tzinfo=UTC)) == pytest.approx(2451545.5)


def test_julian_day_requires_a_timezone():
    """A naive datetime here means the sun lands hours away, silently."""
    with pytest.raises(ValueError):
        julian_day(datetime(2026, 7, 14, 9, 11, 36))


def test_solstice_sun_is_overhead_at_the_tropic_of_cancer():
    elevation, _ = _max_elevation_over_day(datetime(2026, 6, 21), 23.44, 0.0)
    assert elevation > 89.0


def test_northern_hemisphere_noon_sun_is_due_south():
    _, azimuth = _max_elevation_over_day(datetime(2026, 7, 14), 46.2, 6.14)
    assert azimuth == pytest.approx(180.0, abs=1.0)


def test_southern_hemisphere_noon_sun_is_due_north():
    _, azimuth = _max_elevation_over_day(datetime(2026, 7, 14), -33.9, 151.2)
    assert azimuth % 360.0 == pytest.approx(0.0, abs=1.0) or \
           azimuth == pytest.approx(360.0, abs=1.0)


def test_sun_rises_in_the_east_and_sets_in_the_west():
    """Morning azimuth under 180, afternoon over. Catches a sign flip on the
    hour angle, which would mirror every cast shadow in the scene."""
    morning = solar_position(datetime(2026, 7, 14, 6, 30, tzinfo=UTC), 46.2, 6.14)
    evening = solar_position(datetime(2026, 7, 14, 17, 30, tzinfo=UTC), 46.2, 6.14)
    assert 45.0 < morning[0] < 180.0
    assert 180.0 < evening[0] < 315.0


def test_polar_night():
    """Nothing above the horizon at the South Pole in June, all day."""
    elevation, _ = _max_elevation_over_day(datetime(2026, 6, 21), -89.9, 0.0)
    assert elevation < 0.0


def test_equation_of_time_is_actually_applied():
    """In early November solar noon runs ~16 minutes ahead of clock noon. If the
    equation of time were dropped, this would come out near zero and the sun
    would be up to 4 degrees off — visible in a hard cast shadow."""
    _, true_noon_azimuth = _max_elevation_over_day(datetime(2026, 11, 3), 46.2, 0.0, step_minutes=1)
    clock_noon_azimuth, _ = solar_position(datetime(2026, 11, 3, 12, 0, tzinfo=UTC), 46.2, 0.0)
    assert true_noon_azimuth == pytest.approx(180.0, abs=0.6)
    assert abs(clock_noon_azimuth - 180.0) > 2.0


def test_refraction_lifts_the_sun_near_the_horizon():
    when = datetime(2026, 7, 14, 4, 0, tzinfo=UTC)
    with_refraction = solar_position(when, 46.2, 6.14, refraction=True)[1]
    without = solar_position(when, 46.2, 6.14, refraction=False)[1]
    assert 0.0 < with_refraction - without < 1.0


def test_longitude_shifts_solar_noon_by_four_minutes_per_degree():
    a = _max_elevation_over_day(datetime(2026, 7, 14), 46.2, 0.0, step_minutes=1)
    b = _max_elevation_over_day(datetime(2026, 7, 14), 46.2, 15.0, step_minutes=1)
    assert a[0] == pytest.approx(b[0], abs=0.05)


@pytest.mark.parametrize("when,lat,lon", [
    (datetime(2026, 7, 14, 9, 11, 36, tzinfo=UTC), 46.2044, 6.1432),
    (datetime(2026, 12, 21, 15, 0, 0, tzinfo=UTC), 60.17, 24.94),
    (datetime(2026, 3, 20, 22, 30, 0, tzinfo=UTC), -33.87, 151.21),
    (datetime(2026, 9, 5, 18, 45, 0, tzinfo=UTC), 37.77, -122.42),
])
def test_agrees_with_astral(when, lat, lon):
    """Independent second opinion. Half a degree is far tighter than the compass
    heading this gets combined with, so any disagreement inside it is noise."""
    astral = pytest.importorskip("astral", reason="optional cross-check dependency")
    from astral.sun import azimuth as astral_azimuth, elevation as astral_elevation

    observer = astral.Observer(latitude=lat, longitude=lon)
    mine_az, mine_el = solar_position(when, lat, lon)
    assert mine_el == pytest.approx(astral_elevation(observer, when), abs=0.5)
    if mine_el > 5.0:      # azimuth is ill-conditioned when the sun is on the horizon
        assert mine_az == pytest.approx(astral_azimuth(observer, when), abs=0.5)
