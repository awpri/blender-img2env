# EXIF fixtures

`exiftool -j -n -G0:1` dumps. No pixels — every one of these exercises a code
path that only reads metadata, which is the whole of milestone M1.

| file | what it pins |
|---|---|
| `img7096_exif.json` | The reference photo from `LIGHTING_ADDENDUM.md`. 14 mm ultra-wide, portrait, ProRAW, with the GPS coordinates missing exactly as the real export has them. This is the M1 acceptance fixture. |
| `orientation_{1,3,6,8}_exif.json` | One physical camera pose, four ways of holding the phone. They must all report the same pitch and roll. |
| `sunny_geneva_exif.json` | Has real coordinates and a UTC timestamp, so the solar ephemeris actually runs. |

## Where the orientation vectors come from

Not from running the code — that would only prove the axis map is
self-consistent. They are hand-derived from the physical pose, so a wrong
mapping fails the test instead of agreeing with itself.

The camera is pitched 3.28° above horizontal with 0.71° of roll. Gravity in
display-space camera coordinates is therefore `(-0.01246, 0.99829, -0.05720)`,
by `gravity_from_pitch_roll`, which is a closed-form expression with no axis
table in it.

Then, for each way of holding the phone, gravity in *device* coordinates:

| Orientation | phone held | device axis pointing down | vector |
|---|---|---|---|
| 6 | portrait, top edge up | −Y | `-0.01226 -0.98264 0.05630` |
| 8 | portrait, top edge down | +Y | `0.01226 0.98264 0.05630` |
| 1 | landscape, home button right | −X | `-0.98264 0.01226 0.05630` |
| 3 | landscape, home button left | +X | `0.98264 -0.01226 0.05630` |

The cross-check that makes these trustworthy: **the Z component is +0.05630 in
all four**. Device Z is the screen normal, and pitching the camera up by 3.28°
tips the screen back by 3.28° no matter how the phone is rotated about its own
optical axis. Any axis map that does not preserve that is wrong.

Magnitudes are 0.98433 g, matching the real capture — the phone was moving
slightly. It is left in rather than normalised so the tests run against
realistically imperfect input.
