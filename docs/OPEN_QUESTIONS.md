# Open questions

Where the work stopped, and what to check first. Written at v0.9.0 so the next
session does not have to re-derive any of it.

The headline: the user reports **CG objects cast no visible shadow onto the
plate**, while the same pipeline measurably does cast one here. Several rounds
of fixes have not closed that gap. Each fix was a real bug — they are listed at
the bottom — but none was the one that matters to them.

---

## 1. RESOLVED — this was my error, not a defect

The previous version of this file claimed `make_proxy_material` was silently
dropping three nodes and that the smoke test was therefore lying. **Both claims
were wrong.**

The four-node material is deliberate. Commit `ddb5128` removed the
Is-Camera-Ray/Transparent branch on purpose, because it cancels the shadow
catcher: a camera ray that passes straight through never hits the catcher, so
Cycles has no surface on which to compute the shadow ratio. Measured there,
same scene, only the material differing:

    Is Camera Ray -> Transparent :   0 shadow pixels
    plain Principled             : 283 shadow pixels, mean alpha 0.82

The smoke test was updated in the same commit and no longer asserts those
nodes. There was never a contradiction — I compared the current code against a
stale memory of it.

## 1b. So does it work? On this machine, yes — measured.

A full solve of `IMG_9920.DNG` with a cube in front of the camera, rendered
twice with the caster shown and hidden:

    14,637 of 486,824 pixels darkened by the cube (3.0%)
    mean darkening 0.156

So the committed pipeline does put a shadow on the plate for that exact photo.
Which means the remaining failure is something about the scene in front of the
user, not the code path — and guessing at it from here has repeatedly failed.

`Measure Shadow` (next to Diagnose in the panel) now does the same measurement
in whatever scene it is run in, and says explicitly when the number is zero.
That turns "there are still no shadows" into a number both sides can read.

## 2. Does the Shadow Catcher pass contain anything? (still worth knowing)

v0.8.0 changed the compositor to multiply the plate by the Shadow Catcher
pass, on the reasoning that Cycles returns shadow-catcher shadows as a
multiplier with zero alpha rather than as dark premultiplied pixels. A test
render on `IMG_9920.DNG` did then show a shadow — but the user still sees none.

**The pass content has never actually been measured on a real solve.** If it is
1.0 everywhere, the multiply is inert and the v0.8.0 fix does nothing, which
would fit the reports exactly.

Measure it by wiring `R_LAYERS ▸ Shadow Catcher` straight to the compositor
output, rendering to EXR, and reporting min/mean. A scratch script that does
this got as far as failing on question 1 above.

Less urgent than it looked, now that the end-to-end measurement above shows a
shadow does arrive. But it is still unmeasured, and if the pass were inert the
shadow would have to be arriving by some other route than intended.

## 3. Smaller, genuinely open

- **Segmentation coverage.** SAM 2 finds few regions; raising the resolution to
  1600 did not help much. `points_per_batch` is the more likely lever than
  resolution — its automatic mode is a point grid — but this has not been
  measured against the plate.
- **Shadows on segments vs the proxy.** With several shadow catchers in the
  scene, shadows were reported on the segments only when the proxy was hidden.
  Multiple catchers may not interact the way the single-catcher case does.
- **Ground-plane inlier band** is a fixed 0.15 m window, so `depth_scale`
  scales the noise with it and confidence falls (0.48 → 0.18 on IMG_7263).
  Needs a scene with flat known ground to tune against.

---

## Fixed along the way, for the record

All real, none of them the shadow bug:

| | |
|---|---|
| Compositor pasted the plate at its own resolution | centre crop at any Render % below 100 |
| Light sliders read only at solve time | dragging them did nothing |
| Blender kept stale submodules on upgrade | new version number, old code |
| `props` unassigned in a panel `draw()` | section rendered empty, buttons never created |
| Sky irradiance constant guessed at 2.0 | measured ~42; exposure hinged entirely on sky strength |
| Sun/sky split uncalibrated | sky 4x the sun, so light came from everywhere and cast nothing |
| Exposure probe leaked `file_format` | every later render silently wrote EXR; gobo bake crashed |
| Glass regions made camera-visible | rendered a second sheet over the one in the photo |
