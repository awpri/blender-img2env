# Open questions

Where the work stopped, and what to check first. Written at v0.9.0 so the next
session does not have to re-derive any of it.

**Largely closed at v0.12.0.** The composite now matches the photograph: no
doubling, no saturation, no dark layer, no colour striping, measured hue shift
against the plate of +0.003. What remains is listed under "Smaller" below.

The long-running "no shadow" report turned out to be several separate bugs
stacked, all of the same shape — adding light to, or taking light from, a
photograph that already contained it. They are listed at the bottom.

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

## 2. RESOLVED — measured, and it needed two corrections

Measured on a solved scene with the bounce proxy running:

    min 0.244   mean 1.002   max 1.476   91% of the frame exactly 1.0

It is a genuine multiplier, but it is neither bounded at 1 nor neutral, and
both mattered:

* **Above 1** wherever the bounce proxy lands, because the catcher really does
  receive that light. Multiplying the plate by it blew the photograph out.
* **Coloured**, so clamping per channel turned (1.2, 1.0, 0.8) into
  (1.0, 1.0, 0.8) — a tint, not a darkening. That was the blue and yellow
  striping.

It is now collapsed to luminance and capped as a scalar, so it can only darken
and only neutrally. A real shadow is slightly blue under open sky, but that
colour is already in the plate; this pass says how MUCH light the CG removed,
not what colour to make the result.

## 2b. Historical note (superseded)

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
| Bounce proxy counted as CG by the shadow-catcher pass | see below |

### The bounce proxy was re-lighting the photograph

The artifacts around the glass and the train were this. The shadow-catcher
pass is a ratio: light reaching the catcher WITH the CG objects over the light
reaching it WITHOUT them. Cycles counted the bounce proxy as CG, so it appeared
in the numerator only. Being solid to diffuse and glossy rays, it occluded sky
from the shadow proxy standing right behind it and substituted plate-derived
emission — dimmer than the sky under the canopy, brighter than it around the
glass. The ratio left 1 in both directions and the compositor applied the
difference to a photograph that already looked exactly the way it should.

Measured on IMG_9920, pixels more than 2% darkened, and the mean multiplier:

| | before | after |
|---|---|---|
| whole frame | 14.43%, mean 35.0 | 0.86%, mean 0.994 |
| canopy | 26.92% | 2.33% |
| train | 12.23% | 0.05% |

The mean of 35 is the bright half of the same bug, and is why the glass had
fringes rather than just shadows.

The fix is one flag: the bounce proxy is marked a shadow catcher too. It is
camera-invisible so it never catches anything; the flag only tells Cycles which
side of the ratio it belongs on. On the non-CG side it appears in both legs and
cancels, and because Cycles keeps shadow catchers visible to indirect rays it
goes on lighting CG unchanged (a white sphere sat at 1.13x the no-bounce
brightness before and after).

Light linking was tried first, excluding the catchers as receivers of the
bounce. It fails, because the problem is occlusion and light linking governs
illumination: the emission stopped, the occlusion did not, and the darkened
fraction went to 80%. `tools/render_checks.py` now measures both halves — that
attempt would have passed a check that only looked at the plate.
