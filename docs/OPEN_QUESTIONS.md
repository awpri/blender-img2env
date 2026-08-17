# Open questions

Where the work stopped, and what to check first. Written at v0.9.0 so the next
session does not have to re-derive any of it.

The headline: **CG objects still cast no visible shadow onto the plate**, and
several rounds of fixes have not resolved it. Each of those fixes was a real
bug — they are listed at the bottom — but none was the one that matters.

---

## 1. The verification is not checking what ships. Resolve this first.

`tools/blender_smoke_test.py` contains:

```python
nodes = {n.type for n in material.node_tree.nodes}
assert {"MIX_SHADER", "BSDF_TRANSPARENT", "LIGHT_PATH", "TEX_IMAGE"} <= nodes
```

It passes. But calling `proxy.make_proxy_material()` directly in the same
Blender produces only four nodes:

```
DIRECT call -> ['BSDF_PRINCIPLED', 'OUTPUT_MATERIAL', 'TEX_COORD', 'TEX_IMAGE']
  output fed by: BSDF_PRINCIPLED
```

and a real solve leaves `Photo3D_ProxyMat` with those same four. The Mix
Shader, Transparent BSDF and Light Path are absent, and the Principled feeds
the output directly.

Both cannot be true of the same code. Until that contradiction is explained,
**no "all Blender checks passed" result from this suite means anything about
the proxy shader**, and several such results were reported during development.

Likely candidates, in order:

- `nodes.new()` silently failing for those three identifiers on Blender 5.2,
  with the smoke test inspecting a material built at a different moment.
- The smoke test reading a *different* material than the one the solve ships
  (a leftover from an earlier build step in the same session).

Reproduce with `tools/` scratch scripts or:

```bash
blender --background --factory-startup --python-expr "
import sys; sys.path.insert(0,'addon')
import bpy, photo3d; photo3d.register()
from photo3d import proxy
m = proxy.make_proxy_material(bpy.data.images.new('t',8,8))
print(sorted(n.type for n in m.node_tree.nodes))"
```

## 2. Does the Shadow Catcher pass contain anything?

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

Note the interaction: **if the proxy is a plain Principled surface** (per
question 1) that is *correct* for shadow catching, so the missing nodes are
probably not the shadow bug. Do not conflate the two.

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
