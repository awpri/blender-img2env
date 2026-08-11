# Decisions

Things that were genuinely ambiguous, what was chosen, and what would change
the choice. The handoff asked for several of these explicitly.

---

## 1. DNG `WarpRectilinear` opcodes are NOT applied

**Decision: accept the distortion. Document it. Do not correct it in the solver.**

Apple's ISP rectifies the processed JPEG. A ProRAW DNG defers that to a
`WarpRectilinear` opcode in `OpcodeList3`, and LibRaw/rawpy do not apply it. So
moving to DNG for the highlight headroom can reintroduce barrel distortion that
the JPEG had already removed — on a 14 mm ultra-wide that is several pixels at
the frame corners.

Why accept it:

- The lighting plate is integrated over solid angle. A few pixels of edge
  distortion changes nothing about how much green a patch of grass emits.
- The display plate for a DNG source comes out of the same undistorted develop,
  so the plate and the CG agree *with each other*, which is what a composite
  needs. The error is against reality, not between layers.
- Applying the opcodes properly means implementing the DNG spec's warp model.
  That is a real piece of work to fix an error that is invisible on the terrain
  this pipeline is aimed at.

What to do if it bites — a long straight edge near a frame border is the case:
the compositor's Lens Distortion node, or solve `k1` once. It is a fixed lens,
so one coefficient solved once is good forever.

`server/raw.py:warp_opcodes()` detects and reports the opcodes so the situation
is stated rather than discovered. `test_raw.py` pins the reporting.

---

## 2. 35mm-equivalent focal length: long edge, by default

**Decision: `f_px = f35 / 36 * long_edge`, with a `diagonal` alternative exposed.**

The two readings of "35mm equivalent":

| convention | formula | on a 4:3 frame |
|---|---|---|
| long edge | `f35 / 36 mm × long edge px` | the default |
| diagonal | `f35 / 43.27 mm × diagonal px` | ~4% longer |

Four percent of focal length on a 14 mm ultra-wide is several degrees of field
of view — enough to watch the backplate drift away from the CG toward the frame
edges. Apple does not document which it uses.

Long edge is the default because it matches Blender's `sensor_fit='AUTO'` with
`sensor_width=36` exactly, so the intrinsics and the Blender camera cannot
disagree.

**If the M2 backplate check shows a consistent scale mismatch, switch the
convention. Do not nudge `focal_override`** — that hides a systematic error
behind a per-photo fudge, and the next photo will need a different fudge.

---

## 3. View transform is Standard, not AgX

**Decision: default `Standard`, with AgX available.**

Blender 4.x+ defaults to AgX. For a composite that is wrong: the plate is a
display-referred sRGB PNG, the compositor decodes it to linear, and AgX then
re-grades it on the way out. The backplate stops matching the photograph it
came from.

`Standard` makes sRGB → linear → sRGB an identity, so the plate passes through
untouched and only the CG needs to be matched to it.

---

## 4. Roll is positive when world-up leans right in the image

The sign of roll is a free choice; both conventions exist. Pinned so that the
reference photo `IMG_7096` reports **+0.7°** rather than −0.7°, matching the
figure quoted in `LIGHTING_ADDENDUM.md`.

Formally: `roll = atan2(-g_x, g_y)` for gravity in camera coordinates.
Positive means the world's up direction leans to the right in the displayed
frame — the horizon runs downhill to the right.

`pitch` needs no such choice: up is positive, and everyone agrees.

---

## 5. The shared core lives in the add-on, and the server imports it

`addon/photo3d/coords.py` and `imaging.py` import numpy and stdlib only. The
server reaches them through `server/_core.py`, which puts `addon/` on the path.

The alternative — a third top-level package both sides depend on — is cleaner
on paper and worse in practice: the add-on must be self-contained when zipped
into Blender, so it would have to vendor a copy at build time, and a vendored
copy drifts. When it drifts, the symptom is a camera that solves differently in
Blender than on the server, which reads as an accelerometer bug rather than as
the stale file it is.

`addon/photo3d/__init__.py` guards its bpy imports so the package is importable
outside Blender. `test_addon.py` enforces both halves of this: that the shared
modules never import bpy, and that no add-on module imports anything that would
have to be installed into Blender's bundled Python.

---

## 6. Solar position is implemented here, not taken from astral

`server/solar.py` is the NOAA algorithm in ~60 lines of stdlib.

- It removes a dependency from the part of the pipeline that must work offline
  forever.
- It is testable against published astronomy rather than against itself.

`astral` is still used, but only as an independent cross-check in the test
suite, and those tests skip cleanly when it is absent. Agreement is within 0.5°
across four continents and four seasons — far tighter than the ±5–15° compass
heading it gets combined with, so it is never the limiting error.

---

## 7. Blender 5.x compatibility shims

The add-on targets 4.2 LTS and is **tested on 5.2 LTS**, which is what is
installed here. Three things moved, all found by running
`tools/blender_smoke_test.py`:

| 4.2 | 5.x | handled in |
|---|---|---|
| `Scene.node_tree` | `Scene.compositing_node_group` (a node-group datablock) | `solve._compositor_tree` |
| `CompositorNodeComposite` | removed; use `NodeGroupOutput` + an interface socket | `solve._compositor_tree` |
| Alpha Over `inputs[1]`/`[2]` | `inputs["Background"]`/`["Foreground"]` | `solve._alpha_over_sockets` |
| `sky_type='NISHITA'` | `'MULTIPLE_SCATTERING'` | `solve._set_physical_sky` |

Each shim picks by feature detection, not by version number, so a 4.2 install
still works and a 6.0 rename fails loudly in the smoke test rather than
silently building a broken scene.

---

## 8. Transport: paths in, `.npy` out

Unchanged from the handoff, restated because it is load-bearing:

- **Images cross as filesystem paths, never base64.** A 48 MP plate is ~25 MB;
  base64 adds a third on top plus a JSON parse at both ends.
- **Depth crosses as `.npy` float32, never PNG.** PNG quantisation destroys
  metric precision, and with it the entire collision story.

`test_server.py` asserts the response stays under 8 KB, so a future convenience
that inlines an image fails immediately.
