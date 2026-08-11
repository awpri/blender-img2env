# Decisions

Things that were genuinely ambiguous, what was chosen, and what would change
the choice. The handoff asked for several of these explicitly.

---

## 1. `WarpRectilinear` turned out to be a non-issue on this camera

**Original concern:** a ProRAW DNG defers Apple's lens rectification to a
`WarpRectilinear` opcode that LibRaw does not apply, so moving to DNG for the
highlight headroom might reintroduce barrel distortion the JPEG had removed.

**Finding, from the real file:** `IMG_7263.DNG` (iPhone 17 Pro, ultra-wide)
carries **no opcode tags at all**. Its main image is
`PhotometricInterpretation: Linear Raw` — already demosaiced and already
geometrically corrected. Apple did the rectification before writing the file,
which is exactly why there is nothing left to defer.

```
$ exiftool -a -u -G1 '-Opcode*' IMG_7263.DNG
(nothing)
$ exiftool -G1 -PhotometricInterpretation IMG_7263.DNG
[SubIFD]  Photometric Interpretation : Linear Raw
```

So there is no distortion to accept and no correction to apply. **Decision:
unchanged in effect — do not implement opcode application** — but the reasoning
is now "there are none" rather than "we tolerate them".

`server/raw.py:warp_opcodes()` still detects and reports them, because other
cameras and future firmware may well write them, and a silent geometry change
is the worst way to find that out.
`test_real_photos.py::test_this_dng_carries_no_warp_opcodes` pins the finding
so the claim above is not taken on trust.

If a long straight edge near a frame border ever does bow: the compositor's
Lens Distortion node, or solve `k1` once. It is a fixed lens, so one
coefficient is good forever.

---

## 1b. LibRaw cannot read this camera's ProRAW; Core Image can

This is the real M6 problem, and it is not the one the blueprint anticipated.

```
rawpy 0.27.0 / LibRaw 0.22.1 on IMG_7263.DNG
  -> LibRawFileUnsupportedError: 'Unsupported file format or not RAW file'
```

The file is **DNG 1.7 with JPEG XL compression**. LibRaw 0.22 does not support
it, so the entire `rawpy → linear EXR` path in the blueprint fails at the first
call on the target camera.

**Decision: try Apple's own decoder first on macOS, keep rawpy as the portable
fallback.** `CIRAWFilter` with `boostAmount = 0` disables the tone curve and
returns linear extended-range data — it is the same decoder Photos uses, it is
offline, and it needs no extra download beyond `pyobjc-framework-Quartz`.

Measured on `IMG_7263.DNG`:

| | value |
|---|---|
| max | 2.890 |
| median | 0.210 |
| pixels above 1.0 | 0.52% |
| pixels above 2.0 | 0.063% |

That is real headroom, and it is the whole justification for the DNG path: a
display-referred plate clips all of it, and the clipped regions carry most of
the scene's light energy.

Ordering is Core Image → rawpy on macOS, because on the target camera the
portable decoder is precisely the one that does not work. If neither can read
the file, the error names all three routes out, including the free Adobe DNG
Converter.

The lighting plate is developed at a 2048 px long edge by default. Lighting is
low-frequency — the first diffuse bounce blurs it anyway — and a full 48 MP
float32 RGBA buffer is ~780 MB of unified memory for no visible gain.

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

## 3b. Marigold does not rescue the hard scenes; the ground plane does

Tested, because "absolute scale does not matter, I will rescale by hand" is
only true if the *shape* is right. A uniform scale error can be undone in
Blender. A shape error cannot.

The test: fit an affine model — exactly the ambiguity Marigold has — using two
independent constraints, the visible ground being a plane perpendicular to
measured gravity, and a person of known pixel height being 1.75 m. If the two
agree, the geometry is self-consistent and rescaling works. Consistency of 1.00
means they agree.

| | alpine, 14 mm | station, 24 mm |
|---|---|---|
| Depth Pro, affine-fitted in disparity | **2.62** | 0.79 |
| Marigold, affine in depth | **2.40** | 0.89 |
| Marigold, affine in disparity | **2.42** | degenerate |

Marigold lands within a few percent of Depth Pro on the scene that matters.
The alpine failure is not the affine ambiguity — it is that the *relative*
geometry is wrong, and no two-parameter fit repairs that. Marigold also cannot
be used alone for placement, because affine-invariance means an unknown offset
as well as an unknown scale, and a wrong offset warps the scene: flat ground
becomes curved, and that warp does not come out with a scale slider.

**Decision: keep Depth Pro as the backbone; do not promote Marigold. For scenes
where metric depth is unreliable, place objects on a gravity-derived ground
plane instead of on the depth mesh.**

The plane is the good answer because the camera rotation was built so that
world +Z is truly up — measured by the accelerometer, not inferred. So `z = 0` is
genuinely level, and the only uncertain quantity is how far below the lens it
sits: one number, which the user can drag, and which sets the scene's scale.

The depth mesh keeps doing what it is reliably good at even when distorted —
occlusion and shadow catching, where relative ordering is what matters — while
the plane does the standing-on. That splits the problem along the line where
the accuracy actually falls.

`Proxy Geometry ▸ Add Ground Plane`. The smoke test asserts its normal is
exactly +Z whatever the camera was doing.

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
