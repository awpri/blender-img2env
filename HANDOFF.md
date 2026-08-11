# HANDOFF — brief for a coding agent

Paste this whole file as the opening message, with `BLUEPRINT.md`,
`LIGHTING_ADDENDUM.md`, `solver_server.py`, `photo3d_blender_addon.py` and
`photo3d_radiance_addon.py` in the working directory.

---

## Context

Building an offline photo-to-3D compositing pipeline on an M4 Pro Mac. A single
iPhone ProRAW photo goes in; a solved Blender scene comes out, with a matched
camera, proxy geometry that occludes and collides, and lighting derived from the
image. Minecraft assets get composited into it.

There is existing scaffold code. **It has never been executed.** Treat it as a
detailed design document that happens to be valid Python, not as a working
program. Expect API drift against Blender 4.2+, expect sign errors in the
coordinate math, expect the exiftool field names to differ slightly.

The reference photo is `IMG_7096` — 14 mm equivalent (ultra-wide), portrait,
ProRAW. Sample EXIF is quoted in `LIGHTING_ADDENDUM.md`.

---

## Repo layout to build toward

```
photo3d/
├── server/
│   ├── solver_server.py      # FastAPI daemon, models resident
│   ├── exif.py               # EXIF → intrinsics, gravity, GPS, sun
│   ├── depth.py              # Depth Pro (+ optional Marigold fit)
│   ├── raw.py                # DNG → linear EXR lighting plate
│   ├── shading.py            # intrinsic decomposition → shadow mask
│   └── tests/
└── addon/
    └── photo3d/              # multi-file Blender add-on
        ├── __init__.py
        ├── solve.py
        ├── proxy.py
        ├── radiance.py
        └── ui.py
```

Split the two monolithic add-on files into that package. Keep one panel with
sections, not two tabs.

---

## Milestones, in order. Do not start one before the previous passes.

### M1 — EXIF solve, no ML at all
Parse EXIF, return intrinsics + gravity + heading + sun. Prove it against the
level-floor calibration shot: reported pitch and roll within 2° of zero.

**Acceptance:** `pytest` on a fixture photo returns pitch 3.3° ± 0.5, roll 0.7° ±
0.5, heading 76.9°, focal 14 mm.

Gotcha: iOS accelerometer sign conventions vary with `Orientation`. The existing
`gravity_from_exif()` handles orientations 1/3/6/8 and is **unverified**. Build a
small fixture set of the same scene shot in all four orientations and make them
agree.

### M2 — camera into Blender
Solve → camera object whose backplate lines up. Judged by eye: the Blender floor
grid should sit on the real ground.

Gotcha: the OpenCV↔Blender basis conversion and the yaw correction in
`camera_matrix_from_solve()` are the most likely place for a sign error.
Write a unit test on the rotation math with synthetic gravity vectors before
touching Blender.

### M3 — depth → proxy mesh
Depth Pro at 1536 px, unproject, discontinuity-cull, mesh into Blender, passive
rigid body. A default cube with an active rigid body must fall and land on the
terrain at plausible scale.

**Acceptance:** cube resting height within 15% of ground truth on a shot where
a known object gives scale.

Gotcha: 14 mm frames have huge depth range and metric depth degrades badly past
~100 m. Clamp and don't chase accuracy out there.

### M4 — shadow proxy shader + compositor
Shadow catcher, Window-projected plate, Alpha Over. A chrome sphere must reflect
the actual ground beneath it.

### M5 — bounce proxy
Emissive twin, ray-visibility split per the table in `LIGHTING_ADDENDUM.md`.
Verify no double-counting by rendering a white sphere with the bounce proxy
enabled and disabled and comparing.

**This is the highest-value milestone. Prioritise it over M6 and M7.**

### M6 — DNG lighting plate
`rawpy` → linear 32-bit EXR, separate from the display plate. Check whether
LibRaw applies the DNG `WarpRectilinear` opcodes; if not, decide whether to
apply them or accept the distortion, and document which.

### M7 — automatic shadow extraction + gobo
Replace the luminance-ratio mask in `photo3d_radiance_addon.py` with intrinsic
image decomposition (`compphoto/Intrinsic`), which separates reflectance from
shading properly. The shading layer *is* the mask. Then bake the ortho gobo.

Gotcha: the ortho-camera bake temporarily mutates render settings and materials.
Make it exception-safe — a failed render currently leaves the scene broken.

---

## Hard constraints

- **Never install packages into Blender's bundled Python.** The add-on may import
  only `bpy`, `mathutils`, `numpy`, and stdlib.
- **Images cross the process boundary as filesystem paths**, never base64. A
  48 MP plate is ~25 MB.
- **Depth crosses as `.npy` float32**, never as PNG. PNG quantisation destroys
  metric precision and the whole collision story with it.
- Everything runs offline. No cloud APIs.
- Cycles only. EEVEE Next handles `Is Camera Ray` differently.
- Models stay resident in the daemon. A cold reload per photo is a bug.

## Explicitly out of scope

- Do not add a GUI beyond the Blender N-panel.
- Do not build a job queue, database, or web frontend.
- Do not wrap the depth models in ComfyUI (reasoning in `BLUEPRINT.md`).
- Do not implement Minecraft asset import — MCprep already does it.

## Style

Small, testable functions. The coordinate math gets unit tests with synthetic
inputs before it touches Blender. Comment the *why* on anything involving a
coordinate convention — that is where this will break.
