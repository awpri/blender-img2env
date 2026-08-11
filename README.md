# Photo3D

One iPhone ProRAW photo in; a solved Blender scene out — matched camera, proxy
geometry that occludes and collides, and lighting derived from the image
itself. Fully offline, aimed at an M4 Pro Mac. Minecraft assets get composited
into the result.

**The premise:** your iPhone already recorded the camera's orientation. Apple's
MakerNote carries an `AccelerationVector` — the gravity vector in device
coordinates at the moment of capture. That is pitch and roll *measured by
hardware*, not inferred from vanishing lines. Combined with the compass heading
and a solar ephemeris, the entire camera solve is deterministic:

| unknown | source | accuracy |
|---|---|---|
| focal length | `FocalLengthIn35mmFormat` | exact |
| pitch + roll | gravity vector | ~1° (accelerometer noise) |
| yaw | `GPSImgDirection` | ~5–15° (magnetic interference) |
| height above ground | metric depth + gravity-constrained plane fit | ~10% |
| sun azimuth + elevation | GPS + timestamp → solar ephemeris | arcminutes |

Nothing on that list is an AI guess.

---

## Layout

```
addon/photo3d/         Blender add-on. bpy, mathutils, numpy, stdlib. Nothing else.
├── coords.py          all coordinate maths — bpy-free, heavily tested
├── imaging.py         blur, masks, bounce calibration — bpy-free, tested
├── client.py          urllib transport to the daemon
├── props.py           one PropertyGroup
├── solve.py           camera, sun, sky, compositor, the solve operator
├── proxy.py           depth → mesh, the proxy shader, the collider
├── radiance.py        bounce proxy, sun gobo, panorama
└── ui.py              one panel, with sections

server/                the solve daemon. Runs in its own venv, models resident.
├── exif.py            EXIF → intrinsics, gravity, GPS, sun
├── solar.py           NOAA solar position, stdlib only
├── depth.py           Depth Pro (+ optional Marigold fit), ground plane
├── raw.py             DNG → linear EXR lighting plate
├── shading.py         intrinsic decomposition → shadow mask
├── solver_server.py   FastAPI, one endpoint that matters
└── tests/             450+ tests, no Blender and no torch required

docs/                  BLUEPRINT, LIGHTING_ADDENDUM, HANDOFF, SETUP,
                       DECISIONS (the ambiguous calls), VERIFICATION (milestones)
legacy/                the original unexecuted scaffold, and what was wrong with it
tools/                 Blender smoke test, add-on zip builder
```

### Two processes, on purpose

Blender ships its own Python. Installing torch into it is possible and is a
recurring source of broken installs, and it means reloading 1.9 GB of weights
every time Blender restarts. So the daemon runs in a normal venv and Blender
talks to it over localhost.

Across that boundary: **images are filesystem paths, never base64** (a 48 MP
plate is ~25 MB), and **depth is `.npy` float32, never PNG** (quantisation
destroys metric precision and with it the whole collision story). The add-on
imports nothing that is not already in Blender — `test_addon.py` enforces that
mechanically, and the zip builder refuses to package a violation.

---

## Setup

```bash
make venv          # solver venv at ~/.venvs/photo3d with everything installed
make test          # 450+ tests, ~1 second
make addon         # photo3d.zip → Blender ▸ Preferences ▸ Add-ons ▸ Install from Disk
make serve         # daemon on 127.0.0.1:8765
```

Two git-only dependencies the venv target cannot install for you:

```bash
~/.venvs/photo3d/bin/pip install git+https://github.com/apple/ml-depth-pro.git
~/.venvs/photo3d/bin/pip install git+https://github.com/compphoto/Intrinsic.git
```

Then fetch Depth Pro's weights with its `get_pretrained_models.sh` (~1.9 GB),
and `brew install exiftool` — it is the only thing that reads Apple's MakerNote.

Everything must be run from the repository root; the paths in these commands
and in `docs/VERIFICATION.md` are relative to it.

In Blender: **N-panel ▸ Photo3D**, pick a photo, **Solve Photo**. First solve
~15 s while weights load; every one after ~1–3 s.

`Solve Camera Only` skips depth entirely and needs no models at all — useful
for checking the camera before committing to a full solve.

---

## State of the milestones

The scaffold this was built from had never been executed. Everything below has
been rebuilt and tested; M1, M2 and M6 have now been run against the real
photographs, and the add-on against a real Blender. M3–M5 and M7 still need
Depth Pro weights and a render. `docs/VERIFICATION.md` says exactly what
remains manual for each milestone.

| | status |
|---|---|
| **M1** EXIF solve | **passes on the real `IMG_7096.HEIC`**: pitch 3.279°, roll 0.715°, heading 76.944°, 14 mm, zero warnings. Four-orientation set agrees within 0.25°. |
| **M2** camera into Blender | rotation maths pinned by 200+ synthetic cases and re-checked against the real solve; camera verified inside Blender. Needs the backplate eye-check. |
| **M3** depth → proxy | ground-plane recovery within 2% on analytic depth; mesh, winding and rigid body verified in Blender. The cube-resting-height acceptance needs Depth Pro weights and a GPU. |
| **M4** shadow proxy + compositor | node graph asserted, including Window coords and the Alpha Over order. Chrome-sphere check needs a render. |
| **M5** bounce proxy | ray-visibility split asserted both ways; strength calibration tested; A/B toggle built for the double-counting check. Now driven by the linear EXR when there is one. Needs the white-sphere render. |
| **M6** DNG lighting plate | **works on the real `IMG_7263.DNG`** — linear EXR, max 2.890 against a median of 0.210, 0.52% of pixels above diffuse white. Required a decoder change; see below. |
| **M7** shadow extraction + gobo | intrinsic decomposition with a tested fallback; the bake is now exception-safe, verified by simulating a mid-render failure. |

### Two things the real files changed

**LibRaw cannot read this camera's ProRAW.** `IMG_7263.DNG` is DNG 1.7 with
JPEG XL compression, and rawpy/LibRaw 0.22 rejects it outright — so the
blueprint's `rawpy → EXR` path fails at the first call. macOS **Core Image** is
tried first now: `CIRAWFilter` with `boostAmount = 0` gives linear data with the
headroom intact, offline, using Apple's own decoder. rawpy stays as the
portable fallback. Install `pyobjc-framework-Quartz` for it.

**`WarpRectilinear` was never a problem here.** The DNG is
`PhotometricInterpretation: Linear Raw` with no opcode tags at all — Apple
rectified it before writing the file. `docs/DECISIONS.md` §1 has the evidence;
the detection code stays because other cameras may differ.

## Verified against Blender 5.2 LTS

Targets 4.2 LTS, tested on 5.2, which found three real API breaks:
`Scene.node_tree` → `Scene.compositing_node_group`, `CompositorNodeComposite`
removed, and `sky_type='NISHITA'` → `'MULTIPLE_SCATTERING'`. All three are
handled by feature detection rather than version checks, so both work.

```bash
make smoke     # 17 checks inside a real Blender, including that a failed
               # gobo bake leaves the scene exactly as it found it
```

---

## What this does not do

- No GUI beyond the N-panel.
- No job queue, database or web frontend.
- No ComfyUI wrapper — reasoning in `docs/BLUEPRINT.md`.
- No Minecraft asset import. Install **MCprep**; it already does it, better.
