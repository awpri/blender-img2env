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

Run everything from the repository root — the paths below are relative to it.

```bash
brew install exiftool          # the only thing that reads Apple's MakerNote
make venv                      # ~/.venvs/photo3d + torch + Depth Pro
make weights                   # the 1.9 GB checkpoint
make addon                     # photo3d.zip for Blender
```

`make venv` picks Python 3.12 or 3.11 if either is present, because that is
what the ML stack is tested against. Override with `make venv PYTHON=python3.14`.

Optional, only needed for the M7 sun gobo:

```bash
~/.venvs/photo3d/bin/pip install git+https://github.com/compphoto/Intrinsic.git
```

### Daily use

Leave the daemon running in its own terminal — the models stay resident in that
process, and a cold reload per photo is exactly what it exists to avoid.

```bash
make serve                     # foreground, on 127.0.0.1:8765
make health                    # in another terminal: what is loaded?
```

Then in Blender: **press N** in the 3D viewport, choose the **Photo3D** tab,
set **Photo** to your file, and press **Solve Photo**. First solve ~15 s while
the weights load; every one after ~1–3 s.

**Solve Camera Only** skips depth entirely and needs no models or weights at
all — the fastest way to check the camera before committing to a full solve.

The add-on installs via **Edit ▸ Preferences ▸ Add-ons ▸ Install from Disk**,
pointed at the `photo3d.zip` that `make addon` writes.

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
| **M3** depth → proxy | runs end to end on the real DNG (45 s cold, MPS). Ground-plane recovery within 2% on analytic depth. **But Depth Pro's metric scale is ~5x out on these scenes** — calibrate before trusting it, see below. |
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

### Calibrate the depth scale before trusting geometry

Using the people in both photographs as the known object:

| photo | true distance | Depth Pro says | implied human height |
|---|---|---|---|
| `IMG_7263.DNG` | 11.1 m | 2.20 m | 0.35 m |
| `IMG_7096.HEIC` | 13.5 m | 2.69 m | 0.34 m |

An adult is 1.7 m, so both under-read by **~5x** at working distance. The error
is progressive — about 1.5x at 1 m, 5x at 12 m, 400x at 4 km — so no single
number fixes the whole frame, but one calibrated at the distance you work at
recovers the near and mid field, which is all that collides or casts shadows.

```bash
python tools/calibrate_scale.py testphotos/IMG_7263.DNG \
    --top 4660 --bottom 5155 --x 3264 --height 1.75
```

Put the printed factor in the panel's **Depth scale**. It belongs to the lens
and the kind of scene, not the individual photo. `docs/VERIFICATION.md` shows
how this was confirmed rather than assumed.

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
