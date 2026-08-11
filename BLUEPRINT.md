# Photo3D — architecture blueprint

Single 48MP photo → solved Blender scene with matched camera, occluding/colliding
proxy geometry, and physically-correct lighting. Target: M4 Pro Mac, fully offline.

---

## The one idea that changes everything

**Your iPhone already recorded the camera's orientation.** Apple's MakerNote
writes an `AccelerationVector` — the gravity vector in device coordinates at the
moment of capture. That is pitch and roll, *measured by hardware*, not inferred.

This is why fSpy was the wrong tool for your Swiss photos and why you were right
to be suspicious of it. fSpy exists to recover orientation from vanishing lines.
On jagged alpine terrain there are none. But you never needed to recover it —
it's already in the file.

Check yours right now:

```bash
exiftool -Apple:all -GPS:all -FocalLengthIn35mmFormat IMG_XXXX.HEIC
```

You're looking for `Acceleration Vector`, `GPS Latitude/Longitude`,
`GPS Img Direction`, and `Date/Time Original`. If those are present, the entire
camera solve is deterministic:

| Unknown | Source | Accuracy |
|---|---|---|
| Focal length | `FocalLengthIn35mmFormat` | exact |
| Pitch + roll | gravity vector | ~1° (accelerometer noise) |
| Yaw | `GPSImgDirection` (compass) | ~5–15° (magnetic interference) |
| Height above ground | metric depth + gravity-constrained plane fit | ~10% |
| Sun azimuth + elevation | GPS + timestamp → solar ephemeris | arcminutes |

Nothing on that list is an AI guess. Yaw and sun position are only meaningful
*together* — get the compass heading right and the sun lands in the correct part
of your sky automatically, which means CG shadows fall in the same direction as
the real ones in the plate. That's the thing that sells a composite, and Stager
cannot do it at all.

Fallback path if `AccelerationVector` is missing on your files: horizon-line
detection (roll from tilt, pitch from the horizon's offset from the principal
point) works well on landscape photos and is a far better fit than vanishing
points for terrain. Worth writing only if the EXIF check comes back empty.

---

## Component choices

### Depth: Depth Pro as the backbone, Marigold as optional garnish

Marigold — which you already installed — produces *affine-invariant* depth. It is
beautiful and detailed and has **no absolute scale and no absolute offset**. You
cannot get a camera height or a metre-accurate collision surface from it alone.
A Minecraft block dropped onto a Marigold mesh lands at an arbitrary size.

Apple's **Depth Pro** outputs metric depth (real metres), has unusually sharp
boundaries, and is native to Apple Silicon. That's the backbone.

The `refine` flag fits Marigold's relative depth to Depth Pro's metric scale by
least squares *in disparity space* (where the relationship is genuinely affine),
giving you Marigold's micro-detail at Depth Pro's scale. Leave it off until you
find a shot where silhouette detail actually matters — it roughly triples solve
time for a mostly invisible mesh.

### Skip ComfyUI for this

ComfyUI is a graph editor for iterating on image generation. Here you want one
deterministic function call, and the graph adds a serialization layer, a node
API, and a browser you don't need. `solver_server.py` is a small FastAPI daemon
that keeps the model resident in unified memory — first photo ~15 s (weight
load), every subsequent photo ~1–3 s. Keep ComfyUI around for texture work and
outpainting; it's the wrong shape for this loop.

### Why an out-of-process daemon, not scripts inside Blender

Blender ships its own Python interpreter. Installing torch into it is possible
and is a recurring source of broken installs, and it means the model reloads
every time you restart Blender. The daemon runs in a normal venv, Blender talks
to it over localhost HTTP, and **images are passed by filesystem path, never
base64** — a 48MP HEIC is ~25 MB and base64 would add 33% plus a JSON parse on
both ends for no reason. Depth comes back as a `.npy` path. That's your
lowest-latency handshake.

### Your RTX 3070 Ti is an option

Since Jarvis is already running: the daemon is device-agnostic. Set
`DEVICE = torch.device("cuda")`, run it on the Windows box, and point `ENDPOINT`
at it over LAN. Depth Pro fits in 8 GB. Probably not worth it — MPS on an M4 Pro
is fast enough here and keeps the loop offline and local — but the door is open.

---

## The proxy geometry problem, stated honestly

Monocular depth gives you a **bas-relief**, not a scene. Every surface facing
away from the camera is missing. Practical consequences:

- **Occlusion: works well.** The visible surface is exactly what needs to occlude.
- **Ground collision: works well.** You have a metrically-scaled floor.
- **Shadows onto the plate: works well** where geometry is roughly right.
- **Objects passing *behind* things: works, with caveats.** A creeper walking
  behind a boulder disappears correctly, then reappears — but there's no back of
  the boulder for it to be lit by or bounce off.
- **CG object casting a shadow onto a tree it's standing beside: unreliable.**
  The tree is a flat-backed shell.

The single most important implementation detail is **discontinuity culling**:
delete any quad whose four corners straddle a large disparity jump. Without it
you get a rubber sheet stretching from every foreground silhouette back to the
mountains, and shadows crawl up that sheet. `edge_threshold` controls it —
0.08 is a reasonable start; lower it if you see stretched skirts, raise it if
silhouettes develop holes.

### Where a custom tool is genuinely worth building

**Segmentation-driven occluder splitting.** Depth alone gives soft, wobbly
silhouettes at exactly the edges the eye scrutinizes. The upgrade: run SAM 2 on
the plate, take the masks for the few objects that actually matter (the rock in
front, the fence post, the person), and build each as its own closed occluder
card at its median depth with a clean alpha edge. Then use the depth mesh only
for ground and midground.

This doesn't exist as an add-on. It's maybe 200 lines on top of what's here, and
it's the highest-leverage thing you could add after the base pipeline works.

**Guided depth upsampling.** Depth runs at ~1.5K, your plate is 8K. A joint
bilateral / guided filter of the depth against the full-res RGB snaps depth edges
back onto real image edges. `cv2.ximgproc.guidedFilter`, about 10 lines. Do this
before segmentation work; it may make the segmentation work unnecessary.

---

## The proxy shader (your requirement #4)

Invisible to the eye, fully present to every other ray:

```
Texture Coordinate ──[Window]──► Image Texture (plate, Cubic)
                                        │
                                        ▼ Color
                                 Principled BSDF ──┐
                                  Roughness 0.55   ├──► Mix Shader ──► Output
                                 Transparent BSDF ─┘        ▲
                                                            │
                     Light Path ──[Is Camera Ray]───────────┘
```

`Is Camera Ray` drives Fac. Camera rays → Transparent (you see the plate).
Glossy, diffuse and transmission rays → the photo-textured surface.

**Window coordinates are the key.** They're screen-space, so the plate reprojects
onto the proxy from exactly the render camera's viewpoint. That is what makes a
polished-metal Minecraft block reflect *the actual gravel it is standing on*,
with the real colour and the real texture, rather than a grey approximation.

Combined with `is_shadow_catcher = True` on the object and
`film_transparent = True` on the scene, you get shadows, contact darkening,
occlusion, and correct reflections in one setup, over a full-resolution plate
composited via Alpha Over.

Cycles only. EEVEE Next handles `Is Camera Ray` differently and doesn't do
screen-space reflections off geometry that isn't rendered.

---

## Minecraft specifics

**Crisp textures.** In the Image Texture node set **Interpolation: Closest**.
Cycles does no mipmapping in that mode, so a 16×16 block texture stays 16 hard
pixels at any scale — which is precisely the thing Stager wouldn't let you do.
The `Crispify Textures` button in the panel sets it across everything selected.

Minification aliasing is the trade-off: distant blocks will shimmer. Fix with
sample count (512+) rather than by reintroducing filtering.

**Assets.** Install **MCprep** (free Blender add-on). It ships rigged mobs, does
material setup with Closest interpolation and correct emission for glowstone /
lava / eyes, and imports worlds. Blockbench for custom models, Mineways or
jmc2obj for chunks of real worlds. MCprep alone probably covers 80% of what you
described.

**Nametags and XP orbs.** These aren't geometry problems, they're billboard
problems: plane + emission shader + `TRACK_TO` constraint on the camera, and for
nametags a semi-transparent dark quad behind text with Closest-interpolated
bitmap font. XP orbs get a small sine driver on Z and a random phase offset per
instance via geometry nodes. Straightforward once the scene is solved — write
these as a second pass, not part of the solver.

---

## Setup

```bash
# solver
python3.11 -m venv ~/.venvs/photo3d && source ~/.venvs/photo3d/bin/activate
pip install fastapi uvicorn torch torchvision pillow pillow-heif numpy astral diffusers
pip install git+https://github.com/apple/ml-depth-pro.git
brew install exiftool
uvicorn solver_server:app --port 8765
```

Blender: `Edit ▸ Preferences ▸ Add-ons ▸ Install…` →
`photo3d_blender_addon.py`. Panel appears under **N ▸ Photo3D**.

The add-on imports numpy, which Blender bundles. It imports nothing else
external — that's deliberate.

---

## Calibration, once, before you trust anything

1. **Gravity axes.** Photograph a level floor holding the phone vertically.
   Solve. The 3D grid floor should be level and the horizon should sit where the
   real one does. If it's tipped 90°, the axis mapping in `gravity_from_exif()`
   needs a swap for your iOS version — fix it there and it's fixed forever.
2. **Sky rotation.** Shoot something with an obvious hard shadow. Solve, render
   a test cube, compare shadow directions. Any constant error goes into
   `sky_rotation_offset` and stays there.
3. **Height.** Solve a photo where you know your eye height. Compare against the
   reported `camera_height_m`. Consistent bias means your `Orientation` branch
   is picking the wrong case.

Two calibration shots buy you correctness on every Switzerland photo afterward.

---

## Suggested order of work

1. Run the exiftool check. Everything above depends on that answer.
2. Get solver + add-on running; verify the grid lands on the ground.
3. Drop a default cube with rigid body, confirm it falls and lands on terrain.
4. Install MCprep, bring in one block, check reflections pick up the ground.
5. Calibrate sun direction against a real shadow.
6. *Then* build nametags, XP orbs, mob animation.
7. Guided depth upsampling if edges bother you; SAM 2 occluders if they still do.
