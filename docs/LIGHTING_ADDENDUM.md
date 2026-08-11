# Addendum — image-derived lighting, DNG, and the 14mm problem

Supersedes the lighting section of `BLUEPRINT.md`. You were right: solar
ephemeris is necessary and nowhere near sufficient.

---

## What your EXIF actually says

```
Acceleration Vector : -0.01226  -0.98264  0.05630
GPS Img Direction   : 76.944  (True North)
GPS Date/Time Stamp : 2026:07:14  09:11:36 UTC
Focal 35mm          : 14 mm
Color Temperature   : 5337 K
Image Capture Type  : ProRAW
```

Decoded:

| | |
|---|---|
| Orientation | **portrait**, gravity dominant on device Y |
| Camera pitch | **+3.3° above horizontal** |
| Camera roll | **0.7°** |
| Heading | **76.9° true** — facing ENE |
| Local capture time | **11:11 CEST** |
| Vector magnitude | 0.984 g — slight motion, harmless |

The camera solve is fully determined. No vanishing points, no reference cube.

**Two problems in that block, though.**

**1. There is no `GPS Latitude` / `GPS Longitude`.** You have the *Ref* fields
(North, East) and nothing they refer to. The JPEG export stripped the
coordinates. Without them there is no solar ephemeris at all — the entire sun
half of the pipeline is dark. Fix at export: Photos ▸ File ▸ Export ▸ Export
Unmodified Original, or Export with **Location Information** ticked. Verify with
`exiftool -GPSLatitude -GPSLongitude`. The DNG originals will have it.

**2. 14 mm is the ultra-wide.** That has real consequences:

- Barrel distortion at the frame edges is significant. The pipeline assumes a
  pinhole camera, so straight real-world edges will drift a few pixels away from
  their CG counterparts near the corners — visible on a long building edge, not
  on a mountainside.
- The processed JPEG is already rectified by Apple's ISP. A ProRAW DNG carries
  `WarpRectilinear` opcodes that LibRaw/rawpy **do not apply by default**. If you
  switch to DNG for the lighting benefits below, you may reintroduce distortion
  the JPEG had removed. Check by rendering a wireframe over a shot containing a
  long straight edge near the frame border.
- If it bites: the compositor's Lens Distortion node, or solve `k1` once for the
  14 mm module and reuse it forever — it's a fixed lens.

At 14 mm you also have enormous depth range in frame, which is where monocular
depth is weakest. Expect the far mountains to be metrically wrong. Doesn't
matter for compositing — nothing collides with a peak 4 km away — but don't
trust the numbers out there.

---

## Yes, use DNG. Here is the actual reason.

Not resolution. **Highlight headroom.**

Your JPEG reports `HDR Headroom: 1.01`, meaning essentially none. Every pixel
brighter than diffuse white — the sky, a specular glint off a lake, a window in
an interior — is clipped to 1.0. Those clipped regions carry most of the light
energy in the scene. A lighting solution derived from a clipped image
systematically underestimates the sky and produces flat, grey-looking CG.

ProRAW DNG gives you 12–14 stops of linear scene radiance with the tone curve
not yet applied. That is the difference between guessing at the sky's brightness
and measuring it.

Server-side change:

```python
import rawpy
with rawpy.imread(path) as raw:
    rgb = raw.postprocess(
        gamma=(1, 1),              # stay linear
        no_auto_bright=True,       # no exposure stretch
        output_bps=16,
        output_color=rawpy.ColorSpace.sRGB,
        use_camera_wb=True,
    )
```

Save as 32-bit EXR for the lighting path; keep the SDR PNG for the visible
backplate. Two plates, two jobs — the beauty plate can stay Apple-graded while
the lighting plate stays linear.

`Color Temperature: 5337` is the ISP's illuminant estimate. Use it to set the
sun/sky colour so CG neutrals match plate neutrals without eyeballing.

---

## The three lighting mechanisms

### 1. Bounce proxy — the photograph as an emitter

The single most important idea, and the one that answers your objection
directly.

The depth proxy already carries the plate reprojected through Window
coordinates. Make a second copy of it **emissive** and you have converted the
photograph into geometry-accurate area lighting. Grass one metre below a
floating Minecraft block emits green upward with correct inverse-square falloff
and correct solid angle, because the emitter *is* the grass, at its real
measured distance. A red barn to camera-left throws warm light on the left face
of a block. None of that requires an estimator — it is measured radiance on
measured geometry.

Ray-visibility split, so nothing double-counts:

| | shadow proxy | bounce proxy |
|---|---|---|
| camera | matte (shadow catcher) | invisible |
| diffuse / glossy | **off** | **on**, emissive |
| shadow | on | off |
| collision | yes | no |

**This is also your entire interior answer.** Indoors there is no sun and no
sky worth modelling; there are walls, a floor, a window, a lamp — all of which
are in the photograph and all of which become emitters. Run the bounce proxy,
skip the ephemeris, done. Interiors are arguably the *easier* case.

Limits worth knowing: only what the camera saw emits, so light from behind the
camera is missing (the panorama step below covers that), and a display-referred
plate underestimates the window blowout (the DNG step above covers that).

### 2. Sun gobo — steal the dapple

Light through larch branches cannot be reconstructed from a single photo. The
tree's canopy geometry isn't recoverable and never will be.

But the shadow pattern it casts is *already in your plate*, printed on the
ground. So take it rather than simulate it:

1. Extract a shade mask: local luminance ÷ heavily blurred luminance. Albedo
   varies slowly across grass, illumination under foliage varies fast, so the
   ratio isolates shadow.
2. Project that mask onto the ground proxy through the same Window coords.
3. Render it from an **orthographic camera aligned with the solved sun
   direction**. That converts the pattern from camera space into sun space,
   which is the only space where a cucoloris means anything.
4. Hang the result on a plane in front of the sun as a Transparent BSDF whose
   Color is the mask, visible to shadow rays only.

A Minecraft block set down in dappled shade now receives the same broken light
the real ground does, and its cast shadow breaks up in the same pattern. This is
a real film-lighting technique (a cucoloris) and I know of nothing that
automates it from a plate — this is genuinely new code.

Caveat: the mask conflates dark albedo with shade. A dark rock reads as shadow.
Works well over uniform ground; degrades over mixed terrain. Mask output is
clamped at 0.15 so the worst case is a slightly-too-dark patch rather than a
black hole.

### 3. Panorama for sky and off-frame

At 14 mm you already capture ~104° horizontally, so the bounce proxy covers most
of the hemisphere that matters. What's missing is behind-camera and above-frame.

Options in ascending cost: Nishita sky driven by the solved sun (already built,
free, correct for clear-sky alpine); ComfyUI outpainting to equirectangular then
inverse tone-mapped (your ComfyUI install finally earns its keep — this is
exactly what it's good at); DiffusionLight, which inpaints a chrome ball and
reads the reflection as an HDR probe, best quality, slowest on MPS.

Use the panorama for distant reflections and rim light. Let the bounce proxy own
everything within ~10 m, because there it is simply more accurate.

---

## Order of operations

1. Re-export with GPS. Nothing solar works until you do.
2. Solve → shadow proxy → verify the grid sits on the ground.
3. **Bounce proxy + Estimate strength.** Biggest visual jump, works everywhere,
   independent of the sun. Do this before anything else lighting-wise.
4. Sun lamp + Nishita sky from ephemeris — direct light and hard shadow direction.
5. Sun gobo, only if the shot has dappled shade.
6. DNG lighting plate, when flat/grey CG starts bothering you.
7. Panorama, last and optional.

Steps 3 and 4 are the ones that matter. 5 through 7 are polish.
