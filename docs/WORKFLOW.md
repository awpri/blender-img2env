# How to actually use this

The order matters, and several steps only make sense on certain photographs.
This is the sequence, what each step is for, and how to tell it worked.

---

## 0. Start the daemon

```bash
make serve          # leave running in its own terminal
```

In Blender: **N** in the 3D viewport → **Photo3D** tab.

---

## 1. Solve

Set **Photo**, press **Solve Photo**. ~20 s warm, ~45 s cold.

**Check it worked:** the *Measured* box shows pitch, roll, heading, height and a
ground confidence. Press **Look Through Camera** — the horizon in the plate
should sit level, and the floor grid should lie on the real ground.

If the grid is tipped, the solve is wrong and nothing downstream will save it.
If the backplate lines up in the middle but drifts at the edges, that is the
focal convention — switch it under *Camera & Lens*, don't nudge the focal.

**Solve Camera Only** skips depth entirely. Use it to check the camera in a
second before committing to a full solve.

---

## 2. Decide what your objects stand on

Two options, and the right one depends on the photograph.

**The depth mesh** (`Photo3D_Proxy`, built automatically). Real terrain shape,
so objects sit on bumps and behind rocks correctly. Trust it when the scene is
human-scale and the lens is normal — the station photo measures within 5% at
20 m.

**The ground plane** (*Proxy Geometry ▸ Add Ground Plane*). Dead level, because
its orientation comes from the accelerometer rather than from any model. Only
its height is a guess, and you can drag it in Z. Use it when metric depth is
unreliable — ultra-wide frames, huge depth range, cluttered scenes — or
whenever you just want a dependable floor.

You can have both: the mesh occludes and catches shadows, the plane is what
things land on. Occlusion only needs relative ordering, which survives even
when the metres are wrong.

**Check it worked:** *Add Drop Test Cube*, press play. It should land, not fall
through and not hover.

---

## 3. Bring in your model

MCprep for Minecraft assets. Then **Minecraft ▸ Crispify Textures** with your
objects selected — it forces nearest-neighbour so a 16×16 texture stays 16 hard
pixels at any scale. It deliberately leaves the photographic plate on Cubic.

Distant blocks will shimmer. Fix that with sample count (512+), not by
reintroducing filtering.

---

## 4. Shadows — these work with no extra steps

Your object casts a shadow onto the proxy or the ground plane automatically:
the proxy is a Cycles shadow catcher, the film is transparent, and the
compositor lays the render over the plate with Alpha Over.

You need a light for a shadow to exist. The solve adds a sun automatically when
the photo has GPS and a timestamp. No GPS, no sun — add one by hand and aim it
to match the shadows already in the plate.

**Check it worked:** F12. The shadow should appear on the photograph. If the
object renders but casts nothing, check there is a `Photo3D_Sun` in the scene
and that its direction matches the real shadows.

`tools/render_checks.py` verifies this by actually rendering.

---

## 5. Bounce light — do this before anything else lighting-wise

**What it is.** Your photograph, turned into a light source. The depth mesh is
duplicated, made emissive, and textured with the plate through Window
coordinates — so grass one metre below a floating block emits green *upward*,
with correct falloff and correct solid angle, because the emitter is the actual
grass at its actual measured distance. A red wall to camera-left throws warm
light on the left face of your block. None of that is estimated.

**This is the single biggest visual jump, and it is the whole answer for
interiors** — indoors there is no sun and no sky worth modelling, only walls, a
floor, a window and a lamp, all of which are in the photograph.

**How to use it:**

1. *Bounce Light ▸ Estimate* — solves the emission strength from the plate
   instead of leaving you to guess. It assumes a ground albedo; set that first
   (grass 0.18, asphalt 0.10, gravel 0.25, snow 0.75).
2. *Build Bounce Proxy*.

That is it. It reports which plate it used — the linear EXR if your photo was
ProRAW, otherwise the display plate with a note that the highlights are clipped.

**Check it worked:** put a white matte sphere in the scene and press
*Toggle Bounce*, rendering each way. With bounce on, the sphere should pick up
colour from the ground — green over grass, warm over stone. It should **not**
be roughly twice as bright: that would mean both proxies are emitting and the
light is being counted twice. The toggle is symmetric on purpose, handing
indirect duties back to the shadow proxy when the bounce is off, so the two
renders are genuinely comparable.

---

## 5b. Match the exposure — do this or everything looks blown out

The sun and sky default to 4.0 and 1.0, which are arbitrary numbers. Arbitrary
numbers put CG several stops brighter than the photograph it is standing in,
and the result reads as "the compositing doesn't work" rather than "the key
light is too strong". Nothing else in the pipeline tells you, because every
other check is geometric.

*Sun & Sky ▸ Match Exposure to Plate* solves it: the plate says what radiance
the real ground has, the assumed albedo says what irradiance produced it, and
both strengths are scaled by that one factor — so whatever sun-to-sky ratio you
set is preserved.

Set **Ground albedo** first, under Bounce Light. It is the same number both
calibrations use.

---

## 6. Sun and sky

Set automatically from the solar ephemeris. Two knobs worth knowing:

- **Sun strength / Sky strength** — relative brightness of key versus ambient.
- **Sky rotation offset** — Nishita's zero-rotation reference is not true north.
  Calibrate once: shoot something with a hard shadow, render a test cube,
  compare shadow directions, put the difference here, then never touch it again.

---

## 7. Sun gobo — only for dappled shade

**Skip this unless your photo has broken light on the ground**, light coming
through foliage or a lattice. On open ground it does nothing useful.

**What it is.** Light through larch branches cannot be reconstructed — the
canopy geometry is not recoverable from one photo and never will be. But the
shadow pattern it casts is already printed on your ground in the plate. So the
pattern is stolen rather than simulated: extracted as a mask, projected onto
the ground, re-rendered from an orthographic camera aligned with the solved sun
direction (the only space where a cucoloris means anything), and hung in front
of the sun as a shadow-only card. This is a real film-lighting technique.

**How to use it:**

1. *Sun Gobo ▸ Extract Shade Mask* — asks the daemon for the mask. It reports
   which method it used and what fraction of the frame it thinks is in shade.
2. If that fraction is near zero, stop: either the shot has no dapple, or
   **Mask blur is too small**. The extraction only sees shadows *smaller* than
   its own blur kernel — in the middle of a shadow wider than the blur, the
   mask reads as fully lit. Raise Mask blur until the fraction looks right.
3. *Bake Sun Gobo*. Needs a `Photo3D_Sun`, so it needs a photo with GPS.
4. Lower **Mask contrast** if the shade reads too hard.

**Check it worked:** an object set down in the dappled area should receive
broken light, and its own cast shadow should break up in the same pattern.

**Caveat worth knowing:** the fallback extraction conflates dark paint with
shade — a dark rock reads as shadow. It is clamped so the worst case is a
slightly-too-dark patch rather than a hole punched in the sunlight. Installing
`compphoto/Intrinsic` gets you proper reflectance/shading separation, which
does not make that mistake.

---

## 7b. Glass, water and metal

The proxy arrives as one matte surface, which is right for ground and rock and
wrong for anything transmissive. A glass shelter modelled as matte grey will
not transmit a CG object standing behind it.

*Surface Materials ▸ Edit Proxy Faces*, select the faces covering the surface
(hover and press **L** to grab a connected patch, or box-select), then click
**Glass**, **Water**, **Metal**, **Polished** or **Matte**.

That splits the selection into its own object with a real material. Glass and
water are made camera-visible and stop being shadow catchers, because a matte
cannot refract — so CG behind them is genuinely seen through them. Metal and
matte keep catching shadows and take their colour from the plate.

Doing it by hand is not a placeholder: glass is defined by what is behind it,
which monocular depth cannot see, and you know which surface is glass. When
segmentation lands it will feed this same operator rather than replace it.

---

## 8. Panorama — last, and optional

Fills in sky and behind-camera for distant reflections and rim light. The
bounce proxy is more accurate within ~10 m, so let it own the near field and
use the panorama only for distance.

---

## Order of operations, condensed

1. Solve, and look through the camera.
2. Ground plane or depth mesh; drop-test cube.
3. Your model; crispify.
4. Render — shadows already work.
5. **Bounce proxy** ← the biggest single improvement, works everywhere, indoors too.
6. Sun and sky; calibrate the sky rotation offset once.
7. Gobo, only if the shot has dappled shade.
8. Panorama, only if you need distant reflections.

Steps 5 and 6 are the ones that matter. 7 and 8 are polish.


---

## Selecting an irregular region

Box-dragging rectangles is the wrong tool for a window that is not a rectangle
on screen. Blender has three select tools and the toolbar is the unambiguous
way in: the icons down the top-left of the viewport, click-and-hold the first
one to swap between **Select Box**, **Select Circle** and **Select Lasso**.
**W** cycles them.

For painting over an awkward shape, **Select Circle** is usually fastest: press
**C**, drag over the faces like a brush, scroll to resize the brush, middle-drag
to erase, right-click or Esc to finish.

Also worth knowing in face mode:

| | |
|---|---|
| **C** then drag | circle select, brush-style |
| **Ctrl + right-drag** | freehand lasso |
| **B** | box select |
| **Ctrl + numpad +** | grow the selection by one ring |
| **Shift + click** | add or remove one face |

**L** (select linked) is not much use on the proxy: the depth mesh is one
connected grid, so it takes nearly everything.

---

## Turning a window into a light

A sunlit window or a bright canopy panel is **clipped to white in the plate**.
Its real brightness is the one thing the photograph could not record, so the
bounce proxy cannot emit it correctly no matter how it is calibrated — it emits
what the file says, and the file says "white".

So mark it and give it a real wattage. Select the faces covering the window,
then *Surface Materials ▸ Make Light From Selection*. It creates an area light
at that surface, the size of the region, oriented along its normal, tinted with
the colour sampled from the plate — the warm yellow of light through a canopy
panel comes out as roughly (1.00, 0.88, 0.74).

The power is a starting point, not a measurement, and it is deliberately yours
to set: a clipped highlight carries no magnitude. Raise it until CG objects
sitting in that patch of light look like they belong in it.
