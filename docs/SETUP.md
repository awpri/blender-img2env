# SETUP — everything to install, in order

macOS / Apple Silicon. Roughly 45 minutes, most of it downloads.

---

## 1. Base tools

```bash
# Homebrew, if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install exiftool python@3.11 git
brew install --cask blender
```

| | why |
|---|---|
| **Blender 4.2 LTS+** | the whole target. Cask pulls the Apple Silicon build. |
| **exiftool** | the only thing that reads Apple MakerNote gravity/GPS. Non-negotiable. |
| **python@3.11** | for the solver daemon. Never touch Blender's bundled Python. |
| **git** | cloning model repos |

Optional but recommended: `brew install --cask visual-studio-code`

---

## 2. Solver environment

```bash
python3.11 -m venv ~/.venvs/photo3d
source ~/.venvs/photo3d/bin/activate
pip install --upgrade pip

pip install fastapi uvicorn numpy pillow pillow-heif astral rawpy opencv-python
pip install torch torchvision                      # MPS support is built in on macOS
pip install diffusers transformers accelerate      # only needed for Marigold
pip install git+https://github.com/apple/ml-depth-pro.git
pip install git+https://github.com/compphoto/Intrinsic.git
```

Confirm the GPU is live before anything else:

```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

Then fetch Depth Pro's weights (~1.9 GB):

```bash
git clone https://github.com/apple/ml-depth-pro.git ~/src/ml-depth-pro
cd ~/src/ml-depth-pro && source get_pretrained_models.sh
```

---

## 3. Blender add-ons

**MCprep** — https://github.com/Moo-Ack-Productions/MCprep/releases
Download the `.zip`, then in Blender: `Edit ▸ Preferences ▸ Add-ons ▸ Install…`

This is the one you'd be silly to skip. Rigged mobs, correct nearest-neighbour
material setup, emission on glowstone and mob eyes, world import. It removes
most of what you'd otherwise be doing by hand.

Same install path for the two files already written:
- `photo3d_blender_addon.py`
- `photo3d_radiance_addon.py`

Both appear under the **Photo3D** tab in the N-panel.

---

## 4. Asset tools (as needed, not now)

| tool | for |
|---|---|
| **Blockbench** (free, blockbench.net) | custom blocks, items, nametag geometry |
| **Mineways** or **jmc2obj** | exporting chunks of an actual Minecraft world |
| **ComfyUI** (you have it) | panorama outpainting, texture work — see below |

---

## 5. First run

```bash
source ~/.venvs/photo3d/bin/activate
cd /path/to/photo3d
uvicorn solver_server:app --port 8765
```

Leave it running. In Blender, N-panel ▸ Photo3D ▸ pick a photo ▸ **Solve Photo**.

First solve ~15 s (weight load), every one after ~2 s.

---

## 6. Before you trust any of it

1. **Re-export a photo WITH location data.** Photos ▸ File ▸ Export ▸ Export
   Unmodified Original. Check:
   `exiftool -GPSLatitude -GPSLongitude IMG_XXXX.dng`
   Your current JPEG has the Ref fields and no coordinates, which means no sun.
2. **Level-floor test.** Photograph a level floor, phone upright. Solve. The
   Blender grid should sit flat and the horizon should land where the real one
   does. If it's tipped 90°, fix the axis mapping in `gravity_from_exif()` once
   and it's fixed forever.
3. **Shadow direction test.** Any shot with a hard shadow. Render a test cube,
   compare shadow angles. Constant error goes into `sky_rotation_offset`.

Two calibration shots buy correctness on every photo afterward.
