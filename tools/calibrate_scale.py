"""Solve the depth scale factor from an object of known size.

Depth Pro's metric output saturates on wide-angle scenes with a large depth
range. Measured on both reference photographs:

    IMG_7263.DNG  person 495 px tall, true ~1.75 m, model says 2.24 m away
                  -> implied height 0.35 m, a 5.0x under-read
    IMG_7096.HEIC hiker  395 px tall, true ~1.70 m, model says 2.69 m away
                  -> implied height 0.34 m, a 5.0x under-read

The far field is worse and unfixable — 4 km peaks read under 10 m — but the
near field is only about 1.5x out, so one multiplier calibrated at the distance
you actually work at recovers usable geometry where CG objects sit, collide and
cast shadows. That is the whole of what M3 needs.

Usage — measure a known object's top and bottom row in the FULL-RESOLUTION
plate (Blender's image editor shows pixel coordinates, so does Preview):

    python tools/calibrate_scale.py testphotos/IMG_7263.DNG \\
        --top 4660 --bottom 5155 --x 3264 --height 1.75

Then put the printed number in the add-on's "Depth scale" field, or pass
depth_scale to /solve. It is a property of the lens and the kind of scene, not
of the individual photograph, so calibrate once and reuse.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import depth as depth_mod          # noqa: E402
from server import exif as exif_mod            # noqa: E402
from server import raw as raw_mod              # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="the photograph, as handed to /solve")
    parser.add_argument("--top", type=int, required=True,
                        help="row of the object's top edge, full-res pixels")
    parser.add_argument("--bottom", type=int, required=True,
                        help="row of the object's bottom edge, full-res pixels")
    parser.add_argument("--x", type=int, required=True,
                        help="column through the object, full-res pixels")
    parser.add_argument("--height", type=float, default=1.75,
                        help="the object's true height in metres (default: an adult)")
    parser.add_argument("--depth-long-edge", type=int, default=1536)
    args = parser.parse_args()

    state = exif_mod.camera_state_from_path(args.image)
    intrinsics = state.intrinsics

    image = raw_mod.load_display_image(args.image) if not raw_mod.is_raw(args.image) \
        else raw_mod.load_display_image(
            raw_mod.make_plates(args.image, Path.home() / ".cache" / "photo3d",
                                want_lighting_plate=False).display_png)

    from PIL import Image
    small = image.resize((round(image.width * args.depth_long_edge / max(image.size)),
                          round(image.height * args.depth_long_edge / max(image.size))),
                         Image.LANCZOS)
    f_px_small = intrinsics.fx * (small.width / float(intrinsics.width))

    print(f"running Depth Pro at {small.width}x{small.height} (f_px {f_px_small:.1f}) ...")
    depth = depth_mod.run_depth_pro(small, f_px_small)

    # Sample in the depth raster, at the object's mid-height.
    scale = depth.shape[1] / float(intrinsics.width)
    col = int(args.x * scale)
    row = int((args.top + args.bottom) / 2 * scale)
    patch = depth[max(0, row - 5):row + 5, max(0, col - 4):col + 4]
    if patch.size == 0:
        print("those coordinates fall outside the image", file=sys.stderr)
        return 1
    z = float(np.median(patch))

    pixel_height_full = abs(args.bottom - args.top)
    implied = depth_mod.implied_size(pixel_height_full, z, intrinsics.fx)
    factor = depth_mod.scale_for_known_object(pixel_height_full, args.height,
                                              z, intrinsics.fx)

    print()
    print(f"  object          {pixel_height_full} px tall at ({args.x}, "
          f"{(args.top + args.bottom) // 2})")
    print(f"  reported depth  {z:.2f} m")
    print(f"  implied height  {implied:.2f} m   (you said it is {args.height:.2f} m)")
    print(f"  true distance   {args.height * intrinsics.fx / pixel_height_full:.1f} m")
    print()
    print(f"  depth_scale = {factor:.2f}")
    print()
    if factor > 1.3:
        print(f"  Depth Pro is under-reading by {factor:.1f}x at this distance. That is")
        print("  expected on a wide frame with a big depth range; it is why this")
        print("  tool exists. The far field stays wrong whatever you do — keep the")
        print("  proxy's far clip tight.")
    elif factor < 0.77:
        print(f"  Depth Pro is over-reading by {1 / factor:.1f}x here.")
    else:
        print("  Depth is already close to correct at this distance; leave it at 1.0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
