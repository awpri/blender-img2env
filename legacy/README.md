# legacy — the original scaffold

These three files are the design as it was handed over. **They were never
executed.** They are kept because they are the specification the current code
was built from, and because the comments in them explain intent that is worth
having on record.

| file | superseded by |
|---|---|
| `solver_server.py` | `server/` — split into exif, solar, depth, raw, shading, solver_server |
| `photo3d_blender_addon.py` | `addon/photo3d/` — split into coords, client, props, solve, proxy, ui |
| `photo3d_radiance_addon.py` | `addon/photo3d/radiance.py`, merged into the same panel |

## What was wrong with them

Found by writing tests and by running the add-on in a real Blender. None of
this is a criticism of the scaffold — it is what "has never been executed"
means in practice.

**The device-to-camera gravity map was wrong.** `gravity_from_exif()` applied
`(g[0], -g[1], -g[2])` and then rotated for orientation. The rotations do not
compose to the physical axis relationship, so orientations 1, 3 and 8 solved to
different camera poses for the same physical scene. The corrected table is in
`addon/photo3d/coords.py`, derived from the device axes rather than guessed,
and `test_coords.py` pins all four against hand-derived level cases.

**The yaw correction was unstable.** `up_cam.rotation_difference(Vector((0,0,1)))`
picks the minimal rotation, which is degenerate when the camera points near the
nadir, and the subsequent `atan2(fwd.x, fwd.y)` correction assumed the residual
was a pure yaw. Replaced with orthonormal triad matching, which has no gimbal
case and no ordering convention to get wrong.

**`solve_camera_height` could return a negative height,** putting the Blender
camera underground on ceiling-dominant frames. Now rejected explicitly.

**Blender 5.x API drift**, all three found by running the smoke test:
- `Scene.node_tree` became `Scene.compositing_node_group`
- `CompositorNodeComposite` was removed in favour of a Group Output
- `sky_type = 'NISHITA'` became `'MULTIPLE_SCATTERING'`

**The gobo bake left the scene broken on failure.** Render settings and the
proxy's materials were mutated, then restored only on the success path. Now a
`try/finally` context manager, which is the specific gotcha the handoff flagged.

**`extract_shadow_mask` used `np.apply_along_axis(np.convolve)`,** which is
roughly 200x slower than a summed-area blur at the radii in use — slow enough
to stall Blender's UI thread mid-bake.

**A DNG's `IFD0` describes the embedded thumbnail,** so an unqualified
`ImageWidth` lookup could return 256 and scale the entire solve. The tag lookup
is now group-aware.

**`Crispify` set every texture to Closest,** including the photographic plate,
which wants smoothing.
