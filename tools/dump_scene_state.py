"""Print what a Photo3D scene actually contains.

Run inside the Blender that has the problem scene open:
    Scripting workspace -> Open -> tools/dump_scene_state.py -> Run Script
then copy the System Console / terminal output.

Reads only. Nothing here changes the scene.
"""
import bpy

p = bpy.context.scene.photo3d
sc = bpy.context.scene
print("\n================ PHOTO3D SCENE STATE ================")
import photo3d
print(f"add-on version   {photo3d.bl_info['version']}")
print(f"engine {sc.render.engine}  film_transparent {sc.render.film_transparent}  "
      f"compositing {sc.render.use_compositing}  res% {sc.render.resolution_percentage}")
vl = sc.view_layers[0]
print(f"shadow catcher pass: {getattr(vl.cycles, 'use_pass_shadow_catcher', '?')}")
print(f"sun_strength {p.sun_strength}  sky_strength {p.sky_strength}  "
      f"sun_share {getattr(p, 'sun_share', '?')}")
print(f"bounce_strength {p.bounce_strength}  saturation {p.bounce_saturation}")
print(f"proxy_blocks_light {getattr(p, 'proxy_blocks_light', '?')}  "
      f"mesh_long_edge {p.mesh_long_edge}")

print("\n--- mesh objects ---")
print(f"{'name':28s} {'cam':4s}{'diff':5s}{'glos':5s}{'shad':5s}{'CATCH':6s} faces  material")
for o in sorted(sc.objects, key=lambda o: o.name):
    if o.type != "MESH":
        continue
    mat = o.material_slots[0].material.name if o.material_slots and o.material_slots[0].material else "-"
    print(f"{o.name:28s} {int(o.visible_camera):<4}{int(o.visible_diffuse):<5}"
          f"{int(o.visible_glossy):<5}{int(o.visible_shadow):<5}"
          f"{int(o.is_shadow_catcher):<6}{len(o.data.polygons):<7}{mat}")

print("\n--- lights ---")
for o in sc.objects:
    if o.type == "LIGHT":
        print(f"{o.name:28s} {o.data.type:6s} energy {o.data.energy:.3f}")

print("\n--- sun / gobo ---")
sun = bpy.data.objects.get("Photo3D_Sun")
if sun and sun.data.use_nodes and sun.data.node_tree:
    for n in sun.data.node_tree.nodes:
        print(f"   sun node {n.type:18s} {n.name}")
else:
    print("   sun has no node tree (no gobo baked into the lamp)")
for name in ("Photo3D_Gobo", "Photo3D_Ground"):
    o = bpy.data.objects.get(name)
    print(f"   {name}: {'present' if o else 'absent'}"
          + (f" catcher={int(o.is_shadow_catcher)} cam={int(o.visible_camera)}" if o else ""))

print("\n--- light linking ---")
for o in bpy.data.objects:
    ll = getattr(o, "light_linking", None)
    if ll and (ll.receiver_collection or ll.blocker_collection):
        print(f"   {o.name}: receiver={ll.receiver_collection} blocker={ll.blocker_collection}")

print("\n--- world ---")
w = sc.world
print(f"world: {w.name if w else None}")
if w and w.use_nodes:
    for n in w.node_tree.nodes:
        print(f"   {n.type:18s} {n.name}")

print("\n--- compositor ---")
tree = getattr(sc, "compositing_node_group", None) or (sc.node_tree if sc.use_nodes else None)
if tree is None:
    print("   NO COMPOSITOR TREE")
else:
    for n in tree.nodes:
        extra = ""
        if n.type == "IMAGE" and n.image:
            extra = f" image={n.image.name} {tuple(n.image.size)}"
        if n.type in ("MIX_RGB", "MIX"):
            extra = f" blend={getattr(n, 'blend_type', '?')}"
        if n.type == "MATH":
            extra = f" op={n.operation}"
        print(f"   {n.type:18s} {n.name}{extra}")
print("================ END ================\n")
