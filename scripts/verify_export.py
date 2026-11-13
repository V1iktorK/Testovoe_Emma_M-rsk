# -*- coding: utf-8 -*-
"""
Приёмочная проверка ЭКСПОРТИРОВАННЫХ OBJ (независимо от генератора).

1) Текстовая проверка файла: есть ли v / vn / f / usemtl / s (группы сглаживания),
   все ли грани - квады, нет ли n-гонов, какой mtllib.
2) Обратный импорт в пустую сцену: габариты, топология, замкнутость, материалы.

    blender -b --factory-startup -P verify_export.py -- <папка проекта>
"""

import bpy
import bmesh
import os
import sys

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
root = argv[0] if argv else "."
OBJ_DIR = root


def text_scan(path):
    """Разбор OBJ как текста - то, что увидит импортёр 3ds Max."""
    n_v = n_vn = n_vt = n_f = 0
    quad = tri = other = 0
    mtllib = usemtl = s_lines = 0
    umin = vmin = 1e9
    umax = vmax = -1e9
    bands = {"shell(0.36-1)": 0, "coamings(0.30-0.35)": 0, "transom(0.19-0.29)": 0, "bulwark(0.155-0.185)": 0, "deck/bottom(0-0.145)": 0}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                n_v += 1
            elif line.startswith("vn "):
                n_vn += 1
            elif line.startswith("vt "):
                n_vt += 1
                parts = line.split()
                u, v = float(parts[1]), float(parts[2])
                umin, umax = min(umin, u), max(umax, u)
                vmin, vmax = min(vmin, v), max(vmax, v)
                if v >= 0.355:
                    bands["shell(0.36-1)"] += 1
                elif v >= 0.295:
                    bands["coamings(0.30-0.35)"] += 1
                elif v >= 0.185:
                    bands["transom(0.19-0.29)"] += 1
                elif v >= 0.15:
                    bands["bulwark(0.155-0.185)"] += 1
                else:
                    bands["deck/bottom(0-0.145)"] += 1
            elif line.startswith("f "):
                n_f += 1
                k = len(line.split()) - 1
                if k == 4:
                    quad += 1
                elif k == 3:
                    tri += 1
                else:
                    other += 1
            elif line.startswith("mtllib"):
                mtllib += 1
            elif line.startswith("usemtl"):
                usemtl += 1
            elif line.startswith("s "):
                s_lines += 1
    print("[TEXT] %s" % os.path.basename(path))
    print("  v=%d vn=%d vt=%d f=%d | quads=%d tris=%d other=%d" % (n_v, n_vn, n_vt, n_f, quad, tri, other))
    print("  mtllib=%d usemtl_blocks=%d smoothing_group_lines=%d" % (mtllib, usemtl, s_lines))
    if n_vt:
        print("  UV bounds: u=%.3f..%.3f  v=%.3f..%.3f" % (umin, umax, vmin, vmax))
        print("  UV по областям: " + ", ".join("%s=%d" % (k, c) for k, c in bands.items()))
    else:
        print("  UV отсутствуют")


def import_and_stats(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.obj_import(filepath=path, forward_axis="Y", up_axis="Z")
    obs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    print("[IMPORT] %s  objects=%d" % (os.path.basename(path), len(obs)))
    for ob in obs:
        me = ob.data
        bm = bmesh.new()
        bm.from_mesh(me)
        non_manifold = sum(1 for e in bm.edges if not e.is_manifold)
        vol = bm.calc_volume(signed=False)
        bm.free()
        xs = [v.co.x for v in me.vertices]
        ys = [v.co.y for v in me.vertices]
        zs = [v.co.z for v in me.vertices]
        nq = sum(1 for p in me.polygons if len(p.vertices) == 4)
        nt = sum(1 for p in me.polygons if len(p.vertices) == 3)
        print("  name=%s  verts=%d faces=%d edges=%d" % (ob.name, len(me.vertices), len(me.polygons), len(me.edges)))
        print("  quads=%d tris=%d (quad share %.1f%%)" % (nq, nt, 100.0 * nq / max(1, len(me.polygons))))
        print("  size X=%.4f  Y=%.4f  Z=%.4f" % (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)))
        print("  min X=%.4f Y=%.4f Z=%.4f | max X=%.4f Y=%.4f Z=%.4f"
              % (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs)))
        print("  non_manifold_edges=%d  volume=%.4f" % (non_manifold, vol))
        print("  materials=%s" % ", ".join(m.name if m else "None" for m in me.materials))


for name, expect in (("Emma_Maersk_hull_1-100_mm.obj", "LOA = 3977.1 (мм)"),
                     ("Emma_Maersk_hull_1-100_m.obj", "LOA = 3.9771 (м)")):
    p = os.path.join(OBJ_DIR, name)
    if not os.path.exists(p):
        print("MISSING %s" % p)
        continue
    print("=" * 70)
    print("FILE %s   ожидается %s" % (name, expect))
    text_scan(p)
    import_and_stats(p)

print("VERIFY_DONE")
