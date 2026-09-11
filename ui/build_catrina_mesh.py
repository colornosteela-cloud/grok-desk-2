#!/usr/bin/env python3
"""Convert Las_Catrina.obj into a compact articulated mesh for the MiniOS twin."""
from __future__ import annotations

import gzip
import math
import struct
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CANDIDATES = [
    Path("/home/roni/teela/Las_Catrina.obj"),
    _HERE.parent / "teela" / "Las_Catrina.obj",
    Path.home() / "teela" / "Las_Catrina.obj",
]
SRC = next((p for p in _CANDIDATES if p.is_file()), _CANDIDATES[0])
DST = _HERE / "las-catrina.bin.gz"

# OBJ groups → bone names. CAD is supine along +X, +Y left, +Z anterior.
GROUP_BONE = {
    0: "pelvis",
    1: "head",
    2: "l_thigh",
    3: "l_foot",
    4: "r_foot",
    5: "neck",
    6: "l_uarm",
    7: "r_uarm",
    8: "r_farm",
    9: "l_farm",
    10: "l_hand",
    11: "r_hand",
    12: "r_thigh",
    13: "r_shin",
    14: "l_shin",
    15: "thorax",
}
PARENT = {
    "pelvis": "",
    "torso_lower": "pelvis",
    "torso_upper": "torso_lower",
    "neck": "torso_upper",
    "head": "neck",
    "l_uarm": "torso_upper",
    "l_farm": "l_uarm",
    "l_hand": "l_farm",
    "r_uarm": "torso_upper",
    "r_farm": "r_uarm",
    "r_hand": "r_farm",
    "l_thigh": "pelvis",
    "l_shin": "l_thigh",
    "l_foot": "l_shin",
    "r_thigh": "pelvis",
    "r_shin": "r_thigh",
    "r_foot": "r_shin",
}
ORDER = [
    "pelvis", "torso_lower", "torso_upper", "neck", "head",
    "l_uarm", "l_farm", "l_hand",
    "r_uarm", "r_farm", "r_hand",
    "l_thigh", "l_shin", "l_foot",
    "r_thigh", "r_shin", "r_foot",
]


def xform(x: float, y: float, z: float, cy: float, xmax: float, zmin: float, scale: float) -> tuple[float, float, float]:
    # Standing Y-up, +X right, +Z forward.
    wx = (cy - y) * scale
    wy = (xmax - x) * scale
    wz = (z - zmin) * scale
    return wx, wy, wz


def main() -> None:
    print("reading", SRC)
    verts: list[tuple[float, float, float]] = []
    groups: list[list[tuple[int, int, int]]] = []
    cur: list[tuple[int, int, int]] = []
    gidx = -1
    mins = [1e9, 1e9, 1e9]
    maxs = [-1e9, -1e9, -1e9]
    with SRC.open() as fh:
        for line in fh:
            if line.startswith("v "):
                a = line.split()
                x, y, z = float(a[1]), float(a[2]), float(a[3])
                verts.append((x, y, z))
                if x < mins[0]:
                    mins[0] = x
                if y < mins[1]:
                    mins[1] = y
                if z < mins[2]:
                    mins[2] = z
                if x > maxs[0]:
                    maxs[0] = x
                if y > maxs[1]:
                    maxs[1] = y
                if z > maxs[2]:
                    maxs[2] = z
            elif line.startswith("g "):
                if gidx >= 0:
                    groups.append(cur)
                gidx += 1
                cur = []
            elif line.startswith("f "):
                ids = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                if len(ids) >= 3:
                    cur.append((ids[0], ids[1], ids[2]))
                    for i in range(2, len(ids) - 1):
                        cur.append((ids[0], ids[i], ids[i + 1]))
        if gidx >= 0:
            groups.append(cur)
    cy = 0.5 * (mins[1] + maxs[1])
    height = maxs[0] - mins[0]
    scale = 1.68 / height
    print("groups", len(groups), "verts", len(verts), "scale", round(scale, 5))

    world: list[tuple[float, float, float]] = [xform(x, y, z, cy, maxs[0], mins[2], scale) for x, y, z in verts]
    parts: dict[str, dict] = {}
    for gi, faces in enumerate(groups):
        name = GROUP_BONE[gi]
        used: dict[int, int] = {}
        loc: list[tuple[float, float, float]] = []
        idx: list[int] = []
        for a, b, c in faces:
            for src in (a, b, c):
                if src not in used:
                    used[src] = len(loc)
                    loc.append(world[src])
                idx.append(used[src])
        sx = sy = sz = 0.0
        for px, py, pz in loc:
            sx += px
            sy += py
            sz += pz
        n = max(1, len(loc))
        centroid = (sx / n, sy / n, sz / n)
        parts[name] = {"verts": loc, "idx": idx, "centroid": centroid, "used": used}
        print(f"  {name:8} faces={len(faces):7d} verts={len(loc):6d} c=({centroid[0]:.3f},{centroid[1]:.3f},{centroid[2]:.3f})")

    def pack_faces(face_list: list[tuple[int, int, int]]) -> dict:
        used: dict[int, int] = {}
        loc: list[tuple[float, float, float]] = []
        out: list[int] = []
        for a, b, c in face_list:
            for src in (a, b, c):
                if src not in used:
                    used[src] = len(loc)
                    loc.append(world[src])
                out.append(used[src])
        sx = sy = sz = 0.0
        for px, py, pz in loc:
            sx += px
            sy += py
            sz += pz
        n = max(1, len(loc))
        return {"verts": loc, "idx": out, "centroid": (sx / n, sy / n, sz / n), "used": {}}

    # Match teela_v2: 72,700 lowest thoracic faces = torso_lower, remainder = torso_upper.
    thorax_faces = groups[15]
    scored = []
    for a, b, c in thorax_faces:
        cy = (world[a][1] + world[b][1] + world[c][1]) / 3.0
        scored.append((cy, (a, b, c)))
    scored.sort(key=lambda t: t[0])
    n_lower = 72700
    parts["torso_lower"] = pack_faces([f for _, f in scored[:n_lower]])
    parts["torso_upper"] = pack_faces([f for _, f in scored[n_lower:]])
    parts.pop("thorax", None)
    print(
        f"  torso_lower verts={len(parts['torso_lower']['verts'])}  "
        f"torso_upper verts={len(parts['torso_upper']['verts'])}"
    )

    def sample(verts: list[tuple[float, float, float]], count: int) -> list[tuple[float, float, float]]:
        if len(verts) <= count:
            return verts
        step = len(verts) / count
        return [verts[int(i * step)] for i in range(count)]

    def contact_pivot(
        parent_verts: list[tuple[float, float, float]],
        child_verts: list[tuple[float, float, float]],
    ) -> tuple[float, float, float]:
        """Midpoint of the closest parent/child vertex pairs — the actual joint."""
        ps = sample(parent_verts, 900)
        cs = sample(child_verts, 900)
        pairs: list[tuple[float, tuple[float, float, float], tuple[float, float, float]]] = []
        for c in cs:
            md = 1e9
            mp = ps[0]
            cx, cy, cz = c
            for p in ps:
                d = (cx - p[0]) ** 2 + (cy - p[1]) ** 2 + (cz - p[2]) ** 2
                if d < md:
                    md = d
                    mp = p
            pairs.append((md, c, mp))
        pairs.sort(key=lambda t: t[0])
        k = max(16, len(pairs) // 25)
        chunk = pairs[:k]
        sx = sy = sz = 0.0
        for _, c, p in chunk:
            sx += 0.5 * (c[0] + p[0])
            sy += 0.5 * (c[1] + p[1])
            sz += 0.5 * (c[2] + p[2])
        n = max(1, len(chunk))
        return (sx / n, sy / n, sz / n)

    pivots: dict[str, tuple[float, float, float]] = {}
    for name in ORDER:
        parent = PARENT[name]
        if not parent:
            pivots[name] = parts[name]["centroid"]
            continue
        pivots[name] = contact_pivot(parts[parent]["verts"], parts[name]["verts"])

    # Keep feet on the ground in rest pose.
    foot_y = min(v[1] for name in ("l_foot", "r_foot") for v in parts[name]["verts"])
    for name in ORDER:
        px, py, pz = pivots[name]
        pivots[name] = (px, py - foot_y, pz)
        parts[name]["verts"] = [(x, y - foot_y, z) for x, y, z in parts[name]["verts"]]

    # Smooth normals.
    for name in ORDER:
        vs = parts[name]["verts"]
        idx = parts[name]["idx"]
        nrm = [[0.0, 0.0, 0.0] for _ in vs]
        for i in range(0, len(idx), 3):
            a, b, c = idx[i], idx[i + 1], idx[i + 2]
            ax, ay, az = vs[a]
            bx, by, bz = vs[b]
            cx, cy, cz = vs[c]
            ux, uy, uz = bx - ax, by - ay, bz - az
            vx, vy, vz = cx - ax, cy - ay, cz - az
            nx = uy * vz - uz * vy
            ny = uz * vx - ux * vz
            nz = ux * vy - uy * vx
            for vi in (a, b, c):
                nrm[vi][0] += nx
                nrm[vi][1] += ny
                nrm[vi][2] += nz
        out_n = []
        for nx, ny, nz in nrm:
            L = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
            # Flip so outward normals face +Z (anatomical front).
            out_n.append((-nx / L, -ny / L, -nz / L))
        parts[name]["nrm"] = out_n

    buf = bytearray()
    buf += b"TEELAMES"
    buf += struct.pack("<I", 1)
    buf += struct.pack("<I", len(ORDER))
    for name in ORDER:
        parent = PARENT[name]
        pidx = ORDER.index(parent) if parent else -1
        px, py, pz = pivots[name]
        vs = parts[name]["verts"]
        ns = parts[name]["nrm"]
        idx = parts[name]["idx"]
        name_b = name.encode("ascii")
        buf += struct.pack("<B", len(name_b)) + name_b
        buf += struct.pack("<h", pidx)
        buf += struct.pack("<3f", px, py, pz)
        buf += struct.pack("<I", len(vs))
        buf += struct.pack("<I", len(idx))
        for x, y, z in vs:
            buf += struct.pack("<3f", x, y, z)
        for x, y, z in ns:
            buf += struct.pack("<3f", x, y, z)
        buf += struct.pack(f"<{len(idx)}I", *idx)
        print(f"packed {name} pivot=({px:.3f},{py:.3f},{pz:.3f}) parent={parent or '-'}")

    raw = bytes(buf)
    print("uncompressed", round(len(raw) / 1e6, 2), "MB")
    DST.write_bytes(gzip.compress(raw, compresslevel=6))
    print("wrote", DST, round(DST.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
