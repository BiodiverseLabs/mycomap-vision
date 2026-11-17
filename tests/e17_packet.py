"""A synthetic E17 review packet (e17review.py): invented names, fake accessions, generated
photos with no EXIF. No real E17 data exists in tests, and none is ever looked for."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw


def hit(no, name, identity, coverage, accession=None):
    h = {"hit_no": no, "name": name, "identity": identity, "coverage": coverage}
    if accession is not None:
        h["accession"] = accession
    return h


def seq(n, local=(), genbank=(), withdrawn=False, length=612):
    return {"n": n, "withdrawn": withdrawn, "length": length,
            "local": [hit(i + 1, *h) for i, h in enumerate(local)],
            "genbank": [hit(i + 1, *h) for i, h in enumerate(genbank)]}


def item(code, name, place="Ontario", when="September 2025", photos=((800, 600),),
         sequences=()):
    return {"schema": "e17-item/1", "code": code, "name_under_review": name, "place": place,
            "when": when,
            "photos": [{"n": i + 1, "width": w, "height": h} for i, (w, h) in enumerate(photos)],
            "sequences": list(sequences)}


ITEMS = [
    item("K7Q2M", "Fakeus exemplaris", photos=((960, 720), (720, 960), (960, 720)),
         sequences=[
             seq(1, local=[("Fakeus sp. 'IN07'", 99.7, 100), ("Fakeus sp. 'IN07'", 99.5, 100),
                           ("Fakeus exemplaris", 97.9, 99), ("Fakeus exemplaris", 97.6, 99),
                           ("Fakeus sp. 'NY11'", 96.8, 98), ("unnamed local sequence", 95.2, 97)],
                 genbank=[("Fakeus exemplaris", 98.0, 99, "ZZ000001.1"),
                          ("Fakeus amoenus", 96.9, 98, "ZZ000002.1"),
                          ("Fakeus sp.", 96.4, 97, "ZZ000003.1"),
                          ("Otherella quieta", 86.1, 91, "NR_000004.1")]),
             seq(2, withdrawn=True, length=598,
                 local=[("Fakeus sp. 'IN07'", 98.8, 96)],
                 genbank=[("Fakeus amoenus", 96.3, 95, "ZZ000005.1"),
                          ("Fakeus exemplaris", 96.1, 94, "ZZ000006.1")]),
         ]),
    item("00042", "Mockia fictiva", place="Quebec", when="not recorded",
         photos=((800, 600), (800, 600)),
         sequences=[seq(1, length=None,
                        local=[("Mockia fictiva", 99.9, 100), ("Mockia inventa", 97.2, 99)],
                        genbank=[("Mockia fictiva", 99.6, 100, "ZZ000011.1")])]),
    item("B3XQ9", "Inventia placeholderi", place="Ohio", when="October 2024",
         photos=((640, 480),),
         sequences=[seq(1, local=[("unnamed local sequence", 99.1, 100),
                                  ("Inventia sp. 'ABCD123-45'", 98.0, 99)],
                        genbank=[("Phantomyces vagus", 91.0, 88, "ZZ000021.1")])]),
    item("PLM4T", "Fakeus sp. 'IN01'", place="Michigan", when="August 2023",
         photos=((800, 600), (600, 800), (800, 600), (800, 600)),
         sequences=[seq(1, local=[("Fakeus sp. 'IN01'", 100, 100)], genbank=[]),
                    seq(2, local=[], genbank=[("Fakeus exemplaris", 95.5, 90, "ZZ000031.1")])]),
    item("ZQ8R1", "Testaria nullius", place="Indiana", when="June 2025", photos=(),
         sequences=[]),
    item("W2N6D", "Mockia inventa", place="New York", when="July 2025",
         photos=((800, 600),),
         sequences=[seq(1, local=[("Mockia fictiva", 98.4, 100), ("Mockia inventa", 98.4, 99)],
                        genbank=[("Mockia inventa", 99.0, 100, "ZZ000041.1"),
                                 ("Mockia", 97.0, 100, "ZZ000042.1")])]),
]
# The builder's shuffle: display order is not the code order.
POSITIONS = {"K7Q2M": 3, "00042": 1, "B3XQ9": 5, "PLM4T": 2, "ZQ8R1": 6, "W2N6D": 4}

COLOURS = [(138, 106, 74), (196, 154, 108), (90, 125, 58), (61, 90, 58), (233, 226, 208),
           (185, 167, 137)]


def jpeg(width, height, seed=0, **save) -> bytes:
    """A generated photo with no EXIF: a soft gradient and a few shapes."""
    a, b = COLOURS[seed % len(COLOURS)], COLOURS[(seed + 2) % len(COLOURS)]
    img = Image.new("RGB", (width, height), a)
    draw = ImageDraw.Draw(img)
    for y in range(0, height, 4):
        t = y / max(1, height - 1)
        draw.rectangle([0, y, width, y + 4], fill=tuple(int(a[i] * (1 - t) + b[i] * t) for i in range(3)))
    cx, cy, r = width // 2, int(height * 0.55), min(width, height) // 5
    draw.ellipse([cx - r, cy - r // 2, cx + r, cy + r // 3], fill=(120, 80, 50))
    draw.rectangle([cx - r // 6, cy, cx + r // 6, cy + r], fill=(230, 220, 200))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80, **save)
    return buf.getvalue()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(folder: Path, items=None, positions=None, photo_bytes=None, manifest_edit=None) -> str:
    """Write a packet to `folder`; returns the pinned sha (sha256 of manifest.json).
    `photo_bytes(code, n, width, height)` may replace a photo's bytes; `manifest_edit` may
    change the manifest before it is written."""
    if positions is None:
        positions = POSITIONS if items is None else {}
    items = copy.deepcopy(ITEMS if items is None else items)
    folder = Path(folder)
    (folder / "items").mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "e17-packet/1", "count": len(items), "items": [], "photos": []}
    for k, it in enumerate(items):
        code = it["code"] if isinstance(it, dict) and isinstance(it.get("code"), str) else f"X{k:04d}"
        data = json.dumps(it, indent=1).encode("utf-8")
        (folder / "items" / f"{code}.json").write_bytes(data)
        manifest["items"].append({"code": code, "position": positions.get(code, k + 1),
                                  "sha256": sha(data)})
        for ph in (it.get("photos") or []) if isinstance(it, dict) else []:
            n, w, h = ph["n"], ph["width"], ph["height"]
            data = photo_bytes(code, n, w, h) if photo_bytes else None
            data = data if data is not None else jpeg(w, h, seed=k + n)
            p = folder / "photos" / code / f"{n}.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            manifest["photos"].append({"path": f"photos/{code}/{n}.jpg", "sha256": sha(data)})
    manifest["items"].sort(key=lambda m: m["position"])
    if manifest_edit:
        manifest_edit(manifest)
    data = json.dumps(manifest, indent=1).encode("utf-8")
    (folder / "manifest.json").write_bytes(data)
    return sha(data)
