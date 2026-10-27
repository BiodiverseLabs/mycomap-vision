"""Recovered labels: a specimen crop for a field-slip photo a person excluded (Steve, 2026-10-26).

People keep excluding field-slip ("label") photos in Human Feedback as before. For each photo a
person excluded with the slip reason, a detector looks for a specimen on the WHOLE photo; where
it finds one it proposes a crop box (the specimen's bounding box with a 15% margin, square
where the photo allows, one box per well-separated cluster). The box goes to the "Recovered
labels" queue, where it is included unless a person rejects it.

The crop is kept as COORDINATES on the original photo, never as a new image file: some photos
carry no-derivatives licences, the public read-only Human Feedback viewer must never show a
derivative, and the original keeps its provenance. Embedding and training apply the box.

The rules are the research lane's (experiments/field-slip-attention/regions.py and
crop_test.py, research 7bda042), moved from the photo's centre square to the whole photo:

    OWLv2 (google/owlv2-base-patch16-ensemble, Apache-2.0) finds boxes for open-vocabulary
    prompts; SAM (facebook/sam-vit-base, Apache-2.0) turns each box into a mask.
    slip      paper label / handwritten note / index card / printed label / paper with
              writing, score >= 0.20
    specimen  mushroom / fungus / toadstool / bracket fungus / mushroom cap / small mushroom /
              crust fungus / jelly fungus / slime mold / mushroom stem, score >= 0.12; a box
              over 60% of the frame is the whole scene, dropped
    Where a specimen mask lies on a slip the pixel is specimen. A hole in the slip mask (inside
    its outline, neither slip nor specimen, >= 0.2% of the frame) is something lying on the
    slip, and counts as specimen.

The geometry below needs numpy only, so it is tested without the models. The models load on
the first `Detector.detect` (transformers, torch: not installed on the box; the detector runs
as a batch job, `mv feedback slip-crops`).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

OWL_MODEL = "google/owlv2-base-patch16-ensemble"
SAM_MODEL = "facebook/sam-vit-base"
PROMPTS = {"slip": ("a paper label", "a handwritten note", "an index card", "a printed label",
                    "a piece of paper with writing"),
           "specimen": ("a mushroom", "a fungus", "a toadstool", "a bracket fungus",
                        "a mushroom cap", "a small mushroom", "a crust fungus", "a jelly fungus",
                        "a slime mold", "a mushroom stem")}
THRESH = {"slip": 0.20, "specimen": 0.12}
MAX_BOX = 0.60          # a specimen box over this share of the frame is the whole scene
MIN_HOLE = 0.002        # a hole in the slip at least this share of the frame is a specimen
MAX_BOXES = 6           # boxes per kind sent to SAM, best first after NMS
MARGIN = 0.15           # around the specimen's bounding box, each side
# A cluster smaller than this share of the frame proposes no crop: the research kept a photo
# when its specimen covered >= 2 of the 24 x 24 patches (2 / 576 of the square).
MIN_CLUSTER = 2 / 576
GAP = 0.05              # specimen parts closer than this share of the short side are one cluster
SIDE = 1024             # the long side the detector runs at
CELL = 8                # px of the detection image per cell of the grid the geometry runs on
GROW_MAX_RATIO = 3.0    # keep_detections(grow=...): a weak box at most this x its sure box
VERSION = "slipcrop-1"  # names the rules and models above; kept with every proposal


# --- geometry (numpy only) ----------------------------------------------------------------------

def components(grid: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected components of a 2-D bool grid: (labels, count), labels 1..count, 0 off."""
    h, w = grid.shape
    lab = np.zeros((h, w), np.int32)
    n = 0
    for y0, x0 in zip(*np.nonzero(grid)):
        if lab[y0, x0]:
            continue
        n += 1
        lab[y0, x0] = n
        todo = deque([(y0, x0)])
        while todo:
            y, x = todo.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and grid[yy, xx] and not lab[yy, xx]:
                        lab[yy, xx] = n
                        todo.append((yy, xx))
    return lab, n


def fill_holes(grid: np.ndarray) -> np.ndarray:
    """The grid with every enclosed hole filled: background not connected to the border."""
    lab, n = components(~grid)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    inside = ~grid & ~np.isin(lab, list(edge))
    return grid | inside


def dilate(grid: np.ndarray, r: int) -> np.ndarray:
    """Square dilation by r cells."""
    out = grid.copy()
    h, w = grid.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
            xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
            out[yd, xd] |= grid[ys, xs]
    return out


def to_grid(mask: np.ndarray, cell: int = CELL) -> np.ndarray:
    """A full-resolution bool mask as a coarse grid: a cell is on when any pixel in it is."""
    h, w = mask.shape
    gh, gw = -(-h // cell), -(-w // cell)
    pad = np.zeros((gh * cell, gw * cell), bool)
    pad[:h, :w] = mask
    return pad.reshape(gh, cell, gw, cell).any(axis=(1, 3))


def specimen_region(slip: np.ndarray, spec: np.ndarray, min_hole: float = MIN_HOLE
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(slip, specimen, on_slip) grids: a specimen pixel on a slip is specimen; a hole in the
    slip of at least `min_hole` of the frame is something lying on it, counted as specimen
    (and also returned alone as on_slip)."""
    slip = slip & ~spec
    holes = fill_holes(slip | spec) & ~slip & ~spec
    lab, n = components(holes)
    on_slip = np.zeros_like(holes)
    floor = min_hole * holes.size
    for i in range(1, n + 1):
        part = lab == i
        if part.sum() >= floor:
            on_slip |= part
    return slip, spec | on_slip, on_slip


def square_box(l: float, t: float, r: float, b: float, w: int, h: int,
               margin: float = MARGIN) -> tuple[int, int, int, int]:
    """A box (original pixels) grown by `margin` of its size on each side, made square where
    the photo allows (so a centre crop of it keeps it all), and kept inside the photo."""
    bw, bh = r - l, b - t
    l, r = l - margin * bw, r + margin * bw
    t, b = t - margin * bh, b + margin * bh
    side = min(max(r - l, b - t), w, h)
    cx, cy = (l + r) / 2, (t + b) / 2
    # a box wider (or taller) than the photo's short side stays as wide, not square
    bw2, bh2 = min(max(r - l, side), w), min(max(b - t, side), h)
    l = min(max(cx - bw2 / 2, 0), w - bw2)
    t = min(max(cy - bh2 / 2, 0), h - bh2)
    return int(round(l)), int(round(t)), int(round(l + bw2)), int(round(t + bh2))


def _overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def crop_boxes(spec: np.ndarray, w: int, h: int, gap: float = GAP,
               min_cluster: float = MIN_CLUSTER, margin: float = MARGIN) -> list[dict]:
    """The crop boxes for a specimen grid that covers a w x h photo: one per well-separated
    cluster (parts closer than `gap` of the short side join), each the cluster's bounding box
    with `margin`, square where possible; clusters whose boxes overlap become one. Each is
    {"box": [l, t, r, b] in original pixels, "area": the cluster's share of the frame,
    "cluster": the cluster's own tight box before the margin and squaring}, the smallest cluster
    first (the riskiest, where false finds sit)."""
    gh, gw = spec.shape
    sx, sy = w / gw, h / gh
    r = max(1, int(round(gap * min(gh, gw))))
    lab, n = components(dilate(spec, r))
    found = []
    for i in range(1, n + 1):
        part = spec & (lab == i)
        area = float(part.sum()) / spec.size
        if area < min_cluster:
            continue
        ys, xs = np.nonzero(part)
        found.append({"raw": [xs.min() * sx, ys.min() * sy, (xs.max() + 1) * sx,
                              (ys.max() + 1) * sy], "area": area})
    merged = True
    while merged:                               # overlapping crops would show one thing twice
        merged = False
        boxes = [square_box(*c["raw"], w, h, margin) for c in found]
        for i in range(len(found)):
            for j in range(i + 1, len(found)):
                if _overlap(boxes[i], boxes[j]):
                    a, b = found[i]["raw"], found[j]["raw"]
                    found[i] = {"raw": [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]),
                                        max(a[3], b[3])], "area": found[i]["area"] + found[j]["area"]}
                    del found[j]
                    merged = True
                    break
            if merged:
                break
    out = [{"box": list(square_box(*c["raw"], w, h, margin)), "area": round(c["area"], 5),
            "cluster": [int(round(v)) for v in c["raw"]]}
           for c in found]
    return sorted(out, key=lambda c: c["area"])


def nms(boxes: list, scores: list, iou: float = 0.5) -> list[int]:
    order = np.argsort(-np.asarray(scores, float))
    keep: list[int] = []
    for i in order:
        ok = True
        for j in keep:
            xa, ya = max(boxes[i][0], boxes[j][0]), max(boxes[i][1], boxes[j][1])
            xb, yb = min(boxes[i][2], boxes[j][2]), min(boxes[i][3], boxes[j][3])
            inter = max(0, xb - xa) * max(0, yb - ya)
            a = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
            b = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
            if inter / max(a + b - inter, 1e-9) > iou:
                ok = False
                break
        if ok:
            keep.append(int(i))
    return keep


def _touch(a, b, pad: float) -> bool:
    return (a[0] - pad < b[2] and b[0] - pad < a[2] and a[1] - pad < b[3] and b[1] - pad < a[3])


def keep_detections(dets: list[tuple[list, float, str]], frame_area: float,
                    grow: float | None = None) -> dict:
    """OWLv2's (box, score, kind) detections by kind, at each kind's threshold, the
    whole-scene specimen boxes dropped, NMS'd and capped at MAX_BOXES, best first.

    With `grow` (a lower specimen score): a weaker specimen box is kept too when it touches or
    overlaps a kept one, repeatedly, so a cap OWLv2 scores low beside the stem it is sure of
    joins it (Steve, 2026-10-27: a Hygrocybe crop got the stem, not the cap at 0.04). A weak
    box touching nothing kept (a logo printed on the slip) stays out."""
    found: dict[str, list] = {"slip": [], "specimen": []}
    weak = []
    for b, s, k in dets:
        big = (b[2] - b[0]) * (b[3] - b[1]) > MAX_BOX * frame_area
        if s >= THRESH[k] and not (k == "specimen" and big):
            found[k].append((b, s))
        elif k == "specimen" and grow is not None and s >= grow and not big:
            weak.append((b, s))
    # One step only (no chaining: weak boxes linked to weak boxes spread over the scene), and a
    # weak box at most GROW_MAX_RATIO times the area of the sure box it overlaps.
    area = lambda bx: max(1e-9, (bx[2] - bx[0]) * (bx[3] - bx[1]))  # noqa: E731
    seeds = list(found["specimen"])
    for w in weak:
        if any(_touch(w[0], f[0], 0.0) and area(w[0]) <= GROW_MAX_RATIO * area(f[0])
               for f in seeds):
            found["specimen"].append(w)
    out = {}
    for k, items in found.items():
        keep = nms([i[0] for i in items], [i[1] for i in items])[:MAX_BOXES]
        out[k] = [items[i] for i in keep]
    return out


def version(grow: float | None = None) -> str:
    """The proposal's version: VERSION, and the touching-box rule when it is on."""
    return VERSION if grow is None else f"{VERSION}+grow{grow:g}"


# --- the detector (transformers + torch, loaded on first use) -----------------------------------

@dataclass
class Detector:
    device: str = "cpu"
    threads: int | None = None
    grow: float | None = None          # keep_detections' lower score for touching boxes
    _m: dict = field(default_factory=dict, repr=False)

    def _load(self):
        if self._m:
            return self._m
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor, SamModel, SamProcessor
        if self.threads:
            torch.set_num_threads(self.threads)
        dt = torch.float16 if self.device == "cuda" else torch.float32
        self._m = {"torch": torch, "dt": dt,
                   "owl_p": Owlv2Processor.from_pretrained(OWL_MODEL),
                   "owl": Owlv2ForObjectDetection.from_pretrained(OWL_MODEL, torch_dtype=dt)
                   .eval().to(self.device),
                   "sam_p": SamProcessor.from_pretrained(SAM_MODEL),
                   "sam": SamModel.from_pretrained(SAM_MODEL, torch_dtype=dt).eval().to(self.device)}
        return self._m

    def _put(self, batch):
        m = self._m
        return {k: (v.to(self.device, m["dt"]) if v.is_floating_point() else v.to(self.device))
                if hasattr(v, "to") else v for k, v in batch.items()}

    def masks(self, img) -> tuple[np.ndarray, np.ndarray, dict]:
        """(slip, specimen) full-resolution masks of the detection image and the kept boxes."""
        m = self._load()
        torch = m["torch"]
        W, H = img.size
        texts = list(PROMPTS["slip"] + PROMPTS["specimen"])
        kinds = ["slip"] * len(PROMPTS["slip"]) + ["specimen"] * len(PROMPTS["specimen"])
        with torch.no_grad():
            inp = m["owl_p"](text=[texts], images=img, return_tensors="pt")
            o = m["owl"](**self._put(inp))
            o.logits, o.pred_boxes = o.logits.float().cpu(), o.pred_boxes.float().cpu()
            # OWLv2 pads the photo to a square at the bottom/right: boxes are on that square
            s = max(W, H)
            res = m["owl_p"].post_process_object_detection(
                o, threshold=min(0.05, self.grow or 0.05), target_sizes=torch.tensor([[s, s]]))[0]
        dets = [([min(max(v, 0.0), (W, H, W, H)[i]) for i, v in enumerate(b)], sc, kinds[lb])
                for b, sc, lb in zip(res["boxes"].tolist(), res["scores"].tolist(),
                                     res["labels"].tolist())]
        kept = keep_detections(dets, W * H, self.grow)
        out, emb = {}, None
        for k in ("slip", "specimen"):
            mk = np.zeros((H, W), bool)
            items = kept[k]
            if items:
                if emb is None:
                    with torch.no_grad():
                        si = m["sam_p"](img, return_tensors="pt")
                        emb = m["sam"].get_image_embeddings(si["pixel_values"].to(self.device, m["dt"]))
                with torch.no_grad():
                    si = m["sam_p"](img, input_boxes=[[i[0] for i in items]], return_tensors="pt")
                    so = m["sam"](image_embeddings=emb,
                                  input_boxes=si["input_boxes"].to(self.device, m["dt"]),
                                  multimask_output=True)
                    ms = m["sam_p"].image_processor.post_process_masks(
                        so.pred_masks.float().cpu(), si["original_sizes"],
                        si["reshaped_input_sizes"])[0]
                best = so.iou_scores[0].float().cpu().argmax(-1)
                for j, (b, _s) in enumerate(items):
                    mm = ms[j, int(best[j])].numpy().astype(bool)
                    x0, y0, x1, y1 = [int(round(v)) for v in b]
                    inbox = np.zeros_like(mm)        # a mask leaking past its box is background
                    inbox[max(0, y0):y1, max(0, x0):x1] = True
                    mk |= mm & inbox
            out[k] = mk
        boxes = {k: [{"box": [round(v, 1) for v in b], "score": round(s, 3)} for b, s in v]
                 for k, v in kept.items()}
        return out["slip"], out["specimen"], boxes

    def detect(self, photo) -> dict:
        """A photo (PIL, EXIF already applied) -> the proposal: crop boxes on the ORIGINAL
        photo's pixels, smallest specimen first, and the areas found."""
        from PIL import Image
        img = photo.convert("RGB")
        w, h = img.size
        k = SIDE / max(w, h)
        small = img.resize((max(1, round(w * k)), max(1, round(h * k))), Image.BICUBIC) \
            if k < 1 else img
        slip_m, spec_m, found = self.masks(small)
        slip, spec, on_slip = specimen_region(to_grid(slip_m), to_grid(spec_m))
        # OWLv2's kept boxes with their scores, on the original photo's pixels like the crops
        back = w / small.size[0]
        found = {kind: [{"box": [round(v * back, 1) for v in d["box"]], "score": d["score"]}
                        for d in dets] for kind, dets in found.items()}
        return {"version": version(self.grow), "width": w, "height": h,
                "slip_area": round(float(slip.mean()), 4),
                "specimen_area": round(float(spec.mean()), 4),
                "on_slip_area": round(float(on_slip.mean()), 4),
                "crops": crop_boxes(spec, w, h), "detections": found}
