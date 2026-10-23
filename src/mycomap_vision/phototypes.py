"""What kind of photo is it? Photo-type tags on Identify (Steve, 2026-10-22).

Each photo sent to Identify (the site, API v1, the app's sessions) is tagged by the
photo-type tagger v2: one softmax head over the photo's FROZEN BioCLIP 2 vector, the same
vector the not-a-fungus gate reads (fungusgate.py), so the tower runs once for both. The
head was trained in the private research repo on Human Feedback decisions (t10, "tagger
v2") and banded per class and confidence by eye (t11; the bands t13 seeded with): high is
a measured precision of 85% or more, medium 50-85%, low below.

What a tag does:
  - micrograph, slip (a label or field slip) and habitat photos in the HIGH band are left
    out of the observation's answer, the way a gated photo is. Each can be counted anyway
    (`include`: the photo indices asked for).
  - every other kind (UV, chemical, spore print, dried, people, tree, drawing, not_fungus)
    is tagged only, and still counts.
  - when every photo the gate kept would be left out, none is: the answer is made from all
    of them and says so. The gate refuses when every photo is gated, because a photo that
    isn't a fungus can't be named; a micrograph, label or habitat photo of a fungus still
    carries some of it (the DNA references hold such photos too), so a weaker answer beats
    none, and the left-out set would be empty anyway.

The head is a deploy-time file (`mv photo-tagger build` makes it from the research output),
never committed: trained models count as data until their photos' permissions allow
release (CLAUDE.md). It is served only when MV_PHOTO_TAGGER names it, so code without the
file changes nothing. Its json sidecar holds the .npz's sha256 (checked at load) and the
research model's sha256, which must be a pinned one (PINNED): v2.1 (t15) is not. A head
reads the frozen vectors of ONE base model, the one it was trained on (v2: bioclip-2, the
224 px BioCLIP 2); it is served only where the gate reads that same base. A head trained on
another base's vectors is pinned with its own sha and base: v2.1-336 (research t20, v2.1's
recipe on bioclip-2-336 vectors, banded by eye in t21; Steve approved the refit 2026-10-22)
is the head for the 336 px gate the site serves.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config

NAME = "photo-tagger"
SETTING = "MV_PHOTO_TAGGER"

# Research models this server may serve, by the sha256 of their model.npz: the version's
# name, the base model whose frozen vectors it was trained on (and so must read), and where
# the model and its bands came from in the research repo (exp/photo-type-tagger).
PINNED = {
    "526ca22f62e66643193f3d97ba9ab0684d9ab602d8b841895ea1b41d310230c1":
        {"version": "v2", "base": "bioclip-2",
         "source": "photo-type tagger t10 (research exp/photo-type-tagger)",
         "bands_from": "t13_v2_seed.json (by-eye bands, t11)"},
    # v2.1's recipe on bioclip-2-336 vectors, for the 336 px gate (t20; research a27bb67).
    "128f4f874dad41253e57433ddd196a54e2f8929600df82bf706986db86ada1e6":
        {"version": "v2.1-336", "base": "bioclip-2-336",
         "source": "photo-type tagger t20 (research exp/photo-type-tagger a27bb67)",
         "bands_from": "t21_336_bands.json (by-eye bands, t21)"},
}

SPECIMEN = "specimen"
HIGH, MEDIUM, LOW = "high", "medium", "low"
BANDS = (HIGH, MEDIUM, LOW)
# Kinds left out of the answer when tagged in the HIGH band (Steve, 2026-10-22).
LEAVE_OUT = ("micrograph", "slip", "habitat")
# A chip on the page for these bands; a low-band tag is mostly wrong, so it shows none.
CHIP_BANDS = (HIGH, MEDIUM)

# The chip's words, per kind.
LABELS = {
    "micrograph": "Microscopy", "slip": "Label", "habitat": "Habitat", "uv": "UV",
    "chemical": "Chemical test", "spore_print": "Spore print", "dried": "Dried",
    "people": "People", "tree": "Phylo tree", "drawing": "Drawing", "not_fungus": "Not a fungus",
}
# How a hint names a left-out kind.
LOOKS = {"micrograph": "a micrograph", "slip": "a label or field slip",
         "habitat": "a habitat photo"}
LOOK_MANY = {"micrograph": "micrographs", "slip": "labels or field slips",
             "habitat": "habitat photos"}


# When every photo would be left out, none is; the answer says this (Steve, 2026-10-22).
KEPT_ALL_NOTE = "Only non-specimen photos: this answer is less reliable."


def base_mismatch(tagger_base: str, gate_base: str) -> str:
    """Why a tagger can't be served beside this gate: it would need another tower's vectors."""
    return (f"the tagger head reads {tagger_base} frozen vectors but the gate reads "
            f"{gate_base}; serve a head trained on {gate_base} frozen vectors (pinned with its "
            "own sha in phototypes.PINNED)")


class Refused(ValueError):
    """A tagger file that must not be served: sha256 mismatch, unpinned model, bad shape."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def band_of(bands: dict, kind: str, p: float, floor: float) -> str | None:
    """The band of a tag: None for a specimen photo; LOW under the floor or under every
    measured range; else the first range of `bands[kind]` ([[min p, band], ...], highest
    first) that `p` reaches."""
    if kind == SPECIMEN:
        return None
    if p < floor:
        return LOW
    for lo, band in bands.get(kind) or []:
        if p >= float(lo):
            return str(band)
    return LOW


@dataclass
class PhotoTagger:
    W: np.ndarray                  # (dim, classes)
    b: np.ndarray                  # (classes,)
    classes: tuple[str, ...]
    bands: dict                    # {kind: [[min p, band], ...] highest first}
    floor: float                   # below this every tag is LOW
    base: str = "bioclip-2"        # whose frozen image vectors it reads
    version: str | None = None     # PINNED name of the research model ("v2")
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        self.W = np.asarray(self.W, np.float32)
        self.b = np.asarray(self.b, np.float32)
        self.classes = tuple(str(c) for c in self.classes)
        if self.W.ndim != 2 or self.W.shape[1] != len(self.classes) or self.b.shape != (
                len(self.classes),):
            raise Refused(f"head shape {self.W.shape} / {self.b.shape} for "
                          f"{len(self.classes)} classes")
        if SPECIMEN not in self.classes:
            raise Refused("the head has no specimen class")
        unknown = sorted(set(self.bands) - set(self.classes))
        if unknown:
            raise Refused(f"bands for classes the head doesn't have: {unknown}")
        for kind, ranges in self.bands.items():
            for lo, band in ranges:
                if band not in BANDS or not 0.0 <= float(lo) <= 1.0:
                    raise Refused(f"bad band {kind}: {lo}, {band}")

    def probs(self, base_vectors: np.ndarray) -> np.ndarray:
        """The softmax over the classes for each photo's frozen base vector (any scale)."""
        v = np.asarray(base_vectors, np.float32)
        v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        z = v @ self.W + self.b
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    def tag(self, base_vectors: np.ndarray) -> list[dict]:
        """One tag per photo: {"type", "p", "band", "label"}. `type` is the top class
        ("specimen": an ordinary photo of the fungus); `label` is the chip's words, or None
        when the page shows no chip (a specimen photo, or a low band)."""
        out = []
        for row in self.probs(base_vectors):
            k = int(row.argmax())
            kind, p = self.classes[k], float(row[k])
            band = band_of(self.bands, kind, p, self.floor)
            out.append({"type": kind, "p": round(p, 4), "band": band,
                        "label": LABELS.get(kind, kind) if band in CHIP_BANDS else None})
        return out

    def fingerprint(self) -> str:
        """A short hash of what decides the tags (head, classes, bands, floor, base): a tag
        made under another fingerprint is made again, and kept answers key on it."""
        h = hashlib.sha256(json.dumps([list(self.classes), self.bands, self.floor, self.base],
                                      sort_keys=True).encode())
        h.update(self.W.tobytes())
        h.update(self.b.tobytes())
        return h.hexdigest()[:16]

    def summary(self) -> dict:
        """What /api/health says about the tagger."""
        return {"version": self.version, "base": self.base, "fingerprint": self.fingerprint(),
                "leave_out": [k for k in LEAVE_OUT if k in self.classes], "leave_out_band": HIGH}

    # --- files -----------------------------------------------------------------------

    def save(self, path: Path) -> Path:
        """`path` (.npz) and its json sidecar, which records the .npz's sha256."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            np.savez(f, W=self.W, b=self.b, classes=np.array(self.classes))
        meta = {"base": self.base, "bands": self.bands, "floor": self.floor,
                "version": self.version, "sha256": sha256_file(path),
                "provenance": self.provenance}
        sidecar(path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "PhotoTagger":
        """The tagger at `path`, or Refused: a missing file or sidecar, an .npz whose sha256
        isn't the one its sidecar records, or a research model that isn't PINNED."""
        path = Path(path)
        meta_path = sidecar(path)
        if not (path.is_file() and meta_path.is_file()):
            raise Refused(f"{path} or its {meta_path.name} is missing")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        got = sha256_file(path)
        if got != meta.get("sha256"):
            raise Refused(f"{path.name} has sha256 {got[:12]}…, its sidecar says "
                          f"{str(meta.get('sha256'))[:12]}…")
        source = (meta.get("provenance") or {}).get("source_sha256")
        if source not in PINNED:
            raise Refused(f"the research model {str(source)[:12]}… is not a pinned one")
        pin = PINNED[source]
        if meta.get("base") != pin["base"]:
            raise Refused(f"{path.name} says it reads {meta.get('base')} vectors, but tagger "
                          f"{pin['version']} was trained on {pin['base']} vectors")
        with np.load(path) as f:
            return cls(f["W"], f["b"], tuple(str(c) for c in f["classes"]), meta["bands"],
                       float(meta["floor"]), pin["base"], pin["version"],
                       meta.get("provenance") or {})

    @classmethod
    def from_setting(cls) -> "PhotoTagger | None":
        """The tagger MV_PHOTO_TAGGER names (Refused if it can't be served), or None when
        the setting is unset: then photos are not tagged."""
        named = config.setting(SETTING)
        return cls.load(Path(named)) if named else None


def sidecar(path: Path) -> Path:
    return Path(path).with_suffix(".json")


def build(model_npz: Path, seed_json: Path, built_at: str | None = None) -> PhotoTagger:
    """The served tagger from the research output: a pinned model.npz (W, b, classes; v2:
    PT_OUT/t10, v2.1-336: PT_OUT/t20) and the bands json made for it, which records the
    model's sha256 (v2: PT_OUT/t13_v2_seed.json, v2.1-336: PT_OUT/t21_336_bands.json).
    Refused unless the model is PINNED and the bands file names it. Its base is the one
    PINNED records for it (v2: bioclip-2, v2.1-336: bioclip-2-336)."""
    model_sha = sha256_file(Path(model_npz))
    if model_sha not in PINNED:
        raise Refused(f"{model_npz} (sha256 {model_sha[:12]}…) is not a pinned tagger model")
    seed = json.loads(Path(seed_json).read_text(encoding="utf-8"))
    if seed.get("model_sha256") != model_sha:
        raise Refused(f"{Path(seed_json).name} bands were made for model "
                      f"{str(seed.get('model_sha256'))[:12]}…, not {model_sha[:12]}…")
    with np.load(model_npz) as f:
        W, b, classes = f["W"], f["b"], tuple(str(c) for c in f["classes"])
    bands = {k: [[float(lo), str(band)] for lo, band in v] for k, v in seed["bands"].items()}
    pin = PINNED[model_sha]
    return PhotoTagger(W, b, classes, bands, float(seed["floor"]), pin["base"], pin["version"],
                       {"source_sha256": model_sha,
                        "seed_sha256": sha256_file(Path(seed_json)),
                        "source": pin.get("source"),
                        "bands_from": pin.get("bands_from"),
                        "built_at": built_at})


# --- the decision ----------------------------------------------------------------------

def would_leave_out(tag: dict | None) -> bool:
    """A tag that leaves its photo out of the answer: a LEAVE_OUT kind in the HIGH band."""
    return bool(tag) and tag.get("type") in LEAVE_OUT and tag.get("band") == HIGH


def decide(tags: list[dict | None], counted: list[int], include=()) -> dict:
    """Which of the photos the gate kept (`counted`, indices) are left out by their tag.
    `include`: indices asked to count anyway. When every counted photo would be left out,
    none is (`kept_all`). {"left_out", "included", "kept_all"}, indices ascending."""
    asked = {int(i) for i in include}
    would = [i for i in counted if would_leave_out(tags[i])]
    included = [i for i in would if i in asked]
    left = [i for i in would if i not in asked]
    kept_all = bool(left) and len(left) == len(counted)
    return {"left_out": [] if kept_all else left, "included": included, "kept_all": kept_all}


def annotate(tags: list[dict | None], decision: dict) -> list[dict | None]:
    """Each photo's tag with whether it was left out and whether it was counted anyway."""
    left, inc = set(decision["left_out"]), set(decision["included"])
    return [None if t is None else {**t, "left_out": i in left, "included": i in inc}
            for i, t in enumerate(tags)]


def _photos(numbers: list[int]) -> str:
    n = [str(i + 1) for i in numbers]
    return (f"Photo {n[0]}" if len(n) == 1 else
            "Photos " + ", ".join(n[:-1]) + f" and {n[-1]}")


def hints(tags: list[dict | None], decision: dict) -> list[str]:
    """What the page and API say about the photos the tags left out (or would have)."""
    if decision["kept_all"]:
        return [KEPT_ALL_NOTE]
    left = decision["left_out"]
    if not left:
        return []
    kinds = {tags[i]["type"] for i in left}
    if len(left) == 1:
        return [f"{_photos(left)} looks like {LOOKS[next(iter(kinds))]}, so it was left out "
                "of the answer."]
    what = (LOOK_MANY[next(iter(kinds))] if len(kinds) == 1
            else "micrographs, labels or habitat photos")
    return [f"{_photos(left)} look like {what}, so they were left out of the answer."]


def parse_include(value, photos: int | None = None) -> list[int]:
    """The photos asked to count anyway, as sorted unique 0-based indices, from a list of
    ints or a comma-separated string ("" or None: none). ValueError when one isn't a
    photo index (0 to `photos` - 1, or to 9 when the count isn't known yet)."""
    if value is None or value == "":
        return []
    items = value.split(",") if isinstance(value, str) else value
    if not isinstance(items, (list, tuple)):
        raise ValueError("include is a list of photo indices")
    top = (photos if photos is not None else 10) - 1
    out = set()
    for x in items:
        if isinstance(x, bool):
            raise ValueError("include is a list of photo indices")
        try:
            i = int(str(x).strip()) if isinstance(x, str) else x
        except ValueError:
            raise ValueError("include is a list of photo indices") from None
        if not isinstance(i, int) or not 0 <= i <= top:
            raise ValueError(f"include holds photo indices 0 to {top}")
        out.add(i)
    return sorted(out)
