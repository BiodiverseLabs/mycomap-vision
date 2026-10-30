"""The query photo filters of paper E34 (b) and E35 (a3-spec §1): which of a scored find's
photos a filtered comparator leaves out before it answers, by pinned rules over each photo's
frozen BioCLIP 2 (224) vector. The filter touches the query photos only, never the reference.

The rules are the pinned E34 flag function (research exp/photo-type-tagger 206422a,
experiments/photo-type-tagger/e34_flags.py `flags`), read from its bundle
(e34_flag_rules.npz) and the 224 review gate (gate-e10/tuned-revised), unchanged:

  R6_micrograph    micrograph probe p >= 0.5
  R6_view          view softmax p >= 0.02 for habitat or dried (float16, as the rule ran)
  R6_gate          the gate's p(fungus) < its not_fungus threshold
  R6b_slip         slip probe p >= 0.7 and float16(label cosine) >= 0.42 and micrograph p < 0.5
  R7_uv, R7_phylo_tree, R7_drawing   tagger v1 probes at 0.98, 0.95, 0.95

and, declared in -a3:

  R6_view_habitat  the habitat column of R6_view alone (E35 b-habitat)
  V21_spore_print, V21_chemical   photo-type tagger v2.1 (t15 model, softmax over its classes
                   on the same vectors) p >= the declared threshold (0.5, t17's High band)

Every file is pinned by sha256 and refused when it differs. Nothing is fitted here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

BUNDLE_RULES = ("R6_micrograph", "R6_view", "R6_view_habitat", "R6_gate", "R6b_slip",
                "R7_uv", "R7_phylo_tree", "R7_drawing")
TAGGER_RULES = ("V21_spore_print", "V21_chemical")
RULES = BUNDLE_RULES + TAGGER_RULES
TAGGER_V1 = {"uv": 0.98, "phylo_tree": 0.95, "drawing": 0.95}
BACKBONE = "bioclip-2"


class PinError(RuntimeError):
    """A filter file is not the one the config pins."""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pinned(path, sha256: str, what: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise PinError(f"{what}: {path} not found")
    got = sha256_file(path)
    if got != sha256:
        raise PinError(f"{what}: {path.name} has sha256 {got[:12]}, not the pinned {sha256[:12]}")
    return path


def unit(a) -> np.ndarray:
    a = np.asarray(a, np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def _h16(a) -> np.ndarray:
    # the rules ran on float16-stored values (s6b's view p, s20's cosines): compare as they did
    return np.asarray(a, np.float16).astype(np.float64)


def flags(vectors, bundle: dict, gate=None, tagger: dict | None = None,
          threshold: float = 0.5, rules=RULES) -> dict[str, np.ndarray]:
    """Each rule named in `rules`: a bool per photo, from frozen BioCLIP 2 vectors (n, 768).
    `bundle`: the arrays of e34_flag_rules.npz; `gate`: a FungusGate (R6_gate); `tagger`:
    {"W", "b", "classes"} of tagger v2.1 (V21_*)."""
    v = unit(vectors)
    z = bundle
    out: dict[str, np.ndarray] = {}

    def sig(w, b):
        return 1 / (1 + np.exp(-(v @ np.asarray(w, np.float32) + float(b))))
    micro = sig(z["micro_w"], z["micro_b"])
    need = set(rules)
    if need & {"R6_view", "R6_view_habitat"}:
        logits = float(z["view_scale"]) * (v @ np.asarray(z["view_text"], np.float32).T)
        logits -= logits.max(1, keepdims=True)
        e = np.exp(logits)
        vp = e / e.sum(1, keepdims=True)
        names = [str(x) for x in z["view_names"]]
        habitat = _h16(vp[:, names.index("habitat")]) >= 0.02
        out["R6_view"] = habitat | (_h16(vp[:, names.index("dried")]) >= 0.02)
        out["R6_view_habitat"] = habitat
    out["R6_micrograph"] = micro >= 0.5
    if "R6_gate" in need:
        if gate is None:
            raise PinError("R6_gate needs the pinned gate")
        out["R6_gate"] = gate.p_fungus(v) < gate.thresholds.not_fungus
    if "R6b_slip" in need:
        out["R6b_slip"] = ((sig(z["slip_w"], z["slip_b"]) >= 0.7)
                           & (_h16(v @ np.asarray(z["label_text"], np.float32)) >= 0.42)
                           & (micro < 0.5))
    for k, t in TAGGER_V1.items():
        if f"R7_{k}" in need:
            out[f"R7_{k}"] = sig(z[f"{k}_w"], z[f"{k}_b"]) >= t
    if need & set(TAGGER_RULES):
        if tagger is None:
            raise PinError("the V21 rules need the pinned tagger v2.1 model")
        logits = v @ np.asarray(tagger["W"], np.float32) + np.asarray(tagger["b"], np.float32)
        logits -= logits.max(1, keepdims=True)
        e = np.exp(logits)
        p = e / e.sum(1, keepdims=True)
        classes = [str(c) for c in tagger["classes"]]
        for r in TAGGER_RULES:
            if r in need:
                out[r] = p[:, classes.index(r[len("V21_"):])] >= threshold
    return {r: out[r] for r in rules}


def load(spec: dict, resolve) -> dict:
    """The pinned files a filter spec names, read and checked: {"bundle", "gate", "tagger",
    "threshold"}. PinError on any file that differs from its pin."""
    from .fungusgate import GATE_NAME, FungusGate
    b = spec["bundle"]
    with np.load(pinned(resolve(b["file"]), b["sha256"], "query_photo_filter bundle")) as f:
        bundle = {k: f[k] for k in f.files}
    gate = None
    if "R6_gate" in spec["rules"]:
        g = spec["gate"]
        root = Path(resolve(g["dir"]))
        for name, sha in g["files"].items():
            pinned(root / name, sha, "query_photo_filter gate")
        if set(g["files"]) != {f"{GATE_NAME}.npz", f"{GATE_NAME}.json"}:
            raise PinError(f"query_photo_filter gate: pins {GATE_NAME}.npz and .json")
        gate = FungusGate.load(root)
    tagger = None
    if set(spec["rules"]) & set(TAGGER_RULES):
        t = spec["tagger_v21"]
        with np.load(pinned(resolve(t["file"]), t["sha256"], "query_photo_filter tagger")) as f:
            tagger = {"W": f["W"], "b": f["b"], "classes": f["classes"]}
        if t.get("bands"):
            pinned(resolve(t["bands"]["file"]), t["bands"]["sha256"], "query_photo_filter bands")
    return {"bundle": bundle, "gate": gate, "tagger": tagger,
            "threshold": float((spec.get("tagger_v21") or {}).get("threshold", 0.5))}


def check_spec(cid: str, spec) -> None:
    """A comparator's query_photo_filter (a3-spec §1): {"rules": [RULES], "bundle": {file,
    sha256}, "gate": {dir, files: {name: sha256}} (with R6_gate), "tagger_v21": {file,
    sha256, bands: {file, sha256}, threshold} (with a V21 rule), "backbone": "bioclip-2",
    "backbone_sha256": the hub checkpoint's, "empty": "miss_and_drop_row"}."""
    import re

    def sha_ok(x):
        return isinstance(x, str) and re.fullmatch(r"[0-9a-f]{64}", x)
    if not isinstance(spec, dict):
        raise ValueError(f"comparator {cid}: query_photo_filter is an object")
    rules = spec.get("rules")
    if not isinstance(rules, list) or not rules or len(set(rules)) != len(rules):
        raise ValueError(f"comparator {cid}: query_photo_filter rules is a list of distinct rules")
    unknown = [r for r in rules if r not in RULES]
    if unknown:
        raise ValueError(f"comparator {cid}: unknown query_photo_filter rule(s) {unknown}; "
                         f"known: {list(RULES)}")
    b = spec.get("bundle")
    if not (isinstance(b, dict) and b.get("file") and sha_ok(b.get("sha256"))):
        raise ValueError(f"comparator {cid}: query_photo_filter bundle is {{file, sha256}}")
    if "R6_gate" in rules:
        g = spec.get("gate")
        if not (isinstance(g, dict) and g.get("dir") and isinstance(g.get("files"), dict)
                and g["files"] and all(sha_ok(x) for x in g["files"].values())):
            raise ValueError(f"comparator {cid}: R6_gate pins the gate: {{dir, files: "
                             "{name: sha256}}")
    if set(rules) & set(TAGGER_RULES):
        t = spec.get("tagger_v21")
        th = (t or {}).get("threshold")
        if not (isinstance(t, dict) and t.get("file") and sha_ok(t.get("sha256"))
                and isinstance(th, (int, float)) and not isinstance(th, bool) and 0 < th < 1):
            raise ValueError(f"comparator {cid}: a V21 rule pins tagger_v21 {{file, sha256, "
                             "threshold in (0, 1)}")
        bands = t.get("bands")
        if bands is not None and not (isinstance(bands, dict) and bands.get("file")
                                      and sha_ok(bands.get("sha256"))):
            raise ValueError(f"comparator {cid}: tagger_v21 bands is {{file, sha256}}")
    if spec.get("backbone") != BACKBONE or not sha_ok(spec.get("backbone_sha256")):
        raise ValueError(f"comparator {cid}: query_photo_filter backbone is {BACKBONE!r}, its "
                         "hub checkpoint pinned by backbone_sha256")
    if spec.get("empty") != "miss_and_drop_row":
        raise ValueError(f"comparator {cid}: query_photo_filter empty is \"miss_and_drop_row\"")
