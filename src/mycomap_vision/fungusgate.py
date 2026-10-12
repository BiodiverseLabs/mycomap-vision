"""Is this photo a fungus at all? A gate in front of the identifier.

Vision answers by nearest DNA-verified specimen, and its confidence is a softmax over
the candidates it holds, so a photo of a duck still gets a "best" mushroom, often with a
high stated chance: calibration only holds for fungus photos. The gate looks at each
photo first and says `fungus`, `unsure` or `not_fungus`.

Signal: BioCLIP 2 zero-shot. BioCLIP 2 is a CLIP model trained on the tree of life, so
its text tower can name a photo's kingdom. Each photo's FROZEN BioCLIP 2 vector is
compared with text vectors for a set of classes (fungus orders and growth forms on one
side; birds, mammals, insects, plants, slime moulds, people, buildings, food, objects,
scenery on the other); p(fungus) is the softmax mass on the fungus classes. The text
vectors are computed once (`mv fungus-gate build`) and kept in data/models; serving
needs only the image tower.

The fine-tuned model's vectors no longer line up with the text tower (fungus photos
score a median p(fungus) of about 0.1 on them), so the gate needs the frozen vector.
The fine-tune changed only the last blocks, so the backbone runs the shared blocks once
and forks (models.FinetunedBackbone.encode_with_base): about 1/6 more compute a photo.

Thresholds are chosen on the held-out DEV split and the not-a-fungus set only, never on
a test split (experiment entry 2026-10-09-not-fungus-gate, in the private research repo).

A gated photo is left out of the observation's answer (a duck among mushroom photos
must not sink it); when every photo is gated, the observation gets no species answer
and no stated chance at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config

FUNGUS, UNSURE, NOT_FUNGUS = "fungus", "unsure", "not_fungus"
MODES = ("zero-shot", "probe", "both")
VERDICTS = (FUNGUS, UNSURE, NOT_FUNGUS)
GATE_NAME = "fungus-gate"

# Messages the API and page show.
MESSAGES = {
    NOT_FUNGUS: "This doesn't look like a fungus.",
    UNSURE: "Not sure this is a fungus.",
}

# What a find is said to look like when it isn't judged a fungus and its photos score high
# for an in-scope class (Steve, 2026-10-11: "This looks like a slime mould. Slime moulds
# aren't fungi, but MycoMap holds some DNA-verified ones").
LOOKS_LIKE = {
    "slime": {
        "message": "This looks like a slime mould.",
        "hint": ("This looks like a slime mould. Slime moulds aren't fungi, but MycoMap holds "
                 "some DNA-verified ones; the closest are below, without a stated chance."),
    },
}

TEMPLATES = ("a photo of {}.", "a photo of a {}.")

# (class, side, texts). Taxonomic strings follow BioCLIP's training captions; common
# names cover what taxonomy can't (people, objects, scenery). Side "fungus" counts
# toward p(fungus); "other" against it. Lichens are fungi (iNat files them under
# Fungi, and some are DNA-verified MycoMap records), so they count as fungus.
CLASSES: list[tuple[str, str, tuple[str, ...]]] = [
    ("agaric", "fungus", ("Fungi Basidiomycota Agaricomycetes Agaricales", "mushroom",
                          "gilled mushroom")),
    ("bolete", "fungus", ("Fungi Basidiomycota Agaricomycetes Boletales", "bolete mushroom")),
    ("polypore", "fungus", ("Fungi Basidiomycota Agaricomycetes Polyporales", "bracket fungus",
                            "shelf fungus on wood")),
    ("russula", "fungus", ("Fungi Basidiomycota Agaricomycetes Russulales",)),
    ("chanterelle", "fungus", ("Fungi Basidiomycota Agaricomycetes Cantharellales",
                               "coral fungus")),
    ("jelly", "fungus", ("Fungi Basidiomycota Tremellomycetes Tremellales", "jelly fungus")),
    ("puffball", "fungus", ("Fungi Basidiomycota Agaricomycetes Phallales", "puffball",
                            "stinkhorn")),
    ("cup", "fungus", ("Fungi Ascomycota Pezizomycetes Pezizales", "cup fungus", "morel")),
    ("asco", "fungus", ("Fungi Ascomycota Sordariomycetes", "Fungi Ascomycota Leotiomycetes",
                        "Fungi")),
    ("fungus-generic", "fungus", ("fungus", "mushrooms on the forest floor",
                                  "a mushroom specimen")),
    ("lichen", "fungus", ("Fungi Ascomycota Lecanoromycetes", "lichen")),
    ("bird", "other", ("Animalia Chordata Aves", "bird", "duck")),
    ("mammal", "other", ("Animalia Chordata Mammalia", "mammal")),
    ("herp", "other", ("Animalia Chordata Reptilia", "Animalia Chordata Amphibia", "frog",
                       "snake")),
    ("fish", "other", ("Animalia Chordata Actinopterygii", "fish")),
    ("insect", "other", ("Animalia Arthropoda Insecta", "insect",
                         "Animalia Arthropoda Arachnida", "spider")),
    ("mollusc", "other", ("Animalia Mollusca Gastropoda", "snail", "slug")),
    ("other-invert", "other", ("Animalia Annelida", "Animalia Cnidaria Anthozoa",
                               "Animalia Porifera", "worm")),
    ("flower", "other", ("Plantae Tracheophyta Magnoliopsida", "flower", "flowering plant")),
    ("monocot", "other", ("Plantae Tracheophyta Liliopsida", "grass")),
    ("fern", "other", ("Plantae Tracheophyta Polypodiopsida", "fern")),
    ("conifer", "other", ("Plantae Tracheophyta Pinopsida", "pine cone", "conifer")),
    ("moss", "other", ("Plantae Bryophyta", "moss", "Plantae Marchantiophyta")),
    ("tree", "other", ("tree", "leaves")),
    ("slime", "other", ("Protozoa Mycetozoa Myxomycetes", "slime mold")),
    ("person", "other", ("person", "people", "human face")),
    ("building", "other", ("building", "house", "city street")),
    ("vehicle", "other", ("car", "vehicle")),
    ("food", "other", ("food", "plate of food", "bread")),
    ("object", "other", ("household object", "toy", "furniture", "electronic device")),
    ("scenery", "other", ("landscape", "sky", "beach")),
    ("pet", "other", ("dog", "cat")),
]


def observation_verdict(verdicts: list[str]) -> str:
    """A find is a fungus when any of its photos is; not sure when none is but some
    photo is unsure; not a fungus when every photo is judged not one."""
    if FUNGUS in verdicts:
        return FUNGUS
    if UNSURE in verdicts:
        return UNSURE
    return NOT_FUNGUS


def find_looks_like(photos: list[dict], overall: str) -> str | None:
    """What a find not judged a fungus looks like: the class its photos name (LOOKS_LIKE),
    or None. A find judged a fungus never gets one."""
    if overall == FUNGUS:
        return None
    named = [p["looks_like"] for p in photos if p.get("looks_like")]
    return named[0] if named else None


def gate_hints(verdicts: list[str], overall: str, looks_like: str | None = None) -> list[str]:
    """What the page says about the gate, before the usual hints."""
    if looks_like is not None and overall != FUNGUS:
        return [LOOKS_LIKE[looks_like]["hint"]]
    if overall == NOT_FUNGUS:
        return ["This doesn't look like a fungus, so no identification is given. Vision "
                "only knows DNA-verified fungi; if it is one, add a clear photo of the "
                "fruiting body."]
    hints = []
    if overall == UNSURE:
        hints.append("Not sure this is a fungus, so no chance is stated: the names below are "
                     "only the closest DNA-verified look-alikes. A clear photo of the cap, "
                     "underside and stem would help.")
    left_out = [i + 1 for i, v in enumerate(verdicts) if v == NOT_FUNGUS]
    if left_out:
        which = (f"Photo {left_out[0]} doesn't" if len(left_out) == 1 else
                 "Photos " + ", ".join(map(str, left_out[:-1])) + f" and {left_out[-1]} don't")
        hints.append(f"{which} look like a fungus, so {'it was' if len(left_out) == 1 else 'they were'} "
                     "left out of the answer.")
    return hints


def prompts(classes=CLASSES) -> tuple[list[str], list[str], list[str]]:
    """(class per prompt, side per prompt, prompt text). A taxonomic string gets only
    "a photo of {}." ("a photo of a Fungi ..." reads wrong)."""
    names, sides, texts = [], [], []
    for name, side, ts in classes:
        for t in ts:
            for tpl in TEMPLATES:
                if tpl == TEMPLATES[1] and t[0].isupper():
                    continue
                names.append(name)
                sides.append(side)
                texts.append(tpl.format(t))
    return names, sides, texts


@dataclass
class Thresholds:
    """p(fungus) below `not_fungus` -> not a fungus; below `unsure` -> not sure."""
    not_fungus: float
    unsure: float

    def verdict(self, p: float) -> str:
        if p < self.not_fungus:
            return NOT_FUNGUS
        if p < self.unsure:
            return UNSURE
        return FUNGUS


@dataclass
class FungusGate:
    classes: np.ndarray           # class of each prompt
    sides: np.ndarray             # "fungus" | "other" per prompt
    vectors: np.ndarray           # (prompts, dim), unit length
    scale: float                  # the CLIP logit scale (100 for BioCLIP 2)
    thresholds: Thresholds
    base: str = "bioclip-2"       # whose frozen image vectors the text vectors match
    provenance: dict = field(default_factory=dict)
    # A linear fungus-vs-not probe on the same frozen vectors (fit_logistic), optional.
    probe_w: np.ndarray | None = None
    probe_b: float = 0.0
    # Which scores make p(fungus): "zero-shot", "probe", or "both" (their log-odds added).
    mode: str = "zero-shot"
    # Zero-shot classes in scope although not fungi (Steve, 2026-10-11: slime moulds): a photo
    # judged "not a fungus" whose probability for such a class is at least the value given is
    # lifted to "not sure" (the names, no chance). A verdict is never lowered by this.
    lift_to_unsure: dict = field(default_factory=dict)
    # Zero-shot classes named to the user (LOOKS_LIKE): a photo not judged a fungus whose
    # probability for the class is at least the value given "looks like" it (slime: 0.5).
    looks_like: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"unknown gate mode {self.mode!r}")
        if self.mode != "zero-shot" and self.probe_w is None:
            raise ValueError(f"gate mode {self.mode!r} needs a probe")
        for name, given in (("lift_to_unsure", self.lift_to_unsure), ("looks_like", self.looks_like)):
            unknown = sorted(set(given) - set(np.asarray(self.classes).tolist()))
            if unknown:
                raise ValueError(f"{name} names classes the gate doesn't have: {unknown}")
        unworded = sorted(set(self.looks_like) - set(LOOKS_LIKE))
        if unworded:
            raise ValueError(f"looks_like has no wording for {unworded}")
        self.classes = np.asarray(self.classes)
        self.sides = np.asarray(self.sides)
        self._class_names = sorted(set(self.classes.tolist()))
        self._fungus_class = np.array([self.sides[self.classes == c][0] == "fungus"
                                       for c in self._class_names])

    def p_fungus(self, base_vectors: np.ndarray) -> np.ndarray:
        """p(fungus) per photo from its frozen base vector (any scale): the zero-shot
        chance, the probe's, or the two combined (their log-odds added), by `mode`."""
        v = np.asarray(base_vectors, dtype=np.float32)
        v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        z = np.zeros(len(v))
        if self.mode in ("zero-shot", "both"):
            p = np.clip(self.zero_shot(v), 1e-9, 1 - 1e-9)
            z += np.log(p / (1 - p))
        if self.mode in ("probe", "both"):
            z += v @ np.asarray(self.probe_w, np.float32) + self.probe_b
        return 1 / (1 + np.exp(-z))

    def class_probs(self, v: np.ndarray) -> np.ndarray:
        """BioCLIP 2 zero-shot probability of every class (columns: sorted class names), for
        unit-length base vectors."""
        logits = (v @ self.vectors.T.astype(np.float32)) * self.scale
        top = logits.max(axis=1, keepdims=True)
        # A class's score: the log of its prompts' mean likelihood.
        per_class = np.stack([np.log(np.exp(logits[:, self.classes == c] - top).mean(axis=1))
                              for c in self._class_names], axis=1) + top
        p = np.exp(per_class - per_class.max(axis=1, keepdims=True))
        return p / p.sum(axis=1, keepdims=True)

    def zero_shot(self, v: np.ndarray) -> np.ndarray:
        """BioCLIP 2 zero-shot p(fungus) for unit-length base vectors."""
        return self.class_probs(v)[:, self._fungus_class].sum(axis=1)

    def _first_at_or_above(self, base_vectors: np.ndarray, rule: dict) -> list[str | None]:
        """Per photo, the first class of `rule` ({class: threshold}) the photo reaches, or None."""
        if not rule:
            return [None] * len(base_vectors)
        v = np.asarray(base_vectors, dtype=np.float32)
        v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        out: list[str | None] = []
        for row in self.class_probs(v):
            hit = [c for c, t in rule.items() if row[self._class_names.index(c)] >= t]
            out.append(hit[0] if hit else None)
        return out

    def lifted(self, base_vectors: np.ndarray) -> list[str | None]:
        """Per photo, the in-scope class that lifts a "not a fungus" verdict, or None."""
        return self._first_at_or_above(base_vectors, self.lift_to_unsure)

    def judge(self, base_vectors: np.ndarray) -> list[dict]:
        """One verdict per photo: {"verdict", "p_fungus", "message"}, plus "lifted_by" (the
        in-scope class, e.g. "slime") when a "not a fungus" verdict was lifted to "not sure"."""
        out = []
        looks = self._first_at_or_above(base_vectors, self.looks_like)
        for p, by, like in zip(self.p_fungus(base_vectors).tolist(), self.lifted(base_vectors),
                               looks):
            v = self.thresholds.verdict(p)
            entry = {"verdict": v, "p_fungus": round(float(p), 4)}
            if v == NOT_FUNGUS and by is not None:
                v = entry["verdict"] = UNSURE
                entry["lifted_by"] = by
            entry["message"] = MESSAGES.get(v)
            if v != FUNGUS and like is not None:       # a fungus is never relabelled
                entry["looks_like"] = like
                entry["message"] = LOOKS_LIKE[like]["message"]
            out.append(entry)
        return out

    def fingerprint(self) -> str:
        """A short hash of what decides the verdicts (mode, base, thresholds, probe, text
        vectors): a stored verdict made under another fingerprint should be judged again."""
        import hashlib
        h = hashlib.sha256(f"{self.mode}|{self.base}|{self.thresholds.not_fungus}|"
                           f"{self.thresholds.unsure}|{self.scale}|{self.probe_b}|"
                           f"{sorted(self.lift_to_unsure.items())}|"
                           f"{sorted(self.looks_like.items())}".encode())
        if self.probe_w is not None:
            h.update(np.asarray(self.probe_w, np.float32).tobytes())
        h.update(np.asarray(self.vectors, np.float32).tobytes())
        return h.hexdigest()[:16]

    def summary(self) -> dict:
        """What /api/health says about the gate."""
        return {"mode": self.mode, "base": self.base, "fingerprint": self.fingerprint(),
                "thresholds": {"not_fungus": self.thresholds.not_fungus,
                               "unsure": self.thresholds.unsure},
                "lift_to_unsure": dict(self.lift_to_unsure),
                "looks_like": dict(self.looks_like)}

    # --- files -----------------------------------------------------------------------

    def save(self, root: Path | None = None) -> Path:
        root = root or config.DATA_DIR / "models"
        root.mkdir(parents=True, exist_ok=True)
        extra = ({} if self.probe_w is None else
                 {"probe_w": np.asarray(self.probe_w, np.float32)})
        np.savez(root / f"{GATE_NAME}.npz", classes=self.classes, sides=self.sides,
                 vectors=self.vectors.astype(np.float32), **extra)
        meta = {"base": self.base, "scale": self.scale, "mode": self.mode,
                "probe_b": self.probe_b, "lift_to_unsure": dict(self.lift_to_unsure),
                "looks_like": dict(self.looks_like),
                "thresholds": {"not_fungus": self.thresholds.not_fungus,
                               "unsure": self.thresholds.unsure},
                "provenance": self.provenance}
        path = root / f"{GATE_NAME}.json"
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, root: Path | None = None) -> "FungusGate | None":
        """The saved gate, or None when there is none (the identifier then runs ungated).
        MV_FUNGUS_GATE_DIR points at another folder (a candidate gate, for a preview)."""
        root = root or Path(config.setting("MV_FUNGUS_GATE_DIR") or config.DATA_DIR / "models")
        meta_path, vec_path = root / f"{GATE_NAME}.json", root / f"{GATE_NAME}.npz"
        if not (meta_path.is_file() and vec_path.is_file()):
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        with np.load(vec_path) as f:
            return cls(f["classes"], f["sides"], f["vectors"], float(meta["scale"]),
                       Thresholds(**meta["thresholds"]), meta.get("base", "bioclip-2"),
                       meta.get("provenance", {}),
                       f["probe_w"] if "probe_w" in f.files else None,
                       float(meta.get("probe_b", 0.0)), meta.get("mode", "zero-shot"),
                       dict(meta.get("lift_to_unsure") or {}),
                       dict(meta.get("looks_like") or {}))


# --- choosing thresholds (on development sets only) ------------------------------------

def auroc(fungus: np.ndarray, other: np.ndarray) -> float:
    """Chance a random fungus photo scores above a random non-fungus one (ties half)."""
    f, o = np.asarray(fungus, float), np.asarray(other, float)
    if not len(f) or not len(o):
        return float("nan")
    allv = np.concatenate([f, o])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sorted_v = allv[order]
    i = 0
    while i < len(allv):                        # average ranks over ties
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(f)].sum() - len(f) * (len(f) + 1) / 2) / (len(f) * len(o)))


def threshold_for_pass_rate(fungus: np.ndarray, pass_rate: float) -> float:
    """The largest threshold that still lets at least `pass_rate` of fungus photos
    through (a photo passes when its score >= the threshold)."""
    f = np.sort(np.asarray(fungus, float))
    k = int(np.floor((1 - pass_rate) * len(f) + 1e-9))   # photos allowed below
    return float(f[min(k, len(f) - 1)])


def rates_at(threshold: float, fungus: np.ndarray, other: np.ndarray) -> dict:
    """Fungus photos passed (TPR) and non-fungus photos passed (FPR) at a threshold."""
    f, o = np.asarray(fungus, float), np.asarray(other, float)
    return {"threshold": float(threshold),
            "tpr": float((f >= threshold).mean()) if len(f) else float("nan"),
            "fpr": float((o >= threshold).mean()) if len(o) else float("nan"),
            "fungus_rejected": int((f < threshold).sum()),
            "other_passed": int((o >= threshold).sum())}


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 25
                 ) -> tuple[np.ndarray, float]:
    """A class-balanced logistic regression with an L2 penalty (bias unpenalised), by
    Newton's method: (weights, bias). Rows of X are unit-length base vectors; y is 1 for
    a fungus photo, 0 for not. numpy only (torch's autograd crashed under load here)."""
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    n, d = X.shape
    A = np.hstack([X, np.ones((n, 1))])
    sw = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean())) / n
    reg = np.full(d + 1, 2 * l2)
    reg[-1] = 0.0
    theta = np.zeros(d + 1)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(A @ theta)))
        grad = A.T @ (sw * (p - y)) + reg * theta
        hess = (A * (sw * p * (1 - p))[:, None]).T @ A + np.diag(reg)
        step = np.linalg.solve(hess, grad)
        theta -= step
        if np.abs(step).max() < 1e-6:
            break
    return theta[:-1].astype(np.float32), float(theta[-1])


def build(thresholds: Thresholds, base_spec: str = "hf-hub:imageomics/bioclip-2",
          base: str = "bioclip-2", provenance: dict | None = None,
          probe: tuple[np.ndarray, float] | None = None, mode: str = "zero-shot") -> FungusGate:
    """Text vectors for every prompt, from the base model's text tower (CPU is fine:
    a few seconds). Needs open_clip and the base model's weights."""
    import open_clip
    import torch
    model, _, _ = open_clip.create_model_and_transforms(base_spec)
    tokenizer = open_clip.get_tokenizer(base_spec)
    classes, sides, texts = prompts()
    with torch.no_grad():
        v = model.encode_text(tokenizer(texts)).float().numpy()
        scale = float(model.logit_scale.exp())
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return FungusGate(np.array(classes), np.array(sides), v, scale, thresholds, base,
                      provenance or {}, probe[0] if probe else None,
                      float(probe[1]) if probe else 0.0, mode)
