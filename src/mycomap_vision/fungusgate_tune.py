"""Re-runnable tuning of the not-a-fungus gate (fungusgate.py), on development data only.

    mv fungus-gate embed --list photos.tsv --out vectors.npz
        The base model's (frozen) vectors for a list of photo files (key<TAB>path), from
        the same one-pass fork the server uses (FinetunedBackbone.encode_with_base).
    mv fungus-gate tune --dev dev.npz --ood ood.npz --ood-manifest ood.csv
                        --exclude mislinked.tsv [--negatives neg.npz] --out DIR
        Fits the linear probe, scores every mode on the held-out DEV split and the
        not-a-fungus set, picks the thresholds by the rule below on the declared grid,
        and writes DIR/fungus-gate.{json,npz} (the gate) and DIR/report.json.

The probe's training data never overlaps the evaluation sets: positives are photos of a
random 20,000 genuine (iNat-linked) reference records; negatives are photos of reference
records that are not iNat records at all (ids of other sources fetched as iNat ids, so
their photos are random iNat observations), taken only where iNat files them under
Animalia or Plantae, plus optional extra negatives (Open Images test split). Evaluation
uses the held-out DEV split (never in the references) and a separate not-a-fungus set
(CC iNat photos + Open Images validation split). A test split is never read.

The threshold rule, declared before looking (fixed in code): on the grid 0.05, 0.10, ...
0.95 of p(fungus), "not a fungus" is the largest value at which NO DEV record loses its
answer (every photo below it), and "not sure" the largest value at which at most 0.5% of
DEV records are shown without a chance. Exploratory until the dataset release v1 freeze:
confirm there.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config, fungusgate

GRID = tuple(round(0.05 * i, 2) for i in range(1, 20))
MAX_UNSURE_SHARE = 0.005
# Slime moulds are in scope (Steve, 2026-10-11): "not sure", never "not a fungus". Declared
# before looking: the zero-shot slime-mould class lifts a "not a fungus" photo to "not sure"
# at the lowest grid value at which at most 1% of everyday non-fungi are lifted.
LIFT_CLASS = "slime"
MAX_EVERYDAY_LIFTED = 0.01
POSITIVE_RECORDS = 20_000
SEED = 20261009
FUNGI, ANIMALIA, PLANTAE = 47170, 1, 47126
EVERYDAY = ("animal", "plant", "person", "object", "scene", "food")


@dataclass
class Choice:
    not_fungus: float | None
    unsure: float | None
    records: int
    no_answer: int
    unsure_records: int
    photo_left_out: int


def choose_thresholds(record_scores: list[np.ndarray], grid=GRID,
                      max_unsure_share: float = MAX_UNSURE_SHARE) -> Choice:
    """The declared rule on p(fungus) per photo, grouped by DEV record. None when no grid
    value keeps every DEV record's answer."""
    best = np.array([s.max() for s in record_scores])
    worst = np.array([s.min() for s in record_scores])
    n = len(best)
    a = max((t for t in grid if (best < t).sum() == 0), default=None)
    if a is None:
        return Choice(None, None, n, 0, 0, 0)
    b = max(t for t in grid if t >= a and ((best >= a) & (best < t)).sum() <= max_unsure_share * n)
    return Choice(a, b, n, int((best < a).sum()), int(((best >= a) & (best < b)).sum()),
                  int(((worst < a) & (best >= a)).sum()))


def choose_lift(p_class_everyday: np.ndarray, judged_not_fungus: np.ndarray, grid=GRID,
                max_lifted: float = MAX_EVERYDAY_LIFTED) -> float | None:
    """The lowest grid value of a class's probability at which at most `max_lifted` of the
    everyday non-fungi would move from "not a fungus" to "not sure"; None if none does."""
    p = np.asarray(p_class_everyday, float)
    nf = np.asarray(judged_not_fungus, bool)
    return min((t for t in grid if (nf & (p >= t)).mean() <= max_lifted), default=None)


def kingdom(ancestor_ids) -> str:
    """'fungi', 'animal', 'plant' or 'other' from an iNat taxon's ancestor ids."""
    if isinstance(ancestor_ids, str):
        ancestor_ids = json.loads(ancestor_ids) if ancestor_ids.strip().startswith("[") else \
            [int(x) for x in ancestor_ids.replace("/", ",").split(",") if x.strip()]
    ids = set(int(x) for x in (ancestor_ids or []))
    for k, name in ((FUNGI, "fungi"), (ANIMALIA, "animal"), (PLANTAE, "plant")):
        if k in ids:
            return name
    return "other"


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def photo_manifest(rows: list[dict], folder: Path, out: Path) -> int:
    """The not-a-fungus set's manifest: each photo's source ids, licence, owner, url and
    sha256, so the set can be rebuilt and checked by anyone."""
    fields = ["file", "group", "category", "source", "observation_id", "photo_id", "taxon_id",
              "taxon_name", "owner", "license", "url", "sha256"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for r in rows:
            w.writerow({**{k: r.get(k, "") for k in fields},
                        "sha256": sha256_file(folder / r["file"])})
    return len(rows)


# --- embedding through the serving path --------------------------------------------------

def embed_list(backbone_name: str, list_path: Path, out: Path, batch: int = 32, log=print) -> int:
    """key<TAB>path lines -> out.npz with keys, frozen (base) and ft (served) vectors."""
    from PIL import Image

    from .models import load_backbone
    bb = load_backbone(backbone_name)
    items = [l.rstrip("\n").split("\t") for l in open(list_path, encoding="utf-8") if l.strip()]
    keys, F, T = [], [], []
    for i in range(0, len(items), batch):
        chunk, ims = [], []
        for k, p in items[i:i + batch]:
            try:
                im = Image.open(p)
                if im.format == "JPEG":
                    im.draft("RGB", (448, 448))     # libjpeg scales while decoding, >= 448 px
                ims.append(bb.prepare(im.convert("RGB")))
                chunk.append(k)
            except Exception as e:                  # an unreadable photo is skipped, said
                log(f"skip {k}: {e}")
        if not ims:
            continue
        pair = bb.encode_with_base(ims) if hasattr(bb, "encode_with_base") else None
        if pair is None:
            raise ValueError(f"{backbone_name} gives no base vectors (not a forkable fine-tune)")
        T.append(pair[0].astype(np.float16))
        F.append(pair[1].astype(np.float16))
        keys += chunk
        log(f"{len(keys)}/{len(items)}")
    np.savez(out, keys=np.array(keys), frozen=np.concatenate(F), ft=np.concatenate(T))
    return len(keys)


# --- tuning --------------------------------------------------------------------------------

def probe_training_set(conn, exclude: set[str], embeddings_root: Path,
                       extra_negatives: np.ndarray | None = None, base: str = "bioclip-2"):
    """(X, y, provenance) for the probe; see the module docstring."""
    from .embed import load_embeddings
    ids, vecs = load_embeddings(conn, base, embeddings_root)
    row = {int(p): i for i, p in enumerate(ids.tolist())}
    photos = defaultdict(list)
    for pid, o in conn.execute("select photo_id, observation_id from observation_photos"):
        if int(pid) in row:
            photos[str(o)].append(row[int(pid)])
    kind = {str(o): kingdom(a) for o, a in conn.execute(
        "select observation_id, taxon_ancestor_ids from inat_observations")}
    neg_obs = sorted(o for o in exclude if kind.get(o) in ("animal", "plant") and photos.get(o))
    good = sorted(o for o in photos if o not in exclude)
    rng = np.random.default_rng(SEED)
    pos_obs = [good[i] for i in rng.permutation(len(good))[:POSITIVE_RECORDS]]
    pos = [i for o in pos_obs for i in photos[o]]
    neg = [i for o in neg_obs for i in photos[o]]
    extra = extra_negatives if extra_negatives is not None else np.zeros((0, vecs.shape[1]))
    X = unit(np.concatenate([vecs[pos], vecs[neg], extra]))
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg) + len(extra))]
    h = hashlib.sha256()
    for part in (sorted(pos_obs), neg_obs):
        h.update("\x1f".join(part).encode())
        h.update(b"\x1e")
    prov = {"positive_records": len(pos_obs), "positive_photos": len(pos),
            "negative_records": len(neg_obs), "negative_photos": len(neg),
            "extra_negatives": int(len(extra)), "reference_photos": int(len(ids)),
            "training_records_sha256": h.hexdigest(), "seed": SEED}
    return X, y, prov


def dev_records(conn, benchmark: str, keys: list[str]) -> list[str]:
    """The DEV record of each photo key (a photo id of the held-out benchmark)."""
    obs = {str(p): str(o) for p, o in conn.execute(
        "select hp.photo_id, hp.observation_id from heldout_photos hp join heldout_records r "
        "on r.benchmark = hp.benchmark and r.observation_id = hp.observation_id "
        "where hp.benchmark = ? and r.split = 'dev'", (benchmark,))}
    missing = [k for k in keys if k not in obs]
    if missing:
        raise ValueError(f"{len(missing)} photos are not DEV photos of {benchmark} "
                         "(a test split is never used here)")
    return [obs[k] for k in keys]


def tune(conn, dev_npz: Path, ood_npz: Path, ood_manifest: Path, exclude_tsv: Path,
         out_dir: Path, benchmark: str = "heldout-2026-10-08", negatives_npz: Path | None = None,
         embeddings_root: Path | None = None, gate_dir: Path | None = None,
         served: str | None = None, log=print) -> dict:
    """See the module docstring. With `served` (a backbone name), the lighter gate: the
    probe alone, fitted and scored on that backbone's own (served) vectors, so serving
    needs no second set of weights and no extra pass; the gate's base is that backbone."""
    out_dir.mkdir(parents=True, exist_ok=True)
    exclude = {r["observation_id"] for r in csv.DictReader(open(exclude_tsv, encoding="utf-8"),
                                                          delimiter="\t")}
    field, base = ("ft", served) if served else ("frozen", "bioclip-2")
    extra = unit(np.load(negatives_npz)[field]) if negatives_npz else None
    X, y, prov = probe_training_set(conn, exclude, embeddings_root or config.DATA_DIR /
                                    "embeddings" / base, extra, base=base)
    log(f"probe: {int(y.sum()):,} fungus, {int((1 - y).sum()):,} not-fungus photos")
    w, b = fungusgate.fit_logistic(X, y)
    del X
    text = fungusgate.FungusGate.load(gate_dir) if gate_dir else None
    if text is None:
        text = fungusgate.build(fungusgate.Thresholds(0.0, 0.0))
    dev, ood = np.load(dev_npz), np.load(ood_npz)
    keys = [str(k) for k in dev["keys"]]
    recs = dev_records(conn, benchmark, keys)
    meta = {r["file"]: r for r in csv.DictReader(open(ood_manifest, encoding="utf-8"))}
    cat = np.array([meta[str(k)]["category"] for k in ood["keys"]])
    groups = [meta[str(k)]["group"] for k in ood["keys"]]
    everyday = np.isin(cat, EVERYDAY)
    report = {"benchmark": benchmark, "split": "dev", "reproducibility": "exploratory-pre-freeze",
              "code_version": config.code_version(), "probe": prov, "vectors": base,
              "dev": {"photos": len(keys), "records": len(set(recs))},
              "not_a_fungus": {"photos": int(len(cat)), "everyday": int(everyday.sum()),
                               "manifest_sha256": sha256_file(ood_manifest)},
              "rule": {"grid": list(GRID), "max_unsure_share": MAX_UNSURE_SHARE},
              "created_at": datetime.now(timezone.utc).isoformat(), "modes": {}}
    by_rec = defaultdict(list)
    for i, r in enumerate(recs):
        by_rec[r].append(i)
    chosen = None
    for mode in (("probe",) if served else fungusgate.MODES):
        g = fungusgate.FungusGate(text.classes, text.sides, text.vectors, text.scale,
                                  fungusgate.Thresholds(0, 0), probe_w=w, probe_b=b, mode=mode)
        pd, po = g.p_fungus(dev[field]), g.p_fungus(ood[field])
        c = choose_thresholds([pd[ix] for ix in by_rec.values()])
        entry = {"auroc_everyday": fungusgate.auroc(pd, po[everyday]),
                 "auroc_all": fungusgate.auroc(pd, po), "choice": c.__dict__,
                 "fpr_at_95_tpr_everyday": fungusgate.rates_at(
                     fungusgate.threshold_for_pass_rate(pd, 0.95), pd, po[everyday])["fpr"]}
        if c.not_fungus is not None:
            m = po[everyday]
            entry["everyday"] = {"not_a_fungus": float((m < c.not_fungus).mean()),
                                 "not_sure": float(((m >= c.not_fungus) & (m < c.unsure)).mean()),
                                 "passed": float((m >= c.unsure).mean())}
            entry["by_category"] = {
                k: {"n": int((cat == k).sum()),
                    "not_a_fungus": float((po[cat == k] < c.not_fungus).mean()),
                    "passed": float((po[cat == k] >= c.unsure).mean())}
                for k in sorted(set(cat.tolist()))}
            if chosen is None or entry["everyday"]["passed"] < chosen[1]["everyday"]["passed"]:
                chosen = (mode, entry, c)
        report["modes"][mode] = entry
        log(f"{mode}: AUROC everyday {entry['auroc_everyday']:.4f}, rule {c.not_fungus}/{c.unsure}")
    if chosen is None:
        raise ValueError("no mode keeps every DEV record's answer at any grid value")
    mode, entry, c = chosen
    lift = {}
    if mode in ("zero-shot", "both") and not served:
        g = fungusgate.FungusGate(text.classes, text.sides, text.vectors, text.scale,
                                  fungusgate.Thresholds(c.not_fungus, c.unsure), probe_w=w,
                                  probe_b=b, mode=mode)
        k = g._class_names.index(LIFT_CLASS)
        p_class = g.class_probs(unit(ood["frozen"]))[:, k]
        po = g.p_fungus(ood["frozen"])
        s = choose_lift(p_class[everyday], po[everyday] < c.not_fungus)
        if s is not None:
            lift = {LIFT_CLASS: s}
            g.lift_to_unsure = lift
            ver = [v["verdict"] for v in g.judge(ood["frozen"])]
            report["lift"] = {"class": LIFT_CLASS, "threshold": s,
                              "rule": f"lowest grid value with at most {MAX_EVERYDAY_LIFTED:.0%} "
                                      "of everyday non-fungi lifted",
                              "by_group": {grp: {v: sum(1 for x, gg in zip(ver, groups) if gg == grp
                                                        and x == v) for v in fungusgate.VERDICTS}
                                           for grp in sorted(set(groups))}}
            pd_dev = g.judge(dev["frozen"])
            report["lift"]["dev_photos_lifted"] = sum(1 for v in pd_dev if v.get("lifted_by"))
    gate = fungusgate.FungusGate(text.classes, text.sides, text.vectors, text.scale,
                                 fungusgate.Thresholds(c.not_fungus, c.unsure), base,
                                 {"tuned_on": f"{benchmark} dev + not-a-fungus set",
                                  "reproducibility": "exploratory-pre-freeze",
                                  "code_version": report["code_version"],
                                  "training_records_sha256": prov["training_records_sha256"]},
                                 w, b, mode, lift)
    gate.save(out_dir)
    report["chosen"] = {"mode": mode, "not_fungus": c.not_fungus, "unsure": c.unsure,
                        "why": "fewest everyday non-fungi passed among modes meeting the rule"}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
