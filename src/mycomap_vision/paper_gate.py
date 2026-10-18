"""The not-a-fungus gate in the paper's scoring (paper E10; PLAN section 7).

The config's "gate" block names the tuned gate (its folder and fingerprint), where the
gate's vectors for the scored photos come from, and, optionally, the sealed OOD set (its
vectors and manifest). The scored photos' vectors are either a file made beforehand
(`test_vectors`, `mv fungus-gate embed`: keys are the release's manifest photo ids), or
made by the run itself through the served model's fork (`embed_with`: the backbone), so a
sealed test is never embedded before the run that scores it; the vectors are then kept
beside the report and their sha256 recorded. Every file is pinned by sha256; the gate by
its fingerprint (FungusGate.fingerprint).

- Each scored find is judged as the server judges it: every photo by FungusGate.judge,
  the find by observation_verdict (a fungus when any photo is one). A find judged "not a
  fungus" loses its answer. The claim stands when at most `max_lost_share` of finds do.
- Served numbers: each Vision comparator gets a "<id> served" twin in which a refused
  find is wrong in every cell; the plain row is the gate-off number.
- The OOD set: each photo is its own find. The headline let-through (judged a fungus) and
  TNR at 95% TPR are on everyday + non-fungal near-OOD; everyday alone is checked against
  `everyday_let_through`; lichens and fungal pathogens get their own rows (named / not
  sure / refused), split by whether the release trains on the taxon (in_v1_labels).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import fungusgate
from . import paper_outcomes as po
from . import paper_stats as ps
from .fungusgate_tune import sha256_file

HEADLINE = ("everyday", "near-ood")        # manifest "kind": the non-fungal OOD photos
IN_SCOPE = ("fungal-pathogen", "near-ood-fungus")
SEALED_BY = "the tag paper-analysis-freeze-v1"
VERDICT_WORDS = {fungusgate.FUNGUS: "named", fungusgate.UNSURE: "not_sure",
                 fungusgate.NOT_FUNGUS: "refused"}


def pinned(spec: dict, resolve) -> Path:
    """A {file, sha256} entry of the block: the file, refused when its sha256 differs."""
    path = resolve(spec["file"])
    got = sha256_file(path)
    if got != spec.get("sha256"):
        raise ValueError(f"gate input {path}: sha256 {got[:12]}, the config pins "
                         f"{str(spec.get('sha256'))[:12]}")
    return path


def load_gate(block: dict, resolve) -> fungusgate.FungusGate:
    gate = fungusgate.FungusGate.load(resolve(block["dir"]))
    if gate is None:
        raise ValueError(f"no gate in {block['dir']}")
    if gate.fingerprint() != block.get("fingerprint"):
        raise ValueError(f"gate {block['dir']}: fingerprint {gate.fingerprint()}, the config "
                         f"pins {block.get('fingerprint')}")
    return gate


def read_vectors(path: Path) -> dict[str, np.ndarray]:
    """key -> frozen base vector, from a `mv fungus-gate embed` file."""
    with np.load(path) as f:
        return dict(zip((str(k) for k in f["keys"]), f["frozen"]))


def judge_finds(gate, photos: dict[str, list[dict]], vectors: dict[str, np.ndarray]
                ) -> dict[str, dict]:
    """Each find's verdict and best p(fungus) from its photos' vectors. A scored photo with
    no vector is refused: the gate's numbers would otherwise hide finds."""
    missing = [str(p["manifest_photo_id"]) for ph in photos.values() for p in ph
               if str(p["manifest_photo_id"]) not in vectors]
    if missing:
        raise ValueError(f"{len(missing):,} scored photos have no gate vector (e.g. "
                         f"{missing[:3]}): embed them with `mv fungus-gate embed`")
    out = {}
    for key, ph in photos.items():
        if not ph:
            continue
        v = np.stack([vectors[str(p["manifest_photo_id"])] for p in ph]).astype(np.float32)
        judged = gate.judge(v)
        out[key] = {"verdict": fungusgate.observation_verdict([j["verdict"] for j in judged]),
                    "p_best": max(j["p_fungus"] for j in judged)}
    return out


def served(outcome: dict, verdicts: dict[str, dict]) -> dict:
    """The served twin of a comparator's outcomes: a find the gate refuses is wrong in
    every cell it could be scored on (and unanswered); every other find as it was."""
    out = {}
    for key, o in outcome.items():
        v = verdicts.get(key)
        if v is not None and v["verdict"] == fungusgate.NOT_FUNGUS:
            out[key] = {**o, "answered": False,
                        **{c: (None if o.get(c) is None else False) for c in po.CELLS}}
        else:
            out[key] = o
    return out


def _share(flags: list[bool], clusters: list, reps: int, seed: int) -> dict:
    n, k = len(flags), int(sum(flags))
    return {"n": n, "count": k, "share": k / n if n else None,
            "ci95": ps.cluster_ci(flags, clusters, reps, seed) if n else None}


def test_report(finds, verdicts: dict[str, dict], reps: int, seed: int,
                max_lost_share: float) -> dict:
    """The scored finds through the gate: refused (lost) and not-sure shares with the
    observer-clustered interval, and the claim check on the point estimate."""
    judged = [f for f in finds if f.key in verdicts]
    who = [f.observer for f in judged]
    lost = _share([verdicts[f.key]["verdict"] == fungusgate.NOT_FUNGUS for f in judged],
                  who, reps, seed)
    unsure = _share([verdicts[f.key]["verdict"] == fungusgate.UNSURE for f in judged],
                    who, reps, seed)
    return {"finds": len(judged), "lost": lost, "not_sure": unsure,
            "claim": {"max_lost_share": max_lost_share,
                      "stands": lost["share"] is not None and lost["share"] <= max_lost_share}}


def read_manifest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def ood_report(gate, rows: list[dict], vectors: dict[str, np.ndarray],
               fungus_p: np.ndarray, reps: int, seed: int, everyday_let_through: float) -> dict:
    """The sealed OOD set through the gate (one photo = one find, clustered by owner).
    `fungus_p`: the scored finds' best p(fungus), for AUROC and the threshold at 95% TPR."""
    missing = [r["file"] for r in rows if r["file"] not in vectors]
    if missing:
        raise ValueError(f"{len(missing):,} OOD photos have no gate vector (e.g. {missing[:3]})")
    v = np.stack([vectors[r["file"]] for r in rows]).astype(np.float32)
    judged = gate.judge(v)
    p = np.array([j["p_fungus"] for j in judged])
    verdict = [j["verdict"] for j in judged]
    owner = [r.get("owner") or r["file"] for r in rows]
    t95 = fungusgate.threshold_for_pass_rate(fungus_p, 0.95) if len(fungus_p) else None

    def row(ix: list[int]) -> dict:
        who = [owner[i] for i in ix]
        out = {w: _share([verdict[i] == k for i in ix], who, reps, seed)
               for k, w in VERDICT_WORDS.items()}
        out["photos"] = len(ix)
        if t95 is not None and ix:
            out["tnr_at_95_tpr"] = float((p[ix] < t95).mean())
        if len(fungus_p) and ix:
            out["auroc"] = fungusgate.auroc(fungus_p, p[ix])
        return out

    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["kind"]].append(i)
    headline = [i for k in HEADLINE for i in by.get(k, [])]
    everyday = by.get("everyday", [])
    report = {"photos": len(rows), "threshold_at_95_tpr": t95,
              "headline": {"kinds": list(HEADLINE), **row(headline)},
              "everyday": row(everyday), "near_ood_nonfungal": row(by.get("near-ood", [])),
              "in_scope": {}, "by_category": {}}
    let = report["everyday"]["named"]["share"]
    report["everyday"]["check"] = {"let_through_at_most": everyday_let_through,
                                   "holds": let is not None and let <= everyday_let_through}
    for kind in IN_SCOPE:
        for trains in ("yes", "no"):
            ix = [i for i in by.get(kind, []) if rows[i].get("in_v1_labels") == trains]
            if ix:
                report["in_scope"][f"{kind} (trained on: {trains})"] = row(ix)
    cats = defaultdict(list)
    for i, r in enumerate(rows):
        cats[r["category"]].append(i)
    report["by_category"] = {c: row(ix) for c, ix in sorted(cats.items())}
    return report


def photo_list(conn, photos: dict[str, list[dict]], path: Path) -> Path:
    """key<TAB>path of every scored photo's local large copy (never S3), for
    `fungusgate_tune.embed_list`; a photo with no local copy is refused."""
    from .backbone_screen import photos_to_embed
    ids = [int(p["manifest_photo_id"]) for ph in photos.values() for p in ph]
    todo = photos_to_embed(conn, "(gate list: every photo)", ids)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for where, items in todo.items():
            for pid, rel_path in items:
                fh.write(f"{pid}\t{(Path(where) / rel_path).as_posix()}\n")
    return path


def gate_vectors(block: dict, photos, resolve, out: Path | None, conn=None,
                 embed_list=None) -> Path:
    """The scored photos' gate vectors: the pinned `test_vectors` file, or made now through
    `embed_with`'s fork into `out` (embed_list: fungusgate_tune.embed_list)."""
    if block.get("test_vectors"):
        return pinned(block["test_vectors"], resolve)
    if out is None or conn is None:
        raise ValueError("gate: embed_with needs the run's output folder and manifest")
    if embed_list is None:
        from .fungusgate_tune import embed_list
    out.mkdir(parents=True, exist_ok=True)
    lst = photo_list(conn, photos, out / "gate-vectors.list")
    npz = out / "gate-vectors.npz"
    embed_list(block["embed_with"], lst, npz)
    return npz


def run_gate(block: dict, finds, photos, outcomes: dict[str, dict], vision_ids: list[str],
             resolve, reps: int, seed: int, out: Path | None = None, conn=None,
             embed_list=None, sealed_open: bool = False) -> tuple[dict, dict[str, dict]]:
    """(report, served outcomes to add) for the config's gate block. The OOD set is scored
    only when `sealed_open` (the freeze guard opened the sealed test for this run)."""
    gate = load_gate(block, resolve)
    tv = gate_vectors(block, photos, resolve, out, conn, embed_list)
    verdicts = judge_finds(gate, photos, read_vectors(tv))
    claims = block.get("claims") or {}
    report = {"gate": {**gate.summary(), "dir": str(block["dir"])},
              "test_vectors": {"file": str(tv), "sha256": sha256_file(tv),
                               "made_by": "pinned" if block.get("test_vectors")
                               else f"embedded by this run through {block['embed_with']}"},
              "test": test_report(finds, verdicts, reps, seed,
                                  float(claims.get("max_lost_share", 0.005)))}
    if block.get("ood") and not sealed_open:
        # The OOD set is sealed like the test: no verdict, rate or even a read of its
        # vectors until the run that opens the sealed test (the freeze guard passed).
        report["ood"] = {"not_scored": f"sealed until the sealed test run ({SEALED_BY})"}
    elif block.get("ood"):
        ov = pinned(block["ood"]["vectors"], resolve)
        om = pinned(block["ood"]["manifest"], resolve)
        report["ood"] = {"vectors_sha256": sha256_file(ov), "manifest_sha256": sha256_file(om),
                         **ood_report(gate, read_manifest(om), read_vectors(ov),
                                      np.array([v["p_best"] for v in verdicts.values()]),
                                      reps, seed,
                                      float(claims.get("everyday_let_through", 0.02)))}
    twins = {f"{cid} served": served(outcomes[cid], verdicts) for cid in vision_ids
             if cid in outcomes and outcomes[cid]}
    report["served_comparators"] = sorted(twins)
    return report, twins
