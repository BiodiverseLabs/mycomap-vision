"""Evaluating on a dataset release: which records are scored, against which reference, and
the Picek vs Vision comparison on them (`mv compare --release <id> --eval val|test|sealed`).

A release fixes its own time split (docs/dataset-release.md §5; cutoffs are read from the
release, never set here): tune on `val`, report on `test`; `sealed` (the paper's fresh set)
only once the release has one. The reference a model answers from is every split before
the scored one: `train` for `val`, `train` + `val` for `test` and `sealed`.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import dataset

REFERENCE_FOR = {"val": ("train",), "test": ("train", "val"), "sealed": ("train", "val")}


def check_eval_split(rel, split: str, allow_test: bool = False) -> None:
    """`val` by default; `test` only when asked for explicitly; `sealed` only when the
    release has it."""
    if split not in dataset.EVAL_SPLITS:
        raise ValueError(f"--eval is one of {', '.join(dataset.EVAL_SPLITS)}")
    if split == "test" and not allow_test:
        raise ValueError("the test split is reported once per experiment: pass --allow-test "
                         "to score it (tune on val)")
    if split not in rel.splits():
        raise ValueError(f"release {rel.id} has no {split} split"
                         + (" (the paper's sealed set isn't in it yet)" if split == "sealed"
                            else ""))


def to_record(row: dict, photo_ids: list[int], place: dict):
    """A release record as evaluate.Record (photo_rows hold manifest photo ids)."""
    from ...evaluate import Record
    lat, lon = place.get(str(row["record_key"]), (None, None))
    species = row.get("species") or ""
    try:
        projects = tuple(json.loads(row.get("green_projects") or "[]"))
    except ValueError:
        projects = ()
    return Record(observation_id=str(row["record_key"]), species=species,
                  genus=row.get("genus") or "", family=row.get("family") or "",
                  validated_on=row.get("validated_on"), observer=None,
                  photo_rows=list(photo_ids), latitude=lat, longitude=lon,
                  observed_on=row.get("observed_on"), projects=projects,
                  taxon="" if species else (row.get("label") or ""))


def records_of(conn, rel, splits, keep_photo=None) -> list:
    """The release's records of `splits` with their photos (manifest ids; only those
    `keep_photo` accepts), records left without a photo dropped."""
    rows = [r for s in splits for r in rel.records(s)]
    photos = rel.photos(r["record_key"] for r in rows)
    place = {str(o): (lat, lon) for o, lat, lon in conn.execute(
        "select observation_id, latitude, longitude from records")}
    out = []
    for r in rows:
        ids = [dataset.photo_id(p["photo_key"]) for p in photos[r["record_key"]]]
        ids = [i for i in ids if keep_photo is None or keep_photo(i)]
        if ids:
            out.append(to_record(r, ids, place))
    return out


def model_release(name: str) -> dict | None:
    """The dataset release a registered model was trained on (its JSON), or None."""
    from ...finetune import models_dir
    path = Path(models_dir()) / f"{name}.json"
    if not path.is_file():
        return None
    return (json.loads(path.read_text(encoding="utf-8")).get("dataset_release") or None)


def shared_records(conn, rel, backbones: list[str], split: str, embeddings_root=None):
    """evaluate.SharedSet for a release: the scored split's records against the reference
    splits before it, on the photos every backbone embedded. A model trained on another
    release is refused; the cutoff (the newest reference validation) keeps
    evaluate.check_not_trained_on_test's rule: a model that saw the scored split is refused."""
    from ...embed import load_embeddings
    from ...evaluate import SharedSet, record_set_hash
    for b in backbones:
        trained = model_release(b)
        if trained and trained.get("dataset_release") != rel.id:
            raise ValueError(f"{b} was trained on release {trained.get('dataset_release')}, "
                             f"not {rel.id}: compare models on the release they trained on")
    loaded = {}
    for b in backbones:
        root = embeddings_root / b if embeddings_root else None
        ids, vecs = load_embeddings(conn, b, root)
        if not len(ids):
            raise ValueError(f"no embeddings for {b!r}; run mv embed --backbone {b}")
        loaded[b] = (ids, vecs)
    common = set.intersection(*(set(int(i) for i in ids.tolist()) for ids, _ in loaded.values()))
    ref = records_of(conn, rel, REFERENCE_FOR[split], common.__contains__)
    test = records_of(conn, rel, (split,), common.__contains__)
    if not ref or not test:
        raise ValueError(f"release {rel.id}: no {split} or reference records with photos "
                         f"every backbone embedded")
    cutoff = max((r.validated_on for r in ref if r.validated_on), default="")
    record_set = record_set_hash(ref, test)
    return SharedSet(loaded, ref, test, cutoff, record_set, len(common))


def compare(conn, rel, backbones: list[str], methods: list[str], split: str = "val",
            allow_test: bool = False, embeddings_root=None, **kw) -> dict:
    """mv compare on a release's split: the same evaluate.compare (every method, the
    standard summary, the scoreboard), its rows citing the release and the split."""
    from ... import evaluate
    check_eval_split(rel, split, allow_test)
    shared = shared_records(conn, rel, backbones, split, embeddings_root)
    out = evaluate.compare(conn, backbones, methods, test_days=0, shared=shared,
                           extra={"dataset_release": rel.cite(), "eval_split": split,
                                  "reference_splits": list(REFERENCE_FOR[split])}, **kw)
    out.update({"dataset_release": rel.cite(), "eval_split": split,
                "reference_splits": list(REFERENCE_FOR[split])})
    return out
