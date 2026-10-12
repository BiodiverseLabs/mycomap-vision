"""Which records a dataset release scores, against which reference (docs/dataset-release.md).

A release holds train + val (val = its newest weeks, cutoff fixed inside it). Its test
set is a SEPARATE release, `<id>-test` (fresh records, disjoint by record and photo bytes,
never trainable). So:

- tune on val: `mv compare --release v1` (the builder's: reference = train, scored = val)
  and `mv external run --release v1` (scored = val, reference = train);
- report on test: `mv compare --release v1 --test-release v1-test` (compare_on_test here:
  reference = v1's train + val, scored = v1-test) and `mv external run --release v1-test`.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import dataset


def scored_split(rel, eval_split: str | None = None) -> tuple[str, tuple[str, ...]]:
    """(the split scored, the splits of the base release used as reference): val against
    train for a release, test against its base's train + val for a test release."""
    if dataset.is_test_release(rel):
        split, ref = "test", ("train", "val")
    else:
        split, ref = "val", ("train",)
    if eval_split not in (None, split):
        raise ValueError(f"release {rel.id} is scored on its {split} split"
                         + (f": its test set is a separate release ({rel.id}-test)"
                            if split == "val" else ""))
    return split, ref


def base_release(rel, root: Path | None = None):
    """The release a test release was made for (opened like it: a draft only as a draft)."""
    return dataset.load(rel.manifest["test_for"], allow_draft=bool(rel.manifest.get("draft")),
                        root=root)


def model_release(name: str) -> dict | None:
    """The dataset release a registered model was trained on (its JSON), or None."""
    from ...finetune import models_dir
    path = Path(models_dir()) / f"{name}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("dataset_release") or None


def compare_on_test(conn, rel, test_rel, backbones: list[str], methods: list[str],
                    embeddings_root=None, **kw) -> dict:
    """The Picek vs Vision report: every backbone x method scored on the test release's
    records against the base release's train + val (evaluate.compare with that record
    set), rows citing both releases. A test release made for another release, or a model
    trained on another release, is refused; so is a model that saw the test records
    (evaluate.check_not_trained_on_test, cutoff = the newest reference validation)."""
    from ... import evaluate
    from ...embed import load_embeddings
    if test_rel.manifest.get("test_for") != rel.id:
        raise ValueError(f"{test_rel.id} is not a test release of {rel.id}")
    for b in backbones:
        trained = model_release(b)
        if trained and trained.get("dataset_release") != rel.id:
            raise ValueError(f"{b} was trained on release {trained.get('dataset_release')}, "
                             f"not {rel.id}")
    rel.check_photos_match(conn)
    test_rel.check_photos_match(conn)
    loaded = {}
    for b in backbones:
        root = embeddings_root / b if embeddings_root else None
        ids, vecs = load_embeddings(conn, b, root)
        if not len(ids):
            raise ValueError(f"no embeddings for {b!r}; run mv embed --backbone {b}")
        loaded[b] = (ids, vecs)
    common = set.intersection(*(set(int(i) for i in ids.tolist()) for ids, _ in loaded.values()))
    same = {p: p for p in common}
    ref = rel.as_records(("train", "val"), same)
    test = test_rel.as_records("test", same)
    if not ref or not test:
        raise ValueError("no reference or test records with photos every backbone embedded")
    cutoff = max((r.validated_on for r in ref if r.validated_on), default="")
    shared = evaluate.SharedSet(loaded, ref, test, cutoff,
                                evaluate.record_set_hash(ref, test), len(common))
    cites = {"dataset_release": dataset.cite(rel), "test_release": dataset.cite(test_rel)}
    out = evaluate.compare(conn, backbones, methods, test_days=0, shared=shared,
                           extra=cites, **kw)
    out.update(cites)
    return out
