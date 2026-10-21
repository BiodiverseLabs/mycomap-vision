"""The Picek group's fungi classifier recipe, trained on our data (a baseline for the paper).

Lukas Picek's group (BVRA, Univ. of West Bohemia; a PI of the EU FunDive project, whose
model goes into PlutoF GO) trains a plain image classifier over every species, fully
fine-tuned from an ImageNet-21k vision transformer, with an imbalance-aware loss, heavy
augmentation and a metadata prior. This module repeats that recipe on Vision's labels and
records, so the paper can tell what is method and what is data: the same training records,
their method.

Sources (cited per piece below):
- FungiTastic, best closed-set model: Hugging Face BVRA/beit_base_patch16_384.in1k_ft_
  fungitastic_384 (config.yaml) and the training code in BohemianVRA/FungiTastic
  (baselines/closed_set/train.py), which runs on the `fgvc` library
  (github.com/BohemianVRA/FGVC-Tools: losses, augmentations, trainer).
- Danish Fungi 2020 (DF20), Picek et al., WACV 2022: the production ViT-L/16 384 and the
  metadata prior of its section 5.2.

The recipe (preset "fungitastic-beit-b384"):
- timm `beit_base_patch16_384.in22k_ft_in22k_in1k`, every weight trained, 384 x 384, a new
  linear head over all species.
- SGD, momentum 0.9, weight decay 0, learning rate 0.01; ReduceLROnPlateau on the
  validation loss (factor 0.9, patience 1, eps 1e-6); effective batch 256 (here micro-batch
  x gradient accumulation, with mixed precision and gradient checkpointing so it fits an
  8 GB GPU); 50 epochs; the checkpoint kept is the epoch with the best validation macro-F1.
- Seesaw loss (Wang et al., CVPR 2021; mmdet SeesawLoss), p = 0.8, q = 2.0, class counts
  from the training set (as fgvc's SeesawLoss takes them). Plain cross-entropy for their
  ViT-B variant.
- Training augmentation "vit_heavy": RandomResizedCrop(384, scale 0.8-1) then
  RandAugment(num_ops=2, magnitude=20); evaluation: resize to 384 x 384, no crop; both
  normalised with mean = std = 0.5.
- An observation's answer: each photo's logits divided by a temperature fitted on the
  validation set (NLL), averaged over the photos, softmax.
- Metadata prior (DF20 section 5.2): see MetadataPrior.

What we must do differently, and why (also in docs/PLAN.md, "Picek replication"):
- Splits: their train / validation / test are by year. Ours: the comparison's reference
  records (everything validated up to the cutoff, newest 28 days before the newest
  record), of which the last `val_days` are the validation slice (never trained on: it
  picks the epoch, drives the plateau schedule and fits the temperature). The test weeks
  and the held-out benchmarks are never seen (evaluate.check_not_trained_on_test,
  build_data).
- Classes are species labels only (Vision's labels: green records, observation names,
  provisional codes count as species). A record named with one word has no species label
  and is not trained on; their data has species labels only, too.
- Seesaw's pairwise matrix is C x C in fgvc; with ~18,000 species that is 1.3 GB, so the
  same factors are formed per batch row (B x C), in log space. Same loss.
- Training photos are decoded in PIL's draft mode (DCT scaling to the smallest size still
  at least the crop: 1024 -> 512 px), which leaves what the network sees at 384 px
  essentially unchanged and makes decoding cheaper. Validation decodes in full, exactly as
  the embedding at test time does.
- The metadata prior: month only (we have no habitat or substrate), smoothed (see
  MetadataPrior), and an optional coarse place prior that is OUR extension, kept as its
  own method so the paper can report their method and ours separately.

The trained model is registered like a fine-tuned backbone (data/models/<name>.json and
<name>.pt, plus <name>.classifier.npz holding the head, temperature and metadata counts,
and <name>.records.csv: every record it trained or validated on). Its "embedding" is the
pre-logit feature, stored so that the classifier can be applied to it exactly (see
encode_features), which lets `mv embed`, `mv compare` and `mv heldout predict` score it
with the methods `classifier`, `classifier+month` and `classifier+month+place` (and the
similarity methods too, on its features).
"""

from __future__ import annotations

import csv
import io
import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

KIND = "classifier"           # meta["kind"] of a model made here; its spec is classifier:<name>
LOG_FLOOR = math.log(1e-12)    # a species the classifier has no class for

from ... import sources  # noqa: E402  (record sources: MIGRATION, INAT, PHOTO_SOURCES)
# The recipe's presets and hyperparameters live in presets.py; read here as retrain.X too.
from .presets import (EXCLUDE_MODES, MEAN, PRESETS, STD, EpochRule, PicekConfig,  # noqa: E402,F401
                      Preset)

# The epoch rule's pre-declared block and gain (EpochRule). A spec names
# only the minimum and the cap ('preset@15-50'); these two are not free parameters.
RULE_BLOCK_EPOCHS = 5
RULE_MIN_GAIN = 0.005


def parse_spec(spec: str) -> tuple[str, int]:
    """'fungitastic-beit-b384@15' -> (preset, 15 epochs); no '@': the preset's epochs.
    'preset@15-50' trains by the epoch rule (epoch_rule_of): 15 to 50 epochs, and 50 is
    returned (the most it can take). A third part, 'preset@15@440', is the photo cache
    (cache_of)."""
    preset, _, rest = spec.partition("@")
    epochs, _, cache = rest.partition("@")
    if preset not in PRESETS:
        raise ValueError(f"unknown Picek preset {preset!r}: {', '.join(sorted(PRESETS))}")
    lo, dash, hi = epochs.partition("-")
    ok = (lo.isdigit() and int(lo) > 0 and (not dash or (hi.isdigit() and int(hi) >= int(lo))))
    if epochs and not ok:
        raise ValueError(f"bad epoch count in {spec!r}: use preset@epochs, e.g. {preset}@15, "
                         f"or preset@min-max for the epoch rule, e.g. {preset}@15-50")
    if cache and not (cache.isdigit() and int(cache) >= PRESETS[preset].image_size):
        raise ValueError(f"bad photo cache size in {spec!r}: preset@epochs@px, px at least "
                         f"the input size ({PRESETS[preset].image_size}), e.g. {preset}@15@440")
    return preset, int(hi or lo) if epochs else PRESETS[preset].epochs


def epoch_rule_of(spec: str) -> EpochRule | None:
    """The epoch rule a spec asks for ('preset@15-50' -> min 15, max 50, blocks of
    RULE_BLOCK_EPOCHS, gain RULE_MIN_GAIN), else None (a fixed count)."""
    parse_spec(spec)
    epochs = spec.split("@")[1] if "@" in spec else ""
    if "-" not in epochs:
        return None
    lo, hi = epochs.split("-")
    return EpochRule(int(lo), int(hi), RULE_BLOCK_EPOCHS, RULE_MIN_GAIN)


def block_decision(rule: EpochRule, val_macro_f1: list, epoch: int) -> dict | None:
    """The epoch rule's decision after `epoch` (1-based), from the validation macro-F1 of
    every epoch so far (`val_macro_f1[0]` is epoch 1's; None counts as no score). None when
    `epoch` is not one of the rule's block boundaries. Otherwise a record of the decision,
    written to progress.json and the model's metadata so it can be audited later:
    the best macro-F1 so far, the best at the previous boundary, the gain, continue or stop,
    and why. The first boundary (the minimum) has nothing to compare with and continues."""
    bounds = rule.boundaries()
    if epoch not in bounds:
        return None
    scores = [-1.0 if f is None else float(f) for f in val_macro_f1[:epoch]]
    best = max(scores)
    at = bounds.index(epoch)
    previous = max(scores[:bounds[at - 1]]) if at else None
    gain = None if previous is None else best - previous
    # The gain is a difference of fractions: 1e-9 keeps 0.305 - 0.300 a gain of 0.005.
    enough = gain is None or gain >= rule.min_gain - 1e-9
    if epoch >= rule.max_epochs:
        decision, reason = "stop", (f"reached the cap of {rule.max_epochs} epochs"
                                    + ("" if gain is None else f" (this block's gain {gain:+.4f})"))
    elif gain is None:
        decision, reason = "continue", (f"the minimum of {rule.min_epochs} epochs is done; "
                                        f"blocks of {rule.block} epochs follow")
    elif enough:
        decision, reason = "continue", (f"best val macro-F1 rose {gain:+.4f} over the block "
                                        f"(at least {rule.min_gain})")
    else:
        decision, reason = "stop", (f"best val macro-F1 rose only {gain:+.4f} over the block "
                                    f"(under {rule.min_gain})")
    return {"epoch": epoch, "best_val_macro_f1": round(best, 6),
            "best_epoch": 1 + scores.index(best),
            "previous_best_val_macro_f1": None if previous is None else round(previous, 6),
            "gain": None if gain is None else round(gain, 6), "decision": decision,
            "reason": reason,
            "decided_on": "validation macro-F1 (the release's val split; never a test split)"}


def cache_of(spec: str) -> int | None:
    """The photo cache's shorter side a spec asks for ('preset@15@440' -> 440), else None."""
    parse_spec(spec)
    parts = spec.split("@")
    return int(parts[2]) if len(parts) > 2 and parts[2] else None


# --- the training data --------------------------------------------------------------------

def month_of(day: str | None) -> int | None:
    """0-11, or None for no date (and the 1970-01-01 placeholder, dates.py)."""
    from ...dates import real_date
    d = real_date(day)
    return int(d[5:7]) - 1 if d else None


def cell_of(lat, lon, degrees: float) -> int | None:
    """A lat/lon grid cell id (degrees x degrees), or None without a place."""
    if lat is None or lon is None:
        return None
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat == 0 and lon == 0):
        return None
    rows = int(math.ceil(180 / degrees))
    i = min(int((lat + 90) // degrees), rows - 1)
    j = int((lon + 180) // degrees)
    return i * 1000 + j


def exclusion_ids(conn, mode: str, ids_file: str | None = None) -> set[str] | None:
    """The extra records to leave out for `mode`: None for "match" (load_records already
    leaves sealed benchmarks out, as for every Vision model), every heldout_records id for
    "all". A trainer's manifest carries no benchmark tables, so there the ids come from
    `ids_file` (shipped by the launcher), and "all" without them is refused."""
    if mode in ("match", "sealed"):
        return None
    if mode != "all":
        raise ValueError(f"exclude_benchmarks is one of {', '.join(EXCLUDE_MODES)} (not "
                         f"{mode!r}; sealed benchmarks are never trained on in any mode)")
    if ids_file:
        text = Path(ids_file).read_text(encoding="utf-8")
        return {line.strip() for line in text.splitlines() if line.strip()}
    ids = benchmark_record_ids(conn)
    have = conn.execute("select 1 from sqlite_master where type = 'table' and "
                        "name = 'heldout_records'").fetchone()
    if not have:
        raise ValueError("exclude_benchmarks=all, but this manifest has no benchmark tables "
                         "(a trainer's copy) and no ids file was given")
    return ids


@dataclass
class PicekData:
    classes: list[str]                       # species labels, sorted; index = class id
    genus_of: list[str]                      # each class's genus
    train: list[tuple[int, str, int]]        # (photo_id, path, class)
    val: list[tuple[int, str, int, str]]     # (photo_id, path, class or -1, observation_id)
    class_photos: np.ndarray                 # training photos per class (Seesaw's counts)
    month_counts: np.ndarray                 # (classes, 12) training photos per month
    cells: dict[int, dict[int, int]]         # cell -> {class: training photos} (our prior)
    trained_through: str                     # the comparison cutoff: newest date seen at all
    train_through: str                       # newest date of the training (gradient) slice
    train_records: list[str] = field(repr=False, default_factory=list)
    val_records: list[str] = field(repr=False, default_factory=list)
    excluded_benchmark_records: int = 0
    one_word_records: int = 0
    val_records_unknown_species: int = 0
    record_labels: list[tuple[str, str]] = field(repr=False, default_factory=list)
    release: dict | None = None              # the dataset release's cite(), when from one
    missing_photos: int = 0                  # release photos with no copy in the store
    photo_licences: dict = field(default_factory=dict)   # train + val photos by licence class
    record_sources: dict = field(default_factory=dict)    # train + val records by source
    place_source: str = "the manifest's records"           # where the place counts came from
    cell_degrees: float = 4.0                                # the place prior's grid
    public_trainable_only: bool = False                      # PHOTO_LICENCE_POLICY_PUBLIC


def benchmark_record_ids(conn) -> set[str]:
    """Every record of every held-out benchmark (heldout.py), whatever its split or state:
    the replication never trains or validates on them."""
    have = conn.execute("select 1 from sqlite_master where type = 'table' and "
                        "name = 'heldout_records'").fetchone()
    if not have:
        return set()
    return {str(r[0]) for r in conn.execute("select observation_id from heldout_records")}


def build_data(conn, store_location: str, size: str, test_days: int = 28, val_days: int = 28,
               cell_degrees: float = 4.0, exclude_ids: set[str] | None = None) -> PicekData:
    """The records a comparison would use as reference (every green record with a photo copy
    at `size` in the store, up to the cutoff newest-28-days before the newest record), split
    into training and the validation slice of the last `val_days` before the cutoff. Labels
    are Vision's (evaluate.load_records, which leaves sealed benchmarks out, as for every
    Vision model). `exclude_ids` (exclusion_ids) leaves more records out."""
    from ... import evaluate
    paths = dict(conn.execute("select photo_id, path from photo_copies where store = ? and "
                              "size = ?", (store_location, size)).fetchall())
    if not paths:
        raise ValueError(f"no {size} photos in {store_location}")
    records = evaluate.load_records(conn, {int(p): int(p) for p in paths})
    ref, _test, cutoff = evaluate.split_by_time(records, test_days)
    held = exclude_ids or set()
    excluded = sum(r.observation_id in held for r in ref)
    ref = [r for r in ref if r.observation_id not in held]
    one_word = sum(not r.species for r in ref)
    ref = [r for r in ref if r.species]
    val_from = (date.fromisoformat(cutoff) - timedelta(days=val_days)).isoformat()
    train_recs = [r for r in ref if not (r.validated_on and r.validated_on > val_from)]
    val_recs = [r for r in ref if r.validated_on and r.validated_on > val_from]
    if not train_recs:
        raise ValueError("no training records before the validation slice")
    classes = sorted({r.species for r in train_recs})
    pos = {c: i for i, c in enumerate(classes)}
    genus_of = [""] * len(classes)
    train, month_counts = [], np.zeros((len(classes), 12), dtype=np.int32)
    cells: dict[int, dict[int, int]] = {}
    for r in train_recs:
        k = pos[r.species]
        genus_of[k] = genus_of[k] or r.genus
        m = month_of(r.observed_on)
        cell = cell_of(r.latitude, r.longitude, cell_degrees)
        for pid in r.photo_rows:              # photo ids here
            train.append((int(pid), paths[pid], k))
            if m is not None:
                month_counts[k, m] += 1
            if cell is not None:
                row = cells.setdefault(cell, {})
                row[k] = row.get(k, 0) + 1
    val = [(int(pid), paths[pid], pos.get(r.species, -1), r.observation_id)
           for r in val_recs for pid in r.photo_rows]
    class_photos = np.bincount([k for _, _, k in train], minlength=len(classes)).astype(np.int64)
    newest_train = max((r.validated_on for r in train_recs if r.validated_on), default=cutoff)
    return PicekData(classes, genus_of, train, val, class_photos, month_counts, cells, cutoff,
                     newest_train, [r.observation_id for r in train_recs],
                     [r.observation_id for r in val_recs], excluded, one_word,
                     sum(r.species not in pos for r in val_recs),
                     [(r.observation_id, r.species) for r in train_recs + val_recs])


# --- the training data from a dataset release -------------------------------------------------

# Internal benchmarking uses every photo the release includes, all-rights-reserved ones too
# (Steve, 2026-10-09); only the final public model is CC + permission-granted. Recorded in
# every snapshot made from a release.
PHOTO_LICENCE_POLICY = ("internal benchmarking: every photo the release includes (open, nc "
                        "and arr, ARR with no answer yet included); not for a public model")
PHOTO_LICENCE_POLICY_PUBLIC = ("public: only the photos the public model may learn from (CC "
                               "including NC, or ARR with the owner's permission), in training "
                               "and validation, and the priors' counts from them; a record left "
                               "with none, and a species left with none, drop out")
MAX_MISSING_PHOTOS = 0.01


def check_release_exclusion(rel, mode: str) -> None:
    """The release decides which records train (its splits). `match` follows it; `all`
    (leave every benchmark record out) is refused when the release trains on benchmark
    records, rather than training on other records than Vision."""
    from . import dataset
    if mode not in EXCLUDE_MODES:
        raise ValueError(f"exclude_benchmarks is one of {', '.join(EXCLUDE_MODES)}")
    n = dataset.benchmark_records(rel)
    if mode == "all" and n:
        raise ValueError(f"release {rel.id} trains on {n:,} held-out benchmark records; "
                         "--picek-exclude-benchmarks all would train on other records than "
                         "the release (and Vision). Use match: the release's splits decide.")


def build_data_from_release(conn, rel, store_location: str, size: str,
                            cell_degrees: float = 4.0,
                            place_cells: dict | None = None,
                            public_trainable_only: bool = False) -> PicekData:
    """PicekData from a dataset release (dataset_release.Release): its `train` split
    trains (rel.training_records(); a test release raises SealedLeak), its `val` split
    validates (the plateau schedule, the best macro-F1 epoch and the temperature), and
    nothing else is read. Labels are the release's (`species`; a one-word name isn't
    trained). Photos are the release's originals (manifest_photo_id), read from this
    manifest's store copies (rel.check_photos_match: the same bytes); month and place prior
    counts come from the train split only.

    Places are the release's own (private.sqlite's record_places), so they exist only on
    the laptop. A trainer instance gets no private.sqlite: there the place counts are
    `place_cells`, computed on the laptop at launch (place_cells_doc) and shipped; without
    them the data has no place counts (PLACE_NONE), which train() refuses. The manifest's
    coordinates are never read here.

    `public_trainable_only` (PHOTO_LICENCE_POLICY_PUBLIC): train and validate on the
    release's public-trainable photos only; a record with none drops out before the classes
    are made (so a species with none drops out too), and the month and place counts come
    from the photos kept."""
    from . import dataset
    train_rows = [dict(r) for r in rel.training_records()]
    val_rows = [dict(r) for r in rel.records("val")]
    for r in train_rows + val_rows:
        if r["split"] not in dataset.TRAINABLE:
            raise dataset.SealedLeak(f"{r['record_key']} ({r['split']}) is not trainable")
    if size == "large":
        rel.check_photos_match(conn)
    copies = {int(p): path for p, path in conn.execute(
        "select photo_id, path from photo_copies where store = ? and size = ?",
        (store_location, size))}
    if not copies:
        raise ValueError(f"no {size} photos in {store_location}")
    if rel.has_private:
        place = {str(k): (None if lat is None else lat / 1e6, None if lon is None else lon / 1e6)
                 for k, lat, lon in rel.private.execute(
                     "select record_key, lat_micro, lon_micro from record_places")}
        place_source = PLACE_RELEASE
    else:
        place = {}
        place_source = PLACE_SHIPPED if place_cells is not None else PLACE_NONE
    one_word = sum(not r.get("species") for r in train_rows + val_rows)
    train_recs = [r for r in train_rows if r.get("species")]
    val_recs = [r for r in val_rows if r.get("species")]

    def kept(p) -> bool:
        return not public_trainable_only or int(p["public_trainable"] or 0) == 1
    if public_trainable_only:
        def has_public(rec) -> bool:
            return any(kept(p) for p in rel.photos(rec["record_key"]))
        train_recs = [r for r in train_recs if has_public(r)]
        val_recs = [r for r in val_recs if has_public(r)]
    if not train_recs:
        raise ValueError(f"release {rel.id} has no training records")
    classes = sorted({r["species"] for r in train_recs})
    pos = {c: i for i, c in enumerate(classes)}
    genus_of = [""] * len(classes)
    missing, wanted = 0, 0
    licences, by_source = {}, {}

    def paths_of(rec):
        nonlocal missing, wanted
        out = []
        for p in rel.photos(rec["record_key"]):
            if not kept(p):
                continue
            wanted += 1
            lc = p["license_class"] or "(none)"
            licences[lc] = licences.get(lc, 0) + 1
            pid = int(p["manifest_photo_id"])
            if pid not in copies:
                missing += 1
                continue
            out.append((pid, copies[pid]))
        return out
    train, month_counts = [], np.zeros((len(classes), 12), dtype=np.int32)
    cells: dict[int, dict[int, int]] = {}
    for r in train_recs:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
        k = pos[r["species"]]
        genus_of[k] = genus_of[k] or (r.get("genus") or "")
        m = month_of(r.get("observed_on"))
        lat, lon = place.get(str(r["record_key"]), (None, None))
        cell = cell_of(lat, lon, cell_degrees)
        for pid, path in paths_of(r):
            train.append((pid, path, k))
            if m is not None:
                month_counts[k, m] += 1
            if cell is not None:
                row = cells.setdefault(cell, {})
                row[k] = row.get(k, 0) + 1
    val = []
    for r in val_recs:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
        val += [(pid, path, pos.get(r["species"], -1), r["record_key"])
                for pid, path in paths_of(r)]
    if wanted and missing > MAX_MISSING_PHOTOS * wanted:
        raise ValueError(f"{missing:,} of release {rel.id}'s {wanted:,} train/val photos have "
                         f"no {size} copy in {store_location}")
    if place_cells is not None and not rel.has_private:
        cells = cells_from_doc(place_cells, rel, pos, cell_degrees)
    class_photos = np.bincount([k for _, _, k in train], minlength=len(classes)).astype(np.int64)
    dates = lambda rs: [r["validated_on"] for r in rs if r.get("validated_on")]  # noqa: E731
    newest_seen = max(dates(train_recs + val_recs), default="")
    newest_train = max(dates(train_recs), default=newest_seen)
    return PicekData(classes, genus_of, train, val, class_photos, month_counts, cells,
                     newest_seen, newest_train, [r["record_key"] for r in train_recs],
                     [r["record_key"] for r in val_recs], 0, one_word,
                     sum(r["species"] not in pos for r in val_recs),
                     [(r["record_key"], r["species"]) for r in train_recs + val_recs],
                     release=dataset.cite(rel), missing_photos=missing, photo_licences=licences,
                     record_sources=dict(sorted(by_source.items(), key=lambda kv: -kv[1])),
                     place_source=place_source, cell_degrees=cell_degrees,
                     public_trainable_only=public_trainable_only)


# Where a release run's place prior counts come from (PicekData.place_source).
PLACE_RELEASE = "the release's private.sqlite (record_places), on the laptop"
PLACE_SHIPPED = "counts computed on the laptop at launch and shipped (picek-place-cells.json)"
PLACE_NONE = "none: no private.sqlite here and no shipped counts"
PLACE_CELLS_FILE = "picek-place-cells.json"


def place_cells_doc(rel, data: PicekData) -> dict | None:
    """The place prior's training counts as a document a trainer instance can use without
    private.sqlite: per coarse grid cell, training photos per species (no coordinates,
    no record ids). None when the data has no place counts."""
    if data.place_source == PLACE_NONE:
        return None
    return {"dataset_release": rel.id, "cell_degrees": float(data.cell_degrees),
            "cells": {str(c): {data.classes[k]: int(n) for k, n in sorted(row.items())}
                      for c, row in sorted(data.cells.items())}}


def place_cells_sha256(doc: dict | None) -> str | None:
    if doc is None:
        return None
    import hashlib
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":"))
                          .encode()).hexdigest()


def read_place_cells(path, expect_sha256: str | None = None) -> dict:
    """A shipped place_cells_doc, refused when it is not the one the launch made."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if expect_sha256 and place_cells_sha256(doc) != expect_sha256:
        raise ValueError(f"{path}: the place counts are not the ones the launch made "
                         f"(sha256 {place_cells_sha256(doc)[:12]}, launch "
                         f"{expect_sha256[:12]}): not training")
    return doc


def cells_from_doc(doc: dict, rel, pos: dict[str, int], cell_degrees: float
                   ) -> dict[int, dict[int, int]]:
    """PicekData.cells from a shipped place_cells_doc, for this release and grid."""
    if doc.get("dataset_release") != rel.id:
        raise ValueError(f"the shipped place counts are for {doc.get('dataset_release')!r}, "
                         f"not {rel.id}")
    if float(doc.get("cell_degrees") or 0) != float(cell_degrees):
        raise ValueError(f"the shipped place counts use a {doc.get('cell_degrees')} degree "
                         f"grid, not {cell_degrees}")
    unknown = sorted({s for row in doc["cells"].values() for s in row if s not in pos})
    if unknown:
        raise ValueError(f"the shipped place counts name {len(unknown)} species this release "
                         f"doesn't train (e.g. {unknown[:3]})")
    return {int(c): {pos[s]: int(n) for s, n in row.items()} for c, row in doc["cells"].items()}


def release_snapshot(rel, data: PicekData, exclude_mode: str = "match") -> dict:
    """label_snapshot for a run on a dataset release: the release's identity (its
    labels_hash is the run's labels_hash), what it trains and validates on, the photo
    licence policy and the record sources against the release's own sources."""
    from . import dataset
    ok = allowed_sources(release=rel)
    return {
        "labels_hash": rel.labels_hash, "own_labels_hash": labels_hash(data),
        "dataset_release": dataset.cite(rel), "trained_through": data.trained_through,
        "train_through": data.train_through, "val_cutoff": rel.val_cutoff,
        "exclude_benchmarks": f"{exclude_mode} (the release's splits decide)",
        "benchmark_records_trained": dataset.benchmark_records(rel),
        "validation": "the release's val split (plateau, best macro-F1 epoch, temperature)",
        "records_train": len(data.train_records), "records_validation": len(data.val_records),
        "photos_train": len(data.train), "photos_validation": len(data.val),
        "photos_missing_in_store": data.missing_photos, "species": len(data.classes),
        "one_word_records_left_out": data.one_word_records,
        "photo_licence_policy": (PHOTO_LICENCE_POLICY_PUBLIC if data.public_trainable_only
                                 else PHOTO_LICENCE_POLICY),
        "photo_licences": data.photo_licences,
        "public_trainable_only": data.public_trainable_only,
        "place_prior": {"source": data.place_source,
                        "cells": len(data.cells),
                        "photos_with_place": int(sum(sum(r.values())
                                                     for r in data.cells.values())),
                        "cells_sha256": place_cells_sha256(place_cells_doc(rel, data))},
        "allowed_sources": list(ok),
        "record_sources": {"recorded": True, "migration": "dataset release",
                           "allowed_sources": list(ok),
                           "records": len(data.train_records) + len(data.val_records),
                           "by_source": data.record_sources,
                           "not_allowed": sum(n for s, n in data.record_sources.items()
                                              if s not in ok)},
    }


def labels_hash(data: PicekData) -> str:
    """Which labelling a run trained on: the hash of every (record, species label) of its
    training and validation records, and of the cutoff."""
    import hashlib
    h = hashlib.sha256(f"{data.trained_through}|{data.train_through}\n".encode())
    h.update("\n".join(f"{o}\t{s}" for o, s in sorted(data.record_labels)).encode())
    return h.hexdigest()[:16]


def label_snapshot(conn, data: PicekData, exclude_mode: str = "match",
                   allowed: tuple[str, ...] | None = None) -> dict:
    """What a run's labels were, to prove later which labelling it used: counts, the
    share of provisional names, what was left out and why, the known label problems the
    manifest carries (label conflicts; names that wait for a person, names.py), the
    record sources the run allows and has (source_check), and labels_hash."""
    from ... import names
    provisional = sum(names.parse_name(c).code is not None for c in data.classes)
    is_code = {c: names.parse_name(c).code is not None for c in data.classes}
    labelled = [s for _o, s in data.record_labels]
    rows = names.name_counts(conn)
    spell = names.summary(rows, names.group_name_variants(rows))
    conflicts = conn.execute("select count(*) from records where label_conflict = 1"
                             ).fetchone()[0]
    newest = conn.execute("select max(exported_at) from records").fetchone()[0]
    return {
        "labels_hash": labels_hash(data), "trained_through": data.trained_through,
        "train_through": data.train_through, "exclude_benchmarks": exclude_mode,
        "records_train": len(data.train_records), "records_validation": len(data.val_records),
        "photos_train": len(data.train), "photos_validation": len(data.val),
        "species": len(data.classes),
        "provisional_species_share": round(provisional / max(len(data.classes), 1), 4),
        "provisional_record_share": round(sum(is_code.get(s, names.parse_name(s).code
                                                          is not None) for s in labelled)
                                          / max(len(labelled), 1), 4),
        "one_word_records_left_out": data.one_word_records,
        "excluded_benchmark_records": data.excluded_benchmark_records,
        "known_label_problems": {
            "label_conflict_records": conflicts,
            "names_left_for_a_person": spell["left_for_a_person"],
        },
        "manifest_newest_export": newest,
        "allowed_sources": list(allowed_sources(allowed)),
        "record_sources": source_check(conn, data, allowed),
    }


# --- where the records come from -----------------------------------------------------------
# A record's photos must be its own. Records that are not iNat observations (Mushroom
# Observer, MyCoPortal, .com sequences, GenBank) fetched AS iNat ids carried other
# observations' photos (mammals, birds, plants). The record-sources fix (sources.py, on
# main) normalises records.source to inat | mo | mycoportal | com_sequence | genbank |
# unknown, from .org's own field, and marks a migrated manifest with a
# manifest_migrations row; evaluate.load_records then loads only inat and mo records, each
# with photos of its own source. The checks here stay as the run's own proof. BEFORE that migration records.source holds the old
# guess ('inat' for any numeric id), so it says nothing until the marker is there; an
# unmigrated manifest can't prove where its records come from, and a real launch is
# refused (the dry run only warns). Never decided by the shape of an id.
#
# Which sources a run may train on (Steve, 2026-10-09): the Dataset release's allowed
# sources, the same release for Vision and the replication: iNat, and DNA-validated MO
# records with photos from MO itself. MyCoPortal, .com sequences, GenBank and unknown are
# never allowed. Until a release exists the set is ('inat',): strict.
MIGRATION = sources.MIGRATION
INAT = sources.INAT
PHOTO_SOURCES = sources.PHOTO_SOURCES   # the most any release may allow: inat, mo
SOURCE_COLUMN = "source"
DEFAULT_ALLOWED_SOURCES = (INAT,)


class RecordSourceError(ValueError):
    """A Picek run's records are not proven to come from its allowed sources only."""


def allowed_sources(allowed=None, release=None) -> tuple[str, ...]:
    """The sources a run may train on: `allowed` if given, else the release's own
    `allowed_sources` (a release without it is refused: dataset.Release), else
    DEFAULT_ALLOWED_SOURCES. Refuses an empty set and any source outside PHOTO_SOURCES
    (inat, mo), whoever names it."""
    if allowed is None and release is not None:
        from . import dataset
        allowed = dataset.allowed_sources(release)
    chosen = tuple(sorted(set(DEFAULT_ALLOWED_SOURCES if allowed is None else allowed)))
    bad = sorted(set(chosen) - set(PHOTO_SOURCES))
    if not chosen or bad:
        raise RecordSourceError(
            f"allowed sources must be a non-empty subset of {list(PHOTO_SOURCES)}; "
            f"refused: {bad or 'an empty set'}")
    return chosen


def sources_migrated(conn) -> bool:
    """Whether this manifest's records.source is the record-sources fix's (its migration
    marker row is there). A missing manifest_migrations table = not migrated. Reads only:
    never creates the table (sources.migrated)."""
    return sources.migrated(conn)


def record_sources(conn, record_ids) -> dict[str, int] | None:
    """Records per source among `record_ids`, or None when the manifest's sources are not
    the migrated ones (sources_migrated)."""
    if not sources_migrated(conn):
        return None
    ids = sorted({str(i) for i in record_ids})
    counts: dict[str, int] = {}
    for at in range(0, len(ids), 500):          # reads only: the manifest may be read-only
        part = ids[at:at + 500]
        for s, n in conn.execute(f"select coalesce({SOURCE_COLUMN}, '(none)'), count(*) "
                                 f"from records where observation_id in "
                                 f"({','.join('?' * len(part))}) group by 1", part):
            counts[str(s)] = counts.get(str(s), 0) + int(n)
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def source_check(conn, data: PicekData, allowed=None) -> dict:
    """The training and validation records by recorded source (label snapshot,
    result.json): `recorded` False when the manifest isn't migrated; `not_allowed` counts
    the records whose source is outside the run's allowed sources."""
    ok = allowed_sources(allowed)
    ids = list(data.train_records) + list(data.val_records)
    by = record_sources(conn, ids)
    base = {"migration": MIGRATION, "allowed_sources": list(ok), "records": len(ids)}
    if by is None:
        return {"recorded": False, **base, "by_source": None, "not_allowed": None}
    return {"recorded": True, **base, "by_source": by,
            "not_allowed": sum(n for s, n in by.items() if s not in ok)}


def require_allowed_sources(check: dict, dry_run: bool = False, log=print) -> None:
    """Refuse a run whose training or validation records aren't proven to come from its
    allowed sources: some record's source is outside them, or the manifest isn't migrated.
    A dry run (sends nothing) goes on with a loud warning, so the runbook stays testable."""
    ok = ", ".join(check["allowed_sources"])
    if not check["recorded"]:
        msg = (f"the manifest's record sources are not migrated (no {MIGRATION} marker), "
               f"so this run can't prove that its {check['records']:,} training and "
               f"validation records come from its allowed sources ({ok}); records fetched "
               "by another site's id carry other observations' photos. The launch waits "
               "for a migrated snapshot (docs/PLAN.md, launch day).")
    elif check["not_allowed"]:
        counts = ", ".join(f"{s}: {n:,}" for s, n in check["by_source"].items())
        msg = (f"{check['not_allowed']:,} of the {check['records']:,} training and "
               f"validation records come from sources outside this run's allowed sources "
               f"({ok}): {counts}. The launch waits for a snapshot of allowed sources only "
               "(docs/PLAN.md, launch day).")
    else:
        log(f"Record sources: all {check['records']:,} training and validation records "
            f"come from the allowed sources ({ok}).")
        return
    if dry_run:
        log("WARNING (a real launch is refused): " + msg)
        return
    raise RecordSourceError("not launching: " + msg)


# The coordinator's 2026-10-09 audit: .org's live source per record (observation_id, source
# as .org spells it). Laptop only, informational for a dry run; never shipped.
AUDIT_TSV = Path("audits") / "record-sources-2026-10-09" / "org-sources-live.tsv"
AUDIT_INAT = "iNaturalist"


def audit_not_inat(path: Path, record_ids) -> dict | None:
    """How many of `record_ids` the audit file lists with a source other than iNaturalist,
    by its source, or None without the file."""
    if not path.is_file():
        return None
    ids = {str(i) for i in record_ids}
    listed: dict[str, str] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            oid = (row.get("observation_id") or "").strip()
            if oid in ids:
                listed[oid] = (row.get("source") or "").strip() or "(none)"
    by: dict[str, int] = {}
    for src in listed.values():
        if src != AUDIT_INAT:
            by[src] = by.get(src, 0) + 1
    return {"file": str(path), "records": len(ids), "listed": len(listed),
            "not_inat": sum(by.values()),
            "by_source": dict(sorted(by.items(), key=lambda kv: -kv[1]))}


def run_sources(conn, store_location: str, size: str, test_days: int = 28,
                exclude: str = "match", ids_file: str | None = None, allowed=None) -> dict:
    """source_check for the records a Picek run on this manifest would train and validate
    on (the instance's own check, before its first stage)."""
    data = build_data(conn, store_location, size, test_days, PicekConfig.val_days,
                      exclude_ids=exclusion_ids(conn, exclude, ids_file))
    return source_check(conn, data, allowed)


def format_snapshot(snap: dict) -> str:
    if snap.get("dataset_release"):
        rel = snap["dataset_release"]
        return "\n".join([
            f"Labels for this run: dataset release {rel['dataset_release']}"
            f"{' (DRAFT)' if rel.get('draft') else ''}, labels hash {snap['labels_hash']}, "
            f"MANIFEST {rel['manifest_sha256'][:12]}: {snap['records_train']:,} train + "
            f"{snap['records_validation']:,} val records ({snap['photos_train']:,} + "
            f"{snap['photos_validation']:,} photos), {snap['species']:,} species; "
            f"validation is the release's val split",
            f"  one-word names left out: {snap['one_word_records_left_out']:,}; photos missing "
            f"in the store: {snap['photos_missing_in_store']:,}; benchmark records trained: "
            f"{snap['benchmark_records_trained']:,}",
            f"  photo licences: {', '.join(f'{k} {v:,}' for k, v in snap['photo_licences'].items())}"
            f" ({snap['photo_licence_policy']})",
            "  record sources: " + format_sources(snap.get("record_sources")),
        ])
    p = snap["known_label_problems"]
    return "\n".join([
        f"Labels for this run (hash {snap['labels_hash']}): {snap['records_train']:,} training "
        f"+ {snap['records_validation']:,} validation records, {snap['species']:,} species "
        f"({100 * snap['provisional_species_share']:.1f}% provisional names, "
        f"{100 * snap['provisional_record_share']:.1f}% of records)",
        f"  trained through {snap['trained_through']} (gradients through "
        f"{snap['train_through']}); benchmarks excluded: {snap['exclude_benchmarks']} "
        f"({snap['excluded_benchmark_records']:,} records); one-word names left out: "
        f"{snap['one_word_records_left_out']:,}; newest export {snap['manifest_newest_export']}",
        f"  known label problems in the manifest: {p['label_conflict_records']:,} records with "
        f"conflicting names (left out), {p['names_left_for_a_person']['groups']:,} names "
        f"({p['names_left_for_a_person']['records']:,} records) waiting for a person "
        "(mv name-spellings)",
        "  record sources: " + format_sources(snap.get("record_sources")),
    ])


def format_sources(check: dict | None) -> str:
    if not check or not check["recorded"]:
        return (f"NOT MIGRATED (no {MIGRATION} marker in this manifest: can't prove the "
                "records come from the allowed sources)")
    found = ", ".join(f"{s} {n:,}" for s, n in check["by_source"].items()) or "no records"
    return f"{found} (allowed: {', '.join(check['allowed_sources'])})"


# --- the photo cache --------------------------------------------------------------------

def resize_short_side(img, px: int):
    """The image with its shorter side at most `px` (aspect kept; never enlarged)."""
    from PIL import Image
    w, h = img.size
    scale = px / min(w, h)
    if scale >= 1:
        return img
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)


def cache_photos(store, items, dest: Path, px: int, threads: int = 8, log=print) -> dict:
    """Copy each training photo once, shorter side `px`, JPEG quality 90, to `dest` under its
    own relative path, so training reads small local files instead of 1024 px photos from
    S3: the data loader, not the GPU, sets the pace on a 4-vCPU instance (docs/PLAN.md).
    The network then sees a RandomResizedCrop of the 440 px copy instead of the original
    (decoded at 512 px in draft mode); everything after the read is unchanged. Resumable:
    a file already there is kept. Returns counts and the time taken."""
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image, ImageOps
    dest = Path(dest)
    started = time.monotonic()

    def one(item):
        rel = item[1]
        out = dest / rel
        if out.is_file() and out.stat().st_size:
            return "kept"
        try:
            img = Image.open(io.BytesIO(store.get(rel)))
            if img.format == "JPEG":
                img.draft("RGB", (px, px))
            img = resize_short_side(ImageOps.exif_transpose(img).convert("RGB"), px)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(out.suffix + ".part")
            img.save(tmp, "JPEG", quality=90)
            tmp.replace(out)
            return "made"
        except Exception:
            return "failed"         # the loader skips it as it would an unreadable original
    counts = {"made": 0, "kept": 0, "failed": 0}
    unique = list({rel: (pid, rel) for pid, rel, *_ in items}.values())
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for i, r in enumerate(pool.map(one, unique), 1):
            counts[r] += 1
            if i % 20000 == 0:
                log(f"  photo cache: {i:,}/{len(unique):,}")
    secs = time.monotonic() - started
    return {"short_side": px, "location": str(dest), "photos": len(unique), **counts,
            "seconds": round(secs, 1),
            "photos_per_second": round(len(unique) / secs, 1) if secs else None}


def make_datasets(train_location: str, val_location: str, data: "PicekData", preset: Preset,
                  val_items) -> tuple["Photos", "Photos"]:
    """The training and validation datasets. With a photo cache only train_location
    changes: transforms, draft decoding and items are the same either way."""
    size_px = preset.image_size
    return (Photos(train_location, data.train, train_transform(preset.augment, size_px),
                   draft=size_px),
            # Full decode, exactly as the embedding at test time (embed.decode) reads it.
            Photos(val_location, val_items, eval_transform(size_px), draft=None))


# --- the loss -----------------------------------------------------------------------------

def seesaw_logits(logits, labels, log_counts, p: float = 0.8, q: float = 2.0, eps: float = 1e-2):
    """The logits Seesaw loss takes the cross-entropy of (Wang et al., "Seesaw Loss for
    Long-Tailed Instance Segmentation", CVPR 2021; mmdet's SeesawLoss, as fgvc uses it).

    For a photo of class i, every other class j's logit gets + log S_ij, where
    S_ij = M_ij * C_ij:
      mitigation   M_ij = (N_j / N_i) ** p when N_j < N_i, else 1 (N: training photos), so
                   a rare class j is pushed down less by the common class i's photos;
      compensation C_ij = (sigma_j / sigma_i) ** q when sigma_j > sigma_i (sigma: the
                   softmax, detached; sigma_i clamped at eps as mmdet does), else 1, so a
                   class wrongly scored above the truth is pushed down harder.
    Formed per row (B x C) in log space: fgvc builds the whole C x C matrix, which at
    ~18,000 species is 1.3 GB."""
    import torch
    lc = log_counts.to(logits.device, torch.float32)
    z = logits.float()
    own = lc[labels]
    log_s = p * torch.clamp(lc[None, :] - own[:, None], max=0.0)
    if q:
        log_sigma = torch.log_softmax(z.detach(), dim=1)
        own_sigma = torch.clamp(log_sigma.gather(1, labels[:, None]), min=math.log(eps))
        log_s = log_s + q * torch.clamp(log_sigma - own_sigma, min=0.0)
    log_s = log_s.scatter(1, labels[:, None], 0.0)          # S_ii = 1
    return z + log_s


class SeesawLoss:
    def __init__(self, class_counts, p: float = 0.8, q: float = 2.0):
        import torch
        counts = torch.as_tensor(np.maximum(np.asarray(class_counts, dtype=np.float64), 1.0))
        self.log_counts = torch.log(counts).float()
        self.p, self.q = p, q

    def __call__(self, logits, labels):
        import torch.nn.functional as F
        return F.cross_entropy(seesaw_logits(logits, labels, self.log_counts, self.p, self.q),
                               labels)


def make_loss(kind: str, class_counts, cfg: PicekConfig):
    if kind == "seesaw":
        return SeesawLoss(class_counts, cfg.seesaw_p, cfg.seesaw_q)
    if kind == "ce":
        import torch.nn.functional as F
        return lambda logits, labels: F.cross_entropy(logits.float(), labels)
    raise ValueError(f"unknown loss {kind!r}")


# --- augmentation -------------------------------------------------------------------------

def train_transform(augment: str, size: int):
    """fgvc's "vit_heavy": RandomResizedCrop(size, scale 0.8-1) then RandAugment(2, 20).
    "light" (DF20's production training): RandomResizedCrop(size, scale 0.8-1), horizontal
    and vertical flips, brightness and contrast. Both normalised with mean = std = 0.5."""
    from torchvision import transforms as T
    crop = T.RandomResizedCrop(size, scale=(0.8, 1.0))
    if augment == "vit_heavy":
        middle = [crop, T.RandAugment(num_ops=2, magnitude=20)]
    elif augment == "light":
        middle = [crop, T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                  T.ColorJitter(brightness=0.2, contrast=0.2)]
    else:
        raise ValueError(f"unknown augmentation {augment!r}")
    return T.Compose([*middle, T.ToTensor(), T.Normalize(MEAN, STD)])


def eval_transform(size: int):
    """Resize to size x size (no crop), mean = std = 0.5 (fgvc's test transform)."""
    from torchvision import transforms as T
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])


class Photos:
    """(image tensor, label) per photo, read from a store by location so it pickles into
    DataLoader workers. JPEGs are decoded in draft mode at no less than `draft` px."""

    def __init__(self, location: str, items, transform, draft: int | None = None):
        self.location, self.items, self.transform, self.draft = location, items, transform, draft
        self._store = None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image, ImageOps

        from ...storage import open_store
        if self._store is None:
            self._store = open_store(self.location, Path(self.location))
        try:
            img = Image.open(io.BytesIO(self._store.get(self.items[i][1])))
            if self.draft and img.format == "JPEG":
                img.draft("RGB", (self.draft, self.draft))
            img = ImageOps.exif_transpose(img).convert("RGB")
            return self.transform(img), int(self.items[i][2]), i
        except Exception:
            return None


def collate(batch):
    import torch
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return (torch.stack([b[0] for b in batch]), torch.as_tensor([b[1] for b in batch]),
            torch.as_tensor([b[2] for b in batch]))


# --- metrics ------------------------------------------------------------------------------

from ...metrics import macro_f1  # noqa: E402  (shared with comparisons and the held-out report)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, groups: np.ndarray | None = None
                    ) -> float:
    """The temperature T minimising the NLL of softmax(mean over an observation's photos of
    logits / T) on the validation set (temperature scaling, Guo et al. 2017), only on
    photos whose class the model has. `groups`: each photo's observation; None = per photo."""
    keep = labels >= 0
    z, y = logits[keep].astype(np.float64), labels[keep]
    if groups is not None:
        g = np.asarray(groups)[keep]
        order = np.argsort(g, kind="stable")
        z, y, g = z[order], y[order], g[order]
        starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
        z = np.add.reduceat(z, starts, axis=0) / np.diff(np.r_[starts, len(g)])[:, None]
        y = y[starts]
    if not len(y):
        return 1.0

    def nll(t):
        s = z / t
        s = s - s.max(axis=1, keepdims=True)
        lse = np.log(np.exp(s).sum(axis=1))
        return float(np.mean(lse - s[np.arange(len(y)), y]))
    grid = np.geomspace(0.05, 20.0, 60)
    best = grid[int(np.argmin([nll(t) for t in grid]))]
    lo, hi = best / 1.15, best * 1.15          # refine: golden section in log space
    for _ in range(40):
        a, b = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        if nll(a) < nll(b):
            hi = b
        else:
            lo = a
    return round(float((lo + hi) / 2), 5)


# --- the embedding that carries the classifier's input -------------------------------------

def encode_features(features: np.ndarray, scale: float) -> np.ndarray:
    """Pre-logit features -> unit vectors that keep their length, so the classifier can be
    applied to an L2-normalised embedding exactly. The last of d + 1 dimensions holds
    t = |f| / scale (scale is ~20x the median length, so t ~ 0.05); the first d hold the
    direction scaled by sqrt(1 - t^2). Cosine between two such vectors is the features'
    cosine to within ~t^2, so nearest-specimen methods still work on them."""
    f = np.asarray(features, dtype=np.float32)
    n = np.linalg.norm(f, axis=1, keepdims=True)
    t = np.clip(n / scale, 0.0, 0.99)
    direction = f / np.maximum(n, 1e-12)
    return np.concatenate([direction * np.sqrt(1 - t * t), t], axis=1)


def decode_features(vectors: np.ndarray, scale: float) -> np.ndarray:
    """encode_features' inverse (as stored: float16, normalised again)."""
    v = np.asarray(vectors, dtype=np.float32)
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    t = v[:, -1:]
    d = v[:, :-1]
    return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12) * (t * scale)


# --- training -----------------------------------------------------------------------------

def create_model(preset: Preset, n_classes: int, pretrained: bool = True):
    import timm
    return timm.create_model(preset.timm_name, pretrained=pretrained, num_classes=n_classes)


def features_and_logits(model, x):
    """(pre-logit features, logits) of a timm classifier: forward_head with pre_logits is
    the pooled, normed feature the head is applied to."""
    feats = model.forward_head(model.forward_features(x), pre_logits=True)
    return feats, model.get_classifier()(feats)


def run_validation(model, loader, device, n_classes: int, loss_fn, amp_dtype=None) -> dict:
    """Per-photo logits, features' lengths and labels over the validation photos."""
    import torch
    model.eval()
    logits, labels, rows, norms = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            if batch is None:
                continue
            x, y, i = batch
            with torch.autocast(device, dtype=amp_dtype or torch.float32,
                                enabled=amp_dtype is not None):
                f, z = features_and_logits(model, x.to(device, non_blocking=True))
            logits.append(z.float().cpu().numpy())
            norms.append(f.float().norm(dim=1).cpu().numpy())
            labels.append(y.numpy())
            rows.append(i.numpy())
    model.train()
    if not logits:
        return {"logits": np.zeros((0, n_classes), np.float32), "labels": np.zeros(0, int),
                "rows": np.zeros(0, int), "norms": np.zeros(0), "loss": None, "macro_f1": None,
                "top1": None, "n": 0}
    z, y = np.concatenate(logits), np.concatenate(labels)
    known = y >= 0
    loss = None
    if known.any():
        import torch
        zk, yk, total = z[known], y[known], 0.0
        for s in range(0, len(yk), 1024):           # B x C temporaries stay small
            part = loss_fn(torch.from_numpy(zk[s:s + 1024]), torch.from_numpy(yk[s:s + 1024]))
            total += float(part) * len(yk[s:s + 1024])
        loss = total / len(yk)
    pred = z.argmax(axis=1)
    return {"logits": z, "labels": y, "rows": np.concatenate(rows), "norms": np.concatenate(norms),
            "loss": loss, "n": int(known.sum()),
            "macro_f1": macro_f1(y[known].tolist(), pred[known].tolist()),
            "top1": float((pred[known] == y[known]).mean()) if known.any() else None}


def models_dir() -> Path:
    from ...finetune import models_dir as md
    return md()


CHECKPOINT_SUFFIX = ".block.pt"       # the epoch rule's block checkpoint, beside the model


def train(conn, store, size: str, name: str, cfg: PicekConfig | None = None,
          test_days: int = 28, out_dir: Path | None = None, model_factory=None, log=print,
          should_stop=None, data: PicekData | None = None, on_block=None,
          resume_from: Path | None = None, time_left=None) -> dict:
    """Train the preset's classifier on the reference records (build_data), keep the epoch
    with the best validation macro-F1, fit its temperature, save and register it. Returns
    the metadata. `should_stop()` is asked between steps: when it says stop, the run ends
    with screening.Stopped and no model is saved, as for a fine-tune.

    With `cfg.epoch_rule` (EpochRule) the run trains until the rule stops it
    (block_decision). After every block boundary it writes `<name>.block.pt` (weights,
    optimizer, plateau schedule, the best epoch's weights, the curve and the decisions) and
    calls `on_block(path, state)` (the trainer uploads it and progress.json) BEFORE the next
    block starts. A stop between boundaries loses at most that block: `resume_from` (such a
    file) carries on from it. At a boundary, `time_left()` (seconds, optional) that cannot
    fit another block stops the run there rather than mid-block.

    Every refusal comes before torch is imported, so they hold (and are tested) without it."""
    cfg = (cfg or PicekConfig()).resolved()
    preset = PRESETS[cfg.preset]
    out_dir = out_dir or models_dir()
    started = time.monotonic()
    rel = None
    if cfg.release:
        from . import dataset
        rel = dataset.load(cfg.release, cfg.allow_draft,
                           Path(cfg.release_root) if cfg.release_root else None,
                           without_private=cfg.shipped_release,
                           expect_manifest_sha256=cfg.expect_release_manifest)
        check_release_exclusion(rel, cfg.exclude_benchmarks)
        place_cells = (read_place_cells(cfg.place_cells_file, cfg.expect_place_cells_sha256)
                       if cfg.place_cells_file else None)
        data = data or build_data_from_release(conn, rel, store.location, size,
                                               cfg.cell_degrees, place_cells,
                                               public_trainable_only=cfg.public_trainable_only)
        if data.place_source == PLACE_NONE:
            raise ValueError("no place counts: this copy of the release has no "
                             "private.sqlite and the launch shipped no "
                             f"{PLACE_CELLS_FILE}, so classifier+month+place would silently "
                             "be classifier+month. Not training.")
        snapshot = release_snapshot(rel, data, cfg.exclude_benchmarks)
    else:
        data = data or build_data(conn, store.location, size, test_days, cfg.val_days,
                                  cfg.cell_degrees,
                                  exclusion_ids(conn, cfg.exclude_benchmarks,
                                                cfg.exclude_ids_file))
        snapshot = label_snapshot(conn, data, cfg.exclude_benchmarks)
    log(format_snapshot(snapshot))
    if cfg.expect_labels_hash and snapshot["labels_hash"] != cfg.expect_labels_hash:
        raise ValueError(f"the labels here (hash {snapshot['labels_hash']}) are not the ones "
                         f"the launch recorded ({cfg.expect_labels_hash}): not training")
    if rel is not None:
        require_allowed_sources(snapshot["record_sources"], log=log)
    n_classes = len(data.classes)
    log(f"[{name}] {cfg.preset}: {len(data.train):,} training photos of "
        f"{len(data.train_records):,} records, {n_classes:,} species; validation "
        f"{len(data.val):,} photos of {len(data.val_records):,} records; trained through "
        f"{data.trained_through} (gradients through {data.train_through})")
    import torch

    from ...finetune import batches_with_fallback, register
    from ...screening import Stopped
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = (model_factory or (lambda: create_model(preset, n_classes)))()
    model.to(device).train()
    checkpointing = cfg.grad_checkpointing
    if checkpointing is None:
        checkpointing = (device == "cuda" and torch.cuda.get_device_properties(0).total_memory
                         < 12 * 2**30)
    if checkpointing and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
    amp_dtype = None
    if cfg.amp and device == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                          weight_decay=cfg.weight_decay)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=cfg.plateau_factor, patience=cfg.plateau_patience,
        eps=cfg.plateau_eps)
    loss_fn = make_loss(preset.loss, data.class_photos, cfg)
    accum = max(1, cfg.effective_batch // cfg.micro_batch)
    size_px = preset.image_size
    val_items = data.val
    if cfg.val_max_photos and len(val_items) > cfg.val_max_photos:
        keep = np.random.default_rng(cfg.seed).choice(len(val_items), cfg.val_max_photos,
                                                      replace=False)
        val_items = [val_items[i] for i in sorted(keep)]
    cache = None
    train_location = store.location
    if cfg.cache_px:
        from ... import config as _config
        where = Path(cfg.cache_dir) if cfg.cache_dir else (_config.DATA_DIR / "picek-cache"
                                                           / str(cfg.cache_px))
        log(f"[{name}] caching {len(data.train):,} training photos at {cfg.cache_px} px "
            f"in {where}")
        cache = cache_photos(store, data.train, where, cfg.cache_px, log=log)
        log(f"[{name}] photo cache: {json.dumps(cache)}")
        train_location = str(where)
    train_set, val_set = make_datasets(train_location, store.location, data, preset, val_items)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=cfg.micro_batch * 2, num_workers=cfg.workers, collate_fn=collate,
        pin_memory=device == "cuda")
    micro_per_epoch = len(data.train) // cfg.micro_batch
    steps_per_epoch = max(1, micro_per_epoch // accum)
    history, best = [], {"macro_f1": -1.0}
    step, seen, train_seconds, wait_seconds = 0, 0, 0.0, 0.0
    stopped_early = False
    fallback = {"at": None}
    rule = cfg.epoch_rule
    blocks: list[dict] = []
    start_epoch, resumed_from = 0, None
    stop_reason = None
    epoch_seconds: list[float] = []      # this attempt's epochs, validation included
    ckpt_path = out_dir / f"{name}{CHECKPOINT_SUFFIX}"
    ckpt_identity = {"name": name, "preset": cfg.preset, "labels_hash": snapshot["labels_hash"],
                     "epoch_rule": asdict(rule) if rule else None, "seed": cfg.seed}
    if resume_from is not None and Path(resume_from).is_file():
        if rule is None:
            raise ValueError("a block checkpoint resumes only a run with an epoch rule")
        ck = torch.load(resume_from, map_location="cpu", weights_only=True)
        if ck.get("identity") != ckpt_identity:
            raise ValueError(f"{resume_from} is not a checkpoint of this run "
                             f"({ck.get('identity')} vs {ckpt_identity}): not resuming")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        plateau.load_state_dict(ck["plateau"])
        scaler.load_state_dict(ck["scaler"])
        best = {"macro_f1": ck["best_macro_f1"], "epoch": ck["best_epoch"],
                "state": ck["best_state"]}            # its validation pass is redone at the end
        history, blocks = list(ck["history"]), list(ck["blocks"])
        step, seen = int(ck["step"]), int(ck["seen"])
        start_epoch = resumed_from = int(ck["epoch"])
        log(f"[{name}] RESUMED from the block checkpoint after epoch {start_epoch} (best "
            f"val macro-F1 {best['macro_f1']:.4f} at epoch {best['epoch']})")
        if blocks and blocks[-1]["decision"] == "stop":
            stop_reason = blocks[-1]["reason"]
            start_epoch = cfg.epochs                # the rule had already stopped it

    def save_block(epoch: int) -> None:
        """The block checkpoint (written whole, then renamed), then on_block."""
        tmp = ckpt_path.with_suffix(".tmp")
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"identity": ckpt_identity, "epoch": epoch, "step": step, "seen": seen,
                    "model": model.state_dict(), "optimizer": opt.state_dict(),
                    "plateau": plateau.state_dict(), "scaler": scaler.state_dict(),
                    "best_state": best["state"], "best_epoch": best["epoch"],
                    "best_macro_f1": best["macro_f1"], "history": history,
                    "blocks": blocks}, tmp)
        tmp.replace(ckpt_path)
        if on_block is not None:
            on_block(ckpt_path, {"epochs_done": epoch, "best_epoch": best["epoch"],
                                 "best_val_macro_f1": best["macro_f1"], "blocks": blocks,
                                 "history": history})
    for epoch in range(start_epoch, cfg.epochs):
        order = torch.randperm(len(data.train), generator=torch.Generator().manual_seed(
            cfg.seed + epoch)).tolist()
        micro_steps = steps_per_epoch * accum

        def make_loader(workers: int, left: int, seed: int, order=order, micro_steps=micro_steps):
            done = micro_steps - left
            idx = order[done * cfg.micro_batch: micro_steps * cfg.micro_batch]
            return torch.utils.data.DataLoader(
                torch.utils.data.Subset(train_set, idx), batch_size=cfg.micro_batch,
                shuffle=False, num_workers=workers, collate_fn=collate, drop_last=True,
                pin_memory=device == "cuda", persistent_workers=False)

        micro = 0
        losses = []
        t_epoch = time.monotonic()
        t_wait = time.monotonic()

        def fell_back(at, error):
            fallback["at"] = (epoch, at)
            log(f"[{name}] loader workers crashed ({error}); carrying on without workers")
        for batch in batches_with_fallback(make_loader, cfg.workers, micro_steps, cfg.seed,
                                           lambda: micro, fell_back):
            wait_seconds += time.monotonic() - t_wait
            if should_stop is not None and should_stop():
                kept = (f"the block checkpoint after epoch {blocks[-1]['epoch']} is kept; "
                        "resume the run to carry on from it" if blocks else "not saved")
                raise Stopped(f"stopped in epoch {epoch + 1} at step {step:,} (time limit); "
                              + kept)
            if batch is None:
                micro += 1
                t_wait = time.monotonic()
                continue
            x, y, _ = batch
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device, dtype=amp_dtype or torch.float32,
                                enabled=amp_dtype is not None):
                logits = model(x)
            loss = loss_fn(logits, y) / accum
            scaler.scale(loss).backward()
            losses.append(float(loss.detach()) * accum)
            seen += len(x)
            micro += 1
            if micro % accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    rate = seen / max(time.monotonic() - started, 1e-6)
                    log(f"[{name}] epoch {epoch + 1}, step {step:,}, loss "
                        f"{np.mean(losses[-20 * accum:]):.3f}, {rate:.1f} photos/s overall")
                if cfg.max_steps and step >= cfg.max_steps:
                    stopped_early = True
                    break
            t_wait = time.monotonic()
        if device == "cuda":
            torch.cuda.synchronize()
        train_seconds += time.monotonic() - t_epoch
        v = run_validation(model, val_loader, device, n_classes, loss_fn, amp_dtype)
        if v["loss"] is not None:
            plateau.step(v["loss"])
        row = {"epoch": epoch + 1, "steps": step, "train_loss": round(float(np.mean(losses)), 4)
               if losses else None, "val_loss": v["loss"], "val_macro_f1": v["macro_f1"],
               "val_top1": v["top1"], "val_photos": v["n"], "lr": opt.param_groups[0]["lr"]}
        history.append(row)
        log(f"[{name}] epoch {epoch + 1}/{cfg.epochs}: {json.dumps(row)}")
        if v["macro_f1"] is not None and v["macro_f1"] > best["macro_f1"] or not best.get("state"):
            best = {"macro_f1": v["macro_f1"] if v["macro_f1"] is not None else -1.0,
                    "epoch": epoch + 1, "val": v,
                    "state": {k: t.detach().to("cpu", copy=True)
                              for k, t in model.state_dict().items()}}
        epoch_seconds.append(time.monotonic() - t_epoch)
        if stopped_early:
            stop_reason = f"stopped at max_steps ({cfg.max_steps:,} optimizer steps)"
            break
        decision = block_decision(rule, [h["val_macro_f1"] for h in history],
                                  epoch + 1) if rule else None
        if decision is not None:
            blocks.append(decision)
            log(f"[{name}] epoch rule after epoch {epoch + 1}: {json.dumps(decision)}")
            save_block(epoch + 1)
            if decision["decision"] == "stop":
                stop_reason = decision["reason"]
                break
            if time_left is not None:
                needed = rule.block * float(np.mean(epoch_seconds))
                left = time_left()
                if left is not None and left < needed:
                    raise Stopped(f"stopped at the block boundary after epoch {epoch + 1} "
                                  f"(time limit: {left / 3600:.1f} h left, a block needs "
                                  f"~{needed / 3600:.1f} h); the block checkpoint is kept: "
                                  "resume the run to carry on from it")
    if stop_reason is None:
        stop_reason = f"fixed length: {cfg.epochs} epochs"
    model.load_state_dict(best["state"])
    v = best.get("val")
    if v is None:            # the best epoch came before a resume: score it again
        v = run_validation(model, val_loader, device, n_classes, loss_fn, amp_dtype)
    groups = np.array([val_items[i][3] for i in v["rows"]]) if len(v["rows"]) else None
    temperature = fit_temperature(v["logits"], v["labels"], groups)
    scale = float(20.0 * np.median(v["norms"])) if len(v["norms"]) else 20.0
    head = model.get_classifier()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best["state"], out_dir / f"{name}.pt")
    save_head(out_dir / f"{name}.classifier.npz", data, head.weight.detach().float().cpu().numpy(),
              head.bias.detach().float().cpu().numpy(), temperature, scale, cfg.cell_degrees)
    with open(out_dir / f"{name}.records.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["observation_id", "split"])
        w.writerows([(o, "train") for o in data.train_records]
                     + [(o, "validation") for o in data.val_records])
    elapsed = time.monotonic() - started
    meta = {"name": name, "kind": KIND, "base": preset.timm_name,
            "base_spec": f"timm:{preset.timm_name}", "preset": cfg.preset,
            "preset_note": preset.note, "weights": f"{name}.pt",
            "classifier": f"{name}.classifier.npz", "records_file": f"{name}.records.csv",
            "image_size": size_px, "feature_dim": int(head.weight.shape[1]),
            "trained_through": data.trained_through, "train_through": data.train_through,
            "dataset_release": data.release,
            "trained_splits": ["train"] if data.release else None,
            "validation": {"days": None if data.release else cfg.val_days,
                           "split": "val (the release's)" if data.release else None,
                           "records": len(data.val_records),
                           "photos": len(data.val), "photos_scored": len(val_items),
                           "records_of_species_not_trained": data.val_records_unknown_species},
            "test_days": test_days, "records": len(data.train_records),
            "photos": len(data.train), "species": n_classes, "size": size,
            "excluded_benchmark_records": data.excluded_benchmark_records,
            "one_word_records_left_out": data.one_word_records,
            "steps": step, "photos_seen": seen, "epochs_run": len(history),
            "best_epoch": best.get("epoch"), "best_val_macro_f1": best["macro_f1"],
            "temperature": temperature, "feature_scale": scale, "history": history,
            "epoch_rule": asdict(rule) if rule else None, "blocks": blocks,
            "stop_reason": stop_reason, "resumed_from_epoch": resumed_from,
            "train_photos_per_second": round(seen / train_seconds, 2) if train_seconds else None,
            "loader_wait_share": round(wait_seconds / train_seconds, 3) if train_seconds else None,
            "stopped_at_max_steps": stopped_early, "loader_fallback": fallback["at"],
            "photo_cache": cache, "label_snapshot": snapshot,
            "config": asdict(cfg), "device": device, "grad_checkpointing": bool(checkpointing),
            "minutes": round(elapsed / 60, 1),
            "code_version": __import__("mycomap_vision.config", fromlist=["x"]).code_version(),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    register(conn, meta)
    log(f"[{name}] saved to {out_dir}: best epoch {meta['best_epoch']}, val macro-F1 "
        f"{meta['best_val_macro_f1']}, T {temperature}, {meta['minutes']} min")
    return meta


def save_head(path: Path, data: PicekData, weight, bias, temperature: float, scale: float,
              cell_degrees: float) -> None:
    cell_ids = sorted(data.cells)
    flat = [(c, k, n) for c in cell_ids for k, n in sorted(data.cells[c].items())]
    np.savez_compressed(
        path, classes=np.array(data.classes, dtype=str), genus=np.array(data.genus_of, dtype=str),
        weight=np.asarray(weight, np.float32), bias=np.asarray(bias, np.float32),
        temperature=np.float64(temperature), feature_scale=np.float64(scale),
        class_photos=data.class_photos.astype(np.int64), month_counts=data.month_counts,
        cell_degrees=np.float64(cell_degrees),
        cell_id=np.array([c for c, _, _ in flat], dtype=np.int64),
        cell_class=np.array([k for _, k, _ in flat], dtype=np.int64),
        cell_count=np.array([n for _, _, n in flat], dtype=np.int64))


# --- using it -----------------------------------------------------------------------------

@dataclass
class Head:
    classes: list[str]
    genus: list[str]
    weight: np.ndarray
    bias: np.ndarray
    temperature: float
    feature_scale: float
    class_photos: np.ndarray
    month_counts: np.ndarray
    cell_degrees: float
    cells: dict[int, tuple[np.ndarray, np.ndarray]]      # cell -> (classes, counts)
    genus_idx: np.ndarray = field(repr=False, default=None)     # each class's genus number
    place_totals: np.ndarray = field(repr=False, default=None)  # photos with a place, per class

    def __post_init__(self):
        self.genus_idx = np.unique(np.asarray(self.genus, dtype=str), return_inverse=True)[1]
        totals = np.zeros(len(self.classes))
        for cls, cnt in self.cells.values():
            np.add.at(totals, cls, cnt)
        self.place_totals = totals

    def log_probs(self, vectors: np.ndarray) -> np.ndarray:
        """An observation's log-probabilities over the classes: each photo's logits / T,
        averaged over the photos, log-softmax."""
        from ...methods import log_softmax
        f = decode_features(vectors, self.feature_scale)
        z = (f @ self.weight.T + self.bias) / self.temperature
        return log_softmax(z.mean(axis=0))


def is_classifier(name: str, root: Path | None = None) -> bool:
    path = (root or models_dir()) / f"{name}.json"
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("kind") == KIND
    except (OSError, ValueError):
        return False


def load_head(name: str, root: Path | None = None) -> Head:
    root = root or models_dir()
    path = root / f"{name}.classifier.npz"
    if not path.is_file():
        raise ValueError(f"{name} is not a classifier model (no {path.name} in {root}): the "
                         "classifier methods need a model trained with mv picek-train")
    return _load_head(str(path), path.stat().st_mtime_ns)


@lru_cache(maxsize=4)
def _load_head(path: str, _mtime: int) -> Head:
    with np.load(path) as z:
        cells: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        ids, cls, cnt = z["cell_id"], z["cell_class"], z["cell_count"]
        if len(ids):
            starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
            ends = np.r_[starts[1:], len(ids)]
            for s, e in zip(starts, ends):
                cells[int(ids[s])] = (cls[s:e], cnt[s:e])
        return Head(z["classes"].tolist(), z["genus"].tolist(), z["weight"], z["bias"],
                    float(z["temperature"]), float(z["feature_scale"]), z["class_photos"],
                    z["month_counts"], float(z["cell_degrees"]), cells)


@dataclass
class MetadataPrior:
    """DF20's metadata prior (Picek et al., WACV 2022, section 5.2), per field k:

        p(c | x, m) ∝ p(c | x) * prod_k p(c | m_k) / p(c)

    which by Bayes is p(c | x) * prod_k p(m_k | c) / p(m_k). This is the paper's formula.
    (Their released code multiplies the image probability in twice, squaring p(c | x);
    that is not what the paper describes, and is not repeated here.)

    Fields: month only (habitat and substrate, the other DF20 fields, don't exist in our
    data). `place`, a coarse lat/lon cell, is OUR extension, not their method.

    Smoothing. DF20 estimates p(m | c) from raw counts. With ~18,000 species, 43% known
    from one record, raw counts give probability 0 to every month a species was not yet
    found in and remove it outright. So by default p(m | c) is shrunk towards its genus,
    and the genus towards all fungi:
        p(m | g) = (n_gm + beta_genus * p(m)) / (n_g + beta_genus)
        p(m | c) = (n_cm + beta * p(m | g)) / (n_c + beta)
    with counts in training photos (DF20 counts its training images too). beta = 0 is their
    unsmoothed estimate (a never-seen month gets 1e-12, not 0, so scores stay finite)."""
    month: bool = True
    place: bool = False
    beta: float = 10.0
    beta_genus: float = 10.0

    def log_ratio(self, head: Head, context) -> np.ndarray:
        out = np.zeros(len(head.classes))
        if context is None or not len(head.classes):
            return out
        if self.month:
            m = month_of(getattr(context, "observed_on", None))
            if m is not None:
                out += self._ratio(head, head.month_counts.astype(np.float64), m)
        if self.place:
            cell = cell_of(getattr(context, "latitude", None), getattr(context, "longitude", None),
                           head.cell_degrees)
            if cell is not None:
                out += self._place_ratio(head, cell)
        return out

    def _shrunk(self, head: Head, n_cx: np.ndarray, n_c: np.ndarray, p_x: float) -> np.ndarray:
        """log p(x | c) / p(x) with the genus back-off, for one value x of a field."""
        gi = head.genus_idx
        n_gx = np.bincount(gi, weights=n_cx, minlength=gi.max() + 1)[gi]
        n_g = np.bincount(gi, weights=n_c, minlength=gi.max() + 1)[gi]
        gd = n_g + self.beta_genus
        p_gx = np.where(gd > 0, (n_gx + self.beta_genus * p_x) / np.maximum(gd, 1e-12), p_x)
        denom = n_c + self.beta
        p_cx = np.where(denom > 0, (n_cx + self.beta * p_gx) / np.maximum(denom, 1e-12), p_x)
        return np.log(np.maximum(p_cx, 1e-12)) - math.log(max(p_x, 1e-12))

    def _ratio(self, head: Head, counts: np.ndarray, m: int) -> np.ndarray:
        n_c = counts.sum(axis=1)
        p_m = (counts[:, m].sum() + 1.0) / (counts.sum() + 12.0)
        return self._shrunk(head, counts[:, m], n_c, p_m)

    def _place_ratio(self, head: Head, cell: int) -> np.ndarray:
        n_c = head.place_totals
        n_cx = np.zeros(len(head.classes))
        if cell in head.cells:
            cls, cnt = head.cells[cell]
            np.add.at(n_cx, cls, cnt)
        rows = int(math.ceil(180 / head.cell_degrees))
        n_cells = rows * int(math.ceil(360 / head.cell_degrees))
        p_x = (n_cx.sum() + 1.0) / (n_c.sum() + n_cells)
        return self._shrunk(head, n_cx, n_c, p_x)


class Classifier:
    """The replicated classifier as a Vision method: an observation's log-probabilities
    (Head.log_probs), times the metadata prior when there is one, over the index's
    species. A species the classifier has no class for (only in records after its
    cutoff, or only in its validation slice) gets probability 1e-12; one-word groups the
    same. Its scores are log-probabilities, so its own confidence temperature is 1."""
    name = "classifier"
    prior: MetadataPrior | None = None
    needs_context = False
    trainable = False                    # nothing to fit or cache: the head comes trained
    confidence_temperature = 1.0

    @staticmethod
    def ready() -> bool:
        """Not offered on the website (methods.method_ready): the site offers each method
        with every backbone, and this one works only with its own trained model. It is a
        baseline for the paper, scored by mv compare and mv heldout predict."""
        return False

    def for_backbone(self, backbone: str, root: Path | None = None) -> None:
        self.backbone = backbone
        self.head = load_head(backbone, root)

    def fit(self, vectors, index, records=None) -> None:
        if not hasattr(self, "head"):
            raise ValueError(f"{self.name} needs its model: call for_backbone first "
                             "(evaluate.make_method does)")
        pos = {c: i for i, c in enumerate(self.head.classes)}
        species = index.labels["species"]
        group_species = [species[i] if i >= 0 else None
                         for i in np.asarray(index.label_of["species"]).tolist()]
        self.cols = np.array([pos.get(s, -1) if s else -1 for s in group_species], dtype=np.int64)
        self.n_groups = len(group_species)

    def species_scores(self, query: np.ndarray, context=None) -> np.ndarray:
        from ...methods import log_softmax
        lp = self.head.log_probs(query)
        if self.prior is not None and context is not None:
            lp = log_softmax(lp + self.prior.log_ratio(self.head, context))
        out = np.full(self.n_groups, LOG_FLOOR, dtype=np.float32)
        has = self.cols >= 0
        out[has] = lp[self.cols[has]]
        return out


class ClassifierMonth(Classifier):
    """+ DF20's metadata prior on month (smoothed; MetadataPrior)."""
    name = "classifier+month"
    prior = MetadataPrior(month=True)
    needs_context = True


class ClassifierMonthRaw(Classifier):
    """+ DF20's metadata prior on month exactly as they estimate it: raw counts, no
    smoothing (a month a species was never found in all but removes it)."""
    name = "classifier+month-raw"
    prior = MetadataPrior(month=True, beta=0.0, beta_genus=0.0)
    needs_context = True


class ClassifierMonthPlace(Classifier):
    """+ month (DF20) + a coarse place prior: OUR extension, not their method."""
    name = "classifier+month+place"
    prior = MetadataPrior(month=True, place=True)
    needs_context = True


CLASSIFIER_METHODS = (Classifier, ClassifierMonth, ClassifierMonthRaw, ClassifierMonthPlace)


# --- the backbone -------------------------------------------------------------------------

class ClassifierBackbone:
    """A trained classifier as a backbone: encode() gives its pre-logit features in the
    length-keeping form (encode_features), so the classifier methods apply its head
    exactly; prepare() is its evaluation transform (resize to size x size)."""

    def __init__(self, name: str, model_name: str, root: Path | None = None, model_factory=None):
        import torch
        root = root or models_dir()
        meta = json.loads((root / f"{model_name}.json").read_text(encoding="utf-8"))
        if meta.get("kind") != KIND:
            raise ValueError(f"{model_name} is not a classifier model")
        self.torch, self.name, self.meta = torch, name, meta
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        head = load_head(model_name, root)
        preset = PRESETS[meta["preset"]]
        model = (model_factory or (lambda: create_model(preset, len(head.classes),
                                                        pretrained=False)))()
        state = torch.load(root / meta["weights"], map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        self.model = model.eval().to(self.device)
        self.scale = head.feature_scale
        self.dim = int(meta["feature_dim"]) + 1
        self.transform = eval_transform(int(meta["image_size"]))

    def prepare(self, image):
        return self.transform(image)

    def encode(self, items) -> np.ndarray:
        torch = self.torch
        x = torch.stack([i if isinstance(i, torch.Tensor) else self.transform(i)
                         for i in items]).to(self.device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.float16,
                                                    enabled=self.device == "cuda"):
            f, _ = features_and_logits(self.model, x)
        return encode_features(f.float().cpu().numpy(), self.scale)


# --- measuring speed ----------------------------------------------------------------------

def bench_loader(store, items, preset: str, workers: int, photos: int = 512,
                 micro_batch: int = 16) -> dict:
    """Training photos per second the data loader alone delivers (read, decode, augment) with
    `workers` processes: the CPU side of the training speed."""
    import torch
    p = PRESETS[preset]
    ds = Photos(store.location, items[:photos], train_transform(p.augment, p.image_size),
                draft=p.image_size)
    loader = torch.utils.data.DataLoader(ds, batch_size=micro_batch, num_workers=workers,
                                         collate_fn=collate, shuffle=False)
    it = iter(loader)
    first = next(it)                     # workers started: not counted
    n, t0 = 0, time.monotonic()
    for b in it:
        if b is not None:
            n += len(b[0])
    secs = time.monotonic() - t0
    return {"workers": workers, "photos": n, "seconds": round(secs, 2),
            "photos_per_second": round(n / secs, 1) if secs else None,
            "first_batch": first is not None}


def bench_gpu(preset: str, n_classes: int, micro_batch: int = 16, steps: int = 20,
              grad_checkpointing: bool = True, eval_batch: int = 64) -> dict:
    """Training and evaluation photos per second on the GPU alone (random tensors, so the
    loader isn't counted): forward + backward + SGD step at `micro_batch`, mixed precision,
    with or without gradient checkpointing; and inference at `eval_batch`."""
    import torch
    p = PRESETS[preset]
    dev = "cuda"
    model = create_model(p, n_classes, pretrained=False).to(dev).train()
    if grad_checkpointing:
        model.set_grad_checkpointing(True)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    x = torch.randn(micro_batch, 3, p.image_size, p.image_size, device=dev)
    y = torch.randint(0, n_classes, (micro_batch,), device=dev)
    loss_fn = make_loss(p.loss, np.ones(n_classes), PicekConfig(preset=preset))
    torch.cuda.reset_peak_memory_stats()

    def step():
        with torch.autocast(dev, dtype=dtype):
            z = model(x)
        loss_fn(z, y).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    t0 = time.monotonic()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    train_rate = steps * micro_batch / (time.monotonic() - t0)
    peak = torch.cuda.max_memory_allocated() / 2**30
    model.eval()
    xe = torch.randn(eval_batch, 3, p.image_size, p.image_size, device=dev)
    with torch.inference_mode(), torch.autocast(dev, dtype=torch.float16):
        for _ in range(2):
            features_and_logits(model, xe)
        torch.cuda.synchronize()
        t0 = time.monotonic()
        for _ in range(5):
            features_and_logits(model, xe)
        torch.cuda.synchronize()
    eval_rate = 5 * eval_batch / (time.monotonic() - t0)
    return {"preset": preset, "classes": n_classes, "micro_batch": micro_batch,
            "grad_checkpointing": grad_checkpointing,
            "train_photos_per_second": round(train_rate, 1),
            "eval_photos_per_second": round(eval_rate, 1),
            "peak_gpu_gb": round(peak, 2), "gpu": torch.cuda.get_device_name()}


def estimate_hours(photos: int, epochs: float, train_rate: float, eval_rate: float,
                   val_photos: int = 0) -> float:
    """Wall-clock hours for `epochs` passes over `photos` at `train_rate` photos/s (the
    slower of the loader and the GPU), plus a validation pass per epoch."""
    per_epoch = photos / train_rate + (val_photos / eval_rate if val_photos else 0)
    return round(epochs * per_epoch / 3600, 1)
