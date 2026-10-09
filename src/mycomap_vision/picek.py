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


# --- presets ------------------------------------------------------------------------------

@dataclass(frozen=True)
class Preset:
    timm_name: str
    image_size: int
    loss: str                  # "seesaw" or "ce"
    lr: float
    epochs: int
    augment: str               # "vit_heavy" or "light"
    note: str


PRESETS: dict[str, Preset] = {
    # FungiTastic's best closed-set model (HF BVRA/beit_base_patch16_384.in1k_ft_fungitastic_384).
    "fungitastic-beit-b384": Preset("beit_base_patch16_384.in22k_ft_in22k_in1k", 384, "seesaw",
                                    0.01, 50, "vit_heavy",
                                    "FungiTastic best: BEiT-B/16 384, Seesaw, SGD 0.01, heavy aug"),
    # The same at 224 px (FungiTastic reports 224 models too): the cheaper variant.
    "fungitastic-beit-b224": Preset("beit_base_patch16_224.in22k_ft_in22k_in1k", 224, "seesaw",
                                    0.01, 50, "vit_heavy",
                                    "FungiTastic recipe at 224 px (BEiT-B/16 224)"),
    # FungiTastic's ViT-B variant: plain cross-entropy.
    "vit-b384-ce": Preset("vit_base_patch16_384.augreg_in21k_ft_in1k", 384, "ce", 0.001, 50,
                          "vit_heavy", "ViT-B/16 384, cross-entropy, SGD 0.001, heavy aug"),
    # DF20's production model (Danish Fungi / Atlas of Danish Fungi "FungiVision").
    "df20-vit-l384": Preset("vit_large_patch16_384.augreg_in21k_ft_in1k", 384, "ce", 0.005, 100,
                            "light", "DF20 production: ViT-L/16 384, cross-entropy, light aug"),
}


@dataclass
class PicekConfig:
    preset: str = "fungitastic-beit-b384"
    epochs: int | None = None          # None: the preset's
    lr: float | None = None            # None: the preset's
    effective_batch: int = 256
    micro_batch: int = 16              # photos per forward pass; accumulated to effective_batch
    momentum: float = 0.9
    weight_decay: float = 0.0
    plateau_factor: float = 0.9
    plateau_patience: int = 1
    plateau_eps: float = 1e-6
    seesaw_p: float = 0.8
    seesaw_q: float = 2.0
    val_days: int = 28                 # the validation slice: the last days before the cutoff
    val_max_photos: int | None = None  # cap the validation pass (smoke tests)
    # Gradient checkpointing: None = on only when the GPU has under 12 GB (it costs ~20% of
    # the speed: 44 vs 55 photos/s for BEiT-B 384 on the laptop's 8 GB RTX 4070, where
    # micro-batch 16 without it peaks at 4.5 GB and 32 spills over 8 GB).
    grad_checkpointing: bool | None = None
    amp: bool = True
    workers: int = 6
    seed: int = 0
    max_steps: int | None = None       # cap on optimizer steps (smoke tests)
    cell_degrees: float = 4.0          # the place prior's grid (our extension)
    # Training photos pre-resized once to this shorter side on local disk (cache_photos),
    # so 4 vCPUs can keep the GPU fed; None (the default) reads the originals. Validation
    # always reads the originals, like the embedding at test time.
    cache_px: int | None = None
    cache_dir: str | None = None       # default: <data>/picek-cache/<px>
    # Which benchmark records to leave out (EXCLUDE_MODES). "match": exactly what a Vision
    # fine-tune leaves out (sealed benchmarks, evaluate.load_records), so the two train on
    # the same records; "all": also every record of every benchmark, released or not.
    exclude_benchmarks: str = "match"
    exclude_ids_file: str | None = None   # "all" on a trainer instance: the ids, shipped
    expect_labels_hash: str | None = None # refuse to train on other labels than at launch

    def resolved(self) -> "PicekConfig":
        p = PRESETS[self.preset]
        return replace(self, epochs=self.epochs or p.epochs, lr=self.lr or p.lr)


def parse_spec(spec: str) -> tuple[str, int]:
    """'fungitastic-beit-b384@15' -> (preset, 15 epochs); no '@': the preset's epochs.
    A third part, 'preset@15@440', is the photo cache (cache_of)."""
    preset, _, rest = spec.partition("@")
    epochs, _, cache = rest.partition("@")
    if preset not in PRESETS:
        raise ValueError(f"unknown Picek preset {preset!r}: {', '.join(sorted(PRESETS))}")
    if epochs and not (epochs.isdigit() and int(epochs) > 0):
        raise ValueError(f"bad epoch count in {spec!r}: use preset@epochs, e.g. {preset}@15")
    if cache and not (cache.isdigit() and int(cache) >= PRESETS[preset].image_size):
        raise ValueError(f"bad photo cache size in {spec!r}: preset@epochs@px, px at least "
                         f"the input size ({PRESETS[preset].image_size}), e.g. {preset}@15@440")
    return preset, int(epochs) if epochs else PRESETS[preset].epochs


def cache_of(spec: str) -> int | None:
    """The photo cache's shorter side a spec asks for ('preset@15@440' -> 440), else None."""
    parse_spec(spec)
    parts = spec.split("@")
    return int(parts[2]) if len(parts) > 2 and parts[2] else None


# --- the training data --------------------------------------------------------------------

def month_of(day: str | None) -> int | None:
    """0-11, or None for no date (and the 1970-01-01 placeholder, dates.py)."""
    from .dates import real_date
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


EXCLUDE_MODES = {
    "match": "only sealed benchmarks (as every Vision model; evaluate.load_records)",
    "all": "also every record of every held-out benchmark, sealed, released or not",
}


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
    from . import evaluate
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


def labels_hash(data: PicekData) -> str:
    """Which labelling a run trained on: the hash of every (record, species label) of its
    training and validation records, and of the cutoff."""
    import hashlib
    h = hashlib.sha256(f"{data.trained_through}|{data.train_through}\n".encode())
    h.update("\n".join(f"{o}\t{s}" for o, s in sorted(data.record_labels)).encode())
    return h.hexdigest()[:16]


def label_snapshot(conn, data: PicekData, exclude_mode: str = "match") -> dict:
    """What a run's labels were, to prove later which labelling it used: counts, the
    share of provisional names, what was left out and why, the known label problems the
    manifest carries (label conflicts; names that wait for a person, names.py), and
    labels_hash."""
    from . import names
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
    }


def format_snapshot(snap: dict) -> str:
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
    ])


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

MEAN = STD = (0.5, 0.5, 0.5)


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

        from .storage import open_store
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

from .metrics import macro_f1  # noqa: E402  (shared with comparisons and the held-out report)


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
    from .finetune import models_dir as md
    return md()


def train(conn, store, size: str, name: str, cfg: PicekConfig | None = None,
          test_days: int = 28, out_dir: Path | None = None, model_factory=None, log=print,
          should_stop=None, data: PicekData | None = None) -> dict:
    """Train the preset's classifier on the reference records (build_data), keep the epoch
    with the best validation macro-F1, fit its temperature, save and register it. Returns
    the metadata. `should_stop()` is asked between steps: when it says stop, the best
    epoch so far is NOT saved (screening.Stopped), as for a fine-tune."""
    import torch

    from .finetune import batches_with_fallback, register
    from .screening import Stopped
    cfg = (cfg or PicekConfig()).resolved()
    preset = PRESETS[cfg.preset]
    out_dir = out_dir or models_dir()
    started = time.monotonic()
    data = data or build_data(conn, store.location, size, test_days, cfg.val_days,
                              cfg.cell_degrees,
                              exclusion_ids(conn, cfg.exclude_benchmarks, cfg.exclude_ids_file))
    snapshot = label_snapshot(conn, data, cfg.exclude_benchmarks)
    log(format_snapshot(snapshot))
    if cfg.expect_labels_hash and snapshot["labels_hash"] != cfg.expect_labels_hash:
        raise ValueError(f"the labels here (hash {snapshot['labels_hash']}) are not the ones "
                         f"the launch recorded ({cfg.expect_labels_hash}): not training")
    n_classes = len(data.classes)
    log(f"[{name}] {cfg.preset}: {len(data.train):,} training photos of "
        f"{len(data.train_records):,} records, {n_classes:,} species; validation "
        f"{len(data.val):,} photos of {len(data.val_records):,} records; trained through "
        f"{data.trained_through} (gradients through {data.train_through})")
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
        from . import config as _config
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
    for epoch in range(cfg.epochs):
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
                raise Stopped(f"stopped in epoch {epoch + 1} at step {step:,} (time limit); "
                              "not saved")
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
        if stopped_early:
            break
    model.load_state_dict(best["state"])
    v = best["val"]
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
            "validation": {"days": cfg.val_days, "records": len(data.val_records),
                           "photos": len(data.val), "photos_scored": len(val_items),
                           "records_of_species_not_trained": data.val_records_unknown_species},
            "test_days": test_days, "records": len(data.train_records),
            "photos": len(data.train), "species": n_classes, "size": size,
            "excluded_benchmark_records": data.excluded_benchmark_records,
            "one_word_records_left_out": data.one_word_records,
            "steps": step, "photos_seen": seen, "epochs_run": len(history),
            "best_epoch": best.get("epoch"), "best_val_macro_f1": best["macro_f1"],
            "temperature": temperature, "feature_scale": scale, "history": history,
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
        from .methods import log_softmax
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
        from .methods import log_softmax
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
