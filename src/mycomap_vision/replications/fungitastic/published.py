"""Published fungi classifiers as external, zero-retraining baselines (Picek et al., BVRA).

Steve (2026-10-09): the Danish Fungi (DF20) and FungiTastic models go beside iNat's
computer vision as outside baselines in the paper. They are run as their authors ship
them, never retrained:

    fungitastic-beit-b384  BVRA/beit_base_patch16_384.in1k_ft_fungitastic_384  2,829 classes
    fungitastic-vit-b384   BVRA/vit_base_patch16_384.in1k_ft_fungitastic_384   2,829 classes
    df20-vit-l384          BVRA/vit_large_patch16_384.ft_df20_384              1,604 classes

Weights and their config come from the public Hugging Face repos (timm checkpoints),
stored under <data>/external/bvra/<repo>/ (git-ignored). Licence: CC BY-NC 4.0, weights
and datasets alike: research use, never inside a commercial product.

No label file ships with the weights. The class index is rebuilt from each dataset's
public training metadata, the way the authors' loaders build it (one id per class, the
species of every training row of that id): FungiTastic-Train.csv `category_id`
(BohemianVRA/FungiTastic dataset/fungi.py), DanishFungi2024-train.csv `class_id` (the
"DF20_FIX" set the DF20 Production model trained on, 266,273 rows; DF20-train_metadata_
PROD-2.csv gives the same map). A class is the name the Danish records were made under
(`scientificName`, author stripped: 'Gliophorus perplexus'); GBIF's accepted name for it
(`species`: 'Gliophorus psittacinus') is kept beside it. `build_labels` refuses a map
that is not one name per id, ids 0..N-1, N as the checkpoint's config says.

Their names are read through Vision's (`vision_names`): a class takes the Vision label of
its own name when Vision knows it, else of the accepted name, else its own name as
written; genus is the label's first word and family Vision's (iNat's) for that genus,
else the dataset's. Classes that land on one Vision name are one candidate (their
probabilities summed), so the model is judged in Vision's taxonomy like every other.

Observation level follows the authors' rule: each photo is resized to 384 x 384 (no
crop), normalised with mean = std = 0.5, the logits of the record's photos are averaged
and a softmax gives the record's answer (temperature 1: no temperature is fitted, since
fitting it on dev would tune the baseline on our test labels). Genus and family
candidates sum the probabilities of their classes. Each photo's own top-1 is kept too,
for their per-image metric. Photo only: their metadata prior uses Danish habitat and
substrate, which our records do not have in their vocabulary.

    mv external labels      rebuild and check the class maps from the metadata CSVs
    mv external coverage    how much of a benchmark split and of Vision's North American
                            records each model can name at all (read-only)
    mv external predict     answers on a held-out split, written as JSONL (no manifest
                            writes); `mv heldout import-external` stores them
    mv external baseline    a saved scoreboard comparison's test records (eval_runs rows,
                            with the standard top 1/3/5/10 block)
    mv external report      the agreed protocol tables (published_report.py)
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from ... import config, names
# The checkpoints and protocol constants live in presets.py; read here as published.X too.
from .presets import (LICENCE, METHOD, MODELS, PREFIX, TOP_K, ExternalModel,  # noqa: F401
                      external_dir)

def model_for(name: str) -> ExternalModel:
    """A model by short name, backbone ('external:df20-vit-l384') or Hugging Face id."""
    key = name[len(PREFIX):] if name.startswith(PREFIX) else name
    key = key.removeprefix("hf-hub:")
    for m in MODELS.values():
        if key in (m.short, m.hf_id, m.hf_id.split("/", 1)[1]):
            return m
    raise ValueError(f"unknown external model {name!r}: {', '.join(MODELS)}")


def is_external(backbone: str | None) -> bool:
    return bool(backbone) and backbone.startswith(PREFIX)


# --- the class map ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClassLabel:
    class_id: int
    name: str                    # the training rows' scientificName, author stripped
    accepted: str                # GBIF's accepted species for it (the `species` column)
    family: str
    images: int


_RANK_WORDS = {"var.", "subsp.", "ssp.", "f.", "forma"}
_EPITHET = re.compile(r"[a-z][a-z-]+\Z")


def canonical_name(scientific: str, fallback: str = "") -> str:
    """'Gliophorus perplexus (A.H.Sm. & Hesler) Kovalenko' -> 'Gliophorus perplexus';
    'Amanita muscaria var. formosa Pers.' -> 'Amanita muscaria var. formosa'. A name that
    doesn't read as Genus epithet gives `fallback`."""
    words = (scientific or "").split()
    if len(words) < 2 or not re.fullmatch(r"[A-Z][a-z-]+", words[0]) \
            or not _EPITHET.match(words[1]):
        return (fallback or "").strip()
    out = words[:2]
    if len(words) >= 4 and words[2] in _RANK_WORDS and _EPITHET.match(words[3]):
        out += ["subsp." if words[2] == "ssp." else
                "f." if words[2] == "forma" else words[2], words[3]]
    return " ".join(out)


def build_labels(csv_path: Path, id_column: str, num_classes: int) -> list[ClassLabel]:
    """The class index from a training metadata CSV: per id, the one name its rows carry.
    Refused unless every id 0..num_classes-1 is there with exactly one name."""
    per_id: dict[int, Counter] = defaultdict(Counter)
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw = (row.get(id_column) or "").strip()
            if raw in ("", "-1"):           # FungiTastic's open-set "unknown": not a class
                continue
            cid = int(float(raw))
            name = canonical_name(row.get("scientificName") or "", row.get("species") or "")
            per_id[cid][(name, (row.get("species") or "").strip(),
                         (row.get("family") or "").strip())] += 1
    ids = sorted(per_id)
    if ids != list(range(num_classes)):
        missing = sorted(set(range(num_classes)) - set(ids))[:5]
        extra = sorted(set(ids) - set(range(num_classes)))[:5]
        raise ValueError(f"{Path(csv_path).name}: {len(ids):,} class ids, the checkpoint has "
                         f"{num_classes:,} (missing {missing}, unexpected {extra}); not this "
                         "model's training metadata")
    out = []
    for cid in ids:
        by_name = Counter()
        for (name, _acc, _fam), n in per_id[cid].items():
            by_name[name] += n
        if len(by_name) != 1:
            raise ValueError(f"class {cid} has {len(by_name)} names ({list(by_name)[:3]}): "
                             "the map is ambiguous")
        (name, accepted, family), _n = per_id[cid].most_common(1)[0]
        out.append(ClassLabel(cid, name, accepted, family, sum(per_id[cid].values())))
    return out


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def labels_path(model: ExternalModel, root: Path | None = None) -> Path:
    return model.folder(root) / "labels.json"


def write_labels(model: ExternalModel, root: Path | None = None) -> dict:
    """Build the map from the model's metadata CSV and save it beside the weights, with the
    CSV's hash so the map can be traced to the file it came from."""
    root = root or external_dir()
    source = root / "metadata" / model.metadata
    if not source.is_file():
        raise FileNotFoundError(f"{source} is missing: download the dataset's public metadata "
                                "(docs/PLAN.md, External baselines)")
    labels = build_labels(source, model.id_column, model.num_classes)
    body = {"model": model.hf_id, "classes": len(labels), "source": model.metadata,
            "source_sha256": file_sha256(source), "id_column": model.id_column,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "labels": [asdict(c) for c in labels]}
    path = labels_path(model, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=1), encoding="utf-8")
    renamed = sum(c.name != c.accepted for c in labels)
    return {"model": model.short, "classes": len(labels), "path": str(path),
            "source_sha256": body["source_sha256"][:16],
            "classes_whose_accepted_name_differs": renamed}


def read_labels(model: ExternalModel, root: Path | None = None) -> list[ClassLabel]:
    path = labels_path(model, root)
    if not path.is_file():
        raise FileNotFoundError(f"no class map for {model.short}: run mv external labels")
    body = json.loads(path.read_text(encoding="utf-8"))
    labels = [ClassLabel(**c) for c in body["labels"]]
    if [c.class_id for c in labels] != list(range(model.num_classes)):
        raise ValueError(f"{path} is not a {model.num_classes}-class map")
    return labels


def labels_sha(model: ExternalModel, root: Path | None = None) -> str:
    return file_sha256(labels_path(model, root))[:12]


# --- their names in Vision's ------------------------------------------------------------------

@dataclass
class ClassNames:
    """A model's classes as Vision names them, and the candidates they make at each rank:
    `groups[rank]` the distinct names, `of[rank][class]` the group a class adds to."""
    species: list[str]
    genus: list[str]
    family: list[str]
    model_names: list[str]
    how: Counter = field(default_factory=Counter)
    accepted: list[str] = field(default_factory=list)   # GBIF's accepted name, per class
    groups: dict = field(default_factory=dict)
    of: dict = field(default_factory=dict)

    def __post_init__(self):
        for rank in ("species", "genus", "family"):
            names_ = getattr(self, rank)
            groups = sorted({n for n in names_ if n})
            pos = {n: i for i, n in enumerate(groups)}
            self.groups[rank] = groups
            self.of[rank] = np.array([pos.get(n, -1) for n in names_], dtype=np.int64)


def vision_names(labels: list[ClassLabel], labeller) -> ClassNames:
    """Each class in Vision's labels (heldout.Labeller): its own name's label when Vision
    knows it, else its accepted name's, else its own name as Vision would write it."""
    species, genus, family, how = [], [], [], Counter()
    tax = getattr(labeller, "tax", None)
    for c in labels:
        own, acc = labeller.label(c.name), labeller.label(c.accepted) if c.accepted else ""
        if own in labeller.known:
            label, why = own, "own name, known to Vision"
        elif acc and acc in labeller.known:
            label, why = acc, "accepted name, known to Vision"
        else:
            label, why = own or acc, "not a Vision name"
        how[why] += 1
        g = label.split()[0] if label else ""
        fam = ""
        if tax and g in tax.genera:
            fam = tax.genera[g].get("family") or ""
        fam = fam or labeller.genus_family.get(g, "") or c.family
        species.append(label)
        genus.append(g)
        family.append(fam)
    return ClassNames(species, genus, family, [c.name for c in labels], how,
                      [c.accepted for c in labels])


# --- one record's answer ----------------------------------------------------------------------

def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()


def record_answer(logits: np.ndarray, cn: ClassNames, top_k: int = TOP_K,
                  temperature: float = 1.0) -> dict:
    """The authors' observation rule: mean of the photos' logits, softmax, then per rank the
    top candidates (classes on one name summed), plus each photo's own top-1 class."""
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 2 or not len(logits):
        raise ValueError("one row of logits per photo")
    prob = softmax(logits.mean(axis=0) / temperature)
    out: dict = {}
    for rank in ("species", "genus", "family"):
        of = cn.of[rank]
        keep = of >= 0
        summed = np.bincount(of[keep], weights=prob[keep], minlength=len(cn.groups[rank]))
        order = np.argsort(-summed, kind="stable")[:top_k]
        out[rank] = [{"name": cn.groups[rank][i], "confidence": round(float(summed[i]), 4)}
                     for i in order]
    best = int(np.argmax(prob))
    out["top_class"] = cn.model_names[best]
    out["per_image_top1"] = [cn.species[int(np.argmax(row))] for row in logits]
    out["top_k"] = top_k
    out["temperature"] = temperature
    out["photo_only"] = True
    return out


# --- the network ------------------------------------------------------------------------------

def check_config(model: ExternalModel, root: Path | None = None) -> dict:
    """The downloaded config.json must name this model's architecture, class count and
    input size, so the map and preprocessing match the weights."""
    cfg = json.loads((model.folder(root) / "config.json").read_text(encoding="utf-8"))
    size = (cfg.get("input_size") or (cfg.get("pretrained_cfg") or {}).get("input_size"))
    problems = []
    if cfg.get("architecture") != model.architecture:
        problems.append(f"architecture {cfg.get('architecture')!r}")
    if cfg.get("num_classes") != model.num_classes:
        problems.append(f"{cfg.get('num_classes')} classes")
    if size and list(size)[-2:] != [model.input_size, model.input_size]:
        problems.append(f"input {size}")
    if problems:
        raise ValueError(f"{model.hf_id}: config.json says {', '.join(problems)}")
    return cfg


def checkpoint_id(model: ExternalModel, root: Path | None = None) -> str:
    """What a run answers from, for heldout_runs.reference_hash: the weights' and the class
    map's hashes together, so a new map or new weights are a new run, never mixed."""
    weights = model.folder(root) / "pytorch_model.bin"
    marker = weights.with_suffix(".sha256")
    if marker.is_file() and marker.stat().st_mtime >= weights.stat().st_mtime:
        sha = marker.read_text().strip()
    else:
        sha = file_sha256(weights)
        marker.write_text(sha)
    return f"{model.short}:{sha[:10]}:{labels_sha(model, root)[:8]}"


class Classifier:
    """The checkpoint, as its authors run it: Resize((384, 384)), ToTensor, Normalize(0.5)."""

    def __init__(self, model: ExternalModel, device: str | None = None,
                 root: Path | None = None):
        import timm
        import torch
        from torchvision import transforms as T
        check_config(model, root)
        self.model, self.torch = model, torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        net = timm.create_model(model.architecture, pretrained=False,
                                num_classes=model.num_classes)
        state = torch.load(model.folder(root) / "pytorch_model.bin", map_location="cpu",
                           weights_only=True)
        net.load_state_dict(state, strict=True)
        self.net = net.eval().to(self.device)
        self.transform = T.Compose([T.Resize((model.input_size, model.input_size)),
                                    T.ToTensor(), T.Normalize(model.mean, model.std)])

    def logits(self, images: list) -> np.ndarray:
        torch = self.torch
        batch = torch.stack([self.transform(im) for im in images]).to(self.device)
        with torch.inference_mode():
            if self.device.startswith("cuda"):
                with torch.autocast("cuda", dtype=torch.float16):
                    out = self.net(batch)
            else:
                out = self.net(batch)
        return out.float().cpu().numpy()


class LogitCache:
    """Each photo's logits for one model, in a small sqlite file beside the photos it came
    from (a benchmark's in its own folder), never in the manifest."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("create table if not exists logits (photo_id integer primary key, "
                        "classes integer not null, logits blob not null)")

    def get(self, ids: Iterable[int]) -> dict[int, np.ndarray]:
        out = {}
        for pid in ids:
            row = self.db.execute("select logits from logits where photo_id = ?",
                                  (int(pid),)).fetchone()
            if row:
                out[int(pid)] = np.frombuffer(row[0], dtype=np.float16).astype(np.float32)
        return out

    def put(self, rows: dict[int, np.ndarray]) -> None:
        with self.db:
            self.db.executemany("insert or replace into logits values (?, ?, ?)",
                                [(int(p), len(v), np.asarray(v, np.float16).tobytes())
                                 for p, v in rows.items()])

    def close(self) -> None:
        self.db.close()


def classify_photos(todo: list[tuple[int, Callable[[], bytes]]], classify: Callable,
                    cache: LogitCache, batch_size: int = 16, log=print) -> dict:
    """Logits for photos not in the cache yet: `todo` is (photo id, bytes reader)."""
    from concurrent.futures import ThreadPoolExecutor

    from ...embed import decode
    have = set(cache.get(p for p, _ in todo))
    todo = [(p, read) for p, read in todo if p not in have]
    stats = Counter(already=len(have))

    def load(item):
        pid, read = item
        try:
            return pid, decode(read())
        except Exception as e:          # an unreadable photo is skipped and counted
            return pid, e
    with ThreadPoolExecutor(max_workers=6) as pool:
        for start in range(0, len(todo), batch_size):
            got = list(pool.map(load, todo[start:start + batch_size]))
            ok = [(p, im) for p, im in got if not isinstance(im, Exception)]
            stats["unreadable"] += len(got) - len(ok)
            if ok:
                out = classify([im for _, im in ok])
                cache.put({p: out[i] for i, (p, _) in enumerate(ok)})
                stats["classified"] += len(ok)
            if (start // batch_size) % 50 == 0:
                log(f"  {start + len(got):,}/{len(todo):,} photos classified")
    return dict(stats)


# --- a held-out split, as JSONL ------------------------------------------------------------

def predict_heldout(conn: sqlite3.Connection, bench: str, model: ExternalModel,
                    ids: list[str], out_path: Path, classify: Callable | None = None,
                    labeller=None, size: str = "large", batch_size: int = 16,
                    root: Path | None = None, log=print) -> dict:
    """Answer each record of `ids` from its benchmark photos and write one JSON line per
    record (`mv heldout import-external` stores them). Reads the manifest only; logits are
    cached in the benchmark's folder (<bench>/external/<short>/logits-<size>.sqlite)."""
    from ... import heldout
    records = [r for r in heldout.load_benchmark(conn, bench, ids, size) if r.photos]
    log(f"  {len(records):,} of {len(ids):,} records have {size} photos")
    labeller = labeller or heldout.Labeller(conn)
    cn = vision_names(read_labels(model, root), labeller)
    ref = checkpoint_id(model, root)
    cache = LogitCache(heldout.bench_dir(conn, bench) / "external" / model.short
                       / f"logits-{size}.sqlite")
    try:
        stores: dict = {}
        todo = []
        for rec in records:
            for pid, location, path in rec.photos:
                store = stores.setdefault(location, heldout.open_location(location))
                todo.append((pid, lambda s=store, p=path: s.get(p)))
        if classify is None:
            net = Classifier(model, root=root)
            classify = net.logits
        stats = classify_photos(todo, classify, cache, batch_size, log)
        written = 0
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                got = cache.get(pid for pid, _s, _p in rec.photos)
                rows = [got[pid] for pid, _s, _p in rec.photos if pid in got]
                if not rows:
                    stats["records without readable photos"] += 1
                    continue
                f.write(json.dumps({
                    "benchmark": bench, "observation_id": rec.observation_id,
                    "backbone": model.backbone, "method": METHOD, "size": size,
                    "reference_hash": ref, "photos": len(rows),
                    "result": record_answer(np.stack(rows), cn)}) + "\n")
                written += 1
    finally:
        cache.close()
    return {"model": model.short, "records": written, "reference_hash": ref,
            "class_names": dict(cn.how), "out": str(out_path), **stats}


# --- a saved scoreboard comparison -----------------------------------------------------------

# The standard summary's top-k and reference bands are evaluate's, as for every model.
from ...evaluate import STANDARD_DEPTH, STANDARD_K, standard_band  # noqa: E402


PHOTO_PREFERENCE = {"large": 0, "medium": 1, "original": 2, "small": 3}


def reference_photo_inputs(conn: sqlite3.Connection, photo_ids: list[int],
                           stores: dict | None = None) -> list[tuple[int, Callable[[], bytes]]]:
    """(photo id, reader) for a comparison record's photos: the copy nearest 384 px wide
    first (large, then medium, original, small), on this machine before S3."""
    from ...storage import LocalStore, S3Store
    stores = {} if stores is None else stores
    out = []
    for pid in photo_ids:
        copies = conn.execute("select store, size, path from photo_copies where photo_id = ?",
                              (pid,)).fetchall()
        if not copies:
            continue
        store, _size, path = min(copies, key=lambda c: (PHOTO_PREFERENCE.get(c[1], 9),
                                                        str(c[0]).startswith("s3://")))
        if store not in stores:
            stores[store] = S3Store(store) if str(store).startswith("s3://") \
                else LocalStore(Path(store))
        out.append((pid, lambda s=stores[store], p=path: s.get(p)))
    return out


def score_records(records: list, answers: dict[str, dict], ref_count: Counter,
                  first_photo_only: bool = False, top_k: int = 5) -> dict:
    """evaluate.evaluate()'s shape (per rank, per reference bucket: top-1 and top-5) plus the
    standard block (top 1/3/5/10 by the true species' reference band), as inat_cv's
    score_records writes it. Names are compared with writing set aside (heldout.name_key);
    `answers` hold Vision labels already (vision_names)."""
    from ...evaluate import RANKS, bucket_of
    from ...heldout import name_key
    tally = {rank: defaultdict(Counter) for rank in RANKS}
    standard = {rank: defaultdict(Counter) for rank in RANKS}
    deepest = max(top_k, *STANDARD_K)
    for rec in records:
        ans = answers.get(rec.observation_id)
        if ans is None:
            continue
        if first_photo_only:
            ans = ans["first_photo"]
        else:
            ans = ans["all"]
        b = bucket_of(ref_count.get(rec.unit, 0))
        band = standard_band(ref_count.get(rec.unit, 0))
        for rank in RANKS:
            t = getattr(rec, rank)
            if not t:
                continue
            want = name_key(t)
            top = [name_key(c["name"]) for c in (ans.get(rank) or [])[:deepest]]
            for key in ("all", b):
                c = tally[rank][key]
                c["n"] += 1
                c["top1"] += top[:1] == [want]
                c[f"top{top_k}"] += want in top[:top_k]
            for key in ("all", band):
                c = standard[rank][key]
                c["n"] += 1
                for k in STANDARD_K:
                    c[f"top{k}"] += want in top[:k]
    out = {rank: {k: {"n": c["n"], "top1": round(c["top1"] / c["n"], 4),
                      f"top{top_k}": round(c[f"top{top_k}"] / c["n"], 4)}
                  for k, c in tally[rank].items() if c["n"]}
           for rank in RANKS}
    out["standard"] = {
        "k": list(STANDARD_K), "bands": [label for _, _, label in STANDARD_DEPTH],
        **{rank: {key: {"n": c["n"], **{f"top{k}": round(c[f"top{k}"] / c["n"], 4)
                                       for k in STANDARD_K}}
                  for key, c in standard[rank].items() if c["n"]}
           for rank in RANKS}}
    return out


MAX_WITHOUT_PHOTOS = 0.02


def run_comparison(conn: sqlite3.Connection, comparison_id: str, model: ExternalModel,
                   classify: Callable | None = None, labeller=None,
                   embeddings_root: Path | None = None, cache_path: Path | None = None,
                   batch_size: int = 16, root: Path | None = None, log=print) -> dict:
    """A saved comparison's test records, as inat_cv.run does for iNat: same records, same
    labels, one eval_runs row (external:<short> / mean-logits) replacing any earlier one."""
    from ...evaluate import comparison_backbones, save_run, shared_records
    from ...heldout import Labeller
    backbones = comparison_backbones(conn, comparison_id)
    if not backbones:
        raise ValueError(f"no comparison {comparison_id!r}")
    saved = conn.execute("select record_set, test_days from eval_runs where comparison_id = ? "
                         "limit 1", (comparison_id,)).fetchone()
    shared = shared_records(conn, backbones, test_days=saved[1],
                            embeddings_root=embeddings_root)
    if shared.record_set != saved[0]:
        raise RuntimeError("the comparison's records have changed since it ran (new data or "
                           "embeddings); run mv compare again and score that one")
    ref_count = Counter(r.unit for r in shared.ref)
    stores: dict = {}
    inputs = {rec.observation_id: reference_photo_inputs(conn, list(rec.photo_rows), stores)
              for rec in shared.test}
    bare = [oid for oid, got in inputs.items() if not got]
    if len(bare) > MAX_WITHOUT_PHOTOS * len(shared.test):
        raise RuntimeError(f"{len(bare):,} of {len(shared.test):,} test records have no photo "
                           f"on this machine or in S3 (first: {', '.join(bare[:5])}); nothing "
                           "was scored")
    labeller = labeller or Labeller(conn)
    cn = vision_names(read_labels(model, root), labeller)
    cache = LogitCache(cache_path or model.folder(root) / "logits-reference.sqlite")
    try:
        if classify is None:
            classify = Classifier(model, root=root).logits
        stats = classify_photos([x for got in inputs.values() for x in got], classify,
                                cache, batch_size, log)
        answers = {}
        for rec in shared.test:
            pids = [pid for pid, _r in inputs[rec.observation_id]]
            got = cache.get(pids)
            rows = [got[p] for p in pids if p in got]
            if rows:
                answers[rec.observation_id] = {"all": record_answer(np.stack(rows), cn),
                                               "first_photo": record_answer(rows[0][None], cn)}
    finally:
        cache.close()
    all_photos = score_records(shared.test, answers, ref_count)
    first = score_records(shared.test, answers, ref_count, first_photo_only=True)
    vocab = {k for k in (names_key(n) for n in cn.species) if k}
    in_vocab = sum(1 for r in shared.test if r.species and names_key(r.species) in vocab)
    with conn:
        conn.execute("delete from eval_runs where comparison_id = ? and backbone = ?",
                     (comparison_id, model.backbone))
    run = save_run(conn, comparison_id, model.backbone, METHOD, shared, saved[1], all_photos,
                   first, extra={"external": {"hf_id": model.hf_id, "licence": LICENCE,
                                              "checkpoint": checkpoint_id(model, root),
                                              "photo_only": True, "temperature": 1.0,
                                              "classes": model.num_classes,
                                              "class_names": dict(cn.how)},
                                 "species_in_model_vocabulary": in_vocab,
                                 "records_answered": len(answers)})
    return {"comparison_id": comparison_id, "test_records": len(shared.test),
            "answered": len(answers), "without_photos": len(bare),
            "species_in_model_vocabulary": in_vocab, **stats, "run": run}


def names_key(name: str) -> str:
    from ...heldout import name_key
    return name_key(name) if name else ""
