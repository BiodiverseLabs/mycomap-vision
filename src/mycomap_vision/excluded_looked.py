"""Research > Excluded photos (Steve, 2026-10-26): every photo people left out of Vision in
Human Feedback, by why it was left out, and "where Vision looked" (looked.py) on any of them,
from the served model or the base model it was tuned from, for research on attention.

**Which photos.** A photo item of the "Photos that show no specimen" queue whose latest decision
is `exclude_photo`; its category is that decision's value (photo_review.REASONS). Decisions are
written only by a signed-in person (feedback_api), so a tagger's flag alone never puts a photo
here. People photos (Steve, 2026-10-26) are listed only to Human Feedback reviewers.

**What is shown** is what Human Feedback shows a read-only viewer (feedback_public.py): a photo
anyone may see now (Creative Commons, or its owner said yes), by its public URL; every other
photo a placeholder with no photographer, no bytes and no map. Never who decided, or their note.

**A map** explains the photo's served score for its record's DNA-verified name, with the
record itself (and its cross-post twins) left out of the reference, as if the photo were
someone's new find: the record's other photos are DNA-verified specimens of the same name and
would otherwise light up whatever the photo shares with them. Beside it, where the photo alone
puts that name among every species (top RANK_DEPTH, else "not in its top 10"). Made on request
(signed in), one at a time on the run maps' worker, kept here and open to anyone after.

**The paper's seal** (Paper Coordinator, 2026-10-26; PLAN §8): this page is research-facing, so
no map is made or shown for a photo whose record is not judged "ok" by feedback.SealGate (a v1
record, or first green on a day before the seal); the photo still shows. One switch lifts it,
MV_EXCLUDED_LOOKED_UNSEALED=1, set only once the paper's v1-test scoring is committed. Requests
keep no requester and write nothing to the identify log.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from . import feedback as fb
from . import photo_review as pr

FILE = "excluded-looked.sqlite"
VERSION = 1
MODELS = ("v1", "base")
REVIEWERS_ONLY = frozenset({"people"})
CATEGORY_LABELS = {
    "micrograph": "Micrographs", "slip": "Field slips and labels",
    "habitat": "Habitat and scenes", "people": "People", "not_fungus": "Not a fungus",
    "dried": "Dried specimens", "spore_print": "Spore prints", "uv": "UV reactions",
    "chemical": "Chemical spot tests", "phylo_tree": "Phylogenetic trees / BLAST",
    "drawing_micro": "Drawings – microscopy", "drawing_specimen": "Drawings – specimen",
    "species_description": "Species descriptions", "blurry": "Blurry or badly exposed",
    "other": "Other"}
assert set(CATEGORY_LABELS) == set(pr.REASON_CODES)
RANK_DEPTH = 10
PAGE_MAX = 96
UNSEAL_SETTING = "MV_EXCLUDED_LOOKED_UNSEALED"

# Each item's latest decision, grouped once (0.6 s on the live store's 79k decisions; the
# per-item max of feedback.LATEST_DECISION took 2.5 s).
EXCLUDED_SQL = (
    "with latest as (select item_id, max(id) as id from decisions group by item_id) "
    "select i.subject, i.record_keys, i.links, d.value, d.id from latest "
    "join decisions d on d.id = latest.id join items i on i.id = d.item_id "
    "where i.item_type = ? and d.decision = 'exclude_photo' order by d.id desc")


@dataclass(frozen=True)
class Excluded:
    photo_key: str
    record_key: str
    category: str


def read_excluded(conn: sqlite3.Connection) -> list[Excluded]:
    """Every photo a person's latest decision leaves out, newest decision first. A photo whose
    record can't be told, or whose reason is not a known category, is left off."""
    out = []
    for subject, record_keys, links, value, _ in conn.execute(EXCLUDED_SQL, (pr.ITEM_TYPE,)):
        rk = pr.record_of_item({"record_keys": json.loads(record_keys or "[]"),
                                "links": json.loads(links or "[]")})
        if rk and value in CATEGORY_LABELS:
            out.append(Excluded(subject, rk, value))
    return out


def visible(rows: list[Excluded], reviewer: bool) -> list[Excluded]:
    """The rows this viewer may list: People only to Human Feedback reviewers."""
    return rows if reviewer else [r for r in rows if r.category not in REVIEWERS_ONLY]


def categories(rows: list[Excluded]) -> list[dict]:
    """{code, label, count} per category with photos, most photos first."""
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.category] = counts.get(r.category, 0) + 1
    return [{"code": c, "label": CATEGORY_LABELS[c], "count": n}
            for c, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def select(rows: list[Excluded], category: str, mapped: set[str] | None = None) -> list[Excluded]:
    """The rows of a category ("all": every one), only those in `mapped` when given."""
    if category != "all" and category not in CATEGORY_LABELS:
        raise ValueError(f"no such category: {category}")
    return [r for r in rows if (category == "all" or r.category == category)
            and (mapped is None or r.photo_key in mapped)]


def held_back(gate: "fb.SealGate | None", record_key: str, unsealed: bool) -> bool:
    """Whether the paper's seal holds this record's maps back (module docstring)."""
    return not unsealed and gate is not None and gate.judge(record_key) != "ok"


def rank_of(ident, vector, name: str, hidden) -> int | None:
    """Where the photo alone puts `name` among every species (1-based), its own record left
    out (`hidden`, Identifier.left_out); None when below RANK_DEPTH."""
    import numpy as np
    leave = hidden if hidden is not None and len(hidden) and ident.can_leave_out() else None
    got = ident.identify_vectors_ungated(np.asarray(vector, dtype=np.float32)[None],
                                         top_k=RANK_DEPTH, n_specimens=0, n_photo_specimens=0,
                                         photo_lookup=lambda ids: {}, hidden=leave)
    names = [c["name"] for c in got["ranks"]["species"]]
    return names.index(name) + 1 if name in names else None


def record_url(record_key: str) -> str:
    if record_key.startswith("mo:"):
        return f"https://mushroomobserver.org/observations/{record_key[3:]}"
    return f"https://www.inaturalist.org/observations/{record_key}"


# --- the base model's vectors ------------------------------------------------------------------
# A release's manifest keeps the embeddings rows of every backbone the laptop ever embedded, but
# ships vector files only for the served ones (release.py), so rows alone don't make a backbone
# usable. The base is offered only when its folder holds every shard the manifest names; shipping
# it is a folder copy into <release>/embeddings/<base>/, read by serving.map_embeddings as
# shard-NNNNN.ids.npy (int64 photo ids) and shard-NNNNN.npy (float, or int8 codes with
# shard-NNNNN.scale.npy beside them), row r of a shard being the manifest's (photo_id, shard, r).

def manifest_shards(conn: sqlite3.Connection, backbone: str) -> list[int]:
    return [r[0] for r in conn.execute("select distinct shard from embeddings where backbone = ? "
                                       "order by shard", (backbone,))]


def vectors_shipped(shards: list[int], folder: Path) -> bool:
    """Every shard's two files are in `folder` (the cheap check the server makes)."""
    folder = Path(folder)
    return bool(shards) and folder.is_dir() and all(
        (folder / f"shard-{s:05d}.ids.npy").is_file() and (folder / f"shard-{s:05d}.npy").is_file()
        for s in shards)


def check_vectors(conn: sqlite3.Connection, backbone: str, folder: Path,
                  dim: int | None = None, sample: int = 2000) -> dict:
    """Whether `folder` holds `backbone`'s vectors exactly as `conn`'s manifest names them, and
    loads as the server loads it (serving.map_embeddings): every shard's files, each row's photo
    id, the vectors' shape, dtype and scales, `dim` when given, and unit length on a sample of
    float rows. {"ok", "problems", "shards", "vectors", "dim", "dtype"}."""
    import numpy as np

    from .serving import SCALE_SUFFIX, map_embeddings
    folder = Path(folder)
    problems: list[str] = []
    want: dict[int, dict[int, int]] = {}
    for photo_id, shard, row in conn.execute(
            "select photo_id, shard, row from embeddings where backbone = ?", (backbone,)):
        want.setdefault(int(shard), {})[int(row)] = int(photo_id)
    if not want:
        return {"ok": False, "problems": [f"the manifest has no embeddings rows for {backbone}"],
                "shards": 0, "vectors": 0, "dim": None, "dtype": None}
    dims, dtypes, total = set(), set(), 0
    rng = np.random.default_rng(0)
    for s in sorted(want):
        ids_f, vec_f = folder / f"shard-{s:05d}.ids.npy", folder / f"shard-{s:05d}.npy"
        if not (ids_f.is_file() and vec_f.is_file()):
            problems.append(f"shard {s}: {ids_f.name} or {vec_f.name} is missing")
            continue
        ids = np.load(ids_f)
        v = np.load(vec_f, mmap_mode="r")
        if ids.ndim != 1 or ids.dtype.kind not in "iu":
            problems.append(f"shard {s}: photo ids are {ids.dtype} {ids.shape}, not 1-D integers")
            continue
        if v.ndim != 2 or len(v) != len(ids):
            problems.append(f"shard {s}: {len(ids)} ids but vectors of shape {v.shape}")
            continue
        dims.add(int(v.shape[1]))
        dtypes.add(str(v.dtype))
        rows = want[s]
        wrong = [r for r, pid in rows.items() if r >= len(ids) or int(ids[r]) != pid]
        if wrong:
            problems.append(f"shard {s}: {len(wrong)} manifest rows name another photo "
                            f"(first: row {wrong[0]})")
        if len(ids) != len(rows):
            problems.append(f"shard {s}: {len(ids)} vectors, the manifest names {len(rows)}")
        total += len(ids)
        if v.dtype == np.int8:
            sc_f = folder / f"shard-{s:05d}{SCALE_SUFFIX}"
            if not sc_f.is_file() or np.load(sc_f).shape != (len(ids),):
                problems.append(f"shard {s}: int8 codes without one scale per row in {sc_f.name}")
        elif v.dtype.kind == "f":
            pick = rng.choice(len(v), size=min(len(v), max(1, sample // len(want))), replace=False)
            norms = np.linalg.norm(np.asarray(v[np.sort(pick)], dtype=np.float64), axis=1)
            if not np.all(np.isfinite(norms)) or np.abs(norms - 1).max() > 1e-2:
                problems.append(f"shard {s}: vectors are not unit length (norms "
                                f"{norms.min():.3f}..{norms.max():.3f})")
        else:
            problems.append(f"shard {s}: vectors of dtype {v.dtype}")
    if len(dims) > 1:
        problems.append(f"shards differ in width: {sorted(dims)}")
    if dim is not None and dims and dims != {dim}:
        problems.append(f"vectors are {sorted(dims)} wide, the model's are {dim}")
    if not problems:
        try:
            ids, vecs = map_embeddings(conn, backbone, folder)
            if len(ids) != total:
                problems.append(f"the server's loader read {len(ids)} vectors, not {total}")
        except Exception as e:  # noqa: BLE001 - report what the server would hit
            problems.append(f"the server's loader refused it: {type(e).__name__}: {e}")
    return {"ok": not problems, "problems": problems, "shards": len(want), "vectors": total,
            "dim": next(iter(dims)) if len(dims) == 1 else None,
            "dtype": next(iter(dtypes)) if len(dtypes) == 1 else sorted(dtypes)}


class Maps:
    """The kept maps: one row per photo and model, insert or replace (a newer VERSION), never
    who asked."""

    SCHEMA = ("create table if not exists maps (photo_key text not null, model text not null, "
              "version integer not null, doc text not null, made_at text not null, "
              "primary key (photo_key, model))")

    def __init__(self, path: Path | str):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock, self.conn:
            self.conn.execute(self.SCHEMA)
        self._mapped: dict[str, tuple[int, set[str]]] = {}

    def get(self, photo_key: str, model: str) -> dict | None:
        with self.lock:
            r = self.conn.execute("select doc from maps where photo_key = ? and model = ? "
                                  "and version = ?", (photo_key, model, VERSION)).fetchone()
        return json.loads(r[0]) if r else None

    def keep(self, photo_key: str, model: str, doc: dict) -> None:
        with self.lock, self.conn:
            self.conn.execute("insert or replace into maps values (?,?,?,?,?)",
                              (photo_key, model, VERSION, json.dumps(doc, separators=(",", ":")),
                               doc["made_at"]))
            self._mapped.clear()

    def mapped(self) -> dict[str, set[str]]:
        """photo key -> the models with a kept map."""
        with self.lock:
            out: dict[str, set[str]] = {}
            for k, m in self.conn.execute("select photo_key, model from maps where version = ?",
                                          (VERSION,)):
                out.setdefault(k, set()).add(m)
            return out


class Listing:
    """read_excluded, read again only when the store's decisions change (checked at most every
    `every` seconds), on a read-only connection of its own: the store is in WAL mode, so the
    read never holds up a reviewer's save."""

    def __init__(self, store: fb.Store, every: float = 30.0, clock=time.monotonic):
        self.store, self.every, self.clock = store, every, clock
        self._at, self._version, self._rows = -1e9, None, []
        self._conn: sqlite3.Connection | None = None
        self.lock = threading.Lock()

    def rows(self) -> list[Excluded]:
        with self.lock:
            if self.clock() - self._at >= self.every:
                self._at = self.clock()
                if self._conn is None:
                    self._conn = sqlite3.connect(
                        f"file:{Path(self.store.path).as_posix()}?mode=ro", uri=True,
                        check_same_thread=False)
                version = self._conn.execute("select max(id), count(*) from decisions").fetchone()
                if version != self._version:
                    self._rows = read_excluded(self._conn)
                    self._version = version
            return self._rows
