"""`mv paper-score`: every table of the paper's sealed-test analysis from one command.

    mv paper-score --release v1 --test v1-test --config <research>/paper/analysis-v1.json

The analysis plan (the config: comparators, methods, hypotheses, slices, bootstrap) is
fixed before the sealed test is opened. On the sealed test this command is the only thing
that predicts, embeds or scores, and it refuses to run unless:

- the tag `paper-analysis-freeze-v1` exists in the research repo (config.RESEARCH_DIR),
- the config given is byte-identical to that tag's `paper/analysis-v1.json`, and
- this code is the commit the config names (`code_commit`), unless `--allow-code <sha>`
  names the commit running, which marks every output "post hoc" in capitals.

`--test val` scores the base release's val split against its train split, and a draft
test release (`dry-...-test`) runs freely: both are how this command is tested before the
freeze. Nothing is ever written to the manifest: outputs go to `--out`.

Per find (paper_outcomes.Find): the release's answer key; depth = reference records of
the true species in the reference used (val: train; the test: train + val), errata records
left out; observer (clusters); name kind; photos; slice attributes; near-duplicate flags
(a photo byte-identical to a reference photo, flagged in the test release's near_base
table by its dHash, or at cosine >= the config's threshold to a reference photo on the
frozen backbone). Comparators:

- `vision`: a backbone and method scored here from the stored vectors (photos embedded by
  this command if missing; place methods see only the record's public point);
- `published`: `mv external run`'s <model>.jsonl for the same release split;
- `inat`: the comparisons lane's E14 finds.jsonl (`score`: vision or combined);
- `expert`: its E31 scores.jsonl (`tier`, `view`: best or community, `strict`).

Outputs: report.json (every table, test and input hash), report.md (Steve's tables),
predictions/<comparator>.jsonl (each Vision answer, for audit).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config as cfgmod
from . import paper_outcomes as po
from . import paper_stats as ps
from . import paper_tables as pt

TAG = "paper-analysis-freeze-v1"
CONFIG_IN_REPO = "paper/analysis-v1.json"
CONFIG_FORMAT = 1
KINDS = ("vision", "published", "inat", "expert")
SEASON = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
          6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}


class FrozenError(RuntimeError):
    """The sealed test may not be touched by this command, config or code."""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path) -> str:
    return sha256_bytes(Path(path).read_bytes())


# --- the configuration ----------------------------------------------------------------------------

def load_config(path) -> tuple[dict, bytes]:
    body = Path(path).read_bytes()
    doc = json.loads(body)
    check_config(doc)
    return doc, body


def check_config(doc: dict) -> None:
    if doc.get("format") != CONFIG_FORMAT:
        raise ValueError(f"analysis config format {doc.get('format')!r}, expected {CONFIG_FORMAT}")
    ids = [c["id"] for c in doc.get("comparators", [])]
    if len(ids) != len(set(ids)):
        raise ValueError("comparator ids must be unique")
    for c in doc.get("comparators", []):
        if c.get("kind") not in KINDS:
            raise ValueError(f"comparator {c.get('id')}: kind is one of {KINDS}")
        if c["kind"] == "vision" and not (c.get("backbone") and c.get("method")):
            raise ValueError(f"comparator {c['id']}: a vision comparator names backbone and method")
        if c["kind"] in ("published", "inat", "expert") and not c.get("file"):
            raise ValueError(f"comparator {c['id']}: needs its input file")
    gate = doc.get("gate")
    if gate:
        for k in ("dir", "fingerprint", "test_vectors"):
            if not gate.get(k):
                raise ValueError(f"gate: names {k}")
        # The gate's served twins ("<id> served") may be named by a hypothesis.
        ids = ids + [f"{c['id']} served" for c in doc.get("comparators", [])
                     if c.get("kind") == "vision"]
    for fam, hyps in (doc.get("hypotheses") or {}).items():
        for h in hyps:
            for k in ("a", "b", "comparator"):
                if k in h and h[k] not in ids:
                    raise ValueError(f"{fam}/{h.get('id')}: unknown comparator {h[k]!r}")


def resolve(path: str | Path) -> Path:
    """A path in the config: absolute as given; `data/...` under the data folder
    (config.DATA_DIR); `paper/...` and `experiments/...` under the research repo."""
    p = Path(str(path))
    if p.is_absolute():
        return p
    parts = p.parts
    if parts and parts[0] == "data":
        return cfgmod.DATA_DIR.joinpath(*parts[1:])
    if parts and parts[0] in ("paper", "experiments"):
        return cfgmod.RESEARCH_DIR / p
    return p


def attrs_file(block: dict | None, given) -> Path | None:
    """The per-find attributes file: the config's "attrs" block (file + sha256) pins it; a
    --attrs that is not the same file (by sha256) is refused. Without a block, --attrs as
    given (or none)."""
    if not block:
        return Path(given) if given else None
    path = resolve(block["file"])
    got = sha256_file(path)
    if got != block.get("sha256"):
        raise ValueError(f"attrs {path}: sha256 {got[:12]}, the config pins "
                         f"{str(block.get('sha256'))[:12]}")
    if given and sha256_file(given) != got:
        raise ValueError(f"--attrs {given} is not the attrs file the config pins "
                         f"({got[:12]}): give that one or none")
    return path


def apply_errata(base, block: dict | None) -> dict | None:
    """The config's erratum (file + sha256) on the base release: refused when the file's
    sha256 is not the one pinned, or when the release already carries another erratum
    (a CLI --errata that differs). An erratum the release already carries, the same, is
    kept; without a block, whatever the release carries is used."""
    if not block:
        return None
    from . import errata as errata_mod
    path = resolve(block["file"])
    got = sha256_file(path)
    if got != block.get("sha256"):
        raise ValueError(f"erratum {path}: sha256 {got[:12]}, the config pins "
                         f"{str(block.get('sha256'))[:12]}")
    carried = getattr(base, "errata", None)
    if carried is not None:
        if carried.sha256 != got:
            raise ValueError(f"the release carries erratum {carried.sha256[:12]} (--errata), "
                             f"the config names {got[:12]}: give the config's or none")
        return block
    base.use_errata(errata_mod.read(path))
    return block


# --- the freeze guard -------------------------------------------------------------------------

def _git(repo: Path, *args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True)


def same_commit(a: str, b: str) -> bool:
    """Two commit ids name the same commit: one is a prefix of the other, 7+ hex digits."""
    a, b = (a or "").strip().lower(), (b or "").strip().lower()
    return len(a) >= 7 and len(b) >= 7 and (a.startswith(b) or b.startswith(a))


def guard(sealed: bool, config_bytes: bytes, config: dict, research_dir: Path,
          code_commit: str, allow_code: str | None = None) -> dict:
    """Whether this run may touch the scored records, and how it is labelled. A sealed test
    needs the freeze tag, the tagged config byte for byte, and the frozen code commit (or
    `allow_code` = the commit running: post hoc)."""
    if not sealed:
        if allow_code:
            raise ValueError("--allow-code is only for the sealed test")
        return {"sealed": False, "post_hoc": False}
    if _git(research_dir, "rev-parse", "--verify", "--quiet", f"refs/tags/{TAG}").returncode:
        raise FrozenError(f"the sealed test stays closed: no tag {TAG} in {research_dir}")
    tagged = _git(research_dir, "show", f"{TAG}:{CONFIG_IN_REPO}")
    if tagged.returncode:
        raise FrozenError(f"tag {TAG} has no {CONFIG_IN_REPO}")
    if tagged.stdout != config_bytes:
        crlf = tagged.stdout.replace(b"\r", b"") == config_bytes.replace(b"\r", b"")
        raise FrozenError(f"the config given is not the one tagged {TAG} "
                          + ("(line endings differ: give the tagged bytes, as "
                             f"git show {TAG}:{CONFIG_IN_REPO} prints them)" if crlf else
                             f"(sha256 {sha256_bytes(config_bytes)[:12]} vs "
                             f"{sha256_bytes(tagged.stdout)[:12]})"))
    if code_commit.endswith("-dirty") or code_commit == "unknown":
        raise FrozenError(f"the sealed test is scored only by committed code (this is "
                          f"{code_commit})")
    frozen = str(config.get("code_commit") or "")
    post_hoc = False
    if not same_commit(code_commit, frozen):
        if allow_code and same_commit(code_commit, allow_code):
            post_hoc = True
        else:
            raise FrozenError(f"this code ({code_commit[:12]}) is not the frozen paper-score "
                              f"commit ({frozen[:12] or 'none named'}); --allow-code "
                              f"{code_commit[:12]} runs it as POST HOC")
    return {"sealed": True, "tag": TAG, "post_hoc": post_hoc,
            "config_sha256": sha256_bytes(config_bytes), "code_commit": code_commit,
            "frozen_code_commit": frozen}


# --- finds ----------------------------------------------------------------------------------------

def scored_and_reference(base, test):
    """(scored release, scored split, reference splits of base, sealed?) for `test`: "val"
    (base's val against its train) or a test release made for base (its test split
    against base's train + val)."""
    if test == "val":
        return base, "val", ("train",), False
    from .replications.fungitastic import dataset
    if not dataset.is_test_release(test) or test.manifest.get("test_for") != base.id:
        raise ValueError(f"{test.id} is not a test release of {base.id}")
    return test, "test", ("train", "val"), not test.manifest.get("draft")


def build_finds(scored_rows: list[dict], photos: dict[str, list[dict]],
                ref_rows: list[dict], near: dict[str, dict] | None = None,
                extra_attrs: dict[str, dict] | None = None) -> list[po.Find]:
    """Finds from release rows: depth from the reference rows' species, attributes for the
    slices (source, month, season, licence class, observer new to the reference, name kind)
    and any extra attributes given per record (e.g. ecoregion, distance band)."""
    depth = Counter(r["species"] for r in ref_rows if r.get("species"))
    ref_observers = {r["observer_login"] for r in ref_rows if r.get("observer_login")}
    out = []
    for r in scored_rows:
        key = str(r["record_key"])
        ph = photos.get(key, [])
        month = int(r["observed_on"][5:7]) if (r.get("observed_on") or "")[:4].isdigit() \
            and len(r["observed_on"]) >= 7 else None
        who = r.get("observer_login")
        attrs = {"source": r.get("source"), "name_kind": r.get("name_kind"),
                 "photos": photo_band(len(ph)),
                 "former_heldout": "yes" if r.get("benchmark") else "no",
                 "month": month, "season": SEASON.get(month, "(no date)"),
                 "licence": "arr" if any(p.get("license_class") == "arr" for p in ph)
                 else "cc",
                 "observer_new": "new" if who and who not in ref_observers else "known"}
        attrs.update((extra_attrs or {}).get(key, {}))
        species = r.get("species") or ""
        out.append(po.Find(key, who or f"(unknown observer):{key}", species,
                           r.get("genus") or "", r.get("family") or "",
                           r.get("name_kind") or ("formal" if species else "one-word"),
                           depth.get(species, 0) if species else 0, len(ph), attrs,
                           dict((near or {}).get(key, {}))))
    return out


def photo_band(n: int) -> str:
    return "1" if n <= 1 else "2" if n == 2 else "3" if n == 3 else "4-5" if n <= 5 else "6+"


def read_attrs(path) -> dict[str, dict]:
    """Extra per-find attributes: a TSV with a record_key column."""
    import csv
    with open(path, encoding="utf-8") as fh:
        return {r["record_key"]: {k: v for k, v in r.items() if k != "record_key"}
                for r in csv.DictReader(fh, delimiter="\t")}


# --- near-duplicates ------------------------------------------------------------------------------

def max_cosine(query: np.ndarray, ref: np.ndarray, q_chunk: int = 2048,
               r_chunk: int = 65536) -> np.ndarray:
    """Each query vector's highest cosine to any reference vector (both unit length)."""
    best = np.full(len(query), -np.inf, dtype=np.float32)
    for i in range(0, len(query), q_chunk):
        q = np.asarray(query[i:i + q_chunk], dtype=np.float32)
        for j in range(0, len(ref), r_chunk):
            s = q @ np.asarray(ref[j:j + r_chunk], dtype=np.float32).T
            best[i:i + q_chunk] = np.maximum(best[i:i + q_chunk], s.max(axis=1))
    return best


def near_duplicates(scored_photos: dict[str, list[dict]], ref_shas: set[str],
                    dhash_keys: set[str], cosine_of: dict[int, float] | None,
                    threshold: float) -> dict[str, dict]:
    """Per find: bytes (a photo's original sha256 is a reference photo's), dhash (a photo
    the test build flagged as a near-copy of a base photo, by photo key; or the record), cosine (a photo at or above `threshold`)."""
    out = {}
    for key, ph in scored_photos.items():
        flags = {"bytes": any(p.get("original_sha256") in ref_shas for p in ph),
                 "dhash": key in dhash_keys or any(str(p.get("photo_key")) in dhash_keys
                                                    for p in ph)}
        if cosine_of is not None:
            flags["cosine"] = any(cosine_of.get(int(p["manifest_photo_id"]), -1.0) >= threshold
                                  for p in ph)
        out[key] = flags
    return out


# --- Vision's answers -----------------------------------------------------------------------------

def vision_answers(vectors: np.ndarray, ref_records: list, scored_records: list, method: str,
                   backbone: str, top_k: int = 10, calibration: dict | None = None,
                   truths: dict[str, dict] | None = None) -> dict[str, dict]:
    """Each scored record's ranked species and genus names from `method` fitted on the
    reference records (evaluate.fit_method; the records' photo_rows index `vectors`).

    With `calibration` ({rank: {temperature, floor}}, fixed on val, never refitted here)
    each answer also carries, per calibrated rank, `conf` (the top name's probability),
    `p_true` (the true name's, from `truths` {key: {rank: name}}; 0 when the reference
    lacks it), `sumsq` (for the multiclass Brier score) and `set` (the likely set at the
    floor, likely.likely_set; None without a floor)."""
    from . import evaluate, likely
    index, model = evaluate.fit_method(vectors, ref_records, method, backbone)
    uses_context = getattr(model, "needs_context", False)
    out = {}
    for rec in scored_records:
        rows = rec.photo_rows
        scores = (model.species_scores(vectors[rows], evaluate.context_of(rec)) if uses_context
                  else model.species_scores(vectors[rows]))
        ans = {}
        for rank in ("species", "genus"):
            rs = evaluate.rank_scores(scores, index, rank)
            ans[rank] = evaluate.top_labels(rs, index.labels[rank], top_k) if len(rs) else []
        if calibration:
            ans["cal"] = {}
            for rank, c in calibration.items():
                rs = evaluate.rank_scores(scores, index, rank)
                if not len(rs):
                    continue
                p = likely.probabilities(rs, float(c["temperature"]))
                names = index.labels[rank]
                truth = ((truths or {}).get(rec.observation_id) or {}).get(rank)
                pos = {n: i for i, n in enumerate(names)}.get(truth) if truth else None
                one = {"conf": float(p.max()), "sumsq": float((p * p).sum()),
                       "p_true": float(p[pos]) if pos is not None else 0.0, "set": None}
                if c.get("floor") is not None:
                    chosen, _ = likely.likely_set(p, float(c["floor"]))
                    one["set"] = [names[i] for i in chosen]
                ans["cal"][rank] = one
        out[rec.observation_id] = ans
    return out


LEAVE_OUT = ("observer", "trip")


def leave_out_key(rec, mode: str):
    """What a find shares with the reference records left out for it (E12): its observer,
    or its observer and observation date (a trip)."""
    if mode == "observer":
        return rec.observer
    if mode == "trip":
        return (rec.observer, rec.observed_on)
    raise ValueError(f"leave_out is one of {LEAVE_OUT}")


def vision_answers_leave_out(vectors, ref_records, scored_records, method, backbone, mode,
                             **kw) -> dict[str, dict]:
    """vision_answers with, for each group of finds sharing a leave_out_key, the reference
    records sharing it removed and the method fitted again: what the identifier gives a
    find when nothing of its own observer (or trip) is in the reference. A find with no
    observer leaves nothing out."""
    groups: dict = defaultdict(list)
    for r in scored_records:
        groups[leave_out_key(r, mode) if r.observer else None].append(r)
    out = {}
    for key, recs in groups.items():
        ref = ref_records if key is None else [
            r for r in ref_records if not (r.observer and leave_out_key(r, mode) == key)]
        out.update(vision_answers(vectors, ref, recs, method, backbone, **kw))
    return out


def sized_name(backbone: str, px: int) -> str:
    """The storage name of a backbone's vectors of photos sent at `px` (E30)."""
    return f"{backbone}-px{int(px)}"


class Resized:
    """A backbone that sees each photo as an app would send it at `px`: the longer side
    scaled down to `px` (never up), saved as JPEG at `quality`, decoded again."""

    def __init__(self, backbone, px: int, quality: int = 90):
        self.base, self.px, self.quality = backbone, int(px), quality
        self.name, self.dim = sized_name(backbone.name, px), backbone.dim
        self.device = getattr(backbone, "device", None)

    def prepare(self, image):
        import io

        from PIL import Image
        w, h = image.size
        scale = self.px / max(w, h)
        if scale < 1:
            image = image.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                                 Image.BICUBIC)
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=self.quality)
        image = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        base = getattr(self.base, "prepare", None)
        return base(image) if base else image

    def encode(self, items):
        return self.base.encode(items)


def ensure_embedded(conn, spec: str, photo_ids: list[int], px: int | None = None,
                    loader=None, log=print) -> int:
    """Embed the photos `spec` (at `px`, Resized) has no vector for, from copies on this
    machine (backbone_screen.photos_to_embed: never S3). Returns how many were embedded."""
    from . import backbone_screen, models
    from .embed import embed_photos
    from .storage import open_store
    loader = loader or models.load_backbone
    name = sized_name(models.storage_name(spec), px) if px else models.storage_name(spec)
    todo = backbone_screen.photos_to_embed(conn, name, photo_ids)
    n = sum(len(v) for v in todo.values())
    if not n:
        return 0
    bb = loader(spec)
    if px:
        bb = Resized(bb, px)
    for where, items in todo.items():
        embed_photos(conn, open_store(where, cfgmod.DATA_DIR), bb, items,
                     cfgmod.DATA_DIR / "embeddings" / name, log=log)
    return n


def with_public_points(records: list, public: dict[str, tuple]) -> list:
    """The scored records with their public point only (paper rule: obscured where iNat
    obscures, MO's box centre), never the release's true point; none when there is none."""
    from dataclasses import replace
    return [replace(r, latitude=public.get(r.observation_id, (None, None))[0],
                    longitude=public.get(r.observation_id, (None, None))[1]) for r in records]


def public_points(rel) -> dict[str, tuple]:
    if not rel.has_private:
        raise ValueError(f"{rel.id} has no private.sqlite here: no public points")
    return {str(k): ((lat / 1e6, lon / 1e6) if lat is not None and kind != "none"
                     else (None, None))
            for k, lat, lon, kind in rel.private.execute(
                "select record_key, public_lat_micro, public_lon_micro, public_kind "
                "from record_public_places")}


# --- the run --------------------------------------------------------------------------------------

CROSSWALK_MODES = ("reviewed", "full")


def crosswalk_rule(spec: dict, names, conn) -> tuple:
    """same(answer, truth) for a published model scored through the GBIF crosswalk.

    The frozen cache (`cache`, sha256 pinned) gives each name's accepted keys (Vision's
    label of it first, heldout.Labeller). `mode` "reviewed" (the primary score): a join
    counts only when its (their answer, our label) pair is marked "credit" in the reviewed
    list (`review`, sha256 pinned); any other join, including one never reviewed (first
    met on the sealed test), is scored by exact names and counted. `mode` "full": every
    join counts (the row beside). Returns (same, info, counter of uncredited joins)."""
    import csv
    from collections import Counter

    from . import heldout
    from .replications.fungitastic import crosswalk as gbif
    mode = spec.get("mode", "reviewed")
    if mode not in CROSSWALK_MODES:
        raise ValueError(f"crosswalk mode is one of {CROSSWALK_MODES}")
    cache = resolve(spec["cache"])
    got = sha256_file(cache)
    if spec.get("cache_sha256") and got != spec["cache_sha256"]:
        raise ValueError(f"crosswalk cache {cache}: sha256 {got[:12]}, the config pins "
                         f"{spec['cache_sha256'][:12]}")
    credited, rev_sha = set(), None
    if mode == "reviewed":
        if not spec.get("review"):
            raise ValueError("a reviewed crosswalk needs its review list")
        rev_sha = sha256_file(resolve(spec["review"]))
        if spec.get("review_sha256") and rev_sha != spec["review_sha256"]:
            raise ValueError(f"crosswalk review {spec['review']}: sha256 {rev_sha[:12]}, the "
                             f"config pins {spec['review_sha256'][:12]}")
        with open(resolve(spec["review"]), encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                if (r.get("review") or "").strip().lower() == "credit":
                    credited.add((r["their_answer"], r["our_label"]))
    labeller = heldout.Labeller(conn, extra=list(names))
    matcher = gbif.Matcher(cache)
    try:
        cw = gbif.Crosswalk.from_cache(matcher, {labeller.label(n) for n in names if n})
    finally:
        matcher.close()
    uncredited = Counter()

    def same(a: str, t: str) -> bool:
        if not cw.same(labeller.label(a), labeller.label(t)):
            return False
        if mode == "full" or (a, t) in credited:
            return True
        uncredited[(a, t)] += 1
        return False
    return same, {"mode": mode, "cache_sha256": got, "review_sha256": rev_sha,
                  "credited_pairs": len(credited)}, uncredited


def outcomes_for(comp: dict, finds: list[po.Find], answers: dict | None = None,
                 conn=None) -> tuple[dict, dict]:
    """(outcomes by record key, input info) for one comparator."""
    by_key = {f.key: f for f in finds}
    info: dict = {"kind": comp["kind"]}
    if comp["kind"] == "vision":
        collapse = bool(comp.get("collapse_infraspecific"))
        return {k: po.from_ranked(a["species"], a["genus"], by_key[k], collapse=collapse)
                for k, a in (answers or {}).items() if k in by_key}, info
    path = resolve(comp["file"])
    lines = po.read_jsonl(path)
    info.update(file=str(comp["file"]), sha256=sha256_file(path), lines=len(lines))
    out = {}
    if comp["kind"] == "published":
        same, uncredited = None, None
        if comp.get("crosswalk"):
            names = {a["name"] for line in lines for rank in ("species", "genus")
                     for a in line["result"].get(rank) or []}
            names |= {x for f in finds for x in (f.species, f.genus) if x}
            same, info["crosswalk"], uncredited = crosswalk_rule(comp["crosswalk"], names, conn)
        for line in lines:
            k = str(line["observation_id"])
            if k in by_key:
                out[k] = po.published_outcome(line, by_key[k], same,
                                              bool(comp.get("collapse_infraspecific")))
        if uncredited is not None:
            # Joins the review did not credit (excluded, or never reviewed): exact names.
            info["crosswalk"]["uncredited_joins"] = {
                "pairs": len(uncredited), "answers": sum(uncredited.values()),
                "top": [{"their_answer": a, "our_label": t, "answers": n}
                        for (a, t), n in uncredited.most_common(30)]}
    elif comp["kind"] == "inat":
        po.check_format(lines, "e14")
        disagree = 0
        for line in lines:
            k = str(line["record_key"])
            if k in by_key:
                out[k] = po.inat_outcome(line, by_key[k], comp.get("score", "vision"))
                disagree += bool(line["truth"].get("formal_species")) != (
                    by_key[k].name_kind == "formal")
        info["formal_disagreements"] = disagree
        if out and disagree > comp.get("max_formal_disagreement", 0.005) * len(out):
            raise ValueError(f"{comp['id']}: {disagree} finds disagree with the release on "
                             "whether the true name is formal")
    else:
        po.check_format(lines, "e31")
        for line in lines:
            k = str(line["record_key"])
            if k in by_key:
                o = po.expert_outcome(line, str(comp.get("tier", "1")), comp.get("view", "best"),
                                      bool(comp.get("strict")))
                if o is not None:
                    out[k] = o
        # A find the experts' file does not score is not a find they got wrong.
        for f in finds:
            if f.key not in out:
                out[f.key] = {c: None for c in po.CELLS} | {"answered": False}
    info["finds_with_lines"] = len(out)
    return out, info


def run(conn, base, test, config: dict, config_bytes: bytes, out: Path,
        attrs_path=None, allow_code: str | None = None,
        research_dir: Path | None = None, embed_missing=None, log=print) -> dict:
    """Score every comparator of `config` on `test` ("val" or a test release of `base`)
    and write report.json, report.md and the Vision predictions under `out`."""
    from .embed import load_embeddings
    from .replications.fungitastic import dataset
    scored_rel, split, ref_splits, sealed = scored_and_reference(base, test)
    g = guard(sealed, config_bytes, config, research_dir or cfgmod.RESEARCH_DIR,
              cfgmod.code_version(), allow_code)
    apply_errata(base, config.get("errata"))
    attrs_path = attrs_file(config.get("attrs"), attrs_path)
    # An erratum given to the base release (--errata; Release.use_errata) has already left
    # its records out of every split base reads: references and depth alike.
    errata = set(base.errata.keys) if getattr(base, "errata", None) is not None else set()
    reps, seed = int(config.get("bootstrap", {}).get("reps", 10_000)), \
        int(config.get("bootstrap", {}).get("seed", 0))
    ref_rows = [dict(r) for r in base.records(ref_splits)]
    scored_rows = [dict(r) for r in scored_rel.records(split)]
    scored_keys = {str(r["record_key"]) for r in scored_rows}
    photos: dict[str, list[dict]] = defaultdict(list)      # the scored finds' photos only
    for p in scored_rel.db.execute("select * from photos where included = 1 order by "
                                   "record_key, position, photo_key"):
        if str(p["record_key"]) in scored_keys:
            photos[str(p["record_key"])].append(dict(p))
    ref_keys = {str(r["record_key"]) for r in ref_rows}
    ref_photo_rows = [dict(p) for p in base.db.execute(
        "select manifest_photo_id, original_sha256, record_key from photos where included = 1")
        if str(p["record_key"]) in ref_keys]
    ref_shas = {p["original_sha256"] for p in ref_photo_rows if p["original_sha256"]}
    # dHash near-copy flags: the test release's own near_base table (part of its
    # release_hash), the one source; empty for val and for a build made without them.
    near_base = scored_rel.near_base() if scored_rel is not base else {}
    dhash = {pk for pk, v in near_base.items() if v.get("base_photo_key")}
    nd_cfg = config.get("near_duplicates", {})
    cosine_of = None
    if nd_cfg.get("backbone"):
        b = nd_cfg["backbone"]
        want = [int(p["manifest_photo_id"]) for ph in photos.values() for p in ph]
        if embed_missing is not None:
            embed_missing(b, want, None)
        ids, vecs = load_embeddings(conn, b)
        row = {int(p): i for i, p in enumerate(ids.tolist())}
        q = [p for p in want if p in row]
        r = [row[int(p["manifest_photo_id"])] for p in ref_photo_rows
             if int(p["manifest_photo_id"]) in row]
        sims = max_cosine(vecs[[row[p] for p in q]], vecs[r]) if q and r else np.zeros(0)
        cosine_of = dict(zip(q, sims.tolist()))
    near = near_duplicates(photos, ref_shas, dhash, cosine_of,
                           float(nd_cfg.get("cosine", 0.99)))
    finds = build_finds(scored_rows, photos, ref_rows, near,
                        read_attrs(attrs_path) if attrs_path else None)
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(exist_ok=True)
    outcomes, inputs, calib, top1, per_photo = {}, {}, {}, {}, {}
    public = None
    for comp in config["comparators"]:
        answers = None
        if comp["kind"] == "vision":
            b = comp["backbone"]
            px = comp.get("photo_px")
            wanted = [int(p["manifest_photo_id"]) for ph in photos.values() for p in ph]
            if embed_missing is not None:
                embed_missing(b, wanted, None)
            ids, vecs = load_embeddings(conn, b)
            row = {int(p): i for i, p in enumerate(ids.tolist())}
            ref = base.as_records(ref_splits, row, public_trainable_only=(
                comp.get("reference") == "public_trainable_only"))
            if comp.get("reference_as_of"):           # E7b: the reference as it stood then
                ref = [r for r in ref if r.validated_on and r.validated_on <= comp["reference_as_of"]]
            if px:
                # E30: the scored photos as sent at `px` (references stay as stored).
                qname = sized_name(b, px)
                if embed_missing is not None:
                    embed_missing(b, wanted, px)
                qids, qvecs = load_embeddings(conn, qname)
                offset = len(vecs)
                vecs = np.concatenate([vecs, qvecs]) if len(qvecs) else vecs
                scored = scored_rel.as_records(split, {int(p): offset + i
                                                       for i, p in enumerate(qids.tolist())})
            else:
                scored = scored_rel.as_records(split, row)
            if comp.get("place"):
                public = public if public is not None else public_points(scored_rel)
                scored = with_public_points(scored, public)
            log(f"[{comp['id']}] {b} / {comp['method']}: {len(scored):,} finds with vectors, "
                f"{len(ref):,} reference records")
            truths = {f.key: {"species": f.species, "genus": f.genus, "family": f.family}
                      for f in finds}
            if comp.get("photos") == "first":            # E19: the first photo only
                from dataclasses import replace
                scored = [replace(r, photo_rows=r.photo_rows[:1]) for r in scored]
            if comp.get("photos") == "each":             # E19: every photo on its own
                from dataclasses import replace
                singles = [replace(r, observation_id=f"{r.observation_id}#{i}",
                                   photo_rows=[row_])
                           for r in scored for i, row_ in enumerate(r.photo_rows)]
                ans_each = vision_answers(vecs, ref, singles, comp["method"], b,
                                          int(config.get("top_k", 10)))
                per_photo[comp["id"]] = per_photo_table(finds, ans_each, reps, seed)
                outcomes[comp["id"]], inputs[comp["id"]] = {}, {
                    "kind": "vision", "backbone": b, "method": comp["method"],
                    "photos": "each", "photo_answers": len(ans_each)}
                top1[comp["id"]] = {}
                continue
            kw = {"top_k": int(config.get("top_k", 10)), "calibration": comp.get("calibration"),
                  "truths": truths}
            answers = (vision_answers_leave_out(vecs, ref, scored, comp["method"], b,
                                                comp["leave_out"], **kw)
                       if comp.get("leave_out") else
                       vision_answers(vecs, ref, scored, comp["method"], b, **kw))
            if comp.get("calibration"):
                calib[comp["id"]] = {k: a["cal"] for k, a in answers.items() if "cal" in a}
            path = out / "predictions" / f"{comp['id']}.jsonl"
            with open(path, "w", encoding="utf-8") as fh:
                for k in sorted(answers):
                    fh.write(json.dumps({"record_key": k, **answers[k]}) + "\n")
        outcomes[comp["id"]], inputs[comp["id"]] = outcomes_for(comp, finds, answers, conn)
        top1[comp["id"]] = top_answers(comp, answers)
        if comp["kind"] == "inat" and comp.get("calibration_from_score", True):
            calib[comp["id"]] = {str(line["record_key"]): po.inat_confidence(
                line, comp.get("score", "vision")) for line in po.read_jsonl(resolve(comp["file"]))}
        if answers is not None:
            inputs[comp["id"]].update(backbone=comp["backbone"], method=comp["method"],
                                      place=bool(comp.get("place")),
                                      reference_records=len(ref),
                                      reference=comp.get("reference", "every included photo"),
                                      finds_without_vectors=len(finds) - len(answers),
                                      predictions_sha256=sha256_file(path))
    gate_report = None
    if config.get("gate"):
        # E10: the gate on the scored finds and the OOD set; each Vision comparator gets a
        # "<id> served" twin where a refused find is wrong (the plain row is gate-off).
        from . import paper_gate
        gate_report, twins = paper_gate.run_gate(
            config["gate"], finds, photos, outcomes,
            [c["id"] for c in config["comparators"] if c["kind"] == "vision"],
            resolve, reps, seed)
        outcomes.update(twins)
        for cid in twins:
            inputs[cid] = {"kind": "vision served", "of": cid.rsplit(" served", 1)[0]}
    report = {
        "post_hoc": g["post_hoc"],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "guard": g, "code_commit": cfgmod.code_version(),
        "config_sha256": sha256_bytes(config_bytes),
        "dataset_release": dataset.cite(base),
        "scored": {"release": scored_rel.id, "split": split, "reference_splits": list(ref_splits),
                   **({"test_release": dataset.cite(scored_rel)} if scored_rel is not base else {})},
        "errata": {**(dataset.cite(base).get("errata") or {}),
                   "reference_records_left_out": sum(
                       1 for (k,) in base.db.execute(
                           "select record_key from records where included = 1 and split in "
                           f"({','.join('?' * len(ref_splits))})", ref_splits)
                       if str(k) in errata)},
        "attrs": {"file": str(attrs_path) if attrs_path else None,
                  "sha256": sha256_file(attrs_path) if attrs_path else None},
        "dhash_flags": {"source": "the scored release's near_base table",
                        "photos_hashed": len(near_base), "photos_flagged": len(dhash),
                        "near_base": (scored_rel.manifest.get("near_base")
                                      if scored_rel is not base else None)},
        "bootstrap": {"reps": reps, "seed": seed, "clusters": "observer"},
        "inputs": inputs,
        "finds": {"scored": len(finds), "by_name_kind": dict(Counter(f.name_kind for f in finds))},
        "standard": pt.all_standard(finds, outcomes, reps, seed),
        # The same tables on named subsets of the finds (config "subsets": {name: {attr:
        # value}}), e.g. val excluding the former held-out benchmark, the half that decides.
        "subsets": {name: {"filter": flt, **pt.all_standard(
            [f for f in finds if all(str(f.attrs.get(k)) == str(v) for k, v in flt.items())],
            outcomes, reps, seed)} for name, flt in (config.get("subsets") or {}).items()},
        "hypotheses": pt.families(config.get("hypotheses") or {}, finds, outcomes, reps, seed,
                                  float(config.get("alpha", 0.05))),
        "per_photo": per_photo,
        "gate": gate_report,
        "calibration": calibration_report(finds, calib, outcomes,
                                          config.get("calibration_metrics", {}), reps, seed),
        "errors": errors_report(finds, top1, {**config.get("errors", {}), **(
            {"deadly": config["deadly_taxa"]} if config.get("deadly_taxa")
            and not (config.get("errors") or {}).get("deadly") else {})}, base, scored_rel),
        "slices": pt.slices(finds, outcomes, config.get("slices", {}).get("attrs", []),
                            config.get("slices", {}).get("cells", ["species_strict@1",
                                                                   "genus_strict@1"]),
                            config.get("slices", {}).get("comparators", list(outcomes)),
                            reps, seed, int(config.get("slices", {}).get("min_finds", 30))),
    }
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    (out / "report.md").write_text(format_report(report, config), encoding="utf-8")
    log(f"paper-score: {len(finds):,} finds; report in {out}"
        + ("  *** POST HOC ***" if g["post_hoc"] else ""))
    return report


def per_photo_table(finds, answers_each: dict, reps: int, seed: int) -> dict:
    """E19, each photo alone: top-1 and top-5 per photo (species strict, genus strict) with
    the observer-clustered interval, primary scope; a photo counts toward its find's
    observer, so a find with many photos weighs more, as a photo-level number does."""
    by_key = {f.key: f for f in pt.in_scope(finds)}
    outs = {}
    for pk, a in answers_each.items():
        f = by_key.get(pk.split("#", 1)[0])
        if f is not None:
            outs[pk] = (f, po.from_ranked(a["species"], a["genus"], f))
    table = {}
    for cell in ("species_strict@1", "species_strict@5", "genus_strict@1", "genus_strict@5"):
        vals = [(f.observer, o[cell]) for f, o in outs.values() if o[cell] is not None]
        n, k = len(vals), sum(1 for _w, v in vals if v)
        table[cell] = {"photos": n, "right": k, "rate": k / n if n else None,
                       "ci95": ps.cluster_ci([v for _w, v in vals], [w for w, _v in vals],
                                             reps, seed) if n else None}
    return table


def top_answers(comp: dict, answers: dict | None) -> dict[str, str]:
    """Each find's top species name from a comparator ({} for the experts: one name per
    identifier, judged by the lane; no answer names are carried)."""
    if comp["kind"] == "vision":
        return {k: (a["species"] or [""])[0] for k, a in (answers or {}).items()}
    if comp["kind"] == "published":
        return {str(line["observation_id"]): ((line["result"].get("species") or [{}])[0]
                                              .get("name") or "")
                for line in po.read_jsonl(resolve(comp["file"]))}
    if comp["kind"] == "inat":
        sc = comp.get("score", "vision")
        return {str(line["record_key"]): ((line.get(sc) or {}).get("species") or [[0, ""]])[0][1]
                for line in po.read_jsonl(resolve(comp["file"]))}
    return {}


def taxonomy_ranks(base, scored_rel, path=None) -> tuple[dict[str, dict], str | None]:
    """genus -> {family, order}: iNat's taxonomy cache (taxonomy.py) where it has the genus,
    else the releases' own family with no order; and the cache file's sha256."""
    from . import taxonomy
    ranks: dict[str, dict] = {}
    for rel in {id(base): base, id(scored_rel): scored_rel}.values():
        for g, fam in rel.db.execute("select distinct genus, family from records "
                                     "where included = 1 and genus is not null"):
            if g and fam:
                ranks.setdefault(g, {"family": fam, "order": ""})
    cache = Path(path) if path else taxonomy.cache_path()
    tax = taxonomy.load(cache)
    if tax is not None:
        for g, r in tax.genera.items():
            ranks[g] = {"family": r.get("family", ""), "order": r.get("order", "")}
    return ranks, (sha256_file(cache) if cache.is_file() else None)


def errors_report(finds, top1: dict, cfg: dict, base, scored_rel) -> dict:
    """E29 and E21 for the comparators named (default: every one with answer names)."""
    from . import paper_errors as pe
    base_finds = pt.in_scope(finds)
    ranks, tax_sha = taxonomy_ranks(base, scored_rel, cfg.get("taxonomy"))
    deadly = pe.read_deadly({**cfg["deadly"], **({"file": str(resolve(cfg["deadly"]["file"]))}
                                                if "file" in cfg["deadly"] else {})}
                            ) if cfg.get("deadly") else None
    out = {"taxonomy_sha256": tax_sha, "comparators": {}}
    for cid in cfg.get("comparators") or [c for c, t in top1.items() if t]:
        t = top1.get(cid) or {}
        row = {"confused_pairs": pe.confused_pairs(base_finds, t, int(cfg.get("pairs", 20))),
               "distance": pe.mistake_distance(base_finds, t, ranks)}
        if deadly:
            row["deadly"] = pe.deadly_taxa(base_finds, t, deadly)
        out["comparators"][cid] = row
    return out


def calibration_report(finds, calib: dict, outcomes: dict, cfg: dict, reps: int,
                       seed: int) -> dict:
    """E13 and E11 for every comparator with confidences (paper_calibration), primary scope."""
    from . import paper_calibration as pc
    base = pt.in_scope(finds)
    return {cid: {rank: pc.summarise(base, cal, outcomes[cid], rank, cfg, reps, seed,
                                     lambda f: pt.band(f.depth))
                  for rank in ("species", "genus")            # the ranks with outcome cells
                  if any(rank in (c or {}) for c in cal.values())}
            for cid, cal in calib.items()}


# --- the readable report --------------------------------------------------------------------------

def _pct(r: dict) -> str:
    if not r or r.get("rate") is None:
        return "-"
    s = f"{100 * r['rate']:.1f}"
    if r.get("ci95"):
        s += f" ({100 * r['ci95'][0]:.1f}-{100 * r['ci95'][1]:.1f})"
    return s


def format_report(report: dict, config: dict) -> str:
    """Steve's format: per comparator (photos only and photos + place in separate groups),
    top 1/3/5/10 by species strict / s.l. / complex and genus strict / s.l., n on every
    table; species top-1 and top-5 by reference band; both denominators."""
    lines = []
    if report["post_hoc"]:
        lines += ["# *** POST HOC: run from a commit other than the frozen one ***", ""]
    sc = report["scored"]
    lines += [f"# Paper scores: {sc['release']} {sc['split']} against "
              f"{report['dataset_release']['dataset_release']} {'+'.join(sc['reference_splits'])}",
              "", f"Near-duplicates: {report['standard']['near_duplicates']}", ""]
    groups = {"photos only": [], "photos + place and date": []}
    for c in config["comparators"]:
        group = groups["photos + place and date" if c.get("place") or str(
            c.get("score", "")).startswith("combined") else "photos only"]
        group.append(c["id"])
        if f"{c['id']} served" in report["standard"]["comparators"]:
            group.append(f"{c['id']} served")         # the gate on: a refusal is wrong
    for label, cids in groups.items():
        if not cids:
            continue
        lines += [f"## {label}", ""]
        for cid in cids:
            comp = report["standard"]["comparators"][cid]["primary"]
            for den in pt.DENOMINATORS:
                t = comp[den]
                lines += [f"### {cid} ({den}: {t['finds']:,} finds, {t['answered']:,} answered)",
                          "", "| | top 1 | top 3 | top 5 | top 10 | n |", "|---|---|---|---|---|---|"]
                for row, cells in t["table"].items():
                    lines.append(f"| {row} | " + " | ".join(_pct(cells[f'top{k}'])
                                                           for k in po.K)
                                 + f" | {cells['top1']['n']:,} |")
                lines += ["", "| species by reference records | top 1 | top 5 | n |",
                          "|---|---|---|---|"]
                for b, r in t["species_by_reference_band"].items():
                    lines.append(f"| {b} | {_pct(r['top1'])} | {_pct(r['top5'])} | "
                                 f"{r['top1']['n']:,} |")
                lines.append("")
    if report.get("gate"):
        lines += _gate_lines(report["gate"])
    lines += ["## Hypotheses", ""]
    for fam, hyps in report["hypotheses"].items():
        lines += [f"### Family {fam} (Holm within the family)", "",
                  "| | n | difference / slope | p | p (Holm) | reject |", "|---|---|---|---|---|---|"]
        for hid, h in hyps.items():
            if "not_tested" in h:
                lines.append(f"| {hid} | - | not tested: {h['not_tested']} | - | - | - |")
                continue
            pts = lambda v: "-" if v is None else f"{100 * v:.1f}"  # noqa: E731
            if "slope" in h:
                eff, n = f"{h['slope']['estimate']:.3f} (se {h['slope']['se_robust']:.3f})", h["n"]
            elif "superiority" in h:
                eff = (f"{pts(h['superiority']['difference'])} pts; band lower bound "
                       f"{pts(h['noninferiority']['lower_bound'])}")
                n = h["superiority"]["n"]
            else:
                eff, n = f"{pts(h['difference'])} pts", h["n"]
            lines.append(f"| {hid} | {n:,} | {eff} | {h['p_test']:.4g} | {h['p_holm']:.4g} | "
                         f"{'yes' if h['reject'] else 'no'} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def _gate_lines(gr: dict) -> list[str]:
    """E10: the gate on the scored finds, and the OOD set's rows."""
    def share(x):
        if not x or x.get("share") is None:
            return "-"
        ci = x.get("ci95")
        return (f"{100 * x['share']:.1f}% ({x['count']:,}/{x['n']:,}"
                + (f"; {100 * ci[0]:.1f}-{100 * ci[1]:.1f}" if ci else "") + ")")
    t = gr["test"]
    g = gr["gate"]
    lines = ["## Not-a-fungus gate", "",
             f"Gate {g['mode']} {g['fingerprint']}, thresholds {g['thresholds']['not_fungus']}"
             f" / {g['thresholds']['unsure']}.", "",
             f"Scored finds: {t['finds']:,}; lose their answer {share(t['lost'])}; not sure "
             f"{share(t['not_sure'])}. Claim (at most {100 * t['claim']['max_lost_share']:.1f}% "
             f"lose their answer): {'stands' if t['claim']['stands'] else 'FAILS'}.", ""]
    ood = gr.get("ood")
    if ood:
        lines += ["| OOD photos | n | named | not sure | refused | TNR @ 95% TPR | AUROC |",
                  "|---|---|---|---|---|---|---|"]
        rows = [("everyday + non-fungal near-OOD", ood["headline"]),
                ("everyday", ood["everyday"]), ("non-fungal near-OOD", ood["near_ood_nonfungal"]),
                *ood["in_scope"].items()]
        for name, r in rows:
            tnr, au = r.get("tnr_at_95_tpr"), r.get("auroc")
            lines.append(f"| {name} | {r['photos']:,} | {share(r['named'])} | "
                         f"{share(r['not_sure'])} | {share(r['refused'])} | "
                         f"{'-' if tnr is None else f'{100 * tnr:.1f}%'} | "
                         f"{'-' if au is None else f'{au:.3f}'} |")
        chk = ood["everyday"]["check"]
        lines += ["", f"Everyday let-through at most {100 * chk['let_through_at_most']:.0f}%: "
                      f"{'holds' if chk['holds'] else 'does NOT hold'}.", ""]
    return lines
