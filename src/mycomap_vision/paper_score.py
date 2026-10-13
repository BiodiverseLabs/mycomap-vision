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
(a photo byte-identical to a reference photo, listed in the dHash flags file, or at cosine
>= the config's threshold to a reference photo on the frozen backbone). Comparators:

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
    for fam, hyps in (doc.get("hypotheses") or {}).items():
        for h in hyps:
            for k in ("a", "b", "comparator"):
                if k in h and h[k] not in ids:
                    raise ValueError(f"{fam}/{h.get('id')}: unknown comparator {h[k]!r}")


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

def read_errata(path) -> tuple[set[str], str | None]:
    from .backbone_screen import read_errata as read
    return read(path)


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
    """Per find: bytes (a photo's original sha256 is a reference photo's), dhash (listed in
    the flags file, by record key or photo key), cosine (a photo at or above `threshold`)."""
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
                   backbone: str, top_k: int = 10) -> dict[str, dict]:
    """Each scored record's ranked species and genus names from `method` fitted on the
    reference records (evaluate.fit_method; the records' photo_rows index `vectors`)."""
    from . import evaluate
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
        out[rec.observation_id] = ans
    return out


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

def outcomes_for(comp: dict, finds: list[po.Find], answers: dict | None = None) -> tuple[dict, dict]:
    """(outcomes by record key, input info) for one comparator."""
    by_key = {f.key: f for f in finds}
    info: dict = {"kind": comp["kind"]}
    if comp["kind"] == "vision":
        return {k: po.from_ranked(a["species"], a["genus"], by_key[k])
                for k, a in (answers or {}).items() if k in by_key}, info
    lines = po.read_jsonl(comp["file"])
    info.update(file=str(comp["file"]), sha256=sha256_file(comp["file"]), lines=len(lines))
    out = {}
    if comp["kind"] == "published":
        for line in lines:
            k = str(line["observation_id"])
            if k in by_key:
                out[k] = po.published_outcome(line, by_key[k])
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
        errata_path=None, dhash_path=None, attrs_path=None, allow_code: str | None = None,
        research_dir: Path | None = None, embed_missing=None, log=print) -> dict:
    """Score every comparator of `config` on `test` ("val" or a test release of `base`)
    and write report.json, report.md and the Vision predictions under `out`."""
    from .embed import load_embeddings
    from .replications.fungitastic import dataset
    scored_rel, split, ref_splits, sealed = scored_and_reference(base, test)
    g = guard(sealed, config_bytes, config, research_dir or cfgmod.RESEARCH_DIR,
              cfgmod.code_version(), allow_code)
    errata, errata_sha = read_errata(errata_path)
    reps, seed = int(config.get("bootstrap", {}).get("reps", 10_000)), \
        int(config.get("bootstrap", {}).get("seed", 0))
    ref_rows = [dict(r) for r in base.records(ref_splits) if str(r["record_key"]) not in errata]
    scored_rows = [dict(r) for r in scored_rel.records(split)]
    photos: dict[str, list[dict]] = defaultdict(list)
    for p in scored_rel.db.execute("select * from photos where included = 1 order by "
                                   "record_key, position, photo_key"):
        photos[str(p["record_key"])].append(dict(p))
    ref_keys = {str(r["record_key"]) for r in ref_rows}
    ref_photo_rows = [dict(p) for p in base.db.execute(
        "select manifest_photo_id, original_sha256, record_key from photos where included = 1")
        if str(p["record_key"]) in ref_keys]
    ref_shas = {p["original_sha256"] for p in ref_photo_rows if p["original_sha256"]}
    dhash = set()
    if dhash_path:
        dhash, _ = read_errata(dhash_path)
    nd_cfg = config.get("near_duplicates", {})
    cosine_of = None
    if nd_cfg.get("backbone"):
        b = nd_cfg["backbone"]
        want = [int(p["manifest_photo_id"]) for ph in photos.values() for p in ph]
        if embed_missing is not None:
            embed_missing(b, want)
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
    outcomes, inputs = {}, {}
    public = None
    for comp in config["comparators"]:
        answers = None
        if comp["kind"] == "vision":
            b = comp["backbone"]
            if embed_missing is not None:
                embed_missing(b, [int(p["manifest_photo_id"]) for ph in photos.values()
                                  for p in ph])
            ids, vecs = load_embeddings(conn, b)
            row = {int(p): i for i, p in enumerate(ids.tolist())}
            ref = [r for r in base.as_records(ref_splits, row) if r.observation_id not in errata]
            scored = scored_rel.as_records(split, row)
            if comp.get("place"):
                public = public if public is not None else public_points(scored_rel)
                scored = with_public_points(scored, public)
            log(f"[{comp['id']}] {b} / {comp['method']}: {len(scored):,} finds with vectors, "
                f"{len(ref):,} reference records")
            answers = vision_answers(vecs, ref, scored, comp["method"], b,
                                     int(config.get("top_k", 10)))
            path = out / "predictions" / f"{comp['id']}.jsonl"
            with open(path, "w", encoding="utf-8") as fh:
                for k in sorted(answers):
                    fh.write(json.dumps({"record_key": k, **answers[k]}) + "\n")
        outcomes[comp["id"]], inputs[comp["id"]] = outcomes_for(comp, finds, answers)
        if answers is not None:
            inputs[comp["id"]].update(backbone=comp["backbone"], method=comp["method"],
                                      place=bool(comp.get("place")),
                                      finds_without_vectors=len(finds) - len(answers),
                                      predictions_sha256=sha256_file(path))
    report = {
        "post_hoc": g["post_hoc"],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "guard": g, "code_commit": cfgmod.code_version(),
        "config_sha256": sha256_bytes(config_bytes),
        "dataset_release": dataset.cite(base),
        "scored": {"release": scored_rel.id, "split": split, "reference_splits": list(ref_splits),
                   **({"test_release": dataset.cite(scored_rel)} if scored_rel is not base else {})},
        "errata": {"file": str(errata_path) if errata_path else None, "sha256": errata_sha,
                   "reference_records_left_out": len(errata & {str(r["record_key"]) for r in
                                                               base.records(ref_splits)})},
        "dhash_flags": {"file": str(dhash_path) if dhash_path else None,
                        "sha256": sha256_file(dhash_path) if dhash_path else None},
        "bootstrap": {"reps": reps, "seed": seed, "clusters": "observer"},
        "inputs": inputs,
        "finds": {"scored": len(finds), "by_name_kind": dict(Counter(f.name_kind for f in finds))},
        "standard": pt.all_standard(finds, outcomes, reps, seed),
        "hypotheses": pt.families(config.get("hypotheses") or {}, finds, outcomes, reps, seed,
                                  float(config.get("alpha", 0.05))),
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
        groups["photos + place and date" if c.get("place") or str(c.get("score", "")).startswith("combined")
               else "photos only"].append(c["id"])
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
