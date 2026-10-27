"""`mv paper-score`: every table of the paper's sealed-test analysis from one command.

    mv paper-score --release v1 --test v1-test --config <research>/paper/analysis-v1.json

The analysis plan (the config: comparators, methods, hypotheses, slices, bootstrap) is
fixed before the sealed test is opened. On the sealed test this command is the only thing
that predicts, embeds or scores, and it refuses to run unless:

- the tag `paper-analysis-freeze-v1` exists in the research repo (config.RESEARCH_DIR),
- the config given is byte-identical to that tag's `paper/analysis-v1.json` (CRLF line
  endings read as LF, as git stores it: one sha256 from any checkout), and
- this code is the commit the config names (`code_commit`), unless `--allow-code <sha>`
  names the commit running, which marks every output "post hoc" in capitals.

The one exception to "byte-identical" is the tagged config's `post_freeze` list: the fields
it names (each a "TBD..." placeholder at the tag, with the rule that fills it) may be filled
after the tag, and the freeze block may gain parts (never change one). Every other difference
is refused, and each filled field is recorded with its sha256 (PLAN §8).

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
- `expert`: its E31 scores.jsonl (`tier`: 1, 2 or pooled 1+2; `view`: first, best or
  community; `strict`).

Outputs: report.json (every table, test and input hash), report.md (Steve's tables),
predictions/<comparator>.jsonl (each Vision answer, for audit).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from contextlib import contextmanager
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
EXPERT_TIERS = {"1", "2"}      # E31: >= 10,000 and 1,000-9,999 iNat IDs; "1+2" pools them
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
    """The config and its bytes as git stores them. Research pins *.json to LF line endings
    (.gitattributes), so a copy rewritten with CRLF (an editor, a builder on Windows) is
    taken with LF: its sha256, the report's config_sha256, the checkpoint stamps and the
    sealed guard's comparison with the tagged config are then the one value a clean checkout
    gives. A carriage return anywhere else is refused."""
    body = Path(path).read_bytes().replace(b"\r\n", b"\n")
    if b"\r" in body:
        raise ValueError(f"{path}: a carriage return outside a CRLF line ending; save the "
                         "config with LF line endings")
    doc = json.loads(body)
    check_config(doc)
    return doc, body


def config_sha_line(path, body: bytes) -> str:
    """What paper-score prints about the config: the sha256 of the bytes it scores with (LF,
    as git stores them), and whether the file on disk has CRLF line endings."""
    crlf = b"\r\n" in Path(path).read_bytes()
    return (f"config {Path(path).name}: sha256 {sha256_bytes(body)} (LF line endings, as git "
            "stores it" + ("; the file on disk has CRLF line endings, read as LF" if crlf else "")
            + ")")


def check_config(doc: dict) -> None:
    if doc.get("format") != CONFIG_FORMAT:
        raise ValueError(f"analysis config format {doc.get('format')!r}, expected {CONFIG_FORMAT}")
    from . import name_equiv
    if doc.get("answer_key") not in name_equiv.ANSWER_KEYS:
        raise ValueError(f"the config names its answer key (\"answer_key\": one of "
                         f"{name_equiv.ANSWER_KEYS}), not {doc.get('answer_key')!r}")
    ids = [c["id"] for c in doc.get("comparators", [])]
    if len(ids) != len(set(ids)):
        raise ValueError("comparator ids must be unique")
    check_post_freeze(doc)
    # A post_freeze field still a "TBD..." placeholder is checked once it is filled (the
    # sealed run's config), not before; the rest of its comparator is checked now.
    pending = {e["path"] for e in doc.get("post_freeze") or []
               if is_placeholder(config_path_get(doc, e["path"]))}
    for c in doc.get("comparators", []):
        def later(key, c=c):
            return f"comparators/{c.get('id')}/{key}" in pending
        if c.get("kind") not in KINDS:
            raise ValueError(f"comparator {c.get('id')}: kind is one of {KINDS}")
        if c["kind"] == "vision" and not (c.get("backbone") and c.get("method")):
            raise ValueError(f"comparator {c['id']}: a vision comparator names backbone and method")
        if c.get("answer_key") not in (None, doc["answer_key"], "exact"):
            raise ValueError(f"comparator {c['id']}: a comparator's own answer_key is only "
                             "\"exact\" (v1's published rule, a sensitivity row)")
        if c["kind"] == "vision" and not later("method"):
            # Any method the scoring code knows (an adopted E32 arm, e.g.
            # nearest+mean-records-balanced, once it is in methods.METHODS); a typo is
            # refused here, never first met when the sealed test is opened.
            from .evaluate import METHODS
            if c["method"] not in METHODS:
                raise ValueError(f"comparator {c['id']}: unknown method {c['method']!r}")
            from . import twins as twins_mod
            if twins_mod.is_grouped(c["method"]) and not doc.get("twins"):
                raise ValueError(f"comparator {c['id']}: {c['method']} scores by reference "
                                 "record, so the config pins its cross-post twin groups "
                                 "(a top-level \"twins\" block)")
        if not later("reference") and c.get("reference") not in REFERENCES:
            raise ValueError(f"comparator {c['id']}: reference is one of "
                             f"{[r for r in REFERENCES if r]} (or none: every included photo), "
                             f"not {c.get('reference')!r}")
        if c.get("reference") and c["kind"] != "vision":
            raise ValueError(f"comparator {c['id']}: a reference is for a Vision comparator")
        if (c["kind"] == "vision" and not later("method") and "place" in c["method"]
                and not c.get("place")):
            # Without "place": true the scored records keep the release's true point.
            raise ValueError(f"comparator {c['id']}: {c['method']} reads each find's place, "
                             "so it takes \"place\": true (the public point, never the "
                             "release's true one)")
        if c.get("place_setting") and not later("place_setting") and not later("method"):
            if "place" not in str(c.get("method", "")):
                raise ValueError(f"comparator {c['id']}: place_setting is for a place method, "
                                 f"not {c.get('method')!r}")
            from .replications.fungitastic.retrain import CLASSIFIER_METHODS
            if c["method"] in {m.name for m in CLASSIFIER_METHODS}:
                # Its place prior is the model's own training counts (retrain.MetadataPrior),
                # so a placeprior setting would only be refused after the test was read.
                raise ValueError(f"comparator {c['id']}: {c['method']}'s place prior is the "
                                 "trained model's own counts, not a place setting: leave "
                                 "place_setting out")
            from . import placeprior
            _, setting = placeprior.place_setting(c["place_setting"])   # refuses an unknown name
            if "inat" in str(setting.get("pen_known", "")):
                # PlaceBoost keeps no pair counts, so nearest_inat_km cannot take a find's
                # own iNat observation out: such a setting would leak (score it through
                # placetune's TermBuilder instead).
                raise ValueError(f"comparator {c['id']}: place setting {c['place_setting']!r} "
                                 "reads iNat distances, which cannot leave a find's own "
                                 "observation out here")
        if c["kind"] == "expert" and c.get("view") != "community" and not (
                set(str(c.get("tier", "1")).split("+")) <= EXPERT_TIERS):
            raise ValueError(f"comparator {c['id']}: expert tier is 1, 2 or 1+2, "
                             f"not {c.get('tier')!r}")
        if c["kind"] in ("published", "inat", "expert") and not c.get("file"):
            raise ValueError(f"comparator {c['id']}: needs its input file")
    check_place_pins(doc)
    psr = doc.get("per_species_rows")
    if psr is not None:
        unknown = set(psr.get("metrics") or []) - set(pt.PER_SPECIES_METRICS)
        if unknown:
            raise ValueError(f"per_species_rows: unknown metrics {sorted(unknown)} (known: "
                             f"{', '.join(pt.PER_SPECIES_METRICS)})")
        if not str(psr.get("cell", "species_strict@1")).startswith("species_"):
            raise ValueError("per_species_rows: the cell is a species cell")
    gate = doc.get("gate")
    if gate:
        for k in ("dir", "fingerprint"):
            if not gate.get(k):
                raise ValueError(f"gate: names {k}")
        if bool(gate.get("test_vectors")) == bool(gate.get("embed_with")):
            raise ValueError("gate: names one of test_vectors (a file) or embed_with "
                             "(the served backbone whose fork the run embeds through)")
        # The gate's served twins ("<id> served") may be named by a hypothesis.
        ids = ids + [f"{c['id']} served" for c in doc.get("comparators", [])
                     if c.get("kind") == "vision"]
    for fam, hyps in (doc.get("hypotheses") or {}).items():
        for h in hyps:
            for k in ("a", "b", "comparator"):
                if k in h and h[k] not in ids:
                    raise ValueError(f"{fam}/{h.get('id')}: unknown comparator {h[k]!r}")
    check_secondary_rows(doc)


def check_secondary_rows(doc: dict) -> None:
    """The secondary_rows block (E14, Steve 27 Oct): rows of known conditions, hypotheses
    that exist, created_after an ISO date or {"config": path} naming a field that exists."""
    block = doc.get("secondary_rows")
    if block is None:
        return
    from . import paper_tables as ptab
    hyp_ids = {h["id"] for hs in (doc.get("hypotheses") or {}).values() for h in hs}
    for hid in block.get("for") or []:
        if hid not in hyp_ids:
            raise ValueError(f"secondary_rows: unknown hypothesis {hid!r}")
    ids = [r.get("id") for r in block.get("rows") or []]
    if not ids or None in ids or len(ids) != len(set(ids)):
        raise ValueError("secondary_rows: every row has a unique id")

    def check(where, rid):
        for k, v in where.items():
            if k not in ptab.WHERE_KEYS:
                raise ValueError(f"secondary_rows {rid}: unknown condition {k!r} (one of "
                                 f"{ptab.WHERE_KEYS})")
            if k == "any_of":
                if not isinstance(v, list) or not v:
                    raise ValueError(f"secondary_rows {rid}: any_of is a list of conditions")
                for w in v:
                    check(w, rid)
            elif k == "created_after":
                if isinstance(v, dict):
                    try:
                        config_path_get(doc, v["config"])
                    except KeyError:
                        raise ValueError(f"secondary_rows {rid}: created_after names "
                                         f"{v.get('config')!r}, no such field") from None
                elif not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(v)):
                    raise ValueError(f"secondary_rows {rid}: created_after is a date "
                                     "(YYYY-MM-DD) or {\"config\": path}")
    for r in block.get("rows") or []:
        check(r.get("where") or {}, r["id"])


def resolved_secondary_rows(config: dict) -> dict:
    """The block with each {"config": path} date read from the config; a row whose date is
    still a TBD placeholder is marked not run."""
    block = json.loads(json.dumps(config.get("secondary_rows") or {}))

    def fill(where):
        for k, v in list(where.items()):
            if k == "any_of":
                for w in v:
                    fill(w)
            elif k == "created_after" and isinstance(v, dict):
                val = config_path_get(config, v["config"])
                if is_placeholder(val):
                    raise LookupError(v["config"])
                where[k] = str(val)
    for r in block.get("rows") or []:
        try:
            fill(r.get("where") or {})
        except LookupError as e:
            r["not_run"] = f"not run: TBD ({e.args[0]} is still a placeholder)"
    return block


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


# The twins block's file for each reference: val is scored against train, the sealed test
# against train + val (each file holds the groups among those splits only).
TWINS_FOR_REFERENCE = {("train",): "validation", ("train", "val"): "sealed_test"}


def twin_groups(block: dict | None, ref_splits) -> tuple[dict | None, dict | None]:
    """({observation_id: specimen}, cite) from the config's "twins" block for a reference
    of `ref_splits`: its "validation" file for train, its "sealed_test" file for train +
    val. Refused when the block has no file for this reference, the file is missing, or
    its sha256 is not the one pinned. (None, None) without a block."""
    if not block:
        return None, None
    which = TWINS_FOR_REFERENCE.get(tuple(ref_splits))
    entry = block.get(which) if which else None
    if not entry or not entry.get("file") or str(entry["file"]).startswith("TBD"):
        raise ValueError(f"twins: no {which or '?'} file pinned for a reference of "
                         f"{' + '.join(ref_splits)}")
    path = resolve(entry["file"])
    if not path.is_file():
        raise ValueError(f"twins: {path} is missing")
    got = sha256_file(path)
    if got != entry.get("sha256"):
        raise ValueError(f"twins {path.name}: sha256 {got[:12]}, the config pins "
                         f"{str(entry.get('sha256'))[:12]}")
    from . import twins as twins_mod
    return twins_mod.read(path), {"for": which, **twins_mod.cite(path)}


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


# --- fields filled after the tag (post_freeze) ---------------------------------------------------

# PLAN §8: test-time outputs made at scoring by the pinned code (E14, E31a, the external
# models' v1-test files, the attrs file), and models and lists declared before the tag but
# finished after it by a pre-declared rule (E34 / E35).
POST_FREEZE_KINDS = ("test_output", "declared_after_freeze")
POST_FREEZE_PROTECTED = ("code_commit", "post_freeze", "freeze")


def is_placeholder(value) -> bool:
    return isinstance(value, str) and value.startswith("TBD")


def config_path_get(doc, path: str):
    """The value at `path` in the config: segments split by "/"; inside a list of objects a
    segment picks the one entry whose "id" it is. KeyError when nothing is there, or when the
    path does not end at an object's key."""
    parts = str(path).split("/")
    if not path or any(not seg for seg in parts):
        raise KeyError(path)
    node = doc
    for i, seg in enumerate(parts):
        if isinstance(node, dict) and seg in node:
            node = node[seg]
        elif isinstance(node, list) and i < len(parts) - 1:
            hits = [x for x in node if isinstance(x, dict) and x.get("id") == seg]
            if len(hits) != 1:
                raise KeyError(path)
            node = hits[0]
        else:
            raise KeyError(path)
    return node


def check_post_freeze(doc: dict) -> None:
    """The config's post_freeze list is well formed: each entry names a path that exists, the
    rule that fills it and its kind; no path is protected, repeated, or inside another."""
    entries = doc.get("post_freeze")
    if entries is None:
        return
    if not isinstance(entries, list):
        raise ValueError("post_freeze: a list of {path, rule, kind}")
    paths = []
    for e in entries:
        path = e.get("path") if isinstance(e, dict) else None
        if not isinstance(path, str) or not path:
            raise ValueError(f"post_freeze: every entry names a path ({e!r})")
        if not str(e.get("rule") or "").strip():
            raise ValueError(f"post_freeze {path}: names the rule that fills it")
        if e.get("kind") not in POST_FREEZE_KINDS:
            raise ValueError(f"post_freeze {path}: kind is one of {POST_FREEZE_KINDS}")
        if path.split("/")[0] in POST_FREEZE_PROTECTED:
            raise ValueError(f"post_freeze {path}: {path.split('/')[0]} is never filled after "
                             "the tag")
        try:
            config_path_get(doc, path)
        except KeyError:
            raise ValueError(f"post_freeze {path}: no such field in the config (the path ends "
                             "at an object's key; a list entry is picked by its id)") from None
        paths.append(path)
    for a in paths:
        for b in paths:
            if a == b and paths.count(a) > 1:
                raise ValueError(f"post_freeze {a}: named twice")
            if a != b and b.startswith(a + "/"):
                raise ValueError(f"post_freeze {b}: inside {a}, also named")


def post_freeze_differences(tagged: dict, given: dict) -> list[str]:
    """Where `given` differs from the tagged config other than in the tagged post_freeze
    fields and by parts added to the freeze block (empty: the same plan). Key sets count: a
    key added or dropped anywhere else is a difference."""
    allowed = {e["path"] for e in tagged.get("post_freeze") or []}
    out: list[str] = []

    def grow(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in a:
                if k not in b:
                    out.append(f"{path}/{k}: frozen at the tag, missing now")
                else:
                    grow(a[k], b[k], f"{path}/{k}")
        elif a != b:
            out.append(f"{path}: changed since the tag (the freeze block may only gain parts)")

    def walk(a, b, path):
        if path in allowed:
            return
        if path == "freeze":
            if a is None:
                out.append("freeze: no freeze block in the tagged config")
            else:
                grow(a, b, "freeze")
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                p = f"{path}/{k}" if path else k
                if k not in b:
                    out.append(f"{p}: in the tagged config, missing now")
                elif k not in a:
                    out.append(f"{p}: not in the tagged config")
                else:
                    walk(a[k], b[k], p)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                out.append(f"{path}: {len(a)} entries tagged, {len(b)} now")
                return
            for i, (x, y) in enumerate(zip(a, b)):
                xid = x.get("id") if isinstance(x, dict) else None
                yid = y.get("id") if isinstance(y, dict) else None
                if xid != yid:
                    out.append(f"{path}/{i}: entry {xid!r} tagged, {yid!r} now")
                    continue
                walk(x, y, f"{path}/{xid if xid is not None else i}")
        elif a != b:
            out.append(f"{path}: changed since the tag")

    walk(tagged, given, "")
    return out


def post_freeze_records(tagged: dict, given: dict) -> list[dict]:
    """Per tagged post_freeze entry, what the scoring record keeps: its path, kind and rule,
    whether it is filled, and the sha256 of its value (canonical JSON)."""
    from .paper_freeze import canonical_sha
    out = []
    for e in tagged.get("post_freeze") or []:
        value = config_path_get(given, e["path"])
        filled = not is_placeholder(value)
        out.append({"path": e["path"], "kind": e["kind"], "rule": e["rule"], "filled": filled,
                    **({"value_sha256": canonical_sha(value)} if filled else {})})
    return out


def post_freeze_files(config: dict, records: list[dict], resolve_path) -> None:
    """A filled post_freeze ".../file" field: the file's own sha256 goes into its record. When
    the object beside it pins a sha256 (filled, or fixed at the tag), the file must match."""
    from .paper_freeze import sha256_file
    for rec in records:
        if not (rec["filled"] and rec["path"].endswith("/file")):
            continue
        p = Path(resolve_path(config_path_get(config, rec["path"])))
        if not p.is_file():
            raise FrozenError(f"post_freeze {rec['path']}: {p.name} not found, so its sha256 "
                              "cannot be recorded")
        rec["file_sha256"] = sha256_file(p)
        try:
            pinned = config_path_get(config, rec["path"][:-len("file")] + "sha256")
        except KeyError:
            continue
        if not is_placeholder(pinned) and pinned != rec["file_sha256"]:
            raise FrozenError(f"post_freeze {rec['path']}: the file's sha256 "
                              f"{rec['file_sha256'][:12]} is not the {str(pinned)[:12]} "
                              "pinned beside it")


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
    tagged_doc = json.loads(tagged.stdout)
    # The tagged post_freeze fields are placeholders: filling one never changes a value the
    # tag decided.
    try:
        check_post_freeze(tagged_doc)
    except ValueError as e:
        raise FrozenError(f"the tagged config's post_freeze list: {e}") from None
    for e in tagged_doc.get("post_freeze") or []:
        if not is_placeholder(config_path_get(tagged_doc, e["path"])):
            raise FrozenError(f"post_freeze {e['path']}: not a \"TBD...\" placeholder in the "
                              f"config tagged {TAG}")
    if tagged.stdout != config_bytes:
        crlf = tagged.stdout.replace(b"\r", b"") == config_bytes.replace(b"\r", b"")
        if crlf:
            raise FrozenError(f"the config given is not the one tagged {TAG} (line endings "
                              f"differ: give the tagged bytes, as git show "
                              f"{TAG}:{CONFIG_IN_REPO} prints them)")
        diff = post_freeze_differences(tagged_doc, config)
        if diff:
            raise FrozenError(f"the config given is not the one tagged {TAG} outside its "
                              f"post_freeze fields (sha256 {sha256_bytes(config_bytes)[:12]} vs "
                              f"{sha256_bytes(tagged.stdout)[:12]}): " + "; ".join(diff[:8])
                              + (f" (+{len(diff) - 8} more)" if len(diff) > 8 else ""))
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
            "frozen_code_commit": frozen,
            **({"tagged_config_sha256": sha256_bytes(tagged.stdout),
                "post_freeze": post_freeze_records(tagged_doc, config)}
               if "post_freeze" in tagged_doc else {})}


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
    from . import val_review as vr
    # PLAN rule 7: depth by answer key (a species pooled with its varieties; a hybrid
    # code its own), the one definition (name_equiv.ReferenceDepth)
    from .name_equiv import ReferenceDepth
    depth = ReferenceDepth(r["species"] for r in ref_rows if r.get("species"))
    unseen = vr.unseen_observer([r.get("observer_login") for r in scored_rows],
                                [r.get("observer_login") for r in ref_rows])
    out = []
    for r, is_new in zip(scored_rows, unseen.tolist()):
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
                 "observer_new": "new" if is_new else "known"}
        attrs.update((extra_attrs or {}).get(key, {}))
        species = r.get("species") or ""
        out.append(po.Find(key, who or f"(unknown observer):{key}", species,
                           r.get("genus") or "", r.get("family") or "",
                           r.get("name_kind") or ("formal" if species else "one-word"),
                           depth(species), len(ph), attrs,
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


def val_dhash_flags(base, block: dict, photos: dict[str, list[dict]],
                    ref_photo_rows: list[dict]) -> set[str]:
    """Photo keys of the scored val photos within dHash distance 4 of a reference photo,
    from the base release's dHash index (pinned by sha256, checked against the release)."""
    from . import dhash as dh
    from . import val_review as vr
    path = resolve(block["file"])
    got = sha256_file(path)
    if got != block.get("sha256"):
        raise ValueError(f"dHash index {path}: sha256 {got[:12]}, the config pins "
                         f"{str(block.get('sha256'))[:12]}")
    index = dh.load_index(path)
    dh.check_index(index, base)
    of_key = dict(zip(index["photo_key"], index["dhash"].tolist()))
    queries = {str(p["photo_key"]): np.uint64(of_key[str(p["photo_key"])])
               for ph in photos.values() for p in ph if str(p["photo_key"]) in of_key}
    train = np.asarray([of_key[str(p["photo_key"])] for p in ref_photo_rows
                        if str(p["photo_key"]) in of_key], dtype=np.uint64)
    flags = vr.near_train_flags(queries, train, chunk=64) if queries and len(train) else {}
    return {pk for pk, f in flags.items() if f}


def build_exclusions(rel) -> dict:
    """The test release's own exclusion counts by source and reason (records and photos),
    from its manifest; every reason but `ok` is something the build left out."""
    counts = (rel.manifest or {}).get("counts") or {}
    out = {}
    for what in ("records", "photos"):
        by = counts.get(f"{what}_by_source_reason") or {}
        left = {k: v for k, v in sorted(by.items()) if not k.endswith("/ok")}
        out[what] = {"by_source_reason": left, "left_out": sum(left.values()),
                     "total": counts.get(what)}
    return out


def read_created(block: dict) -> dict[str, str]:
    """{record key: the iNat observation's created_at}, from the pinned file of the config's
    generalisation.pretraining block (JSONL or TSV with record_key and created_at)."""
    import csv
    path = resolve(block["file"])
    got = sha256_file(path)
    if got != block.get("sha256"):
        raise ValueError(f"created_at file {path}: sha256 {got[:12]}, the config pins "
                         f"{str(block.get('sha256'))[:12]}")
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".tsv":
        rows = list(csv.DictReader(text.splitlines(), delimiter="	"))
    else:
        rows = [json.loads(x) for x in text.splitlines() if x.strip()]
    return {str(r["record_key"]): str(r.get("created_at") or "") for r in rows}


GENERALISATION_ROWS = {"unseen observers": {"observer_new": "new"},
                       "recent cohort": {"recent_cohort": "yes"}}


def add_generalisation_attrs(finds, scored_rows: list[dict], recent_after: str | None,
                             created: dict[str, str] | None, created_after: str | None) -> None:
    """recent_cohort (observed after `recent_after`: the base release's val cutoff, so on
    v1-test the finds observed after v1's newest records) and after_pretraining (the iNat
    observation created after `created_after`, the backbone's pretraining snapshot), as
    yes / no / (no date)."""
    from . import val_review as vr
    observed = {str(r["record_key"]): (r.get("observed_on") or "")[:10] for r in scored_rows}
    dates = [observed.get(f.key, "") for f in finds]
    recent = (vr.recent_cohort(dates, recent_after).tolist() if recent_after
              else [False] * len(finds))
    for f, d, is_recent in zip(finds, dates, recent):
        f.attrs["recent_cohort"] = ("(no date)" if not d[:4].isdigit() or not recent_after
                                    else "yes" if is_recent else "no")
        if created is not None:
            # Only iNat finds are in the snapshot (an MO find falls outside the iNat/GBIF
            # export cutoffs); an observation gone from iNat has no date.
            c = (created.get(f.key) or "")[:10]
            f.attrs["after_pretraining"] = ("(not in the snapshot)" if f.key not in created
                                            else "(no date)" if not c[:4].isdigit()
                                            else "yes" if c > created_after else "no")


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

@contextmanager
def place_setting_used(name: str | None):
    """The named place setting (placeprior.PLACE_SETTINGS) in force while a place method is
    built: MV_PLACE_SETTING set to it, then restored. An unknown name is refused; None
    leaves the environment's own (recorded by the caller)."""
    if not name:
        yield
        return
    from . import placeprior
    placeprior.place_setting(name)                    # refuses an unknown name
    env = placeprior.SETTING_ENV
    before = os.environ.get(env)
    os.environ[env] = name
    try:
        yield
    finally:
        if before is None:
            os.environ.pop(env, None)
        else:
            os.environ[env] = before


# A Vision comparator's reference: every included photo (none given) or the photos the
# public model may learn from (O1: the headline public model with the openly licensed
# reference). Its finds' depth and "observer new" are counted in that reference.
REFERENCES = (None, "public_trainable_only")


def reference_rows(base, ref_splits, reference: str | None) -> list[dict]:
    """The reference records as release rows: every included record of `ref_splits`, or
    with "public_trainable_only" those with at least one public-trainable included photo
    (as Release.as_records keeps them)."""
    rows = [dict(r) for r in base.records(ref_splits)]
    if reference != "public_trainable_only":
        return rows
    keep = {str(k) for (k,) in base.db.execute(
        "select distinct record_key from photos where included = 1 and public_trainable = 1")}
    return [r for r in rows if str(r["record_key"]) in keep]


def finds_in_reference(finds: list, scored_rows: list[dict], ref_rows: list[dict]) -> list:
    """The same finds with depth (ReferenceDepth, answer-key units) and "observer_new"
    counted in another reference (`ref_rows`); every other attribute as it was."""
    from dataclasses import replace

    from . import val_review as vr
    from .name_equiv import ReferenceDepth
    depth = ReferenceDepth(r["species"] for r in ref_rows if r.get("species"))
    new = vr.unseen_observer([r.get("observer_login") for r in scored_rows],
                             [r.get("observer_login") for r in ref_rows])
    new_of = {str(r["record_key"]): bool(n) for r, n in zip(scored_rows, new.tolist())}
    return [replace(f, depth=depth(f.species),
                    attrs={**f.attrs, "observer_new": "new" if new_of.get(f.key) else "known"})
            for f in finds]


def check_place_pins(doc: dict) -> None:
    """Every comparator's place_setting is pinned by the config's top-level "place_settings"
    block ({name: {sha256, ...}}): the sha256 of the setting's values as the code has them
    (placeprior.place_setting; paper_freeze.canonical_sha, as the freeze spec hashes them).
    Refused when the name has no entry or the values differ, so a setting changed in code
    after the decision is never scored under the decided name."""
    from . import placeprior
    from .paper_freeze import canonical_sha
    pins = doc.get("place_settings") or {}
    for c in doc.get("comparators", []):
        name = c.get("place_setting")
        if not name:
            continue
        pin = pins.get(name)
        if not isinstance(pin, dict) or not pin.get("sha256"):
            raise ValueError(f"comparator {c['id']}: place setting {name!r} has no entry in "
                             "the config's \"place_settings\" block (its sha256 pins the "
                             "values)")
        got = canonical_sha(placeprior.place_setting(name)[1])
        if got != pin["sha256"]:
            raise ValueError(f"comparator {c['id']}: place setting {name!r} is {got[:12]} in "
                             f"this code, the config pins {str(pin['sha256'])[:12]}")


def recoverability_units(base, ref_rows: list[dict], rule: str) -> tuple[dict, dict]:
    """PLAN C4's denominators as {name: [(reference records, formal?)]}, (b) first: every
    species name in the base release (all its records, included or not; a name with no
    reference record at 0), and (a) the species of `ref_rows`; both by the answer key `rule`
    (a species pools with its varieties). Also their counts, for the report."""
    from .name_equiv import ReferenceDepth
    from .name_equiv import answer_key as _key
    rd = ReferenceDepth((r["species"] for r in ref_rows if r.get("species")), rule)
    formal_of: dict[str, bool] = {}

    def note(species, kind):
        k = _key(species, rule)
        if k:
            formal_of[k] = formal_of.get(k, False) or kind == "formal"
        return k
    seen = {note(r["species"], r.get("name_kind")) for r in ref_rows if r.get("species")}
    known = {note(sp, kind) for sp, kind in base.db.execute(
        "select species, name_kind from records where species is not null and species != ''")}
    known |= seen
    seen.discard("")
    known.discard("")
    units = {pt.KNOWN: [(rd.counts.get(k, 0), formal_of.get(k, False)) for k in sorted(known)],
             pt.SEEN: [(rd.counts[k], formal_of.get(k, False)) for k in sorted(seen)]}
    return units, {name: {"species": len(u), "with_reference": sum(d > 0 for d, _ in u)}
                   for name, u in units.items()}


def vision_answers(vectors: np.ndarray, ref_records: list, scored_records: list, method: str,
                   backbone: str, top_k: int = 10, calibration: dict | None = None,
                   truths: dict[str, dict] | None = None, place_setting: str | None = None,
                   info: dict | None = None, twins: dict | None = None) -> dict[str, dict]:
    """Each scored record's ranked species and genus names from `method` fitted on the
    reference records (evaluate.fit_method; the records' photo_rows index `vectors`).

    With `calibration` ({rank: {temperature, floor}}, fixed on val, never refitted here)
    each answer also carries, per calibrated rank, `conf` (the top name's probability),
    `p_true` (the true name's, from `truths` {key: {rank: name}}; 0 when the reference
    lacks it), `sumsq` (for the multiclass Brier score) and `set` (the likely set at the
    floor, likely.likely_set; None without a floor).

    `place_setting` names the place setting a place method is built with; `info` gets the
    setting the fitted method actually uses. `twins` ({observation_id: specimen}, the
    config's groups file): a record-grouped method counts a cross-post pair as one specimen."""
    from . import evaluate
    index, model = fitted_method(vectors, ref_records, method, backbone, place_setting, info,
                                 twins)
    uses_context = getattr(model, "needs_context", False)

    def score(rec):
        rows = rec.photo_rows
        return (model.species_scores(vectors[rows], evaluate.context_of(rec)) if uses_context
                else model.species_scores(vectors[rows]))
    return answers_from(index, scored_records, score, top_k, calibration, truths)


def fitted_method(vectors, ref_records, method, backbone, place_setting=None, info=None,
                  twins: dict | None = None):
    """(index, model) for `method` fitted on the reference records, with the named place
    setting built in (refused if the method ends up with another). A record-grouped method
    (twins.is_grouped) groups the cross-post twins of `twins` present in this reference
    (twins.specimen_of: a lone twin is an ordinary record), and is refused without them."""
    from . import evaluate
    from . import twins as twins_mod
    specimen_of = None
    if twins_mod.is_grouped(method):
        if twins is None:
            raise ValueError(f"{method} scores by reference record: give the config's twin "
                             "groups")
        specimen_of = twins_mod.specimen_of(twins, ref_records)
    if info is not None and specimen_of is not None:
        info["twin_records"] = len(specimen_of)
    with place_setting_used(place_setting):
        index, model = evaluate.fit_method(
            vectors, ref_records, method, backbone,
            **({"specimen_of": specimen_of} if specimen_of is not None else {}))
    used = getattr(getattr(model, "prior", None), "setting_name", None)
    if place_setting and used != place_setting:
        raise ValueError(f"{method} was built with place setting {used!r}, not {place_setting!r}")
    if info is not None and used is not None:
        info["place_setting"] = used
    return index, model


def answers_from(index, scored_records, score, top_k: int = 10,
                 calibration: dict | None = None, truths: dict | None = None) -> dict:
    """Each scored record's ranked names (and calibration figures) from `score(rec)`, the
    record's species scores over index.species (-inf: a species absent for this record)."""
    from . import evaluate, likely
    position = {rank: {n: i for i, n in enumerate(index.labels[rank])}
                for rank in (calibration or {})}
    out = {}
    for rec in scored_records:
        scores = score(rec)
        ans = {}
        for rank in ("species", "genus"):
            rs = evaluate.rank_scores(scores, index, rank)
            # a name at -inf is absent for this record (a masked leave-out), never an answer;
            # with none at -inf this is top_labels over every name, as before
            ok = np.flatnonzero(np.isfinite(rs))
            names = index.labels[rank]
            ans[rank] = (evaluate.top_labels(rs[ok], [names[j] for j in ok], top_k)
                         if len(ok) else [])
        if calibration:
            ans["cal"] = {}
            for rank, c in calibration.items():
                rs = evaluate.rank_scores(scores, index, rank)
                if not len(rs):
                    continue
                p = likely.probabilities(rs, float(c["temperature"]))
                names = index.labels[rank]
                truth = ((truths or {}).get(rec.observation_id) or {}).get(rank)
                pos = position[rank].get(truth) if truth else None
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


MASKED_LEAVE_OUT = ("nearest", "nearest+mean", "nearest+mean-records",
                    "nearest+mean-records-balanced")


def vision_answers_leave_out(vectors, ref_records, scored_records, method, backbone, mode,
                             **kw) -> dict[str, dict]:
    """vision_answers with, for each group of finds sharing a leave_out_key, the reference
    records sharing it left out: what the identifier gives a find when nothing of its own
    observer (or trip) is in the reference. A find with no observer leaves nothing out.

    `nearest`, `nearest+mean` and the record-grouped arms fit once and hide each group's
    own reference columns (masked_leave_out_answers; equal to a refit without them,
    tested); any other method,
    including a place method (its prior is built from the reference records), is fitted
    again per group. Either way the calibration temperatures and list floors are the
    config's, fixed on val, never refitted per group."""
    if method in MASKED_LEAVE_OUT:
        return masked_leave_out_answers(vectors, ref_records, scored_records, method, backbone,
                                        mode, **kw)
    return refit_leave_out_answers(vectors, ref_records, scored_records, method, backbone,
                                   mode, **kw)


def masked_leave_out_answers(vectors, ref_records, scored_records, method, backbone, mode,
                             top_k: int = 10, calibration: dict | None = None,
                             truths: dict | None = None, place_setting=None, info=None,
                             twins: dict | None = None) -> dict:
    """The leave-out without a refit: the method is fitted once on every reference record;
    a find's scores are taken with the reference columns of its own group hidden (nearest:
    those columns out of each species' best match; nearest+mean: NearestAndMean.leave_out,
    which takes each touched species' two best matches and its average again without them;
    the record arms: NearestAndMeanRecords.leave_out, by record, so a hidden twin leaves its
    pair's photos in their specimen, as a refit without it would).
    A species whose every reference record is in the group gets -inf, so it is no answer,
    as it would be missing from a refitted index."""
    from . import methods
    index, model = fitted_method(vectors, ref_records, method, backbone, place_setting, info,
                                 twins)
    col_of_row = {int(r): i for i, r in enumerate(np.asarray(index.cols).tolist())}
    hidden_of: dict = defaultdict(list)
    for r in ref_records:
        if r.observer:
            hidden_of[leave_out_key(r, mode)].extend(
                col_of_row[int(p)] for p in r.photo_rows if int(p) in col_of_row)
    hidden_of = {k: np.unique(np.asarray(v, dtype=np.int64)) for k, v in hidden_of.items()}
    empty = np.zeros(0, dtype=np.int64)
    # The query-free part of the leave-out (the kept columns, the averages taken again) is
    # the same for every find of a group: built once per group (model.leave_out_plan), the
    # finds taken group by group so only the current group's plan is held.
    plan_of: dict = {}

    def plan(key, hidden):
        if key not in plan_of:
            plan_of.clear()
            plan_of[key] = model.leave_out_plan(hidden)
        return plan_of[key]

    def score(rec):
        q = np.asarray(vectors[rec.photo_rows], dtype=np.float32)
        key = leave_out_key(rec, mode) if rec.observer else None
        hidden = hidden_of.get(key, empty) if rec.observer else empty
        sims = model.photo_sims(q)
        if method == "nearest":
            if len(hidden):
                sims = np.array(sims, dtype=np.float32, copy=True)
                sims[:, hidden] = -np.inf
            return methods.species_scores(sims, index)
        per = model.per_photo_scores(sims, q)
        if len(hidden):
            per = model.leave_out_apply(per, sims, q, plan(key, hidden))
        return per.mean(axis=0)
    by_group = sorted(scored_records, key=lambda r: (not r.observer,
                                                     repr(leave_out_key(r, mode))
                                                     if r.observer else ""))
    return answers_from(index, by_group, score, top_k, calibration, truths)


def refit_leave_out_answers(vectors, ref_records, scored_records, method, backbone, mode,
                            **kw) -> dict[str, dict]:
    """The leave-out by fitting the method again per group, without the group's records."""
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


def embed_fresh(conn, spec: str, photo_ids: list[int], px: int | None = None, loader=None,
                log=print) -> tuple[np.ndarray, np.ndarray, int]:
    """The sealed test's photos embedded now by the frozen model `spec` (at `px`), from the
    copies on this machine, in memory: (photo ids, vectors, how many of them already had a
    stored vector, which is never read). PLAN §8: the site's nightly layer embeds newly
    validated records under the served backbone names, so a stored vector of a test photo
    may come from other weights. Nothing is written to the manifest; a photo with no local
    copy, or one that cannot be read, is refused."""
    import sqlite3
    import tempfile
    from . import models
    from .embed import embed_photos, load_embeddings
    from .storage import open_store
    loader = loader or models.load_backbone
    want = list(dict.fromkeys(int(p) for p in photo_ids))
    bb = loader(spec)
    if px:
        bb = Resized(bb, px)
    stored = 0                     # under the name embed_photos stores this model's rows
    if conn.execute("select 1 from sqlite_master where name = 'embeddings'").fetchone():
        have = {r[0] for r in conn.execute("select photo_id from embeddings where backbone = ?",
                                           (bb.name,))}
        stored = sum(1 for p in want if p in have)
    todo: dict[str, list] = defaultdict(list)
    missing = []
    for pid in want:
        row = conn.execute("select store, path from photo_copies where photo_id = ? and "
                           "size = 'large' and store not like 's3:%' order by store limit 1",
                           (pid,)).fetchone()
        if row is None:
            missing.append(pid)
        else:
            todo[row[0]].append((pid, row[1]))
    if missing:
        raise FrozenError(f"{len(missing):,} sealed-test photos have no large copy on this "
                          f"machine (e.g. {missing[:5]}): copy them first; the sealed test is "
                          "embedded fresh, never from stored vectors")
    with tempfile.TemporaryDirectory() as d:
        mem = sqlite3.connect(":memory:")
        skipped = []
        for where, items in todo.items():
            st = embed_photos(mem, open_store(where, cfgmod.DATA_DIR), bb, items,
                              Path(d) / "vectors", log=log)
            skipped += st.skipped
        if skipped:
            raise FrozenError(f"{len(skipped):,} sealed-test photos could not be embedded "
                              f"(e.g. {skipped[:3]})")
        ids, vecs = load_embeddings(mem, bb.name, root=Path(d) / "vectors")
        mem.close()
    return ids, vecs, stored


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


def _pinned(spec: dict, what: str) -> Path:
    path = resolve(spec["file"])
    got = sha256_file(path)
    if got != spec.get("sha256"):
        raise ValueError(f"{what} {path}: sha256 {got[:12]}, the config pins "
                         f"{str(spec.get('sha256'))[:12]}")
    return path


def answer_key_taxonomy(block: dict | None, truths=()) -> tuple[object, dict]:
    """The frozen iNat taxonomy for the answer key's canonical step (config
    "answer_key_taxonomy": the open-data snapshot, the successors file and the shared label
    vocabulary, each pinned by sha256, a different file refused; optional answer-side
    `extras` files, likewise pinned). The labels are the vocabulary file (never built here)
    plus any scored truth outside it. Extras may only add answer taxa: if one changes how
    any label resolves, the run is refused. Returns (taxonomy or None, its citation)."""
    if not block:
        return None, {}
    import json as _json

    from . import name_equiv
    snap = _pinned(block["snapshot"], "taxonomy snapshot")
    succ = _pinned(block["successors"], "taxonomy successors")
    vocab = name_equiv.read_label_vocabulary(_pinned(block["label_vocabulary"],
                                                     "label vocabulary"))
    if block["label_vocabulary"].get("labels") not in (None, len(vocab)):
        raise ValueError(f"label vocabulary: {len(vocab):,} labels, the config says "
                         f"{block['label_vocabulary']['labels']:,}")
    known = set(vocab)
    outside = sorted({t for t in truths if t and t not in known})
    labels = vocab + outside
    tx = name_equiv.Taxonomy.load(snap, labels, succ)
    cite = {**tx.cite, "label_vocabulary_sha256": block["label_vocabulary"]["sha256"],
            "labels": len(vocab), "truths_outside_vocabulary": len(outside)}
    extras: dict = {}
    specs = extras_files(block)
    for spec in specs:
        doc = _json.loads(_pinned(spec, "taxonomy extras").read_text(encoding="utf-8"))
        taxa = doc.get("taxa", doc)
        extras.update({int(k): v for k, v in taxa.items()})
    if extras:
        before = {lab: tx.species_id(lab) for lab in labels}
        tx2 = name_equiv.Taxonomy(tx.base, labels, tx.cite, extras=extras)
        moved = [lab for lab in labels if tx2.species_id(lab) != before[lab]]
        if moved:
            raise ValueError(f"taxonomy extras change how {len(moved):,} labels resolve "
                             f"(e.g. {moved[:3]}): extras may only add answer taxa")
        tx = tx2
        cite.update(extras=[{"file": str(x["file"]), "sha256": x["sha256"]} for x in specs],
                    extra_taxa=len(extras))
    return tx, cite


def extras_files(block: dict | None) -> list[dict]:
    """The answer-side taxonomy extras files of an answer_key_taxonomy block, each
    {file, sha256}: a list of them, or (as the analysis config documents them) a mapping
    whose entries with a file and a sha256 are the files and whose other entries (a rule,
    a note such as "fetched at scoring") are text."""
    raw = (block or {}).get("extras")
    if not raw:
        return []
    items = raw if isinstance(raw, list) else list(raw.values())
    return [x for x in items if isinstance(x, dict) and x.get("file") and x.get("sha256")]


def truth_resolution(taxonomy, species: str) -> str:
    """A find's truth_resolves attribute: "yes" when its species label resolves one-to-one
    to an iNat species at the snapshot (a formal name, or a provisional one by its exact
    name, rule 4), "no", or "(no species)"."""
    return "(no species)" if not species else "yes" if taxonomy.resolves(species) else "no"


def genus_known(taxonomy, genus: str, species: str | None) -> bool:
    """Steve, 18 Oct: can iNat name this truth's genus? The shared Taxonomy.genus_known (one
    rule for paper-score and E31), on the truth's label (its species, else its genus) and
    the release genus."""
    return bool(taxonomy.genus_known(species or genus, genus or None))


def answer_key(config: dict, comp: dict | None = None, taxonomy=None) -> po.AnswerKey:
    """The answer key a comparator is judged by: the config's (its own "exact" for a
    sensitivity row, which takes no canonical step), with the frozen taxonomy."""
    rule = (comp or {}).get("answer_key") or config["answer_key"]
    return po.AnswerKey(rule, taxonomy if rule != "exact" else None)


def outcomes_for(comp: dict, finds: list[po.Find], answers: dict | None = None,
                 conn=None, key: po.AnswerKey | None = None) -> tuple[dict, dict]:
    """(outcomes by record key, input info) for one comparator, every answer judged by
    `key` (the answer key; paper_outcomes.AnswerKey)."""
    by_key = {f.key: f for f in finds}
    key = key or po.AnswerKey()
    info: dict = {"kind": comp["kind"], "answer_key": key.rule}
    if comp["kind"] == "vision":
        collapse = bool(comp.get("collapse_infraspecific"))
        return {k: po.from_ranked(a["species"], a["genus"], by_key[k], collapse=collapse,
                                  key=key)
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
                                              bool(comp.get("collapse_infraspecific")), key)
        if uncredited is not None:
            # Joins the review did not credit (excluded, or never reviewed): exact names.
            info["crosswalk"]["uncredited_joins"] = {
                "pairs": len(uncredited), "answers": sum(uncredited.values()),
                "top": [{"their_answer": a, "our_label": t, "answers": n}
                        for (a, t), n in uncredited.most_common(30)]}
    elif comp["kind"] == "inat":
        po.check_format(lines, "e14")
        disagree, lane = 0, Counter()
        for line in lines:
            k = str(line["record_key"])
            if k in by_key:
                score = comp.get("score", "vision")
                out[k] = po.inat_outcome(line, by_key[k], score, key)
                for rank, d in po.inat_hits_disagree(line, out[k], score).items():
                    lane[rank] += d
                disagree += bool(line["truth"].get("formal_species")) != (
                    by_key[k].name_kind == "formal")
        info["formal_disagreements"] = disagree
        # The lane's taxon-id verdicts: a diagnostic beside the answer key, never scored.
        info["lane_verdicts_disagree_top1"] = dict(lane)
        if out and disagree > comp.get("max_formal_disagreement", 0.005) * len(out):
            raise ValueError(f"{comp['id']}: {disagree} finds disagree with the release on "
                             "whether the true name is formal")
    else:
        po.check_format(lines, "e31")
        lane = Counter()
        tier, view = str(comp.get("tier", "1")), comp.get("view", "best")
        for line in lines:
            k = str(line["record_key"])
            if k in by_key:
                o = po.expert_outcome(line, tier, view, bool(comp.get("strict")),
                                      find=by_key[k], key=key)
                for rank, d in po.expert_verdicts_disagree(line, o, tier, view).items():
                    lane[rank] += d
                if o is not None:
                    out[k] = o
        info["lane_verdicts_disagree"] = dict(lane)
        # A find the experts' file does not score is not a find they got wrong.
        for f in finds:
            if f.key not in out:
                out[f.key] = {c: None for c in po.CELLS} | {"answered": False}
    info["finds_with_lines"] = len(out)
    return out, info


def checkpoint_stamp(config_bytes: bytes, code_commit: str, comp: dict, finds: int) -> dict:
    """What a comparator's checkpoint must match to be reused: the config byte for byte, the
    code that scored it, the comparator's own entry, and the number of finds scored."""
    return {"config_sha256": sha256_bytes(config_bytes), "code_commit": code_commit,
            "comparator_sha256": sha256_bytes(json.dumps(comp, sort_keys=True).encode()),
            "finds": finds}


def write_checkpoint(path: Path, answers: dict, stamp: dict, info: dict) -> None:
    """The comparator's predictions (record_key + answer per line, sorted), written to a
    temporary file and moved into place, then its stamp beside it (<id>.meta.json): a run
    that dies mid-write leaves no checkpoint that looks complete."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for k in sorted(answers):
            fh.write(json.dumps({"record_key": k, **answers[k]}) + "\n")
    os.replace(tmp, path)
    meta = path.with_name(path.stem + ".meta.json")
    meta.write_text(json.dumps({**stamp, "answers": len(answers), "info": info}, indent=1),
                    encoding="utf-8")


def load_checkpoint(path: Path, stamp: dict, info: dict) -> dict | None:
    """The comparator's answers from its checkpoint when its stamp matches `stamp` in every
    part (and `info`, e.g. the place setting used, restored); None otherwise."""
    meta = path.with_name(path.stem + ".meta.json")
    if not (path.is_file() and meta.is_file()):
        return None
    m = json.loads(meta.read_text(encoding="utf-8"))
    if any(m.get(k) != v for k, v in stamp.items()):
        return None
    answers = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            answers[row.pop("record_key")] = row
    if len(answers) != m.get("answers"):
        return None
    info.update(m.get("info") or {})
    return answers


def run(conn, base, test, config: dict, config_bytes: bytes, out: Path,
        attrs_path=None, allow_code: str | None = None,
        research_dir: Path | None = None, embed_missing=None, gate_embed=None,
        log=print, resume: bool = False) -> dict:
    """Score every comparator of `config` on `test` ("val" or a test release of `base`)
    and write report.json, report.md and the Vision predictions under `out`."""
    from .embed import load_embeddings
    from .replications.fungitastic import dataset
    scored_rel, split, ref_splits, sealed = scored_and_reference(base, test)
    check_place_pins(config)          # before anything is read
    # The code that runs is the code checked out now: read once, here, so a commit made
    # while the run is going never relabels its report.
    code_commit = cfgmod.code_version()
    g = guard(sealed, config_bytes, config, research_dir or cfgmod.RESEARCH_DIR,
              code_commit, allow_code)
    if g.get("post_freeze"):
        post_freeze_files(config, g["post_freeze"], resolve)
    apply_errata(base, config.get("errata"))
    attrs_path = attrs_file(config.get("attrs"), attrs_path)
    # The whole predictor by hash (PLAN §8): the sealed test refuses any part that is not
    # the frozen one; val and draft runs say what differs and go on.
    from . import paper_freeze
    # (the taxonomy hashed is the one the answer key and E29 read)
    freeze_now = paper_freeze.compute(config, base, conn, resolve,
                                      taxonomy_path=(config.get("errors") or {}).get("taxonomy"))
    freeze_diff = paper_freeze.check(config.get("freeze"), freeze_now)
    if g["sealed"] and freeze_diff:
        raise FrozenError("the predictor is not the one the config freezes: "
                          + "; ".join(freeze_diff[:8])
                          + (f" (+{len(freeze_diff) - 8} more)" if len(freeze_diff) > 8 else ""))
    for d in freeze_diff:
        log(f"[freeze] {d}")
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
        "select manifest_photo_id, original_sha256, record_key, photo_key from photos "
        "where included = 1") if str(p["record_key"]) in ref_keys]
    ref_shas = {p["original_sha256"] for p in ref_photo_rows if p["original_sha256"]}
    nd_cfg = config.get("near_duplicates", {})
    # dHash near-copy flags: a test release's own near_base table (part of its
    # release_hash); on val, the base release's pinned dHash index through the same rule
    # (val_review.near_train_flags, dHash distance <= 4 to a reference photo).
    near_base = scored_rel.near_base() if scored_rel is not base else {}
    dhash = {pk for pk, v in near_base.items() if v.get("base_photo_key")}
    dhash_source = "the scored release's near_base table"
    if scored_rel is base and nd_cfg.get("dhash_index"):
        dhash = val_dhash_flags(base, nd_cfg["dhash_index"], photos, ref_photo_rows)
        near_base = {pk: {} for pk in dhash}
        dhash_source = f"the base release's dHash index ({nd_cfg['dhash_index']['file']})"
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
    gen = config.get("generalisation") or {}
    recent_after = gen.get("recent_after") or getattr(base, "val_cutoff", None)
    pre = gen.get("pretraining")
    add_generalisation_attrs(finds, scored_rows, recent_after,
                             read_created(pre) if pre else None, (pre or {}).get("after"))
    gen_rows = {**GENERALISATION_ROWS, **({f"created after {pre['after']}":
                                           {"after_pretraining": "yes"}} if pre else {}),
                **(gen.get("rows") or {})}
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(exist_ok=True)
    outcomes, inputs, calib, top1, per_photo = {}, {}, {}, {}, {}
    public = None
    # Every comparator's answers are judged by the config's answer key (one function,
    # paper_outcomes.AnswerKey) with the pinned frozen taxonomy; a comparator's own "exact"
    # is a sensitivity row.
    taxonomy, taxonomy_cite = answer_key_taxonomy(config.get("answer_key_taxonomy"),
                                                  {f.species for f in finds})
    keys = {c["id"]: answer_key(config, c, taxonomy) for c in config["comparators"]}
    # E32 arm (c), O8: a record-grouped method counts a cross-post pair as one specimen, by
    # the config's pinned groups file for this reference (train: "validation"; train + val:
    # "sealed_test"); refused when the block, the file or its sha256 is not as pinned.
    from . import twins as twins_mod
    groups, twins_cite = None, None
    if any(c["kind"] == "vision" and twins_mod.is_grouped(c["method"])
           for c in config["comparators"]):
        if not config.get("twins"):
            raise ValueError("a record-grouped method needs the config's \"twins\" block")
        groups, twins_cite = twin_groups(config["twins"], ref_splits)
        log(f"[twins] {twins_cite['file']} ({twins_cite['for']}): {twins_cite['records']:,} "
            f"records in {twins_cite['groups']:,} groups")
    # PLAN rule 7: a species truth that does not resolve one-to-one leaves the iNat- and
    # expert-paired tests (hypotheses with "truth_resolves": true), never their miss.
    if taxonomy is not None:
        for f in finds:
            f.attrs["truth_resolves"] = truth_resolution(taxonomy, f.species)
            f.attrs["truth_genus_known"] = ("(no genus)" if not f.genus else
                                            "yes" if genus_known(taxonomy, f.genus, f.species)
                                            else "no")
    # O1: a comparator with its own reference (the public model's openly licensed one) has
    # its finds' depth and "observer new" counted there; every other comparator shares
    # `finds`. The pairing of tests is by find key either way.
    refs_used = sorted({c["reference"] for c in config["comparators"] if c.get("reference")})
    ref_rows_of = {r: reference_rows(base, ref_splits, r) for r in refs_used}
    finds_by_ref = {r: finds_in_reference(finds, scored_rows, ref_rows_of[r]) for r in refs_used}
    finds_of = {c["id"]: finds_by_ref[c["reference"]] for c in config["comparators"]
                if c.get("reference")}
    fresh: dict = {}               # the sealed test: (backbone, px) -> its fresh vectors
    test_vectors: dict = {}
    for comp in config["comparators"]:
        answers = None
        if comp["kind"] == "vision":
            b = comp["backbone"]
            px = comp.get("photo_px")
            wanted = [int(p["manifest_photo_id"]) for ph in photos.values() for p in ph]
            if embed_missing is not None and not g["sealed"]:
                embed_missing(b, wanted, None)
            ids, vecs = load_embeddings(conn, b)
            row = {int(p): i for i, p in enumerate(ids.tolist())}
            ref = base.as_records(ref_splits, row, public_trainable_only=(
                comp.get("reference") == "public_trainable_only"))
            if comp.get("reference_as_of"):           # E7b: the reference as it stood then
                ref = [r for r in ref if r.validated_on and r.validated_on <= comp["reference_as_of"]]
            if g["sealed"]:
                # PLAN §8: the sealed test's photos are embedded now by the frozen model (its
                # files pinned by the freeze block), never read from stored vectors; their
                # vectors_sha is recorded (a test output). References stay as stored, pinned
                # by the freeze block's vectors part.
                if (b, px) not in fresh:
                    fresh[(b, px)] = embed_fresh(conn, b, wanted, px, log=log)
                    fids, fvecs, stored = fresh[(b, px)]
                    from .paper_freeze import vectors_sha
                    test_vectors[sized_name(b, px) if px else b] = {
                        **vectors_sha(fids, fvecs, wanted), "stored_vectors_not_read": stored}
                    if stored:
                        log(f"[fresh] {sized_name(b, px) if px else b}: {stored:,} stored "
                            "vectors of sealed-test photos not read (embedded fresh)")
                fids, fvecs, _ = fresh[(b, px)]
                offset = len(vecs)
                vecs = np.concatenate([vecs, np.asarray(fvecs, dtype=vecs.dtype)]) \
                    if len(fvecs) else vecs
                scored = scored_rel.as_records(split, {int(p): offset + i
                                                       for i, p in enumerate(fids.tolist())})
            elif px:
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
                                          int(config.get("top_k", 10)), twins=groups)
                per_photo[comp["id"]] = per_photo_table(finds, ans_each, reps, seed,
                                                        keys[comp["id"]])
                outcomes[comp["id"]], inputs[comp["id"]] = {}, {
                    "kind": "vision", "backbone": b, "method": comp["method"],
                    "photos": "each", "photo_answers": len(ans_each)}
                top1[comp["id"]] = {}
                continue
            place_info: dict = {}
            kw = {"top_k": int(config.get("top_k", 10)), "calibration": comp.get("calibration"),
                  "truths": truths, "place_setting": comp.get("place_setting"),
                  "info": place_info, "twins": groups}
            # A checkpoint per comparator (--resume): reused only when the config, the code
            # and this comparator's entry are the ones that scored it.
            path = out / "predictions" / f"{comp['id']}.jsonl"
            stamp = checkpoint_stamp(config_bytes, code_commit, comp, len(scored))
            answers = load_checkpoint(path, stamp, place_info) if resume else None
            if answers is not None:
                log(f"[{comp['id']}] resumed from its checkpoint ({len(answers):,} finds)")
            else:
                answers = (vision_answers_leave_out(vecs, ref, scored, comp["method"], b,
                                                    comp["leave_out"], **kw)
                           if comp.get("leave_out") else
                           vision_answers(vecs, ref, scored, comp["method"], b, **kw))
                write_checkpoint(path, answers, stamp, place_info)
            if comp.get("calibration"):
                calib[comp["id"]] = {k: a["cal"] for k, a in answers.items() if "cal" in a}
        outcomes[comp["id"]], inputs[comp["id"]] = outcomes_for(
            comp, finds, answers, conn, keys[comp["id"]])
        top1[comp["id"]] = top_answers(comp, answers)
        if comp["kind"] == "inat" and comp.get("calibration_from_score", True):
            calib[comp["id"]] = {str(line["record_key"]): po.inat_confidence(
                line, comp.get("score", "vision")) for line in po.read_jsonl(resolve(comp["file"]))}
        if answers is not None:
            inputs[comp["id"]].update(backbone=comp["backbone"], method=comp["method"],
                                      place=bool(comp.get("place")),
                                      reference_records=len(ref),
                                      reference=comp.get("reference", "every included photo"),
                                      **({"place_setting": place_info["place_setting"]}
                                         if place_info.get("place_setting") else {}),
                                      **({"twins": {**twins_cite,
                                                    "grouped_records": place_info.get(
                                                        "twin_records", 0)}}
                                         if twins_mod.is_grouped(comp["method"]) else {}),
                                      finds_without_vectors=len(finds) - len(answers),
                                      predictions_sha256=sha256_file(path))
    # PLAN rule 5 (19 Oct): per-species rows beside the per-find rate, species grouped by
    # the config's answer key; secondary, outside every hypothesis test.
    from .name_equiv import answer_key as _species_key
    per_species_kw = ({"per_species_rows": config["per_species_rows"],
                       "species_key": lambda name: _species_key(name, config["answer_key"])}
                      if config.get("per_species_rows") else {})
    # PLAN C4: species as answer-key units with their reference records (ReferenceDepth,
    # the same count as each find's depth; 0 with none) and whether the name is formal, over
    # two denominators: (b, leading) every species name in the base release's records,
    # included or not; (a) the species in the reference this run scores against.
    recover_units, recover_cite, in_scope_primary = None, None, None
    recover_units_of: dict = {}
    if config.get("recoverability"):
        recover_units, recover_cite = recoverability_units(base, ref_rows,
                                                           config["answer_key"])
        # a comparator with its own reference: its species seen, and the release's names'
        # depth, counted there
        by_ref = {r: recoverability_units(base, ref_rows_of[r], config["answer_key"])
                  for r in refs_used}
        recover_units_of = {cid: by_ref[c["reference"]][0] for c in config["comparators"]
                            for cid in [c["id"]] if c.get("reference")}
        recover_cite = {"full reference": recover_cite,
                        **{r: by_ref[r][1] for r in refs_used}} if refs_used else recover_cite
        in_scope_primary = pt.in_scope(finds, "primary", "all")
    gate_report = None
    if config.get("gate"):
        # E10: the gate on the scored finds and the OOD set; each Vision comparator gets a
        # "<id> served" twin where a refused find is wrong (the plain row is gate-off).
        from . import paper_gate
        gate_report, twins = paper_gate.run_gate(
            config["gate"], finds, photos, outcomes,
            [c["id"] for c in config["comparators"] if c["kind"] == "vision"],
            resolve, reps, seed, out=out / "gate", conn=conn, embed_list=gate_embed,
            sealed_open=bool(g["sealed"]))
        outcomes.update(twins)
        for cid in twins:
            inputs[cid] = {"kind": "vision served", "of": cid.rsplit(" served", 1)[0]}
    report = {
        "post_hoc": g["post_hoc"],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "guard": g, "code_commit": code_commit,
        "config_sha256": sha256_bytes(config_bytes),
        "answer_key": config["answer_key"],
        "answer_key_taxonomy": taxonomy_cite or None,
        # coverage (aggregate only): species truths that do not resolve one-to-one
        "truth_resolution": ({k: sum(1 for f in finds if f.attrs.get("truth_resolves") == k)
                              for k in ("yes", "no", "(no species)")}
                             if taxonomy is not None else None),
        "truth_genus_known": ({k: sum(1 for f in finds if f.attrs.get("truth_genus_known") == k)
                               for k in ("yes", "no", "(no genus)")}
                              if taxonomy is not None else None),
        "freeze": {"predictor": freeze_now, "differences": freeze_diff,
                   **({"test_vectors": test_vectors} if test_vectors else {})},
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
        # What the test build left out before scoring (its own reasons, e.g. a byte copy
        # of a base photo or a cross-post twin of a v1 record), beside the scoring's
        # near-copy exclusion (standard.near_duplicates): both counted in the report.
        "build_exclusions": build_exclusions(scored_rel) if scored_rel is not base else None,
        "dhash_flags": {"source": dhash_source,
                        "photos_hashed": len(near_base), "photos_flagged": len(dhash),
                        "near_base": (scored_rel.manifest.get("near_base")
                                      if scored_rel is not base else None)},
        "bootstrap": {"reps": reps, "seed": seed, "clusters": "observer"},
        "inputs": inputs,
        "finds": {"scored": len(finds), "by_name_kind": dict(Counter(f.name_kind for f in finds))},
        "standard": pt.all_standard(finds, outcomes, reps, seed, finds_of=finds_of,
                                    **per_species_kw),
        "references": {r: {"records": len(ref_rows_of[r]),
                           "comparators": [c for c, fs in finds_of.items()
                                           if fs is finds_by_ref[r]]} for r in refs_used},
        # The same tables on named subsets of the finds (config "subsets": {name: {attr:
        # value}}), e.g. val excluding the former held-out benchmark, the half that decides.
        "subsets": {name: {"filter": flt, **pt.all_standard(
            [f for f in finds if all(str(f.attrs.get(k)) == str(v) for k, v in flt.items())],
            outcomes, reps, seed, finds_of={
                cid: [f for f in fs if all(str(f.attrs.get(k)) == str(v)
                                           for k, v in flt.items())]
                for cid, fs in finds_of.items()}, **per_species_kw)}
            for name, flt in (config.get("subsets") or {}).items()},
        "hypotheses": pt.families(config.get("hypotheses") or {}, finds, outcomes, reps, seed,
                                  finds_of=finds_of, alpha=
                                  float(config.get("alpha", 0.05))),
        # E14 (Steve 27 Oct): the same hypotheses on secondary rows, descriptive, no Holm
        **({"secondary_rows": pt.secondary_rows(resolved_secondary_rows(config),
                                                config.get("hypotheses") or {}, finds,
                                                outcomes, reps, seed, finds_of=finds_of)}
           if config.get("secondary_rows") else {}),
        "per_photo": per_photo,
        # PLAN C4 (19 Oct): species recoverability, descriptive and model-based.
        "recoverability": ({cid: pt.recoverability(
            pt.in_scope(finds_of.get(cid, finds), "primary", "all"), o,
            recover_units_of.get(cid, recover_units), reps, seed)
            for cid, o in outcomes.items() if o} if config.get("recoverability") else None),
        "recoverability_units": recover_cite,
        "gate": gate_report,
        "calibration": calibration_report(finds, calib, outcomes,
                                          config.get("calibration_metrics", {}), reps, seed,
                                          config["comparators"]),
        "errors": errors_report(finds, top1, {**config.get("errors", {}), **(
            {"deadly": config["deadly_taxa"]} if config.get("deadly_taxa")
            and not (config.get("errors") or {}).get("deadly") else {})}, base, scored_rel,
            keys),
        "generalisation": {"recent_after": recent_after,
                           "pretraining": ({k: pre.get(k) for k in ("file", "sha256", "after")}
                                           if pre else None),
                           **pt.generalisation(
                               finds, outcomes, gen_rows,
                               gen.get("cells", ["species_strict@1", "genus_strict@1"]),
                               gen.get("comparators") or [c for c in outcomes],
                               reps, seed)},
        "slices": pt.slices(finds, outcomes, config.get("slices", {}).get("attrs", []),
                            config.get("slices", {}).get("cells", ["species_strict@1",
                                                                   "genus_strict@1"]),
                            config.get("slices", {}).get("comparators", list(outcomes)),
                            reps, seed, int(config.get("slices", {}).get("min_finds", 30)),
                            finds_of=finds_of),
    }
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    (out / "report.md").write_text(format_report(report, config), encoding="utf-8")
    log(f"paper-score: {len(finds):,} finds; report in {out}"
        + ("  *** POST HOC ***" if g["post_hoc"] else ""))
    return report


def per_photo_table(finds, answers_each: dict, reps: int, seed: int,
                    key: po.AnswerKey | None = None) -> dict:
    """E19, each photo alone: top-1 and top-5 per photo (species strict, genus strict) with
    the observer-clustered interval, primary scope; a photo counts toward its find's
    observer, so a find with many photos weighs more, as a photo-level number does."""
    by_key = {f.key: f for f in pt.in_scope(finds)}
    outs = {}
    for pk, a in answers_each.items():
        f = by_key.get(pk.split("#", 1)[0])
        if f is not None:
            outs[pk] = (f, po.from_ranked(a["species"], a["genus"], f, key=key))
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


def errors_report(finds, top1: dict, cfg: dict, base, scored_rel, keys: dict | None = None
                  ) -> dict:
    """E29 and E21 for the comparators named (default: every one with answer names); a
    wrong answer is one the comparator's answer key does not credit."""
    from . import paper_errors as pe
    base_finds = pt.in_scope(finds)
    ranks, tax_sha = taxonomy_ranks(base, scored_rel, cfg.get("taxonomy"))
    deadly = pe.read_deadly({**cfg["deadly"], **({"file": str(resolve(cfg["deadly"]["file"]))}
                                                if "file" in cfg["deadly"] else {})}
                            ) if cfg.get("deadly") else None
    out = {"taxonomy_sha256": tax_sha, "comparators": {}}
    for cid in cfg.get("comparators") or [c for c, t in top1.items() if t]:
        t = top1.get(cid) or {}
        right = (keys or {}).get(cid, po.AnswerKey()).species
        row = {"confused_pairs": pe.confused_pairs(base_finds, t, int(cfg.get("pairs", 20)),
                                                   right),
               "distance": pe.mistake_distance(base_finds, t, ranks, right)}
        if deadly:
            row["deadly"] = pe.deadly_taxa(base_finds, t, deadly, right)
        out["comparators"][cid] = row
    return out


def calibration_report(finds, calib: dict, outcomes: dict, cfg: dict, reps: int,
                       seed: int, comparators: list | None = None) -> dict:
    """E13 and E11 for every comparator with confidences (paper_calibration), primary scope.
    A comparator's own calibration block may give the likely-set coverage its floor was fit
    for ({rank: {temperature, floor, target}}): that target is printed beside the coverage
    achieved, in place of the config-wide calibration_metrics.list_target."""
    from . import paper_calibration as pc
    base = pt.in_scope(finds)
    own = {c["id"]: {rank: v["target"] for rank, v in (c.get("calibration") or {}).items()
                     if isinstance(v, dict) and v.get("target") is not None}
           for c in comparators or []}

    def cfg_for(cid):
        if not own.get(cid):
            return cfg
        merged = dict(cfg.get("list_target") or {}) if isinstance(cfg.get("list_target"),
                                                                  dict) else {}
        return {**cfg, "list_target": {**merged, **own[cid]}}
    return {cid: {rank: pc.summarise(base, cal, outcomes[cid], rank, cfg_for(cid), reps, seed,
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
              ""]
    bx = report.get("build_exclusions")
    if bx:
        for what in ("records", "photos"):
            b = bx[what]
            lines.append(f"Left out by the test build ({what}): {b['left_out']:,} of "
                         f"{b['total'] or 0:,}" + (": " + ", ".join(
                             f"{k} {v:,}" for k, v in b["by_source_reason"].items())
                             if b["by_source_reason"] else "") + ".")
        lines.append("")
    nd = report["standard"]["near_duplicates"]
    lines += [f"Near-copies left out of the primary numbers at scoring: "
              f"{nd['near_duplicate_finds']:,} of {nd['finds']:,} finds ("
              + (", ".join(f"{k} {v:,}" for k, v in nd["by_kind"].items()) or "none")
              + "); every table also has a with-near-copies scope.", ""]
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
                ps_ = t.get("per_species")
                if ps_ and ps_["n_species"]:
                    ci = ps_["macro_ci95"]
                    lines += ["", f"Per species ({ps_['cell']}; secondary, no test): macro "
                              f"{100 * ps_['macro_species_top1']:.1f}%"
                              + (f" ({100 * ci[0]:.1f}-{100 * ci[1]:.1f})" if ci else "")
                              + f"; {ps_['species_ge1_right']:,} of {ps_['n_species']:,} "
                              f"species with a find right "
                              f"({100 * ps_['share_species_ge1_right']:.1f}%); "
                              f"{ps_['finds']:,} finds."]
                lines.append("")
    if report.get("gate"):
        lines += _gate_lines(report["gate"])
    if report.get("calibration"):
        lines += _calibration_lines(report["calibration"], config)
    if report.get("generalisation"):
        lines += _generalisation_lines(report["generalisation"])
    if report.get("recoverability"):
        lines += _recoverability_lines(report["recoverability"])
    if report.get("freeze"):
        diff = report["freeze"]["differences"]
        lines += ["## Freeze", "", "The predictor matches the config's freeze block."
                  if not diff else f"{len(diff)} part(s) differ from the config's freeze "
                  "block:", ""] + [f"- {d}" for d in diff[:40]] + [""]
    tv = (report.get("freeze") or {}).get("test_vectors")
    if tv:
        lines += ["The scored photos were embedded fresh by the frozen model (stored vectors "
                  "of them are never read):", "", "| backbone | photos | sha256 | stored, not "
                  "read |", "|---|---|---|---|"]
        lines += [f"| {b} | {v['photos']:,} | {v['sha256'][:12]} | "
                  f"{v['stored_vectors_not_read']:,} |" for b, v in tv.items()] + [""]
    pf = (report.get("guard") or {}).get("post_freeze")
    if pf:
        lines += [f"Filled after the tag (post_freeze): {sum(r['filled'] for r in pf)} of "
                  f"{len(pf)}; the config tagged had sha256 "
                  f"{report['guard']['tagged_config_sha256'][:12]}.", "",
                  "| field | kind | filled | sha256 |", "|---|---|---|---|"]
        lines += [f"| {r['path']} | {r['kind']} | {'yes' if r['filled'] else 'no'} | "
                  f"{(r.get('file_sha256') or r.get('value_sha256') or '-')[:12]} |"
                  for r in pf] + [""]
    if report.get("secondary_rows"):
        lines += ["## Secondary rows (descriptive, outside Holm)", "",
                  "| row | finds | share | unknown | hypothesis | n | difference | p |",
                  "|---|---|---|---|---|---|---|---|"]
        for rid, r in report["secondary_rows"].items():
            if "not_run" in r:
                lines.append(f"| {rid} | - | - | - | {r['not_run']} | | | |")
                continue
            for hid, h in r["hypotheses"].items():
                d = h.get("difference")
                lines.append(f"| {rid} | {r['finds']:,} | "
                             f"{100 * (r['share_of_scored'] or 0):.1f}% | {r['unknown']:,} | "
                             f"{hid} | {h.get('n', '-')} | "
                             + ("-" if d is None else f"{100 * d:+.1f} pts")
                             + f" | {h.get('p', h.get('not_tested', '-'))} |")
        lines.append("")
    lines += ["## Hypotheses", ""]
    for fam, hyps in report["hypotheses"].items():
        lines += [f"### Family {fam} (" + ("outside Holm: disclosed, no decision"
                                           if fam in pt.OUTSIDE_HOLM else
                                           "Holm within the family") + ")", "",
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
            holm = "-" if h.get("p_holm") is None else f"{h['p_holm']:.4g}"
            decision = "-" if h.get("reject") is None else ("yes" if h["reject"] else "no")
            lines.append(f"| {hid} | {n:,} | {eff} | {h['p_test']:.4g} | {holm} | {decision} |")
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
    if ood and "not_scored" in ood:
        lines += [f"OOD set: not scored ({ood['not_scored']}).", ""]
    elif ood:
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


def _calibration_lines(cal: dict, config: dict) -> list[str]:
    """E13 and E11 per comparator; a place method's rows stand on their own (it shares the
    photos-only calibration, so its reliability is reported separately, never pooled)."""
    place = {c["id"] for c in config.get("comparators", []) if c.get("place")}
    pct = lambda v: "-" if v is None else f"{100 * v:.1f}%"  # noqa: E731
    lines = ["## Calibration and confidence (E13, E11)", "",
             "| comparator | rank | n | ECE | Brier | NLL | rule | precision (lower 95) "
             "| coverage | likely sets: achieved (95% CI) / target, mean size |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for cid, ranks in cal.items():
        for rank, r in ranks.items():
            sc = r.get("scores") or {}
            ls = r.get("likely_sets")
            lst = "-" if not ls else (
                f"{pct(ls['coverage'])}"
                + (f" ({pct(ls['ci95'][0])}-{pct(ls['ci95'][1])})" if ls.get("ci95") else "")
                + f" / {pct(ls.get('target'))}, {ls['mean_size']:.2f}")
            num = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
            stats = (f"{num(r['reliability']['ece'])} | {num(sc.get('brier'))} | "
                     f"{num(sc.get('nll'))}")
            for rule in r.get("rules") or [{}]:
                lines.append(
                    f"| {cid}{' (place)' if cid in place else ''} | {rank} | {r['n']:,} | "
                    f"{stats} | {rule.get('threshold', '-')} | {pct(rule.get('precision'))} "
                    f"({pct(rule.get('lower_bound_one_sided_95'))}) | "
                    f"{pct(rule.get('coverage'))} | {lst} |")
    lines.append("")
    unk = [(cid, ranks["species"]["unknown_species"]) for cid, ranks in cal.items()
           if "unknown_species" in ranks.get("species", {})]
    if unk:
        lines += ["| unknown species (no v1 reference) | known / unknown finds | AUROC "
                  "| confident and wrong: all / unknown | genus right, unknown |",
                  "|---|---|---|---|---|"]
        for cid, u in unk:
            cw = u.get("confident_wrong") or {}
            au = u.get("auroc_known_vs_unknown")
            gb = u.get("genus_top1_without_species_reference") or {}
            lines.append(f"| {cid} | {u['known']:,} / {u['unknown']:,} | "
                         f"{'-' if au is None else f'{au:.3f}'} | "
                         f"{pct((cw.get('all') or {}).get('wrong_share'))} / "
                         f"{pct((cw.get('unknown') or {}).get('wrong_share'))} | "
                         f"{pct(gb.get('rate'))} |")
        lines.append("")
    return lines


def _recoverability_lines(rec: dict) -> list[str]:
    """PLAN C4: species recoverability per comparator (descriptive, model-based)."""
    lines = ["## Species recoverability (C4; descriptive, model-based)", "",
             "Expected share of the reference's species one find would name right: each "
             "species' reference band x that band's per-find accuracy (species strict, by the "
             "answer key), averaged over species. Assumes accuracy within a band does not "
             "depend on the species. Intervals: bootstrap over finds.", ""]
    for cid, r in rec.items():
        lines += [f"### {cid}", "", "| of the | | species | top 1 | top 3 | top 10 |",
                  "|---|---|---|---|---|---|"]
        for dname, den in r["denominators"].items():
            for name, row in den["rows"].items():
                cells = []
                for k in (1, 3, 10):
                    c = row[f"top{k}"]
                    cells.append("-" if c["estimate"] is None else
                                 f"{100 * c['estimate']:.1f}%" + (
                                     f" ({100 * c['ci95'][0]:.1f}-{100 * c['ci95'][1]:.1f})"
                                     if c["ci95"] else ""))
                lines.append(f"| {dname} | {name} | {row['species']:,} | " + " | ".join(cells)
                             + " |")
        lines.append("")
    return lines


def _generalisation_lines(gen: dict) -> list[str]:
    """E20 named rows (unseen observers, recent cohort, created after the pretraining
    snapshot) and E12's near-duplicate sensitivity: intervals only."""
    lines = ["## Generalisation (E20, E12, P15)", "",
             f"Recent cohort = observed after {gen.get('recent_after') or '-'}.", "",
             "| row | comparator | cell | in the row | the other finds |", "|---|---|---|---|---|"]
    for name, row in gen["rows"].items():
        for cid, cells in row.items():
            if not isinstance(cells, dict) or cid == "filter":
                continue
            for cell, r in cells.items():
                lines.append(f"| {name} ({row['finds']:,} / {row['others']:,}) | {cid} | {cell} "
                             f"| {_pct(r['in'])} | {_pct(r['out'])} |")
    nd = gen["near_duplicates"]
    lines += ["", f"Near-duplicates: {nd['near_duplicate_finds']:,} of {nd['finds']:,} finds "
              f"({', '.join(f'{k} {v:,}' for k, v in nd['by_kind'].items()) or 'none'}).", "",
              "| comparator | cell | primary (near-duplicates out) | near-duplicates kept |",
              "|---|---|---|---|"]
    for cid, cells in nd.items():
        if not isinstance(cells, dict) or cid == "by_kind":
            continue
        for cell, r in cells.items():
            lines.append(f"| {cid} | {cell} | {_pct(r['primary'])} | "
                         f"{_pct(r['with_near_duplicates'])} |")
    return lines + [""]
