"""Expert identifiers on iNaturalist against the DNA name (the paper's E31).

For each DNA-validated iNat observation: the identifications people made on it BEFORE the
DNA result was visible, by identifiers with many iNat identifications, scored against the
validated name.

- "Before visible" is before the earliest of: a sequence for the record created on
  mycomap.org or mycomap.com, an observation field carrying DNA results added on iNat (its
  timestamp), a name pushed by MycoMap (the provisional / override name fields, or a push
  logged on our side), and any comment or identification remark mentioning a sequence.
  Choosing the earliest keeps the cutoff on the safe side.
- Each identifier's ID is taken as it stood at the cutoff, even if later withdrawn or
  replaced: their latest identification made before it, unless that one was provably
  withdrawn before it.
- Tier 1: identifiers with >= 10,000 iNat identifications, tier 2: 1,000-9,999 (counts read
  at the snapshot). The observer, the record's MycoMap validators and MycoMap's push
  accounts are never counted.
- Species is scored on formally named truths, genus on all; an identification above
  species counts at genus only (stopping early is coverage, not error). Matching is by iNat
  taxon id, following a retired name to its one current successor.

Data comes from iNat's public API with no token, at most one request a second for both
hosts together, backing off on 429 and 5xx: iNat's v2 observation search (200 a request:
community taxon, and each identifier's identification count) and the older
`www.inaturalist.org/observations/<id>.json`, the only public answer that carries
observation-field timestamps and identification update times. Raw answers are cached, so a
rerun resumes free. The cache holds logins, so it stays in the private data folder; the
snapshot built from it keeps numeric ids only and is hashed; the scoring files carry a keyed
hash of each identifier, never an id or login.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import hmac
import json
import random
import re
import secrets
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import requests

from . import config
from .ratelimit import MinInterval

V2 = "https://api.inaturalist.org/v2"
V1 = "https://api.inaturalist.org/v1"
RAILS = "https://www.inaturalist.org"
BATCH = 200
V2_FIELDS = ("(id:!t,community_taxon_id:!t,user:(id:!t),"
             "identifications:(id:!t,current:!t,user:(id:!t,identifications_count:!t)))")
TIER1, TIER2 = 10_000, 1_000
# The research fetch names itself and a contact (MycoMap's public address, not a person's).
USER_AGENT = config.USER_AGENT.rstrip(")") + "; research; contact info@mycomap.org)"
MAX_THROTTLES = 5          # 429s in one run before it stops and asks a person
SNAPSHOT_FORMAT = 1
SCORES_FORMAT = 3   # 2: taxon ids + names on every ID, community, first_pooled, truth
                    # 3: verdicts by the shared answer key; by-id and plain-string beside

# Observation fields that carry a DNA result (matched by name, so a field iNat users add
# later is caught too) and the name fields that follow a DNA result. Both end the window.
# Not "consensus" alone: "Mushroom Observer Consensus Name" is MO's community name.
DNA_FIELD = re.compile(r"\b(DNA|sequenc\w*|barcod\w*|BLAST|GenBank|Reads in Consensus|RiC|"
                       r"Trace File\w*|nanopore|sanger|ITS|LSU)\b", re.I)
# Name fields pushed with or after a DNA result (a temporary code made for it, a flag that
# the iNat ID disagrees with the DNA): counted as a name push, keeping the cutoff early.
NAME_PUSH_FIELDS = {10675: "Provisional Species Name", 20259: "Species Name Override",
                    17527: "New Temporary Name Created", 21348: "ID Update Needed"}
# Remarks: "sequence", "sequenced", "DNA", "barcode", "BLAST", "GenBank", "ITS" (capitals
# only, so "its" is not a hit), "RiC", "nanopore", "Sanger".
MENTION = re.compile(r"(?i:\bsequenc\w*|\bDNA\b|\bbarcod\w*|\bBLAST\w*|\bgen ?bank\b|"
                     r"\bnanopore\b|\bsanger\b|\bmolecular\w*)|\bITS\d?\b|\bRiC\b")
SPECIES_LEVEL = 10      # iNat rank_level: species 10, subspecies/variety/form below
GENUS_LEVEL = 20
RANK_LEVEL = {"form": 5, "variety": 5, "subspecies": 5, "hybrid": 10, "species": 10,
              "complex": 11, "subsection": 12, "section": 13, "subgenus": 15, "genus": 20,
              "subtribe": 24, "tribe": 25, "supertribe": 26, "subfamily": 27, "family": 30}


# --- time --------------------------------------------------------------------------

def utc(ts: str | None) -> str | None:
    """Any iNat or SQL timestamp -> 'YYYY-MM-DDTHH:MM:SS.ffffffZ' (UTC), or None."""
    if not ts:
        return None
    s = str(ts).strip().replace(" ", "T")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    d = datetime.fromisoformat(s)
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)   # SQL routes answer in UTC
    return d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# --- iNat access (cached, paced, resumable) ------------------------------------------

class HistoryClient:
    """Public iNat reads for the snapshot. One pace for every host (<= 1 request a second);
    429 and 5xx back off (honouring Retry-After) and retry; answers are cached by key."""

    def __init__(self, cache_dir: Path, session: requests.Session | None = None,
                 interval: float = 1.05, sleep: Callable[[float], None] = time.sleep,
                 max_calls: int | None = None):
        self.cache_dir = cache_dir
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.session.headers["Accept"] = "application/json"
        self.throttles = 0
        self.pace = MinInterval(interval, sleep=sleep)
        self.calls = 0
        self.max_calls = max_calls

    def path(self, kind: str, key: str) -> Path:
        return self.cache_dir / kind / f"{key}.json.gz"

    def cached(self, kind: str, key: str) -> dict | None:
        p = self.path(kind, key)
        if not p.exists():
            return None
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)

    def _store(self, kind: str, key: str, body: dict) -> None:
        p = self.path(kind, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(body, f)
        tmp.replace(p)                      # never a half-written cache file

    def _get(self, url: str, params: dict | None = None) -> tuple[int, dict | None]:
        for attempt in range(6):
            if self.max_calls is not None and self.calls >= self.max_calls:
                raise CallBudgetSpent(f"stopped at the call budget ({self.max_calls})")
            self.pace.wait()
            self.calls += 1
            try:
                r = self.session.get(url, params=params, timeout=60)
            except requests.RequestException:
                self.pace.pause(10 * (attempt + 1))
                continue
            if r.status_code == 200:
                return 200, r.json()
            if r.status_code in (404, 410):
                return r.status_code, None  # deleted or never public: recorded as such
            if r.status_code == 403:
                raise Refused("iNat answered 403 (refused): stopped; tell the paper coordinator")
            if r.status_code == 429:
                self.throttles += 1
                if self.throttles >= MAX_THROTTLES:
                    raise Refused(f"iNat throttled this run {self.throttles} times (429): "
                                  "stopped; tell the paper coordinator")
            if r.status_code == 429 or r.status_code >= 500:
                wait = 60 * (attempt + 1)
                try:
                    wait = max(wait, int(r.headers.get("Retry-After", 0)))
                except ValueError:
                    pass
                self.pace.pause(wait)
                continue
            raise RuntimeError(f"iNat answered {r.status_code} for {url}: {r.text[:200]}")
        raise RuntimeError("iNat kept failing; stopped. Rerun to resume (answers are cached).")

    def observation(self, oid: str) -> dict:
        """The older observation JSON: identifications with update times, observation
        fields with timestamps, comments. A missing observation caches as {'missing': code}."""
        got = self.cached("rails", oid)
        if got is None:
            code, body = self._get(f"{RAILS}/observations/{oid}.json")
            got = body if body is not None else {"missing": code}
            got["_fetched_at"] = utc(datetime.now(timezone.utc).isoformat())
            self._store("rails", oid, got)
        return got

    def v2_batch(self, ids: list[str]) -> dict[str, dict]:
        """Community taxon and identifier counts for up to 200 observations; cached per id."""
        out, want = {}, []
        for oid in ids:
            got = self.cached("v2", oid)
            if got is None:
                want.append(oid)
            else:
                out[oid] = got
        for i in range(0, len(want), BATCH):
            chunk = want[i:i + BATCH]
            _, body = self._get(f"{V2}/observations", {"id": ",".join(chunk), "per_page": BATCH,
                                                       "fields": V2_FIELDS})
            found = {str(r["id"]): r for r in (body or {}).get("results") or []}
            now = utc(datetime.now(timezone.utc).isoformat())
            for oid in chunk:
                r = found.get(oid) or {"missing": 404}
                r["_fetched_at"] = now
                self._store("v2", oid, r)
                out[oid] = r
        return out

    def user_counts(self, uid: int) -> int | None:
        got = self.cached("users", str(uid))
        if got is None:
            code, body = self._get(f"{V1}/users/{uid}")
            got = (body or {}).get("results", [{}])[0] if body else {"missing": code}
            got = {"identifications_count": got.get("identifications_count"),
                   "missing": got.get("missing"),
                   "_fetched_at": utc(datetime.now(timezone.utc).isoformat())}
            self._store("users", str(uid), got)
        return got.get("identifications_count")

    def taxa(self, ids: Iterable[int]) -> dict[int, dict]:
        """iNat taxa by id (ancestry, active flag, current successor), 30 a request."""
        ids = sorted({int(t) for t in ids})
        out, want = {}, []
        for t in ids:
            got = self.cached("taxa", str(t))
            (want.append(t) if got is None else out.__setitem__(t, got))
        for i in range(0, len(want), 30):
            chunk = want[i:i + 30]
            _, body = self._get(f"{V1}/taxa/{','.join(map(str, chunk))}")
            found = {int(r["id"]): r for r in (body or {}).get("results") or []}
            for t in chunk:
                r = found.get(t)
                slim = {"id": t, "missing": r is None}
                if r:
                    slim.update(name=r.get("name"), rank=r.get("rank"),
                                rank_level=r.get("rank_level"), is_active=r.get("is_active"),
                                ancestor_ids=r.get("ancestor_ids") or [],
                                current_synonymous_taxon_ids=r.get(
                                    "current_synonymous_taxon_ids") or [])
                self._store("taxa", str(t), slim)
                out[t] = slim
        return out


class CallBudgetSpent(RuntimeError):
    pass


class Refused(RuntimeError):
    """iNat refused (403) or kept throttling (429): stop the run and ask a person."""


# --- the snapshot (numeric ids only) --------------------------------------------------

def _path_of(taxon: dict) -> list[int]:
    anc = [int(x) for x in str(taxon.get("ancestry") or "").split("/") if x.strip().isdigit()]
    return anc + [int(taxon["id"])]


def slim_observation(oid: str, rails: dict, v2: dict) -> dict:
    """One observation's snapshot row: no logins, names, coordinates or field values."""
    if rails.get("missing"):
        return {"observation_id": oid, "missing": rails["missing"],
                "fetched_at": rails.get("_fetched_at")}
    counts = {}
    for ident in (v2.get("identifications") or []):
        u = ident.get("user") or {}
        if u.get("id") is not None:
            counts[int(u["id"])] = u.get("identifications_count")
    idents = []
    for i in rails.get("identifications") or []:
        t = i.get("taxon") or {}
        if not t.get("id"):
            continue
        idents.append({
            "id": i["id"], "user_id": int(i["user_id"]),
            "taxon_id": int(t["id"]), "taxon_rank": t.get("rank"),
            "rank_level": t.get("rank_level") or RANK_LEVEL.get(t.get("rank")),
            "taxon_path": _path_of(t), "taxon_active": t.get("is_active"),
            "created_at": utc(i.get("created_at")), "updated_at": utc(i.get("updated_at")),
            "current": bool(i.get("current")), "category": i.get("category"),
            "disagreement": bool(i.get("disagreement")),
            "taxon_change": i.get("taxon_change_id") is not None,
            "remark_mentions_dna": bool(MENTION.search(i.get("body") or "")),
        })
    ofvs = [{"field_id": v.get("observation_field_id"),
             "name": (v.get("observation_field") or {}).get("name"),
             "created_at": utc(v.get("created_at")), "updated_at": utc(v.get("updated_at")),
             "user_id": v.get("user_id")}
            for v in rails.get("observation_field_values") or []]
    comments = [{"user_id": c.get("user_id"), "created_at": utc(c.get("created_at")),
                 "mentions_dna": bool(MENTION.search(c.get("body") or ""))}
                for c in rails.get("comments") or []]
    return {"observation_id": oid, "observer_id": rails.get("user_id"),
            "created_at": utc(rails.get("created_at")),
            "community_taxon_id_now": v2.get("community_taxon_id"),
            "quality_grade_now": rails.get("quality_grade"),
            "identifications": sorted(idents, key=lambda r: (r["created_at"] or "", r["id"])),
            "ofvs": ofvs, "comments": comments, "identifier_counts": counts,
            "fetched_at": rails.get("_fetched_at")}


def fetch(client: HistoryClient, ids: list[str], log=print) -> dict:
    """Fetch (or reuse from the cache) every observation's history and counts. Resumable:
    an interrupted run picks up where it stopped. Returns the counts, the requests made and
    the wall hours, for the paper's cost ledger."""
    stats = Counter()
    start, calls0 = time.monotonic(), client.calls
    for i in range(0, len(ids), BATCH):
        chunk = ids[i:i + BATCH]
        client.v2_batch(chunk)
        for oid in chunk:
            had = client.cached("rails", oid) is not None
            client.observation(oid)
            stats["cached" if had else "fetched"] += 1
        log(f"  {min(i + BATCH, len(ids)):,}/{len(ids):,} observations, {client.calls:,} calls")
    return {**stats, "requests": client.calls - calls0,
            "wall_hours": round((time.monotonic() - start) / 3600, 3)}


CREATED_FILE = "created_at.tsv"


def build_snapshot(client: HistoryClient, ids: list[str], out: Path, meta: dict,
                   record_keys: dict[str, str] | None = None) -> dict:
    """Write the slim snapshot (gzip JSONL, one observation a line, then the identifier
    counts) from the cache, fetching counts for identifiers v2 did not carry. Returns
    the manifest with sha256s; nothing is fetched for an observation not yet cached.
    With `record_keys` (iNat id -> release record_key) it also writes created_at.tsv:
    record_key, observation_id, created_at (iNat's observation creation, UTC; empty when
    the observation is gone), the join paper-score uses for the training-cutoff rows."""
    rows, missing_cache = [], []
    for oid in ids:
        rails, v2 = client.cached("rails", oid), client.cached("v2", oid)
        if rails is None or v2 is None:
            missing_cache.append(oid)
            continue
        rows.append(slim_observation(oid, rails, v2))
    if missing_cache:
        raise RuntimeError(f"{len(missing_cache):,} observations are not fetched yet "
                           f"(first: {missing_cache[:3]}); run the fetch first")
    for r in rows:   # identifiers on the observation that v2 did not carry a count for
        for ident in r.get("identifications") or []:
            uid = ident["user_id"]
            if r["identifier_counts"].get(uid) is None:
                r["identifier_counts"][uid] = client.user_counts(uid)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    manifest = {"format": SNAPSHOT_FORMAT, **meta, "observations": len(rows),
                "missing_on_inat": sum(1 for r in rows if r.get("missing")),
                "ids_sha256": ids_sha256(ids), "snapshot_file": out.name,
                "snapshot_sha256": sha256_file(out),
                "built_at": utc(datetime.now(timezone.utc).isoformat())}
    if record_keys is not None:
        created = out.with_name(CREATED_FILE)
        with open(created, "w", encoding="utf-8", newline="\n") as f:
            f.write("record_key\tobservation_id\tcreated_at\n")
            for r in sorted(rows, key=lambda r: record_keys[r["observation_id"]]):
                f.write(f"{record_keys[r['observation_id']]}\t{r['observation_id']}\t"
                        f"{r.get('created_at') or ''}\n")
        manifest.update(created_file=created.name, created_sha256=sha256_file(created))
    out.with_name(out.name.replace(".jsonl.gz", "") + ".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def read_snapshot(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for r in rows:   # JSON object keys are strings
        r["identifier_counts"] = {int(k): v for k, v in (r.get("identifier_counts") or {}).items()}
    return rows


def ids_sha256(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(set(map(str, ids)))).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --- the sample --------------------------------------------------------------------

def draw_sample(ids: Iterable[str], n: int, seed: int) -> list[str]:
    """A seeded random n of `ids` (sorted first, so the draw depends only on the set and
    the seed), returned sorted."""
    pool = sorted(set(map(str, ids)), key=lambda s: (len(s), s))
    if n >= len(pool):
        return pool
    return sorted(random.Random(seed).sample(pool, n), key=lambda s: (len(s), s))


# --- when the DNA became visible ------------------------------------------------------

@dataclass
class Cutoff:
    at: str | None          # UTC; None when no source is known
    source: str             # sequence_created | dna_field | name_push | comment | validated_on | none
    sources: dict = field(default_factory=dict)   # every source's earliest time


def dna_cutoff(obs: dict, ours: Iterable[tuple[str, str]] = (),
               validated_on: str | None = None) -> Cutoff:
    """The earliest moment the DNA could be seen. `ours`: (source, time) pairs from our
    databases (sequence_created, name_push). iNat side: a DNA-result field or a pushed name
    field added, a comment or identification remark mentioning a sequence. With none known,
    the start of the first validation day (validated_on) stands in, flagged as such."""
    best: dict[str, str] = {}

    def offer(source: str, ts: str | None) -> None:
        ts = utc(ts)
        if ts and (source not in best or ts < best[source]):
            best[source] = ts

    for source, ts in ours:
        offer(source, ts)
    for v in obs.get("ofvs") or []:
        if v.get("field_id") in NAME_PUSH_FIELDS:
            offer("name_push", v.get("created_at"))
        elif DNA_FIELD.search(v.get("name") or ""):
            offer("dna_field", v.get("created_at"))
    for c in obs.get("comments") or []:
        if c.get("mentions_dna"):
            offer("comment", c.get("created_at"))
    for i in obs.get("identifications") or []:
        if i.get("remark_mentions_dna"):
            offer("comment", i.get("created_at"))
    if best:
        source = min(best, key=lambda s: (best[s], s))
        return Cutoff(best[source], source, best)
    if validated_on:
        return Cutoff(utc(f"{validated_on}T00:00:00"), "validated_on", {})
    return Cutoff(None, "none", {})


# --- identifications as they stood ----------------------------------------------------

def standing_ids(obs: dict, cutoff: str) -> dict[int, dict]:
    """Each user's identification as it stood at `cutoff`: their latest one made before
    it, unless it was withdrawn before it. A replaced ID (a later one by the same user)
    was current until the replacement, so it stands if the replacement came after the
    cutoff. A withdrawn ID (no later one) shows its withdrawal only through its update
    time, which is at or after the withdrawal: an update before the cutoff proves it was
    withdrawn by then; otherwise it is taken as standing (counted in `ambiguous`)."""
    by_user: dict[int, list[dict]] = defaultdict(list)
    for i in obs.get("identifications") or []:
        by_user[i["user_id"]].append(i)
    out = {}
    for uid, ids in by_user.items():
        ids.sort(key=lambda r: (r["created_at"] or "", r["id"]))
        before = [r for r in ids if r["created_at"] and r["created_at"] < cutoff]
        if not before:
            continue
        lead = before[-1]
        later = [r for r in ids if r["created_at"] and r["created_at"] >= cutoff]
        if not lead["current"] and not later:
            if lead["updated_at"] and lead["updated_at"] < cutoff:
                continue                       # withdrawn before the cutoff
            out[uid] = {**lead, "withdrawal_ambiguous": True}
            continue
        out[uid] = {**lead, "withdrawal_ambiguous": False}
    return out


def community_taxon(ids: Iterable[dict]) -> int | None:
    """iNat's community taxon over one standing ID per user: the deepest taxon with at
    least two IDs at or below it and more than 2/3 agreement. An ID elsewhere in the tree
    counts against; an ID of an ancestor counts against only when it was marked as a
    disagreement."""
    ids = [i for i in ids if i.get("taxon_path")]
    if len(ids) < 2:
        return None
    to_taxon: dict[int, list[int]] = {}       # taxon -> its path from the root
    for i in ids:
        p = i["taxon_path"]
        for n, t in enumerate(p):
            to_taxon.setdefault(t, p[:n + 1])
    best = None
    for t, path in to_taxon.items():
        under = against = 0
        for i in ids:
            if t in i["taxon_path"]:
                under += 1
            elif i["taxon_id"] in path:
                against += bool(i.get("disagreement"))
            else:
                against += 1
        if under < 2 or under / (under + against) <= 2 / 3:
            continue
        key = (len(path), under / (under + against), under, -t)
        if best is None or key > best[0]:
            best = (key, t)
    return best[1] if best else None


# --- scoring -----------------------------------------------------------------------

def tier_of(count: int | None) -> str:
    if count is None:
        return "unknown"
    return "1" if count >= TIER1 else "2" if count >= TIER2 else "below"


@dataclass
class TruthTaxa:
    species: int | None      # iNat taxon id of the validated name, when a formal species
    genus: int | None
    formal_species: bool     # scored at species (a formally named species truth)


class TaxonResolver:
    """An iNat taxon id -> its current (species id, genus id), following a retired taxon
    to its single current successor. Ancestry comes from the identification itself; a
    retired one is looked up (HistoryClient.taxa, cached)."""

    def __init__(self, taxa: dict[int, dict]):
        self.taxa = taxa

    def current(self, taxon_id: int, path: list[int], rank_level) -> tuple[list[int], float | None]:
        t = self.taxa.get(taxon_id)
        if t and t.get("is_active") is False:
            succ = t.get("current_synonymous_taxon_ids") or []
            if len(succ) == 1 and succ[0] in self.taxa and not self.taxa[succ[0]].get("missing"):
                s = self.taxa[succ[0]]
                return list(s.get("ancestor_ids") or []) + ([s["id"]] if s["id"] not in (
                    s.get("ancestor_ids") or []) else []), s.get("rank_level")
        return path, rank_level

    def ref(self, taxon_id: int | None) -> dict | None:
        """{'id', 'name'} of an iNat taxon in this taxonomy (name None if unknown here)."""
        if taxon_id is None:
            return None
        return {"id": taxon_id, "name": (self.taxa.get(taxon_id) or {}).get("name")}

    def levels(self, taxon_id: int, path: list[int], rank_level) -> tuple[int | None, int | None]:
        """(species id or None, genus id or None) of an identification."""
        path, level = self.current(taxon_id, path, rank_level)
        if not path:
            return None, None
        lv = {t: (self.taxa.get(t) or {}).get("rank_level") for t in path}
        genus = next((t for t in reversed(path) if lv.get(t) == GENUS_LEVEL), None)
        species = None
        if level is not None and level <= SPECIES_LEVEL:
            species = next((t for t in reversed(path) if lv.get(t) == SPECIES_LEVEL),
                           path[-1] if level == SPECIES_LEVEL else None)
        if genus is None and level == GENUS_LEVEL:
            genus = path[-1]
        return species, genus


def hasher(salt: bytes) -> Callable[[int], str]:
    return lambda uid: hmac.new(salt, str(uid).encode(), hashlib.sha256).hexdigest()[:16]


def load_salt(path: Path) -> bytes:
    """The private key for identifier hashes; made once, kept in data/secrets."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(32), encoding="utf-8")
    return bytes.fromhex(path.read_text(encoding="utf-8").strip())


@dataclass(frozen=True)
class KeyTruth:
    """The truth as the shared answer key sees it (name_equiv, PLAN rule 4): the label and
    its genus, against the frozen taxonomy. Species is scored only on a formal label that
    resolves one-to-one to an iNat species (`species_scored`)."""
    taxonomy: object
    label: str
    genus: str | None
    species_scored: bool
    genus_known: bool = True


def key_truth(rec: dict, taxonomy) -> KeyTruth:
    label = rec.get("label") or ""
    formal = rec.get("name_kind") == "formal" and rec.get("label_rank") in ("species", None)
    return KeyTruth(taxonomy, label, rec.get("genus"),
                    bool(formal and label and taxonomy.species_id(label) is not None),
                    taxonomy.genus_known(label, rec.get("genus")))


def verdicts(raw_id: int | None, sp: int | None, ge: int | None, truth: TruthTaxa,
             kt: KeyTruth | None) -> dict:
    """Right or wrong at species and genus for one iNat taxon (an identification or the
    community taxon). With the answer key (kt): `species_right`/`genus_right` by
    name_equiv.species_credit / genus_match on the raw taxon id against the label, beside
    `*_by_id` (iNat taxon ids, E31's method before 18 Oct) and `*_plain` (the taxon's
    current name as a string, no taxonomy), on the same finds. Without it: by id only."""
    by_sp = (sp == truth.species) if (sp is not None and truth.formal_species
                                      and truth.species) else None
    by_ge = (ge == truth.genus) if (ge is not None and truth.genus) else None
    if kt is None:
        return {"species_right": by_sp, "genus_right": by_ge}
    from . import name_equiv as N
    tax = kt.taxonomy
    on_sp = sp is not None and kt.species_scored
    on_ge = ge is not None and bool(kt.genus)
    return {
        "species_right": N.species_credit(raw_id, kt.label, taxonomy=tax) if on_sp else None,
        "genus_right": N.genus_match(raw_id, kt.genus, tax, truth_species=kt.label)["strict"]
        if on_ge else None,
        "species_right_by_id": (sp == truth.species) if (on_sp and truth.species) else None,
        "genus_right_by_id": by_ge if on_ge else None,
        "species_right_plain": N.species_credit(tax.name_of(raw_id), kt.label) if on_sp else None,
        "genus_right_plain": N.genus_match(tax.name_of(ge), kt.genus)["strict"] if on_ge else None,
    }


def score_find(obs: dict, cutoff: Cutoff, truth: TruthTaxa, resolver: TaxonResolver,
               validators: set[int], key: Callable[[int], str],
               push_accounts: set[int] = frozenset(), kt: KeyTruth | None = None) -> dict:
    """One find's line for the scoring file: each person's standing ID at the cutoff (and
    their latest one after it, flagged) with role, tier and correctness, the community
    taxon then, and the strict flag. Roles: observer, validator (of this record on
    MycoMap), push_account (MycoMap's), identifier; only identifiers are scored."""
    line = {"observation_id": obs["observation_id"], "cutoff": cutoff.at,
            "cutoff_source": cutoff.source, "cutoff_kind": kind_of(cutoff.source),
            "cutoff_sources": cutoff.sources,
            "truth_species_formal": truth.formal_species,
            "truth_species_scored": kt.species_scored if kt is not None
            else bool(truth.formal_species and truth.species),
            "truth_genus_known": kt.genus_known if kt is not None else True,
            "truth_species_taxon": truth.species, "truth_genus_taxon": truth.genus}
    if obs.get("missing") or not cutoff.at:
        return {**line, "status": "missing_on_inat" if obs.get("missing") else "no_cutoff",
                "ids": []}
    standing = standing_ids(obs, cutoff.at)
    observer = obs.get("observer_id")
    def role_of(uid):
        return ("observer" if uid == observer else "validator" if uid in validators
                else "push_account" if uid in push_accounts else "identifier")

    def entry(uid, i, before):
        sp, ge = resolver.levels(i["taxon_id"], i["taxon_path"], i["rank_level"])
        return {
            "taxon": {**resolver.ref(i["taxon_id"]), "rank": i.get("taxon_rank")},
            "species_taxon": resolver.ref(sp), "genus_taxon": resolver.ref(ge),
            "identifier": key(uid), "role": role_of(uid),
            "tier": tier_of(obs["identifier_counts"].get(uid)),
            "made_at": i["created_at"], "before_cutoff": before,
            "current_now": i["current"],
            "withdrawal_ambiguous": i.get("withdrawal_ambiguous", False) if before else None,
            "at_species": sp is not None, "at_genus": ge is not None,
            **verdicts(i["taxon_id"], sp, ge, truth, kt),
        }
    ids = [entry(uid, i, True)
           for uid, i in sorted(standing.items(), key=lambda kv: kv[1]["created_at"])]
    latest_after: dict[int, dict] = {}
    for i in obs.get("identifications") or []:
        if i["created_at"] and i["created_at"] >= cutoff.at:
            latest_after[i["user_id"]] = i            # identifications are in time order
    ids += [entry(uid, i, False)
            for uid, i in sorted(latest_after.items(), key=lambda kv: kv[1]["created_at"])]
    ct = community_taxon(standing.values())
    ct_sp = ct_ge = None
    if ct is not None:
        src = next(i for i in standing.values() if ct in i["taxon_path"])
        path = src["taxon_path"][:src["taxon_path"].index(ct) + 1]
        level = (resolver.taxa.get(ct) or {}).get("rank_level")
        if level is None and ct == src["taxon_id"]:
            level = src["rank_level"]
        ct_sp, ct_ge = resolver.levels(ct, path, level)
    obs_lead = standing.get(observer)
    observer_named_species = bool(obs_lead and obs_lead["rank_level"] is not None
                                  and obs_lead["rank_level"] <= SPECIES_LEVEL)
    after = sum(1 for i in obs.get("identifications") or []
                if i["created_at"] and i["created_at"] >= cutoff.at and i["user_id"] != observer)
    return {**line, "status": "ok", "ids": ids,
            "community": {"present": ct is not None,
                          "taxon": {**resolver.ref(ct),
                                    "rank": (resolver.taxa.get(ct) or {}).get("rank")}
                          if ct is not None else None,
                          "species_taxon": resolver.ref(ct_sp), "genus_taxon": resolver.ref(ct_ge),
                          "at_species": ct_sp is not None, "at_genus": ct_ge is not None,
                          **verdicts(ct, ct_sp, ct_ge, truth, kt)},
            "observer_named_species_before_cutoff": observer_named_species,
            "ids_after_cutoff_by_others": after,
            "first_pooled": first_pooled(ids)}


def erratum_status(rec: dict, errata) -> str | None:
    """'erratum_<reason>' for a record the release's erratum lists (never scored), or None."""
    reason = errata.reasons.get(rec["record_key"]) if errata is not None else None
    return f"erratum_{reason}" if reason else None


def sensitivity_block(lines: list[dict]) -> dict:
    """The headline summaries under another verdict (with_verdicts): tiers 1 and 2, H4's
    pooled first ID."""
    return {"tier1": summarise(lines, "1"), "tier2": summarise(lines, "2"),
            "pooled_first": summarise_pooled(lines)}


def genus_counted(r: dict) -> bool:
    """Genus is scored against iNat-side comparators on this find (rule 7: iNat can name
    the truth's genus); `with_genus_kept_in` is the sensitivity that keeps every truth."""
    return bool(r.get("truth_genus_known", True))


def with_genus_kept_in(lines: list[dict]) -> list[dict]:
    return [{**r, "truth_genus_known": True} for r in lines]


def scored_species(r: dict) -> bool:
    """Species is scored on this find: a formal truth (format 3: one the key resolves)."""
    return bool(r.get("truth_species_scored", r["truth_species_formal"]))


def with_verdicts(lines: list[dict], which: str) -> list[dict]:
    """Copies of format-3 lines with a sensitivity's verdicts ('by_id', 'plain') in place of
    the answer key's, for the same summaries."""
    def swap(d):
        if not d:
            return d
        d = dict(d)
        for rank in ("species", "genus"):
            if f"{rank}_right_{which}" in d:
                d[f"{rank}_right"] = d[f"{rank}_right_{which}"]
        return d

    def swap_first(fp):
        if not fp:
            return fp
        fp = dict(fp)
        for rank in ("species", "genus"):
            if fp.get(rank) and f"right_{which}" in fp[rank]:
                fp[rank] = {**fp[rank], "right": fp[rank][f"right_{which}"]}
        return fp
    return [{**r, "ids": [swap(i) for i in r.get("ids") or []],
             "community": swap(r.get("community")),
             "first_pooled": swap_first(r.get("first_pooled"))} for r in lines]


# H4 (Steve, 18 Oct): the comparator is the FIRST identification before the cutoff by an
# identifier with >= 1,000 iNat identifications (tiers 1 and 2 pooled), at each rank.
POOLED_TIERS = ("1", "2")


def first_pooled(ids: list[dict]) -> dict:
    """Per rank, the earliest standing identifier ID before the cutoff from tiers 1+2 that
    is at that rank (whatever the truth): {'species': {'right', 'tier', 'taxon',
    'species_taxon' or 'genus_taxon'} or None, 'genus': ...}. `right` is E31's own
    taxon-id verdict, None where it cannot score (a provisional truth, or one iNat has no
    taxon for); paper-score re-scores by name_equiv from the taxa."""
    mine = sorted((i for i in ids if i["role"] == "identifier" and i["before_cutoff"]
                   and i["tier"] in POOLED_TIERS), key=lambda i: i["made_at"] or "")
    out = {"tiers": "+".join(POOLED_TIERS)}
    for rank in ("species", "genus"):
        first = next((i for i in mine if i[f"at_{rank}"]), None)
        out[rank] = {"right": first[f"{rank}_right"], "tier": first["tier"],
                     "taxon": first.get("taxon"),
                     f"{rank}_taxon": first.get(f"{rank}_taxon"),
                     **{f"right_{v}": first.get(f"{rank}_right_{v}") for v in ("by_id", "plain")
                        if f"{rank}_right_{v}" in first}} if first else None
    return out


def summarise_pooled(lines: list[dict], resamples: int = 10_000, seed: int = 20261013) -> dict:
    """H4's comparator on its own: the first tiers-1+2 ID's accuracy at species (formal
    truths that have one) and genus, with coverage and observer-clustered intervals; `all`
    and `strict` (finds the observer had not named at species before the cutoff)."""
    def block(rows):
        formal = [r for r in rows if scored_species(r)]
        out = {"finds": len(rows), "formal_species_finds": len(formal)}
        known = [r for r in rows if genus_counted(r)]
        out["genus_known_finds"] = len(known)
        for rank, base in (("species", formal), ("genus", known)):
            got = [((r.get("observer") or r["observation_id"]),
                    int(r["first_pooled"][rank]["right"]), 1)
                   for r in base if ((r.get("first_pooled") or {}).get(rank) or {}).get(
                       "right") is not None]
            out[f"first_{rank}"] = clustered_rate(got, resamples, seed)
            out[f"coverage_{rank}"] = round(len(got) / len(base), 4) if base else None
        return out
    ok = [r for r in lines if r["status"] == "ok"]
    strict = [r for r in ok if not r.get("observer_named_species_before_cutoff")]
    return {"tiers": "+".join(POOLED_TIERS), "view": "first", "all": block(ok),
            "strict": block(strict)}


def summarise(lines: list[dict], tier: str = "1", count_push_accounts: bool = False,
              drop_ambiguous: bool = False) -> dict:
    """Descriptive per-find accuracy (E31b): best tier-N, each tier-N ID, community taxon;
    species over formal truths with a species-level ID, genus over finds with a genus ID;
    coverage over all scored finds. `strict` drops finds the observer had named at species.
    Sensitivities: `count_push_accounts` scores MycoMap's push account where it did not
    validate the find; `drop_ambiguous` leaves out IDs withdrawn at an unknown time."""
    roles = {"identifier", "push_account"} if count_push_accounts else {"identifier"}
    def block(rows):
        c = Counter()
        for r in rows:
            if r["status"] != "ok":
                continue
            c["finds"] += 1
            mine = [i for i in r["ids"] if i["role"] in roles and i["tier"] == tier
                    and i["before_cutoff"]
                    and not (drop_ambiguous and i.get("withdrawal_ambiguous"))]
            for rank in ("species", "genus"):
                if rank == "genus" and not genus_counted(r):
                    continue
                got = [i[f"{rank}_right"] for i in mine if i[f"{rank}_right"] is not None]
                if got:
                    c[f"best_{rank}_n"] += 1
                    c[f"best_{rank}_right"] += any(got)
                    c[f"each_{rank}_n"] += len(got)
                    c[f"each_{rank}_right"] += sum(got)
                cr = r["community"][f"{rank}_right"]
                if cr is not None:
                    c[f"community_{rank}_n"] += 1
                    c[f"community_{rank}_right"] += cr
            c["formal_finds"] += scored_species(r)
            c["genus_finds"] += genus_counted(r)
        out = {"finds": c["finds"], "formal_species_finds": c["formal_finds"]}
        for view in ("best", "each", "community"):
            for rank in ("species", "genus"):
                n = c[f"{view}_{rank}_n"]
                out[f"{view}_{rank}"] = {"n": n, "right": c[f"{view}_{rank}_right"],
                                         "accuracy": round(c[f"{view}_{rank}_right"] / n, 4)
                                         if n else None}
        out["coverage_best_species"] = round(c["best_species_n"] / c["formal_finds"], 4) \
            if c["formal_finds"] else None
        out["genus_known_finds"] = c["genus_finds"]
        out["coverage_best_genus"] = round(c["best_genus_n"] / c["genus_finds"], 4) \
            if c["genus_finds"] else None
        return out
    strict = [r for r in lines if not r.get("observer_named_species_before_cutoff")]
    return {"tier": tier, "all": block(lines), "strict": block(strict),
            "cutoff_kinds": dict(Counter(r.get("cutoff_kind", r["cutoff_source"]) for r in lines)),
            "status": dict(Counter(r["status"] for r in lines))}


# --- E31b slices: by genus, depth band and year, with observer-clustered intervals -----

# PLAN §9 (descriptive B): "by genus, by depth band, by year"; §7: bootstrap clustered by
# observer, 10,000 resamples, percentile. Depth = included v1 `train` records of the find's
# label (the find itself left out), in the standard bands. The year bands and the genus
# floor are fixed here, before any slice is looked at.
DEPTH_BANDS = ((0, 0, "0"), (1, 4, "1-4"), (5, 19, "5-19"), (20, 99, "20-99"),
               (100, 10**9, "100+"))
YEAR_BANDS = ((0, 2017, "<=2017"), (2018, 2019, "2018-2019"), (2020, 2021, "2020-2021"),
              (2022, 2023, "2022-2023"), (2024, 9999, "2024+"))
GENUS_MIN_FINDS = 100
SLICE_METRICS = ("best_species", "best_genus", "each_species", "community_species",
                 "community_genus", "coverage_best_species")


def band_of(value: int | None, bands) -> str:
    if value is None:
        return "unknown"
    for lo, hi, label in bands:
        if lo <= value <= hi:
            return label
    return "unknown"


def find_counts(r: dict, tier: str) -> dict[str, tuple[int, int]]:
    """One scored find's (right, n) toward each slice metric, counted as `summarise` does."""
    mine = [i for i in r["ids"] if i["role"] == "identifier" and i["tier"] == tier
            and i["before_cutoff"]]
    out = {}
    for rank in ("species", "genus"):
        got = [i[f"{rank}_right"] for i in mine if i[f"{rank}_right"] is not None]
        if rank == "genus" and not genus_counted(r):
            got = []
        out[f"best_{rank}"] = (int(any(got)), 1) if got else (0, 0)
        if rank == "species":
            out["each_species"] = (sum(got), len(got))
        cr = r["community"][f"{rank}_right"]
        if rank == "genus" and not genus_counted(r):
            cr = None
        out[f"community_{rank}"] = (int(cr), 1) if cr is not None else (0, 0)
    formal = scored_species(r)
    out["coverage_best_species"] = (out["best_species"][1], 1) if formal else (0, 0)
    return out


def clustered_rate(rows: list[tuple[str, int, int]], resamples: int = 10_000,
                   seed: int = 20261013) -> dict:
    """k/n with a percentile bootstrap that resamples observers (clusters) with
    replacement. rows: (cluster, right, n) per find."""
    import numpy as np
    k, n = sum(r[1] for r in rows), sum(r[2] for r in rows)
    out = {"n": n, "right": k, "rate": round(k / n, 4) if n else None, "ci95": None,
           "observers": 0}
    by: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for c, rk, rn in rows:
        if rn:
            by[c][0] += rk
            by[c][1] += rn
    out["observers"] = len(by)
    if not n:
        return out
    ks = np.array([v[0] for v in by.values()], dtype=float)
    ns = np.array([v[1] for v in by.values()], dtype=float)
    rng = np.random.default_rng(seed)
    rates = np.empty(resamples)
    for start in range(0, resamples, 1000):           # bounded memory with many clusters
        m = min(1000, resamples - start)
        w = rng.multinomial(len(ks), np.full(len(ks), 1 / len(ks)), size=m)
        rates[start:start + m] = (w @ ks) / (w @ ns)   # every cluster here has n > 0
    lo, hi = np.percentile(rates, [2.5, 97.5])
    out["ci95"] = [round(float(lo), 4), round(float(hi), 4)]
    return out


def slices(lines: list[dict], tier: str = "1", resamples: int = 10_000,
           seed: int = 20261013) -> dict:
    """E31b's descriptive table: overall, by year band, by depth band and by genus (genera
    with >= GENUS_MIN_FINDS scored finds; the rest pooled as 'other'), each metric with an
    observer-clustered 95% interval. Scored ('ok') finds only; push accounts never count."""
    ok = [r for r in lines if r["status"] == "ok"]
    big = Counter(r.get("truth_genus") for r in ok)

    def genus_key(r):
        g = r.get("truth_genus")
        return g if g and big[g] >= GENUS_MIN_FINDS else "other"

    def year_key(r):
        y = str(r.get("observed_year") or "")
        return band_of(int(y) if y.isdigit() else None, YEAR_BANDS)

    def depth_key(r):
        return band_of(r.get("truth_train_refs"), DEPTH_BANDS)

    def table(rows):
        counts = [(r.get("observer") or r["observation_id"], find_counts(r, tier)) for r in rows]
        return {"finds": len(rows),
                **{m: clustered_rate([(c, fc[m][0], fc[m][1]) for c, fc in counts],
                                     resamples, seed) for m in SLICE_METRICS}}

    out = {"tier": tier, "resamples": resamples, "seed": seed, "cluster": "observer",
           "genus_min_finds": GENUS_MIN_FINDS, "overall": table(ok)}
    for name, keyf in (("year", year_key), ("depth", depth_key), ("genus", genus_key)):
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in ok:
            groups[keyf(r)].append(r)
        out[f"by_{name}"] = {g: table(rs) for g, rs in sorted(groups.items())}
    return out


def train_refs(rel) -> Counter:
    """Included v1 `train` records per label (the depth a find's species had in training)."""
    return Counter(r[0] for r in rel.db.execute(
        "select label from records where included = 1 and split = 'train'"))


# --- iNat's taxonomy (open data, no calls) and the truth ------------------------------

FUNGI, PROTOZOA = 47170, 47686     # slime moulds are under Protozoa on iNat


def load_open_taxa(taxa_csv_gz: Path, roots: tuple[int, ...] = (FUNGI, PROTOZOA)) -> dict[int, dict]:
    """iNat open data taxa.csv.gz (taxon_id, ancestry, rank_level, rank, name, active) ->
    {id: {id, name, rank, rank_level, is_active, ancestor_ids}} for the taxa under `roots`."""
    keep = set(map(str, roots))
    out: dict[int, dict] = {}
    with gzip.open(taxa_csv_gz, "rt", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="	"):
            anc = (row.get("ancestry") or "").split("/")
            if row["taxon_id"] not in keep and not keep.intersection(anc):
                continue
            try:
                level = float(row["rank_level"]) if row.get("rank_level") else None
            except ValueError:
                level = None
            out[int(row["taxon_id"])] = {
                "id": int(row["taxon_id"]), "name": row.get("name"), "rank": row.get("rank"),
                "rank_level": level, "is_active": str(row.get("active")).lower() == "true",
                "ancestor_ids": [int(x) for x in anc if x.isdigit()]}
    return out


def complete_taxa(client: HistoryClient, taxa: dict[int, dict], snapshot: list[dict]) -> dict[int, dict]:
    """Add what the open data lacks for the snapshot's identifications: taxa newer than
    the download, and the current successor of every retired taxon (both from iNat's
    taxa endpoint, cached)."""
    seen = {t for r in snapshot for i in r.get("identifications") or [] for t in i["taxon_path"]}
    want = {t for t in seen if t not in taxa or not taxa[t]["is_active"]}
    got = client.taxa(want) if want else {}
    succ = {s for g in got.values() for s in g.get("current_synonymous_taxon_ids") or []
            if s not in taxa}
    got.update(client.taxa(succ) if succ else {})
    merged = dict(taxa)
    for t, g in got.items():
        if not g.get("missing"):
            merged[t] = {**merged.get(t, {}), **g}
    return merged


def truth_taxa(rec: dict, by_name: dict[tuple[str, str], int], taxa: dict[int, dict]) -> TruthTaxa:
    """The validated name in iNat's taxonomy. A formal species: its iNat species and that
    species' genus. Otherwise (provisional code, one word, or unknown to iNat): the genus
    by name. Species is scored only on formal names (iNat has no provisional codes)."""
    formal = rec.get("name_kind") == "formal" and rec.get("label_rank") in ("species", None)         and bool(rec.get("species"))
    sp = by_name.get(((rec.get("species") or "").lower(), "species")) if formal else None
    genus = None
    if sp is not None:
        genus = next((a for a in reversed(taxa[sp]["ancestor_ids"])
                      if (taxa.get(a) or {}).get("rank_level") == GENUS_LEVEL), None)
    if genus is None and rec.get("genus"):
        genus = by_name.get((rec["genus"].lower(), "genus"))
    return TruthTaxa(species=sp, genus=genus, formal_species=formal)


def name_index(taxa: dict[int, dict]) -> dict[tuple[str, str], int]:
    """(name lower, rank) -> taxon id; an active taxon beats a retired one."""
    out: dict[tuple[str, str], int] = {}
    for t in sorted(taxa.values(), key=lambda t: (t["is_active"], -t["id"])):
        out[((t.get("name") or "").lower(), t.get("rank") or "")] = t["id"]
    return out


# --- the run (E31a on a sealed test release, E31b on a seeded sample of v1) ------------

FREEZE_TAG = "paper-analysis-freeze-v1"
OPEN_TAXA = "occurrence/inat-open-data-2026-09-27/taxa.csv.gz"


def work_dir(name: str) -> Path:
    """Private outputs of one set (data/, never committed)."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", name):
        raise ValueError(f"set name {name!r}: lower case letters, digits, . _ -")
    return config.DATA_DIR / "paper-comparisons" / "e31" / name


def cache_dir() -> Path:
    return config.DATA_DIR / "paper-comparisons" / "inat-history-cache"


def inat_records(rel) -> list[dict]:
    """The release's included iNat records as dicts (MO records are not on iNat)."""
    return [dict(r) for r in rel.db.execute(
        "select * from records where included = 1 and source = 'inat' order by record_key")]


def write_ids(path: Path, ids: list[str], meta: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    info = {**meta, "ids": len(ids), "ids_sha256": ids_sha256(ids), "file": path.name}
    path.with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def read_ids(path: Path) -> list[str]:
    return [s.strip() for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def freeze_tagged(research_dir: Path | None = None) -> bool:
    import subprocess
    r = subprocess.run(["git", "-C", str(research_dir or config.RESEARCH_DIR), "tag", "-l",
                        FREEZE_TAG], capture_output=True, text=True, check=False)
    return r.returncode == 0 and r.stdout.strip() == FREEZE_TAG


def read_ours(path: Path | None) -> dict[str, list[tuple[str, str]]]:
    """Our side of the DNA-visible date: CSV observation_id, source, at (UTC)."""
    out: dict[str, list[tuple[str, str]]] = defaultdict(list)
    if path:
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("at"):
                    out[str(row["observation_id"]).strip()].append((row["source"].strip(), row["at"]))
    return out


def read_validators(path: Path | None) -> dict[str, set[int]]:
    """CSV observation_id, inat_user_id: who validated the record on MycoMap."""
    out: dict[str, set[int]] = defaultdict(set)
    if path:
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if str(row.get("inat_user_id") or "").strip().isdigit():
                    out[str(row["observation_id"]).strip()].add(int(row["inat_user_id"]))
    return out


def not_this_observation(obs: dict, rec: dict, mo_ids: set[str]) -> str | None:
    """Why an iNat observation is not the validated find, or None. v1 carries some Mushroom
    Observer finds a second time as iNat records under the same number (their iNat
    namesake is a bird or a plant), and an observation nobody ever placed in Fungi or
    Protozoa cannot be the specimen either."""
    if rec.get("source_id") in mo_ids:
        return "mo_id_collision"
    paths = [i["taxon_path"] for i in obs.get("identifications") or []]
    if paths and not any(FUNGI in p or PROTOZOA in p for p in paths):
        return "not_a_fungus_on_inat"
    return None


def vocabulary(rel, errata=None, path: Path | None = None) -> list[str]:
    """The label vocabulary for the answer key's one-to-one rule (PLAN rule 4): the shared
    file when given (one label a line), else every distinct label on the release's included
    records after the erratum, train and val, all ranks."""
    if path is not None:
        from .name_equiv import read_label_vocabulary
        return read_label_vocabulary(path)
    drop = errata.reasons if errata is not None else {}
    return sorted({lab for key, lab in rel.db.execute(
        "select record_key, label from records where included = 1")
        if lab and key not in drop})


def label_resolution_changes(a, b, labels: Iterable[str], genera: Iterable[str]) -> list[str]:
    """Labels and truth genera that two Taxonomy objects resolve differently (species id,
    genus names); empty when extras leave the label side alone."""
    out = [l for l in labels if a.species_id(l) != b.species_id(l)
           or a.genus_names(l) != b.genus_names(l)]
    return out + [g for g in sorted(set(genera)) if g and a.genus_names(g) != b.genus_names(g)]


def answer_key_taxonomy(rel, client: HistoryClient, snap: list[dict], successors: Path | None,
                        out_dir: Path, errata=None, vocabulary_file: Path | None = None):
    """(the shared answer key's Taxonomy, E31's own taxa dict). The key's: iNat open data plus the frozen
    successors TSV, with what E31's identifications add (taxa newer than the download and
    retired ids' successors, from the cached taxa lookups) as name_equiv's answer-side
    extras (never applied to a taxon a label lookup reaches); the run refuses if they
    change how any label or truth genus resolves. The extras are written to taxonomy-extras.json
    and cited by sha256 with the vocabulary's."""
    from . import name_equiv as N
    labels = vocabulary(rel, errata, vocabulary_file)
    genera = sorted({g for (g,) in rel.db.execute(
        "select distinct genus from records where included = 1") if g})
    frozen = N.Taxonomy.load(config.DATA_DIR / OPEN_TAXA, labels=labels, successors=successors)
    base = {t: dict(v) for t, v in frozen.taxa.items()}
    merged = complete_taxa(client, {t: dict(v) for t, v in base.items()}, snap)
    # every entry the cached lookups add or change; name_equiv applies them to answers only
    # (never to a taxon a label lookup reaches) and the guard below re-checks that
    extras = {t: merged[t] for t in sorted(merged) if base.get(t) != merged[t]}
    extras_path = out_dir / "taxonomy-extras.json"
    extras_path.write_text(json.dumps({str(t): v for t, v in extras.items()}, sort_keys=True,
                                      indent=0), encoding="utf-8")
    vocab_sha = (sha256_file(vocabulary_file) if vocabulary_file else
                 hashlib.sha256(("\n".join(labels) + "\n").encode("utf-8")).hexdigest())
    # name_equiv's extras are answer-only and the snapshot wins on a shared id, so only the
    # taxa the snapshot lacks take effect; changes to snapshot entries are counted, unused
    tax = N.Taxonomy(base, labels, extras=extras, cite={**frozen.cite, "extras": extras_path.name,
                                         "extras_taxa": len(extras),
                                         "extras_sha256": sha256_file(extras_path),
                                         "vocabulary": str(vocabulary_file.name)
                                         if vocabulary_file else "included labels after erratum",
                                         "vocabulary_labels": len(labels),
                                         "vocabulary_sha256": vocab_sha})
    changed = label_resolution_changes(frozen, tax, labels, genera)
    if changed:
        raise RuntimeError(f"taxonomy extras change how {len(changed)} labels resolve "
                           f"(e.g. {changed[:5]}): PLAN rule 4 forbids it")
    tax.cite["labels_kept_apart"] = sum(len(v) for v in tax.shared_by.values())
    tax.cite["extras_not_applied_label_reach"] = len(set(extras) & tax.reached)
    tax.cite["answer_key"] = N.ANSWER_KEY
    return tax, merged          # E31's own taxonomy (by-id verdicts, ID ranks): every lookup


def score(rel, name: str, client: HistoryClient, ours_csv: Path | None,
          validators_csv: Path | None, push_accounts: set[int], salt_path: Path,
          sealed_ok: bool = False, log=print, successors: Path | None = None,
          errata=None, answer_key: bool = True,
          vocabulary_file: Path | None = None) -> dict:
    """Score a snapshot: one hashed line per find (scores.jsonl) and the tier summaries.
    On a test release this refuses until the analysis freeze tag exists. Verdicts come
    from the shared answer key (name_equiv, frozen taxonomy) unless answer_key=False; an
    erratum's records are listed but not scored (status erratum_<reason>)."""
    if rel.manifest.get("test_for") and not (sealed_ok or freeze_tagged()):
        raise RuntimeError(f"{rel.id} is a sealed test release: scoring waits for the tag "
                           f"{FREEZE_TAG} in the research repo")
    d = work_dir(name)
    snap = read_snapshot(d / "snapshot.jsonl.gz")
    records = {r["source_id"]: r for r in inat_records(rel)}
    if answer_key:
        tax, taxa = answer_key_taxonomy(rel, client, snap, successors, d, errata,
                                        vocabulary_file)
    else:
        tax, taxa = None, complete_taxa(client, load_open_taxa(config.DATA_DIR / OPEN_TAXA), snap)
    by_name = name_index(taxa)
    resolver = TaxonResolver(taxa)
    ours, validators = read_ours(ours_csv), read_validators(validators_csv)
    unresolved = 0
    if validators_csv:
        with open(validators_csv, encoding="utf-8", newline="") as f:
            unresolved = sum(1 for r in csv.DictReader(f)
                             if r.get("kind") == "human" and not r.get("inat_user_id"))
    key = hasher(load_salt(salt_path))
    mo_ids = {r[0] for r in rel.db.execute("select source_id from records where source = 'mo'")}
    refs = train_refs(rel)          # empty on a test release: depth is then not reported
    lines = []
    for o in snap:
        rec = records.get(o["observation_id"])
        if rec is None:
            raise RuntimeError(f"{o['observation_id']} is in the snapshot but not in {rel.id}")
        mine, backdated = drop_backdated(ours.get(o["observation_id"], []), o.get("created_at"))
        cut = dna_cutoff(o, mine, rec.get("validated_on"))
        line = score_find(o, cut, truth_taxa(rec, by_name, taxa), resolver,
                          validators.get(o["observation_id"], set()), key, push_accounts,
                          key_truth(rec, tax) if tax is not None else None)
        line["status"] = not_this_observation(o, rec, mo_ids) or line["status"]
        line["status"] = erratum_status(rec, errata) or line["status"]
        line["backdated_trashed_dropped"] = backdated
        label_genus = by_name.get(((rec.get("genus") or "").lower(), "genus"))
        tt = truth_taxa(rec, by_name, taxa)
        line["truth_taxa"] = {"label": rec.get("label"), "label_genus": rec.get("genus"),
                              "species": resolver.ref(tt.species),
                              "genus": resolver.ref(tt.genus),
                              "label_genus_taxon": resolver.ref(label_genus)}
        line.update(record_key=rec["record_key"], split=rec.get("split"),
                    name_kind=rec.get("name_kind"), observed_year=(rec.get("observed_on") or "")[:4],
                    truth_genus=rec.get("genus"), observer=key(o["observer_id"])
                    if o.get("observer_id") else None, format=SCORES_FORMAT,
                    truth_train_refs=(refs[rec.get("label")] - (
                        rec.get("split") == "train" and bool(rec.get("included"))))
                    if refs else None)
        lines.append(line)
    out = d / "scores.jsonl"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(json.dumps(line, sort_keys=True) + "\n")
    summary = {"set": name, **rel.cite(), "code_version": config.code_version(),
               # the iNat identification-history snapshot (not the taxonomy: answer_key.taxa_sha256)
               "identification_snapshot_sha256": sha256_file(d / "snapshot.jsonl.gz"),
               "snapshot_sha256": sha256_file(d / "snapshot.jsonl.gz"),   # same; older readers
               "scores_sha256": sha256_file(out), "finds": len(lines),
               "validator_lists": bool(validators_csv), "our_dates": bool(ours_csv),
               "push_accounts_excluded": len(push_accounts),
               "answer_key": tax.cite if tax is not None else None,
               "errata": errata.cite() if errata is not None else None,
               "genus_not_known_to_inat": {
                   "finds": sum(1 for l in lines if l["status"] == "ok" and not genus_counted(l)),
                   "identifier_calls": sum(
                       1 for l in lines if l["status"] == "ok" and not genus_counted(l)
                       for i in l["ids"] if i["role"] == "identifier" and i["before_cutoff"]
                       and i.get("genus_right") is not None)},
               "formal_species_not_resolved": sum(
                   1 for l in lines if l["status"] == "ok" and l["truth_species_formal"]
                   and not l["truth_species_scored"]),
               "tier1": summarise(lines, "1"), "tier2": summarise(lines, "2"),
               "pooled_first": summarise_pooled(lines),
               "slices": {t: slices(lines, t) for t in ("1", "2")},
               "sensitivity": {
                   "push_account_counted_where_not_validator": {
                       t: summarise(lines, t, count_push_accounts=True) for t in ("1", "2")},
                   "without_withdrawal_ambiguous": {
                       t: summarise(lines, t, drop_ambiguous=True) for t in ("1", "2")},
                   "withdrawal_ambiguous_ids": sum(
                       1 for l in lines if l["status"] == "ok" for i in l["ids"]
                       if i["role"] == "identifier" and i["before_cutoff"]
                       and i.get("withdrawal_ambiguous")),
                   "unresolved_validator_rows": unresolved,
                   "genus_kept_in": sensitivity_block(with_genus_kept_in(lines)),
                   **({"taxon_id_only": sensitivity_block(with_verdicts(lines, "by_id")),
                       "plain_strings": sensitivity_block(with_verdicts(lines, "plain"))}
                      if tax is not None else {})}}
    (d / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# --- our side of the cutoff: the read-only SQL routes ---------------------------------

# Source names the SQL gives (paper/e31/sql in the research repo) -> the kind reported.
SOURCE_KIND = {
    "com_seq_created": "sequence_created", "org_seq_created": "sequence_created",
    "com_seq_created_trashed": "sequence_created", "org_seq_created_trashed": "sequence_created",
    "com_genbank_files_prepared": "sequence_created",
    "com_genbank_record_mirrored": "sequence_public",
    "com_validation_verified_first": "validation", "org_validation_verified_first": "validation",
    "com_inat_fields_push_last": "name_push", "org_autovalidate_name_push_last": "name_push",
    "org_name_update_field_push_last": "name_push", "org_name_update_identification": "name_push",
    "org_autovalidate_inat_comment": "comment", "com_inat_dna_field_seen_by": "dna_field",
}
TRASHED = ("com_seq_created_trashed", "org_seq_created_trashed")


def kind_of(source: str) -> str:
    return SOURCE_KIND.get(source, source)


def drop_backdated(ours: list[tuple[str, str]], obs_created: str | None) -> tuple[list, int]:
    """A trashed .com sequence can be an iNat import moved to the trash, whose date is the
    observation's own creation time (backdated, not when MycoMap had DNA): one dated
    within a day of the observation's creation is set aside and counted."""
    if not obs_created:
        return list(ours), 0
    t0 = datetime.fromisoformat(utc(obs_created).replace("Z", "+00:00"))
    keep, dropped = [], 0
    for source, ts in ours:
        if source in TRASHED:
            t = datetime.fromisoformat(utc(ts).replace("Z", "+00:00"))
            if abs((t - t0).total_seconds()) <= 86400:
                dropped += 1
                continue
        keep.append((source, ts))
    return keep, dropped


def run_sql(host: str, template: str, ids: list[str], batch: int = 500,
            runner: Callable[[str, str], str] | None = None) -> list[dict]:
    """Run a read-only SQL template ({ids} = a comma list of numeric ids) over the ssh route
    `host` in batches; tab-separated answers with a header line -> dicts."""
    import subprocess

    def ssh(h: str, sql: str) -> str:
        r = subprocess.run(["ssh", h, sql], capture_output=True, text=True, encoding="utf-8",
                           check=False, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"SQL route failed ({r.returncode}): {r.stderr.strip()[:300]}")
        return r.stdout
    runner = runner or ssh
    body = " ".join(re.sub(r"--[^\n]*", "", template).split())   # ssh refuses a leading --
    rows = []
    for i in range(0, len(ids), batch):
        chunk = [x for x in ids[i:i + batch] if x.isdigit()]
        out = runner(host, body.replace("{ids}", ",".join(chunk)))
        lines = [l for l in out.splitlines() if l.strip()]
        if not lines:
            continue
        head = lines[0].split("\t")
        rows += [dict(zip(head, l.split("\t"))) for l in lines[1:]]
    return rows


def dna_dates(name: str, ids: list[str], sql_dir: Path, org_host: str, com_host: str,
              runner=None, log=print) -> dict:
    """Pull our side of the DNA-visible date and the validators' iNat ids into the set's
    folder: our_dates.csv (observation_id, source, at) and validators.csv
    (observation_id, inat_user_id, kind; a login without an id is kept for the API)."""
    d = work_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    dates = []
    for host, f in ((com_host, "com_dna_visible.sql"), (org_host, "org_dna_visible.sql")):
        for r in run_sql(host, (sql_dir / f).read_text(encoding="utf-8"), ids, runner=runner):
            dates.append({"observation_id": r["inat_observation_id"], "source": r["source"],
                          "at": utc(r["earliest_timestamp_utc"])})
        log(f"  {f}: {len(dates):,} rows so far")
    with open(d / "our_dates.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, ["observation_id", "source", "at"])
        w.writeheader()
        w.writerows(dates)
    vals = run_sql(org_host, (sql_dir / "org_validators.sql").read_text(encoding="utf-8"), ids,
                   runner=runner)
    with open(d / "validators.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, ["observation_id", "inat_user_id", "kind", "login_if_no_id"])
        w.writeheader()
        for r in vals:
            w.writerow({"observation_id": r["inat_observation_id"],
                        "inat_user_id": r.get("validator_inat_user_id") or "",
                        "kind": r.get("validator_kind"),
                        "login_if_no_id": r.get("inat_login_if_no_id") or ""})
    return {"our_dates": len(dates), "validator_rows": len(vals),
            "by_source": dict(Counter(r["source"] for r in dates)),
            "validator_kinds": dict(Counter(r.get("validator_kind") for r in vals)),
            "logins_to_resolve": len({r["inat_login_if_no_id"] for r in vals
                                      if r.get("inat_login_if_no_id")
                                      and not r.get("validator_inat_user_id")})}


def resolve_logins(client: HistoryClient, name: str) -> int:
    """Validators known only by iNat login: their numeric id from iNat (cached), written
    back into validators.csv. Returns how many rows gained an id."""
    p = work_dir(name) / "validators.csv"
    with open(p, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    got = 0
    for r in rows:
        login = r.get("login_if_no_id")
        if login and not r.get("inat_user_id"):
            key = "login-" + hashlib.sha256(login.lower().encode()).hexdigest()[:24]
            hit = client.cached("users", key)
            if hit is None:
                code, body = client._get(f"{V1}/users/{requests.utils.quote(login)}")
                res = (body or {}).get("results") or []
                hit = {"id": res[0].get("id") if res else None, "missing": None if res else code}
                client._store("users", key, hit)
            if hit.get("id"):
                r["inat_user_id"] = str(hit["id"])
                got += 1
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, ["observation_id", "inat_user_id", "kind", "login_if_no_id"])
        w.writeheader()
        w.writerows(rows)
    return got
