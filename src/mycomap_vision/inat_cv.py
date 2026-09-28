"""iNaturalist's own computer vision as a baseline on the scoreboard.

Each photo of a test record is scored by iNat's (unpublished) endpoint
POST /v1/computervision/score_image with aggregated=true, which scores every
rank. A record's answer at each rank is the taxon with the best score on any of
its photos ("strongest photo wins"). Two variants go on the scoreboard:
  - vision-max:   photo only (vision_score)
  - combined-max: photo plus iNat's location model (normalized_combined_score)
and each also reports the first photo alone, the way the iNat app starts.

Truth is judged in iNat's taxonomy: the record's DNA name, its genus and that
genus's family are looked up on iNat, following a retired name to its current
one, and compared by taxon id, so renames aren't counted as misses. The full
name is tried first, since iNat has begun adding provisional names (e.g.
"Clitocybe sp. 'IN13'") as taxa its vision model can suggest; then the plain
binomial. Species accuracy is also given on the names iNat knows, because iNat
can't be right about a name it doesn't have.

Location sent is iNat's own public (possibly obscured) coordinates for the
observation, never the true coordinates MycoMap holds. Responses are cached,
so a rerun costs nothing; requests are paced at 1 per second.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests

from . import config
from .evaluate import (RANKS, Record, SharedSet, bucket_of, comparison_backbones, save_run,
                       shared_records)
from .ratelimit import MinInterval

API = "https://api.inaturalist.org/v1"
BACKBONE = "external:inat-cv"
JWT_FILE = config.DATA_DIR / "secrets" / "inat_jwt.txt"
NOT_AN_EPITHET = {"sp", "spp", "cf", "aff", "sect", "group", "complex", "var", "ssp", "subsp"}


# --- pure parts -----------------------------------------------------------------

def parse_aggregated(body: dict) -> dict[str, dict[int, dict]]:
    """score_image response -> {rank: {taxon_id: {name, vision, combined}}}."""
    out: dict[str, dict[int, dict]] = {rank: {} for rank in RANKS}
    for r in body.get("results") or []:
        taxon = r.get("taxon") or {}
        rank, tid = taxon.get("rank"), taxon.get("id") or r.get("taxon_id")
        if rank not in out or tid is None or not isinstance(taxon.get("name"), str):
            continue
        out[rank][int(tid)] = {
            "name": taxon["name"],
            "vision": float(r.get("vision_score") or 0),
            "combined": float(r.get("normalized_combined_score", r.get("combined_score")) or 0),
        }
    return out


def plain_binomial(name: str) -> str | None:
    """'Russula emetica' -> itself; provisional or open names ('X sp. IN13', "X 'a'") -> None."""
    words = name.strip().split()
    if len(words) < 2:
        return None
    genus, epithet = words[0], words[1].rstrip(".")
    if not re.fullmatch(r"[A-Z][a-z-]+", genus):
        return None
    if not re.fullmatch(r"[a-z][a-z-]+", epithet) or epithet in NOT_AN_EPITHET:
        return None
    if len(words) > 2 and not words[2].startswith(("var.", "subsp.", "f.")):
        return None
    return f"{genus} {epithet}"


def best_per_rank(photos: list[dict[str, dict[int, dict]]], score: str) -> dict[str, list[tuple[int, float]]]:
    """Per rank: taxa ranked by their best score on any photo."""
    out = {}
    for rank in RANKS:
        best: dict[int, float] = {}
        for p in photos:
            for tid, t in p.get(rank, {}).items():
                best[tid] = max(best.get(tid, 0.0), t[score])
        out[rank] = sorted(best.items(), key=lambda kv: -kv[1])
    return out


@dataclass
class Truth:
    species: int | None          # iNat taxon id of the DNA name, when iNat knows it
    genus: int | None
    family: int | None
    species_known: bool          # iNat has a taxon for the DNA name (provisional ones too)


def score_records(records: list[Record], photo_scores: dict[str, list[dict]],
                  truths: dict[str, Truth], ref_count: Counter, score: str,
                  first_photo_only: bool = False, top_k: int = 5) -> dict:
    """Same shape as evaluate.evaluate(): per rank, per reference bucket, top-1 and top-5."""
    tally = {rank: defaultdict(Counter) for rank in RANKS}
    known = Counter()
    for rec in records:
        photos = photo_scores.get(rec.observation_id) or []
        if first_photo_only:
            photos = photos[:1]
        ranked = best_per_rank(photos, score)
        truth = truths[rec.observation_id]
        b = bucket_of(ref_count.get(rec.species, 0))
        for rank in RANKS:
            want = getattr(truth, rank)
            top = [tid for tid, _ in ranked[rank][:top_k]]
            hit1 = want is not None and top[:1] == [want]
            hit5 = want is not None and want in top
            keys = ["all", b]
            if rank == "species" and truth.species_known:
                keys.append("names iNat knows")
            for key in keys:
                c = tally[rank][key]
                c["n"] += 1
                c["top1"] += hit1
                c[f"top{top_k}"] += hit5
    return {rank: {k: {"n": c["n"], "top1": round(c["top1"] / c["n"], 4),
                       f"top{top_k}": round(c[f"top{top_k}"] / c["n"], 4)}
                   for k, c in tally[rank].items() if c["n"]}
            for rank in RANKS}


# --- iNat access (cached, paced) ---------------------------------------------------

class InatClient:
    def __init__(self, jwt: str, cache_dir: Path, session: requests.Session | None = None,
                 interval: float = 1.0, sleep: Callable[[float], None] = time.sleep):
        self.jwt = jwt.strip()
        self.cache_dir = cache_dir
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = config.USER_AGENT
        self.pace = MinInterval(interval, sleep=sleep)
        self.calls = 0

    def _cached(self, key: str, fetch: Callable[[], dict]) -> dict:
        path = self.cache_dir / f"{hashlib.sha1(key.encode()).hexdigest()}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        body = fetch()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body), encoding="utf-8")
        return body

    def _request(self, method: str, url: str, **kw) -> dict:
        for attempt in range(5):
            self.pace.wait()
            self.calls += 1
            try:
                r = self.session.request(method, url, timeout=60, **kw)
            except requests.RequestException:
                # Dropped connections and TLS resets happen; back off and try again.
                self.pace.pause(10 * (attempt + 1))
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 401:
                raise RuntimeError("iNat refused the token (401): it may have expired; "
                                   "tokens last 24 hours. Save a fresh one.")
            if r.status_code == 429 or r.status_code >= 500:
                self.pace.pause(30 * (attempt + 1))
                continue
            raise RuntimeError(f"iNat answered {r.status_code}: {r.text[:200]}")
        raise RuntimeError("iNat kept failing; stopped. Rerun to resume (answers are cached).")

    def score_image(self, photo_id: int, image: bytes, lat: float | None,
                    lng: float | None) -> dict:
        loc = f"{lat:.4f},{lng:.4f}" if lat is not None and lng is not None else "none"
        data = {"aggregated": "true"}
        if loc != "none":
            data.update(lat=str(lat), lng=str(lng))

        def fetch():
            return self._request("POST", f"{API}/computervision/score_image", data=data,
                                 files={"image": (f"{photo_id}.jpg", image, "image/jpeg")},
                                 headers={"Authorization": self.jwt})
        return self._cached(f"score:{photo_id}:{loc}", fetch)

    def get(self, path: str, params: dict | None = None) -> dict:
        key = f"get:{path}?{json.dumps(params or {}, sort_keys=True)}"
        return self._cached(key, lambda: self._request("GET", f"{API}{path}", params=params))


def _ancestor(taxon: dict, rank: str) -> int | None:
    if taxon.get("rank") == rank:
        return int(taxon["id"])
    for a in taxon.get("ancestors") or []:
        if a.get("rank") == rank:
            return int(a["id"])
    return None


def lookup(client, name: str, rank: str) -> dict | None:
    """Exact-name iNat taxon at one rank, active names first, following a retired name to
    its single current one. Returns the full taxon (with ancestors) or None."""
    hit = None
    for active in ("true", "false"):
        body = client.get("/taxa", {"q": name, "rank": rank, "is_active": active, "per_page": 30})
        hit = next((t for t in body.get("results") or []
                    if str(t.get("name", "")).lower() == name.lower()), None)
        if hit:
            break
    if not hit:
        return None
    taxon = (client.get(f"/taxa/{hit['id']}").get("results") or [None])[0]
    if taxon and taxon.get("is_active") is False:
        current = taxon.get("current_synonymous_taxon_ids") or []
        if len(current) == 1:
            taxon = (client.get(f"/taxa/{current[0]}").get("results") or [taxon])[0]
    return taxon


def resolve_truth(client, rec: Record) -> Truth:
    # Full name first: iNat now carries some provisional names as species-rank taxa.
    sp = lookup(client, rec.species, "species")
    binomial = plain_binomial(rec.species)
    if sp is None and binomial and binomial != rec.species:
        sp = lookup(client, binomial, "species")
    genus_taxon = None
    if sp:
        genus_id, family_id = _ancestor(sp, "genus"), _ancestor(sp, "family")
    else:
        genus_taxon = lookup(client, rec.genus, "genus") if rec.genus else None
        genus_id = int(genus_taxon["id"]) if genus_taxon else None
        family_id = _ancestor(genus_taxon, "family") if genus_taxon else None
    return Truth(species=int(sp["id"]) if sp else None, genus=genus_id, family=family_id,
                 species_known=sp is not None)


# --- the run ---------------------------------------------------------------------

def read_jwt(path: Path = JWT_FILE) -> str:
    if not path.is_file() or not path.read_text(encoding="utf-8").strip():
        raise RuntimeError(
            f"No iNat token at {path}. Sign in at https://www.inaturalist.org/users/api_token, "
            "copy the token, then in PowerShell: Get-Clipboard | Set-Content -NoNewline "
            f"'{path}'  (it lasts 24 hours; data/ is git-ignored).")
    return path.read_text(encoding="utf-8").strip()


def photo_inputs(conn: sqlite3.Connection, rec: Record) -> list[tuple[int, Path]]:
    """(photo id, local file) for the record's photos, in position order."""
    out = []
    for pid in rec.photo_rows:            # photo_rows hold photo ids in a SharedSet
        row = conn.execute("select local_path, store from photos where photo_id = ?",
                           (pid,)).fetchone()
        if row and row[0]:
            root = Path(row[1]) if row[1] and not str(row[1]).startswith("s3://") else config.DATA_DIR
            out.append((pid, root / row[0]))
    return out


def run(conn: sqlite3.Connection, comparison_id: str, client: InatClient,
        embeddings_root: Path | None = None, log=print) -> dict:
    backbones = comparison_backbones(conn, comparison_id)
    if not backbones:
        raise ValueError(f"no comparison {comparison_id!r}")
    saved = conn.execute("select record_set, test_days from eval_runs where comparison_id = ? "
                         "limit 1", (comparison_id,)).fetchone()
    shared: SharedSet = shared_records(conn, backbones, test_days=saved[1],
                                       embeddings_root=embeddings_root)
    if shared.record_set != saved[0]:
        raise RuntimeError("the comparison's records have changed since it ran (new data or "
                           "embeddings); run mv compare again and score that one")
    ref_count = Counter(r.species for r in shared.ref)
    photo_scores: dict[str, list[dict]] = {}
    truths: dict[str, Truth] = {}
    years: dict[str, str] = {}
    for i, rec in enumerate(shared.test, 1):
        obs = conn.execute("select inat_latitude, inat_longitude, observed_on from "
                           "inat_observations where observation_id = ?",
                           (rec.observation_id,)).fetchone()
        lat, lng, observed = obs if obs else (None, None, None)
        years[rec.observation_id] = (observed or "")[:4] or "unknown"
        photo_scores[rec.observation_id] = [
            parse_aggregated(client.score_image(pid, path.read_bytes(), lat, lng))
            for pid, path in photo_inputs(conn, rec)]
        truths[rec.observation_id] = resolve_truth(client, rec)
        if i % 10 == 0:
            log(f"  {i}/{len(shared.test)} test records, {client.calls} iNat calls")
    known = sum(t.species_known for t in truths.values())
    runs = []
    for method, score in (("vision-max", "vision"), ("combined-max", "combined")):
        all_photos = score_records(shared.test, photo_scores, truths, ref_count, score)
        first = score_records(shared.test, photo_scores, truths, ref_count, score,
                              first_photo_only=True)
        by_year = {}
        for year in sorted(set(years.values())):
            subset = [r for r in shared.test if years[r.observation_id] == year]
            by_year[year] = score_records(subset, photo_scores, truths, ref_count, score)[
                "species"].get("all")
        runs.append(save_run(conn, comparison_id, BACKBONE, method, shared, saved[1],
                             all_photos, first,
                             extra={"species_names_inat_knows": known,
                                    "species_by_observed_year": by_year}))
    return {"comparison_id": comparison_id, "test_records": len(shared.test),
            "species_names_inat_knows": known, "inat_calls": client.calls, "runs": runs}
