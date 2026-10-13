"""The sealed out-of-distribution test set for the not-a-fungus gate (`mv ood-set`).

A gate tuned on one set of non-fungus photos must be judged on others. This builds a new
set from sources the earlier gate never saw, once, before the paper's sealed test is
opened; nothing scores it until the analysis freeze. Every photo is CC0 or CC BY, with
its licence, author and source page kept, and its sha256 in the manifest, which is
committed (photos never are) before the unseal.

- Everyday non-fungi: Open Images V5 *train* split (the earlier set used its validation
  split for evaluation and its test split for training negatives), human-verified image
  labels, CC BY only, any image labelled Mushroom left out.
- Near-OOD: iNaturalist research-grade observations in North America, CC0 / CC BY photos
  from the open-data bucket (iNat's API at most 1 request/s with config.USER_AGENT), one
  photo per observation: plants (mushroom-like plants among them), mosses and
  liverworts, lichens, galls, insects, plant pathogens and insect pathogens. Lichens and
  most pathogens are fungi: each photo's `kind` says so, and `in_v1_labels` says whether
  its taxon is a label Vision trains on, so the gate's analysis can treat them apart.
- Left out: every photo and observation of the earlier set (its manifests), and every
  observation that is a MycoMap record.

Seeded; a re-run keeps what it already has (resumable). Draws differ if iNat's results
change, so the manifest, not the command, is the record of the set.
"""

from __future__ import annotations

import csv
import hashlib
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

SEED = "ood-v1-2026-10-13"
NA = 97394
MUSHROOM_LABEL = "/m/052sf"
OI_URL = "https://open-images-dataset.s3.amazonaws.com/train/{}.jpg"
CC_LICENCES = ("cc0", "cc-by")

FIELDS = ["file", "source", "split", "group", "category", "kind", "observation_id",
          "photo_id", "taxon_id", "taxon_name", "in_v1_labels", "owner", "license",
          "page_url", "photo_url", "sha256", "bytes"]


@dataclass(frozen=True)
class OIGroup:
    group: str
    category: str
    labels: tuple[str, ...]          # Open Images label ids (any of them)
    want: int


@dataclass(frozen=True)
class InatGroup:
    group: str
    category: str
    kind: str                        # near-ood | fungal-pathogen | near-ood-fungus
    taxon_id: int
    want: int


# ~700 everyday photos (Open Images label ids as in its class descriptions).
OI_GROUPS = (
    OIGroup("oi-people", "person", ("/m/01g317", "/m/0dzct"), 100),
    OIGroup("oi-buildings-scenery", "scene", ("/m/0cgh4", "/m/079cl", "/m/03jm5", "/m/07j7r"), 100),
    OIGroup("oi-vehicles", "object", ("/m/0k4j", "/m/0199g", "/m/019jd"), 60),
    OIGroup("oi-food", "food", ("/m/0663v", "/m/09728", "/m/052lwg6", "/m/021mn", "/m/0jy4k",
                                "/m/033cnk", "/m/0f4s2w", "/m/02xwb"), 120),
    OIGroup("oi-objects", "object", ("/m/0138tl", "/m/018xm", "/m/04dr76w", "/m/0bt_c3",
                                     "/m/050k8", "/m/01m2v", "/m/04brg2", "/m/0c_jw", "/m/09j5n",
                                     "/m/02dl1y", "/m/0242l", "/m/06msq", "/m/0h99cwc"), 160),
    OIGroup("oi-pets", "animal", ("/m/0bt9lr", "/m/01yrx"), 60),
    OIGroup("oi-birds", "animal", ("/m/015p6", "/m/09ddx"), 50),
    OIGroup("oi-flowers-houseplants", "plant", ("/m/0c9ph5", "/m/03fp41"), 50),
)

# ~800 near-OOD photos (iNat taxon ids checked against the taxa API, 2026-10-13).
INAT_GROUPS = (
    InatGroup("dicots", "plant", "near-ood", 47124, 80),
    InatGroup("monocots", "plant", "near-ood", 47163, 40),
    InatGroup("ferns", "plant", "near-ood", 121943, 20),
    InatGroup("conifers", "plant", "near-ood", 136329, 20),
    InatGroup("mushroom-like-plants", "plant", "near-ood", 451956, 40),   # Monotropoideae
    InatGroup("mosses", "moss", "near-ood", 311249, 80),
    InatGroup("liverworts", "moss", "near-ood", 64615, 20),
    InatGroup("lichens", "lichen", "near-ood-fungus", 54743, 120),
    InatGroup("oak-gall-wasps", "gall", "near-ood", 205775, 50),
    InatGroup("gall-midges", "gall", "near-ood", 55699, 25),
    InatGroup("gall-mites", "gall", "near-ood", 55584, 25),
    InatGroup("insects", "insect", "near-ood", 47158, 120),
    InatGroup("rusts", "plant-pathogen", "fungal-pathogen", 69968, 40),
    InatGroup("powdery-mildews", "plant-pathogen", "fungal-pathogen", 55525, 20),
    InatGroup("black-knot", "plant-pathogen", "fungal-pathogen", 58700, 10),
    InatGroup("exobasidium", "plant-pathogen", "fungal-pathogen", 130026, 10),
    InatGroup("downy-mildews", "plant-pathogen", "near-ood", 52597, 10),   # oomycetes
    InatGroup("white-rusts", "plant-pathogen", "near-ood", 59966, 10),     # oomycetes
    InatGroup("cordyceps-family", "insect-pathogen", "fungal-pathogen", 179204, 30),
    InatGroup("entomophthorales", "insect-pathogen", "fungal-pathogen", 55115, 30),
)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def is_jpeg(b: bytes) -> bool:
    return b[:3] == b"\xff\xd8\xff"


def excluded_ids(manifests: list[Path]) -> tuple[set[str], set[str]]:
    """(photo ids, observation ids) of the earlier set's manifests, never drawn again."""
    photos, obs = set(), set()
    for m in manifests:
        with open(m, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("photo_id"):
                    photos.add(str(r["photo_id"]))
                if r.get("observation_id"):
                    obs.add(str(r["observation_id"]))
    return photos, obs


def oi_pool(labels: dict[str, set[str]], meta: dict[str, dict], group: OIGroup,
            used: set[str], excluded: set[str]) -> list[str]:
    """Open Images ids with one of the group's labels, CC BY, not labelled Mushroom, not
    used or excluded, shuffled by the group's seed."""
    want = set(group.labels)
    pool = sorted(i for i, ls in labels.items()
                  if ls & want and MUSHROOM_LABEL not in ls and i not in used
                  and i not in excluded
                  and "creativecommons.org/licenses/by/" in meta.get(i, {}).get("License", ""))
    random.Random(f"{SEED}-{group.group}").shuffle(pool)
    return pool


def pick_inat_photo(obs: dict, open_host: str) -> dict | None:
    """The first CC0 / CC BY photo of an observation that is served from the open-data
    bucket, else None."""
    from .licenses import host_of
    return next((p for p in obs.get("photos") or []
                 if (p.get("license_code") or "") in CC_LICENCES
                 and host_of(p.get("url") or "") == open_host), None)


def read_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_manifest(path: Path, rows: list[dict]) -> str:
    """The manifest (sorted by file), LF line endings; returns its sha256."""
    rows = sorted(rows, key=lambda r: r["file"])
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        w = csv.DictWriter(fh, FIELDS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    tmp.replace(path)
    return sha256_bytes(path.read_bytes())


def summary(rows: list[dict]) -> dict:
    by = defaultdict(lambda: defaultdict(int))
    for r in rows:
        by["source"][r["source"]] += 1
        by["kind"][r["kind"]] += 1
        by["group"][r["group"]] += 1
        by["license"][r["license"]] += 1
    return {"photos": len(rows), **{k: dict(sorted(v.items())) for k, v in by.items()}}


# --- the network side ----------------------------------------------------------------------------

class Paced:
    """GETs at most once per `interval` seconds, retried on dropped connections."""

    def __init__(self, session, interval: float = 1.05, sleep=time.sleep, clock=time.time):
        self.s, self.interval, self.sleep, self.clock, self.last = session, interval, sleep, clock, 0.0

    def get(self, url, **kw):
        import requests
        for attempt in range(5):
            self.sleep(max(0.0, self.interval - (self.clock() - self.last)))
            self.last = self.clock()
            try:
                return self.s.get(url, timeout=60, **kw)
            except requests.RequestException:
                if attempt == 4:
                    raise
                self.sleep(30 * (attempt + 1))


def build(out: Path, oi_dir: Path | None, excluded_manifests: list[Path],
          mycomap_inat_ids: set[str], v1_labels: set[str], session=None, api_get=None,
          photo_get=None, log=print) -> dict:
    """Collect the set into `out` (photos/ + manifest.csv); resumable."""
    import requests

    from . import config
    from .licenses import OPEN_DATA_HOST
    out.mkdir(parents=True, exist_ok=True)
    (out / "photos").mkdir(exist_ok=True)
    rows = read_manifest(out / "manifest.csv")
    have = {r["group"]: sum(1 for x in rows if x["group"] == r["group"]) for r in rows}
    ex_photos, ex_obs = excluded_ids(excluded_manifests)
    if session is None:
        session = requests.Session()
        session.headers["User-Agent"] = config.USER_AGENT
    api = api_get or Paced(session).get
    fetch = photo_get or (lambda url: session.get(url, timeout=60))
    used_photos = {r["photo_id"] for r in rows}
    used_obs = {r["observation_id"] for r in rows if r["observation_id"]}

    def keep(row: dict, body: bytes) -> None:
        path = out / "photos" / row["file"]
        path.write_bytes(body)
        row.update(sha256=sha256_bytes(body), bytes=len(body))
        rows.append(row)
        used_photos.add(str(row["photo_id"]))

    # Near-OOD from iNat.
    for g in INAT_GROUPS:
        got = have.get(g.group, 0)
        page, rng = 1, random.Random(f"{SEED}-{g.group}")
        while got < g.want and page <= 5:
            params = {"taxon_id": g.taxon_id, "photo_license": "cc0,cc-by", "photos": "true",
                      "quality_grade": "research", "place_id": NA, "per_page": 200,
                      "page": page, "order_by": "random"}
            r = api("https://api.inaturalist.org/v1/observations", params=params)
            if r.status_code == 422:                  # random order refused: by id instead
                r = api("https://api.inaturalist.org/v1/observations",
                        params={**params, "order_by": "id"})
            r.raise_for_status()
            results = r.json().get("results") or []
            rng.shuffle(results)
            for o in results:
                if got >= g.want:
                    break
                oid = str(o["id"])
                if oid in ex_obs or oid in used_obs or oid in mycomap_inat_ids:
                    continue
                ph = pick_inat_photo(o, OPEN_DATA_HOST)
                if not ph or str(ph["id"]) in ex_photos or str(ph["id"]) in used_photos:
                    continue
                url = ph["url"].replace("/square.", "/medium.")
                resp = fetch(url)
                if resp.status_code != 200 or not is_jpeg(resp.content):
                    continue
                taxon = o.get("taxon") or {}
                name = taxon.get("name") or ""
                keep({"file": f"inat-{ph['id']}.jpg", "source": "inaturalist-open-data",
                      "split": "", "group": g.group, "category": g.category, "kind": g.kind,
                      "observation_id": oid, "photo_id": str(ph["id"]),
                      "taxon_id": taxon.get("id") or "", "taxon_name": name,
                      "in_v1_labels": "yes" if name in v1_labels or name.split(" ")[0]
                      in v1_labels else "no",
                      "owner": (o.get("user") or {}).get("login") or "",
                      "license": ph["license_code"],
                      "page_url": f"https://www.inaturalist.org/observations/{oid}",
                      "photo_url": url}, resp.content)
                used_obs.add(oid)
                got += 1
            if len(results) < 200:
                break
            page += 1
        log(f"{g.group}: {got}/{g.want}")
        write_manifest(out / "manifest.csv", rows)

    # Everyday from Open Images train.
    if oi_dir is not None:
        labels: dict[str, set[str]] = defaultdict(set)
        with open(oi_dir / "train-annotations-human-imagelabels-boxable.csv", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["Confidence"] == "1":
                    labels[r["ImageID"]].add(r["LabelName"])
        meta = {}
        with open(oi_dir / "train-images-boxable-with-rotation.csv", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["ImageID"] in labels:
                    meta[r["ImageID"]] = r
        for g in OI_GROUPS:
            got = have.get(g.group, 0)
            for iid in oi_pool(labels, meta, g, used_photos, ex_photos):
                if got >= g.want:
                    break
                resp = fetch(OI_URL.format(iid))
                if resp.status_code != 200 or not is_jpeg(resp.content):
                    continue
                m = meta[iid]
                keep({"file": f"oi-{iid}.jpg", "source": "open-images-v5", "split": "train",
                      "group": g.group, "category": g.category, "kind": "everyday",
                      "observation_id": "", "photo_id": iid, "taxon_id": "", "taxon_name": "",
                      "in_v1_labels": "no", "owner": m.get("Author", ""),
                      "license": m.get("License", ""),
                      "page_url": m.get("OriginalLandingURL", ""),
                      "photo_url": OI_URL.format(iid)}, resp.content)
                got += 1
            log(f"{g.group}: {got}/{g.want}")
            write_manifest(out / "manifest.csv", rows)
    sha = write_manifest(out / "manifest.csv", rows)
    return {"out": str(out), "manifest_sha256": sha, **summary(rows)}
