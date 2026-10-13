"""iNaturalist's computer vision on a dataset release's finds (the paper's E14).

Every included photo of each find is sent to iNat's POST /v1/computervision/score_image
(inat_cv.InatClient: a token, cached, 1 request a second, a 401 stops the run), at 1024 px
on the long side at most: our kept originals (iNat 'large' is 1024 px, MO 960 px) as they
are, any larger one cut down. Photos only, with the record's PUBLIC point (obscured where
iNat obscures; an MO box centre) and its observed date: never a true coordinate, never the
observation id, so iNat sees no community ID. Each photo keeps `vision_score` (photo only)
and `combined_score` (with iNat's place model); a find's answer per rank is the best score
on any photo, matched to the truth by iNat taxon id; a find with no suggestion is wrong.

On a sealed test release nothing is sent before the analysis freeze tag; the fallback
sample (`draw_fallback`) is drawn and hashed before that, from the id list alone.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import config
from . import expert_ids as E
from .inat_cv import best_per_rank, parse_aggregated

MAX_SIDE = 1024
TOP = 10
RANKS = ("species", "genus", "family")
FORMAT = 1


def work_dir(name: str) -> Path:
    if not E.re.fullmatch(r"[a-z0-9][a-z0-9._-]*", name):
        raise ValueError(f"set name {name!r}: lower case letters, digits, . _ -")
    return config.DATA_DIR / "paper-comparisons" / "e14" / name


def at_most_1024(body: bytes) -> tuple[bytes, int, int]:
    """The photo as sent: unchanged if its long side is <= 1024 px, else cut to 1024 (EXIF
    orientation applied, JPEG quality 90). Returns (bytes, width, height)."""
    from PIL import Image, ImageOps
    img = Image.open(io.BytesIO(body))
    w, h = img.size
    if max(w, h) <= MAX_SIDE:
        return body, w, h
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue(), img.size[0], img.size[1]


def public_point(place: dict | None) -> tuple[float | None, float | None]:
    """A release's frozen public point (micro-degrees) -> (lat, lng); None without one."""
    if not place or place.get("public_lat_micro") is None or place.get("public_lon_micro") is None:
        return None, None
    return place["public_lat_micro"] / 1e6, place["public_lon_micro"] / 1e6


def real_day(observed_on: str | None) -> str | None:
    from .dates import real_date
    d = real_date(observed_on)
    return d[:10] if d else None


def find_answer(photo_answers: list[dict], score: str, top: int = TOP) -> dict:
    """Per rank, the find's taxa ranked by their best `score` on any photo (top `top`), as
    [taxon_id, name, score]: names let species s.l./complex be scored by name too."""
    best = best_per_rank(photo_answers, score) if photo_answers else {r: [] for r in RANKS}
    names = {rank: {tid: t["name"] for a in photo_answers for tid, t in a.get(rank, {}).items()}
             for rank in RANKS}
    return {rank: [[tid, names[rank].get(tid), round(s, 4)] for tid, s in best.get(rank, [])[:top]]
            for rank in RANKS}


def hits(ranked: list, want: int | None) -> dict:
    """top-1/3/5/10 hits of a ranked [[taxon_id, score], ...] list; no truth or no
    suggestion is a miss."""
    ids = [r[0] for r in ranked]
    return {f"top{k}": want is not None and want in ids[:k] for k in (1, 3, 5, 10)}


def draw_fallback(ids: list[str], n: int, seed: int) -> list[str]:
    """The pre-declared fallback: a seeded random n of the find ids, drawn before the
    unseal from the id list alone (no label, photo or score is read)."""
    return E.draw_sample(ids, n, seed)


PLACE_RULE = ("the release's frozen public point (private.sqlite record_public_places): iNat's "
              "public coordinates, obscured where iNat obscures; an MO box centre; none sent "
              "where none is frozen")


def run(rel, name: str, record_keys: list[str], client, read_photo: Callable[[str], bytes],
        truths: dict[str, E.TruthTaxa], sealed_ok: bool = False, log=print,
        model: dict | None = None) -> dict:
    """Score `record_keys` of `rel`; writes one JSON line per find to finds.jsonl. Resumable:
    every photo's answer is cached by the client. `read_photo(photo_key)` -> original bytes."""
    if rel.manifest.get("test_for") and not (sealed_ok or E.freeze_tagged()):
        raise RuntimeError(f"{rel.id} is a sealed test release: iNat CV waits for the tag "
                           f"{E.FREEZE_TAG} in the research repo")
    places = rel.public_places()
    recs = {r["record_key"]: dict(r) for r in rel.db.execute(
        "select * from records where included = 1")}
    d = work_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    out = d / "finds.jsonl"
    stats = Counter()
    first_scored = last_scored = None
    import time
    start, calls0 = time.monotonic(), client.calls
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for n, key in enumerate(record_keys, 1):
            rec = recs.get(key)
            if rec is None:
                raise RuntimeError(f"{key} is not an included record of {rel.id}")
            lat, lng = public_point(places.get(key))
            day = real_day(rec.get("observed_on"))
            photos = rel.db.execute("select photo_key, manifest_photo_id from photos where "
                                    "record_key = ? and included = 1 order by position, photo_key",
                                    (key,)).fetchall()
            answers, sent = [], []
            for pk, _ in photos:
                def body(pk=pk):
                    b, w, h = at_most_1024(read_photo(pk))
                    sent.append({"photo_key": pk, "sha256": hashlib.sha256(b).hexdigest(),
                                 "width": w, "height": h})
                    return b
                raw = client.score_image(pk, body, lat, lng, day)
                answers.append(parse_aggregated(raw))
                stamp = raw.get("_scored_at")
                if stamp:
                    first_scored = min(first_scored or stamp, stamp)
                    last_scored = max(last_scored or stamp, stamp)
                stats["photos"] += 1
            t = truths[key]
            line = {"format": FORMAT, "record_key": key, "photos": len(photos),
                    "photo_ids": [mid for _, mid in photos], "px": MAX_SIDE,
                    "place_sent": lat is not None, "date_sent": day is not None,
                    "truth": {"species": t.species, "genus": t.genus,
                              "formal_species": t.formal_species,
                              "has_genus": bool(rec.get("genus"))},
                    "vision": find_answer(answers, "vision"),
                    "combined": find_answer(answers, "combined"),
                    "first_photo": {"vision": find_answer(answers[:1], "vision"),
                                    "combined": find_answer(answers[:1], "combined")},
                    "sent_now": sent}
            for score in ("vision", "combined"):
                line[f"{score}_hits"] = {
                    "species": hits(line[score]["species"], t.species) if t.formal_species else None,
                    "genus": hits(line[score]["genus"], t.genus) if rec.get("genus") else None}
            line["answered"] = {s: any(line[s][r] for r in RANKS) for s in ("vision", "combined")}
            stats["no_suggestion"] += not line["answered"]["vision"]
            stats["finds"] += 1
            f.write(json.dumps(line, sort_keys=True) + "\n")
            if n % 25 == 0:
                log(f"  {n:,}/{len(record_keys):,} finds, {client.calls:,} iNat calls")
    meta = {"set": name, **rel.cite(), "code_version": config.code_version(),
            "finds": stats["finds"], "photos": stats["photos"],
            "finds_with_no_suggestion": stats["no_suggestion"],
            "ids_sha256": E.ids_sha256(record_keys), "finds_sha256": E.sha256_file(out),
            "inat_cv_model": model or {"version": None, "note": "not recorded"},
            "place_rule": PLACE_RULE, "px": MAX_SIDE,
            "first_scored_at": first_scored, "last_scored_at": last_scored,
            "requests_this_run": client.calls - calls0,
            "wall_hours_this_run": round((time.monotonic() - start) / 3600, 3),
            "written_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    (d / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def summarise(lines: list[dict]) -> dict:
    """Top-1/3/5/10 per score: species over formal truths, genus over finds with a genus."""
    out = {}
    for score in ("vision", "combined"):
        for rank in ("species", "genus"):
            # a truth iNat has no taxon for stays in, as a miss (iNat can't be right)
            rows = [l[f"{score}_hits"][rank] for l in lines if l[f"{score}_hits"][rank] is not None]
            n = len(rows)
            out[f"{score}_{rank}"] = {"n": n, **{k: round(sum(r[k] for r in rows) / n, 4) if n else None
                                                for k in ("top1", "top3", "top5", "top10")}}
    return out
