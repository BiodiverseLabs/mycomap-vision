"""How many records the sealed test (v1-test) would hold today, by the day each turned green:
the pace that decides when the test closes (paper close rule: >= 3,000 in-scope finds, or
30 Nov). Counts only; no model output is read or made.

A record counts when it is first green at or after the seal (early_answers.sealed: only
evidence of green before the seal clears it), is not one of the base release's candidate
records, and passes the release builder's own inclusion rules for a test release
(dataset_release.collect, as `build --test-for` applies them: fetched at its source, no
label conflict, a fungal label, not a guest, North America, at least one usable photo, the
review lists given; then separate_from_base: a photo byte-identical to a base photo is
left out, and a record left with none). In scope: its country (the record's, as
paper_score.country_code reads it from the release) is one of the test scope's countries.

A photo downloaded but with no original kept here (photos.status 'done', no 'large' copy:
the server box keeps none) counts as usable; such in-scope records are counted apart as
"originals not fetched", since the cut needs their originals on the machine that cuts it.
`strict_originals` (`--strict-originals`) applies the cut's rule: no original, no photo.

Not applied here, so the counts are an upper bound by them: the cut's dHash pass (a
cross-post twin of a base record is left out as `crosspost_twin`; it reads every
original) and any review list not given.

vision_assisted (records.VISION_ASSISTED_FIELD, from .org's green feed) and first_green_at
are counted per day over every post-seal candidate (included or not), and vision_assisted
also over the in-scope finds. The manifest column defaults to 0 when the feed row lacks the
field, so a 0 cannot be told from "not sent"; first_green_at arrives with it in the same
feed version, so its count says whether the feed's newer fields reach post-seal records at
all. No column: "not populated".

GIF photos (the sealed run decodes only the allowed photo formats, so a GIF has no vector for
any model, and a find whose photos are all GIFs has no Vision answer: a miss) are counted per
day among the in-scope finds' usable photos, by the extension of the photo's source URL or
kept copy (iNat keeps the original's), so the count is by extension, not by the bytes.

How to run it: on the laptop, against a copy of the box's live manifest (the nightly
layer's manifest.sqlite, copied consistently with SQLite's backup, never a plain copy of a
database being written), with the v1 dataset release in data/research-releases/v1:
`mv v1-test-count --manifest <copy>`. The box holds no dataset release. The manifest is
opened read-only and nothing is written anywhere.
"""

from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict

DEFAULT_COUNTRIES = ("US", "CA", "PR")       # the -a2 test scope (Steve, 28 Oct)


def open_manifest(path) -> sqlite3.Connection:
    """The manifest opened read-only: any write fails."""
    from pathlib import Path
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def candidate_keys(conn: sqlite3.Connection, seal: str, base_keys: set[str]) -> tuple[dict, int]:
    """({record key: its green fields} for every record first green at or after `seal` and
    not in the base, how many such records the base already holds)."""
    from .early_answers import sealed
    have = {r[1] for r in conn.execute("pragma table_info(records)")}
    extra = [c for c in ("first_green_at", "vision_assisted") if c in have]
    cur = conn.execute("select observation_id, validated_on, first_seen_at"
                       + "".join(f", {c}" for c in extra) + " from records")
    names = [d[0] for d in cur.description]
    out, in_base = {}, 0
    for row in cur:
        rec = dict(zip(names, row))
        if not sealed(rec["validated_on"], rec["first_seen_at"], rec.get("first_green_at"), seal):
            continue
        if rec["observation_id"] in base_keys:
            in_base += 1
            continue
        out[rec["observation_id"]] = rec
    return out, in_base


def green_day(rec: dict) -> str:
    """The UTC day the record turned green (early_answers.green_moment); "unknown" when
    nothing says."""
    from .early_answers import green_moment
    moment, _ = green_moment(rec["validated_on"], rec["first_seen_at"], rec.get("first_green_at"))
    return moment.date().isoformat() if moment else "unknown"


COUNTS = ("candidates", "in_scope", "outside_scope", "excluded", "originals_not_fetched",
          "in_scope_vision_assisted", "vision_assisted_1", "vision_assisted_0",
          "first_green_at_given", "gif_photos", "all_gif_finds")


def is_gif(photo: dict) -> bool:
    """A release photo whose source URL or kept copy has the .gif extension."""
    url = str(photo.get("source_url") or "").lower().split("?")[0]
    return url.endswith(".gif") or str(photo.get("original_path") or "").lower().endswith(".gif")


def count(conn: sqlite3.Connection, *, seal: str, countries=DEFAULT_COUNTRIES, base=None,
          review_records: dict | None = None, review_photos: dict | None = None,
          strict_originals: bool = False) -> dict:
    """The v1-test count by green day. `base`: the base release (dataset_release.Release, or
    anything with id, record_keys() and photo_shas()); None counts by the seal alone."""
    from . import dataset_release as dr
    from .paper_score import country_code
    have = {r[1] for r in conn.execute("pragma table_info(records)")}
    has_va, has_fg = "vision_assisted" in have, "first_green_at" in have
    base_keys = set(base.record_keys()) if base is not None else set()
    cands, in_base = candidate_keys(conn, seal, base_keys)
    built = dr.collect(conn, review_records=review_records, review_photos=review_photos,
                       only_keys=set(cands), need_original=strict_originals)
    dropped = dr.separate_from_base(base, built) if base is not None else 0
    with_original = {p["record_key"] for p in built.photos
                     if p["included"] and p["original_sha256"]}
    usable: dict = defaultdict(list)                   # record key -> its usable photos
    for ph in built.photos:
        if ph["included"]:
            usable[ph["record_key"]].append(ph)
    keep = set(countries)
    days: dict = defaultdict(lambda: dict.fromkeys(COUNTS, 0))
    reasons: Counter = Counter()
    outside_by_country: Counter = Counter()
    for r in built.records:
        rec = cands[r["record_key"]]
        d = days[green_day(rec)]
        d["candidates"] += 1
        if has_va:
            d["vision_assisted_1" if rec.get("vision_assisted") else "vision_assisted_0"] += 1
        if rec.get("first_green_at"):
            d["first_green_at_given"] += 1
        if not r["included"]:
            d["excluded"] += 1
            reasons[r["reason"]] += 1
            continue
        code = country_code(r["country"])
        if code not in keep:
            d["outside_scope"] += 1
            outside_by_country[code] += 1
            continue
        d["in_scope"] += 1
        if r["record_key"] not in with_original:
            d["originals_not_fetched"] += 1
        if rec.get("vision_assisted"):
            d["in_scope_vision_assisted"] += 1
        gifs = sum(is_gif(ph) for ph in usable[r["record_key"]])
        d["gif_photos"] += gifs
        if gifs and gifs == len(usable[r["record_key"]]):
            d["all_gif_finds"] += 1
    rows, running = [], 0
    for day in sorted(days, key=lambda k: (k == "unknown", k)):
        running += days[day]["in_scope"]
        rows.append({"day": day, **days[day], "in_scope_cumulative": running})
    total = {k: sum(r[k] for r in rows) for k in COUNTS}
    newest = conn.execute("select max(exported_at), max(first_seen_at) from records").fetchone()
    return {
        "seal": seal, "countries": list(countries),
        "base": base.id if base is not None else None,
        "manifest_newest_export": newest[0], "manifest_newest_first_seen": newest[1],
        "records_first_green_after_seal": len(cands) + in_base,
        "already_in_base": in_base, "photos_identical_to_base_dropped": dropped,
        "originals": "required (the cut's rule)" if strict_originals else
                     "not required (a downloaded photo counts; records without one counted apart)",
        "vision_assisted_column": "present" if has_va else "not populated",
        "first_green_at_column": "present" if has_fg else "absent",
        "days": rows, "total": total,
        "outside_scope_by_country": dict(sorted(outside_by_country.items())),
        "excluded_by_reason": dict(reasons.most_common()),
        "not_applied": ["crosspost_twin (the cut's dHash pass)"]
                       + ([] if review_records or review_photos else ["review lists"])
                       + ([] if base is not None else ["the base release (seal rule only)"]),
    }


def format_count(rep: dict) -> str:
    """The count as a text table, one line per green day."""
    va = rep["vision_assisted_column"] == "present"
    lines = [f"v1-test count: records first green at or after {rep['seal']}, "
             f"not in {rep['base'] or '(no base given)'}, by the release builder's rules; "
             f"in scope = {', '.join(rep['countries'])}",
             f"manifest: newest export {rep['manifest_newest_export']}, newest first seen "
             f"{rep['manifest_newest_first_seen']}", f"originals {rep['originals']}", "",
             "                                                  originals | every post-seal candidate:",
             "day         in scope  cumulative  outside  excluded  not fetched | candidates"
             "  vision_assisted 1/0  first_green_at given"]
    for r in rep["days"]:
        v = (f"{r['vision_assisted_1']:,}/{r['vision_assisted_0']:,}" if va
             else "not populated")
        lines.append(f"{r['day']:<11} {r['in_scope']:>8,}  {r['in_scope_cumulative']:>10,}  "
                     f"{r['outside_scope']:>7,}  {r['excluded']:>8,}  "
                     f"{r['originals_not_fetched']:>11,} | {r['candidates']:>10,}  {v:>19}  "
                     f"{r['first_green_at_given']:>20,}")
    t = rep["total"]
    lines += ["", f"total in scope {t['in_scope']:,}; outside the scope {t['outside_scope']:,} "
              f"({', '.join(f'{c} {n:,}' for c, n in rep['outside_scope_by_country'].items()) or 'none'}); "
              f"excluded {t['excluded']:,}",
              "excluded by reason: " + (", ".join(f"{k} {n:,}" for k, n in
                                                  rep["excluded_by_reason"].items()) or "none"),
              f"already in the base: {rep['already_in_base']:,}; photos identical to a base "
              f"photo dropped: {rep['photos_identical_to_base_dropped']:,}",
              f"originals not fetched (counted in scope; the cut needs them): "
              f"{t['originals_not_fetched']:,}",
              f"GIF photos among the in-scope finds (by extension; not decodable at the sealed "
              f"run): {t['gif_photos']:,}; in-scope finds whose photos are all GIFs (no Vision "
              f"answer, scored as misses): {t['all_gif_finds']:,}",
              f"vision_assisted over all {t['candidates']:,} post-seal candidates: "
              + (f"{t['vision_assisted_1']:,} assisted, {t['vision_assisted_0']:,} not (0 is "
                 f"also what a feed row without the field gives); in scope "
                 f"{t['in_scope_vision_assisted']:,} assisted" if va else
                 "not populated (no column in this manifest)")
              + f"; first_green_at given for {t['first_green_at_given']:,} of them "
              f"(column {rep['first_green_at_column']})",
              "not applied (an upper bound by them): " + "; ".join(rep["not_applied"])]
    return "\n".join(lines)
