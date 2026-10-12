"""The public outline of the MycoMap Vision API (v1), as data.

One source for every page that documents the API: the Vision site's /api-docs page and the
"MycoMap Vision API" tab on mycomap.org/api-docs both render GET /api/docs/v1.json, which
`outline()` builds. Each endpoint says whether this server has it turned on, so neither
page promises what isn't there. docs/api.md is the full contract; this is what a developer
needs, in plain words, with nothing internal (no backbone or method names, Steve
2026-10-11).

Blocks are deliberately few, so both renderers stay small:
  {"type": "p", "text"}                     inline `code` and [label](https://…) links
  {"type": "list", "items": [text]}
  {"type": "table", "columns": [..], "rows": [[..]]}
  {"type": "code", "lang", "text"}
  {"type": "endpoint", "method", "path", "status", "summary", "auth",
   "fields": [[name, type, required, notes]], "request", "response", "notes": [text]}
"""

from __future__ import annotations

import json

BASE_URL = "https://vision.mycomap.org/api/v1"
PAGE_URL = "https://vision.mycomap.org/api-docs"
STATUSES = ("available", "planned")

ANSWER_EXAMPLE = {
    "request_id": "r_8sK2…",
    "model": {"name": "MycoMap Vision", "version": "2026-10-06",
              "data": {"snapshot": "2026-10-06", "records": 160173, "species": 18688,
                       "genera": 1620},
              "calibrated_on": "2026-10-08"},
    "photos": 2,
    "context_used": {"latitude": 40.1, "longitude": -83.0, "observed_on": "2026-10-11",
                     "place_from": "entered", "date_from": "photo"},
    "gate": {"verdict": "fungus", "message": None, "left_out": [],
             "photos": [{"verdict": "fungus", "p_fungus": 0.98, "message": None}]},
    "answer": {"rank": "genus", "name": "Amanita", "confidence": 0.91},
    "ranks": {
        "species": [{"name": "Amanita muscaria", "score": 0.81, "confidence": 0.62,
                     "confident": False, "reference_records": 412,
                     "place": {"nearest_record_km": 20, "in_season": True,
                               "far_from_known": False, "note": None}}],
        "genus": [{"name": "Amanita", "confidence": 0.91, "confident": True,
                   "reference_records": 3120}],
        "family": ["…"], "order": ["…"], "class": ["…"], "phylum": ["…"]},
    "likely": {"species": {"coverage": 0.9, "checked_coverage": 0.6, "names": ["…"]}},
    "specimens": [{"name": "Amanita muscaria", "similarity": 0.83, "matched_query_photo": 1,
                   "photo_url": "https://…/medium.jpg", "photo_owner": "…",
                   "photo_withheld": False, "source": "inat", "source_id": "123456789",
                   "record_url": "https://www.inaturalist.org/observations/123456789"}],
    "per_photo": [{"photo": 0, "gate": {"verdict": "fungus", "p_fungus": 0.98},
                   "ranks": {"species": ["…"], "genus": ["…"]},
                   "overall_top": {"species": {"name": "Amanita muscaria", "position": 2,
                                               "confidence": 0.21}},
                   "specimens": ["…"]}],
    "hints": ["Add a photo of the underside."],
    "timing": {"queue_ms": 0, "fetch_ms": None, "decode_ms": 41, "embed_ms": 3180,
               "score_ms": 2010, "model_ms": 5190, "total_ms": 5260,
               "photos_embedded": 2, "photos_reused": 0},
    "usage": {"plan": "free", "used": 37, "limit": 100, "period": "2026-10",
              "resets_at": "2026-11-01T00:00:00Z"},
}


def _json(value) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def outline(enabled: dict[str, bool]) -> dict:
    """The outline, with each endpoint `available` when `enabled` says its part is on
    here, else `planned`. Parts: identify, sessions, by_observation, taxon_counts, usage,
    models."""
    def status(part: str) -> str:
        return "available" if enabled.get(part) else "planned"

    photo_fields = [
        ["photos", "file, 1–10", "yes", "Photos of ONE find, several angles. JPEG preferred; PNG and WebP work."],
        ["latitude, longitude", "number", "no", "Decimal degrees, sent together. Used for scoring only, never stored."],
        ["observed_on", "YYYY-MM-DD", "no", "The day it was found. Read from the photo's EXIF date when not sent."],
        ["top_k", "1–10", "no", "Candidates per rank (default 5)."],
        ["client_ref", "string ≤ 64", "no", "Your id for the find. The same find identified again within a day (or after its session) isn't counted twice."],
    ]
    return {
        "title": "MycoMap Vision API",
        "version": "v1",
        "base_url": BASE_URL,
        "page_url": PAGE_URL,
        "summary": ("Identify fungi from photos with MycoMap Vision, which compares them with "
                    "DNA-verified records from MycoMap's community science network."),
        "sections": [
            {"id": "overview", "title": "Overview", "blocks": [
                {"type": "p", "text": "Send photos of a find; get back the most likely species, "
                 "genus and higher groups, the DNA-verified specimens that look most alike, "
                 "and how confident Vision is. Every answer also says, photo by photo, what "
                 "each photo suggests on its own."},
                {"type": "p", "text": f"Base URL: `{BASE_URL}`. Everything is JSON over HTTPS. "
                 "The API is opening in stages: endpoints marked **available** answer now; "
                 "**planned** ones are agreed and coming."},
                {"type": "list", "items": [
                    "Vision only knows fungi that someone has DNA-verified, mostly in North "
                    "America. A species with few verified records is hard for it to name.",
                    "A suggestion is not an identification. Never use it to decide whether "
                    "something is safe to eat.",
                ]},
            ]},
            {"id": "access", "title": "Getting access", "blocks": [
                {"type": "p", "text": "Use a **Vision token**: create one on mycomap.org "
                 "([API keys](https://mycomap.org/api-docs)) with a short note on what it is "
                 "for; it works once approved. Send it on every call:"},
                {"type": "code", "lang": "http", "text": "Authorization: Bearer vision_…"},
                {"type": "p", "text": "`X-API-Key: vision_…` works too. Keep tokens secret; "
                 "revoke one on mycomap.org at any time (Vision stops accepting it within a "
                 "minute). The MycoMap app uses short-lived tokens of its own; you don't need "
                 "those."},
            ]},
            {"id": "identify", "title": "Identify", "blocks": [
                {"type": "endpoint", "method": "POST", "path": "/identify",
                 "status": status("identify"),
                 "summary": "Identify one find from 1–10 photos.",
                 "auth": "Vision token",
                 "fields": photo_fields,
                 "request": ("curl -X POST " + BASE_URL + "/identify \\\n"
                             "  -H 'Authorization: Bearer vision_…' \\\n"
                             "  -F photos=@cap.jpg -F photos=@underside.jpg \\\n"
                             "  -F latitude=40.12 -F longitude=-82.98 -F observed_on=2026-10-11"),
                 "response": _json(ANSWER_EXAMPLE),
                 "notes": ["The answer is described field by field under **The answer**."]},
                {"type": "endpoint", "method": "POST", "path": "/identify/inat/{observation_id}",
                 "status": status("by_observation"),
                 "summary": "Identify an iNaturalist observation from its photos.",
                 "auth": "Vision token",
                 "fields": [["use_place", "boolean", "no", "Use the observation's public place (default true). An obscured location is used as iNaturalist publishes it."],
                            ["use_date", "boolean", "no", "Use its date (default true)."],
                            ["top_k", "1–10", "no", "Candidates per rank."],
                            ["client_ref", "string ≤ 64", "no", "Defaults to inat:<id>, so the same observation within a day is counted once."]],
                 "request": ("curl -X POST " + BASE_URL + "/identify/inat/123456789 \\\n"
                             "  -H 'Authorization: Bearer YOUR_TOKEN' \\\n"
                             "  -H 'Content-Type: application/json' -d '{\"top_k\": 5}'"),
                 "notes": ["Send JSON or no body. Vision reads up to 10 of the observation's "
                           "photos politely, one observation at a time, so expect a few "
                           "seconds more (`timing.fetch_ms`). Photos are kept in memory only.",
                           "The answer is the usual one plus `observation` (the source, its "
                           "link, photos read, whether a place was used, and the source's "
                           "current name for comparison).",
                           "If the observation is itself one of Vision's DNA-verified "
                           "references, it is left out of the comparison, its place and date "
                           "aren't used, nothing is fetched, and `excluded_reference` says so. "
                           "`left_out_photos` counts the reference photos left out.",
                           "Refusals: 404 `observation_not_found`, 422 `no_usable_photos`, "
                           "502 `source_unavailable` (try again after `Retry-After`)."]},
                {"type": "endpoint", "method": "POST", "path": "/identify/mo/{observation_id}",
                 "status": status("by_observation"),
                 "summary": "Identify a Mushroom Observer observation from its photos.",
                 "auth": "Vision token",
                 "fields": [["use_place", "boolean", "no", "As above: Mushroom Observer's GPS unless hidden, else a small named place's centre."],
                            ["use_date", "boolean", "no", "As above."],
                            ["top_k", "1–10", "no", "Candidates per rank."],
                            ["client_ref", "string ≤ 64", "no", "Defaults to mo:<id>."]],
                 "notes": ["Mushroom Observer asks for one request every 5 seconds, photos "
                           "included, so expect about 5 seconds a photo."]},
            ]},
            {"id": "sessions", "title": "Observation sessions", "blocks": [
                {"type": "p", "text": "For apps that send photos one at a time as they are "
                 "taken. Open a session, add each photo, and every answer is the combined "
                 "answer over all the session's photos, recomputed by Vision. Photos are "
                 "never kept: only what is needed to score them, until the session ends "
                 "(30 minutes after the last call, 2 hours at most)."},
                {"type": "endpoint", "method": "POST", "path": "/sessions",
                 "status": status("sessions"), "summary": "Open a session.",
                 "auth": "Vision token",
                 "fields": [["latitude, longitude", "number", "no", "As for identify."],
                            ["observed_on", "YYYY-MM-DD", "no", "As for identify."],
                            ["client_ref", "string ≤ 64", "no", "Your own id for the find."]],
                 "response": _json({"session_id": "s_7Yp2…", "expires_at": "2026-10-11T15:42:00Z",
                                    "max_photos": 10,
                                    "model": {"name": "MycoMap Vision", "version": "2026-10-06"}})},
                {"type": "endpoint", "method": "POST", "path": "/sessions/{session_id}/photos",
                 "status": status("sessions"),
                 "summary": "Add one photo; get its own answer and the updated combined answer.",
                 "auth": "Vision token",
                 "fields": [["photo", "file", "yes", "One photo."],
                            ["photo_ref", "string ≤ 64", "yes", "Your id for the photo. Sending the same one again returns the current answer without counting again."]],
                 "response": _json({"photo": {"photo_ref": "p1", "index": 0, "ranks": "…"},
                                    "combined": "the answer, as for identify",
                                    "photos_scored": 1, "timing": "…", "usage": "…"})},
                {"type": "endpoint", "method": "PATCH", "path": "/sessions/{session_id}",
                 "status": status("sessions"), "summary": "Set or change the place and date.",
                 "auth": "Vision token",
                 "fields": [["latitude, longitude, observed_on", "", "no", "As above."]]},
                {"type": "endpoint", "method": "GET", "path": "/sessions/{session_id}",
                 "status": status("sessions"), "summary": "The current combined answer.",
                 "auth": "Vision token"},
                {"type": "endpoint", "method": "DELETE",
                 "path": "/sessions/{session_id}/photos/{photo_ref}",
                 "status": status("sessions"), "summary": "Remove a photo; the answer is recomputed.",
                 "auth": "Vision token"},
                {"type": "endpoint", "method": "DELETE", "path": "/sessions/{session_id}",
                 "status": status("sessions"), "summary": "End the session (do this when the find is saved).",
                 "auth": "Vision token"},
            ]},
            {"id": "reference", "title": "Reference data", "blocks": [
                {"type": "endpoint", "method": "GET", "path": "/taxon-counts",
                 "status": status("taxon_counts"),
                 "summary": "How many DNA-verified reference records Vision has for each "
                            "species, genus and family.",
                 "auth": "Vision token",
                 "response": _json({"generated_at": "2026-10-11T12:00:00Z",
                                    "model": {"name": "MycoMap Vision", "version": "2026-10-06"},
                                    "species": {"Amanita muscaria": 412},
                                    "genera": {"Amanita": 3120}, "families": {"Amanitaceae": 3300},
                                    "totals": {"records": 152915}}),
                 "notes": ["Cached for an hour; send `If-None-Match` with the last `ETag`."]},
                {"type": "endpoint", "method": "GET", "path": "/models",
                 "status": status("models"), "summary": "The model answering now and its reference set.",
                 "auth": "Vision token"},
                {"type": "endpoint", "method": "GET", "path": "/usage",
                 "status": status("usage"), "summary": "Your token's use and limit this month.",
                 "auth": "Vision token"},
            ]},
            {"id": "answer", "title": "The answer", "blocks": [
                {"type": "table", "columns": ["Field", "What it is"], "rows": [
                    ["`answer`", "Vision's answer in one line: the most precise rank it is confident at (`{rank, name, confidence}`), or `null` when it isn't confident even at family."],
                    ["`ranks`", "The top candidates at species, genus, family, order, class and phylum, combining all the photos. Order and above have names and scores but no confidence yet."],
                    ["`confidence`", "How likely the name is to be right, from 0 to 1, calibrated weekly on newly verified records. `null` when it can't be stated (see `gate`)."],
                    ["`confident`", "`true` for a rank's top name when its confidence is at least 0.70 and the photos show a fungus. In testing with photos prepared as the app sends them, a confident species answer was right about 92–93% of the time; expect less with field photos."],
                    ["`reference_records`", "How many DNA-verified records Vision has of that name. Under 5, Vision is rarely right at species."],
                    ["`place`", "When you send a place or date: distance to the nearest verified record of that name (rounded to 10 km), whether the date is in its known season, and a note when the find would be far from known records."],
                    ["`likely`", "A short list per rank that held the right name about the stated share of the time in testing."],
                    ["`specimens`", "The DNA-verified records that look most alike, with a photo when its licence or its photographer allows, and a link to the record."],
                    ["`per_photo`", "What each photo says on its own, so you can see which photo pulled the answer which way."],
                    ["`gate`", "Whether each photo shows a fungus (`fungus`, `unsure`, `not_fungus`). A photo that isn't one is left out; when none clearly is, names are given without a confidence."],
                    ["`hints`", "Short suggestions, such as which photo would help most."],
                    ["`model`", "`{name, version, data, calibrated_on}`. `version` names the DNA-verified data the model learned from: the date it was exported from mycomap.org, then .2, .3 for later builds on it. Store it with a suggestion to know which model made it. `data` gives that snapshot's date and size (records, species, genera)."],
                    ["`context_used`", "The place (rounded to 0.1°) and date that were used, and whether each came from you or the photo."],
                    ["`timing`", "How long the call took on the server; see below."],
                    ["`usage`", "Your plan, use and limit this month."],
                ]},
            ]},
            {"id": "timing", "title": "Timing", "blocks": [
                {"type": "p", "text": "Every answer reports its time on the server, in "
                 "milliseconds (also as a `Server-Timing` header)."},
                {"type": "table", "columns": ["Field", "Meaning"], "rows": [
                    ["`queue_ms`", "Waiting for other identifications ahead of yours."],
                    ["`fetch_ms`", "Fetching photos from iNaturalist or Mushroom Observer (by-observation calls only)."],
                    ["`decode_ms`", "Reading and resizing your photos."],
                    ["`embed_ms`", "The model reading your photos (grows with the number of photos)."],
                    ["`score_ms`", "Comparing with the reference set and building the answer."],
                    ["`model_ms`", "`embed_ms + score_ms`."],
                    ["`total_ms`", "From arrival to answer, everything included."],
                ]},
            ]},
            {"id": "errors", "title": "Errors", "blocks": [
                {"type": "p", "text": "A refusal is JSON `{\"code\", \"detail\"}`. When trying "
                 "again later can work, it has a `Retry-After` header and "
                 "`retry_after_seconds`."},
                {"type": "table", "columns": ["Status", "code", "When"], "rows": [
                    ["400", "`bad_request`, `too_many_photos`", "A field is missing or wrong; more than 10 photos."],
                    ["401", "`invalid_token`, `token_expired`", "No token, or one Vision doesn't accept."],
                    ["403", "`key_inactive`", "The token is pending approval, disabled or revoked."],
                    ["404", "`not_found`, `session_gone`, `observation_not_found`", "No such route (or not open yet), session or observation."],
                    ["413 / 415", "`too_large`, `unsupported_image`", "Over 25 MB a file, 80 MB a call or 40 MP; not an image."],
                    ["429", "`rate_limited`, `quota_exceeded`", "Too many photos this minute; this month's identifications used up."],
                    ["503", "`busy`, `updating`, `auth_unavailable`", "Vision is busy or updating; try again after `Retry-After` seconds."],
                ]},
            ]},
            {"id": "limits", "title": "Limits", "blocks": [
                {"type": "list", "items": [
                    "Monthly identifications per member, by membership plan (a call, or a "
                    "session from its first photo, counts once; extra photos don't cost more). "
                    "All of a member's tokens share one allowance.",
                    "Up to 20 photos a minute per member, and 10 photos per identification.",
                    "When many people identify at once, calls wait in a queue; one that "
                    "would wait too long is answered 503 `busy` with an honest `Retry-After`.",
                ]},
            ]},
            {"id": "photos", "title": "Photos that work well", "blocks": [
                {"type": "list", "items": [
                    "The model looks at a small image, so a long side of 640–1024 px is plenty "
                    "(JPEG quality 80–85, about 100–250 KB). Bigger files only take longer.",
                    "Keep the photo's EXIF date, or send `observed_on`. Many resizers drop EXIF.",
                    "Location is optional; strip GPS if you'd rather not send it. Nothing is kept.",
                    "Photograph the cap from above, the underside (gills, pores or teeth) and "
                    "the stem with its base. Several angles of one find beat one photo.",
                ]},
            ]},
            {"id": "rules", "title": "Using the results", "blocks": [
                {"type": "list", "items": [
                    "Call it \"MycoMap Vision\" when you show its suggestions.",
                    "Show specimen photos only from the `photo_url` given, with their "
                    "photographer, and link `record_url`. Don't copy or re-host them.",
                    "Vision never reveals where a reference record was found; don't try to "
                    "work it out from the answers.",
                ]},
            ]},
        ],
    }


def endpoints(doc: dict) -> list[dict]:
    return [b for s in doc["sections"] for b in s["blocks"] if b["type"] == "endpoint"]
