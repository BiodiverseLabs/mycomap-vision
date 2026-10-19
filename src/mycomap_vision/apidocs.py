"""The public outline of the MycoMap Vision API (v1), as data.

One source for every page that documents the API: the Vision site's API page (Research > API) and the
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

from .apiv1 import CONFIDENT_AT, confident_rule
from .settled import SPECIES_RUNNER_UP_MAX, SPECIES_SETTLED_MIN

BASE_URL = "https://vision.mycomap.org/api/v1"
PAGE_URL = "https://vision.mycomap.org/research/api"   # /api-docs leads there too
STATUSES = ("available", "planned")

# The model answering on vision.mycomap.org, as GET /api/v1/models gave it when v1 went live
# (2026-10-14). Every example below shows it, so the outline matches what a caller sees.
LIVE_MODEL = {"name": "MycoMap Vision", "version": "2026-10-13",
              "data": {"snapshot": "2026-10-13", "records": 176655, "species": 18944,
                       "genera": 1543},
              "calibrated_on": "2026-10-13"}
LIVE_REFERENCE = {"records": 169129, "species": 17854, "photos": 684135}
# p(fungus) below this, the gate says "not sure" (re-tuned on dataset release v1; was 0.70).
GATE_NOT_SURE_BELOW = 0.5
SHORT_MODEL = {"name": LIVE_MODEL["name"], "version": LIVE_MODEL["version"]}

ANSWER_EXAMPLE = {
    "request_id": "r_8sK2…",
    "model": LIVE_MODEL,
    "photos": 2,
    "context_used": {"latitude": 40.1, "longitude": -83.0, "observed_on": "2026-10-11",
                     "place_from": "entered", "date_from": "photo"},
    "gate": {"verdict": "fungus", "message": None, "left_out": [],
             "photos": [{"verdict": "fungus", "p_fungus": 0.98, "message": None}]},
    "answer": {"rank": "genus", "name": "Amanita", "confidence": 0.91},
    "confidence_withheld": False,
    "species_url": "https://mycomap.org/species/Amanita%20muscaria",
    "ranks": {
        "species": [{"name": "Amanita muscaria", "score": 0.81, "confidence": 0.62,
                     "confident": False, "reference_records": 412,
                     "place": {"nearest_record_km": 20, "in_season": True,
                               "far_from_known": False, "note": None}}],
        "genus": [{"name": "Amanita", "confidence": 0.91, "confident": True,
                   "reference_records": 3120,
                   "place": {"nearest_record_km": 10, "in_season": True,
                             "far_from_known": False, "note": None}}],
        "family": ["…"], "order": ["…"], "class": ["…"], "phylum": ["…"]},
    "prior": {"applied": True},
    "photo_only_top": {},
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
        ["latitude, longitude", "number", "no", "Decimal degrees, sent together. Used to weigh the names and for each name's place cue, never stored."],
        ["observed_on", "YYYY-MM-DD", "no", "The day it was found, used like the place. Read from the photo's EXIF date when not sent."],
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
                 "each photo suggests on its own. Send the find's place and date too: Vision "
                 "uses them to weigh the names, and says how near and how in season each "
                 "name's DNA-verified records are."},
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
                {"type": "p", "text": "Use a **Vision token**: request one on mycomap.org "
                 "([Vision API keys](https://mycomap.org/vision-keys)) with a short note on what "
                 "it is for; it works once a mycomap.org admin approves it. Send it on every "
                 "call:"},
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
                           "So is every other record of the same find: the one it links to "
                           "under the other source, and its Mushroom Observer <-> iNaturalist "
                           "cross-posts (the same picture), so a cross-post never matches "
                           "itself. `left_out_photos` counts the reference photos left out.",
                           "Answers are kept: the same observation with the same options "
                           "comes back at once, without reading it again (`\"cached\": true`), "
                           "until `model.answers_version` moves (a new release or reference "
                           "set). GET /models gives the current `answers_version`.",
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
                           "included, so expect about 5 seconds a photo the first time; a kept "
                           "answer comes back at once, as for iNaturalist."]},
                {"type": "endpoint", "method": "GET", "path": "/answers/{inat|mo}/{observation_id}",
                 "status": status("by_observation"),
                 "summary": "The answer Vision already holds for an observation, if any.",
                 "auth": "Vision token",
                 "fields": [["use_place", "boolean", "no", "As for identify (default true)."],
                            ["use_date", "boolean", "no", "As for identify (default true)."],
                            ["top_k", "1–10", "no", "As for identify (default 5)."]],
                 "request": ("curl " + BASE_URL + "/answers/inat/123456789 \\\n"
                             "  -H 'Authorization: Bearer YOUR_TOKEN'"),
                 "notes": ["Returns a kept answer under the current `model.answers_version`, "
                           "from an earlier identify by observation or from someone's run on "
                           "the Identify page, in the same shape (`\"cached\": true`; "
                           "`kept_from` is `api` or `site`). Nothing is read or scored, and "
                           "nothing is counted.",
                           "404 `not_answered` when Vision holds no such answer."]},
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
                                    "model": SHORT_MODEL})},
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
                                    "model": SHORT_MODEL,
                                    "species": {"Amanita muscaria": 412},
                                    "genera": {"Amanita": 3120}, "families": {"Amanitaceae": 3300},
                                    "totals": {"records": LIVE_REFERENCE["records"]}}),
                 "notes": ["Cached for an hour; send `If-None-Match` with the last `ETag`."]},
                {"type": "endpoint", "method": "GET", "path": "/models",
                 "status": status("models"), "summary": "The model answering now and its reference set.",
                 "auth": "Vision token",
                 "response": _json({"models": [{**LIVE_MODEL, "default": True,
                                                "confident_at": CONFIDENT_AT,
                                                "confident_rule": confident_rule(),
                                                "reference": LIVE_REFERENCE}],
                                    "limits": {"max_photos": 10, "photos_per_minute": 20}}),
                 "notes": ["Also its data snapshot, calibration date, the ranks answered, "
                           "request limits and which parts of the API are on.",
                           "`answers_version` names what Vision answers from now; identify by "
                           "observation keeps its answers until it moves.",
                           "There is one model; leave `model` out of your requests."]},
                {"type": "endpoint", "method": "GET", "path": "/usage",
                 "status": status("usage"), "summary": "Your token's use and limit this month.",
                 "auth": "Vision token",
                 "response": _json({"usage": {"plan": "member", "used": 12, "limit": 300,
                                              "period": "2026-10",
                                              "resets_at": "2026-11-01T00:00:00Z"},
                                    "quotas": "enforce", "pool": "member",
                                    "limit_source": "plan"}),
                 "notes": ["Reading it is never counted. `pool` is `member` when the token "
                           "shares its owner's month, `key` when it has its own limit."]},
            ]},
            {"id": "answer", "title": "The answer", "blocks": [
                {"type": "table", "columns": ["Field", "What it is"], "rows": [
                    ["`answer`", "Vision's answer in one line: the most precise rank it is confident at (`{rank, name, confidence}`), or `null` when it isn't confident even at family."],
                    ["`ranks`", "The top candidates at species, genus, family, order, class and phylum, combining all the photos. Order and above have names and scores but no confidence yet."],
                    ["`confidence`", "How likely the name is to be right, from 0 to 1. Calibrated on the validation split of MycoMap's frozen dataset release v1 (DNA-verified records the model did not learn from), and fitted again when a new model or release is built; `model.calibrated_on` gives the date. `null` when it can't be stated (see `gate`)."],
                    ["`confident`", f"`true` for a rank's top name when the photos show a fungus and the name is settled: at genus and family, confidence at least {CONFIDENT_AT:.2f}; at species, at least {CONFIDENT_AT:.2f}, or at least {SPECIES_SETTLED_MIN:.2f} with the next species below {SPECIES_RUNNER_UP_MAX:.2f} (Steve, 2026-10-18: one rule with the app and .org collection importance; species needed {CONFIDENT_AT:.2f} before). `GET /models` states the rule as `confident_rule`. Confident answers are calibrated on v1's validation split, whose photos are DNA-verified records' own; expect fewer of them with field photos."],
                    ["`reference_records`", "How many DNA-verified records Vision has of that name. Under 5, Vision is rarely right at species."],
                    ["`place`", "On species and genus names, when you send a place or date: distance to the nearest verified record of that name (rounded to 10 km), whether the date is in its known season, and a `note` when the find would be far from known records or out of season."],
                    ["`confused_with`", "On a species name that people often get wrong: what DNA said instead on mycomap.org's DNA-checked iNaturalist finds called that name (up to 3 names with their number of finds), how many finds were called it and how many DNA said were something else, and a one-line `note` to show under the name. Absent otherwise. Rebuilt nightly from [mycomap.org/misid](https://mycomap.org/misid)."],
                    ["`prior`", "`{applied}`: `true` when the place or date you sent moved the scores."],
                    ["`photo_only_top`", "The ranks where place and date changed the top name, with the name the photos alone would have put first (`{}` when they changed none)."],
                    ["`likely`", "A short list per rank that held the right name about the stated share of the time in testing."],
                    ["`specimens`", "The DNA-verified records that look most alike, with a photo when its licence or its photographer allows, and a link to the record."],
                    ["`per_photo`", "What each photo says on its own, so you can see which photo pulled the answer which way."],
                    ["`species_url`", "The top species' page on mycomap.org, or `null`."],
                    ["`confidence_withheld`", "`true` when no photo clearly shows a fungus, so every `confidence` is `null` and nothing is `confident`."],
                    ["`gate`", f"Whether each photo shows a fungus (`fungus`, `unsure`, `not_fungus`), with `p_fungus`; below {GATE_NOT_SURE_BELOW:.2f} a photo is at best `unsure` (the gate was re-tuned on dataset release v1). A photo that isn't one is left out; when none clearly is, names are given without a confidence."],
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
