# MycoMap Vision public API (v1)

Status, 2026-10-13: the contract mycomap.org, the MycoMap app and Vision build against.
Steve decided its open items on 2026-10-11 (§10). Live on vision.mycomap.org, each behind its
own setting (§9): identify by photos and by iNat / MO observation number, observation
sessions, `GET /models`, `GET /usage`, taxon counts, the higher ranks, the queue, the fast
int8 search, and quotas (enforced). A developer-facing outline of the same API is served at
`/api/docs/v1.json` and shown at vision.mycomap.org/research/api (`/api-docs` leads there)
and on mycomap.org/api-docs.

Everything is under `https://vision.mycomap.org/api/v1/`. The website's routes
(`/api/identify`, `/api/models`, …) stay as they are. They gain only the `timing` block.

## 1. Who calls, and how they authenticate

| Caller | Credential | How Vision checks it |
|---|---|---|
| External developer | a **Vision service token** `vision_…` from the .org key manager: `Authorization: Bearer vision_…` (or `X-API-Key: vision_…`) | .org's existing `POST /api/service-keys/introspect` (built for Atlas), service `vision` added; cached on Vision |
| The mobile app, for a signed-in member | a **Vision app token** `mvt.<body>.<sig>`, minted by .org for 15 min (§6.3): `Authorization: Bearer mvt.…` | verified on Vision with the sign-in bridge's Ed25519 public key it already holds. No call to .org |
| mycomap.org's server (e.g. its collection-importance score) | `VISION_API_KEY`, the server-to-server key the two already share | constant-time compare |

The cookie sign-in (`MV_SIGNIN`) doesn't apply under `/api/v1/`; a v1 route is answered
only with one of these credentials, else 401.

The app calls Vision **directly**, not through .org: .org's box is memory-tight, and
proxying would double every photo's transfer and hold Node connections through queue
waits of 5–40 s. Vision can check the token itself, so the proxy would gain nothing.

## 2. Endpoints

| Method + path | What it does |
|---|---|
| `POST /api/v1/identify` | identify from 1–10 uploaded photos in one call (developers; the app's offline-saved records) |
| `POST /api/v1/identify/inat/{observation_id}` | identify from an iNaturalist observation's photos |
| `POST /api/v1/identify/mo/{observation_id}` | identify from a Mushroom Observer observation's photos |
| `GET /api/v1/answers/{inat\|mo}/{observation_id}` | the answer Vision already holds for an observation (§3.2); reads, scores and counts nothing |
| `POST /api/v1/sessions` | open an observation session (the app, while photos are taken) |
| `POST /api/v1/sessions/{id}/photos` | add one photo → that photo's result + the updated combined result |
| `PATCH /api/v1/sessions/{id}` | set or change place/date (GPS often fixes after the first photo) |
| `GET /api/v1/sessions/{id}` | the current combined result |
| `DELETE /api/v1/sessions/{id}/photos/{photo_ref}` | the user removed a photo; re-scored |
| `DELETE /api/v1/sessions/{id}` | end the session (on save or discard); everything deleted |
| `GET /api/v1/models` | the served model, its reference set and calibration date |
| `GET /api/v1/taxon-counts` | DNA-validated reference records per species, genus, family (§7) |
| `GET /api/v1/usage` | this credential's quota and use this period |

Identify-by-observation is a `POST` because every call costs quota and must never be
answered from a proxy's cache.

## 3. Requests

### 3.1 Photos

`POST /api/v1/identify`, `multipart/form-data`:

| Field | Type | Notes |
|---|---|---|
| `photos` | file, 1–10 | the same find, several angles. JPEG preferred; PNG and WebP accepted |
| `latitude`, `longitude` | number, optional | decimal degrees; scoring only, never stored |
| `observed_on` | `YYYY-MM-DD`, optional | the day it was found |
| `model` | string, optional | one of `GET /api/v1/models`; default the served default. One model per request |
| `top_k` | int 1–10, optional | candidates per rank, default 5 |
| `client_ref` | string 1–64, optional | the caller's id for the find: the same find counted in the last 24 h (by its session or an earlier call) isn't counted again (§6.4) |

**Input guidance** (also on the developer page):

- **Size.** The model sees 224 × 224 px. Vision keeps each photo at 448 px on its
  shorter side and shrinks anything larger. **A long side of 640–1024 px is plenty**,
  JPEG quality 80–85, roughly 100–250 KB. Bigger files cost upload and decode time and
  change nothing: vectors from a 448-px copy match the full-size photo's at
  cosine ≥ 0.9997. Hard limits: 25 MB a file, 80 MB a request, 40 MP.
- **Date.** Keep EXIF `DateTimeOriginal`, or send `observed_on`. Many resizers drop EXIF;
  send `observed_on` then.
- **Place.** Optional: fields or EXIF GPS, used for scoring only, never kept. Strip GPS
  if you'd rather not send it.
- **What to photograph.** Cap from above, the underside (gills/pores/teeth), the stem
  with its base. Several angles of one find beat one photo.
- Entered values win over EXIF. A 1970-01-01 date counts as none.

### 3.2 By observation number

`POST /api/v1/identify/inat/123456789` or `/mo/567890`, optional JSON
`{"model", "top_k", "use_place": true, "use_date": true, "client_ref"}`.

- Vision reads the observation and up to 10 of its photos (iNat's in position order,
  MO's primary image first) at the size the reference was embedded from (iNat large,
  1024 px; MO 960 px), keeps them in memory only, and stays within those sites' limits:
  iNat's API one request a second, MO 5 s between every request, photos included (its
  20 a minute), and at least MO's reported run time. One observation at a time across
  all callers, at most 3 waiting behind it (more: 503 `busy`), so expect seconds in
  `timing.fetch_ms` (MO: about 5 s a photo). The caller's photo window and the queue
  are checked once the observation is read and before any photo is: a refused caller
  costs those sites no photo reads.
- Place and date are the observation's **public** ones: iNat's published location
  (obscured as iNat obscures it), MO's GPS unless hidden, else the centre of MO's named
  place when its box is at most 0.5° across. Never a true location Vision may hold.
  `use_place`/`use_date: false` leave them out.
- **Its own record is left out.** When the observation is one of Vision's DNA-validated
  references, it is removed from the reference set for this answer (its photos from
  the nearest matches and from its species' average, exactly as a reference built
  without it), and `excluded_reference: {"source", "source_id"}` says so. Otherwise it
  would match itself. Vision uses that record's stored vectors and fetches nothing
  (`timing.photos_reused`, `fetch_ms` 0). Its place and date aren't used either: the
  record is part of the range and season data. The same applies to any reference photo
  that is one of the observation's own (one photo filed under two observations).
  `left_out_photos` counts the reference photos left out.
- The answer adds `observation`: `{"source", "id", "url", "photos"` (at the source),
  `"photos_used", "place"` (`coordinates`, `obscured`, `area` or null), `"name_at_source",
  "rank_at_source", "photos_shown"}` (the source's current name, for comparison; null for
  a reused reference record, which isn't read). `photos_shown` has one entry per scored
  photo, in `per_photo`'s order: the photo's URL when it is Creative Commons or public
  domain, else null (null as a whole for a reused reference record).
- **Answers are kept.** Each answer is saved on the box per observation, model and
  options (`top_k`, `use_place`, `use_date`) and given again, without reading the
  observation or scoring it, while `model.answers_version` is unchanged: an opaque id
  of what Vision answers from (the release and its reference set as served), which
  moves with a new release, a night's reference changes, a recalibration or a
  photographer's answer. A kept answer says `"cached": true` and has a new
  `request_id` and `timing`; a fresh one says `"cached": false`. A photo added at the
  source after the answer is not seen until the version moves. GET /api/v1/models
  gives the current `answers_version`, so a caller that keeps answers itself can tell
  they still hold.
- Counting: one identification, behind interactive calls in the queue, kept answers
  included. `client_ref` defaults to `inat:<id>` / `mo:<id>`, so the same observation
  for the same pool within a day is counted once (§6.4).
- Refusals: 404 `observation_not_found` (and `not_found` for a source other than `inat`
  or `mo`), 422 `no_usable_photos`, 502 `source_unavailable` (with `Retry-After`), 503
  `busy`.
- **Reading a kept answer.** `GET /api/v1/answers/inat/{id}` or `/mo/{id}`, optional
  query `model`, `top_k`, `use_place`, `use_date` (defaults as above), gives the answer
  Vision holds under the current `answers_version` without reading, scoring or counting
  anything: one kept by identify-by-observation (`"kept_from": "api"`), else one someone
  got on the site's Identify page from a link (`"kept_from": "site"`; same model, place
  and date options, put in this shape: `observation.place` is null, and order, class and
  phylum aren't listed). `"cached": true`. 404 `not_answered` when there is none. .org's
  Validate Records uses it so an answer got on the site shows on the record at once.

### 3.3 Sessions (the app)

Why sessions rather than stateless calls:

- **Merging per-photo answers in the app is wrong.** Each photo returns only its top 5
  per rank. nearest+mean averages scores over every species, so the combined #1 can sit
  at #6 on every single photo. The temperature, the likely sets and genus = max over its
  species also live on Vision. An app merge would copy that scoring and drift with every
  method change.
- **Re-sending every photo each time** is exact but re-embeds: 10 photos = 55 embeds
  instead of 10, on a box with no CPU to spare, and 5× the cellular upload.
- **A session**: one embed per photo, an exact combined answer, small uploads. It costs
  a few KB of server memory per session, lost on restart, which the app handles.
- `POST /api/v1/identify` stays the path for a record saved offline: all its photos go
  once when the phone is back online.

`POST /api/v1/sessions`, JSON, all optional:

```json
{"latitude": 40.12, "longitude": -82.98, "observed_on": "2026-10-11",
 "client_ref": "obs-local-8812", "model": null, "top_k": 5, "keep_photos": false}
```

→ `201 {"session_id": "s_7Yp2…", "expires_at": "…", "max_photos": 10,
"model": {"name", "version", "data", "calibrated_on"}, "context_used": {…}}`

`POST /api/v1/sessions/{id}/photos`, multipart: `photo` (one file; JPEG, long side
1024, q≈85, EXIF date kept) and `photo_ref` (the app's id for it, ≤ 64 chars):

```json
{"photo": {"photo_ref": "p1", "index": 2, "gate": {…}, "ranks": {…},
           "overall_top": {…}, "specimens": [ … ]},
 "combined": { …the result of §4… },
 "photos_scored": 3,
 "timing": { … },
 "usage": { … },
 "session": {"session_id": "s_7Yp2…", "expires_at": "…"}}
```

- **Idempotent on `photo_ref`.** A resend after a dropped response gets the stored
  answer: no re-embed, no extra count. The same `photo_ref` with a different file is
  still answered from the first (the app must not reuse refs).
- `PATCH` → `{"context_used", "combined"}`; `GET` → `{"combined", "context_used",
  "expires_at"}` (`combined.per_photo` holds every photo); `DELETE …/photos/{photo_ref}`
  → `{"combined"}`; `DELETE` session → 204.
- **What is kept.** Per photo: its vector for the served model, the gate's base vector and
  verdict with `p_fungus` (so a gate threshold change can re-judge it), and its
  per-photo answer, all computed once on upload. The photo itself is decoded, embedded
  and dropped in that call; **no photo bytes are kept**. Re-scoring after a new photo,
  a removed one or a place/date change runs `identify_vectors` over the stored vectors
  and never re-embeds. Place and date move the scores only with the
  `nearest+mean+place` method (the place/date prior), the served default whenever a place
  or a date is given (`prior.applied` true). Under a method without the prior they still
  add the per-name `place` cue and are echoed, so a `PATCH` returns a re-scored answer
  either way. A session that never sends a place or a date gets photos-only scores
  (`prior.applied` false): send them at `POST` or by `PATCH` as soon as they are known.
- **Lifetime.** 30 min after the last call, 2 h at most from creation (Steve). A
  session gone to expiry or a restart answers 404 `session_gone`. The app opens a new
  one and resends the photos it keeps locally anyway. Sessions belong to the credential's
  subject; anyone else gets 404.
- `keep_photos: true` is accepted and recorded but changes nothing in v1: **nothing is
  kept** (Steve). Keeping consented photos would need storage and a consent text, and is
  a later decision.
- Memory: ~70 KB a photo (its vectors, its species score row so a new photo never
  re-searches the others, its best 2,000 records, its answer; `docs/capacity.md`). Hard
  cap `MV_SESSIONS_MAX_MB=64`: past it the oldest idle session is dropped, and if all
  were active in the last 2 minutes a new session gets 503 `busy`.

### 3.4 Model and usage

`GET /api/v1/models` (any credential; `Cache-Control: private, max-age=300`): the one
model answering, by its public name only. Requests leave `model` out.

```json
{"models": [{"name": "MycoMap Vision", "version": "2026-10-13",
             "data": {"snapshot": "2026-10-13", "records": 176655, "species": 18944,
                      "genera": 1543},
             "calibrated_on": "2026-10-13", "default": true,
             "reference": {"records": 169129, "species": 17854, "photos": 684135},
             "ranks": ["species", "genus", "family", "order", "class", "phylum"],
             "confident_at": 0.7,
             "likely_coverage": {"species": 0.8, "genus": 0.9, "family": 0.91}}],
 "limits": {"max_photos": 10, "max_file_mb": 25, "max_request_mb": 80, "top_k_max": 10,
            "photos_per_minute": 20},
 "features": {"identify": true, "sessions": true, "by_observation": false,
              "models": true, "usage": true, "taxon_counts": true}}
```

`likely_coverage` (added 2026-10-18) is how often each rank's likely set held the right name
on validation records held back from its fit: what the sets achieved, not their target
(§4, `likely`); `null` for a rank with no fitted set.

`GET /api/v1/usage` (any credential; `no-store`): this credential's use this month. Reading
it is never counted.

```json
{"usage": {"plan": "member", "used": 12, "limit": 300, "period": "2026-10",
           "resets_at": "2026-11-01T00:00:00Z"},
 "quotas": "enforce", "pool": "member", "limit_source": "plan"}
```

`usage` is the block every answer carries (§4.1), `null` when quotas are off or for
mycomap.org's server key (`pool: "none"`). `pool` says whose month it is: `member` (the
member's app and their tokens without an override share one) or `key` (a token with its own
override counts alone). `limit_source` is `plan`, `override` (per token, or per member set
on mycomap.org) or `staff`. Counts arrive at mycomap.org's member page within about 15
minutes; this endpoint is current.

## 4. The answer

Every identify call, and `combined` in a session, is today's per-model result with
`request_id`, `gate`, `confident`, `timing` and `usage` added. A v1 call returns one
model's result, not a `results` list.

```json
{
  "request_id": "r_01J…",
  "model": {"name": "MycoMap Vision", "version": "2026-10-13",
            "data": {"snapshot": "2026-10-13", "records": 176655, "species": 18944,
                     "genera": 1543},
            "calibrated_on": "2026-10-13"},
  "reference": {"records": 169129, "species": 17854, "photos": 684135},
  "photos": 3,
  "context_used": {"latitude": 40.1, "longitude": -83.0, "observed_on": "2026-10-11",
                   "place_from": "entered", "date_from": "photo"},

  "gate": {"verdict": "fungus", "message": null, "left_out": [],
           "photos": [{"verdict": "fungus", "p_fungus": 0.98, "message": null}]},

  "ranks": {
    "species": [{"name": "Amanita muscaria", "score": 0.81, "confidence": 0.62,
                 "confident": true, "reference_records": 412}],
    "genus":   [ … ], "family": [ … ],
    "order":   [ … ], "class": [ … ], "phylum": [ … ]
  },
  "likely": {
    "species": {"coverage": 0.8, "checked_coverage": 0.8, "capped": false,
                "names": [{"name": "Amanita muscaria", "confidence": 0.62,
                           "reference_records": 412}]},
    "genus": { … }, "family": { … }
  },
  "specimens": [{"observation_id": "…", "name": "…", "species": "…", "genus": "…",
                 "family": "…", "similarity": 0.83, "matched_query_photo": 1,
                 "photo_url": "…", "photo_owner": "…", "photo_withheld": false,
                 "source": "inat", "source_id": "…", "record_url": "…",
                 "species_url": "https://mycomap.org/species/…"}],
  "per_photo": [
    {"photo": 0, "photo_ref": "p1",
     "gate": {"verdict": "fungus", "p_fungus": 0.98, "message": null},
     "ranks": {"species": [ … ], "genus": [ … ], "family": [ … ], "order": [ … ],
               "class": [ … ], "phylum": [ … ]},
     "overall_top": {"species": {"name": "Amanita muscaria", "position": 2,
                                 "confidence": 0.21}, "genus": {…}, "…": {}},
     "specimens": [ … ]}
  ],
  "hints": ["…"],
  "confidence_note": "Calibrated on … test records …",
  "timing": { … §4.2 … },
  "usage": {"plan": "member", "used": 37, "limit": 200, "period": "2026-10",
            "resets_at": "2026-11-01T00:00:00Z"}
}
```

- **`model`** (Steve, 2026-10-11: the product never names the backbone underneath):
  `name` is always "MycoMap Vision". `version` reflects the **training data** (Steve,
  2026-10-11): the date the DNA-validated label set behind the served model was
  exported from mycomap.org (`YYYY-MM-DD`), then `.2`, `.3` for a later release built on
  the same export; opaque, matches `[A-Za-z0-9._:-]{1,40}`, never a backbone id. Store it
  on records to say which model made a suggestion. `data` = `{snapshot, records, species,
  genera}`: that export's date and size as Vision uses it (North America, no conflicting
  label, spellings of a name counted once), from the release itself, so it doesn't move
  with the nightly additions (`reference` does). `calibrated_on` is the date its
  confidences were last calibrated. Backbone and method ids never appear in any v1
  answer or refusal (tested), and `prior` carries only `applied`. Today's served model
  (live since 2026-10-14 00:19 UTC): the v1 model, trained 2026-10-13 on dataset release
  v1's `train` split; `version` 2026-10-13, `data` 176,655 records, 18,944 species, 1,543
  genera; `calibrated_on` 2026-10-13 (`GET /api/v1/models`, 2026-10-14). The model before
  it was `version` 2026-10-06.
- **Combined and per-photo, at every rank.** Top-level `ranks`/`likely`/`specimens` are
  the combined answer over the photos the gate kept. `per_photo[i]` is photo *i* alone,
  with `overall_top`: where the combined top name places in that photo's own ranking.
- **Ranks: species, genus, family, order, class, phylum.** The ranks above family come
  from the iNat taxonomy Vision already holds per genus (taxonomy.py), scored as the best
  of their member species, like genus and family. Their `confidence` is `null` until a
  comparison has fitted their temperature (the next `mv compare` after this ships). Until
  then they're names and scores only. `likely` sets cover species, genus and family.
- **`gate`** (the not-a-fungus gate): per photo `fungus` / `unsure` / `not_fungus` with
  `p_fungus`. A `not_fungus` photo is left out of the combined answer (`left_out`). When
  no photo is clearly a fungus, names are still listed but every `confidence` is `null`
  and nothing is `confident`. When every photo is `not_fungus`, `ranks` are empty: still
  a 200, still counted. `gate` is `null` while the gate isn't deployed.
  `gate.looks_like` is `null` or `"slime"` (a slime mould): then `gate.message` is "This
  looks like a slime mould.", the verdict is `unsure` (rarely `not_fungus`) and never
  `fungus`, names are shown and confidence is `null`. A photo's verdict may carry
  `looks_like: "slime"` and/or `lifted_by: "slime"`.
  Since v1 went live (2026-10-14) the gate is the one re-tuned on dataset release v1
  (`mv fungus-gate tune --release v1`, Open Images negatives only): a photo is `not_fungus`
  when `p_fungus` is below 0.05 and `unsure` below 0.50 (was 0.70 before v1). The cut-offs
  belong to the served gate and may change with it; callers read the verdict, not
  `p_fungus`.
- **`confident`** (per candidate; Vision owns the rule, so the website, the app and
  mycomap.org's importance score never set a threshold of their own): `true` for a rank's top name when the
  gate verdict is `fungus` and its calibrated `confidence` is **≥ 0.70**: Steve's rule for
  the website's answer headline (2026-10-11), so web and app always agree. Ranks without
  a fitted confidence (order and above, for now) are never `confident`. The top-level
  **`answer`** is `{"rank", "name", "confidence"}` for the finest rank that is
  `confident`, or `null`. Confident answers are calibrated on v1's validation split (see
  `confidence` below); how often a confident v1 answer is right has not been measured for
  this doc yet (the pre-v1 figures are in §8). Before v1, a one-name `likely` set was tried
  as the rule and was right only 65% of the time at species, so it isn't used. The threshold may change on Vision; callers don't
  change.
- **Place and date** (the place/date prior). At species and genus, wherever a name appears (`ranks`, `likely.*.names`,
  `per_photo[].ranks`), each name carries:

  ```json
  "place": {"nearest_record_km": 120, "in_season": true, "far_from_known": false,
            "note": null}
  ```

  - `nearest_record_km`: to the centre of the nearest DNA record's 0.2° cell (iNat's
    obscuring grid), rounded to 10 km. No bearing, no record id, so it can't place a
    record.
  - `in_season`: a DNA record of it within 30 days of the date, any year.
  - `far_from_known`: `nearest_record_km` > 300.
  - `note`: "far from known records (possible range extension)", "outside its known
    season", both joined by "; ", or `null`.
  - Nulls: `place` is `null` with neither place nor date, and always at family. Without
    a date, `in_season` is `null`; without a place, `nearest_record_km` and
    `far_from_known` are `null`. `in_season` is also `null` for a taxon with under 5
    dated records, and `nearest_record_km` for one with no record with coordinates.

  Top level: `prior: {"applied": bool}` (v1 leaves out the method id), where `applied` means the place/date
  prior moved the scores (only the `nearest+mean+place` method does, and only given a
  place or date). Also `photo_only_top: {rank: name}`: the ranks where the prior changed
  the top name, `{}` otherwise. All of it is computed in
  `Identifier.identify_vectors` from vectors and context, so a session's re-scoring gets
  it with no photos.
- **Often confused with** (`confused_with`, added 2026-10-13; additive, the place keys
  above are unchanged). On a **species** name, wherever it appears (`ranks.species`,
  `likely.species.names`, `per_photo[].ranks.species`), when people often get that name
  wrong:

  ```json
  "confused_with": {"names": [{"name": "Gymnopilus subspectabilis", "finds": 8}, …],
                    "called": 86, "wrong": 29,
                    "note": "Often confused with G. subspectabilis · 9% of DNA-checked finds"}
  ```

  - Source: mycomap.org's "commonly misidentified as" table ([mycomap.org/misid](https://mycomap.org/misid)),
    built nightly from every iNaturalist identification on its DNA-checked iNat records;
    Vision reads it once a night. `called`: DNA-checked finds someone called this name;
    `wrong`: those DNA said were another species (a synonym or a broader name is not
    wrong). The field is present only when .org says the name **prompts**: it is 80% sure
    DNA says something else on at least 25% of the finds called it (a Jeffreys
    half-count; no minimum number of finds). .org owns that rule.
  - `names`: what DNA said instead, most finds first, at most 3, other species only (a
    narrower provisional DNA type of the same species is not listed). `finds`: finds
    called this name that DNA said were that one.
  - `note`: one line to show under the name, about the first of `names`: "Often confused
    with *X* · *n*% of DNA-checked finds" (a share of `called`, rounded; "under 1%"
    below 0.5%), or, while `called` is under 10, "*k* of *n* DNA-checked find(s)". *X*
    has its genus abbreviated ("G. subspectabilis") only when it is the candidate's genus.
  - Absent (the key is left out, never `null`) at genus and above, when the name doesn't
    prompt, when .org has no record of it, and when Vision holds no table yet. A failed
    nightly read keeps the previous night's table.
  - Matching: .org's name key (lower case, single spaces, straight quotes,
    " var. / subsp. / ssp. / f. " dropped) against .org's names, then against their
    aliases (other spellings, iNat and MycoMap synonyms), then Vision's own spelling key
    (temporary codes 'IN7' = 'IN07', quote glyphs) against both. A name .org knows that
    doesn't prompt gets nothing, even if a synonym does.
- **`confidence`**: a softmax with a temperature fitted per rank (by log-likelihood),
  calibrated on v1's validation split (temperature 0.0274; the place-and-date default
  shares it): the served model's nearest+mean comparison on dataset release v1's `val`
  split, erratum r3 applied (comparison 20261013-104759-9d963d; 0.0274 at species, genus
  and family), which the served `nearest+mean+place` reuses
  (`evaluate.latest_calibration(backbone, "nearest+mean")`). It is fitted again when a new
  model or release is built, not weekly;
  `model.calibrated_on` is that comparison's date. It is calibrated on average, not a
  promise per answer.
- **`likely`**: each rank's likely set, the names at or above a probability floor fitted
  on v1's validation split. `coverage` is how often such a set held the right name on
  validation records held back from the fit (the fit's crosscheck): what the sets
  achieved, not the target they aimed at, which a set cut at 15 names can fall short of.
  `checked_coverage` is the same held-back figure (kept for older clients). Served v1
  model, held-back halves of v1 val: species 0.80 (mean 5.0 names), genus 0.90 (1.3),
  family 0.91 (1.0).
- **`reference_records`**: DNA-validated records of that name in the reference set.
- **Photos in `specimens`** follow Vision's rules: shown only if Creative Commons or the
  owner granted permission on mycomap.org, else `photo_url: null`,
  `photo_withheld: true`. Link `record_url`; never re-host.
- **No coordinates** of any reference record ever appear. `context_used` echoes your own
  place rounded to 0.1°.

### 4.1 `usage`

The credential's plan, its use and limit this period, and when the period resets. App
tokens and developer keys show their own pool (§6.4).

### 4.2 Timing

Milliseconds, measured on the server, also sent as a `Server-Timing` header. Also added
to the website's `POST /api/identify`.

| Field | Meaning |
|---|---|
| `queue_ms` | arrival until this request got the model (waiting behind others) |
| `fetch_ms` | by-observation only: fetching from iNat/MO, incl. waiting for the polite-fetch queue; else `null` |
| `decode_ms` | reading, turning upright and shrinking the uploads |
| `embed_ms` | the model turning photos into vectors; grows with photos |
| `score_ms` | gate, scoring against the reference set, building the answer |
| `model_ms` | `embed_ms + score_ms`: "model time" |
| `total_ms` | arrival to answer, everything included: "total server time" |
| `photos_embedded`, `photos_reused` | embedded in this call / taken from stored vectors |

The network between caller and box is in none of these. It's the caller's own elapsed
time minus `total_ms`.

## 5. Errors

Every refusal is JSON `{"code": "<code>", "detail": "<human text>", …}` (`detail` as on
the website's routes). Retryable ones carry `Retry-After` and `retry_after_seconds`.

| Status | `code` | When |
|---|---|---|
| 400 | `bad_request` | malformed field, lat/lng out of range, bad date, unknown model |
| 400 | `too_many_photos` | > 10 in a request or session |
| 401 | `invalid_token` | missing, unknown or malformed credential |
| 401 | `token_expired` | an app token past `exp`: re-mint once and retry |
| 403 | `key_inactive` | key pending, disabled, revoked or owner inactive (`reason`) |
| 404 | `session_gone` | unknown, expired or lost to a restart, or someone else's |
| 404 | `observation_not_found` | |
| 413 | `too_large` | file > 25 MB, request > 80 MB, image > 40 MP |
| 415 | `unsupported_image` | |
| 422 | `no_usable_photos` | an observation with no photo Vision can use |
| 429 | `rate_limited` | the short window (per minute), `Retry-After` |
| 429 | `quota_exceeded` | the period's identifications are used up; `limit`, `used`, `resets_at` |
| 502 | `source_unavailable` | iNat or MO did not answer |
| 503 | `busy` | the queue is full; `estimated_wait_seconds`, `Retry-After` |
| 503 | `updating` | the reference set is being swapped (seconds); `Retry-After: 30` |
| 503 | `auth_unavailable` | .org couldn't be asked about a key Vision has no recent answer for |

429 means the caller's own limits; 503 means the server is out of capacity, so the
caller did nothing wrong (Steve).

**Limits are per credential, never per IP**, for anything authenticated: a foray group
on one hotspot or carrier NAT shares an address. The per-IP limit stays only on the
website's anonymous routes.

**The queue.** Today the box refuses past `MV_MAX_QUEUE` identifications in flight. v1
makes it a FIFO queue in front of the model. `queue_ms` reports the wait. When the
estimated wait (photos ahead × the recent seconds per photo) would pass
`MV_QUEUE_MAX_WAIT` (20 s), the call is refused at once with `busy` and that estimate as
`Retry-After`. Website and app session calls go ahead of developer and by-observation
calls (Steve).

**What the app does**: at most one photo in flight per device, ≤ 10 per
observation, nothing when the user turned suggestions off. On `busy`/`updating` it
keeps the photo queued and retries with backoff and jitter ("suggestions pending").
On `quota_exceeded` it stops until `resets_at` and says so once.

## 6. Keys, plans and quotas: the .org ⇄ Vision contract

**.org owns** tokens, plans, limits and per-key overrides, and shows use. **Vision
enforces and counts** (requests land on Vision) and reports use to .org in batches.
No new secret on the Vision box: it reuses `MV_ORG_VISION_KEY` and the bridge public key.

### 6.1 Developer tokens: the existing service-key system

- .org adds `vision` to `SERVICE_INTROSPECTION_SECRET_ENV`, mapped to `VISION_API_KEY`
  (the key Vision already presents). Tokens are `vision_<base64url>`, hashed, created
  with a purpose, approved, disabled and revoked exactly like `atlas_` tokens, on member
  and admin pages beside the Atlas ones: members request theirs at mycomap.org/vision-keys,
  and admins approve, issue and set limits under Admin → API Keys (`/admin/api-keys`).

### 6.2 Introspection for `vision`

`POST https://mycomap.org/api/service-keys/introspect`,
`Authorization: Bearer <VISION_API_KEY>`, `{"key": "vision_…"}` →

```json
{"active": true, "service": "vision", "keyId": "42", "tier": "standard",
 "plan": "member", "limit": {"monthly": 2000, "period": "2026-10"},
 "limit_source": "plan", "subject": "u_3f9c…"}
```

or `{"active": false, "reason": "unknown|pending|disabled|revoked|account_inactive|wrong_service"}`.

- `limit` is the effective limit after any per-key admin override (`limit_source`:
  `plan` | `override` | `staff`). `monthly: null` means unlimited. Vision enforces
  exactly that. `plan` is for display.
- **`subject` (Vision asks for it):** an opaque, stable id for the owner, the same value
  as an app token's `sub`: `"u_"` + the first 32 characters of base64url
  HMAC-SHA256(secret, `"vision-subject:v1:" + userId`). The secret is
  `VISION_SUBJECT_SECRET` when set; on production it is otherwise derived from the
  sign-in bridge's signing key, never `VISION_API_KEY`.
  Vision treats the value as opaque. A member's keys and app use share one pool without
  Vision learning who they are.
  **One pool per member** (Steve): every key and the app count against the member's
  `subject`. An admin override on a key gives that key its own limit.
- Vision caches positive answers 60 s and negative ones 10 s. If .org is unreachable, a
  cached active answer is used for up to 10 min, then 503 `auth_unavailable`. Revocation
  reaches Vision within 60 s.

### 6.3 App tokens

- `POST https://mycomap.org/api/vision/app-token` (the app's Bearer JWT, or a web
  session) → `{"token": "mvt.<body>.<sig>", "expires_at": "…" (15 min),
  "vision_base": "https://vision.mycomap.org",
  "vision_api": "https://vision.mycomap.org/api/v1", "quota": {"limit", "period"}}`.
  **The app takes Vision's address only from `vision_api`**, so it can move without an
  app release.
- Same Ed25519 key and body/signature format as the sign-in bridge; body
  `{"v": 1, "typ": "vision-api", "sub", "aud": "https://vision.mycomap.org", "iat",
  "exp": iat + 900, "plan", "limit"}`, where `limit: null` means unlimited (staff).
- Vision requires `typ == "vision-api"`, the exact `aud`, and `exp` not passed (30 s skew).
  A bridge sign-in token has no `typ`, so it can't be used here. An app token has no
  `nonce`, so it can't finish a sign-in. Both directions are tested on Vision.
- Plan changes and overrides take effect at the next mint (≤ 15 min). The app refreshes
  before expiry and re-mints once on `token_expired`.

### 6.4 Counting

- **Unit: one identification** = one `identify` call, one by-observation call, or one
  session (counted at its first photo; ≤ 10 photos). Counting per photo would punish
  adding more photos, which gives better answers (Steve).
  A session re-opened for the **same find** (same member pool, same `client_ref`) within
  24 hours of the counted one is free: not counted and not refused at the limit. The app
  re-opens one after a restart, the 30-minute idle end or the 2-hour cap (Steve,
  2026-10-12). Its photos still count as photos. Without a `client_ref`, every session
  counts. The same rule holds for `POST /identify` with a `client_ref`: the app finishes a
  find that way when signal dropped mid-session.
  Modes (`MV_QUOTAS`): `off`; `count` (counted, shown in `usage`, reported, never refused);
  `enforce`. A credential with no limit gets `MV_QUOTA_DEFAULT_MONTHLY` (100); `null` is
  unlimited.
  The CPU cost is per photo, so Vision also keeps a short per-credential window in
  **photos per minute** (`MV_PHOTOS_PER_MINUTE`, default 20) and a global daily ceiling
  for developer tokens (`MV_API_DAILY_PHOTOS`), so API use can't eat the capacity the
  website and app need.
- A `photo_ref` resend, `PATCH`, `GET` and `DELETE` are free. A refusal is not counted.
- Counts per period live in SQLite on the box (they survive restarts).

### 6.5 Reporting use to .org

`POST https://mycomap.org/api/vision/usage`, `Authorization: Bearer <VISION_API_KEY>`,
every ~5 min and once more at period rollover:

```json
{"period": "2026-10", "reported_at": "2026-10-11T14:05:00Z",
 "rows": [{"kind": "key", "keyId": "42", "identifications": 51, "photos": 140,
           "refused": {"quota_exceeded": 3, "rate_limited": 0, "busy": 1},
           "last_used_at": "…"},
          {"kind": "user", "sub": "u_3f9c…", "identifications": 12, "photos": 40,
           "refused": {…}, "last_used_at": "…"}]}
```

Rows are **running totals for the period**. .org keeps the greatest value per
`(period, kind, keyId|sub)`, so repeats, retries and reordering are harmless. Reports
that fail queue on the box and are resent. .org shows use in the key manager (member:
their tokens; admin: per key and per member) along with the last report time. Quota is
enforced from Vision's own count, never from .org's display.

### 6.6 Plan limits

Steve's table (2026-10-11), identifications per month per member, set on .org:

| Plan (.org membership level) | identifications / month |
|---|---|
| free | 100 |
| community-supporter | 500 |
| network-supporter | 2,000 |
| sustaining-supporter | 10,000 |
| partner-organization | 10,000 |
| sustaining-organization | 25,000 |
| institutional-partner | 75,000 |
| staff | unlimited (`null`) |

A per-key admin override replaces the plan's number for that key. The box can't
serve every plan at its ceiling at once: throttled, it answers roughly 1,500–2,500
identifications a day for everyone together (`docs/capacity.md`). So the
photos-per-minute guard and the developer-token daily ceiling (`MV_API_DAILY_PHOTOS`)
stay in front of the monthly numbers.

## 7. Taxon counts

`GET /api/v1/taxon-counts`: `VISION_API_KEY` or any active credential;
`Cache-Control: max-age=3600`, `ETag`, `If-None-Match` → 304.

```json
{"generated_at": "2026-10-11T12:00:00Z",
 "model": {"name": "MycoMap Vision", "version": "2026-10-13", "data": {…}, "calibrated_on": null},
 "nightly_version": 12,
 "counts_are": "reference_records",
 "species": {"Amanita muscaria": 412, "Russula sp. 'IN01'": 7},
 "genera": {"Amanita": 3120},
 "families": {"Amanitaceae": 3300},
 "totals": {"records": 169129, "species": 17854, "genera": …, "families": …}}
```

- Read from the served index (`Identifier.rank_counts`), the same number every candidate
  carries as `reference_records`: exactly what the live model knows. North America, no
  label conflict, embedded, nightly additions in, photos of anyone who withdrew out.
  Records, not photos.
- Keys are Vision's labels: .org observation names with spellings folded (names.py ≡
  .org nameVariants.ts). Temp codes are species keys like any other. Synonyms and s.l. /
  complex groups are **not** folded (scoring-only in Vision). A name not present has 0.
- About 300 KB of JSON, ~60 KB gzipped.

## 8. Reading reference depth and confidence

Pre-v1 numbers (exploratory, measured before dataset release v1 was frozen). The v1 model's
numbers on v1's `val` split are published on the site's Results page
(vision.mycomap.org/research/results) through `mv benchmark-export --comparison`, not here.

Species top-1 by the true species' reference records (the model served then, pre-v1,
nearest+mean, newest 28 days, 1,152 test records, comparison 20261008-012435-4ef7b0):

| reference records | 0 | 1–4 | 5–19 | 20–99 | 100+ |
|---|---|---|---|---|---|
| species right (top 1) | 0% | 13% | 40% | 58% | 68% |
| genus right, by the species' depth | 51% | 59% | 82% | 82% | 91% |

Below 5 a species is effectively unknown to Vision. Gains flatten past ~20 and are small
past ~100. These drift weekly as green records arrive, so callers keep cut-offs as
settings. Still to measure: genus right by **genus** depth.

**How often the top name is right, by its confidence.** Held-out dev benchmark
(heldout-2026-10-08, the 2,986 `dev` records with answers, never in the reference set),
served model, no place/date, answer key = the record's observation name. Species top-1
overall 51%, genus 81%.

| | share of records | right |
|---|---|---|
| species top ≥ 0.9 | 2.1% | 95% |
| species top ≥ 0.7 (**`confident` at species**) | 9.6% | 96% |
| species top ≥ 0.5 | 22.9% | 86% |
| genus top ≥ 0.7, species below (**`answer` at genus**) | 52.2% | 96% |
| neither (answer at family or none) | 38.2% | |
| *one-name species likely set (rejected rule)* | *9.9%* | *61%* |

These are offline numbers: the stored answers of the 2026-10-09 benchmark run (scored by
`mv heldout`), before the not-a-fungus gate, single-photo embedding and int8 serving.

**Through the app's photo path** (integration testing, 2026-10-12; exploratory,
pre-v1): 2,989 dev records of heldout-2026-10-08, through the app's photo path (1024 px
JPEG q85, EXIF dropped) to POST /api/v1/identify, top_k 10, on a **local** Vision server
at main b8f789f (laptop GPU, fp32/fp16 as main serves, no int8), method
nearest+mean+place, fungus gate mode both (fingerprint 73ce44b9afa55ff6). Reference = the
laptop's shared manifest after the MO migration (151,870 records, 17,262 species), not
the box's release. Confident species answers (v1 `answer` at species): right **93.5%**
(273/292, photos only) and **92.2%** (412/447, with place/date); genus 95.8% / 94.9%.
The box's reference set differed. These are pre-v1 figures for the model served then;
don't quote them for the v1 model, which has no measured figure here yet.

The benchmark's photos are DNA-validated records' own iNat photos: all fungi, often
taken with care. App photos at a foray will score lower. So treat these shares as an
upper bound, and expect fewer `confident` answers in the field.

## 9. Settings (each off until turned on; Steve decides)

| Setting | Turns on |
|---|---|
| `MV_API_V1=1` | `POST /api/v1/identify`: credentials, timing, errors, `confident`/`answer`, higher ranks |
| `MV_SESSIONS=1` | the observation sessions (§3.3) |
| `MV_BY_OBSERVATION=1` | identify by iNat or MO observation number (§3.2); needs `MV_API_V1=1` |
| `MV_QUOTAS=off\|count\|enforce` | §6: `count` counts and reports use without refusing, for a trial |
| `MV_TAXON_COUNTS=1` | `GET /api/v1/taxon-counts` (§7) |
| `MV_DEFAULT_METHOD` | the method an identification uses (default `nearest+mean`; the box serves `nearest+mean+place`, which answers as `nearest+mean` without a place or date) |
| `MV_SEARCH_INT8=1` | the fast int8 search of the reference set (docs/capacity.md, option C) |
| `MV_EMBED_ONNX=1` | the image tower as ONNX: not for the 4 GB box (docs/capacity.md, option D) |
| `MV_PHOTOS_PER_MINUTE`, `MV_QUEUE_MAX_WAIT`, `MV_SESSIONS_MAX_MB`, `MV_QUOTA_DEFAULT_MONTHLY`, `MV_USAGE_REPORT_SECONDS` | limits, defaults in .env.example |

mycomap.org's side: the `vision` service in its service keys, the introspection fields,
`/api/vision/app-token`, `/api/vision/usage`, plan limits and per-key overrides, the
member and admin pages, and the app's client.

## 10. Decisions (Steve, 2026-10-11)

1. Server out of capacity → 503 `busy`; the caller's own limits → 429.
2. Quota counts identifications (a call, or a session from its first photo), with a
   photos-per-minute guard per credential.
3. One pool per member (an opaque `subject` from .org), with a per-key admin override.
4. Sessions: 30 min idle, 2 h hard cap.
5. Interactive calls (website, app sessions) ahead of developer and by-observation calls.
6. `keep_photos` is a no-op in v1: no photo is kept.

The box stays 4 GB for now (Steve declined the resize). Session memory has a hard cap,
set in `docs/capacity.md`.
