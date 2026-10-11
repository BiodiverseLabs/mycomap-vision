# MycoMap Vision public API (v1), proposal

Status: **draft for agreement**, 2026-10-11. Nothing here is built or deployed yet.
Written by the Vision API lane with Mobile G (.org keys, plans, quotas, admin, app UI),
whose proposal of the same day is merged in. Mobile H (taxon counts), the not-a-fungus
gate lane and the Vision Coordinator are copied. Steve reviews and decides; items marked
**(decide)** are open.

Everything is under `https://vision.mycomap.org/api/v1/`. The website's routes
(`/api/identify`, `/api/models`, …) stay as they are. They gain only the `timing` block.

## 1. Who calls, and how they authenticate

| Caller | Credential | How Vision checks it |
|---|---|---|
| External developer | a **Vision service token** `vision_…` from the .org key manager: `Authorization: Bearer vision_…` (or `X-API-Key: vision_…`) | .org's existing `POST /api/service-keys/introspect` (built for Atlas), service `vision` added; cached on Vision |
| The mobile app, for a signed-in member | a **Vision app token** `mvt.<body>.<sig>`, minted by .org for 15 min (§6.3): `Authorization: Bearer mvt.…` | verified on Vision with the sign-in bridge's Ed25519 public key it already holds. No call to .org |
| mycomap.org's server (e.g. Mobile H's importance score) | `VISION_API_KEY`, the server-to-server key the two already share | constant-time compare |

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
`{"model", "top_k", "use_place": true, "use_date": true}`.

- Vision fetches the observation and its medium photos from iNat or MO, keeps them in
  memory only, and stays within those sites' limits (iNat ≤ 1 API request/s and the media
  caps in CLAUDE.md; MO its own pacer). One fetch at a time across all callers, so expect
  seconds in `timing.fetch_ms`.
- Place and date are the observation's **public** ones. An obscured iNat location is
  used as iNat publishes it, never a true location Vision may hold.
- **Its own record is left out.** When the observation is one of Vision's DNA-validated
  references, it is removed from the reference set for this answer, and
  `excluded_reference: {"source", "source_id"}` says so. Otherwise it would match
  itself. If Vision already holds that record's vectors, it uses them and fetches
  nothing (`timing.photos_reused`).
- Refusals: 404 `observation_not_found`, 422 `no_usable_photos`, 502
  `source_unavailable` (with `Retry-After`).

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
"model": {"backbone", "method", "version"}, "context_used": {…}}`

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
- **What is kept.** Per photo: its vector for the served model, its gate verdict and its
  per-photo answer, all computed once on upload. The photo itself is decoded, embedded
  and dropped in that call; **no photo bytes are kept**. Re-scoring after a new photo,
  a removed one or a place/date change runs `identify_vectors` over the stored vectors
  and never re-embeds. With today's default method (nearest+mean), place and date don't
  change the scores. They're echoed and used by methods with a place/date prior.
- **Lifetime.** 30 min after the last call, 2 h at most from creation **(decide)**. A
  session gone to expiry or a restart answers 404 `session_gone`. The app opens a new
  one and resends the photos it keeps locally anyway. Sessions belong to the credential's
  subject; anyone else gets 404.
- `keep_photos: true` is accepted and recorded but changes nothing in v1. Keeping
  consented photos (for review or training) needs storage and a consent text Steve
  approves **(decide, later)**.
- Memory: ~3 KB of vector per photo plus its per-photo answer (~5 KB), so 1,000 open
  sessions of 10 photos ≈ 80 MB. Capped by `MV_SESSIONS_MAX` (default 500); past it the
  oldest idle session is dropped.

## 4. The answer

Every identify call, and `combined` in a session, is today's per-model result with
`request_id`, `gate`, `confident`, `timing` and `usage` added. A v1 call returns one
model's result, not a `results` list.

```json
{
  "request_id": "r_01J…",
  "model": {"backbone": "bioclip-2", "method": "nearest+mean", "version": "<release>",
            "calibrated_on": "2026-10-08"},
  "reference": {"records": 152915, "species": 9876, "photos": 593224},
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
    "species": {"coverage": 0.9, "checked_coverage": 0.6, "capped": false,
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
  "confidence_note": "Calibrated on 1,152 test records …",
  "timing": { … §4.2 … },
  "usage": {"plan": "member", "used": 37, "limit": 200, "period": "2026-10",
            "resets_at": "2026-11-01T00:00:00Z"}
}
```

- **Combined and per-photo, at every rank.** Top-level `ranks`/`likely`/`specimens` are
  the combined answer over the photos the gate kept. `per_photo[i]` is photo *i* alone,
  with `overall_top`: where the combined top name places in that photo's own ranking.
- **Ranks: species, genus, family, order, class, phylum.** The ranks above family come
  from the iNat taxonomy Vision already holds per genus (taxonomy.py), scored as the best
  of their member species, like genus and family. Their `confidence` is `null` until a
  comparison has fitted their temperature (the next `mv compare` after this ships). Until
  then they're names and scores only. `likely` sets cover species, genus and family.
- **`gate`** (not-a-fungus gate lane): per photo `fungus` / `unsure` / `not_fungus` with
  `p_fungus`. A `not_fungus` photo is left out of the combined answer (`left_out`). When
  no photo is clearly a fungus, names are still listed but every `confidence` is `null`
  and nothing is `confident`. When every photo is `not_fungus`, `ranks` are empty: still
  a 200, still counted. `gate` is `null` while the gate isn't deployed.
- **`confident`** (per candidate, Vision owns the rule, so the app and Mobile H never
  set a threshold of their own): `true` only for a rank's top name when (1) the gate
  verdict is `fungus`, (2) that rank's `likely` set holds **only that name**, and (3) at
  species, the name has ≥ 5 reference records. Ranks without a likely set (order and
  above, for now) are never `confident`. If measurement shows a better rule (§8), the rule
  changes on Vision and callers don't change.
- **`confidence`**: a softmax with a temperature fitted per rank on the newest weekly
  comparison (by log-likelihood). It is calibrated on average, not a promise per answer.
  `checked_coverage` says how often a likely set really held the right name on unseen
  records (today 0.6 for a 0.9 target at species).
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
| `fetch_ms` | by-observation only: fetching from iNat/MO, incl. waiting for the polite-fetch lane; else `null` |
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
caller did nothing wrong. Agreed with Mobile G. **(decide: Steve's note said
429 for capacity too; either works with an honest `Retry-After`.)**

**Limits are per credential, never per IP**, for anything authenticated: a foray group
on one hotspot or carrier NAT shares an address. The per-IP limit stays only on the
website's anonymous routes.

**The queue.** Today the box refuses past `MV_MAX_QUEUE` identifications in flight. v1
makes it a FIFO queue in front of the model. `queue_ms` reports the wait. When the
estimated wait (photos ahead × the recent seconds per photo) would pass
`MV_QUEUE_MAX_WAIT` (20 s), the call is refused at once with `busy` and that estimate as
`Retry-After`. Website and app session calls go ahead of developer and by-observation
calls **(decide)**.

**What the app does** (Mobile G): at most one photo in flight per device, ≤ 10 per
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
  and admin pages beside the Atlas ones.

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
  `plan` | `override`). Vision enforces exactly that. `plan` is for display.
- **`subject` (Vision asks for it):** an opaque, stable id for the owner, the same value
  as an app token's `sub`, e.g. an HMAC of the user id. Then a member's keys and app
  use can share one pool if Steve wants that, without Vision learning who they are.
  Without it, each key is its own pool and a member with 3 keys gets 3×.
  **(decide: per-key or per-member pool.)**
- Vision caches positive answers 60 s and negative ones 10 s. If .org is unreachable, a
  cached active answer is used for up to 10 min, then 503 `auth_unavailable`. Revocation
  reaches Vision within 60 s.

### 6.3 App tokens

- `POST https://mycomap.org/api/vision/app-token` (the app's Bearer JWT, or a web
  session) → `{"token": "mvt.<body>.<sig>", "expires_at": "…" (15 min),
  "vision_base": "https://vision.mycomap.org", "quota": {"limit", "period"}}`.
- Same Ed25519 key and body/signature format as the sign-in bridge; body
  `{"v": 1, "typ": "vision-api", "sub", "aud": "https://vision.mycomap.org", "iat", "exp",
  "plan", "limit"}`.
- Vision requires `typ == "vision-api"`, the exact `aud`, and `exp` not passed (30 s skew).
  A bridge sign-in token has no `typ`, so it can't be used here. An app token has no
  `nonce`, so it can't finish a sign-in. Both directions are tested on Vision.
- Plan changes and overrides take effect at the next mint (≤ 15 min). The app refreshes
  before expiry and re-mints once on `token_expired`.

### 6.4 Counting

- **Unit: one identification** = one `identify` call, one by-observation call, or one
  session (counted at its first photo; ≤ 10 photos). Counting per photo would punish
  adding more photos, which gives better answers. Proposed by Mobile G **(decide)**.
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

The numbers come from the capacity note (`docs/capacity.md`, next) and Steve.
Placeholders, identifications per month: free member 300, supporting member 3,000,
staff unlimited, per-key admin override any.

## 7. Taxon counts (Mobile H)

`GET /api/v1/taxon-counts`: `VISION_API_KEY` or any active credential;
`Cache-Control: max-age=3600`, `ETag`, `If-None-Match` → 304.

```json
{"generated_at": "2026-10-11T12:00:00Z",
 "model": {"backbone": "bioclip-2", "method": "nearest+mean", "version": "<release>"},
 "nightly_version": 12,
 "counts_are": "reference_records",
 "species": {"Amanita muscaria": 412, "Russula sp. 'IN01'": 7},
 "genera": {"Amanita": 3120},
 "families": {"Amanitaceae": 3300},
 "totals": {"records": 152915, "species": 9876, "genera": 1450, "families": 210}}
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

Species top-1 by the true species' reference records (fine-tuned BioCLIP 2,
nearest+mean, newest 28 days, 1,152 test records, comparison 20261008-012435-4ef7b0):

| reference records | 0 | 1–4 | 5–19 | 20–99 | 100+ |
|---|---|---|---|---|---|
| species right (top 1) | 0% | 13% | 40% | 58% | 68% |
| genus right, by the species' depth | 51% | 59% | 82% | 82% | 91% |

Below 5 a species is effectively unknown to Vision. Gains flatten past ~20 and are small
past ~100. These drift weekly as green records arrive, so callers keep cut-offs as
settings. Before launch, this section gains measured tables: genus right by **genus**
depth, and how often the top name was right when `confident` was true and when its
`confidence` was ≥ 0.5 / 0.7 / 0.9.

## 9. Rollout (each behind a flag, one branch each, Steve reviews)

| Flag | Vision branch | Turns on |
|---|---|---|
| `MV_TAXON_COUNTS=1` | `feat/api-taxon-counts` | §7 (smallest; Mobile H waits on it) |
| `MV_API_V1=1` | `feat/api-v1-identify` | identify by photos, credentials, timing, errors, `confident`, higher ranks, the queue |
| `MV_SESSIONS=1` | `feat/api-v1-sessions` | §3.3 |
| `MV_API_V1_SOURCES=1` | `feat/api-v1-by-observation` | identify by iNat / MO number |
| `MV_QUOTAS=off\|count\|enforce` | `feat/api-v1-quotas` | §6: `count` reports use without refusing, for a trial |

.org (Mobile G): `vision` in `serviceApiKeys.ts`, the introspection fields, 
`/api/vision/app-token`, `/api/vision/usage`, plan limits and per-key override
(additive column), member and admin pages, the app client and UI. Nothing deploys to
Vision without Steve's OK.
