# Atlas for Vision: a suitability prior and an environment modality

Status: **design, decided (Steve, 2026-10-12; section 6)**. Nothing built, fitted or spent yet.
Asked by Steve, 2026-10-11: "can we use atlas for this in some way, as it already has climate
and substrate data in a better model?"

**Short answer.** Yes, but mostly through Atlas's *layers*, not its per-taxon maps. Today
Atlas's maps cover the true species of only 18–24% of held-out records, and a map says nothing
more than 500 km from the taxon's known sites. So the maps can't protect the far finds Steve's
rule is about. The arm most likely to help far finds is a multi-species environment head:
Atlas's climate, soil, host-tree and forest layers, sampled at each record's public point and
trained on v1 `train`. It covers every species, can't leak, and costs CPU minutes. Atlas's
suitability ranks go in the same pre-declared experiment as a second arm, scored against a
leave-`val`-out Atlas fit (~$1–8).

## 0. Facts this rests on

| fact | value | source |
|---|---|---|
| Atlas maps for held-out truths (10/7 release: 8,908 models, 6,434 taxa) | any model 39.7%; full panel (≥20 sites) **24.0%**; full panel + passed null **17.8%** of 13,145 pool records | measured 2026-10-11, `/api/models` × `heldout-2026-10-08/pool.csv`, names folded by case and quotes (an approximation of `atlas_name_key`) |
| "passed" today | scatter null, 73.5% false-pass. Strong maps after the rebuild ≈ 400 taxa, so coverage drops below 17.8% | Atlas m2 calibration, 10/8 |
| Accessible area | 500 km buffer around presences; no row outside it | `R/background.R` `atlas_accessible_area(buffer_km = 500)` |
| Phenology | none in Atlas | `atlasrange.py` docstring |
| Prior export (Atlas `feat/vision-prior`, in the final merge batch) | 20 km Albers cells, rank 0–100 (uint8) per taxon, strong maps only, ~5–15 MB; private trained-ids list per release | Atlas memory, 10/9 |
| Production layers | elevation, bioclim (bio1–19), terrain, soil (SoilGrids), landcover, hosts (genera) + `host_known`, hostsdecay (10 genera), foresttype + `forest_known`. Carbonate bedrock and landform (TWI, northness, heat load) were **dropped** in the 9/30 sweep | `R/layersweep.R` `ATLAS_PRODUCTION_LAYERS`, `R/layers.R` |
| Grids | live release `draft` = 2.5′ (~5 km); `production` = 0.5′ (~1 km) | `atlas_source_resolution` |
| Vision box | 4 GB, cgroup cap 3.5 GB; projected v1 working set ~3.1 GB with int8 vectors, so **~0.4 GB headroom** | `docs/retrain-v1.md` §5 |

Where the served prior loses records (`docs/experiments/2026-10-09-prior-tuning.md`). Test
split, every truth with a DNA reference, species top-1, nearest + species average → setting
(fixed / broken):

| distance to truth's nearest DNA record | n | first choice | boost + far penalty (**served**) | gated |
|---|---|---|---|---|
| < 100 km | 6,113 | 64.3 → 67.9 (+354 / −134) | 64.3 → 68.8 (+417 / −142) | 66.1 (+143 / −35) |
| 100–300 km | 1,890 | 50.1 → 52.6 (+118 / −69) | 50.1 → 47.6 (+70 / −116) | 49.6 (+22 / −31) |
| **300–1,500 km** | 1,306 | 30.8 → 25.4 (+21 / −91) | 30.8 → 27.5 (+19 / −62) | 29.4 (+4 / −22) |
| > 1,500 km | 215 | 11.2 → 7.4 (+1 / −9) | 11.2 → 10.7 (+5 / −6) | 12.1 (+3 / −1) |

The mechanism: ranking up species with nearby records is what pushes a far true species down.
Taking away the absence penalty didn't fix it, and neither did gating.

## 1. Place prior from Atlas suitability

**Term.** `log f(rank(taxon, cell))` replaces or adds to the DNA-distance boost. `f` is the
11-bin factor that `atlasrange.estimate_rank_factor` learns on the training folds; raw ranks
are percentiles within each taxon's own range, so they are never compared across taxa. It is
boost only: f ≥ 1, so no taxon is ever ranked down for a low rank.

**Outside the accessible area: neutral, fixed, not tuned.** This changes `atlasrange`'s
stub, which treats outside as out of range with a tuned penalty. Penalising outside the area
would rebuild the distance prior's far-find loss with a 500 km radius.

**Season.** Atlas has no phenology, so every arm keeps the served DNA season term (0.5 at
10 days) and the far penalty beyond 1,500 km.

**Fallback for taxa without a strong map.** Either the served DNA boost or nothing; both are
arms (P2a/P2b).

**Expected effect by band, from the mechanism above:**

| band | expectation | why |
|---|---|---|
| < 100 km | gains similar to the DNA boost, but only for the ≤ 18% of records with a mapped truth | the maps and DNA records agree near home |
| 100–300 km | **likely better** than served (+70 / −116) | an Atlas rank is smooth across suitable habitat; the 75 km kernel prefers species recorded right at the spot |
| 300–500 km | possible gain | inside the accessible area a far true species can be boosted; the DNA kernel gives ≈ 0 there |
| 500–1,500 km | **loss likely persists** | Atlas is silent, so the true species gets no boost while nearby mapped look-alikes do |
| sparse truths | possible new loss | unmapped truths (most far finds are sparse species) compete with boosted mapped look-alikes |

Verdict: the Atlas maps alone are unlikely to pass Steve's rule. They should shrink the
100–500 km loss for about a fifth of records.

**Environment head (the better route).** A small MLP: ~55 Atlas layer bands plus month
(sin/cos) → 256 → 256 → every v1 species. It uses a presence-only, SINR-style loss and is
trained on v1 `train` only. Rare species shrink toward their genus, as the month prior does.
`log p(c | env, month)` is added at a tuned weight. It has no distance input, so a far find in
suitable habitat isn't penalised for being far, and it covers every species with ≥ 1 train
record. Its risk: sister taxa split east/west in similar climates.

**Pre-declared experiment on v1.** Commit the grid before any result. Everything is
nearest + species average + the served season term + the far penalty, on the v1-retrained
model, in the same run as `retrain-v1.md` §6 item 6 (the prior re-tune):

| arm | place term | settings |
|---|---|---|
| P0 | none (nearest + species average) | reference |
| P1 | served DNA boost, re-tuned on v1 | prior lane's grid |
| P2a / P2b | Atlas f(rank) for mapped taxa; unmapped → served DNA boost (a) or 0 (b) | weight 0.25 / 0.5 / 1 (×2 = 6) |
| P3 | max(DNA boost, Atlas boost) | weight 0.25 / 0.5 / 1 (3) |
| P4 | environment head | weight 0.25 / 0.5 / 1; month in or out (6) |
| P5 | environment head + served DNA boost | env weight 0.25 / 0.5 (2) |

- **Selection** (as in the gated follow-up): on 5 observer folds of `val`, the most species
  right among settings that hold the band on the training folds; otherwise no prior.
- **Pass** (Steve, 10/9): species top-1 above P0 overall, **and** right answers in the
  300–1,500 km band ≥ P0's, both for every truth with a reference and for truths with 6+
  references.
- **Also reported:** each arm against P1. The band is split into 300–500 and 500–1,500 km
  (the accessible-area edge), and every table is split by whether the truth has a strong map.
  Standard summary format throughout.
- **Confirmation:** `v1-test` once, all arms together.
- **Power:** draft `val` had 5,156 records, so ≈ 680 in the band (13%). If `v1-test` is
  ~1,000 records, its band holds ~130 and a "no loss" verdict there is within noise (±8
  records). D5 decides where the band is judged.

## 2. Environment modality in the v1 benchmark (FungiTastic-style)

**Layers.** Atlas's production set, about 55 bands (the exact list comes from Atlas's layer
manifest). Leave out anything derived from MycoMap records, such as effort/sampling layers and
Atlas suitability: those would leak.

Phase 2, after v1: Daymet weather for the 30 and 90 days before the observation date
(precipitation, tmin, tmax). Fruiting follows recent weather, and this is the one signal
Atlas doesn't have.

**Point: the record's public point only.** That is iNat's public (randomised if obscured)
coordinate, or MO's GPS, or the centre of its location box when the GPS is hidden. Never
.org's true point. ~55 continuous bands nearly pin down a 5 km cell (look the values up in the
rasters), so values at a true point would publish the location `v1-cc` withholds. Values at
the public point show nothing the source doesn't already show. An internal true-point row is
allowed as an upper bound (private only).

**Resolution: 5 km (Atlas `draft`, 2.5′).** It matches iNat's obscuration cell (~20 km), the
live Atlas maps, and a ~100 MB serving table (section 4). 1 km is an option (D7).

**Storage: a derived sidecar `v1-env`**, built from v1 the way `v1-cc` is, so v1 itself
doesn't change and the cut isn't blocked:
- an `environment` table keyed by `record_key`;
- integer-scaled values, because the spec allows no floats in hashed columns (bio1 °C×10,
  pH×10, shares ×1000);
- `env_point` (public / box / none) and `env_precision_m`;
- each layer file's build id and sha256;
- a MANIFEST naming v1's `release_hash` as parent, with its own `env_hash`.

Size: ~150k records × 55 × int16 ≈ 16 MB, about 5 MB compressed.

**Needs from v1 (before the cut; D1 yes).** Freeze each record's public point in
`private.sqlite`; otherwise the sidecar has to read iNat's current public point, which drifts
as obscuration changes and records are deleted. **Agreed with the retrain lane (2026-10-12),
which builds it (final):** a `private.sqlite` table `record_public_places`, one row per
candidate record.

| column | meaning |
|---|---|
| `record_key` (PK) | as `release.records` |
| `public_lat_micro`, `public_lon_micro` | integer µ°; null when `public_kind` = `none` |
| `public_kind` | `inat_open`; `inat_obscured` (`obscured` = 1, or `geoprivacy` / `taxon_geoprivacy` obscured: iNat's randomised point in its 0.2° cell); `mo_gps` (shown, not `gps_hidden`); `mo_box_centre` (GPS absent or hidden); `none` (iNat private or no location; MO with neither) |
| `positional_accuracy_m` | `inat_open`: iNat `positional_accuracy` (null if absent); `inat_obscured`: 22,240 (the 0.2° cell); `mo_box_centre`: half the box diagonal; `mo_gps`: null |
| `box_n_micro`, `box_s_micro`, `box_e_micro`, `box_w_micro` | MO location box edges, integer µ°; null unless `mo_box_centre` |
| `source`, `read_at` | `inat` / `mo`; when the public point was read. iNat: anonymous v2 (fetch-inat, so `location` is already obscured, never `private_location`). MO: anonymous api2 via `mv fetch-mo-points` (7,201 records, MO's 5 s pace) |

It is hashed into `private_hash` and never published, and sits beside `record_places`
(.org's own point, kept for the internal place prior). A frozen build refuses unless every
included record has a row (`none` is explicit).

Sidecar mapping:
- `env_point`: public for `inat_*` and `mo_gps`, box for `mo_box_centre`, none for `none`.
- `env_precision_m`: `positional_accuracy_m`.

Checked on the shared manifest (retrain lane, 2026-10-12): 14,942 geoprivacy-obscured and 128
taxon-obscured records count as `inat_obscured`. All 71 geoprivacy-private records and 9 open
records without a location have no anonymous coordinates, so they land as `none` and never get
.org's point.

**Licences.** I'm not a lawyer; this is Steve's call.

| layers | licence | effect on a public release |
|---|---|---|
| bioclim, elevation, terrain (derived) | WorldClim, CC BY-SA 4.0 | the env table is plausibly an adaptation, so **that file** is CC BY-SA 4.0. The records and photos tables are separate files (a collection, not an adaptation) and keep their licences |
| soil, landcover, foresttype (NALCMS, Copernicus fill) | CC BY 4.0 | attribution |
| hosts, hostsdecay | BIGMAP: US public domain; NFI: OGL-Canada | attribution |
| alternatives | ClimateNA 1991–2020 (AdaptWest, CC BY 4.0; already on Atlas's box), CHELSA 2.1 (CC BY 4.0), Daymet (open, cite) | no ShareAlike, but the climate would differ from Atlas's (Steve stayed on WorldClim 10/06) |

Recommendation: keep WorldClim, so Atlas and Vision share one climate, and publish `v1-env`
as its own Zenodo record under CC BY-SA 4.0, linked from `v1-cc`. A Zenodo record carries one
licence, and Atlas's own archives are already CC BY-SA 4.0.

## 3. Leakage

Atlas fits on every validated .org record with coordinates at its pull. v1 `val` is the
newest 8 weeks of validations before the cut, and `v1-test` is validated after it. Three
leaks follow:

1. **`val` records are in the fit.** The map is high at each record's own cell, and a
   range-extension record brings its own 500 km accessible area. That inflates the
   300–1,500 km band, the very band the rule judges.
2. **Any nightly after the cut may include `v1-test`.**
3. **Grades (strong or not) used `val` records.** This is small and gets declared.

| option | CPU-h | cost (spot ≈ $0.02–0.03 / vCPU-h, on-demand $0.05; m7i class, us-east-2, confirm) | valid for benchmarks |
|---|---|---|---|
| A. accept the leak | 0 | $0 | **no**: inflates the far band |
| **B. leave-`val`-out "prior-only" job (`R_train`)**: pull filtered to first validation ≤ val cutoff, the main release's strong taxa, fits + AOA + prior export, no nulls | ~40–160 (Maxent ≈ 48 without nulls for all taxa, strong ≈ half; RF+XGB ≈ +50%; AOA unmeasured) | **~$1–8**, 1–3 h on 1–2 spot workers | **yes, for both `val` and `v1-test`** (train records all predate both) |
| C. full refit on v1 train (nulls, grades, AOA: a second rebuild) | 300–500 | ~$6–25 | yes, and grades are leak-free too |
| D. 5 observer-fold leave-outs | 200–800 | ~$5–40 | adds little over B (folds sit inside `val`) |

Recommendation: **B for benchmarks, the main release for serving** (as the DNA prior serves
every reference). **Decided: B (Steve, 2026-10-12).**

`R_train`, exactly:
- **Same pull as the main rebuild release (`R_main`)**, filtered; not a fresh pull. So the
  two differ only by the records removed.
- **Keep a record** when its first green validation date is on or before v1's val cutoff, or
  when it has no green validation date (v1's own `train` rule). The first green date is the
  earliest `validation_date_i` over the record's .org rows whose `validation_status_i` is
  `yes` (date part), as Vision's `records.build_records` computes `validated_on`.
- **Taxa and models:** the `(taxon, algorithm)` models that are strong in `R_main`, refitted
  with its settings, layer set, grid and seeds. No null tests; grades are taken from
  `R_main` (leak 3, declared). AOA is computed, because the prior uses it. A taxon that falls
  below its minimum sites loses its map and counts as unmapped (listed).
- **Output:** never promoted, pulled to the box, served or archived. Its prior export (ranks,
  taxa, grid) and its trained-ids list stay private.
- Check: `trained-ids(R_train) ∩ (val ∪ v1-test) = 0`, with a positive control
  (`∩ train > 0`). `atlasrange.leak_check` already refuses until those ids exist.
- B keeps the one-way rule: Vision hands Atlas a date (the val cutoff), not records.
- Atlas side: a pull filter on first validation date plus a prior-only job mode (method lane);
  the ops lane runs it.
- The environment head needs none of this. It trains on `train` only, and its layers are
  external rasters.

## 4. Serving cost on the 4 GB box

| option | RAM | disk | latency | verdict |
|---|---|---|---|---|
| read Atlas rasters per query (~400 strong 5 km maps) | low | 0.5–2.3 GB | ~0.4–4 s (400 file opens) | no |
| **precomputed 20 km table** (Atlas export, cell-major uint8, memory-mapped) | ≤ 18 MB | ~18 MB | < 1 ms (one 400-byte row; cell math already in `atlasrange.AlbersGrid`) | **yes** |
| the same at 5 km | ~290 MB | ~290 MB | < 1 ms | no: no gain at obscured places, and it eats most of the headroom |
| environment head: 5 km env table (~0.9M land cells × 55 int16, memory-mapped) + MLP (~9 MB fp16) | ~10 MB + touched pages | ~100 MB | < 5 ms | **yes** |
| for reference: served place prior | ~34 MB | | ~80 ms | |

The worst case, with everything resident, adds ~130 MB against ~0.4 GB headroom. No box
upgrade is needed.

## 5. Plan, order, owners

Nothing here blocks the v1 cut or the retrain.

| # | step | when | owner | cost |
|---|---|---|---|---|
| 1 | decisions D1–D7 | **done 2026-10-12** | Steve, via the Vision coordinator | |
| 2 | freeze the public point in v1 `private.sqlite` (D1) | **before the v1 cut** | the Vision retrain lane, which owns the cut | 0 |
| 3 | commit the experiment grid (section 1) | before any v1 result | Vision prior lane | 0 |
| 4 | cut v1 | Steve's freeze trigger | retrain lane | ~$1.50 |
| 5 | retrain on v1 `train` | after 4 | retrain lane | ~$8.50 |
| 6 | Atlas rebuild + final batch (`feat/vision-prior` export, trained-ids, vision-reader IAM) | Atlas timeline (waits on the MyCoPortal backfill resume) | Atlas operations, in the order set by the Atlas coordinator | Atlas budget |
| 6a | first-green date carried in the pull (private side column, outside the fingerprint) | **before rebuild step 1's fresh pull**, so `R_main`'s own pull has it | Atlas method | 0 |
| 7 | `R_train` job (D2) | after 4 (needs the val cutoff) and 6 | Atlas method lane builds the job mode; ops runs | ~$1–8 |
| 8 | build the `v1-env` sidecar | after 4 | Vision prior lane; Atlas layer files handed over by the ops lane (one-off copy or 12 h presigned link; no IAM change) | 0 |
| 9 | train the environment head (CPU) | after 8 | Vision prior lane | 0 |
| 10 | arms P0–P5 on `val` CV | after 5, 7, 9 | Vision prior lane | 0 |
| 11 | `v1-test` confirmation, once, all arms | when `v1-test` is cut | Vision prior lane | 0 |
| 12 | serve only an arm that passes | after 11 | Vision coordinator merges and deploys; Atlas deploys nothing for this | 0 |

## 6. Decisions (Steve, 2026-10-12, relayed by the Vision coordinator)

| | question | decision |
|---|---|---|
| D1 | Freeze each record's public point in v1 `private.sqlite` before the cut? | **yes**. Columns in section 2, agreed with the retrain lane, which owns the cut |
| D2 | Run the leave-`val`-out Atlas prior job `R_train` (~40–160 CPU-h, ~$1–8)? | **yes**. Spec in section 3; the Atlas ask goes through the Vision coordinator to the Atlas coordinator |
| D3 | Public env table licence | **WorldClim**; `v1-env` is its own Zenodo record under CC BY-SA 4.0 |
| D4 | Include the environment-head arms (P4, P5)? | **yes** |
| D5 | Where the 300–1,500 km band is judged | **`val` CV (n ≈ 680)**, with `v1-test` as a sign check |
| D6 | Daymet recent-weather features | **later** (after v1) |
| D7 | Env resolution | **5 km** (Atlas `draft`, 2.5′) |

Read access (Steve, 2026-10-12): **no IAM change.** The vision-reader policy stays limited to
`private/trained-ids/*/releases/*`. The Atlas ops lane hands over the `R_main` layer files and
manifest and `R_train`'s private prior export as one-off copies or 12 h presigned links (the
`trained.R` pattern).
