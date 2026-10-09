# MycoMap Vision plan

Goal: a public fungal identifier that uses every photo of a record, is trained
only on DNA-validated MycoMap records, and gives a prediction and confidence at
every rank, with advice on how to improve an ID. It runs as its own platform;
mycomap.org links to it and calls its API when needed.

Decisions to date are recorded in the project memory and CLAUDE.md. This file is
the working plan; update it as phases land.

## Phase 0: data and a first honest number (in progress)

- [x] Export green records from mycomap.org (read-only). 164,201 records,
      159,353 in North America, 146 label conflicts held out.
- [x] iNat metadata: 155,858 observations, 593,224 photos with owner and license
      (16% all rights reserved, 1,251 photographers to ask).
- [ ] Download large (1024 px) photos straight into S3 (`mv aws-launch-downloader`,
      ~250 GB, ~2 days for the capped all-rights-reserved part). Waiting on the
      one-time AWS setup in `deploy/aws/README.md`.
- [x] Embedding pipeline (`mv embed`), any timm or open_clip backbone; tried so far: BioCLIP 2
      (~95 photos/s on the laptop 4070), DINOv2-B (~64/s), DINOv2-L at 518 px (~13/s).
- [x] Evaluation harness (`mv compare`): identify the newest 28 days of green
      records from older ones, by nearest DNA-verified specimen; top-1/top-5 at
      family, genus, species, split by reference count (novel, 1, 2, 3-5, 6-30, 31+);
      all photos vs first photo only.
- [ ] iNat computer-vision baseline on the same test records (`mv inat-baseline`:
      best score across a record's photos, photo-only and with iNat's location
      model; needs a 24-hour iNat token).
- [ ] First comparison report.
- [x] Web app and API (`mv serve`) with Identify, Models (scoreboard), Data, How it works;
      confidence calibrated per rank from each model's latest comparison.
- [x] Name spellings (2026-09-29, Steve's decision). The same temporary-code name reaches
      us written several ways (`Inocybe sp. 'PNW18'`, `Inocybe PNW18`, `Inocybe "sp-PNW18"`),
      and each spelling used to count as its own species. Now spellings that differ only in
      how they are written (quotes, `sp.`, spacing, capitals, odd characters, a hyphen) are
      one species, under the house style `Genus sp. 'CODE'`. What needs a person is left
      alone and only reported: a code also in use as a plain name (`Craterellus
      neotubaeformis` / `Craterellus sp. 'neotubaeformis'`), IN7 / IN07, a described code
      (`'AZ brown 01'`). The rule is mycomap.org's (`services/nameVariants.ts`), copied in
      `names.py` and held to the same test file; the manifest keeps every name as .org
      spells it, and labels are worked out when records are loaded. `mv name-spellings`
      lists both kinds. On the manifest of 2026-09-29: 20,609 names become 19,843 labels;
      1,035 names merged (2,546 records re-labelled), 87 left for a person (1,720 records).
      Among records with photos, the newest 28 days (1,973 records) go from 203 species
      with no reference record to 193. Comparisons made before this can't be given an iNat
      baseline any more (their record set changed): run `mv compare` again.
- [x] Family from iNaturalist (2026-09-30, Steve's decision). 30,648 records had a blank
      family and 227 genera carried several families on .org. Now family (and order,
      class, phylum) comes from iNat's taxonomy, one answer per genus, so every record of a
      genus has the same family (`taxonomy.py`). `mv fetch-taxonomy` asks iNat for each
      genus the records use, within Fungi only (a plant or bee genus of the same name is
      ignored), at 1 request/s, and keeps every answer in `data/taxonomy/inat_genera.sqlite`
      (never the manifest; resumable, `--refresh`, `--older-than DAYS`; a release ships
      it). A genus in doubt keeps .org's family and is listed in
      `data/reports/taxonomy-doubts.csv` (`mv taxonomy`). First run, 2026-09-30: 2,050 genus
      names, 2,743 requests in ~46 min; 1,428 applied. Listed: 421 where iNat's family
      differs from most .org records (applied: Clitocybe to Clitocybaceae, Hygrocybe to
      Hygrophoraceae, Galerina to Hymenogastraceae, Cantharellus to Hydnaceae...), 251 with
      no fungal genus of that name on iNat (family names in the genus column, slime moulds,
      bees, plants), 85 inactive (with iNat's replacement, e.g. Marasmiellus to
      Collybiopsis), 72 iNat puts in no family, 32 not a Latin genus name, 87 one-word names
      iNat has at no rank above genus (mostly subgenera and sections: Cyanula, Dermocybe).
      Records: 27,788 blank families filled, 33,326 changed.
- [x] One-word names at genus level only (2026-09-30, Steve's decision). A name of one
      word ("Russula", "Agaricales", "Fungi") used to be a species of its own. Now such a
      record has no species label, adds no species class (in the index, the trained heads
      and fine-tuning it is a genus-level group), is never a species candidate in a result,
      and isn't scored at species; it counts at genus when the word is a genus (iNat says
      so, or, when iNat can't, the genus column agrees) and at family. A record with no
      label left at family, genus or species is left out. On the manifest of 2026-09-29:
      542 one-word names on 2,997 records (2,616 from iNat; 1,968 usable: North America,
      photos, no conflict). 377 names (1,570 records) are genera, 69 (188 records) name
      only a family, 96 (1,239 records: Unknown 506, Fungi 344, Agaricales 97...) have no
      label at any rank Vision scores and are left out.

## Models: modular, side by side, on a scoreboard

A "model" here is a **backbone** (frozen image model -> vector per photo) plus a
**method** (how photo vectors become species scores). Both are swappable:

- Backbones: `models.py`. Any timm or open_clip model works by spec with no code
  (`mv embed --backbone timm:<name>`); favourites get a short alias. Other
  sources (Hugging Face transformers, ONNX, an API) are one loader class each.
- Methods: `evaluate.METHODS`. Now `nearest` (best-matching DNA-verified
  specimen) and `species-mean` (species average vector); trained heads,
  multi-photo attention and the range prior plug in the same way.
- `mv compare --backbones a,b,c --methods nearest,species-mean` scores every
  pair on exactly the same test and reference photos (only photos every backbone
  has embedded), and saves each run to the scoreboard (`eval_runs` in the
  manifest, with the record-set hash and code version). `mv scoreboard` lists
  them; the frontend shows them and runs models side by side on your photos.

Screening policy: try many cheaply, fine-tune few. Frozen screening costs one
embedding pass over a fixed screening set (minutes to an hour on the laptop
GPU) and seconds to score, so 10-15 candidates is fine. Candidates to screen
(aliases in `mv models`): BioCLIP 2, DINOv2 B/L (G too if it fits), DINOv3 B/L
(in timm; Meta's DINOv3 licence needs a read before any public release), SigLIP 2
L/384 (open_clip), EVA-02 L/448, ConvNeXt V2 L as a non-transformer contrast,
plus iNat's own model as an external baseline on the same test records. (timm
has no iNat-trained checkpoints, so that idea is dropped.) The best 2-3 go on
to fine-tuning in phase 1.

## Frontend (started 2026-09-28, grows with every phase)

A web app to try the identifier as it develops. It follows mycomap.org's design
so moving between the two sites feels like one site: the same colour tokens
(myco green and brown), sticky white header with the MycoMap logo, grey nav
links that turn green, brown gradient footer, shadcn "new-york" components and
system/Inter type. Species link to `mycomap.org/species/<name>`, records to iNat.

- `web/`: Vite + React + TypeScript + Tailwind (the .org stack), talking to
- `mycomap_vision/api.py`: FastAPI over the manifest and embeddings.

Pages:
- **Identify**: upload several photos (plus optional place and date), get the
  rank ladder (family, genus, species) with confidence, the closest
  DNA-verified specimens with their photos, and hints on how to improve the ID.
- **Data**: records, photos, licenses, download and embedding progress, names by
  number of DNA-verified records.
- **Models**: the scoreboard; each comparison by rank and by reference count.
- Identify can run two or more models side by side on the same photos.
- **How it works**: plain-language method and its limits.

Later: photo view tags (cap, underside, stem, habitat) with "add an underside
photo" hints, map for location, account-free sharing of a result, and the
production deploy at `vision.mycomap.org` on its own Lightsail box.

## Findings so far (8,000-record sample, 2026-09-28)

Small test sets (68-91 records): read as direction, not precision.

- BioCLIP 2 is far ahead of DINOv2-B (species 25% vs 10%, genus 62% vs 31%).
- Full screen of 8 frozen backbones on the same 68 test records (comparison
  20260928-235515-60274d, best method each, species / genus / family top-1):
  iNat CV with location 29.4 / 67.7 / 70.6; iNat photo-only 27.9 / 64.7 / 67.7;
  BioCLIP 2 25.0 / 61.8 / 71.4; EVA-02 L 13.2 / 33.8 / 44.6; DINOv2-L 10.3 / 36.8 /
  44.6; DINOv2-B 10.3 / 30.9 / 41.1; ConvNeXt V2 L 8.8 / 29.4 / 32.1; SigLIP 2 L
  5.9 / 19.1 / 28.6; DINOv3-B 5.9 / 11.8 / 21.4; DINOv3-L 4.4 / 14.7 / 21.4.
  Laptop speeds (medium photos): DINOv3-B 169/s, DINOv3-L 57/s, BioCLIP 2 ~95/s,
  DINOv2-L 20/s, ConvNeXt V2 L 20/s, SigLIP 2 17/s, EVA-02 L 17/s.
  Nothing generic comes near BioCLIP 2, which was trained on the tree of life.
  Centring the vectors on the reference mean didn't change any of this.
- DINOv3 wasn't tested on equal terms at first: timm runs it at 256 px with
  average pooling, while DINOv2 runs at 518 px on its class token. At 512 px on
  the class token (dinov3-l16-512; comparison 20260929-010339-60274d, same 68
  records) DINOv3-L went from 4.4 / 14.7 to 16.2 / 42.6 species / genus, the best
  general-purpose model, still well behind BioCLIP 2 (25.0 / 61.8). DINOv3-B at
  512: 14.7 / 38.2. About 20 photos/s each on the laptop.
- Fine-tuning BioCLIP 2 helps even on the sample (bioclip-2-ft-sample: last 4
  blocks, 2 passes over the 30,252 reference photos of 8,003 records, 4,317
  species, 23 min on the laptop). Same 91 test records (comparison
  20260929-012938-3f9ac9), species / genus / family top-1, best method each:
  iNat with location 29.7 / 69.2 / 73.6; iNat photo-only 28.6 / 67.0 / 71.4;
  fine-tuned BioCLIP 2 27.5 / 63.7 / 72.0; frozen BioCLIP 2 24.2 / 60.4 / 68.0.
  3 records of 91 separate fine-tuned from frozen at species, so this is a
  direction, not a result; the full-data run decides. The trained heads still
  lose on top of it (linear 9.9%, hybrid 17.6%).
- Large (1024 px) vs medium (500 px) photos make no measurable difference on the
  sample. The four contenders were re-embedded at large (the medium vectors are in
  data/embeddings-archive/<model>-medium) and scored in comparison
  20260929-031844-85e107: 77 test records against 6,805, fewer than before because
  the 5,039 all-rights-reserved sample photos have no large copy yet. On exactly
  those records and photos, species / genus / family, nearest specimen, medium ->
  large: BioCLIP 2 23.4/58.4/65.1 -> 26.0/58.4/68.2; fine-tuned 26.0/59.7/68.2 ->
  26.0/58.4/69.8; DINOv3-L 512 10.4/37.7/41.3 -> 7.8/36.4/42.9; EVA-02 L
  9.1/32.5/46.0 -> 10.4/29.9/38.1. Every change is 1-2 records of 77, in both
  directions. iNat on the same 77: 26.0 / 66.2 / 70.1 photo-only, 27.3 / 66.2 /
  70.1 with location, so frozen and fine-tuned BioCLIP 2 tie iNat at species here
  and trail it by ~8 points at genus. On this set the fine-tuned model ties the
  frozen one at species (it led by 3 records on the 91-record set), with better
  family and first-photo scores (24.7 vs 20.8). The sample can't separate these;
  the full data (~5,000 test records) will. Large photos stay the default for the
  full run: they cost nothing extra there and keep detail for crops later.
- iNat's own model on the same 68 records: 27.9% species photo-only, 29.4% with
  location; BioCLIP 2 (species average), frozen and with only 5% of the reference
  data, reached 25.0% species and a higher family score (71.4% vs 70.6%).
- Trained heads lose on this sample: nearest specimen 24% species; plain linear
  classifier 13%; balanced-softmax linear 3%; hybrid 14-16%. Species here have a
  median of 4 photos, too few for a classifier to add much. Balanced softmax was
  the wrong default (test records arrive at real frequencies), so plain training
  is now the default. Retest heads on the full data, where ~1,200 species have
  30+ records, on the AWS GPU trainer.
- The range-and-season score doesn't help yet. On nearest specimen (82 of 91 test
  records have a location), weight 0 / 0.1 / 0.25 / 0.5 / 1 gave species 24.2 /
  23.1 / 23.1 / 22.0 / 23.1% and genus 59.3 / 59.3 / 56.0 / 51.6 / 47.2%. With
  1-4 records per species the ranges are too thin: a right species found 300 km
  from its few records is penalised. Kept as experimental methods (+prior), not a
  default; retest on the full data, where iNat's own location model adds ~1.5
  points for iNat.

## Phase 1: a better identifier

- Photo view tagging: a cheap LLM labels a seed set, a small head on the
  embeddings does the rest.
- Learned multi-photo weighting (attention over photo embeddings with view tags).
- Fine-tune the best backbone on well-sampled species (metric learning), keep
  nearest-specimen lookup for the long tail (43% of names have one record).
  Built 2026-09-28 (`finetune.py`; `mv finetune` on the laptop, `--finetune` on the
  AWS trainer): the last 4 of 24 blocks train, the rest stay frozen; a cosine
  classifier at species, genus and family starts from the frozen model's class
  averages; photos drawn with weight 1/sqrt(species photos); no hue or saturation
  augmentation (colour identifies fungi). It trains only on the comparison's
  reference records, saves the date it trained through, and `mv compare`
  refuses to score it on anything up to that date. The result is used like any
  backbone (`<base>-ft-<run>`), so nearest, species-mean and the heads all run on
  it, next to the frozen base.
- Hierarchical, calibrated confidence per rank (so 80% means right 80% of the time).
- Crops of the fruiting body from the large photos.

## Phase 2: range and season

- Per-taxon range and phenology from DNA records, corrected for DNA sampling
  effort (all DNA records as the background). Shared with mycomap.org/conservation.
- A separate, capped prior multiplied into the photo score; both shown.

### Occurrence prior from iNat counts (`+occ`, built 2026-10-08)

The DNA-record prior (`+prior`) moved the 99-record held-out pilot only 45 -> 47
of 95: in 22 of 95 the true species had no DNA record within 300 km, so DNA
records are too sparse to draw ranges. Steve's decisions: use iNat occurrence
counts (open data; GBIF later), and treat "out of range" as a wide berth only.

- `mv build-occurrence --observations observations.csv.gz --taxa taxa.csv.gz`
  (local files; `occurrence.py`) keeps Kingdom Fungi by ancestry (lichens
  included; slime molds with `--with-slime-molds`), research and needs-ID
  grades, coordinates inside North America plus 5 degrees, accuracy no worse
  than 25 km. It counts observer-days, not observations (Steve, 2026-10-08: ten
  photos of one species on one walk are one find): distinct (observer, day) per
  0.5-degree cell, per species (a variety also counts for its species) and per
  genus, each deduplicated at its own level, for all fungi together (effort), and
  per 10-degree latitude band and week (season). On the 2026-09-27 export: 283M
  rows read, 9.07M North American fungal observations kept, 21,710 taxa with
  observations, about 12 minutes on the laptop.
- A record is never scored with its own iNat observation counted. The build
  writes a leave-one-out index beside the store (every counted observation's
  uuid hash, what it counted for, and at which levels it alone made its
  observer-day), and a record scored with its uuid
  (`Context.uuid`: comparisons, tuning, advance predictions) has its own find
  taken back out: its observer-day goes only where no other counted observation
  shares it. `--exclude-uuids` still leaves a list out entirely
  (`mv occurrence-exclusions` writes the manifest's).
- Names (`OccurrenceStore.resolve`): the active iNat taxon of exactly that name
  ("var." ignored); a one-word label is its genus. iNat's export records no
  replacement for an inactive name, so an inactive, provisional or unknown name
  gets no species taxon, only its genus. The single active species of an inactive
  name's epithet and family is kept as a flagged guess only (it maps Morchella
  conica to Verpa conica): never for out of range, and for place and season only
  with `epithet_guesses` on (off by default). `mv occurrence-names`, and every
  tuning run (`params-names.csv`), list how every label maps.
- `OccurrencePrior` (`occprior.py`), per species: (a) out of range, a strong
  penalty (6 nats by default), only when it has at least 20 occurrences (iNat
  plus its own DNA records) and none within 1,500 km; a DNA record nearby always
  vetoes it. When a species reaches the 20 only with its DNA records (too few on
  iNat, or none: provisional names), its absence counts only where at least
  `min_dna_effort` (500; tuned on dev) DNA records of any species lie within the
  radius: DNA sampling is uneven (about 1,000 records within 1,500 km of Mexico
  City, 52,700 of Seattle). The same rule at genus level (`genus_rule`, on by default; the tuning
  report shows dev with and without it); never where fewer than 50 fungi of any
  kind were observed within the radius, nor off the map. (b) density: log of the
  species' share near here (150 km kernel) over all fungi's, shrunk to genus by
  20 observations, capped at log 5, weight 0.5. (c) season: the same by week
  within nearby latitude bands, capped at log 5, weight 0.5. (d) unknown to iNat:
  genus for (b) and (c), DNA records for (a), else neutral.
- Range sources are pluggable (`RangeSource`: out of range per radius, a place
  score, a season score; weights and caps applied the same for all). MycoMap
  Atlas is the second source (`atlasrange.py`, a stub against the export the
  Atlas lane proposed: Albers 20 km cells, rank 0-100 within each taxon's
  accessible area; f(rank) learned on dev, never raw ranks across taxa; unlisted
  taxa neutral; iNat for the rest and for season). Not tuned or evaluated until
  Atlas's rebuilt release exists and says what its maps were trained on.
- Methods `nearest+occ`, `species-mean+occ`, `linear+occ`, `hybrid+occ`, offered
  only where the store exists. The store is not in releases and not on the box:
  `+occ` stays on the laptop until dev tuning shows a gain and Steve approves. The existing `+prior` methods are unchanged
  except for confidence.
- Confidence: nearest+prior, species-mean+prior and their +occ versions stated
  ~100% for almost every answer because identify fell back to the cosine
  temperature (0.02). They now carry their own (1.9, from comparison 4ef7b0's
  nearest calibration, or the tuned one); a comparison's calibration still wins.
  linear and hybrid with a prior keep the old behaviour until theirs is measured.
- Every scored record's own find must come out: `mv compare` refuses a +occ run
  whose store counts its test records with no leave-one-out index, warns and
  marks the result for records with no uuid, and records the prior's values and
  their provenance in the scoreboard row; advance predictions skip a candidate
  whose own find can't be taken out.
- `mv tune-occurrence --records dev.csv --scores scores.npz` grid-searches
  radius, penalty, weights and photo temperature for species top-1 on a
  validation set (or `--records comparison:<id>`); the penalty grid includes
  outright exclusion (50 nats) next to the soft values, and 6 nats stays the
  untuned default. It fits the confidence
  temperature per rank, and writes `data/occurrence/params.json` with its
  provenance. `--evaluate-only` scores another set (test.csv) with the saved
  values, temperatures included (nothing is refitted on it); it refuses to
  search on a benchmark's test split (split.json beside the CSV, or overlapping
  its test.csv) without `--allow-test`. `--report-without ids.csv` also reports results without a list (the
  repeat finds). It refuses records registered in `benchmark_holdouts` (except
  split = dev) or in a `--sealed` file. The 2026-10-08 held-out set is no longer
  sealed (Steve): tune on its dev split, check on test.

## Training on an AWS GPU instance (Steve, 2026-09-28)

Embedding the full photo set and training the heads move off the laptop to a GPU
instance started on demand, like the downloader: `mv aws-launch-trainer` starts
it, it reads the large photos straight from S3, embeds them with the chosen
backbones, trains and compares every method, writes embeddings, trained weights
and the scoreboard back to S3, and shuts itself down (with a backstop). The
laptop pulls the results with a merge, as for downloads. The linear classifier
already trains on a CUDA GPU when one is present.

Setup needed first (Steve):
- A GPU quota: AWS console -> Service Quotas -> Amazon EC2 -> "Running On-Demand
  G and VT instances" (and "All G and VT Spot Instance Requests" for spot), e.g.
  8 vCPUs in us-east-2. New accounts often start at 0.
- The ops policy gains read access to the Deep Learning AMI's public parameter
  (/aws/service/deeplearning/ami/...), and spot requests if we use spot.
- Budget alarm in place (a g5/g6.xlarge is roughly a dollar an hour on demand,
  less on spot; one embedding pass of the 593k large photos is 3-4.5 h for
  BioCLIP 2, 9-13 h for DINOv3-L at 512 px).

Built 2026-09-28 (`mv aws-launch-trainer`, `mv aws-pull-trainer`; how to run it in
deploy/aws/README.md): g6.2xlarge from the Deep Learning Base GPU AMI, each run in
its own `runs/<run>/` folder (never the downloader's manifest), each backbone
uploaded as soon as it is embedded, `result.json` last. Waiting on the GPU quota
and the ops-policy update; first run after the download completes.

Fixed before the first run (Steve, 2026-09-30; branch fix/trainer-before-first-run):
- The first run is BioCLIP 2 + its fine-tune alone, with a 24 h limit (now the
  launcher's defaults). The earlier plan (BioCLIP 2 + DINOv3-L 512 + fine-tune,
  14 h) needed ~25 h (19-34) and ran DINOv3 before the fine-tune, so the limit
  would have cut the fine-tune. Stages now run as: each backbone to fine-tune,
  its fine-tune, the fine-tuned model's embedding, then any other backbone.
  The launcher estimates the hours (laptop speeds at large photos x 1.5 for the
  L4: about 16 h for the first run, 29 h with DINOv3-L added) and refuses a run
  that doesn't fit `--max-hours` unless `--allow-over-time`.
- Each finished stage is uploaded at once with a progress.json; the job stops
  itself 45 min before the limit and ends `stopped`; `aws-pull-trainer` brings
  home the finished stages of a stopped or killed run and names what's missing.
  A backbone the run didn't finish, or whose copy doesn't match the run's index,
  never replaces local embeddings.
- Unreadable photos are skipped, listed with the reason (runs/<run>/skipped/) and
  counted; a backbone fails only when more than 1% of its photos are unreadable.
- The run records the exact commit it ran (launch refuses a dirty tree or an
  unpushed commit unless allowed), and the instance installs only pinned, hashed
  packages (requirements/trainer.txt, made by
  deploy/aws/lock_instance_requirements.py for Linux x86_64 / CPython 3.11).
- After the first run, replace the laptop-measured rates in trainer.py with the
  L4's own (progress.json has each stage's speed); DINOv3-L 512 then gets a run of
  its own if it still earns one.

Spot (Steve, 2026-09-30; branch feat/spot-trainer-weekly-taxonomy): the On-Demand
G quota is 0 and its increase is pending, as is the Spot one (filed 9/28), so the
trainer can run on either. `mv aws-launch-trainer --spot [--spot-max-price]` asks
for a one-time Spot g6.2xlarge that AWS terminates when it takes it back. The job
polls the instance metadata (IMDSv2) every 5 s for the two-minute notice; on it,
the stage under way is dropped and the run ends `interrupted` (the finished stages
are already in S3), which the pull treats like `stopped`. `--resume <run>` relaunches
a run under its own id, restores its finished stages on the new instance (checked
against the run's index) and runs only the rest; the estimate counts only those.
Capacity, quota and price refusals are explained. The ops policy gains the Spot
request resource and the Spot service-linked role (the role already exists in the
account, made 2026-09-29). The downloader stays On-Demand. The weekly `mv refresh`
now also asks iNat about genera new since the last taxonomy lookup (at most 20 min,
never failing the refresh; `--no-taxonomy`).

## Phase 3: the platform

- Own Lightsail instance (`vision.mycomap.org`), photos and models in S3, weekly
  job on a spot GPU: new green records → score them first (the weekly test),
  then add them to the reference set and retrain the small heads.
- Prospective test on arrival (Steve, 2026-09-28): every new batch is scored by
  every model, iNat's computer vision included (`mv inat-baseline`), *before*
  its DNA barcode result exists; when the barcodes come back, the predictions are
  validated instantly. Nothing can have seen the answer, so it is the fairest
  test of all, and it covers provisional names, which iNat has begun to carry.
- Model registry with an evaluation report per release; a release goes live only
  if it did not get worse.
- API for mycomap.org (FungAI, Validate Records, record pages).

Hosting kit built 2026-09-29 (docs/deploy.md, deploy/lightsail/): 4 GB Lightsail
box (Steve's choice; 8 GB if the full index won't fit), CPU inference only, all GPU
work stays on the self-terminating instances. The site reads **releases** from S3
(`mv release` / `mv pull-release`, checksummed, previous kept for rollback) through
a read-only IAM user. Sign-in reuses mycomap.org's bridge (Steve: anyone with a
mycomap.org account before launch; later maybe identify-only); the box holds only
the public key.

### Serving speed (roadmap, Steve 2026-09-29)

The box only answers; **all training runs on the EC2 GPU instance** (the box has
MV_FIT_ON_DEMAND=0 and offers only methods whose training ships in a release).
Today: nearest specimen, ~2-3 s a photo on the 4 GB box, fine while use is light.
Lightsail CPUs are burstable: sustained use (or any heavy job) spends the credits
and throttles the box to ~1/5 speed, so real traffic means moving the answering to
a non-burstable compute instance.

- [ ] **8-bit image tower, again with the full-set model.** Re-export the model the
  EC2 run produces (fine-tuned or not) to ONNX and quantize it; also try a gentler
  int8 that keeps the attention layers at full precision, to keep more answers
  identical. Switch only if accuracy holds at species, genus and family.
  First try (2026-09-29, frozen BioCLIP 2, sample index, 2,000 photos, own record
  left out): species 14.1% vs 14.5% full, genus 47.5 vs 48.2, family 47.5 vs 47.4;
  but only 74 / 81 / 85% of answers identical, top confidence moved by 0.025
  typically (0.11 at the 95th percentile); 328 ms a photo on 8 laptop threads vs
  ~1.4 s full precision on 4. Kept full precision. The ONNX export fused LayerNorm
  and GELU but not attention (open_clip's attention doesn't match ORT's patterns).
- [ ] Re-time on the box once its CPU credits have recovered (the 9/29 box timings
  were taken while throttled and don't count).
- [ ] When traffic justifies it: answering on a non-burstable compute instance
  (e.g. c7i with AMX), or a GPU, for well under a second a photo.
