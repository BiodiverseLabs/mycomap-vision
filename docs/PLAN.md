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
  specimen) and `species-mean` (species average vector), and `nearest+mean`
  (experimental: the two best specimen matches blended with the species average;
  see "Is nearest biased toward species with many reference photos?"); trained heads,
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

## Held-out benchmarks (2026-10-08)

`heldout-2026-10-08` is 13,145 North American iNat records that are DNA-validated
(green) on legacy mycomap.com but that Vision never saw: a broken .com -> .org sync
kept them out of its labels, so they are neither in the full-run fine-tune
(`bioclip-2-ft-20261007-165400`, trained through 2026-09-07) nor in its reference
index. Steve (2026-10-08): it is a development benchmark, not the paper's test set.
Its records will join training and the reference index after a relabel and retrain;
first it measures Vision as it is (the "before" number), and it is what dev tuning
uses. The paper gets a fresh ~1,000-record set, frozen later as a sealed benchmark.
Inputs (read-only prod queries) are in `data/benchmarks/heldout-2026-10-08/`: pool.csv
(ids sha256 174bfabab103fe74...), the source snapshot, dev.csv / test.csv / split.json.

- Answer key: the observation's name, the .com record title (field_483). pool.csv's
  com_name is .com's index name, which lags the title on some records; a titles
  export (record_id, index_name, title) with linked43.csv makes the title the answer
  where they differ, and the index name stays for the audit. Names are compared by
  Vision's labels with writing set aside (code formats, `sp-`, a missing `var.`).
  A one-word title is scored at genus and above only and flagged in the label
  audit as possibly stale (a .com refresh). `mv heldout freeze` runs again on a new
  snapshot of the same ids and takes the new names (logged in `heldout_freezes`).
- Split: dev (3,000: the 100-record pilot and 2,900 at random) for tuning and any
  exploring; test (10,145). On a sealed benchmark (`freeze --holdout`) the test
  split is the paper's number: `mv heldout report --split test` says so and every
  such report is recorded (`heldout_test_looks`). On a development benchmark test
  is not sealed.
- Commands (`heldout.py`, `heldout_report.py`): `mv heldout freeze`; `fetch` (iNat
  details, also into `inat_observations` with the uuid, and LARGE photos into the
  benchmark's own folder; resumable, within iNat's limits shared with the reference
  downloader); `predict` (embeds into the benchmark's own index, keeps every photo
  vector, identifies each record with the served index, nearest and nearest+prior,
  iNat's public place by default; saves the top 5 per rank and the full identify
  result, and `--scores-out` writes photo scores in occtune's scored-set format);
  `inat` (iNat's computer vision on the 2,000-record subsample, judged by taxon id);
  `report` (top-1/top-5 per rank with Wilson 95% intervals, McNemar paired tests,
  calibration, likely-set coverage when the results carry it, breakdowns by
  reference depth, unseen species, photos, east/west of -100, same observer and
  day in the reference, provisional vs formal names; the label audit; JSON and a
  per-record CSV, no coordinates).
- Standard summary (Steve, 2026-10-09; `heldout_summary.py`), printed before the JSON
  by every `mv heldout report`: per Vision model, top 1 / 3 / 5 / 10 for species
  strict, species s.l., species complex (beta), genus strict and genus s.l. (s.l. and
  complex from name_equiv once feat/name-equivalence is merged), and species top 1 /
  5 by the true species' reference records (0, 1-4, 5-19, 20-99, 100+); then iNat CV
  and the Vision models on the records all of them answered, or "not run". Every
  table gives its n. Predict keeps answers ten deep from now on; answers stored five
  deep show "n.s." at top 10.
- Before and after: each answer keeps the hash of the reference it was made against
  (its records and their labels), so a run after a relabel sits beside the run
  before it; `report --reference-hash` scores the earlier one.
- Leakage guards. A sealed benchmark's freeze refuses any id the records table
  holds and lists every id in `benchmark_holdouts` (`mv holdout add|list|release`).
  A held-out record is never stored by an export or the nightly update
  (`records.save_records`, `nightly.changes`), never loaded into a reference set,
  comparison, fine-tune or served index (`evaluate.load_records`), and gets no
  reference photo downloaded or embedded (`photos.pending_photos`,
  `embed.photos_to_embed`); a release and a trainer run are refused while the
  records table holds one (`release.publish`, `aws.check_trainer_request`,
  `trainer.run_job`). The server box's nightly update leaves them out once it runs
  a release made after the freeze (the release's manifest carries the list). The
  benchmark's photo copies, vectors and answers live in its own tables and folder,
  never in `photo_copies` or `embeddings`, and no manifest copy that leaves the
  laptop (a release, a trainer or downloader run) carries those tables
  (`manifest.shippable_snapshot`); `predict` refuses a reference index that holds a
  sealed benchmark's record, and skips a record already in the reference or whose
  photo is a reference photo.
- "Never seen" is checked against everything Vision holds, not only today's records:
  freeze notes ids with iNat details, photo copies or embeddings (`was_reference`;
  84 of heldout-2026-10-08's records were Vision records in the 9/28 export, dropped
  by 10/6), a sealed freeze refuses them, and the report breaks scores down by it.
- Runbook for a sealed set (Steve, 2026-10-09). The server box learns which records
  are held out only from a release's manifest. So: deploy this code to the box, then
  right after `mv heldout freeze --holdout` (or `mv holdout add`) cut a release
  (`mv release ... --make-current`, then `mv pull-release` and a restart on the box).
  Until then the box's nightly update can still add the records. Follow-up: ship the
  list on its own (e.g. beside the release, read by the nightly update) so sealing
  needs no release.
- Follow-up: record each fine-tune's trained record ids (finetune.py: the reference
  records of its comparison) beside its weights, so `mv heldout predict` can check a
  benchmark's records against what the model trained on, not only against the
  reference index it answers from.

## Is nearest biased toward species with many reference photos? (2026-10-09, exp/depth-bias)

Why species accuracy is ~40% on held-out records (Steve, 2026-10-09). An audit of 99
held-out dev records (fine-tuned BioCLIP 2, `nearest`) found accuracy rising with the
true species' reference records (0 refs 0/4; 1-4 3/16; 5-19 2/23; 20-99 22/38; 100+
13/18), and in 37 of 55 misses the winning species had more reference records than the
truth. The suspect: `nearest` scores a species by its best match among N photos, a
maximum that grows with N, so deep species get more chances.

Tested on comparison 20261008-012435-4ef7b0 (fine-tuned bioclip-2-ft-20261007-165400;
newest 28 days, 1,152 test records against 152,915; record set reproduced exactly).
Every variant re-scores the same embeddings, with no retraining. Species top-1, by
the true species' reference records (test records 141 / 215 / 296 / 364 / 105):

| scoring | species | 0 | 1-4 | 5-19 | 20-99 | 100+ | pairs vs nearest |
|---|---|---|---|---|---|---|---|
| nearest | 34.5 | 0 | 12.6 | 31.8 | 54.1 | 65.7 | |
| best - 0.005 ln N photos | 33.9 | 0 | 13.0 | 33.4 | 51.9 | 61.0 | +10/-17 |
| best - 0.01 ln N photos | 33.7 | 0 | 14.4 | 34.5 | 51.4 | 55.2 | +24/-33 |
| best - 0.02 ln N photos | 29.1 | 0 | 16.7 | 33.8 | 41.2 | 38.1 | +39/-100 |
| best + 0.005 ln N photos | 35.0 | 0 | 12.6 | 29.4 | 56.0 | 70.5 | +17/-12 |
| best - expected best for other species' photos (x0.25) | 33.8 | 0 | 14.9 | 34.5 | 51.4 | 55.2 | +25/-33 |
| per-species z-score of best | 12.1 | 0 | 18.6 | 16.9 | 11.8 | 2.9 | +35/-286 |
| hubness (CSLS-style, top-50 x0.25) | 34.3 | 0 | 15.3 | 33.8 | 51.6 | 60.0 | +30/-33 |
| per-depth offsets, fitted (2-fold by observer) | 33.9 | 0 | 13.5 | 34.8 | 49.2 | 65.7 | +18/-25 |
| mean of top-2 photos | 36.0 | 0 | 10.7 | 32.8 | 56.6 | 73.3 | +36/-20 |
| mean of top-5 photos | 35.3 | 0 | 8.8 | 29.7 | 57.1 | 77.1 | +53/-44 |
| top-2 over distinct records | 35.6 | 0 | 9.3 | 31.8 | 57.1 | 73.3 | +43/-31 |
| species-mean | 34.3 | 0 | 13.5 | 39.2 | 50.8 | 52.4 | +87/-89 |
| 0.5 nearest + 0.5 species-mean | 37.9 | 0 | 14.0 | 40.9 | 57.4 | 61.9 | +70/-32 |
| **0.6 top-2 + 0.4 species-mean (`nearest+mean`)** | **38.4** | 0 | **13.0** | **39.9** | **58.5** | **67.6** | **+76/-33** |

(N is reference photos; using reference records instead gives the same picture. The
"expected best" and z-score use each species' best match to 6,000 random reference
photos of other species; hubness subtracts the mean of its 50 strongest such matches.)

What this says:
- **Taking depth out of the score loses.** Every penalty on depth, fixed (b ln N),
  measured (expected best match, z-score, hubness) or fitted per depth band, trades a
  few sparse-species records for many more deep ones, overall and in cross-validation.
  A small depth *bonus* is neutral. Test records arrive at real frequencies, so a deep
  species really is more likely, and nearest's implicit lean toward depth is about as
  big as it should be. The audit's "the winner was deeper" is mostly a base rate: the
  truths that get missed are sparse, so nearly any rival is deeper. Among records
  nearest gets right, the runner-up is deeper than the truth only 42% of the time.
- **Most sparse-species misses are not near misses.** For truths with 1-4 reference
  records, nearest ranks the truth #1 for 12%, #2-5 for 14%, #6-20 for 24% and below
  #20 for 46%. No re-scoring of the same similarities can reach those; the photos we
  hold of those species don't look like the query (other angles, stages, observers) or
  the names split what the photos can't (temporary codes inside one lookalike group).
- **What helps is steadier evidence per species, not less depth.** The species average
  and nearest fail in opposite ways: the average is good for a species known from a few
  records (5-19 refs: 39.2% vs 31.8%) and blurred for one with hundreds (52.4% vs
  65.7%); nearest is the reverse, and it hangs on one photo. A blend keeps each where
  it's strong. With the mean of each photo's two best matches in place of the single
  best, the deep species keep their lead too.
- Combining photos is not the problem (nearest-mix gave nothing, 2026-09-29) and neither
  is the trained-head route (balanced softmax lost badly, 2026-09-28).

Depth is a moving target (Steve, 2026-10-09). Every week's green records add references
to species already known, so species keep moving from the sparse bands into the
well-sampled ones; fixing sparse-species accuracy is partly something time does for us,
and the work is iterative. Species-labelled records with photos, by validation date:

| reference set up to | records | species | species with 20+ records |
|---|---|---|---|
| 2025-03-07 | 63,796 | 12,141 | 703 (5.8%) |
| 2025-09-07 | 84,858 | 14,056 | 1,022 (7.3%) |
| 2026-03-07 | 108,421 | 15,653 | 1,350 (8.6%) |
| 2026-06-07 | 139,086 | 17,013 | 1,771 (10.4%) |
| 2026-09-07 | 151,522 | 17,935 | 1,919 (10.7%) |

The well-sampled group nearly tripled in 18 months. New names keep arriving too, and the
share of a weekly batch whose species is sparse depends on which projects reported that
week (in the 28 days after each date above, 31%, 21%, 14%, 37% and 32% of test records
had 0-4 references), so the shift shows in the reference set more than in any one batch.
What follows from it:
- Judge methods by depth band, not only overall: the overall number moves with the
  band mix of each batch and with time. A method has to hold up in each band, so its
  gains carry over as species move between them.
- A depth penalty would cost more every month, since it taxes the bands that keep
  growing. `nearest+mean` gains most in 5-19 and 20-99, the bands sparse species move
  into next, and doesn't lose at 100+.
- Settings fitted on one comparison (k and weight here, the per-method temperatures and
  likely-set floors) are fitted to that comparison's depth mix. Refit them on each new
  comparison and keep an eye on the weekly prospective tests as the mix changes.
- The 1-4 band's ceiling isn't fixed: those species become 5-19 species as records
  arrive, and that's where both methods do far better.

`nearest+mean` (methods.NearestAndMean; k = 2 and weight 0.6 chosen on 4ef7b0, on a
plateau: 0.5-0.65 with top-2 all give 38.2-38.4%). Through `evaluate.evaluate` on
4ef7b0, top-1 / top-5:

| | species | genus | family | first photo only, species |
|---|---|---|---|---|
| nearest | 34.5 / 56.8 | 71.6 / 89.2 | 79.7 / 93.6 | 28.9 |
| nearest+mean | **38.4 / 61.2** | **74.4 / 90.8** | **81.8 / 94.6** | **35.0** |

| true species' refs (n) | nearest sp / ge / fa | nearest+mean sp / ge / fa | species fixed / broken |
|---|---|---|---|
| 0 (141) | - / 50.3 / 66.7 | - / 51.0 / 70.3 | |
| 1-4 (215) | 12.6 / 57.2 / 65.6 | 13.0 / 58.6 / 67.4 | +3 / -2 |
| 5-19 (296) | 31.8 / 76.1 / 84.6 | 39.9 / 82.4 / 87.9 | +33 / -9 |
| 20-99 (364) | 54.1 / 79.6 / 84.2 | 58.5 / 81.8 / 85.9 | +32 / -16 |
| 100+ (105) | 65.7 / 90.5 / 96.1 | 67.6 / 91.4 / 95.1 | +8 / -6 |
| provisional (506) | 27.9 / 70.4 / 79.2 | 32.6 / 73.3 / 81.6 | +33 / -9 |
| formal (615) | 40.0 / 72.7 / 80.0 | 43.1 / 75.3 / 81.9 | +43 / -24 |

Species pairs overall +76 / -33 (sign test p < 0.0001), genus +56 / -25. Species
top-5 rises in every band with references. Provisional names gain most (+4.7
points), which matters since they are about half the records. Calibration is per
method: the species temperature stays 0.037, NLL 3.22 to 2.89; the 90% likely set
reaches 60% coverage (55% for nearest), 69% when the true name is in the reference
set (62%), with 4.1 names on average (3.4).
It costs two similarity passes (photos and species averages) plus a sort per query
photo: scoring the 1,152 records twice (all photos, first photo) took 210 s on the
laptop against nearest's 83 s.

Held-out check: the dev set's vectors aren't there yet (the benchmark wasn't frozen
or fetched when this ran), so the confirmation is still to do: `mv heldout predict`
on dev with `--methods nearest,nearest+mean` once dev is fetched. As a sanity check
only, the 99 audit records (embedded from the audit's own photo copies, all manifest
records as reference; nearest reproduced the audit's 99 answers exactly): species
top-1 40 vs 40 (7 fixed, 7 broken), top-5 72 vs 68; 5-19 refs 5/23 vs 2/23, 100+
10/18 vs 13/18. Too small to read either way, but deep species losing is the risk to
watch on dev.

Before it could become a default: confirm on dev; run `mv compare` with it so its
calibration and likely sets are on the scoreboard; it has no `+prior` / `+occ`
variant yet (its scores are cosine-like, so it would go through AsLogProb).
`identify.py` computes the photo similarities twice for it (once for the specimens
shown, once inside the method); share them if it goes live.

## Uninformative photos: drop or down-weight? (experiment, 2026-10-09)

Steve asked (after the 100-record held-out audit) whether photos dominated by a
voucher slip, a basket of several fungi or habitat should be dropped or weighted
down when `nearest` averages a record's photos. Answer: **no rule earns a change.**
Every rule moves a handful of records each way, and even perfect hand labels gain
nothing. What marks these records is low confidence, and the confidence is
already honest about it. This repeats the 2026-09-29 finding on the 8,000-record
sample (about 1 point at most), now on the full-run model.

**Setup.** Model bioclip-2-ft-20261007-165400, method `nearest`, two sets:
comparison 20261008-012435-4ef7b0 (rebuilt exactly, record set 4ef7b0035bc9: 1,152
test records validated after 2026-09-07, 152,915 reference records, 3.6 photos per
record) and the 100 held-out dev records of the 2026-10-09 audit (re-embedded from
the saved large photos, scored against the whole manifest like the Identify page;
top species matched the audit for 100 of 100). Each test photo's best similarity to
every species was stored once (float32, CPU, as the server scores). A rule only
changes how much each photo counts in the mean. Settings were chosen on 4ef7b0 (by
net records fixed over the three ranks) and then scored on the audit. Paired counts
are vs today's plain mean: +fixed / -broken, \* = sign test p < 0.05. Calibration is
species NLL and ECE (10 bins) at each rule's own fitted temperature. Base on 4ef7b0:
34.3 / 71.6 / 79.5 species / genus / family, NLL 3.219, ECE 0.064 (published
34.5 / 71.6 / 79.7; the laptop's vectors were pulled 19 minutes after 4ef7b0 ran,
so they differ by 2 species records. Temperatures and NLL match to 4 decimals). Base
on the audit: 40.4 / 77.8 / 90.6.

| rule (best setting on 4ef7b0) | 4ef7b0 species | genus | family | sp NLL | audit species / genus / family |
|---|---|---|---|---|---|
| (a) drop a photo whose best match to any reference photo is < 0.52 | 34.3 (+2/-3) | 71.4 (+1/-3) | 79.7 (+3/-1) | 3.209 | no change |
| (a) same, stricter floor 0.65 | 33.5 (+13/-22) | 70.7 (+16/-26) | 78.9 (+14/-21) | 3.266 | - |
| (a') drop a photo > 0.15 below the record's best photo | 34.3 (+11/-11) | 71.3 (+7/-10) | 79.3 (+6/-8) | 3.299 | -1 / 0 / 0 |
| (b) weight by the photo's own confidence^0.25 | 34.7 (+6/-2) | 71.3 (+5/-8) | 79.6 (+7/-6) | 3.195 | +1 / +1 / +1 |
| (b) weight by peak-to-second margin^0.25 | 34.7 (+9/-5) | 71.6 (+13/-13) | 79.9 (+14/-9) | 3.209 | 0 / -2 / -1 |
| (b) weight exp(best match / 0.1) | 34.0 (+18/-22) | 70.9 (+18/-25) | 79.8 (+19/-15) | 3.272 | -2 / -1 / 0 |
| (c) 3+ photos: drop a photo whose top genus no other photo shares | 33.5 (+11/-20) | 71.0 (+14/-20) | 79.0 (+13/-19) | 3.246 | 0 / 0 / 0 |
| (c) same at species | 34.0 (+15/-19) | 70.9 (+10/-18) | 78.5 (+5/-16\*) | 3.274 | - |
| (d) few-shot slip detector: drop slip photos | 34.5 (+7/-5) | 71.7 (+6/-4) | 79.8 (+6/-3) | 3.181 | (trained on the audit) |
| (d) few-shot: drop slip + habitat photos | 34.6 (+10/-7) | 71.6 (+9/-8) | 79.8 (+10/-6) | 3.179 | - |

Stronger versions of every rule lose, often significantly: margin^2 32.3 (+26/-49\*),
exp(best / 0.01) 30.7 (+32/-73\*), keeping only photos within 0.03 of the record's best
31.2 (+25/-60\*). (d): BioCLIP 2's own text tower (prompts on the stored base vectors)
called no photo a slip or basket, and its habitat tag hurt (drop: 33.3, +2/-14\*). So
the detector is a 5-nearest-neighbour vote over the hand labels below (16 of its 21
slip calls on the audit right, leave-one-record-out). It flags 222 of 4,165 4ef7b0
photos as slips. Its small gains, like those of confidence^0.25, are spread over
projects and observers. None reaches significance.

**Hand labels (the 369 audit photos).** 28 are slip-dominated (24 with the specimen
on or beside the slip, 4 without), 2 show a basket of several fungi (both in
Volvopluteus 334798553), 19 are habitat (specimen tiny or out of view), and 4 are a
spore print or microscope screen. 36 of the 100 records have at least one. Each
photo alone, species / genus right: specimen photos 28% / 67%, slip with specimen
12% / 42%, habitat 0% / 11%, slip-only, basket and microscope 0% / 0%. Mean best match
to any reference photo: specimen 0.715, slip 0.59-0.64, habitat 0.639, basket 0.592,
microscope 0.824 (the reference set holds microscope photos too, so a similarity
floor cannot catch them, and it would drop slip photos that still carry the
specimen). **Oracle: dropping exactly the hand-labelled photos**, all kinds: species
40.4 (+2/-2), genus 76.8 (+2/-3), family 89.6 (+1/-2). Slip photos only: species
+1/-1, genus +1/-3. Habitat only: species 0/-1, genus +1/0. Microscope only: species
+1/0. Even a perfect detector would not move the numbers.

**What the audit's "every photo points elsewhere" records are.** 36 records by this
count (35 in the audit's name matching), 102 photos: 88 are ordinary specimen photos,
10 slip-with-specimen, 1 slip-only, 1 habitat, 2 basket. Only 11 of the 36 have any
non-specimen photo. These are look-alikes the model is unsure of (in 6 of them every
photo agrees on the genus), not records spoiled by slips. In Volvopluteus 334798553
the specimen is in all three photos, next to the slip and a yellow neighbour from
the basket. A slip filter would drop every photo there, and the rule falls back to
all of them. On 4ef7b0 such records (362 of 1,121) have mean confidence 0.18 and are
right 22.7% of the time; the rest 0.35 and 39.9%. On the audit: 0.29 and 25.7% vs
0.45 and 48.4%. The confidence already says "unsure" for them, at about the right
rate.

**Headroom.** Some single photo's top species is right in 40.9% of 4ef7b0 records
(combined: 34.3%) and in 45.5% of the audit (40.4%). So a perfect photo picker
could add ~5 points, but no signal tried here (similarity, margin, confidence,
agreement between photos, photo kind) finds that photo better than the plain mean.

**Decision for Steve.** Recommend no change to `nearest`. If one rule is ever
wanted, the least risky are dropping detected slip photos (net +2 / +2 / +3 on
4ef7b0, with no independent check because the detector learned from the audit) or a
mild confidence^0.25 weight (+4 species but -3 genus on 4ef7b0, +1 at each rank on
the audit). Both are within noise. Better levers for these records: the page's
existing hint asking for clear underside and stem photos when two species are
close, the species-complex score for look-alikes, and more references for sparse
species. Scripts and hand labels live in this session's
scratchpad, not the repo (the labels point at private audit photos).

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

## Picek replication: their method, our data (feat/picek-replication, 2026-10-09)

Code, sources, deviations and commands in one place:
[replications/fungitastic/](../replications/fungitastic/README.md) (modules under
`src/mycomap_vision/replications/fungitastic/`, tests under
`tests/replications/fungitastic/`; the old module paths still resolve).

Steve: "replicate their model with our data". Lukas Picek's group (BVRA; a PI of the EU
FunDive project, whose model goes into PlutoF GO) built the Atlas of Danish Fungi's
FungiVision (DF20) and FungiTastic. Trained on exactly Vision's training records, their
recipe is the paper's baseline. It separates method from data: if it does about as well
as Vision on the same records, the gap to the published Danish numbers is data; if it
does better, so is the method.

**Recipe** (`replications/fungitastic/retrain.py`, its constants in `presets.py`; preset
`fungitastic-beit-b384`; sources are the HF config of
BVRA/beit_base_patch16_384.in1k_ft_fungitastic_384, BohemianVRA/FungiTastic
baselines/closed_set/train.py and the fgvc library):
- timm `beit_base_patch16_384.in22k_ft_in22k_in1k`, full fine-tune, 384 x 384, a new linear
  head over every species.
- SGD with momentum 0.9 and no weight decay, learning rate 0.01, ReduceLROnPlateau on the
  validation loss (factor 0.9, patience 1, eps 1e-6). Effective batch 256 (micro-batch 16 x
  16 accumulation steps, bf16 autocast and gradient checkpointing, so it fits 8 GB). 50
  epochs, keeping the epoch with the best validation macro-F1.
- Seesaw loss (p 0.8, q 2.0) on the training set's class counts. Plain cross-entropy for
  the ViT variants.
- Training augmentation "vit_heavy": RandomResizedCrop(384, scale 0.8-1) then
  RandAugment(2, 20). Evaluation: resize to 384 x 384, no crop. Mean and std 0.5.
- A record's answer: each photo's logits divided by T (fitted on validation by NLL),
  averaged over the photos, then softmax.
- DF20's metadata prior (paper section 5.2): p(c|x,m) ∝ p(c|x) · p(m|c)/p(m) per field.
  This is the paper's formula; their released code multiplies p(c|x) in twice, which is not
  repeated here.
- Other presets, same code: `vit-b384-ce` (their ViT-B: CE, lr 0.001), `df20-vit-l384`
  (DF20's production ViT-L/16 384: CE, lr 0.005, light augmentation, 100 epochs) and
  `fungitastic-beit-b224` (the 224 px BEiT).

**Data and protocol.** Labels are Vision's own (evaluate.load_records: green records,
observation names, temporary codes as species). Training uses the reference records of
a comparison: everything validated up to the cutoff, which is 28 days before the newest
record. The last 28 days before that cutoff are the validation slice, which is never
trained on. It picks the epoch, drives the plateau schedule and fits T; their splits are
by year too. `trained_through` is the cutoff, so `mv compare` refuses to score the model
on anything it saw. Benchmark records: `--picek-exclude-benchmarks match` (the default)
leaves out exactly what every Vision model leaves out (sealed benchmarks), so the two
train on the same records; `all` also leaves out every benchmark's records (a trainer's
manifest carries no benchmark tables, so their ids are shipped with the run). `<name>.records.csv` lists every record trained or validated on, which is the
follow-up the held-out section asked for.

**Deviations, and why:**
- One-word names have no species label, so they are not trained on. Their data is
  species-level too.
- Seesaw is formed per batch row in log space. fgvc's C x C matrix would be 1.3 GB at
  18k species. It is the same loss (tested against the published formula).
- Training JPEGs decode in PIL draft mode at no less than the crop size (1024 -> 512 px before the
  384 crop), which is cheaper and leaves the 384 input essentially unchanged. Validation decodes in full, as test-time embedding does.
- The month prior is smoothed. DF20 uses raw counts; with 43% of species known from one
  record, raw counts remove a species outright in any month it was not yet found. p(m|c)
  is shrunk toward its genus (beta 10 photos) and the genus toward all fungi (beta 10).
  `classifier+month-raw` is their unsmoothed estimate, floored at 1e-12 so scores stay
  finite. The betas are not tuned yet: tune them on heldout dev only.
- Habitat and substrate, DF20's other fields, don't exist in our data.
- `classifier+month+place` adds a coarse place prior (4-degree cells, the same shrinkage).
  It is OUR extension, not their method, and is kept as its own method so the paper can
  report "their method" and "their method + our place prior" apart.
- A species with no class in the classifier gets probability 1e-12 when scored against an
  index that has it. Such species are those first validated after the cutoff, or only in
  the validation slice. Nearest-specimen methods get those records for free; this is part
  of the method difference, and the paper should say so.

**Scoring.** The model is registered like a fine-tune (data/models/<name>.json + .pt, plus
<name>.classifier.npz with the head, T and the month and place counts). Its embedding is
the pre-logit feature in a length-keeping unit vector (the last dimension carries |f|), so
the head applies exactly to the stored float16 vectors and cosine still works for
nearest. The methods are `classifier`, `classifier+month`, `classifier+month-raw` and
`classifier+month+place`. They work in `mv compare` (other backbones skip them) and in
`mv heldout predict`, which writes the standard heldout_runs and heldout_predictions
rows, so `mv heldout report` gives Steve's summary tables unchanged. They are not offered
on the website. Every comparison now reports species macro-F1 (scikit-learn's macro over
truth ∪ predictions, as fgvc reports it); `mv compare --per-image` adds top-1 with each
photo answered alone. The held-out report adds species macro-F1 and per-photo top-1 for
every Vision model, and prints them under the standard summary. Public name:
"Danish Fungi method (BEiT, Picek et al.), trained on our DNA-verified records".

**How to run.**
- Laptop smoke: `mv picek-train --max-steps 50 --effective-batch 32 --val-max-photos 300`.
- Speed: `mv picek-bench` (the loader and the GPU measured apart).
- Full run on AWS (not launched): `mv aws-launch-trainer --backbones none --finetune none
  --picek fungitastic-beit-b384@15 --methods
  classifier,classifier+month,classifier+month-raw,classifier+month+place,nearest
  --instance-type g6.xlarge --max-hours 52` (see the launch-day runbook below). The instance trains, embeds every photo with
  the model and compares. Then, on the laptop: `mv aws-pull-trainer`, `mv heldout predict
  --backbone picek-... --methods classifier,classifier+month --split dev`, and
  `mv heldout report`. Dev only: test is not scored.
- Photo cache (`preset@epochs@440`, or `--cache-px 440`; off by default). The training
  photos are resized once (shorter side 440, JPEG q90) onto the instance's disk, and
  training reads them from there. Validation reads the originals, like the embedding at
  test time. Transforms and draft decoding are unchanged (tested). It changes what the
  crop is taken from (a 440 px copy, not the 512 px draft decode), so it is Steve's call;
  at 384 px it gains little, because the L4 is the limit there (see below).

**Speed and cost.** Measured on the laptop CPU on 2026-10-09 (local large photos; GPU idle,
but the laptop was busy with another session's job), in photos/s for one process, and
for 4 workers:

| loader pipeline | 1 process | 4 workers |
|---|---|---|
| BioCLIP fine-tune (224, light aug) | 83 | 318 |
| Picek BEiT 384, RandAugment(2,20) | 88 | 320 |
| Picek at 224 | 147 | 452 |
| Picek 384 from the 440 px cache | 149 | 444 |
| evaluation at 384 (full decode) | 87 | 310 |

The 440 px cache files average 76 KB, against 364 KB for the originals. Building the
cache runs at 97 photos/s per thread.

The 384 recipe's loader costs the same per photo as the BioCLIP fine-tune's. On g6.xlarge
(4 vCPU), the BioCLIP fine-tune's loader delivered 59 photos/s with S3 reads, so the
replication's loader should give about 59/s there. With the 440 px cache or the 224 px
preset it should give ~100/s.

GPU, measured on the laptop's RTX 4070 (8 GB) with `mv picek-bench` (random tensors,
18,000 classes, bf16), 2026-10-09. Training photos/s, then peak VRAM:

| preset | micro-batch 16, checkpointing | 16, none | 32, checkpointing | 32, none |
|---|---|---|---|---|
| BEiT-B 384 | 44 (1.6 GB) | 55 (4.5 GB) | 46 (2.2 GB) | 13 (7.8 GB: spills past 8 GB) |
| BEiT-B 224 | 142 (1.3 GB) | 179 (2.2 GB) | 146 (1.4 GB) | 183 (3.3 GB) |

Evaluation (inference) runs at 211 photos/s at 384 and 640 at 224. End to end, `mv embed`
of 3,000 large photos took 18 s (165/s, decode-bound).

Gradient checkpointing now turns on only when the GPU has under 12 GB: it costs ~20%, and
an L4 (24 GB) doesn't need it.

**Smoke run (2026-10-09).** It ran against a scratch copy of the manifest, with no writes
to the shared one:
- `mv picek-train` for 60 steps (effective batch 32) on the laptop's 29,295 local large
  photos: 7,727 records, 4,109 species, validation 192 records.
- 34 photos/s end to end, with micro-batch 16, checkpointing on and 4 workers; 21% of the
  time was spent waiting for the loader.
- The model registered, then `mv embed`, then `mv heldout predict --split dev --limit 50`
  with classifier and classifier+month: 49 records answered. `mv heldout report` rendered
  the standard tables and the macro-F1 / per-photo table.
- After 60 steps the accuracy is ~0, as expected: the smoke test proves the path, not the
  model.

**Estimates for g6.xlarge** (~$0.80/h, ~580k training photos). The L4 is taken as 1.1x
the laptop 4070 for training: an ESTIMATE, from the BioCLIP fine-tune's 83/s on the laptop
against 106/s in its L4 rehearsal. That puts BEiT-B 384 at ~60 photos/s on the L4. Train
hours are the slower of loader and GPU; add ~2.6 h for the cache pass, the embedding of
every photo, setup and the comparison.

| run | photos/s (limit) | 15 epochs | 50 epochs |
|---|---|---|---|
| BEiT-B 384, as is | ~59 (loader ≈ GPU) | ~44 h, ~$35 | ~139 h, ~$111 |
| BEiT-B 384, 440 cache | ~60 (GPU) | ~43 h, ~$35 | ~137 h, ~$110 |
| BEiT-B 224 | ~100 (loader) | ~27 h, ~$21 | ~83 h, ~$66 |
| BEiT-B 224, 440 cache | ~170 (loader) | ~18 h, ~$14 | ~51 h, ~$41 |

At 384 the L4 itself is the limit, so neither the cache nor a 16-vCPU g6.4xlarge buys
much there. Only a faster GPU does: an L40S (g6e.xlarge, ~$1.86/h, roughly 2.5-3x an L4,
an estimate) would be ~150-180/s on the GPU, but needs the cache and more vCPUs to feed it
(g6e.2xlarge or larger).

Recommendation: BEiT-B 384 (their best model) for 15 epochs on g6.xlarge without the cache
(~44.5 h, ~$36, `--max-hours 52`), then decide on 50 from its validation curve. The 224
preset is the cheap variant, if the paper can use it.

**Decided (Steve, 2026-10-09):** their recipe as published: BEiT-B/16 384, no photo cache.
Prepare everything, but launch only once Vision's internal labelling is fixed and final.

### Launch day (runbook)

Everything below runs on **this PC** (Steve's Windows laptop). Shell commands are **Git
Bash**; `aws login` can run in PowerShell or Git Bash. Nothing runs on the .org boxes.

**Which code goes.** `mv aws-launch-trainer` ships `git archive` of the HEAD of the
checkout whose code is running (`config.REPO_ROOT`). The installed `mv` runs the main
checkout's code, so it ships whatever branch that checkout has checked out. A dry run
prints the commit and checks the archive holds the replication's code
(src/mycomap_vision/replications/fungitastic/: `__init__.py`, presets.py, retrain.py), the
trainer and requirements/trainer.txt (timm 1.0.30, torch 2.14.0, torchvision 0.29.0, the
same as the laptop). feat/replications-fungitastic carries feat/picek-replication and
feat/external-bvra-baselines together. Either:
- (a) merge feat/replications-fungitastic to main (Steve reviews), pull main in
  `C:\Users\info\Projects\mycomap-vision`, and use `mv` there; or
- (b) launch from the branch worktree with
  `cd /c/Users/info/Projects/mycomap-vision-replications && PYTHONPATH=src
  ../mycomap-vision/.venv/Scripts/python -m mycomap_vision.cli ...`, after copying the
  main checkout's `.env` beside it (git-ignored) and setting
  `MV_DATA_DIR=C:/Users/info/Projects/mycomap-vision/data`.

Launch refuses uncommitted changes and a commit that isn't pushed.

**On the instance.** The pinned, hashed requirements are installed. The BEiT weights
(`timm/beit_base_patch16_384.in22k_ft_in22k_in1k`, ~350 MB) download from the Hugging
Face hub into /opt/mv/hf. They are public, so no token is needed (the laptop downloaded
them without one). The disk is the AMI's root size + 60 GB: ~135 GB with the 75 GB Base
GPU AMI. `--ec2-check` reads the real size. The run needs about 3 GB for weights and
embeddings, plus the manifest.

**Preconditions:**
1. Labels final. The internal labelling fixes are done, and Vision's manifest is refreshed
   with them (`mv export-records` or the weekly `mv refresh`; then `mv name-spellings`
   and `mv taxonomy` look clean enough).
   - **iNat-only snapshot (launch guard, 2026-10-09).** The label audit found 8,868 Vision
     reference/training records (5.8%) that are Mushroom Observer, MyCoPortal or .com
     sequence ids fetched AS iNat ids, so they carry other observations' photos (mammals,
     birds, plants); labels hash 27240f89 includes them. The launch waits for the
     record-sources fix (feat/record-sources-mo: `records.source` normalised to inat | mo |
     mycoportal | com_sequence | genbank | unknown) and a snapshot migrated by it. The
     guard (`retrain.require_inat_only`, run by `mv aws-launch-trainer --picek` and again
     by the instance before its first stage) first asks whether the manifest is migrated:
     the `record-sources-v1` row in `manifest_migrations` (`retrain.sources_migrated`;
     TODO: the fix's `sources.migrated` once merged). Before that row, `records.source` is
     the old guess ('inat' for any numeric id) and proves nothing, so a real launch is
     refused (the dry run only warns). Once migrated, a record is iNat only when
     `records.source = 'inat'`; any other training or validation record refuses the
     launch, with the count by source. The breakdown goes into the label snapshot
     (`record_sources`: picek-labels.json and result.json). A dry run also reports, for
     information, how many training records the laptop's audit file
     (`data/audits/record-sources-2026-10-09/org-sources-live.tsv`) lists as non-iNat;
     that file never ships.
2. The Vision model the replication is compared against is trained on the same manifest
   state. Same `trained_through` (the cutoff 28 days before the newest record) and same
   benchmark exclusion.
3. Launch-day decision: `--picek-exclude-benchmarks`.
   - `match` (the default): leave out exactly what the Vision models leave out, i.e.
     sealed benchmarks only. The released 13,145 heldout-2026-10-08 records then train
     both models once they are in the manifest.
   - `all`: also leave out every benchmark's records.
   - For a fair comparison, use what the compared Vision fine-tune used. That is `match`
     for a fine-tune made by `mv finetune` or the trainer. The fresh ~1,000-record paper
     set will be sealed (`mv heldout freeze --holdout`), so it is out either way.
4. The branch is merged, or the worktree is ready (above); `git status` is clean and
   pushed.
5. `aws login` (profile `mycomap-vision`). The On-Demand G quota of 4 vCPU is enough for
   g6.xlarge.

**Commands:**
```bash
# 1. Dry run: builds and checks everything, sends nothing. Read the label snapshot it
#    prints: records, species, provisional share, known label problems, labels hash.
mv aws-launch-trainer --backbones none --finetune none --picek fungitastic-beit-b384@15 \
  --methods classifier,classifier+month,classifier+month-raw,classifier+month+place,nearest \
  --instance-type g6.xlarge --max-hours 52 --dry-run [--ec2-check]
# 2. Launch: the same without --dry-run. Note the run id and the labels hash.
mv aws-launch-trainer --backbones none --finetune none --picek fungitastic-beit-b384@15 \
  --methods classifier,classifier+month,classifier+month-raw,classifier+month+place,nearest \
  --instance-type g6.xlarge --max-hours 52
# 3. Monitor: progress.json shows each epoch's validation loss, macro-F1, top-1 and
#    learning rate. The log is uploaded every 15 min.
aws s3 cp s3://<bucket>/runs/<run>/progress.json - --profile mycomap-vision
aws s3 cp s3://<bucket>/runs/<run>/train.log - --profile mycomap-vision | tail -50
# 4. Pull: weights, head, records list, embeddings, scoreboard rows.
mv aws-pull-trainer --run <run>
# 5. The held-out dev set (dev only; test is not scored).
mv heldout predict --name heldout-2026-10-08 --split dev \
  --backbone picek-fungitastic-beit-b384-<run> \
  --methods classifier,classifier+month,classifier+month-raw,classifier+month+place
mv heldout report --name heldout-2026-10-08 --split dev
```

Use `--max-hours 52`, not 48. The estimate is 44.5 h, and the job stops itself 45 min
before the limit. A Picek stage that is stopped saves nothing: the model of a cut-short
run is not the model asked for, as with fine-tuning. The label snapshot is printed at
launch, uploaded as `runs/<run>/picek-labels.json`, and recorded in the model's json and
in result.json. The instance refuses to train when its labels hash differs from the
launch's.

**Expected:** ~44.5 h. That is ~42 h of training at ~59 photos/s (the L4 and the 4-vCPU
loader are about equal) plus 1.6 h embedding plus 1 h setup and comparison, ~$36 on
demand at $0.80/h. The speed is an estimate until progress.json shows the real
photos/s; check it in the first hour (step 3). If it is below ~45/s, the run won't fit
52 h: stop the instance and raise --max-hours.

**Before deciding on 50 epochs** (from progress.json's per-epoch history):
- Is validation macro-F1 still rising over the last 3-4 epochs (more than ~0.5 points an
  epoch)? Then more epochs would pay. If it flattened by epoch ~10, 50 epochs (~$110, ~6
  days) buys little.
- Did the learning rate drop (ReduceLROnPlateau: x0.9 each time validation loss fails to
  improve for 2 epochs)? A few drops by epoch 15 means it is converging; none means it is
  still early, and a longer run would help.
- Is the validation loss rising while macro-F1 still improves? That is overfitting in
  confidence (the temperature fit handles calibration); is the best epoch the last one?
- Compare with Vision on heldout dev (species top-1 and macro-F1, by reference depth)
  before spending more.

**Open launch-day decisions (Steve):** when the labelling is final (the precondition);
`--picek-exclude-benchmarks match` or `all` (match = what the compared Vision fine-tune
used); 15 epochs now, 50 later from the curve; the month-prior smoothing (betas, tune on dev
only); whether the paper reports `classifier+month` (smoothed) or `classifier+month-raw`
(their estimate) as "their method"; launch from main after review, or from the branch.

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

## Name equivalence for scoring (beta, 2026-10-09)

Scoring only, never training labels. `name_equiv.py` gives every scorer the same three
readings of "the species answer was right" and two of "the genus was right"; strict is
the scoreboard's own top-1, and the others are extra columns beside it, never in its
place. `mv compare --name-scores` (or `evaluate(..., name_scores=True)`) adds them to
a run's report as `name_equivalence`, marked `beta`.

- **strict**: the same name, its spellings folded together (names.py), as Vision's labels.
  Strict does NOT fold gender endings: names.py mirrors .org's nameVariants.ts rule for
  rule (one shared fixture), and strict is the scoreboard top-1 and the training label.
  A temporary code is scored exactly like a named species (Steve, 2026-10-08: "this will
  be half of records. Temp codes are just as good as names"): the same code in the same
  genus is a strict match however it is written ('CA4' / 'CA04'); any other code is
  wrong. No record is left out or downgraded for having a code as its true name.
- **s.l.**: strict, or two genera of one group with the same *described* epithet, its
  Latin gender ending aside (Steve, 2026-10-08: yes): -us/-a/-um and -er/-ra/-rum
  ('rimosa' / 'rimosum', 'ruber' / 'rubrum') are one ending, -is/-e another, and the two
  declensions never join ('acris' / 'acra'). Groups
  live in `src/mycomap_vision/genus_groups.json`, easy to extend (Steve, for genera
  recently split whose taxonomy is still fluid): Cortinarius s.l. (Calonarius,
  Phlegmacium, Thaxterogaster, Aureonarius, Cystinarius, Hygronarius, Mystinarius,
  Volvanarius) and Inocybe s.l. (Inosperma, Pseudosperma, Mallocybe, Nothocybe). A
  provisional code is not joined across genera: codes are numbered within a genus, so
  Calonarius sp. 'IN06' and Cortinarius sp. 'IN06' are usually two taxa. In the North
  American reference set 46 described epithets appear under 2+ Cortinarius-group genera
  and 1 (unicolor) under 2+ Inocybe-group genera; counted over all records the
  benchmark session found 49 and 6.
- **complex (beta)**: s.l., or genera of one group with the same epithet stem: a
  described name and the provisional names split from it ('fallax' / 'fallax-PNW03'), two
  provisional names on one stem ('schweinitzii-IN01' / '-IN02'), a species and its
  subspecies or variety. The stem of a code is its lower-case start of 4+ letters
  ('fallax-PNW03' -> fallax), gender-folded like a described epithet; the code letters
  and numbers are never folded. A bare code ('CA04', 'PNW01') has no stem and never
  joins anything. Complex counts whichever side is the code, answer or truth. Steve:
  "we'll have to think about this more, but make a beta".
- **genus strict / s.l.**: the same genus / genera of one group.

On the benchmark session's audit (99 species-level held-out records, nearest top-1):
species strict 40.4%, s.l. 40.4%, complex 43.4%; genus strict 77.0%, s.l. 80.0%.
Gender folding changed none of them. 42 of the 99 true names are temporary codes
(13 named exactly, 31%).

## External baselines: DF20 / FungiTastic (2026-10-09, feat/external-bvra-baselines)

Code, sources, deviations and commands in one place:
[replications/fungitastic/](../replications/fungitastic/README.md) (modules under
`src/mycomap_vision/replications/fungitastic/`, tests under
`tests/replications/fungitastic/`; the old module paths still resolve).

Steve (2026-10-09): the Picek group's published classifiers go beside iNat's computer
vision as outside, zero-retraining baselines in the paper. `replications/fungitastic/`
`published.py`, `published_report.py`; `mv external labels | coverage | predict |
baseline | report`, `mv heldout import-external`.

- **Models** (timm checkpoints, public Hugging Face repos, no login; weights under
  `data/external/bvra/<repo>/`, git-ignored): `external:fungitastic-beit-b384`
  (BVRA/beit_base_patch16_384.in1k_ft_fungitastic_384, 2,829 classes; its config.yaml
  calls the dataset DF24), `external:fungitastic-vit-b384`
  (BVRA/vit_base_patch16_384.in1k_ft_fungitastic_384, 2,829) and `external:df20-vit-l384`
  (BVRA/vit_large_patch16_384.ft_df20_384, DF20 "Production", 1,604). Each config.json
  is checked against the architecture, class count and 384 px input before loading.
- **Licence: CC BY-NC 4.0** (weights and DF20 / FungiTastic data alike): research use
  only, never inside a commercial product. Cite Picek et al. 2022 (WACV, DF20) and
  Picek et al. 2024 (FungiTastic, arXiv:2408.13632).
- **Label map.** None ships with the weights. Rebuilt from the public training metadata
  (`data/external/metadata/`, no login): FungiTastic-Train.csv `category_id` (from
  cmp.felk.cvut.cz's FungiTastic metadata.zip, as BohemianVRA/FungiTastic dataset/fungi.py
  reads it; 433,702 rows, 2,829 ids) and DanishFungi2024-train.csv `class_id` (266,273 rows,
  the "DF20_FIX" set the DF20 Production model's config.yaml names; DF20-train_metadata_
  PROD-2.csv gives the identical id -> species map). Ids are the training rows'
  `scientificName` in alphabetical order; a class is that name without its author
  ('Gliophorus perplexus'), GBIF's accepted name (`species`) kept beside it (248
  FungiTastic and 147 DF20 classes differ). `build_labels` refuses anything but one name
  per id, ids 0..N-1. **Verified** on CPU with 11 Vision reference records of common
  species shared with Denmark (Trametes versicolor, Laetiporus sulphureus, Pleurotus
  ostreatus, Hypholoma fasciculare, Lycoperdon perlatum, Stereum hirsutum, Schizophyllum
  commune, Armillaria mellea, Mycena galericulata, Fomitopsis betulina, Boletus edulis):
  top-1 right for 8 / 7 / 8 of 11 (BEiT / ViT-B / DF20), and every miss a plausible
  confuser with the right species in the top 3 (T. versicolor -> T. hirsuta, A. mellea
  -> A. lutea, B. edulis -> Calocybe gambosa / B. reticulatus); a shifted map would give
  unrelated names.
- **Names.** A class takes Vision's label of its own name when Vision knows it, else of its
  accepted name (DF20's 'Piptoporus betulinus' -> Fomitopsis betulina), else its own name;
  classes on one Vision name are one candidate (probabilities summed); genus is the
  label's first word, family Vision's (iNat's) for the genus. FungiTastic: 1,254 classes
  are Vision names as they are, 47 by their accepted name, 1,528 are not Vision names.
  DF20: 877, 24, 703.
- **GBIF synonym crosswalk** (coordinator for Steve, 2026-10-09: yes; `crosswalk.py`).
  Scoring only, never Vision's labels. Every formal species name a report compares (answer
  keys, answers, classes; never a temporary code or a one-word name) is matched exactly, no
  fuzzy matching, in the GBIF Backbone (`/v1/species/match`, strict) and in the Catalogue
  of Life eXtended Release through GBIF (`/v2/species/match`, checklistKey COL XR); the
  Backbone alone lacks recent combinations such as 'Collybia nuda' (= Danish 'Lepista
  nuda'). Two names are one species when they share an accepted key in either. Cached in
  `data/external/gbif/match.sqlite` (`mv external crosswalk`, 4 requests a second, the
  project's User-Agent; no iNat call); reports read the cache only. Every table is given
  twice, by exact names and with the crosswalk, and one judge scores every model in a
  table, Vision included, so a synonym can make a Vision answer right too.
- **Protocol.** Photo only: their metadata prior is Danish habitat, substrate and month
  frequencies, and a month prior learnt from Danish records says nothing about a North
  American season, so none is used (coordinator, 2026-10-09). Each
  photo resized to 384 x 384, no crop, mean = std = 0.5 (model cards); a record's answer is
  the softmax of its photos' mean logits (the authors' observation rule), temperature 1:
  fitting it on dev would tune a baseline on our labels. Each photo's own top-1 is kept for
  their per-image metric. `mv external predict` reads the manifest only and writes JSONL
  (logits cached in the benchmark's folder); `mv heldout import-external` stores it as
  heldout_predictions with a heldout_runs row keyed by the checkpoint id (weights + class
  map hashes), so `mv heldout report` scores it like any model (its run is never taken as
  the breakdowns' reference). `mv external baseline --comparison <id>` adds an eval_runs row
  on a scoreboard comparison's test records, with the standard top 1/3/5/10 block in
  feat/compare-approaches' shape (its constants are imported from evaluate once that
  branch is merged). Scoring, since these models have no temporary codes: (i) genus and
  family on all records; (ii) species on records whose true name is formal; (iii) same
  vocabulary: on records whose true name is in model C's vocabulary, C, Vision as served
  and Vision restricted to C's names (Vision's stored list re-ranked, never predicted
  again; a list left shorter than k makes top-k a lower bound, and the table says on how
  many records); (iv) coverage. `mv external report` refuses a sealed test split.
- **Coverage (iv)**, true names read by Vision's labels (heldout-2026-10-08 dev, 3,000
  records; and Vision's 159,388 North American records, guests left out):

  | | FungiTastic (both) dev | DF20 dev | FungiTastic NA | DF20 NA |
  |---|---|---|---|---|
  | temporary code | 36.8% | 36.8% | 41.4% | 41.4% |
  | formal, in vocabulary | 17.0% (509) | 13.8% (414) | 16.2% | 13.1% |
  | formal, s.l. only | 0.0% | 0.2% | 0.0% | 0.1% |
  | formal, not in vocabulary | 43.9% | 46.9% | 41.5% | 44.6% |
  | one word | 2.4% | 2.4% | 0.9% | 0.9% |
  | genus in vocabulary | 87.8% | 82.4% (+2.0% s.l.) | 91.1% | 85.4% (+1.9% s.l.) |

  So at most 17% of our DNA-verified records (28% of those with a formal name) are names a
  Danish model can say at all; distinct species in its vocabulary: 253 of 1,860 on dev and
  1,224 of 18,600 North American (FungiTastic), 197 and 853 (DF20).

- **Dev results (2026-10-09, full 3,000-record dev split; 2,989 answered by the outside
  models, 12,206 photos; GPU wall 194 / 188 / 285 s).** Photo only, temperature 1. Top 1 / 3
  / 5 / 10 strict; "x" = with the GBIF crosswalk (8,987 of 9,730 names matched; it applies
  to every model). Vision = bioclip-2-ft-20261007-165400 as stored (10 deep).

  | | (i) genus, all (n 2,966) | (i) family | (ii) formal species (n 1,818) | (ii) x |
  |---|---|---|---|---|
  | FungiTastic BEiT-B | 54.6 / 68.6 / 73.8 / 78.7 | 66.8 / 81.0 / 86.1 / 91.3 | 14.0 / 20.0 / 21.7 / 23.3 | 15.8 / 22.3 / 24.4 / 26.4 |
  | FungiTastic ViT-B | 50.8 / 66.0 / 71.6 / 77.7 | 61.8 / 79.1 / 84.3 / 89.9 | 13.5 / 18.0 / 19.7 / 22.5 | 15.3 / 20.3 / 22.2 / 25.2 |
  | DF20 ViT-L | 52.4 / 65.7 / 69.2 / 73.7 | 64.7 / 80.4 / 84.6 / 89.6 | 12.6 / 17.0 / 18.1 / 19.5 | 14.7 / 19.9 / 21.3 / 23.0 |
  | Vision nearest | 79.0 / 90.5 / 93.3 / 95.7 | 86.4 / 94.7 / 96.4 / 98.1 | 54.4 / 72.9 / 79.5 / 85.6 | 54.5 / 73.1 / 79.7 / 85.8 |
  | Vision nearest+prior@org | 79.5 / 91.1 / 93.7 / 95.8 | 86.7 / 94.8 / 96.8 / 98.2 | 57.1 / 74.2 / 80.0 / 85.6 | 57.2 / 74.4 / 80.2 / 85.6 |

  (iii) Same vocabulary, species top 1 / 3 / 5 / 10 and macro-F1 (exact names; with the
  crosswalk n rises to 575 FungiTastic, 483 DF20 and the numbers move by about a point):

  | | FungiTastic records (n 508) | DF20 records (n 413) |
  |---|---|---|
  | the outside model | BEiT 50.0 / 71.7 / 77.6 / 83.5, F1 49.7; ViT-B 48.2 / 64.4 / 70.5 / 80.5, F1 45.4 | 55.5 / 74.8 / 79.7 / 86.0, F1 54.9 |
  | Vision nearest | 60.4 / 79.7 / 84.1 / 88.6, F1 57.9 | 60.5 / 78.7 / 83.5 / 87.4, F1 58.7 |
  | Vision nearest+prior@org | 63.2 / 78.5 / 83.9 / 89.0, F1 61.2 | 63.2 / 78.0 / 83.8 / 88.6, F1 62.0 |
  | Vision nearest, restricted | 77.2 / 87.8 / 88.2 / 88.6, F1 70.5 | 78.5 / 87.2 / 87.4 / 87.4, F1 72.4 |

  Restricted top 3+ are lower bounds: Vision's stored list keeps under 3 of the model's
  names on about 45% of those records (top 1 on 11-15). Per-image species top 1 (their
  metric), formal species: BEiT 10.5%, ViT-B 9.8%, DF20 9.2% of 7,206 photos. Full tables:
  `mv external report --name heldout-2026-10-08` (both scorers) and `mv heldout report`.
