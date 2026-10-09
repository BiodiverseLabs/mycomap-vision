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
