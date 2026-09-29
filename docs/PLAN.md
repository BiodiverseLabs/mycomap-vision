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
  less on spot; one full embedding pass should take a few hours).

Built 2026-09-28 (`mv aws-launch-trainer`, `mv aws-pull-trainer`; how to run it in
deploy/aws/README.md): g6.2xlarge from the Deep Learning Base GPU AMI, each run in
its own `runs/<run>/` folder (never the downloader's manifest), each backbone
uploaded as soon as it is embedded, `result.json` last. Waiting on the GPU quota
and the ops-policy update; first run after the download completes.

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
