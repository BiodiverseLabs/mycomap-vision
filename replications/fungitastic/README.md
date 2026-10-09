# Replicating FungiTastic and Danish Fungi 2020

This folder is the starting point for the code we use to compare MycoMap Vision with
the fungi classifiers of Lukas Picek's group (BVRA, University of West Bohemia). It
says what we ran, how we changed their method and why, and how to run it again.

## Their work

- **Danish Fungi 2020 (DF20).** A fungi image dataset from the Atlas of Danish Fungi,
  with photos, species labels and metadata (month, habitat, substrate). Picek, L., et
  al. 2022. *Danish Fungi 2020: Not Just Another Image Recognition Dataset.* WACV 2022.
- **The DF20 recognition system and its metadata prior.** Picek, L., et al. 2022.
  *Automatic Fungi Recognition: Deep Learning Meets Mycology.* Sensors 22(2): 633.
- **FungiTastic.** A larger benchmark built from the same Danish records, with
  closed-set, open-set and few-shot tasks. Picek, L., et al. 2025. *FungiTastic: A
  Multi-Modal Dataset and Benchmark for Image Categorization.* CVPR Workshops 2025.
  arXiv:2408.13632.
- **Their code.** [BohemianVRA/FungiTastic](https://github.com/BohemianVRA/FungiTastic)
  (`baselines/closed_set/train.py`), which trains with the `fgvc` library
  ([BohemianVRA/FGVC-Tools](https://github.com/BohemianVRA/FGVC-Tools)).
- **Their published checkpoints**, on Hugging Face, licence **CC BY-NC 4.0**:
  - `BVRA/beit_base_patch16_384.in1k_ft_fungitastic_384`: FungiTastic, BEiT-B/16, 2,829 classes
  - `BVRA/vit_base_patch16_384.in1k_ft_fungitastic_384`: FungiTastic, ViT-B/16, 2,829 classes
  - `BVRA/vit_large_patch16_384.ft_df20_384`: DF20 "Production", ViT-L/16, 1,604 classes

## What we did

Our records are North American and DNA-verified. Many of them are provisional species
with temporary codes. So we did two things.

### 1. Scored their published checkpoints on our records

We run the three checkpoints as their authors ship them, never retrained. Each photo
is resized to 384 x 384 with no crop and normalised with mean = std = 0.5. A record's
answer is the softmax of the mean of its photos' logits (their observation rule).
Photo only: no metadata prior.

Their models cannot name a provisional species, and many of our formal names are not
in their vocabulary. So we score them with a protocol for models without provisional
names, every table with its n:

- **Coverage:** how many records have a true name the model can say at all.
- **Genus and family** on all records.
- **Species** on the records whose true name is formal (not a temporary code).
- **Same vocabulary:** on the records whose true name is in the model's vocabulary,
  the model, Vision unrestricted, and Vision restricted to the model's names.
- **Synonyms:** every table is given twice, by exact names and with a GBIF Backbone +
  Catalogue of Life crosswalk (exact matches only, no fuzzy matching). The crosswalk
  is applied to both sides, Vision included. It is used for scoring only, never for
  Vision's labels.

### 2. Retrained their recipe on our data

We train their FungiTastic recipe on exactly the records Vision trains on. If it does
about as well as Vision on the same records, the gap to their published Danish numbers
is data. If Vision does clearly better on the same records, part of the gap is method.

The recipe (preset `fungitastic-beit-b384`):

- timm `beit_base_patch16_384.in22k_ft_in22k_in1k` (BEiT-B/16), 384 x 384, full
  fine-tune, a new linear head over every species.
- SGD (momentum 0.9, no weight decay, learning rate 0.01), ReduceLROnPlateau on the
  validation loss (factor 0.9, patience 1).
- Seesaw loss (p 0.8, q 2.0) on the training set's class counts.
- RandomResizedCrop (scale 0.8 to 1), then RandAugment(2, 20).
- Effective batch 256 (micro-batch 16 x 16 accumulation steps, bf16).
- The epoch kept is the one with the best validation macro-F1.
- A record's answer: its photos' logits divided by a temperature fitted on validation,
  averaged, then softmax.
- The month prior from DF20's paper: p(c | x, m) ∝ p(c | x) · p(m | c) / p(m).

Other presets use the same code: `fungitastic-beit-b224`, `vit-b384-ce` (their ViT-B)
and `df20-vit-l384` (DF20's production ViT-L).

## Where we differ from them, and why

Retraining their recipe:

- **Seesaw is formed per batch row.** `fgvc` builds a C x C matrix, which is 1.3 GB at
  our ~18,000 species. We form the same factors per batch row, in log space. It is the
  same loss, tested against the published formula.
- **One-word names are not trained.** A record named only "Russula" has no species
  label. Their data is species-level too.
- **The month prior is smoothed toward the genus.** DF20 uses raw counts. With about
  43% of our species known from one record, raw counts would rule a species out in any
  month it was not yet found. We shrink p(m | c) toward its genus, and the genus
  toward all fungi. Their raw estimate is kept as its own method
  (`classifier+month-raw`).
- **No habitat or substrate.** Our records do not have DF20's other metadata fields,
  so the prior is month only.
- **Training photos are decoded at reduced size.** JPEGs are decoded in PIL's draft
  mode at no less than the crop size (1024 to 512 px, before the 384 px crop). This is
  cheaper and leaves the 384 px input essentially unchanged. Validation decodes in full.
- **The place prior is ours.** `classifier+month+place` adds a coarse place prior.
  It is not their method, and it is reported separately.
- **Their released code multiplies p(c | x) in twice** when it applies the prior. We
  use the formula in their paper.
- **Unseen species get ~0 probability.** A classifier has no class for a species first
  validated after its training cutoff, so such a species gets probability 1e-12.
  Vision's nearest-specimen methods can still name it. This is part of the method
  difference.
- **Splits are by time.** Theirs are by year. Ours: everything validated up to a
  cutoff trains; the last 28 days before the cutoff are the validation slice, never
  trained on.

Running their published checkpoints:

- **Temperature is fixed at 1.** Fitting it on our data would tune their baseline on
  our labels.
- **No Danish month prior.** A month prior learnt from Danish records says nothing
  about a North American season.
- **The class map is rebuilt.** None ships with the weights. We rebuild it from their
  public training metadata the way their data loaders do, and check it on common
  species shared with Denmark.

## How to run it

Install the repository as in the [top-level README](../../README.md) (Setup), with the
`embed` extras and a CUDA build of PyTorch. Commands are for **Git Bash**. Each step
says which machine it runs on: **the laptop (this PC)**, or **an AWS g6.xlarge via
`mv aws-launch-trainer`**. `<benchmark>` is a held-out benchmark frozen with
`mv heldout freeze`.

### Published checkpoints (all on the laptop (this PC), GPU)

1. Download each checkpoint's `pytorch_model.bin` and `config.json` from Hugging Face
   into `data/external/bvra/<repo name>/`. Put the public training metadata in
   `data/external/metadata/`: `FungiTastic/FungiTastic-Train.csv` (from the
   FungiTastic metadata download) and `DanishFungi2024-train.csv`.
2. Build and check the class maps:
   ```bash
   mv external labels
   ```
3. Coverage (read-only):
   ```bash
   mv external coverage --name <benchmark> --split dev
   ```
4. Answers, written as JSONL (reads the manifest only), then stored:
   ```bash
   mv external predict --name <benchmark> --split dev
   mv heldout import-external --name <benchmark> --backbone external:fungitastic-beit-b384 \
     --results <the JSONL predict wrote>
   ```
   Repeat the import for `external:fungitastic-vit-b384` and `external:df20-vit-l384`.
5. The synonym crosswalk (asks GBIF, 4 requests a second, cached):
   ```bash
   mv external crosswalk --name <benchmark> --split dev
   ```
6. The tables:
   ```bash
   mv external report --name <benchmark> --split dev
   mv heldout report --name <benchmark> --split dev
   ```
   `mv external baseline --comparison <id>` adds the checkpoints to a saved scoreboard
   comparison instead.

### Retraining their recipe

- Smoke test, on the laptop (this PC):
  ```bash
  mv picek-train --max-steps 50 --effective-batch 32 --val-max-photos 300
  ```
- Speed of the data loader and the GPU, measured apart, on the laptop (this PC):
  ```bash
  mv picek-bench
  ```
- Dry run of the full run, on the laptop (this PC). It builds and checks everything
  and sends nothing:
  ```bash
  mv aws-launch-trainer --backbones none --finetune none --picek fungitastic-beit-b384@15 \
    --methods classifier,classifier+month,classifier+month-raw,classifier+month+place,nearest \
    --instance-type g6.xlarge --max-hours 52 --dry-run
  ```
- The full run, on an AWS g6.xlarge via `mv aws-launch-trainer`: the same command
  without `--dry-run`, launched from the laptop (this PC). The instance trains, embeds
  every photo with the model, compares, uploads and shuts itself down.
- Back on the laptop (this PC):
  ```bash
  mv aws-pull-trainer --run <run>
  mv heldout predict --name <benchmark> --split dev --backbone picek-fungitastic-beit-b384-<run> \
    --methods classifier,classifier+month,classifier+month-raw,classifier+month+place
  mv heldout report --name <benchmark> --split dev
  ```

Before a real launch, follow the launch-day runbook in [docs/PLAN.md](../../docs/PLAN.md)
("Picek replication", "Launch day (runbook)"): its preconditions, the label snapshot to
read, what to watch in the first hour and how to decide on more epochs.

## Where the results are

- Standard evaluation protocol: [docs/PLAN.md](../../docs/PLAN.md), "Held-out
  benchmarks", and the Vision site's [protocols page](https://vision.mycomap.org/research/protocols).
- Results: the Vision site's [benchmarks](https://vision.mycomap.org/research/benchmarks)
  and [results](https://vision.mycomap.org/research/results) pages, and the paper (in
  preparation).

Development set only, not a paper result: on our 3,000-record development split, 17% of
records have a formal species name in FungiTastic's vocabulary (13.8% for DF20). Most of
the rest are provisional species or formal names their models never saw.

## What is not in this repository

- Photos, the manifest, labels, record lists, benchmark record ids and trained weights.
  They are private under MycoMap's data rules (see [SECURITY.md](../../SECURITY.md)).
- Their checkpoints and metadata. Download them from their authors, under their
  licence (CC BY-NC 4.0: research use only, never in a commercial product).

## Status

- Published checkpoints: built and run on the development split. The paper's numbers
  come later, on a sealed test set.
- Retrain: prepared and dry-run tested, not yet run. It waits for Vision's labels to be
  final, so both models train on the same labels.

## The code

All under [`src/mycomap_vision/replications/fungitastic/`](../../src/mycomap_vision/replications/fungitastic/):

| file | what it holds |
|---|---|
| `presets.py` | the recipe's presets and hyperparameters; the three published checkpoints |
| `retrain.py` | their recipe on our records: data, Seesaw, augmentation, training, temperature, the month prior, the classifier methods |
| `published.py` | running their checkpoints: the class map, the observation rule, answers on a benchmark or a comparison |
| `published_report.py` | the comparison protocol: coverage, genus and family, formal species, same vocabulary |
| `crosswalk.py` | the GBIF Backbone + Catalogue of Life synonym crosswalk |

Tests: [`tests/replications/fungitastic/`](../../tests/replications/fungitastic/). The
commands (`mv picek-train`, `mv picek-bench`, `mv aws-launch-trainer --picek`,
`mv external ...`, `mv heldout import-external`) are defined in `src/mycomap_vision/cli.py`.
The old module paths (`mycomap_vision.picek`, `.external`, `.external_report`, `.gbif`)
still work: each is the moved module itself.
