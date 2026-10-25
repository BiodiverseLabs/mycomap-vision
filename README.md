# MycoMap Vision

[![CI](https://github.com/BiodiverseLabs/mycomap-vision/actions/workflows/ci.yml/badge.svg)](https://github.com/BiodiverseLabs/mycomap-vision/actions/workflows/ci.yml)

Photo identification for North American fungi whose names come only from records
confirmed by DNA sequencing: the model is fine-tuned on those records' labels, and every
specimen it compares a photo with is one of them. (The general image model it starts
from was pretrained elsewhere, on names that were not DNA-checked.) Give it one or more photos of a find and it
suggests a name at every rank (species, genus, family) with a calibrated
confidence, shows the DNA-verified specimens that look most alike, and says what
photo would help (for example, the gills).

It is a project of [MycoMap](https://mycomap.org), a nonprofit. The site is
**[vision.mycomap.org](https://vision.mycomap.org)**; before public launch it is
open to anyone with a mycomap.org account.

## Status (October 2026)

**Live:** the Identify site, answering from about 600,000 photos of DNA-validated
records with a fine-tuned model; a not-a-fungus check in front of it; "likely" name
lists; a place-and-date prior, used whenever a find has a place or date; nightly
updates as new records are validated on mycomap.org; and a public API (v1) for
developers, mycomap.org and its mobile app, with keys from mycomap.org
([docs/api.md](docs/api.md), [vision.mycomap.org/research/api](https://vision.mycomap.org/research/api)).

**The model, MycoMap Vision v1**, is [BioCLIP 2](https://imageomics.github.io/bioclip-2/)
(Imageomics) fine-tuned on the DNA-verified records of dataset release v1, reading
photos at 336 × 336 px. A twin, **MycoMap Vision v1-open**, is trained the same way on
openly licensed (Creative Commons) photos only, so that it can one day be released
with its data; on photos alone it scores within half a point of v1 (species top-1
55.9% vs 56.3% on the same finds).

**Accuracy** of v1 as served (photos plus the place prior) on dataset release v1's
validation split, record-grouped (each find counts once, scored over all its photos):

| Rank | n | Top-1 | Top-3 | Top-5 | Top-10 |
| --- | --- | --- | --- | --- | --- |
| Species | 12,798 | 59.2% | 75.3% | 80.2% | 85.1% |
| Genus | 12,865 | 85.8% | 93.9% | 95.7% | 97.3% |
| Family | 12,759 | 90.6% | 96.3% | 97.6% | 98.5% |

The live figures are public at
[vision.mycomap.org/api/how-good](https://vision.mycomap.org/api/how-good); the method,
breakdowns and benchmarks are on the site's [Research](https://vision.mycomap.org/research)
pages ([Results](https://vision.mycomap.org/research/results)). These are validation
numbers, not a final test: a sealed test set (`v1-test`, records validated after v1 was
cut) is scored once, after it closes.

Accuracy follows how many DNA-verified references a species has: species with 100+
references are named first about 77% of the time, species with 1–4 about 20%, and a
species with none cannot be named at all. On a sample, genera with 100+ references
matched iNaturalist's model, and the gap was in genera with few or none. A full
comparison with iNaturalist's computer vision is still to come.

**Dataset release v1** was frozen on 12 October 2026: 169,135 records (155,781 train,
13,354 validation) and 680,047 photos, so anyone can reproduce every result
([docs/dataset-release.md](docs/dataset-release.md)). The v1 retrain followed a
pre-declared evaluation ([docs/retrain-v1.md](docs/retrain-v1.md)). Lower numbers
elsewhere in the docs (for example 34.5% or 54.2% species top-1) are development
numbers from earlier models on a dataset that was still changing, kept as history.

## How it works

1. **Labels come only from DNA.** A record is used when a project on mycomap.org
   validated its name with a DNA sequence. Community identifications from
   iNaturalist or Mushroom Observer are never used as labels.
2. **Every photo of a find counts.** Each photo becomes a vector from an image model
   (a fine-tuned [BioCLIP 2](https://imageomics.github.io/bioclip-2/)), and a find is
   scored over all its photos.
3. **Nearest DNA-verified specimens.** Candidate names are ranked by how closely
   their reference photos match, blended with each species' average photo. There is
   no classifier head to retrain, so a newly validated record joins the reference
   set overnight.
4. **Confidence per rank.** Scores are calibrated on dataset release v1's validation
   split (its newest eight weeks of records), and fitted again for each new model or
   release; "likely" lists aim to hold the right name 9 times in 10.
5. **Honest evaluation.** Tests use the newest records (split by time, never at
   random) and records held out of everything, and are reported by how many
   references each species has.

More: [docs/PLAN.md](docs/PLAN.md) (the working plan and findings). The experiment
write-ups and the paper draft hold unpublished results, so they are not in this repo:
signed-in mycomap.org members read them on the site under
[Research](https://vision.mycomap.org/research).

## Run it yourself

The code, tests and web app run without any private data. Identifying needs a
trained release, which is private for now (see [Data and privacy](#data-and-privacy)).

Requirements: Python 3.12 (3.11 works; the pinned requirements are compiled for
3.12). For the web app, Node 22 and pnpm 10 (through corepack).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --require-hashes -r requirements/dev.txt
pip install -e . --no-deps
pytest -q                            # what CI runs; no data or GPU needed
```

The web app (`web/`, Vite + React):

```bash
cd web
corepack enable
pnpm install
pnpm build    # mv serve then serves web/dist and the API on :8010
pnpm dev      # or a dev server on :5010 that proxies /api to :8010
```

To compute embeddings, install PyTorch for your GPU first (from
download.pytorch.org), then `pip install -e ".[embed]"`. The pipeline and every
command are in [docs/cli.md](docs/cli.md). The command is called `mv`, which
shadows the Unix `mv` in an activated venv.

## Repository layout

| Path | What's there |
| --- | --- |
| `src/mycomap_vision/` | the Python package: data pipeline, models, evaluation, API server (`api.py`), CLI (`cli.py`) |
| `web/` | the vision.mycomap.org site (Vite, React, Tailwind) |
| `tests/` | the pytest suite (runs without data) |
| `docs/` | plan, data rules, CLI, deployment, dataset releases |
| `replications/fungitastic/` | scoring and retraining the FungiTastic and Danish Fungi 2020 classifiers on our records |
| `deploy/aws/`, `deploy/lightsail/` | self-terminating GPU and download instances; the web server |
| `scripts/` | analysis scripts for the label audit (leave-one-out scan) |
| `requirements/` | hash-pinned dependencies: `dev` (tests), `serve` and `box` (server), `trainer` and `downloader` (instances) |

## Documentation

| Doc | For | What it covers |
| --- | --- | --- |
| [docs/README.md](docs/README.md) | everyone | every doc, one line each, with the historical ones marked |
| [docs/data.md](docs/data.md) | everyone | which records are used; labels, photos, licences, photographer permissions |
| [docs/cli.md](docs/cli.md) | developers | the pipeline and every `mv` command |
| [docs/api.md](docs/api.md) | developers | the public API (v1): credentials, requests, the answer, errors, quotas |
| [docs/PLAN.md](docs/PLAN.md) | researchers, developers | the working plan, decisions and findings, newest phase first |
| [docs/dataset-release.md](docs/dataset-release.md) | researchers | frozen, verifiable dataset releases |
| [docs/retrain-v1.md](docs/retrain-v1.md) | researchers | the pre-declared evaluation for the v1 retrain |
| [replications/fungitastic/](replications/fungitastic/README.md) | researchers | comparison with the Picek group's classifiers |
| [docs/deploy.md](docs/deploy.md), [deploy/aws/](deploy/aws/README.md) | operators | the server and the cloud instances |
| [SECURITY.md](SECURITY.md) | everyone | reporting problems; what stays private |
| [CLAUDE.md](CLAUDE.md) | contributors | project rules (written for AI coding assistants, binding on everyone) |

## Glossary

- **Green / validated**: a record whose name a mycomap.org project confirmed with a
  DNA sequence. Only these records provide labels.
- **mycomap.org**: MycoMap's main site, where records, sequences and validations
  live. Vision only reads from it. The "legacy site" is MycoMap's older platform.
- **Reference set**: the DNA-validated photos a new photo is compared against.
- **Serving release** vs **dataset release**: what the website loads (`mv release`)
  vs a frozen research snapshot (`mv dataset`).
- **The box**: the small server that runs vision.mycomap.org. It only answers
  requests; all training runs on short-lived GPU instances.
- **Exploratory-pre-freeze**: a result measured before dataset release v1, on data
  that kept changing.
- **Steve**: Steve Russell, MycoMap's founder and this project's owner. "Steve's
  decision" records who decided.
- **Lanes, coordinator, sessions**: the owner builds the project with several AI
  coding sessions (Claude Code) working in parallel, each on one area (a "lane").
  One coordinating session merges and deploys. The docs mention them where they
  record who did what.

## Data and privacy

The data is not in this repository and never will be. The photos belong to their
photographers (many are non-commercial or all rights reserved), and records can hold
the true locations of obscured observations of rare species. The manifest,
embeddings, contributor lists and trained models stay private until photographers'
permissions allow otherwise; a Creative-Commons-only dataset release is planned.
Photographers can grant or withdraw permission on mycomap.org, and a withdrawal
takes effect at once. Details: [docs/data.md](docs/data.md), [SECURITY.md](SECURITY.md).

**Your photos on the site.** Every identification on the site gets its own link,
`/r/<id>`, so it can be shared (`src/mycomap_vision/runs.py`). Vision keeps the answer
with it and, for uploaded photos, copies at most 1024 px wide with the camera details
(EXIF: exact place, time, camera) removed; the place is kept only to 0.1 degree, as the
page shows it. Each run also records who ran it (their mycomap.org account), so they can
delete it; that is never shown. Only people given the link can open it: the id is
random, the pages are not indexed, and nothing lists them. A run is kept until the
person who ran it, or a mycomap.org admin, deletes it. Runs from an iNaturalist or
Mushroom Observer link keep no photos, only the observation's photo addresses at its
source. Shared photos are never used to train Vision or as references. The developer
API and the app keep nothing new.

## Contributing and contact

Issues and pull requests are welcome. CI must pass, and every change comes with
tests. Report security problems privately (see [SECURITY.md](SECURITY.md)). Contact:
**info@mycomap.org**.

## Licence

The code is released under the [MIT License](LICENSE). The licence covers this
repository's code and documentation only. It does not cover the photos, records or
trained models, which are not in the repository (see [Data and privacy](#data-and-privacy)).
One image in the repository is not MIT: the photo in `web/public/og-card.jpg` (the site's
link-preview card) is Jesse Bauer's, from [Unsplash](https://unsplash.com/photos/pzwH-a4aF3s),
used under the [Unsplash License](https://unsplash.com/license); its source is recorded
in `scripts/make_og_card.py`.
If you use MycoMap Vision in research, please cite it ([CITATION.cff](CITATION.cff)).
