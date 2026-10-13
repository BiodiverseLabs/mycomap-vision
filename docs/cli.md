# The `mv` command line

Everything runs through one command, `mv` (installed by `pip install -e .`).
`mv --help` and `mv <command> --help` are the full reference; this page groups the
commands and shows how they fit together.

> On Linux and macOS, an activated venv puts this `mv` ahead of the system's `mv`
> (move). Call it as `.venv/bin/mv` (Windows: `.venv/Scripts/mv`), or
> `python -m mycomap_vision.cli`, if you don't want that.

Most commands need the private data under `data/` and the settings in `.env`
(see [.env.example](../.env.example)); the tests do not.

## The pipeline

1. `mv export-records`: pull validated records from mycomap.org into the manifest,
   each with its source (`mv record-sources` shows what an export would change).
2. `mv fetch-inat`: photo lists, licences and owners from iNat (1 request/s), for iNat
   records; `mv fetch-mo`: the same from Mushroom Observer for MO records (5 s apart).
3. `mv aws-launch-downloader`: large (1024 px) photos straight into the private S3
   bucket from a self-terminating EC2 instance, within iNat's limits. See
   [deploy/aws/README.md](../deploy/aws/README.md). (`mv download-photos` does the
   same into a local folder or any `--dest`.)
4. `mv embed --backbone <alias or timm:/open_clip: spec>`: one vector per photo.
   Full runs (every photo, plus fine-tuning) go to a GPU instance with
   `mv aws-launch-trainer` and come back with `mv aws-pull-trainer`.
5. `mv compare --backbones a,b --methods nearest,nearest+mean`: score models on the
   same photos (newest 28 days vs older records); `mv scoreboard` lists results;
   `mv models` lists backbones and methods.
6. `mv release --backbones <model> --make-current`: publish what the public site
   serves to S3; the server pulls it with `mv pull-release`
   ([docs/deploy.md](deploy.md)).

`mv refresh` is the weekly loop (export, fetch, taxonomy of new genera, download,
embed, compare). On the server, `mv nightly` adds new, renamed and removed records
between releases.

**Taxonomy.** `mv fetch-taxonomy [--refresh | --older-than DAYS]` asks iNat for the
family, order, class and phylum of every genus the records use (1 request/s,
resumable, kept in `data/taxonomy/inat_genera.sqlite`); `mv taxonomy` reports what
that changes and writes the genera in doubt to `data/reports/taxonomy-doubts.csv` (no
network). `mv refresh` asks only about genera no lookup has answered yet, for at most
20 minutes (`--taxonomy-minutes`; the next refresh carries on), and goes on without it
when iNat is down; `--no-taxonomy` skips it.

## Commands by group

### Records, photos and labels

| Command | What it does |
| --- | --- |
| `export-records` | pull validated records from mycomap.org (read-only) |
| `record-sources` | what an export would change in record sources (dry run), or `--apply` |
| `fetch-inat` | iNat photo lists, licences and owners |
| `fetch-mo` | Mushroom Observer image lists, licences and owners (MO's pace) |
| `mo-image-list` / `import-mo-zip` | list the MO images wanted; store them from a zip of MO's image store |
| `download-photos` | download photos within iNat's limits |
| `copy-photos` | copy photos already held from one store to another, hash-checked |
| `fetch-taxonomy` / `taxonomy` | iNat's family, order, class, phylum per genus; what it changes |
| `name-spellings` | names written more than one way (read-only) |
| `guests` | records whose DNA name is a yeast or parasite of the fungus in the photo |
| `permissions` | photographers' answers from mycomap.org |
| `refresh-licenses` | re-read photo licences from iNat |
| `contributors` | the contributor list as CSV (private) |
| `status` | counts of records, photos and licences |
| `refresh` | the weekly loop |

### Models and evaluation

| Command | What it does |
| --- | --- |
| `embed` / `screen` | vectors from a frozen backbone; embed and compare several candidates |
| `archive-embeddings` | set a backbone's vectors aside, e.g. to embed again at another size |
| `finetune` | fine-tune a backbone's last blocks (smoke test locally; full runs on AWS) |
| `compare` | score backbones × methods on the same photos; saved to the scoreboard |
| `scoreboard` / `models` | saved results; known backbones and methods |
| `standard-summary` | the standard tables (top 1/3/5/10, species in the broad sense, depth bands) |
| `calibrate-sets` | fit "likely" name lists that hold the right name about 9 times in 10 |
| `inat-baseline` | add iNat's computer vision to a comparison, on the same records |
| `fungus-gate` | the not-a-fungus gate in front of Identify |
| `int8-check` | score a release with float16 vs int8 reference vectors |
| `model-sanity` | the pre-declared sanity gate for a new model ([retrain-v1.md](retrain-v1.md)) |
| `candidates` / `predict-pending` / `prospective` | predict records still waiting for DNA, and check later |

### Benchmarks and research

| Command | What it does |
| --- | --- |
| `holdout`, `heldout` | records held out of everything; held-out benchmarks (freeze, fetch, predict, report) |
| `dataset`, `manifest-freeze` | frozen, verifiable dataset releases ([dataset-release.md](dataset-release.md)) |
| `external`, `picek-train`, `picek-bench` | published fungi classifiers and their recipe ([replications/fungitastic](../replications/fungitastic/README.md)) |
| `build-occurrence`, `occurrence-names`, `occurrence-exclusions`, `tune-occurrence` | the occurrence prior from iNat open data |
| `scoreboard-export` / `-import`, `benchmark-export` / `-import` | move aggregate results to the server (no record ids, names, observers or places); `benchmark-export --comparison <id> [--split val\|test] [--sealed]` publishes a comparison on a frozen dataset release, marked `reproduced-on-<release>` |

### Cloud and serving

| Command | What it does |
| --- | --- |
| `aws-policies` | print the IAM policies for your bucket and region |
| `aws-launch-downloader`, `aws-pull-manifest` | photos into S3 from a self-terminating instance; merge its results |
| `aws-launch-trainer`, `aws-pull-trainer` | embed, fine-tune and compare on a GPU instance; bring the run home |
| `aws-launch-dataset-build` | cut a dataset release on an instance in the bucket's region |
| `aws-backup` | copy the manifest, embeddings and reports to S3 |
| `serve` | the API and the built web app (port 8010) |
| `release`, `pull-release` | publish what the site serves; download, verify and switch to it on the server |
| `nightly` | the server's nightly update |

Commands named `aws-*-job` run on the instances themselves.
