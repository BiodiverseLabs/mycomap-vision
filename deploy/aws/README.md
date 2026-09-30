# AWS setup (one time, done by a person)

Photos live in a private S3 bucket. A small EC2 instance downloads them straight
into the bucket and shuts itself down when done.

Nobody's keys go in the repo or reach an AI assistant: you create a limited user
and enter its key yourself.

## 0. Settings

Put your bucket name, region and AWS profile name in `.env` (see
`.env.example`): `MV_S3_BUCKET`, `MV_AWS_REGION`, `MV_AWS_PROFILE`, and optionally
`MV_INSTANCE_ROLE` (default `mycomap-vision-instance`). Then print the two IAM
policies, filled in for them:

```
.venv/Scripts/mv aws-policies
```

The templates are [`instance-policy.template.json`](instance-policy.template.json)
and [`ops-policy.template.json`](ops-policy.template.json).

## 1. Install the AWS CLI (PowerShell)

```
winget install Amazon.AWSCLI
```

## 2. Instance role (AWS console → IAM → Roles → Create role)

- Trusted entity: **AWS service**, use case **EC2**.
- Skip the managed policies. Name it exactly as `MV_INSTANCE_ROLE`.
- Open the role → **Add permissions → Create inline policy → JSON**, paste the
  printed `instance-policy.json`, name it `vision-bucket`.

It can read and write the vision bucket and nothing else, and cannot delete.

## 3. Ops user (IAM → Users → Create user)

- A name such as `mycomap-vision-ops`, no console access.
- **Attach policies directly → Create policy → JSON**, paste the printed
  `ops-policy.json`, name it, attach it.
- Open the user → **Security credentials → Create access key → Command Line
  Interface**. Keep the window open for step 4.

It can create and use only the vision bucket (not delete it or make it public),
launch instances only when tagged `Project=mycomap-vision`, stop or terminate
only those, and hand them only the role above.

## 4. Enter the key (PowerShell, typed by you)

```
aws configure --profile <MV_AWS_PROFILE>
```

Use your `MV_AWS_REGION`, output `json`. Check it with:

```
aws sts get-caller-identity --profile <MV_AWS_PROFILE>
```

## 5. Optional: budget alarm (Billing → Budgets → Create budget)

A monthly cost budget, e.g. $25, emailing you at 80%.

## Running the downloader

```
.venv/Scripts/mv aws-launch-downloader            # large photos, stops after 120 h at most
```

It creates the bucket (private, encrypted) if needed, uploads the committed code
and a snapshot of the manifest, and starts a `t3.small`. The instance's log is
copied to `s3://<bucket>/runs/<run>/download.log` every 15 minutes, and the
manifest to `s3://<bucket>/manifest/manifest.sqlite` every 10.

The downloader always runs On-Demand: it copies its manifest to S3 only every
10 minutes, so an instance taken back would lose up to 10 minutes of bookkeeping
and could fetch those photos from iNat again, against iNat's hourly and daily caps.

Don't write to the local manifest while it runs. When the instance has gone
(terminated), bring its manifest home:

```
.venv/Scripts/mv aws-pull-manifest
```

## Running the GPU trainer

The trainer embeds every large photo in S3 with the chosen backbones, fine-tunes
some of them, compares every backbone x method on the newest weeks, uploads the
results to `s3://<bucket>/runs/<run>/` and shuts itself down (backstop: max
hours + 30 min). It starts from the Deep Learning Base GPU AMI (Amazon Linux
2023) on a `g6.2xlarge` (one NVIDIA L4, 8 vCPUs, about $1/hour on demand).

One-time setup:

1. **GPU quota** (AWS console, your region → Service Quotas → Amazon Elastic
   Compute Cloud (Amazon EC2) → "Running On-Demand G and VT instances" →
   Request increase). It is counted in vCPUs: 8 covers one g6.2xlarge. New
   accounts often start at 0, and approval can take a day. For `--spot` the
   quota is "All G and VT Spot Instance Requests" (also vCPUs, 8 for one
   g6.2xlarge); either one is enough to run.
2. **Ops policy**: it now also reads the Deep Learning AMI's public parameter
   (`/aws/service/deeplearning/ami/*`). Print it with `mv aws-policies` and
   replace the ops user's policy JSON (IAM → Users → the ops user → Permissions →
   the policy → Edit → JSON). The instance role is unchanged.
3. **For Spot** (once): the ops policy now also lets `RunInstances` create the
   Spot request that comes with a Spot instance
   (`spot-instances-request/*`, still only with the `Project=mycomap-vision`
   tag), and create EC2's Spot service-linked role (`AWSServiceRoleForEC2Spot`,
   nothing else) if the account doesn't have it yet. Replace the ops user's
   policy JSON with the one `mv aws-policies` prints, as in step 2. The role
   itself can also be made by hand, once, by an admin:
   `aws iam create-service-linked-role --aws-service-name spot.amazonaws.com`
   (an error saying it already exists is fine).

### The first run

```
.venv/Scripts/mv aws-launch-trainer
```

With no options this is the recommended first run: BioCLIP 2, then fine-tuning
it (its last 4 blocks, about 2 passes over the photos), then embedding with the
fine-tuned model (`bioclip-2-ft-<run>`), with a 24-hour limit. The spelled-out
form is `--backbones bioclip-2 --finetune bioclip-2 --max-hours 24`.

Before anything is paid for, the launcher prints a time estimate and refuses a
run that would not fit in `--max-hours`. The speeds behind it (`EMBED_RATES`,
`FINETUNE_RATES` in `src/mycomap_vision/trainer.py`) were measured on the laptop
at large photos, and the estimate multiplies them by 1.5 to stay safe on the L4.
For the 593,214 large photos the first run comes to about 16 h (11 h at laptop
speed): 4.5 h to embed, 6 h to fine-tune, 4.5 h to embed again, and an hour for
setup and the comparison. Adding `dinov3-l16-512` would add about 13 h (29 h in
all), so it waits for a second run, or pass `--allow-over-time` to launch anyway.
Once a real run has finished, replace the rates with its own (progress.json
records each stage's speed).

The work goes in this order: each backbone to fine-tune, its fine-tune and the
fine-tuned model's embedding come first; other backbones follow. If time runs
out, it is the extra backbones that are left undone.

### What it ships

The launcher sends the exact commit you are on (`git archive <sha>`) and refuses
when tracked files have uncommitted changes (they would not be sent;
`--allow-dirty` to go anyway) or when the commit is on no remote branch (push it
first; `--allow-unpushed` to go anyway). The instance records that sha as the
run's `code_version` in progress.json, result.json and every scoreboard row.

The instance installs only pinned, hash-checked packages from
`requirements/trainer.txt` (`pip install --require-hashes`); the downloader uses
`requirements/downloader.txt`. torch comes from PyPI with its CUDA runtime as
wheels; the Base AMI brings the NVIDIA driver and no torch, so nothing is
installed over a CUDA build of the image's. torch 2.14 is built for CUDA 13,
which needs NVIDIA driver 580 or newer: the instance checks that torch can see
the GPU and stops at once if it can't. To update the pins (needs PyPI access):

```
.venv/Scripts/python deploy/aws/lock_instance_requirements.py
```

(pip-compile can't be used for these: it resolves for the machine it runs on,
and on Windows it leaves out torch's Linux-only CUDA packages.)

### While it runs, and when it stops

The log is copied to `s3://<bucket>/runs/<run>/train.log` every 15 minutes. As
each stage finishes, its outputs are uploaded at once (embeddings, fine-tuned
weights, a small `index.sqlite`) and `runs/<run>/progress.json` is rewritten: the
stages done, their photo counts, speeds and times, the code version, and the
state (`running`, `comparing`, `finished`, `stopped`, or `interrupted` on Spot).

The job stops itself 45 minutes before `--max-hours` (between batches, or
between fine-tuning steps), uploads what finished, and ends `stopped`; a stage it
was in the middle of is dropped, never uploaded half done. The comparison is
then left to the laptop (`mv compare` after pulling).

A photo that can't be read (corrupt, truncated, missing) is skipped and listed
with the reason in `runs/<run>/skipped/<backbone>.tsv`, and counted in the
progress. If more than 1% of a backbone's photos can't be read, something
systemic is wrong and that backbone fails instead.

### On Spot

```
.venv/Scripts/mv aws-launch-trainer --spot [--spot-max-price 0.60]
```

`--spot` asks for a one-time Spot `g6.2xlarge` instead of On-Demand: the same
instance, usually well under the On-Demand price, but AWS may take it back. With
no `--spot-max-price` the most it pays is the On-Demand price. If EC2 won't start
it, the launcher says why in plain words: no capacity free right now
(`InsufficientInstanceCapacity`: try again later), the Spot quota used up or still
0 (`MaxSpotInstanceCountExceeded`), the price above the cap (`SpotMaxPriceTooLow`),
or the On-Demand quota too low (`VcpuLimitExceeded`). Nothing is running then; the
run's code and manifest in S3 cost next to nothing.

When AWS takes a Spot instance back it gives two minutes' notice in the instance
metadata. The job asks for it every 5 seconds; on notice it drops the stage under
way (never uploaded half done, as at the time limit), writes progress.json with
the state `interrupted` (the time limit gives `stopped`), and exits. The stages it
had finished are already in S3. There is no manifest copy then (no time for it;
the pull reads the run's `index.sqlite`), and no comparison.

### Resuming a run

```
.venv/Scripts/mv aws-launch-trainer --resume <run> [--spot] [--max-hours 24]
```

continues a run that was interrupted, stopped by its time limit, or killed, under
its own run id. It keeps the run's own backbones, fine-tunes, methods, photo size,
test days and manifest (`runs/<run>/manifest-in.sqlite`; other options for those
are ignored, with a warning). The instance downloads the stages progress.json
lists as done (embeddings, fine-tuned weights, index rows), checks them against
the run's index as a pull does, and runs only the others; a done stage whose
files are incomplete is run again. The estimate counts only what is left. It
ships the commit you are on (`runs/<run>/code-resume-<time>.tar.gz`); if that
differs from the run's first commit the launcher says so, and progress.json keeps
each earlier attempt (`attempts`: when, its state and commit). It refuses while
the run's instance is still alive (it could still write to the folder), and a run
that is already complete.

### Bringing it home

```
.venv/Scripts/mv aws-pull-trainer --run <run>
```

It works on a complete run (result.json), a stopped or interrupted one, or one
killed outright (it then reads progress.json), and says which stages are missing
and why, and how to run just those (`--resume <run>`). Each
backbone the run embedded completely replaces the local embeddings (the old ones
move to `data/embeddings-archive/<backbone>-before-<run>/`); a backbone it did
not finish, or whose downloaded copy doesn't match the run's index, never
replaces anything. The run's comparison, when there is one, joins the
scoreboard. What was pulled, with the run's commit, is kept in
`data/aws/run-<run>/pulled.json`. Pulling the same run twice changes nothing,
and pulling again after a run was still going adds the stages finished since.
