# CLAUDE.md

Instructions for AI coding assistants (Claude Code) working in this repository. Human
contributors: the rules below apply to you too; README.md is the place to start.

MycoMap Vision: a photo identifier for fungi fine-tuned on, and answering from,
DNA-validated MycoMap records (the base model's pretraining is not DNA-checked). See README.md for the pipeline and docs/PLAN.md for the plan.
Deployment-specific notes live in `CLAUDE.local.md` (git-ignored) when present.

## Rules

- **Read-only toward every outside system.** Records are read from a read-only
  SQL route (`MV_ORG_SQL_SSH_HOST`). Never write to mycomap.org, iNat or any
  other outside service from this repo. One exception (Steve, 2026-10-18): a Human
  Feedback decision about a whole name opens a Name Update request on mycomap.org
  (`feedback_org.py`), which changes nothing outside .org until a person with Name
  Update rights there presses Fix; renames come back through .org's name-change log
  into the nightly layer only (`name_changes.py`), never a frozen release.
- **Respect iNat's limits.** API: 1 request/s at most (iNat asks for ≤60/min).
  Media: CC photos from the AWS Open Data bucket; all-rights-reserved photos from
  `static.inaturalist.org` under 5 GB/hour and 24 GB/day (we cap at 4 and 20).
  Only the project owner may change the day cap for a run (`--static-day-gb`);
  the 4 GB hourly cap is never raised.
  Always send `config.USER_AGENT`. Never put a user's email in a header or URL.
- **Keep provenance.** Every photo keeps owner, license, license history and
  hash. All-rights-reserved photos are for training only while permission is
  sought; never publish or redistribute them. Identify results show a photo only
  if it is Creative Commons licensed or its owner granted permission on
  mycomap.org, and leave out the photos of anyone who declined or withdrew
  (`identify.pick_shown` + `permissions.py`; the rule for use is in
  `evaluate.load_records`). New code that shows or uses photos must go through them.
  One narrow exception (Steve, 2026-10-18): Human Feedback reviewers with access to the
  item (mycomap.org admins, and project leaders and contributors for their projects'
  items) may see all-rights-reserved photos on the signed-in Human Feedback review
  pages, and nowhere else. They are served only by the authenticated
  `/api/feedback/photo/` endpoint, which checks the reviewer's access to a queued record
  of the photo, from Vision's own copy or fetched from the photo's source (iNat, MO;
  paced and budgeted, `photo_review.SourceFetcher`), with
  `Cache-Control: private, no-store` and no public URL; never to anonymous visitors,
  API callers, or anyone without access to the item; and never a photo whose owner
  declined or withdrew (or an MO photo marked not for export). The rule lives in
  `photo_review.ReviewPhotos`; it does not extend to public pages, Identify or the API.
- **Keep the data private.** `data/` (photos, manifest, embeddings, contributor
  lists) and the S3 bucket are never committed, published or made public. Trained
  models and embedding indexes count as data until their photo permissions allow
  release.
- **Never expose coordinates.** Records can carry true locations of observations
  that are obscured on iNat (often rare or sensitive species). No API response,
  page, export or model may reveal them.
- **Site runs are kept, under their own link** (Steve, 2026-10-20; `runs.py`). Every
  identification on the site is kept at `/r/<id>`: the answer without anyone's own
  collections, 1024 px photo copies re-encoded without EXIF, the place only to 0.1 degree,
  and who ran it (mycomap.org id and name, never served). Open to anyone with the link,
  never listed or indexed; deleted only by the person who ran it or a mycomap.org admin; no
  expiry. Stored in the bucket's `site-runs/` (the box may Put/Get/Delete there and
  nothing more). API v1 and the app's sessions keep nothing. New code that keeps or shows
  a run goes through `runs.py`. A run's link-preview card (`linkcards.py`, for Facebook, X,
  iMessage) carries only its answer and its own photos: never the place, the date, the
  observation or who ran it.
- **Every answer is logged, insert-only** (Steve, 2026-10-20; `identifylog.py`, docs/api.md
  §11). One row per answer given or served, on every path, in `state/identify-log.sqlite`,
  for the paper's truth-independence check: when, the path, the observation, fresh or kept,
  versions, who asked, the run link; no photo or answer. Never UPDATE or DELETE it (the
  table refuses); write only through `log_answer` in `api.py`, which queues and never
  raises. New code that answers an identification logs it there. Kept until the paper
  closes.
- **Unpublished results stay out of this public repo.** Experiment entries and the paper
  draft live in the private research checkout (`config.RESEARCH_DIR`: `MV_RESEARCH_DIR`, or
  `../mycomap-vision-research`), under `experiments/` and `paper/`; the site shows them to
  signed-in members. `tests/test_public_repo.py` refuses `docs/experiments/` and `docs/paper/`.
- **No secrets in the repo.** Settings come from the environment or the
  git-ignored `.env` (see `.env.example`). No keys, hostnames of private
  services or bucket names in code.
- **Labels are DNA-validated only** ("green in a project" on mycomap.org). No iNat
  or Mushroom Observer community IDs as labels, not even as weak labels, for now.
- **Evaluate honestly.** Test sets are split by time (the newest weeks of green
  records), and results are also reported per observer and per project, because
  weekly batches are lumpy. Never evaluate on a random split.

## Conventions

- Python 3.11+, venv at `.venv`, `pip install -r requirements/dev.txt` then
  `pip install -e . --no-deps`. Web app: `web/`, pnpm.
- Tests next to every change: `.venv/Scripts/python -m pytest -q`. Name tests
  after the rule they protect, and check a new test fails when the guard is broken.
- Windows host: use Git Bash or PowerShell, not WSL.
