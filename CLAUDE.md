# CLAUDE.md

MycoMap Vision: a photo identifier for fungi trained on DNA-validated MycoMap
records. See README.md for the pipeline and docs/PLAN.md for the plan.
Deployment-specific notes live in `CLAUDE.local.md` (git-ignored) when present.

## Rules

- **Read-only toward every outside system.** Records are read from a read-only
  SQL route (`MV_ORG_SQL_SSH_HOST`). Never write to mycomap.org, iNat or any
  other outside service from this repo.
- **Respect iNat's limits.** API: 1 request/s at most (iNat asks for ≤60/min).
  Media: CC photos from the AWS Open Data bucket; all-rights-reserved photos from
  `static.inaturalist.org` under 5 GB/hour and 24 GB/day (we cap at 4 and 20).
  Only Steve may raise the day cap for a run (`--static-day-gb`), as he did on
  2026-09-29 to finish the all-rights-reserved photos (MycoMap works with iNat
  regularly); the 4 GB hourly cap is never raised.
  Always send `config.USER_AGENT`. Never put a user's email in a header or URL.
- **Keep provenance.** Every photo keeps owner, license, license history and
  hash. All-rights-reserved photos are for training only while permission is
  sought; never publish or redistribute them. Show one only with its owner's
  grant from mycomap.org, and leave out the photos of anyone who declined or
  withdrew (`permissions.py`; the rule for use is in `evaluate.load_records`).
  New code that shows or uses photos must go through those two.
- **Keep the data private.** `data/` (photos, manifest, embeddings, contributor
  lists) and the S3 bucket are never committed, published or made public. Trained
  models and embedding indexes count as data until their photo permissions allow
  release.
- **Never expose coordinates.** Records can carry true locations of observations
  that are obscured on iNat (often rare or sensitive species). No API response,
  page, export or model may reveal them.
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
