# CLAUDE.md

MycoMap Vision: a photo identifier for fungi trained on DNA-validated MycoMap
records. See README.md for the pipeline and data layout.

## Rules

- **mycomap.org is read-only from here.** Records come from `ssh SQL-ROUTE-HOST`
  (DB-enforced read-only, 60 s statement cap). Never write to .org, .com, iNat
  or Mushroom Observer from this repo.
- **Respect iNat's limits.** API: 1 request/s at most (iNat asks for ≤60/min).
  Media: CC photos from the AWS Open Data bucket; all-rights-reserved photos from
  `static.inaturalist.org` under 5 GB/hour and 24 GB/day (we cap at 4 and 20).
  Always send `config.USER_AGENT`. Never put a user's email in a header or URL.
- **Keep provenance.** Every photo keeps owner, license, license history and
  hash. All-rights-reserved photos are for training only while permission is
  sought; never publish or redistribute them.
- **Labels are DNA-validated only** ("green in a project" on .org). No iNat or
  Mushroom Observer community IDs as labels, not even as weak labels, for now.
- **Evaluate honestly.** Test sets are split by time (the newest weeks of green
  records), and results are also reported per observer and per project, because
  weekly batches are lumpy. Never evaluate on a random split.
- `data/` holds contributors' images and live data. Never commit it.

## Conventions

- Python 3.12, venv at `.venv`, `pip install -e ".[dev]"`.
- Tests next to every change: `.venv/Scripts/python -m pytest -q`. Name tests
  after the rule they protect, and check a new test fails when the guard is broken.
- Windows host: use Git Bash or PowerShell, not WSL.
