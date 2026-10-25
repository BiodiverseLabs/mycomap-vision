# Docs

Every document in this folder, one line each. Start with the repository's
[README](../README.md). Rules binding on every contributor are in [CLAUDE.md](../CLAUDE.md).

**Historical** marks a design note or measurement written for a decision that has since
been made: it is kept as the record of why, and a "status as of" note at its top says
what changed. Its figures and plans are not today's.

| Doc | Kind | What it covers |
| --- | --- | --- |
| [data.md](data.md) | reference | which records are used; labels, photos, licences and photographer permissions |
| [cli.md](cli.md) | reference | the pipeline and every `mv` command |
| [api.md](api.md) | reference | the public API (v1): credentials, requests, the answer, errors, quotas |
| [deploy.md](deploy.md) | operations | the vision.mycomap.org server: setup, releases, nightly updates, memory |
| [dataset-release.md](dataset-release.md) | specification | frozen, verifiable dataset releases (v1 cut 2026-10-12), errata, the sealed test |
| [retrain-v1.md](retrain-v1.md) | specification | the runbook and pre-declared evaluation for the retrain on dataset release v1 |
| [PLAN.md](PLAN.md) | working plan | the plan, decisions and findings, newest phase first; older sections are history |
| [capacity.md](capacity.md) | historical (2026-10-11) | CPU and memory cost of an identification on the original 4 GB box, and the upgrade options |
| [atlas-environment.md](atlas-environment.md) | historical design (2026-10-12) | using Atlas's suitability maps and environment layers as a place prior and an extra input |
| [design-your-collections.md](design-your-collections.md) | historical design (2026-10-18) | the Identify result's "Your collections" and best-match-per-photo panels, as specified before they were built |

Elsewhere in the repository: [replications/fungitastic/](../replications/fungitastic/README.md)
(the comparison with the Picek group's classifiers), [deploy/aws/](../deploy/aws/README.md)
(the self-terminating GPU and download instances) and [SECURITY.md](../SECURITY.md).
