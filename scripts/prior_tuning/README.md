# Prior tuning (exp/prior-tuning, 2026-10-09)

The runs behind `docs/experiments/2026-10-09-prior-tuning.md`. Reproducibility:
exploratory-pre-freeze (re-run on Dataset release v1; the chosen setting is provisional).

One command does it all and writes run.json (manifest, reference hash, code commit, grid):

    python scripts/prior_tuning/run_all.py --manifest <manifest.sqlite> --out <private dir>
    python scripts/prior_tuning/run_all.py --release <release dir> --out <private dir> --confirm-test

The steps it runs, each usable alone: CPU only, on a COPY of the
manifest (`sqlite3` backup), never the shared one. The score and component files they write
hold .org's true coordinates or values derived from them: keep them in a private scratch
folder, never in the repo.

1. `score_sets.py <manifest copy> <out>`: nearest and nearest+mean scores of every held-out
   dev and test record against the reference index (full species vectors).
2. `build_components.py <manifest copy> <out> <split> [part nparts]`, then
   `build_components.py merge <out> <split> <nparts>`: every prior term the declared grid
   (`mycomap_vision.priortune`) needs, per record and candidate.
3. `run_tuning.py <manifest copy> <out> dev`: every setting, nested 5-fold CV by observer,
   the standard tables, calibration and range edges; writes the choice.
   `run_tuning.py <manifest copy> <out> test`: the one chosen setting on test, once.

numpy only: importing torch next to OpenBLAS crashed these runs (access violation), so the
scripts block it.
