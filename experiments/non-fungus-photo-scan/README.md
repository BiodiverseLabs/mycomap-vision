# Non-fungus photo scan (2026-10-09)

Standalone, read-only code for docs/experiments/2026-10-09-non-fungus-photo-scan.md.
Nothing here writes to the shared manifest, the benchmark's tables or any outside
service. Every output goes to `data/audits/non-fungus-scan/` (private, never committed).

Run from this folder with `PYTHONPATH=<repo>/src` and `MV_DATA_DIR` set to the data folder:

1. `zs_score.py <manifest>`: zero-shot class probabilities for every stored photo vector,
   from BioCLIP-2's text tower and class prompts (CPU, about 30 s).
2. `frame.py <scratch manifest copy>`: one row per reference photo (the records
   `evaluate.load_records` serves), with its probabilities.
3. `srcmap.py`: reads the read-only export of each green record's `.org` source
   (`ssh mycomap-sql "select observation_id, source from observations where 'yes' in
   (...validation_status_1..3)"` saved as `org-green-source.tsv`).
4. `sheets.py`: numbered contact sheets for the hand check, read from Vision's own photo store.
5. `review_list.py`: the review list CSV and the counts.
6. `rescore.py <scratch manifest copy>`: the held-out dev split against the current
   reference index and against indexes with the flagged photos and/or the mislinked
   records left out, in the standard summary format with paired fixed / broken counts.

Use a scratch copy of the manifest (sqlite backup from a `mode=ro` connection) for
`frame.py` and `rescore.py`: `load_records` and `Identifier` run schema scripts.
