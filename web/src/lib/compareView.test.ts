import { test } from "node:test";
import assert from "node:assert/strict";
import { bandShares, bars, comparisons, incompleteRun, releaseResultsPending, scoreboardSections, defaultComparison, depthRows, groupRelease, heat, heldoutBars, heldoutDepth, heldoutKs,
         onRelease, orderBenchmarks,
         kChoices, splitModelKey, type PublishedBenchmark } from "./compareView";
import type { RunReport, ScoreRun } from "./api";

const run = (id: number, backbone: string, method: string, over: Partial<ScoreRun> = {}): ScoreRun => ({
  id, comparison_id: "c", backbone, method, cutoff: "2026-09-07", test_days: 28, n_reference: 100,
  n_test: 10, record_set: "rs1", species_top1: 0.3, genus_top1: 0.7, family_top1: 0.8,
  species_top1_first_photo: null, code_version: null, created_at: "", ...over,
});

const std = (top1: number, top3: number, top5: number, top10: number, n = 10) =>
  ({ n, top1, top3, top5, top10 });

function report(withStandard: boolean, speciesTop1: number): RunReport {
  const all = { n: 10, top1: speciesTop1, top5: speciesTop1 + 0.2 };
  const allPhotos: Record<string, unknown> = {
    family: { all }, genus: { all }, species: { all },
  };
  if (withStandard) {
    allPhotos.standard = {
      k: [1, 3, 5, 10], bands: ["0", "1-4", "5-19", "20-99", "100+"],
      species: { all: std(speciesTop1, 0.5, 0.6, 0.7), "1-4": std(0.1, 0.2, 0.3, 0.4, 4),
                 "100+": std(0.9, 1, 1, 1, 6) },
      genus: { all: std(0.7, 0.8, 0.9, 0.95) }, family: { all: std(0.8, 0.9, 0.95, 1) },
    };
  }
  return { backbone: "b", method: "m", all_photos: allPhotos, first_photo_only: {} } as unknown as RunReport;
}

test("only models measured on the same records share a chart, most recently measured first", () => {
  const runs = [run(1, "ft", "nearest", { created_at: "2026-10-08" }), run(2, "ft", "species-mean"),
                run(3, "ft", "nearest", { record_set: "rs0", cutoff: "2026-09-20", created_at: "2026-10-06" }),
                run(4, "external:inat-cv", "combined-max", { comparison_id: "other" })];
  const groups = comparisons(runs);
  assert.deepEqual(groups.map((g) => g.map((r) => r.id)), [[1, 2, 4], [3]],
    "an iNat row imported under its own comparison id still joins its records' chart; "
    + "a later cutoff does not outrank a more recent measurement");
});

test("two comparisons of the same records show each approach once, its newest run", () => {
  const runs = [run(1, "ft", "nearest", { comparison_id: "a", created_at: "2026-10-06" }),
                run(2, "ft", "nearest", { comparison_id: "b", created_at: "2026-10-07" }),
                run(3, "ft", "nearest-mix", { comparison_id: "b", created_at: "2026-10-07" })];
  assert.deepEqual(comparisons(runs)[0].map((r) => r.id), [2, 3]);
});

test("the first chart is the newest that scores the served model beside another approach", () => {
  const lone = [run(9, "ft", "nearest", { record_set: "new" })];
  const others = [run(5, "dino", "nearest", { record_set: "x" }), run(6, "dino", "species-mean", { record_set: "x" })];
  const pair = [run(1, "ft", "nearest"), run(2, "ft", "species-mean")];
  assert.equal(defaultComparison([lone, others, pair], ["ft"]), 2);
  assert.equal(defaultComparison([lone, others, pair]), 1, "nothing served: any pair");
  assert.equal(defaultComparison([lone]), 0);
});

test("bars rank approaches at the chosen rank and top-k, and mark the outside baseline", () => {
  const runs = [run(1, "ft", "nearest"), run(2, "external:inat-cv", "combined-max")];
  const reports = new Map([[1, report(true, 0.34)], [2, report(true, 0.4)]]);
  const b = bars(runs, reports, "species", 1, ["ft"]);
  assert.deepEqual(b.map((x) => [x.id, x.value, x.external, x.served]),
    [[2, 0.4, true, false], [1, 0.34, false, true]]);
  assert.deepEqual(bars(runs, reports, "species", 10).map((x) => x.value), [0.7, 0.7]);
});

test("the served marker follows the server's default method", () => {
  const runs = [run(1, "ft", "nearest"), run(2, "ft", "nearest+mean")];
  const reports = new Map([[1, report(true, 0.34)], [2, report(true, 0.38)]]);
  const served = (m: string) => bars(runs, reports, "species", 1, ["ft"], m).filter((b) => b.served).map((b) => b.id);
  assert.deepEqual(served("nearest+mean"), [2]);
  assert.deepEqual(served("nearest"), [1]);
});

test("runs saved before the standard summary still chart at top 1 and 5, never invent top 3 or 10", () => {
  const runs = [run(1, "ft", "nearest"), run(2, "ft", "species-mean")];
  const reports = new Map([[1, report(true, 0.34)], [2, report(false, 0.3)]]);
  assert.deepEqual(kChoices(runs, reports), [1, 3, 5, 10], "offered once any run has them");
  assert.deepEqual(kChoices(runs.slice(1), reports), [1, 5]);
  const top3 = bars(runs, reports, "species", 3);
  assert.equal(top3.find((x) => x.id === 2)?.value, null);
  assert.equal(bars(runs, reports, "species", 5).find((x) => x.id === 2)?.value, 0.5);
  assert.deepEqual(kChoices(runs.slice(0, 1), reports), [1, 3, 5, 10]);
});

test("the depth table lists runs with the standard summary and counts the rest", () => {
  const runs = [run(1, "ft", "nearest"), run(2, "ft", "species-mean")];
  const reports = new Map([[1, report(true, 0.34)], [2, report(false, 0.3)]]);
  const { rows, missing } = depthRows(runs, reports);
  assert.equal(missing, 1);
  assert.deepEqual(rows[0].cells.map((c) => c?.top1 ?? null), [null, 0.1, null, null, 0.9]);
});

test("heat shading darkens with the share and keeps its text readable", () => {
  assert.equal(heat(null).bg, "transparent");
  assert.equal(heat(0).ink, "#0b0b0b");
  assert.equal(heat(1).ink, "#ffffff");
  assert.notEqual(heat(0.2).bg, heat(0.8).bg);
});


const bench = (withSummary: boolean): PublishedBenchmark => ({
  benchmark: "heldout-x", split: "test", sealed: false, records: 100, scored_records: 99,
  imported_at: "",
  models: {
    "ft/nearest": { records: 99, species: { top1: { n: 99, rate: 0.48, ci95: [0.47, 0.49] },
                                            top5: { n: 99, rate: 0.73 } } },
    "ft/nearest+prior@org": { records: 99, species: { top1: { n: 99, rate: 0.51 } } },
  },
  species_by_reference_records: { "ft/nearest": { "1-4": { species: { n: 10, rate: 0.14 } } } },
  summary: withSummary ? { models: {
    "ft/nearest": { rows: { "species strict": { top1: { n: 99, rate: 0.48 }, top3: { n: 99, rate: 0.66 },
                                                top5: { n: 99, rate: 0.73 }, top10: { n: 99, rate: 0.8 } } } },
    "ft/nearest+prior@org": { rows: { "species strict": { top1: { n: 99, rate: 0.51 }, top3: null,
                                                          top5: null, top10: null } } },
  } } : null,
});

test("held-out model keys keep their place on the method", () => {
  assert.deepEqual(splitModelKey("ft/nearest+prior@org"), { backbone: "ft", method: "nearest+prior@org" });
  assert.deepEqual(splitModelKey("external:inat-cv/combined-max"),
    { backbone: "external:inat-cv", method: "combined-max" });
});

test("held-out bars rank models, carry the interval, and offer top 3/10 only when stored", () => {
  const b = bench(false);
  assert.deepEqual(heldoutKs(b), [1, 5]);
  const top1 = heldoutBars(b, "species", 1);
  assert.deepEqual(top1.map((x) => [x.key, x.value]), [["ft/nearest+prior@org", 0.51], ["ft/nearest", 0.48]]);
  assert.deepEqual(top1[1].ci, [0.47, 0.49]);
  assert.equal(heldoutBars(b, "species", 3)[0].value, null);
  assert.deepEqual(heldoutKs(bench(true)), [1, 3, 5, 10], "one model stored ten deep");
  assert.equal(heldoutBars(bench(true), "species", 10).find((x) => x.key === "ft/nearest+prior@org")?.value,
    null, "the one stored five deep is not measured at 10");
  assert.equal(heldoutBars(bench(true), "species", 3).find((x) => x.key === "ft/nearest")?.value, 0.66);
});

test("held-out depth rows use the standard bands", () => {
  const rows = heldoutDepth(bench(false));
  assert.deepEqual(rows[0].cells.map((c) => c?.rate ?? null), [null, 0.14, null, null, null]);
});

test("v1 results come first; pre-freeze benchmarks follow as development history", () => {
  const old = { ...bench(true), benchmark: "heldout-2026-10-08", reproducibility: "exploratory-pre-freeze" };
  const legacy = { ...bench(true), benchmark: "heldout-older" };          // imported before the field existed
  const val = { ...bench(true), benchmark: "v1", split: "val", reproducibility: "reproduced-on-v1" };
  const test_ = { ...bench(true), benchmark: "v1-test", sealed: true, reproducibility: "reproduced-on-v1-test" };
  assert.deepEqual([old, legacy, val, test_].map(onRelease), [false, false, true, true]);
  assert.deepEqual(orderBenchmarks([old, val, legacy, test_]).map((b) => b.benchmark),
    ["v1-test", "v1", "heldout-2026-10-08", "heldout-older"]);
});

test("a comparison measured on a release is shown first, over a newer pre-freeze one", () => {
  const preFreeze = [run(1, "ft", "nearest"), run(2, "ft", "nearest+mean")];
  const v1 = [run(3, "ft-v1", "nearest", { record_set: "rs2", dataset_release: "v1" }),
              run(4, "ft-v1", "nearest+mean", { record_set: "rs2", dataset_release: null })];
  assert.equal(groupRelease(v1), "v1");
  assert.equal(groupRelease(preFreeze), null);
  assert.equal(defaultComparison([preFreeze, v1], ["ft"]), 1);
  assert.equal(defaultComparison([preFreeze], ["ft"]), 0, "no release yet: as before");
});

test("the Results page says release results are still to come until one is published", () => {
  const preFreeze = [run(1, "ft", "nearest"), run(2, "ft", "nearest+mean")];
  const old = { ...bench(true), reproducibility: "exploratory-pre-freeze" };
  const val = { ...bench(true), benchmark: "v1", split: "val", reproducibility: "reproduced-on-v1" };
  assert.equal(releaseResultsPending([], []), true);
  assert.equal(releaseResultsPending(preFreeze, [old]), true, "development history only");
  assert.equal(releaseResultsPending(preFreeze, [old, val]), false, "an imported v1 benchmark");
  const onV1 = [run(3, "ft-v1", "nearest", { record_set: "rs2", dataset_release: "v1" })];
  assert.equal(releaseResultsPending([...preFreeze, ...onV1], []), false, "a v1 comparison");
});

test("each depth band's head shows its share of all the comparison's finds, in whole percent", () => {
  // Dataset release v1's validation split, species (run 169 on the box): 12,798 finds.
  assert.deepEqual(bandShares([649, 1391, 2864, 5736, 2158]), [5, 11, 22, 45, 17]);
  assert.deepEqual(bandShares([0, 1, 1, 2, 0]), [0, 25, 25, 50, 0]);
  // A band whose count is unknown has no share; the others still share the known finds.
  assert.deepEqual(bandShares([null, 50, 50, null, null]), [null, 50, 50, null, null]);
  assert.deepEqual(bandShares([null, null, null, null, null]), [null, null, null, null, null]);
  assert.deepEqual(bandShares([0, 0, 0, 0, 0]), [null, null, null, null, null]);
});

test("the scoreboard puts every dataset release comparison above development history, frozen against fine-tuned first", () => {
  const v1 = (id: number, backbone: string, method: string, set: string, n_test: number, species: number,
              created = "2026-10-19") =>
    run(id, backbone, method, { record_set: set, n_test, dataset_release: "v1", species_top1: species,
                                cutoff: "2026-08-17", created_at: created, comparison_id: `c-${id}` });
  const runs = [
    // The box's v1 comparisons (runs 168/169: 12,874 records, after the erratum; 163/166:
    // 13,354 records, frozen and fine-tuned), each saved under several comparison ids.
    v1(169, "bioclip-2-336-ft-20261018-171420", "nearest+mean-records-balanced+place", "36fd87", 12874, 0.5917, "2026-10-19T21"),
    v1(168, "bioclip-2-336-ft-20261018-171420", "nearest+mean-records-balanced", "36fd87", 12874, 0.5632, "2026-10-19T19"),
    v1(166, "bioclip-2-336-ft-20261018-171420", "nearest+mean", "d7bd13", 13354, 0.5726, "2026-10-19T06"),
    v1(163, "bioclip-2-336", "nearest+mean", "d7bd13", 13354, 0.5279, "2026-10-19T05"),
    v1(145, "bioclip-2-ft-20261013-003428", "nearest+mean", "d7bd13", 13354, 0.5513, "2026-10-13"),
    run(139, "bioclip-2-ft-20261007-165400", "nearest+mean", { record_set: "4ef7b0", n_test: 1152, comparison_id: "20261008-012435-4ef7b0" }),
    run(132, "external:inat-cv", "vision-max", { record_set: "4ef7b0", n_test: 1152, comparison_id: "20261008-012435-4ef7b0", species_top1: 0.0107 }),
    run(133, "external:inat-cv", "combined-max", { record_set: "4ef7b0", n_test: 1152, comparison_id: "20261008-012435-4ef7b0", species_top1: 0.0116 }),
  ];
  const { release, history } = scoreboardSections(runs);
  assert.deepEqual(release.map((g) => g.map((r) => r.id)), [[166, 145, 163], [169, 168]]);
  assert.deepEqual(history.map((g) => g.map((r) => r.id)), [[139]]);
});

test("iNaturalist's incomplete development run of 8 October is left out, and only that one", () => {
  assert.ok(incompleteRun(run(132, "external:inat-cv", "vision-max", { comparison_id: "20261008-012435-4ef7b0" })));
  assert.ok(!incompleteRun(run(139, "bioclip-2", "nearest", { comparison_id: "20261008-012435-4ef7b0" })));
  assert.ok(!incompleteRun(run(116, "external:inat-cv", "vision-max", { comparison_id: "20261006-183858-b9ca61" })));
});
