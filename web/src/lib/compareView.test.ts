import { test } from "node:test";
import assert from "node:assert/strict";
import { bars, comparisons, defaultComparison, depthRows, heat, heldoutBars, heldoutDepth, heldoutKs,
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
