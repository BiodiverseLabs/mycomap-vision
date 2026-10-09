import { test } from "node:test";
import assert from "node:assert/strict";
import { bars, comparisons, defaultComparison, depthRows, heat, kChoices } from "./compareView";
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

test("only models measured on the same records share a chart, newest records first", () => {
  const runs = [run(1, "ft", "nearest"), run(2, "ft", "species-mean"),
                run(3, "ft", "nearest", { record_set: "rs0", cutoff: "2026-08-01" }),
                run(4, "external:inat-cv", "combined-max", { comparison_id: "other" })];
  const groups = comparisons(runs);
  assert.deepEqual(groups.map((g) => g.map((r) => r.id)), [[1, 2, 4], [3]],
    "an iNat row imported under its own comparison id still joins its records' chart");
});

test("the first chart is the newest one that compares two or more approaches", () => {
  const lone = [run(9, "ft", "nearest", { record_set: "new", cutoff: "2026-10-01" })];
  const pair = [run(1, "ft", "nearest"), run(2, "ft", "species-mean")];
  assert.equal(defaultComparison([lone, pair]), 1);
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
  assert.deepEqual(kChoices(runs, reports), [1, 5]);
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
