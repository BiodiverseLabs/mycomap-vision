import { test } from "node:test";
import assert from "node:assert/strict";
import { awaitsDecision, datasetChoices, datasetOf, defaultDataset, filterExperiments, reproducibilityWords, statusCounts,
         type ExperimentSummary } from "./experiments";

const e = (slug: string, date: string, status: ExperimentSummary["status"], decision = "pending"): ExperimentSummary => ({
  title: slug, slug, date, status, reproducibility: "exploratory-pre-freeze", question: "q",
  headline: "h", verdict: "v", decision,
});

const LIST = [e("a", "2026-09-28", "adopted", "Steve: yes"), e("b", "2026-10-09", "done"),
              e("c", "2026-10-08", "dropped", "Steve: no"), e("d", "2026-10-09", "running")];

test("experiments list newest first, and filter to one status", () => {
  assert.deepEqual(filterExperiments(LIST, null).map((x) => x.slug), ["b", "d", "c", "a"]);
  assert.deepEqual(filterExperiments(LIST, "dropped").map((x) => x.slug), ["c"]);
  assert.deepEqual(filterExperiments(LIST, "planned"), []);
});

test("the filter shows how many entries each status has", () => {
  assert.deepEqual(statusCounts(LIST), { planned: 0, running: 1, done: 1, adopted: 1, dropped: 1 });
});

test("a pre-v1 result is labelled with its dataset; a reproduced one names its release", () => {
  const before = reproducibilityWords("exploratory-pre-freeze");
  assert.deepEqual([before.label, before.provisional], ["Pre-v1 exploratory dataset", true]);
  assert.match(before.note, /re-checked on dataset release v1/);
  const after = reproducibilityWords("reproduced-on-v1");
  assert.deepEqual([after.provisional, after.label], [false, "Dataset release v1"]);
});

test("an entry whose decision is pending is flagged as waiting", () => {
  assert.equal(awaitsDecision(LIST[1]), true);
  assert.equal(awaitsDecision(LIST[0]), false);
  assert.equal(awaitsDecision(e("x", "2026-10-09", "running", "pending (launch after relabel)")), true);
});

test("an exploratory entry re-run on v1 stays pre-v1 and points at its re-run", () => {
  const w = reproducibilityWords("exploratory-pre-freeze", "v1");
  assert.equal(w.label, "Pre-v1 exploratory dataset · re-run on v1");
  assert.match(w.note, /re-run on dataset release v1 holds the numbers that count/);
  assert.equal(reproducibilityWords("reproduced-on-v1", "v1").label, "Dataset release v1");
});

test("the page shows one dataset at a time: v1 first once it has entries, pre-v1 one choice away", () => {
  const rerun = { ...e("r", "2026-10-13", "done"), reproducibility: "reproduced-on-v1" };
  const old = { ...e("o", "2026-10-09", "adopted", "Steve: yes"), reproduced_by: "r", reproduced_on: "v1" };
  const list = [rerun, old, e("x", "2026-10-08", "done")];
  assert.deepEqual([datasetOf(rerun), datasetOf(old)], ["v1", "pre-v1"]);
  assert.deepEqual(datasetChoices(list).map((c) => [c.id, c.n]), [["v1", 1], ["pre-v1", 2]]);
  assert.equal(defaultDataset(list), "v1");
  assert.equal(defaultDataset([old]), "pre-v1", "until v1 has an entry");
  assert.deepEqual(filterExperiments(list, null, "pre-v1").map((x) => x.slug), ["o", "x"]);
  assert.deepEqual(filterExperiments(list, "adopted", "v1"), []);
});
