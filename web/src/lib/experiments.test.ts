import { test } from "node:test";
import assert from "node:assert/strict";
import { awaitsDecision, filterExperiments, reproCounts, reproducibilityWords, reproKind, statusCounts,
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

test("a pre-freeze result reads as exploratory and provisional; a reproduced one names its release", () => {
  const before = reproducibilityWords("exploratory-pre-freeze");
  assert.equal(before.provisional, true);
  assert.match(before.note, /provisional until re-run on dataset release v1/);
  const after = reproducibilityWords("reproduced-on-v1");
  assert.deepEqual([after.provisional, after.label], [false, "Reproducible on release v1"]);
});

test("an entry whose decision is pending is flagged as waiting", () => {
  assert.equal(awaitsDecision(LIST[1]), true);
  assert.equal(awaitsDecision(LIST[0]), false);
  assert.equal(awaitsDecision(e("x", "2026-10-09", "running", "pending (launch after relabel)")), true);
});

test("an exploratory entry re-run on v1 reads as method history, not provisional", () => {
  const w = reproducibilityWords("exploratory-pre-freeze", "v1");
  assert.equal(w.provisional, false);
  assert.match(w.label, /Method history · re-run on release v1/);
  assert.match(w.note, /re-run's numbers are the ones that count/);
  // A reproduced entry's own words don't change with the argument.
  assert.equal(reproducibilityWords("reproduced-on-v1", "v1").label, "Reproducible on release v1");
});

test("the list filters to v1 entries, method history or still-exploratory ones", () => {
  const rerun = { ...e("r", "2026-10-13", "done"), reproducibility: "reproduced-on-v1" };
  const old = { ...e("o", "2026-10-09", "adopted", "Steve: yes"), reproduced_by: "r", reproduced_on: "v1" };
  const list = [rerun, old, e("x", "2026-10-08", "done")];
  assert.deepEqual([reproKind(rerun), reproKind(old), reproKind(list[2])], ["release", "history", "exploratory"]);
  assert.deepEqual(reproCounts(list), { release: 1, history: 1, exploratory: 1 });
  assert.deepEqual(filterExperiments(list, null, "release").map((x) => x.slug), ["r"]);
  assert.deepEqual(filterExperiments(list, "adopted", "history").map((x) => x.slug), ["o"]);
  assert.deepEqual(filterExperiments(list, "adopted", "release"), []);
});
