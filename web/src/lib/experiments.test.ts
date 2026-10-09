import { test } from "node:test";
import assert from "node:assert/strict";
import { awaitsDecision, filterExperiments, statusCounts, type ExperimentSummary } from "./experiments";

const e = (slug: string, date: string, status: ExperimentSummary["status"], decision = "pending"): ExperimentSummary => ({
  title: slug, slug, date, status, question: "q", headline: "h", verdict: "v", decision,
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

test("an entry whose decision is pending is flagged as waiting", () => {
  assert.equal(awaitsDecision(LIST[1]), true);
  assert.equal(awaitsDecision(LIST[0]), false);
  assert.equal(awaitsDecision(e("x", "2026-10-09", "running", "pending (launch after relabel)")), true);
});
