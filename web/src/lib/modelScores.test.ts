// The v1 / v2 dropdown's choices and wording (modelScores.ts).
import { test } from "node:test";
import assert from "node:assert/strict";
import { DAILY_V1_ONLY, DAILY_V2, fortyDays } from "./dailyScores.fixture";
import { CHART_DAYS, defaultModel, depthRows, modelOptions, rangeText, scoredDays, scoresFor, topKLadder, V1, withN } from "./modelScores";

test("v1 is always a choice; a newer model only once the server has its scores", () => {
  assert.deepEqual(modelOptions(undefined).map((o) => o.id), ["v1"]);
  assert.deepEqual(modelOptions(DAILY_V1_ONLY).map((o) => o.id), ["v1"]);
  assert.deepEqual(modelOptions(DAILY_V2), [{ id: "v1", label: "MycoMap Vision v1" }, { id: "v2", label: "MycoMap Vision v2" }]);
  // Listed but with no scores sent: not a choice.
  assert.deepEqual(modelOptions({ ...DAILY_V2, models: {} }).map((o) => o.id), ["v1"]);
});

test("the dropdown opens on v2 when the server offers it, else on v1", () => {
  assert.equal(defaultModel(DAILY_V2), "v2");
  assert.equal(defaultModel(DAILY_V1_ONLY), V1);
  assert.equal(defaultModel(undefined), V1);
  assert.equal(defaultModel({ ...DAILY_V2, models: {} }), V1, "a default it can't show falls back to v1");
});

test("a later model joins from the server's answer alone", () => {
  const open = { ...DAILY_V2.models.v2, id: "v2-open" };
  const d = { ...DAILY_V2, available: ["v1", "v2", "v2-open"], models: { ...DAILY_V2.models, "v2-open": open } };
  assert.deepEqual(modelOptions(d).map((o) => o.label), ["MycoMap Vision v1", "MycoMap Vision v2", "MycoMap Vision v2-open"]);
  assert.equal(scoresFor(d, "v2-open")?.id, "v2-open");
});

test("v1 never reads numbers from the daily scores", () => {
  assert.equal(scoresFor(DAILY_V2, V1), null);
});

test("a daily rate is never written without its n, and the range is said plainly", () => {
  const day = DAILY_V2.models.v2.days[1];
  assert.equal(withN(day.species), "62% (n = 152)");
  assert.equal(withN(day.species, 1), "61.8% (n = 152)");
  assert.equal(withN(DAILY_V2.models.v2.days[0].species), "no records (n = 0)");
  assert.equal(rangeText(DAILY_V2.models.v2.cumulative!.species), "between 58.1% and 67.7%");
  assert.equal(rangeText(day.species), null);
  assert.deepEqual(scoredDays(DAILY_V2.models.v2).map((d) => d.day), ["2026-11-23", "2026-11-24", "2026-11-25"]);
  assert.deepEqual(scoredDays(DAILY_V2.models.v2, 1).map((d) => d.day), ["2026-11-25"]);
});

test("the chart's window is the last 30 calendar days to the newest", () => {
  const m = fortyDays().models.v2;
  assert.equal(CHART_DAYS, 30);
  const drawn = scoredDays(m, CHART_DAYS);
  assert.equal(drawn.length, 30);
  assert.equal(drawn[0].day, m.days[10].day);
  assert.equal(scoredDays(m).length, 40);
});

test("a depth band with too few finds has no rate, whatever the server sent", () => {
  const rows = depthRows(DAILY_V2.models.v2)!;
  assert.deepEqual(rows[0], { band: "0", n: 21, top1: null, tooFew: true });
  assert.deepEqual(rows[1], { band: "1-4", n: 64, top1: 0.2656, tooFew: false });
  const odd = { ...DAILY_V2.models.v2, species_by_depth: [{ band: "0", n: 3, top1: 1, top3: 1, top5: 1, top10: 1, too_few: true }] };
  assert.equal(depthRows(odd)![0].top1, null);
  assert.equal(depthRows({ ...DAILY_V2.models.v2, species_by_depth: null }), null);
});

test("the top-k ladder reads species and genus at each k", () => {
  assert.deepEqual(topKLadder(DAILY_V2.models.v2)?.[1], { k: 3, species: 0.7825, genus: 0.9454 });
  assert.equal(topKLadder({ ...DAILY_V2.models.v2, top_k: null }), null);
});
