// The v1 / v2 dropdown's choices and wording (modelScores.ts).
import { test } from "node:test";
import assert from "node:assert/strict";
import { DAILY_V1_ONLY, DAILY_V2 } from "./dailyScores.fixture";
import { defaultModel, modelOptions, rangeText, scoredDays, scoresFor, V1, withN } from "./modelScores";

test("v1 is always a choice; a newer model only once the server has its scores", () => {
  assert.deepEqual(modelOptions(undefined).map((o) => o.id), ["v1"]);
  assert.deepEqual(modelOptions(DAILY_V1_ONLY).map((o) => o.id), ["v1"]);
  assert.deepEqual(modelOptions(DAILY_V2), [{ id: "v1", label: "Vision v1" }, { id: "v2", label: "Vision v2" }]);
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
  assert.deepEqual(modelOptions(d).map((o) => o.label), ["Vision v1", "Vision v2", "Vision v2-open"]);
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
