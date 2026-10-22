import { test } from "node:test";
import assert from "node:assert/strict";
import {
  CLEAN_LABEL, CLEAN_SECONDS, depthLabel, latestLabel, modelCredit, namesByRecords,
  nearestPoint, recordingSince, servedModel, shortDate, speedView, topKRows,
} from "./howItWorks";
import type { HowGood, ModelsInfo, SpeedDay, Stats } from "./api";

/** `n` consecutive UTC days from 23 September 2026, each with its own made-up test value. */
function days(n: number, start = "2026-09-23"): SpeedDay[] {
  const t0 = Date.parse(`${start}T00:00:00Z`);
  return Array.from({ length: n }, (_, i) => ({
    day: new Date(t0 + i * 86_400_000).toISOString().slice(0, 10),
    seconds_per_photo: Number((1 + (i % 7) * 0.2).toFixed(2)),
  }));
}

test("with no day recorded the speed chart draws nothing", () => {
  assert.deepEqual(speedView([]), { state: "none" });
  assert.deepEqual(speedView(undefined), { state: "none" }, "while loading, or an older server");
  assert.deepEqual(speedView([{ day: "nonsense", seconds_per_photo: 1 },
                              { day: "2026-10-22", seconds_per_photo: Number.NaN }]),
                   { state: "none" }, "a malformed day is never drawn");
});

test("with fewer than three days the days are listed with when recording began, no line", () => {
  const one = speedView([{ day: "2026-10-22", seconds_per_photo: 1.24 }]);
  assert.equal(one.state, "few");
  assert.ok(one.state === "few");
  assert.equal(one.since, "2026-10-22");
  assert.equal(recordingSince(one.since), "Recording since 22 October 2026");
  const two = speedView([{ day: "2026-10-23", seconds_per_photo: 2 }, { day: "2026-10-22", seconds_per_photo: 1 }]);
  assert.ok(two.state === "few");
  assert.deepEqual(two.days.map((d) => d.day), ["2026-10-22", "2026-10-23"], "oldest first");
  assert.equal(two.latest.day, "2026-10-23");
});

test("thirty days make a line from 0 up, by date, with the clean measurement drawn across", () => {
  const v = speedView(days(30));
  assert.ok(v.state === "chart");
  assert.equal(v.points.length, 30);
  assert.equal(v.points[0].x, 0);
  assert.equal(v.points[29].x, 1);
  assert.equal(v.since, "2026-09-23");
  assert.equal(v.latest.day, "2026-10-22");
  assert.equal(v.yTicks[0].v, 0, "the axis starts at zero");
  assert.equal(v.yTicks[0].y, 1);
  assert.ok(v.yMax >= 2.2 * 1.1 - 1e-9, "room above the slowest day");
  assert.equal(v.yTicks[v.yTicks.length - 1].v, v.yMax);
  assert.equal(v.ref.v, CLEAN_SECONDS);
  assert.ok(Math.abs(v.ref.y - (1 - CLEAN_SECONDS / v.yMax)) < 1e-9);
  assert.equal(CLEAN_LABEL, "Clean measurement, 1.1 s");
  for (const p of v.points) assert.ok(Math.abs(p.y - (1 - p.seconds_per_photo / v.yMax)) < 1e-9);
  assert.deepEqual(v.xTicks.map((t) => t.label)[0], "23 Sep");
  assert.equal(v.xTicks[v.xTicks.length - 1].label, "22 Oct");
  assert.ok(v.xTicks.length <= 5);
});

test("a day without photos leaves its gap in time, and a fast week still shows the reference", () => {
  const v = speedView([{ day: "2026-10-01", seconds_per_photo: 0.4 },
                       { day: "2026-10-02", seconds_per_photo: 0.5 },
                       { day: "2026-10-11", seconds_per_photo: 0.6 }]);
  assert.ok(v.state === "chart");
  assert.ok(Math.abs(v.points[1].x - 0.1) < 1e-9, "1 day of 10, not 1 point of 2");
  assert.ok(v.yMax > CLEAN_SECONDS, "the 1.1 s line fits even when every day was faster");
});

test("the crosshair snaps to the nearest day", () => {
  const pts = [{ x: 0 }, { x: 0.5 }, { x: 1 }];
  assert.equal(nearestPoint(pts, 0.3), pts[1]);
  assert.equal(nearestPoint(pts, -2), pts[0]);
  assert.equal(nearestPoint([], 0.5), null);
});

test("the newest figure says today only when it is today's", () => {
  const d = { day: "2026-10-22", seconds_per_photo: 2.28 };
  assert.equal(latestLabel(d, "2026-10-22"), "2.28 s today");
  assert.equal(latestLabel(d, "2026-10-23"), "2.28 s on 22 Oct");
  assert.equal(shortDate("2026-01-05"), "5 Jan");
});

test("names by record count follow the server's bands, the single-record names first", () => {
  const s = { names: 19406, names_by_records: { "31+": 1411, "1": 8091, "2": 2589, "3-5": 3200, "6-30": 4115 } } as unknown as Stats;
  const rows = namesByRecords(s)!;
  assert.deepEqual(rows.map((r) => [r.label, r.n]),
                   [["1 record", 8091], ["2", 2589], ["3–5", 3200], ["6–30", 4115], ["31+", 1411]]);
  assert.ok(Math.abs(rows[0].share - 8091 / 19406) < 1e-12);
  assert.equal(namesByRecords(undefined), null);
  assert.equal(namesByRecords({ names: 0, names_by_records: {} } as unknown as Stats), null);
});

test("the top-k ladder shows only what the run measured", () => {
  const h: HowGood = { run: null, top_k: { k: [1, 3, 5, 10],
    species: { n: 10, top1: 0.59, top3: 0.75, top5: 0.8, top10: 0.85 },
    genus: { n: 10, top1: 0.86, top3: 0.94, top5: null, top10: 0.97 } } };
  assert.deepEqual(topKRows(h), [{ k: 1, species: 0.59, genus: 0.86 }, { k: 3, species: 0.75, genus: 0.94 },
                                 { k: 5, species: 0.8, genus: null }, { k: 10, species: 0.85, genus: 0.97 }]);
  assert.equal(topKRows({ run: null, top_k: null }), null, "an older run: no ladder");
  assert.equal(topKRows(undefined), null);
  assert.equal(depthLabel("20-99"), "20–99 records");
  assert.equal(depthLabel("0"), "0 records");
});

test("the model card is the served model the results are for", () => {
  const m = { ready: ["a", "b"], methods: [], max_models: 1, backbones: [
    { backbone: "a", spec: "", note: "", embedded_photos: 10, photos_per_second: null },
    { backbone: "b", spec: "", note: "", embedded_photos: 20, photos_per_second: null },
    { backbone: "c", spec: "", note: "", embedded_photos: 30, photos_per_second: null }] } as unknown as ModelsInfo;
  assert.equal(servedModel(m, "b")?.backbone, "b");
  assert.equal(servedModel(m, "c")?.backbone, "a", "a model not served is never the card");
  assert.equal(servedModel(undefined), null);
});

test("the small model credit names the base model, its size, its training data and who made it", () => {
  const c = modelCredit("bioclip-2-336-ft-20261018-171420");
  assert.equal(c.name, "BioCLIP 2, 336 px, fine-tuned (public + non-public training data)");
  assert.equal(c.title, "BioCLIP 2, 336 px, fine-tuned");
  assert.equal(c.detail, "(public + non-public training data)");
  assert.equal(c.madeBy, "Built on BioCLIP 2 by Imageomics, fine-tuned on DNA-verified MycoMap records.");
  assert.equal(modelCredit("dinov2-b14").madeBy, null, "no credit made up for another base");
});
