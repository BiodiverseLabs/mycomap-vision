import { test } from "node:test";
import assert from "node:assert/strict";
import {
  depthLabel, easternDay, latestLabel, modelCredit, namesByRecords, nearestPoint, recordingSince,
  rollingAverage, ROLLING_LABEL, servedModel, shortDate, speedView, topKRows,
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
  assert.equal(one.state === "few" && one.rolling, null, "one day has no 7-day average");
  assert.equal(two.rolling, 1.5, "two days do");
});

test("thirty days make a line from 0 up, by date, with its 7-day average beside it", () => {
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
  assert.equal(v.hasRolling, true);
  assert.equal(ROLLING_LABEL, "7-day average");
  for (const p of v.points) {
    assert.ok(Math.abs(p.y - (1 - p.seconds_per_photo / v.yMax)) < 1e-9);
    assert.ok(p.rolling != null && Math.abs(p.ry! - (1 - p.rolling / v.yMax)) < 1e-9);
  }
  assert.ok(Math.abs(v.points[29].rolling! - rollingAverage(days(30))[29].seconds) < 1e-12);
  assert.deepEqual(v.xTicks.map((t) => t.label)[0], "23 Sep");
  assert.equal(v.xTicks[v.xTicks.length - 1].label, "22 Oct");
  assert.ok(v.xTicks.length <= 5);
});

test("a day without photos leaves its gap in time, and the axis fits a fast week", () => {
  const v = speedView([{ day: "2026-10-01", seconds_per_photo: 0.4 },
                       { day: "2026-10-02", seconds_per_photo: 0.5 },
                       { day: "2026-10-11", seconds_per_photo: 0.6 }]);
  assert.ok(v.state === "chart");
  assert.ok(Math.abs(v.points[1].x - 0.1) < 1e-9, "1 day of 10, not 1 point of 2");
  assert.ok(v.yMax >= 0.66 - 1e-9 && v.yMax <= 1, "the axis is sized to the days, not to a fixed figure");
});

// The 7-day average (Steve, 2026-10-22): each day with the recorded days among the 6 before it,
// an unweighted mean of daily averages; a day without photos is skipped, never a 0.
test("the 7-day average needs 2 days: none for 0 or 1", () => {
  assert.deepEqual(rollingAverage([]), []);
  assert.deepEqual(rollingAverage([{ day: "2026-10-22", seconds_per_photo: 1.2 }]), []);
});

test("with 2 days the 7-day average is their plain mean from the second day", () => {
  assert.deepEqual(rollingAverage([{ day: "2026-10-21", seconds_per_photo: 1 },
                                   { day: "2026-10-22", seconds_per_photo: 2 }]),
                   [{ day: "2026-10-21", seconds: 1 }, { day: "2026-10-22", seconds: 1.5 }]);
});

test("over 10 days with a gap, each day averages the recorded days in its 7 calendar days", () => {
  // 1-4 October and 7-12 October: 5 and 6 October had no photos.
  const d = ["01", "02", "03", "04", "07", "08", "09", "10", "11", "12"]
    .map((dd, i) => ({ day: `2026-10-${dd}`, seconds_per_photo: i + 1 }));
  const r = rollingAverage(d);
  assert.equal(r.length, 10);
  const mean = (...xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
  assert.equal(r[0].seconds, 1);
  assert.equal(r[3].seconds, mean(1, 2, 3, 4));
  // 7 Oct: the window is 1-7 Oct; the 2 empty days are skipped, not counted as 0.
  assert.equal(r[4].day, "2026-10-07");
  assert.equal(r[4].seconds, mean(1, 2, 3, 4, 5));
  // 8 Oct: 2-8 Oct, so 1 Oct has left the window.
  assert.equal(r[5].seconds, mean(2, 3, 4, 5, 6));
  // 12 Oct: 6-12 Oct, six recorded days.
  assert.equal(r[9].seconds, mean(5, 6, 7, 8, 9, 10));
  const v = speedView(d);
  assert.ok(v.state === "chart");
  assert.deepEqual(v.points.map((p) => p.rolling), r.map((x) => x.seconds));
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

test("the small model credit names the version, its training data, and what it is built on", () => {
  const c = modelCredit("bioclip-2-336-ft-20261018-171420");
  assert.equal(c.name, "MycoMap Vision v1 (public + non-public training data)");
  assert.equal(c.title, "MycoMap Vision v1");
  assert.equal(c.detail, "(public + non-public training data)");
  assert.equal(c.madeBy, "MycoMap Vision is BioCLIP 2 (Imageomics) fine-tuned on DNA-verified records.");
  assert.equal(modelCredit("bioclip-2-336-ft-20261020-173106").title, "MycoMap Vision v1-open");
  assert.equal(modelCredit("dinov2-b14").madeBy, null, "no credit made up for another base");
});

test("today beside the speed figure is the US Eastern day, as the server counts it", () => {
  assert.equal(easternDay(new Date("2026-10-23T02:30:00Z")), "2026-10-22");   // 10:30 pm EDT
  assert.equal(easternDay(new Date("2026-10-23T04:00:00Z")), "2026-10-23");   // midnight EDT
  assert.equal(easternDay(new Date("2026-12-01T04:59:00Z")), "2026-11-30");   // 11:59 pm EST
});
