import assert from "node:assert/strict";
import { test } from "node:test";
import { fullRows, GIVE_UP_MS, inatForPage, inatState, newInatRef, pollInterval, POLL_MS, settledView,
         topRows, type InatView } from "./inatSuggestions";

const SHOWN: InatView = {
  lines: [{ taxon_id: 1, name: "Russula emetica", rank: "species", score: 82.4, label: "iNaturalist suggests · 82%" }],
  not_fungus: null, full: null,
};

test("nothing from the server (no iNat account, signed out, off) shows nothing", () => {
  assert.equal(inatState(undefined), "none");
  assert.equal(inatState(null), "none");
  assert.equal(inatState({ lines: [], not_fungus: null, full: null }), "none");
});

test("while iNat is still out after Vision's answer the page says it is checking, then shows the lines", () => {
  assert.equal(inatState({ pending: true, ref: "site-x" }), "checking");
  assert.equal(inatState(SHOWN), "shown");
});

test("the full response alone is still shown, in its dropdown", () => {
  assert.equal(inatState({ lines: [], not_fungus: null, full: [{ results: [{ name: "Carya ovata", score: 40 }] }] }), "shown");
});

test("the page asks again only while pending and within the wait, then drops it quietly", () => {
  const pending: InatView = { pending: true, ref: "site-x" };
  assert.equal(pollInterval(pending, 0, 1000), POLL_MS);
  assert.equal(pollInterval(pending, 0, GIVE_UP_MS), false);
  assert.equal(pollInterval(SHOWN, 0, 1000), false);
  assert.equal(settledView(pending, 0, GIVE_UP_MS), null);
  assert.deepEqual(settledView(SHOWN, 0, GIVE_UP_MS * 2), SHOWN);
});

test("the dropdown lists every photo's suggestions best first with rounded scores", () => {
  const rows = fullRows([
    { results: [{ name: "B", rank: "genus", score: 12.6 }, { name: "A", rank: "species", score: 80.2, seen_nearby: true }] },
    { results: [{ name: "C", score: null }] },
  ]);
  assert.deepEqual(rows.map((r) => [r.photo, r.name, r.score, r.nearby]),
                   [[1, "A", "80%", "yes"], [1, "B", "13%", ""], [2, "C", "", ""]]);
});

test("while Vision is still checking, no iNat lines are asked for or shown", () => {
  assert.equal(inatForPage("checking", SHOWN, "ref-0123456789abcd"), null);
});

test("when Vision fails or times out, the find's iNat lines are asked for by the page's own ref", () => {
  assert.deepEqual(inatForPage("failed", undefined, "ref-0123456789abcd"),
                   { initial: { pending: true, ref: "site-ref-0123456789abcd" }, visionFailed: true });
  assert.equal(inatForPage("failed", undefined, null), null);
});

test("with Vision's answer, what came with it is shown, and nothing when it brought none", () => {
  assert.deepEqual(inatForPage("answered", SHOWN, "x"), { initial: SHOWN, visionFailed: false });
  assert.equal(inatForPage("answered", undefined, "x"), null);
});

test("a page's ref is 16 url-safe characters, new each time", () => {
  const a = newInatRef();
  assert.match(a, /^[A-Za-z0-9_-]{16}$/);
  assert.notEqual(a, newInatRef());
});

test("the dropdown shows the best 10 rows above 1% at first, over every photo; Show all has the rest", () => {
  const many = { results: Array.from({ length: 30 }, (_, i) => ({ name: `T${i}`, rank: "order", score: 30 - i * 1.2 })) };
  const second = { results: [{ name: "Best", rank: "species", score: 95 }, { name: "Tiny", rank: "order", score: 0.4 }] };
  const all = fullRows([many, second]);
  const top = topRows(all);
  assert.equal(top.length, 10);
  assert.equal(top[0].name, "Best");
  assert.ok(top.every((r) => (r.value ?? 0) > 1));
  assert.deepEqual(top.slice(1, 4).map((r) => r.name), ["T0", "T1", "T2"]);
  assert.equal(all.length, 32);
  const sparse = fullRows([{ results: [{ name: "A", score: 40 }, { name: "B", score: 1 }, { name: "C", score: null }] }]);
  assert.deepEqual(topRows(sparse).map((r) => r.name), ["A"]);
});
