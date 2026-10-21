import { test } from "node:test";
import assert from "node:assert/strict";
import type { FlaggedTile, ReviewPhoto } from "./api";
import { initialMarks, SUMMARY_TABS } from "./photoSummary";
import { BAND_FILTERS, bandCount, suggestsExclude } from "./photoReview";

// Tagger v2 (Steve, 2026-10-21): a photo it reads as a specimen photo, or puts in its low band,
// starts as Include; the review filters by its bands; every kind has a tab.

const tile = (id: number, over: Partial<FlaggedTile> = {}) => ({
  item_id: id, status: "open" as const, decision: null, suggested_reason: "slip", ...over,
});

test("the Summary starts a v2 specimen or low-band photo as Include, any other open photo as Exclude", () => {
  const m = initialMarks([
    tile(1),
    tile(2, { suggested_decision: "keep", suggested_reason: null }),
    tile(3, { suggested_decision: "keep", suggested_reason: "uv" }),
    tile(4, { suggested_decision: "exclude" }),
  ]);
  assert.deepEqual(m[1], { decision: "exclude", reason: "slip" });
  assert.deepEqual(m[2], { decision: "keep", reason: null });
  assert.deepEqual(m[3], { decision: "keep", reason: "uv" });   // flipping to Exclude brings UV back
  assert.deepEqual(m[4], { decision: "exclude", reason: "slip" });
});

test("the record view never suggests Exclude for a photo the server suggests keeping", () => {
  const p = (over: Partial<ReviewPhoto>) => ({ flagged: true, in_tab: true, status: "open", ...over }) as ReviewPhoto;
  assert.equal(suggestsExclude(p({}), "slip"), true);
  assert.equal(suggestsExclude(p({ suggested_decision: "keep" }), "slip"), false);
  assert.equal(suggestsExclude(p({ suggested_decision: "exclude" }), "slip"), true);
});

test("the band filter offers all, high, medium, low with the tab's open counts", () => {
  assert.deepEqual(BAND_FILTERS.map((b) => b.code), ["all", "high", "medium", "low"]);
  const data = { tabs: [{ code: "slip", bands: { high: 1200, medium: 3, low: 0 } },
                        { code: "uv", bands: { high: 0, medium: 40, low: 7 } }] };
  assert.equal(bandCount(data, "slip", "high"), " (1,200)");
  assert.equal(bandCount(data, "all", "medium"), " (43)");
  assert.equal(bandCount(data, "slip", "all"), "");
  assert.equal(bandCount(undefined, "slip", "high"), "");
});

test("every v2 kind has a Summary tab", () => {
  const codes = SUMMARY_TABS.map((t) => t.code);
  for (const c of ["micrograph", "slip", "habitat", "people", "uv", "chemical", "spore_print", "dried", "tree",
                   "drawing", "specimen"]) assert.ok(codes.includes(c as never), c);
});
