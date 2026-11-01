import { test } from "node:test";
import assert from "node:assert/strict";
import type { FlaggedTile, ReviewPhoto } from "./api";
import { autoDue, initialMarks, markLoaded, markSeen, NOTE_SHOWN, SUMMARY_TABS, toggle, type SaveStates } from "./photoSummary";
import { BAND_FILTERS, bandCount, chooseDecision, initialChoices, saveBody, suggestsExclude } from "./photoReview";

// Tagger v2 (Steve, 2026-10-21): a photo it reads as a specimen photo starts as Include (a
// low-band photo in a group still starts as Exclude); the review filters by its bands; every
// kind has a tab.

const tile = (id: number, over: Partial<FlaggedTile> = {}) => ({
  item_id: id, status: "open" as const, decision: null, suggested_reason: "slip", ...over,
});

test("the Summary starts a photo the server suggests keeping as Include, any other open photo as Exclude", () => {
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

test("the band filter offers all, high, medium, low and very low with the tab's open counts", () => {
  assert.deepEqual(BAND_FILTERS.map((b) => b.code), ["all", "high", "medium", "low", "very_low"]);
  const data = { tabs: [{ code: "slip", bands: { high: 1200, medium: 3, low: 0 } },
                        { code: "uv", bands: { high: 0, medium: 40, low: 7 } }] };
  assert.equal(bandCount(data, "slip", "high"), " (1,200)");
  assert.equal(bandCount(data, "all", "medium"), " (43)");
  assert.equal(bandCount(data, "slip", "all"), "");
  assert.equal(bandCount(undefined, "slip", "high"), "");
});

test("Very low is the bottom of the Confidence dropdown and counts its tab's open photos", () => {
  assert.deepEqual(BAND_FILTERS[BAND_FILTERS.length - 1], { code: "very_low", label: "Very low" });
  const data: { tabs: { code: string; bands?: Record<string, number> }[] } = { tabs: [{ code: "uv", bands: { high: 417, medium: 439, low: 1266, very_low: 3693 } },
                        { code: "slip", bands: { high: 2, medium: 0, low: 1 } }] };   // an older server: no very_low
  assert.equal(bandCount(data, "uv", "very_low"), " (3,693)");
  assert.equal(bandCount(data, "all", "very_low"), " (3,693)");
  assert.equal(bandCount(data, "slip", "very_low"), "");
});

test("every v2 kind has a Summary tab", () => {
  const codes = SUMMARY_TABS.map((t) => t.code);
  for (const c of ["micrograph", "slip", "habitat", "people", "uv", "chemical", "spore_print", "dried", "tree",
                   "drawing", "specimen"]) assert.ok(codes.includes(c as never), c);
});

// Steve, 2026-11-01: "seed UV, Very low starts as Include". Every other band starts as Exclude.

const uvTile = (id: number, band: "high" | "medium" | "low" | "very_low",
                suggested_decision: "keep" | "exclude" = band === "very_low" ? "keep" : "exclude") =>
  ({ item_id: id, status: "open" as const, decision: null, suggested_reason: "uv", suggested_decision, band });

test("in the Summary a Very low tile starts as Include with its UV reason; High, Medium and Low as Exclude", () => {
  const m = initialMarks([uvTile(1, "high"), uvTile(2, "medium"), uvTile(3, "low"), uvTile(4, "very_low"),
                          uvTile(5, "very_low", "exclude")]);   // the band alone is enough
  assert.deepEqual(m[1], { decision: "exclude", reason: "uv" });
  assert.deepEqual(m[2], { decision: "exclude", reason: "uv" });
  assert.deepEqual(m[3], { decision: "exclude", reason: "uv" });
  assert.deepEqual(m[4], { decision: "keep", reason: "uv" });
  assert.deepEqual(m[5], { decision: "keep", reason: "uv" });
  assert.deepEqual(toggle(m, 4)[4], { decision: "exclude", reason: "uv" });   // flipping brings UV back
});

test("Auto-save on an untouched Very low tile saves Include; an untouched Low tile still saves Exclude", () => {
  const tiles = [uvTile(3, "low"), uvTile(4, "very_low")];
  const marks = initialMarks(tiles);
  const shown: SaveStates = markSeen(markLoaded(markSeen(markLoaded({}, 3), 3), 4), 4);
  assert.deepEqual(autoDue(true, tiles, marks, shown), [
    { item_id: 3, decision: "exclude", reason: "uv", note: NOTE_SHOWN },
    { item_id: 4, decision: "keep", note: NOTE_SHOWN },
  ]);
  assert.deepEqual(autoDue(false, tiles, marks, shown), []);                   // Auto-save off: nothing by itself
});

test("in the Full view a Very low photo starts as a suggested Include that saves as kept; others as suggested Exclude", () => {
  const ph = (key: string, band: "high" | "low" | "very_low") => ({
    photo_key: key, position: Number(key.slice(-1)), shown: "public", flagged: true, in_tab: true, item_id: 1,
    flags: [], suggested_reason: "uv", suggested_decision: band === "very_low" ? "keep" : "exclude", band,
    status: "open", decision: null }) as ReviewPhoto;
  const rec = { photos: [ph("inat:1", "high"), ph("inat:2", "low"), ph("inat:3", "very_low")] };
  const c = initialChoices(rec, "uv", "suggest");
  assert.deepEqual(c["inat:1"], { decision: "exclude", reason: "uv", suggested: true });
  assert.deepEqual(c["inat:2"], { decision: "exclude", reason: "uv", suggested: true });
  assert.deepEqual(c["inat:3"], { decision: "keep", reason: "uv", suggested: true });
  assert.equal(suggestsExclude(rec.photos[2], "uv"), false);
  const saved = saveBody(rec, c);
  assert.ok("body" in saved);
  assert.deepEqual(saved.body.photos, [
    { photo_key: "inat:1", decision: "exclude", reason: "uv" },
    { photo_key: "inat:2", decision: "exclude", reason: "uv" },
    { photo_key: "inat:3", decision: "keep" },
  ]);
  // Exclude on the suggested Include leaves it out as UV
  assert.deepEqual(chooseDecision(c, rec.photos[2], "exclude")["inat:3"], { decision: "exclude", reason: "uv" });
  // nothing preselected when suggestions are off, as before
  assert.deepEqual(initialChoices(rec, "uv", "none"), {});
});
