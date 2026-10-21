import { test } from "node:test";
import assert from "node:assert/strict";
import type { PhotoViewTile } from "./api";
import { batchFor, chips, markSaved, needsSave, saveLine, setNoSpecimen, startChoice, toggleView } from "./photoViews";

const VIEWS = [
  { code: "cap_top", label: "Cap top", suggested: true },
  { code: "gills", label: "Gills", suggested: true },
  { code: "stem_base", label: "Stem and base", suggested: true },
  { code: "in_situ", label: "In situ (growing)", suggested: true },
  { code: "in_hand", label: "In hand", suggested: false },
];

const tile = (id: number, suggested: string[], decision: PhotoViewTile["decision"] = null): PhotoViewTile => ({
  item_id: id, photo_key: `inat:${id}`, record_key: String(id * 10), label: "A x", link: "https://x/obs",
  status: decision ? "decided" : "open", suggested: suggested.map((v, i) => ({ view: v, p: 0.99 - i / 100 })),
  confirmed: decision?.views ?? null, decision, strength: 0.99, shown: "public", position: 0,
  license_code: "cc-by", owner_login: "o", record_photos: 3,
});

test("an open photo starts with the tagger's suggestions on, in the views' order", () => {
  assert.deepEqual(startChoice(tile(1, ["in_situ", "gills"]), VIEWS), { views: ["gills", "in_situ"], noSpecimen: false });
});

test("a decided photo starts as it was saved, including no specimen", () => {
  assert.deepEqual(startChoice(tile(2, ["gills"], { decision: "confirm_views", views: ["cap_top"] }), VIEWS),
                   { views: ["cap_top"], noSpecimen: false });
  assert.deepEqual(startChoice(tile(3, ["gills"], { decision: "no_specimen", views: null }), VIEWS),
                   { views: [], noSpecimen: true });
});

test("tapping a tag toggles it; an unknown view changes nothing; no specimen clears the tags", () => {
  const t = tile(1, ["gills"]);
  let c = toggleView({}, t, "in_hand", VIEWS);
  assert.deepEqual(c[1].views, ["gills", "in_hand"]);
  c = toggleView(c, t, "gills", VIEWS);
  assert.deepEqual(c[1].views, ["in_hand"]);
  assert.equal(toggleView(c, t, "nonsense", VIEWS), c);
  c = setNoSpecimen(c, t, true, VIEWS);
  assert.deepEqual(c[1], { views: [], noSpecimen: true });
  c = setNoSpecimen(c, t, false, VIEWS);
  assert.deepEqual(c[1], { views: ["gills"], noSpecimen: false });
});

test("nothing is written until a Save, and Save page writes exactly the tiles still to save", () => {
  const open1 = tile(1, ["gills"]), open2 = tile(2, ["cap_top"]);
  const done = tile(3, ["gills"], { decision: "confirm_views", views: ["gills"] });
  const tiles = [open1, open2, done];
  // Unchanged open tiles still need saving (the reviewer confirms them); an unchanged decided one does not.
  assert.equal(needsSave(open1, {}, {}, VIEWS), true);
  assert.equal(needsSave(done, {}, {}, VIEWS), false);
  let choices = toggleView({}, open2, "in_hand", VIEWS);
  choices = toggleView(choices, done, "stem_base", VIEWS);
  assert.deepEqual(batchFor(tiles, choices, {}, VIEWS), [
    { item_id: 1, views: ["gills"] }, { item_id: 2, views: ["cap_top", "in_hand"] },
    { item_id: 3, views: ["gills", "stem_base"] }]);
});

test("a tile's own Save writes that tile only", () => {
  const tiles = [tile(1, ["gills"]), tile(2, ["cap_top"])];
  assert.deepEqual(batchFor(tiles, {}, {}, VIEWS, [2]), [{ item_id: 2, views: ["cap_top"] }]);
});

test("a saved tile drops out of the next save until it changes again", () => {
  const tiles = [tile(1, ["gills"]), tile(2, ["cap_top"])];
  let choices = setNoSpecimen({}, tiles[1], true, VIEWS);
  const sent = batchFor(tiles, choices, {}, VIEWS);
  assert.deepEqual(sent, [{ item_id: 1, views: ["gills"] }, { item_id: 2, no_specimen: true }]);
  const saved = markSaved({}, sent);
  assert.deepEqual(batchFor(tiles, choices, saved, VIEWS), []);
  assert.equal(saveLine(batchFor(tiles, choices, saved, VIEWS).length), "All saved");
  choices = toggleView(choices, tiles[0], "in_situ", VIEWS);
  assert.deepEqual(batchFor(tiles, choices, saved, VIEWS), [{ item_id: 1, views: ["gills", "in_situ"] }]);
  assert.equal(saveLine(1), "1 to save");
});

test("a tile shows every suggestion (on or off) and every view switched on", () => {
  const t = tile(1, ["gills", "in_situ"]);
  const c = toggleView(toggleView({}, t, "in_situ", VIEWS), t, "in_hand", VIEWS)[1];
  assert.deepEqual(chips(t, c, VIEWS).map((x) => [x.code, x.on, x.suggested]),
                   [["gills", true, true], ["in_situ", false, true], ["in_hand", true, false]]);
});
