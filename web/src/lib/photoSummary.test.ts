import { test } from "node:test";
import assert from "node:assert/strict";
import type { FlaggedTile } from "./api";
import { batchBody, beginSend, decidedText, dueDecisions, gridColumns, gridKey, initialMarks, licenceText, markLoaded,
         markSeen, markTouched, mergeMarks, moveFocus, nextOffset, NOTE_FLIPPED, NOTE_SHOWN, overlay, photoOfText,
         readTileSize, readView, retry, sameState, sendDone, sendFailed, setDecision, setReason, SUMMARY_KEY_HELP,
         TILE_SIZE_STORAGE_KEY, tileStatus, toggle, VIEW_STORAGE_KEY, visitLine, visitTotals, writeTileSize, writeView,
         type GridKeyState, type Marks, type SaveStates } from "./photoSummary";

const micro = { tab: "micrograph", kind: "micrograph probe", p: 0.9, strength: 0.9 };
const gate = { tab: "gate", kind: "not-a-fungus gate", p: 0.0123, strength: 0.9877 };

const tile = (over: Partial<FlaggedTile> = {}): FlaggedTile => ({
  item_id: 1, photo_key: "inat:1000", record_key: "100", label: "Amanita muscaria",
  link: "https://www.inaturalist.org/observations/100", flags: [micro], strength: 0.9,
  suggested_reason: "micrograph", status: "open", decision: null, shown: "public",
  thumb: "https://img.example/1/small.jpg", full: "https://img.example/1/large.jpg", position: 1,
  license_class: "open", license_code: "cc-by", owner_login: "alice", record_photos: 4, ...over,
});

const page = (): FlaggedTile[] => [
  tile(),
  tile({ item_id: 2, photo_key: "inat:1001", flags: [gate], suggested_reason: "not_fungus" }),
  tile({ item_id: 3, photo_key: "mo:77", record_key: "mo:456", suggested_reason: null }),
  tile({ item_id: 4, status: "decided", decision: { decision: "exclude_photo", reason: "slip" } }),
  tile({ item_id: 5, status: "decided", decision: { decision: "keep_photo", reason: null } }),
];

const memory = (init: Record<string, string> = {}) => {
  const m = new Map(Object.entries(init));
  return { getItem: (k: string) => m.get(k) ?? null, setItem: (k: string, v: string) => void m.set(k, v), m };
};
const broken = () => { throw new Error("blocked"); };

// --- marks ---------------------------------------------------------------------------------------

test("every open tile starts as Exclude with its suggested reason; decided tiles show what is stored", () => {
  const m = initialMarks(page());
  assert.deepEqual(m[1], { decision: "exclude", reason: "micrograph" });
  assert.deepEqual(m[2], { decision: "exclude", reason: "not_fungus" });
  assert.deepEqual(m[3], { decision: "exclude", reason: null });
  assert.deepEqual(m[4], { decision: "exclude", reason: "slip", readOnly: true });
  assert.deepEqual(m[5], { decision: "keep", reason: null, readOnly: true });
});

test("the flipper goes Exclude, Include, Exclude and keeps the reason for coming back", () => {
  const m = initialMarks(page());
  const once = toggle(m, 1);
  assert.deepEqual(once[1], { decision: "keep", reason: "micrograph" });
  assert.deepEqual(toggle(once, 1)[1], { decision: "exclude", reason: "micrograph" });
  assert.equal(setDecision(m, 1, "exclude"), m, "already excluded: unchanged");
  assert.equal(toggle(m, 99), m, "no such tile");
});

test("a decided tile is read-only", () => {
  const m = initialMarks(page());
  assert.equal(toggle(m, 4), m);
  assert.equal(setDecision(m, 5, "exclude"), m);
  assert.equal(setReason(m, 4, "people"), m);
});

test("a reason is picked only for an excluded tile", () => {
  const m = initialMarks(page());
  assert.deepEqual(setReason(m, 3, "habitat")[3], { decision: "exclude", reason: "habitat" });
  const kept = toggle(m, 1);
  assert.equal(setReason(kept, 1, "habitat"), kept);
  assert.equal(setReason(m, 1, ""), m);
});

test("the reviewer's own marks win over a page's fresh start, never over a stored decision", () => {
  const fresh = initialMarks(page());
  const edits: Marks = { 1: { decision: "keep", reason: "micrograph" }, 4: { decision: "keep", reason: null } };
  const m = mergeMarks(fresh, edits);
  assert.equal(m[1].decision, "keep");
  assert.equal(m[4].decision, "exclude");
  assert.equal(m[2].decision, "exclude");
});

// --- auto-save -----------------------------------------------------------------------------------

test("the default Exclude is saved only for a tile shown long enough AND loaded", () => {
  const tiles = page();
  const marks = initialMarks(tiles);
  assert.deepEqual(dueDecisions(tiles, marks, {}), [], "nothing shown yet");
  let s: SaveStates = markSeen({}, 1);
  assert.deepEqual(dueDecisions(tiles, marks, s), [], "seen, not loaded");
  s = markLoaded(markLoaded({}, 2), 1);
  assert.deepEqual(dueDecisions(tiles, marks, s), [], "loaded, not seen");
  s = markSeen(markLoaded({}, 1), 1);
  assert.deepEqual(dueDecisions(tiles, marks, s),
                   [{ item_id: 1, decision: "exclude", reason: "micrograph", note: NOTE_SHOWN }]);
});

test("a hidden or failed tile never saves itself, but a flip saves it", () => {
  const tiles = page();
  let marks = initialMarks(tiles);
  let s: SaveStates = markSeen({}, 2);                 // seen, its image never loaded
  assert.deepEqual(dueDecisions(tiles, marks, s), []);
  marks = toggle(marks, 2);
  s = markTouched(s, 2);
  assert.deepEqual(dueDecisions(tiles, marks, s), [{ item_id: 2, decision: "keep", note: NOTE_FLIPPED }]);
});

test("decided tiles are never sent, and an exclusion waits for a reason", () => {
  const tiles = page();
  let marks = initialMarks(tiles);
  let s: SaveStates = {};
  for (const id of [3, 4, 5]) s = markTouched(markLoaded(markSeen(s, id), id), id);
  assert.deepEqual(dueDecisions(tiles, marks, s), []);
  assert.equal(tileStatus(marks[3], s[3]), "needs-reason");
  marks = setReason(marks, 3, "habitat");
  assert.deepEqual(dueDecisions(tiles, marks, s),
                   [{ item_id: 3, decision: "exclude", reason: "habitat", note: NOTE_FLIPPED }]);
});

test("a flip saves at once and becomes the saved state; nothing is sent twice", () => {
  const tiles = page();
  let marks = toggle(initialMarks(tiles), 1);
  let s = markTouched({}, 1);
  const due = dueDecisions(tiles, marks, s);
  assert.deepEqual(batchBody(due), { decisions: [{ item_id: 1, decision: "keep", note: NOTE_FLIPPED }] });
  s = beginSend(s, due);
  assert.equal(tileStatus(marks[1], s[1]), "saving");
  assert.deepEqual(dueDecisions(tiles, marks, s), [], "in flight: not sent again");
  s = sendDone(s, due);
  assert.equal(tileStatus(marks[1], s[1]), "saved");
  assert.deepEqual(dueDecisions(tiles, marks, s), [], "saved in this state: nothing to send");
  // Flipped away and back before anything was sent: back at the saved state, nothing goes.
  marks = toggle(toggle(marks, 1), 1);
  s = markTouched(markTouched(s, 1), 1);
  assert.equal(marks[1].decision, "keep");
  assert.deepEqual(dueDecisions(tiles, marks, s), []);
  assert.equal(batchBody([]), null);
});

test("a flip during a send is sent after it", () => {
  const tiles = page();
  let marks = initialMarks(tiles);
  let s = markLoaded(markSeen({}, 1), 1);
  const first = dueDecisions(tiles, marks, s);
  s = beginSend(s, first);
  marks = toggle(marks, 1);
  s = markTouched(s, 1);
  assert.deepEqual(dueDecisions(tiles, marks, s), [], "waits for the send in flight");
  s = sendDone(s, first);
  assert.deepEqual(dueDecisions(tiles, marks, s), [{ item_id: 1, decision: "keep", note: NOTE_FLIPPED }]);
});

test("a changed reason saves again; a keep's reason never matters", () => {
  const tiles = page();
  let marks = initialMarks(tiles);
  let s = markLoaded(markSeen({}, 1), 1);
  s = sendDone(s, dueDecisions(tiles, marks, s));
  marks = setReason(marks, 1, "habitat");
  s = markTouched(s, 1);
  assert.deepEqual(dueDecisions(tiles, marks, s),
                   [{ item_id: 1, decision: "exclude", reason: "habitat", note: NOTE_FLIPPED }]);
  assert.ok(sameState({ decision: "keep", reason: "a" }, { decision: "keep", reason: null }));
  assert.ok(!sameState({ decision: "exclude", reason: "a" }, { decision: "exclude", reason: "b" }));
  assert.ok(!sameState({ decision: "exclude", reason: "a" }, undefined));
});

test("a refused batch saves none of it; a refused tile waits for a retry or a flip", () => {
  const tiles = page();
  const marks = initialMarks(tiles);
  let s: SaveStates = {};
  for (const id of [1, 2]) s = markLoaded(markSeen(s, id), id);
  const due = dueDecisions(tiles, marks, s);
  assert.equal(due.length, 2);
  s = sendFailed(beginSend(s, due), due, "item 2 is on a record in the paper's sealed test");
  assert.equal(tileStatus(marks[1], s[1]), "failed");
  assert.equal(s[2].failed, "item 2 is on a record in the paper's sealed test");
  assert.equal(s[1].saved, undefined);
  assert.deepEqual(dueDecisions(tiles, marks, s), [], "not sent again by itself");
  s = retry(s, 1);
  assert.deepEqual(dueDecisions(tiles, marks, s).map((d) => d.item_id), [1]);
  s = markTouched(s, 2);
  assert.deepEqual(dueDecisions(tiles, marks, s).map((d) => d.item_id), [1, 2]);
});

test("the visit's totals count each tile's latest saved state", () => {
  const due = [{ item_id: 1, decision: "exclude" as const, reason: "micrograph" },
               { item_id: 2, decision: "keep" as const }, { item_id: 3, decision: "exclude" as const, reason: "x" }];
  let s = sendDone({}, due);
  assert.deepEqual(visitTotals(s), { excluded: 2, included: 1 });
  s = sendDone(s, [{ item_id: 1, decision: "keep" }]);
  assert.deepEqual(visitTotals(s), { excluded: 1, included: 2 });
  assert.equal(visitLine({ excluded: 1200, included: 3 }), "Saved this visit: 1,200 excluded, 3 included");
});

test("Next skips the open tiles saved from this page (they drop out of Open)", () => {
  assert.equal(nextOffset("open", 0, 150, 40), 110);
  assert.equal(nextOffset("open", 150, 150, 150), 150);
  assert.equal(nextOffset("decided", 150, 150, 40), 300);
});

// --- the viewer's words --------------------------------------------------------------------------

test("the viewer's overlay names the species, the observation, the photo, its flags, owner and licence", () => {
  const o = overlay(tile({ flags: [micro, gate] }));
  assert.equal(o.name, "Amanita muscaria");
  assert.deepEqual(o.link, { href: "https://www.inaturalist.org/observations/100", text: "iNat 100" });
  assert.deepEqual(o.lines, ["iNaturalist photo 1000 · photo 2 of 4", "micrograph probe p=0.90",
                             "not a fungus, p(fungus)=0.01", "by alice · CC-BY"]);
  const mo = overlay(tile({ photo_key: "mo:77", record_key: "mo:456", link: "https://mushroomobserver.org/456",
                            label: "", flags: [], owner_login: null, license_code: null, license_class: "arr",
                            position: 0, record_photos: null }));
  assert.equal(mo.name, "(no name)");
  assert.equal(mo.link.text, "MO 456");
  assert.deepEqual(mo.lines, ["Mushroom Observer photo 77 · photo 1", "all rights reserved"]);
  const done = overlay(tile({ status: "decided", decision: { decision: "exclude_photo", reason: "slip" } }),
                       [{ code: "slip", label: "Field slip or label" }]);
  assert.equal(done.lines.at(-1), "Decided: excluded (Field slip or label)");
});

test("photo n of N counts from 1; licences read as their code", () => {
  assert.equal(photoOfText(0, 3), "photo 1 of 3");
  assert.equal(photoOfText(null, 3), "photo ? of 3");
  assert.equal(licenceText("cc-by-nc"), "CC-BY-NC");
  assert.equal(licenceText(null, "open"), null);
  assert.equal(decidedText({ decision: { decision: "keep_photo", reason: null } }), "Decided: included");
  assert.equal(decidedText({ decision: null }), null);
});

// --- per-viewer settings -------------------------------------------------------------------------

test("the view is remembered per viewer, Summary by default and when storage is blocked", () => {
  assert.equal(readView(() => memory()), "summary");
  assert.equal(readView(() => null), "summary");
  assert.equal(readView(broken), "summary");
  assert.equal(readView(() => memory({ [VIEW_STORAGE_KEY]: "nonsense" })), "summary");
  const s = memory();
  writeView(() => s, "full");
  assert.equal(readView(() => s), "full");
  assert.doesNotThrow(() => writeView(broken, "full"));
});

test("the tile size is remembered per viewer, Medium by default and when storage is blocked", () => {
  assert.equal(readTileSize(() => memory()), "m");
  assert.equal(readTileSize(broken), "m");
  assert.equal(readTileSize(() => memory({ [TILE_SIZE_STORAGE_KEY]: "xl" })), "m");
  const s = memory();
  writeTileSize(() => s, "l");
  assert.equal(readTileSize(() => s), "l");
  assert.doesNotThrow(() => writeTileSize(broken, "s"));
});

test("tiles are about 120 / 170 / 240 px, never fewer than two across", () => {
  assert.equal(gridColumns("m"), "repeat(auto-fill, minmax(min(170px, calc(50% - 4px)), 1fr))");
  assert.match(gridColumns("s"), /min\(120px/);
  assert.match(gridColumns("l"), /min\(240px/);
});

// --- keys ----------------------------------------------------------------------------------------

const grid = (over: Partial<GridKeyState> = {}): GridKeyState => ({ typing: false, lightbox: false, columns: 6, ...over });

test("grid keys: arrows move by tile and row, Enter opens, K includes, X excludes, ? help", () => {
  assert.deepEqual(gridKey({ key: "ArrowRight" }, grid()), { type: "move", delta: 1 });
  assert.deepEqual(gridKey({ key: "ArrowLeft" }, grid()), { type: "move", delta: -1 });
  assert.deepEqual(gridKey({ key: "ArrowDown" }, grid()), { type: "move", delta: 6 });
  assert.deepEqual(gridKey({ key: "ArrowUp" }, grid({ columns: 2 })), { type: "move", delta: -2 });
  assert.deepEqual(gridKey({ key: "Enter" }, grid()), { type: "open" });
  assert.equal(gridKey({ key: "Enter" }, grid({ onControl: true })), null);
  assert.deepEqual(gridKey({ key: "k" }, grid()), { type: "choose", choice: "keep" });
  assert.deepEqual(gridKey({ key: "X" }, grid()), { type: "choose", choice: "exclude" });
  assert.deepEqual(gridKey({ key: "?" }, grid()), { type: "help" });
  assert.equal(gridKey({ key: "s" }, grid()), null, "no save key: every flip saves");
  assert.equal(gridKey({ key: "Escape" }, grid()), null);
  assert.deepEqual(gridKey({ key: "Escape" }, grid({ help: true })), { type: "help" });
});

test("keys belong to an input, and to the browser with a modifier", () => {
  assert.equal(gridKey({ key: "x" }, grid({ typing: true })), null);
  assert.equal(gridKey({ key: "ArrowRight", ctrlKey: true }, grid()), null);
});

test("viewer keys: ← / → step through the page, K / X flip, Esc closes", () => {
  const lb = grid({ lightbox: true });
  assert.deepEqual(gridKey({ key: "ArrowRight" }, lb), { type: "step", delta: 1 });
  assert.deepEqual(gridKey({ key: "ArrowLeft" }, lb), { type: "step", delta: -1 });
  assert.equal(gridKey({ key: "ArrowDown" }, lb), null);
  assert.deepEqual(gridKey({ key: "K" }, lb), { type: "choose", choice: "keep" });
  assert.deepEqual(gridKey({ key: "x" }, lb), { type: "choose", choice: "exclude" });
  assert.deepEqual(gridKey({ key: "Escape" }, lb), { type: "close" });
  assert.equal(moveFocus(5, 6, 6), 5);
  assert.equal(moveFocus(0, 6, -1), 0);
  assert.equal(moveFocus(2, 6, 1), 3);
  assert.ok(!SUMMARY_KEY_HELP.some(([k]) => k === "S"));
});
