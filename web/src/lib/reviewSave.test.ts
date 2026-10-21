import { test } from "node:test";
import assert from "node:assert/strict";
import type { ReviewPhoto, ReviewRecord } from "./api";
import { initialChoices, chooseDecision, setReason, type PhotoChoices } from "./photoReview";
import { AUTOSAVE_STORAGE_KEY, autosaveOnText, changedPhotos, confirmLeave, createUnsaved, guardUnload, leaveText,
         pageSaves, readAutosave, savePageLabel, unsavedLine, writeAutosave } from "./reviewSave";

const memory = (init: Record<string, string> = {}) => {
  const m = new Map(Object.entries(init));
  return { getItem: (k: string) => m.get(k) ?? null, setItem: (k: string, v: string) => void m.set(k, v), m };
};
const broken = () => { throw new Error("blocked"); };

// --- the setting ---------------------------------------------------------------------------------

test("Auto-save is off by default, and when storage is missing or blocked", () => {
  assert.equal(readAutosave(() => memory()), false);
  assert.equal(readAutosave(() => null), false);
  assert.equal(readAutosave(broken), false);
  assert.equal(readAutosave(() => memory({ [AUTOSAVE_STORAGE_KEY]: "junk" })), false);
  assert.equal(readAutosave(() => memory({ [AUTOSAVE_STORAGE_KEY]: "off" })), false);
});

test("Auto-save is remembered per viewer once turned on, and off again", () => {
  const s = memory();
  writeAutosave(() => s, true);
  assert.equal(s.m.get(AUTOSAVE_STORAGE_KEY), "on");
  assert.equal(readAutosave(() => s), true);
  writeAutosave(() => s, false);
  assert.equal(readAutosave(() => s), false);
  assert.doesNotThrow(() => writeAutosave(broken, true));
});

// --- the leave guard -----------------------------------------------------------------------------

test("leaving with unsaved changes asks first, and stays when the reviewer says no", () => {
  const asked: string[] = [];
  const no = (m: string) => { asked.push(m); return false; };
  const yes = (m: string) => { asked.push(m); return true; };
  assert.equal(confirmLeave(3, no), false);
  assert.equal(confirmLeave(1, yes), true);
  assert.deepEqual(asked, [leaveText(3), leaveText(1)]);
  assert.match(asked[0], /3 unsaved changes/);
  assert.match(asked[1], /1 unsaved change on/);
});

test("leaving with nothing unsaved never asks", () => {
  const ask = () => { throw new Error("asked"); };
  assert.equal(confirmLeave(0, ask), true);
  assert.equal(confirmLeave(-1, ask), true);
});

test("closing or reloading the tab gets the browser's question only while changes are unsaved", () => {
  const ev = () => {
    const e = { prevented: false, returnValue: undefined as unknown, preventDefault() { e.prevented = true; } };
    return e;
  };
  const kept = ev();
  assert.equal(guardUnload(kept, 2), true);
  assert.equal(kept.prevented, true);
  assert.equal(kept.returnValue, "");
  const free = ev();
  assert.equal(guardUnload(free, 0), false);
  assert.equal(free.prevented, false);
  assert.equal(free.returnValue, undefined);
});

test("the queue picker sees the photo review's unsaved changes until they are gone", () => {
  const u = createUnsaved();
  assert.equal(u.total(), 0);
  u.set("photo-review", 4);
  u.set("other", 1);
  assert.equal(u.total(), 5);
  u.set("photo-review", 0);
  assert.equal(u.total(), 1);
});

test("the page's words", () => {
  assert.equal(savePageLabel(1), "Save page (1 change)");
  assert.equal(savePageLabel(12), "Save page (12 changes)");
  assert.equal(unsavedLine(0), "No unsaved changes");
  assert.equal(unsavedLine(2), "2 unsaved changes");
  assert.match(autosaveOnText(2), /2 unsaved changes will be saved now/);
});

// --- the Full view's Save page ---------------------------------------------------------------------

const photo = (over: Partial<ReviewPhoto> = {}): ReviewPhoto => ({
  photo_key: "inat:1", position: 1, license_class: "cc", shown: "public",
  thumb: "https://img.example/1/medium.jpg", full: "https://img.example/1/large.jpg",
  flagged: false, in_tab: false, item_id: null, flags: [], suggested_reason: null, status: null, decision: null,
  ...over,
});

/** Record `key`: photo 1 flagged (open), photo 2 unflagged, photo 3 decided (left out, slip). */
const record = (key: string): ReviewRecord => ({
  record_key: key, label: "Mycena", link: `https://www.inaturalist.org/observations/${key}`, tab: "micrograph",
  strongest: 0.6, flagged_in_tab: 1,
  photos: [
    photo({ photo_key: `inat:${key}1`, position: 1, flagged: true, in_tab: true, item_id: 1, suggested_reason: "micrograph",
            status: "open" }),
    photo({ photo_key: `inat:${key}2`, position: 2 }),
    photo({ photo_key: `inat:${key}3`, position: 3, flagged: true, in_tab: true, item_id: 3, status: "decided",
            decision: { decision: "exclude_photo", reason: "slip", by: "S" } }),
  ],
});

test("a record's stored decisions are no change; a choice on an open photo, or a changed one, is", () => {
  const r = record("10");
  const start = initialChoices(r, "micrograph");
  assert.deepEqual(changedPhotos(r, start), [], "the decided photo as stored");
  const kept = chooseDecision(start, r.photos[0], "keep");
  assert.deepEqual(changedPhotos(r, kept).map((p) => p.photo_key), ["inat:101"]);
  const reason = setReason(start, "inat:103", "habitat");
  assert.deepEqual(changedPhotos(r, reason).map((p) => p.photo_key), ["inat:103"]);
  const suggested = initialChoices(r, "micrograph", "suggest");
  assert.deepEqual(changedPhotos(r, suggested).map((p) => p.photo_key), ["inat:101"], "a suggestion is written by Save");
});

test("Save page writes exactly the records with changes, each with its own body", () => {
  const recs = [record("10"), record("20"), record("30"), record("40")];
  const choices: Record<string, PhotoChoices> = Object.fromEntries(recs.map((r) => [r.record_key, initialChoices(r, "micrograph")]));
  choices["10"] = chooseDecision(choices["10"], recs[0].photos[0], "keep");
  choices["30"] = chooseDecision(choices["30"], recs[2].photos[0], "exclude");
  choices["40"] = chooseDecision(choices["40"], recs[3].photos[1], "exclude");   // unflagged: no reason yet
  const plan = pageSaves(recs, choices, { notes: { "30": " blurry " } });
  assert.deepEqual(plan.saves.map((s) => [s.key, s.index]), [["10", 0], ["30", 2]], "record 20 has no change");
  assert.deepEqual(plan.saves[0].body.photos, [
    { photo_key: "inat:101", decision: "keep" },
    { photo_key: "inat:103", decision: "exclude", reason: "slip" },
  ]);
  assert.deepEqual(plan.saves[1].body, {
    photos: [{ photo_key: "inat:301", decision: "exclude", reason: "micrograph" },
             { photo_key: "inat:303", decision: "exclude", reason: "slip" }],
    note: "blurry",
  });
  assert.deepEqual(plan.blocked, [{ key: "40", error: "Pick why photo 2 is left out." }]);
  assert.equal(plan.changes, 3);
});

test("Save page skips records already saved or being saved", () => {
  const recs = [record("10"), record("20")];
  const choices = Object.fromEntries(recs.map((r) => [r.record_key, chooseDecision(initialChoices(r, "micrograph"), r.photos[0], "keep")]));
  assert.deepEqual(pageSaves(recs, choices).saves.map((s) => s.key), ["10", "20"]);
  const plan = pageSaves(recs, choices, { saved: { "10": "Saved: …" }, busy: { "20": true } });
  assert.deepEqual(plan, { saves: [], blocked: [], changes: 0 });
});
