import { test } from "node:test";
import assert from "node:assert/strict";
import type { ReviewPhoto, ReviewRecord } from "./api";
import { bulkExcludeFlagged, bulkKeepAll, chosenCount, defaultReason, flagLabel, initialChoices, keyAction,
         nextPageOffset, nextUnsaved, offsetWhenPageDone, openOnLabel, outlineOf, photoLabels, placeholderText,
         savedSummary, saveBody, sortedPhotos, step, tileImage, tileKind, toggleChoice, type KeyState } from "./photoReview";

const photo = (over: Partial<ReviewPhoto> = {}): ReviewPhoto => ({
  photo_key: "inat:1", position: 1, license_class: "cc", shown: "public",
  thumb: "https://img.example/1/medium.jpg", full: "https://img.example/1/large.jpg",
  flagged: false, in_tab: false, item_id: null, flags: [], status: null, decision: null, ...over,
});

const micro = { tab: "micrograph", kind: "micrograph probe", p: 0.6234, strength: 0.6234 };
const habitat = { tab: "habitat", kind: "view habitat", p: 0.3, strength: 0.3 };
const gate = { tab: "gate", kind: "not-a-fungus gate", p: 0.0123, strength: 0.9877 };

/** Record 10: photo 1 flagged micrograph (open), 2 unflagged, 3 flagged habitat only,
 *  4 flagged micrograph and already decided (left out), as the micrograph tab sends it. */
const record = (over: Partial<ReviewRecord> = {}): ReviewRecord => ({
  record_key: "10", label: "Mycena", link: "https://www.inaturalist.org/observations/10", tab: "micrograph",
  strongest: 0.62, flagged_in_tab: 2,
  photos: [
    photo({ photo_key: "inat:3", position: 3, flagged: true, in_tab: false, item_id: 3, flags: [habitat], status: "open" }),
    photo({ photo_key: "inat:1", position: 1, flagged: true, in_tab: true, item_id: 1, flags: [micro], status: "open" }),
    photo({ photo_key: "inat:4", position: 4, flagged: true, in_tab: true, item_id: 4, flags: [micro], status: "decided",
            decision: { decision: "exclude_photo", reason: "micrograph", by: "S" } }),
    photo({ photo_key: "inat:2", position: 2 }),
  ],
  ...over,
});

test("an exclusion's reason starts as the tab's kind, 'other' for the gate and All", () => {
  assert.deepEqual(["micrograph", "slip", "habitat", "people", "gate", "other", "all"].map(defaultReason),
                   ["micrograph", "slip", "habitat", "people", "other", "other", "other"]);
});

test("a record's photos show in position order", () => {
  assert.deepEqual(sortedPhotos(record()).map((p) => p.photo_key), ["inat:1", "inat:2", "inat:3", "inat:4"]);
  const unplaced = { photos: [photo({ photo_key: "x", position: null }), photo({ photo_key: "y", position: 2 })] };
  assert.deepEqual(sortedPhotos(unplaced).map((p) => p.photo_key), ["y", "x"]);
});

test("nothing is preselected except a decided photo, which starts with its stored decision", () => {
  assert.deepEqual(initialChoices(record()), { "inat:4": "exclude" });
  const kept = record({ photos: [photo({ status: "decided", flagged: true,
                                         decision: { decision: "keep_photo", reason: null, by: "S" } })] });
  assert.deepEqual(initialChoices(kept), { "inat:1": "keep" });
  // An open photo never starts chosen, even with an earlier decision on file.
  const reopened = record({ photos: [photo({ status: "open", flagged: true,
                                             decision: { decision: "exclude_photo", reason: "slip", by: "S" } })] });
  assert.deepEqual(initialChoices(reopened), {});
});

test("one click picks Keep or Exclude, a second click on the same one clears it", () => {
  let c = toggleChoice({}, "inat:1", "keep");
  assert.deepEqual(c, { "inat:1": "keep" });
  c = toggleChoice(c, "inat:1", "exclude");
  assert.deepEqual(c, { "inat:1": "exclude" });
  assert.deepEqual(toggleChoice(c, "inat:1", "exclude"), {});
});

test("Keep all keeps every photo, flagged or not", () => {
  assert.deepEqual(bulkKeepAll(record()), { "inat:1": "keep", "inat:2": "keep", "inat:3": "keep", "inat:4": "keep" });
});

test("Exclude all flagged leaves out the tab's flagged photos, every flagged one in All", () => {
  assert.deepEqual(bulkExcludeFlagged(record(), "micrograph", { "inat:2": "keep" }),
                   { "inat:1": "exclude", "inat:4": "exclude", "inat:2": "keep" });
  assert.deepEqual(bulkExcludeFlagged(record(), "all"),
                   { "inat:1": "exclude", "inat:3": "exclude", "inat:4": "exclude" });
});

test("saving sends only the chosen photos, and each exclusion carries the reason", () => {
  const got = saveBody(record(), { "inat:3": "keep", "inat:1": "exclude" }, "micrograph", "  scope shots ");
  assert.deepEqual(got, { body: { photos: [
    { photo_key: "inat:1", decision: "exclude", reason: "micrograph" },
    { photo_key: "inat:3", decision: "keep" }], note: "scope shots" } });
  assert.deepEqual(saveBody(record(), { "inat:2": "keep" }, ""), { body: { photos: [
    { photo_key: "inat:2", decision: "keep" }] } });
});

test("saving refuses a record with nothing chosen, and an exclusion without a known reason", () => {
  assert.deepEqual(saveBody(record(), {}, "micrograph"), { error: "Keep or exclude at least one photo first." });
  assert.ok("error" in saveBody(record(), { "inat:1": "exclude" }, " "));
  assert.ok("error" in saveBody(record(), { "inat:1": "exclude" }, "nope", undefined, ["micrograph", "other"]));
  assert.ok("body" in saveBody(record(), { "inat:1": "exclude" }, "other", undefined, ["micrograph", "other"]));
  // A choice for a photo that is not the record's is never sent.
  assert.ok("error" in saveBody(record(), { "inat:99": "keep" }, "other"));
});

test("Save counts the chosen photos, and a saved card says what was kept and left out", () => {
  assert.equal(chosenCount(record(), {}), 0);
  assert.equal(chosenCount(record(), initialChoices(record())), 1);
  assert.equal(savedSummary({ photos: [{ photo_key: "a", decision: "keep" },
                                       { photo_key: "b", decision: "exclude", reason: "slip" },
                                       { photo_key: "c", decision: "exclude", reason: "slip" }] }),
               "Saved: 1 kept, 2 left out");
});

test("flags read as the probe or view and p to two decimals; the gate in plain words", () => {
  assert.equal(flagLabel(micro), "micrograph probe p=0.62");
  assert.equal(flagLabel(habitat), "view habitat p=0.30");
  assert.equal(flagLabel(gate), "not a fungus, p(fungus)=0.01");
  assert.equal(flagLabel({ tab: "other", kind: "reviewer: left out in photo review", p: null, strength: 0 }),
               "reviewer: left out in photo review");
});

test("photos flagged in the tab get the strong outline, flagged only elsewhere the light one", () => {
  const [p1, p2, p3] = sortedPhotos(record());
  assert.deepEqual([outlineOf(p1), outlineOf(p2), outlineOf(p3)], ["strong", "none", "light"]);
  const both = photo({ flagged: true, in_tab: true, flags: [micro, gate] });
  assert.deepEqual(photoLabels(both, "micrograph"),
                   { strong: ["micrograph probe p=0.62"], light: ["not a fungus, p(fungus)=0.01"] });
  assert.deepEqual(photoLabels(both, "all").strong.length, 2);
  assert.deepEqual(photoLabels(p3, "micrograph"), { strong: [], light: ["view habitat p=0.30"] });
});

test("a hidden photo is a placeholder and never gets an image URL, whatever it carries", () => {
  const hidden = photo({ shown: "hidden", hidden_reason: "all rights reserved",
                         thumb: "https://img.example/x.jpg", full: "/api/feedback/photo/inat:1?size=full" });
  assert.equal(tileKind(hidden), "placeholder");
  assert.equal(tileImage(hidden, "thumb"), null);
  assert.equal(tileImage(hidden, "full"), null);
});

test("an admin's private photo loads only from Vision's own same-origin route", () => {
  const priv = photo({ shown: "private", thumb: "/api/feedback/photo/inat:1?size=thumb",
                       full: "/api/feedback/photo/inat:1?size=full" });
  assert.equal(tileKind(priv), "image");
  assert.equal(tileImage(priv, "full"), "/api/feedback/photo/inat:1?size=full");
  assert.equal(tileImage(photo({ shown: "private", thumb: "https://elsewhere.example/a.jpg" })), null);
  assert.equal(tileImage(photo({ shown: "public", thumb: "http://img.example/a.jpg" })), null);
  assert.equal(tileKind(photo({ shown: "public", thumb: undefined })), "placeholder");
  assert.equal(tileImage(photo(), "full"), "https://img.example/1/large.jpg");
});

test("the record link is named for its site", () => {
  assert.equal(openOnLabel("mo:45"), "Open on MO");
  assert.equal(openOnLabel("123"), "Open on iNat");
});

const list: KeyState = { typing: false, lightbox: false };

test("on the list: j/k move between records, arrows between photos, Shift+K keeps, x excludes", () => {
  assert.deepEqual(keyAction({ key: "j" }, list), { type: "record", step: 1 });
  assert.deepEqual(keyAction({ key: "k" }, list), { type: "record", step: -1 });
  assert.deepEqual(keyAction({ key: "ArrowRight" }, list), { type: "photo", step: 1 });
  assert.deepEqual(keyAction({ key: "ArrowLeft" }, list), { type: "photo", step: -1 });
  assert.deepEqual(keyAction({ key: "K" }, list), { type: "choose", choice: "keep" });
  assert.deepEqual(keyAction({ key: "x" }, list), { type: "choose", choice: "exclude" });
  assert.deepEqual(keyAction({ key: "X" }, list), { type: "choose", choice: "exclude" });
  assert.deepEqual(keyAction({ key: "a" }, list), { type: "keepAll" });
  assert.deepEqual(keyAction({ key: "F" }, list), { type: "excludeFlagged" });
  assert.deepEqual(keyAction({ key: "s" }, list), { type: "save" });
  assert.deepEqual(keyAction({ key: "Enter" }, list), { type: "save" });
  assert.deepEqual(keyAction({ key: "?" }, list), { type: "help" });
  assert.equal(keyAction({ key: "q" }, list), null);
});

test("keys do nothing while typing, with a modifier, or Enter on a focused button", () => {
  assert.equal(keyAction({ key: "j" }, { ...list, typing: true }), null);
  assert.equal(keyAction({ key: "x" }, { typing: true, lightbox: true }), null);
  assert.equal(keyAction({ key: "s", ctrlKey: true }, list), null);
  assert.equal(keyAction({ key: "a", metaKey: true }, list), null);
  assert.equal(keyAction({ key: "Enter" }, { ...list, onControl: true }), null);
  assert.equal(keyAction({ key: "Escape" }, list), null);
  assert.deepEqual(keyAction({ key: "Escape" }, { ...list, help: true }), { type: "help" });
});

test("in the photo viewer: arrows move, K or k keeps, X excludes, Esc closes, nothing else", () => {
  const lb: KeyState = { typing: false, lightbox: true };
  assert.deepEqual(keyAction({ key: "Escape" }, lb), { type: "close" });
  assert.deepEqual(keyAction({ key: "ArrowLeft" }, lb), { type: "photo", step: -1 });
  assert.deepEqual(keyAction({ key: "ArrowRight" }, lb), { type: "photo", step: 1 });
  assert.deepEqual(keyAction({ key: "k" }, lb), { type: "choose", choice: "keep" });
  assert.deepEqual(keyAction({ key: "K" }, lb), { type: "choose", choice: "keep" });
  assert.deepEqual(keyAction({ key: "x" }, lb), { type: "choose", choice: "exclude" });
  assert.equal(keyAction({ key: "j" }, lb), null);
  assert.equal(keyAction({ key: "s" }, lb), null);
});

test("steps stop at the ends", () => {
  assert.equal(step(0, 4, -1), 0);
  assert.equal(step(3, 4, 1), 3);
  assert.equal(step(1, 4, 1), 2);
  assert.equal(step(0, 0, 1), 0);
});

test("after a save focus goes to the next unsaved card, then back to an earlier one, then none", () => {
  const keys = ["a", "b", "c", "d"];
  assert.equal(nextUnsaved(keys, new Set(["b"]), 1), 2);
  assert.equal(nextUnsaved(keys, new Set(["b", "c", "d"]), 3), 0);
  assert.equal(nextUnsaved(keys, new Set(keys), 2), -1);
});

test("a finished page reloads in place in Open (saved records drop out) and moves on in Decided", () => {
  assert.equal(offsetWhenPageDone("open", 10, 10, 50), 10);
  assert.equal(offsetWhenPageDone("decided", 10, 10, 50), 20);
  assert.equal(offsetWhenPageDone("decided", 40, 10, 50), 40);
});

test("Next page in Open steps back by the records saved here, so none are skipped", () => {
  assert.equal(nextPageOffset("open", 0, 10, 6), 4);
  assert.equal(nextPageOffset("open", 20, 10, 0), 30);
  assert.equal(nextPageOffset("decided", 20, 10, 6), 30);
});

test("a photo whose image fails to load becomes the placeholder, saying so; a hidden one keeps its reason", () => {
  const priv = photo({ shown: "private", thumb: "/api/feedback/photo/inat:1?size=thumb" });
  assert.equal(tileKind(priv), "image");
  assert.equal(tileKind(priv, true), "placeholder");
  assert.equal(placeholderText(priv, true), "photo not available here");
  assert.equal(placeholderText(photo({ shown: "hidden", hidden_reason: "the owner declined" }), true),
               "the owner declined");
  assert.equal(placeholderText(photo({ shown: "hidden" })), "not shown here");
});
