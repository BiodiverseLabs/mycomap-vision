import { test } from "node:test";
import assert from "node:assert/strict";
import type { ReviewPhoto, ReviewRecord } from "./api";
import { bulkExcludeFlagged, bulkKeepAll, chooseDecision, chosenCount, cycleReason, decisionsOf, flagLabel,
         initialChoices, keyAction, missingReason, nextPageOffset, nextUnsaved, offsetWhenPageDone, openOnLabel,
         outlineOf, photoLabels, placeholderText, readPreselectMode, savedLine, saveBody, setReason, sortedPhotos,
         step, suggestedCount, tileImage, tileKind, toggleChoice, withMode, writePreselectMode, type KeyState,
         QUICK_REASONS, quickChoice, type PhotoChoices } from "./photoReview";

const photo = (over: Partial<ReviewPhoto> = {}): ReviewPhoto => ({
  photo_key: "inat:1", position: 1, license_class: "cc", shown: "public",
  thumb: "https://img.example/1/medium.jpg", full: "https://img.example/1/large.jpg",
  flagged: false, in_tab: false, item_id: null, flags: [], suggested_reason: null, status: null, decision: null,
  ...over,
});

const micro = { tab: "micrograph", kind: "micrograph probe", p: 0.6234, strength: 0.6234 };
const habitat = { tab: "habitat", kind: "view habitat", p: 0.3, strength: 0.3 };
const slip = { tab: "slip", kind: "view slip", p: 0.4, strength: 0.4 };
const gate = { tab: "gate", kind: "not-a-fungus gate", p: 0.0123, strength: 0.9877 };

/** Record 10: photo 1 flagged micrograph (open), 2 unflagged, 3 flagged habitat only,
 *  4 flagged micrograph and already decided (left out), as the micrograph tab sends it. */
const record = (over: Partial<ReviewRecord> = {}): ReviewRecord => ({
  record_key: "10", label: "Mycena", link: "https://www.inaturalist.org/observations/10", tab: "micrograph",
  strongest: 0.62, flagged_in_tab: 2,
  photos: [
    photo({ photo_key: "inat:3", position: 3, flagged: true, in_tab: false, item_id: 3, flags: [habitat],
            suggested_reason: "habitat", status: "open" }),
    photo({ photo_key: "inat:1", position: 1, flagged: true, in_tab: true, item_id: 1, flags: [micro],
            suggested_reason: "micrograph", status: "open" }),
    photo({ photo_key: "inat:4", position: 4, flagged: true, in_tab: true, item_id: 4, flags: [micro],
            suggested_reason: "micrograph", status: "decided",
            decision: { decision: "exclude_photo", reason: "micrograph", by: "S" } }),
    photo({ photo_key: "inat:2", position: 2 }),
  ],
  ...over,
});

/** Steve's record (2026-10-18), as the All tab sends it: photo 5 a habitat shot, 6 and 7
 *  field slips, 1 the specimen (unflagged). */
const mixed = (): ReviewRecord => record({
  record_key: "20", tab: "all", flagged_in_tab: 3,
  photos: [
    photo({ photo_key: "p1", position: 1 }),
    photo({ photo_key: "p5", position: 5, flagged: true, in_tab: true, item_id: 5, flags: [habitat],
            suggested_reason: "habitat", status: "open" }),
    photo({ photo_key: "p6", position: 6, flagged: true, in_tab: true, item_id: 6, flags: [slip],
            suggested_reason: "slip", status: "open" }),
    photo({ photo_key: "p7", position: 7, flagged: true, in_tab: true, item_id: 7, flags: [slip, gate],
            suggested_reason: "not_fungus", status: "open" }),
  ],
});

const REASONS = ["micrograph", "slip", "habitat", "people", "not_fungus", "other"];

test("a record's photos show in position order", () => {
  assert.deepEqual(sortedPhotos(record()).map((p) => p.photo_key), ["inat:1", "inat:2", "inat:3", "inat:4"]);
  const unplaced = { photos: [photo({ photo_key: "x", position: null }), photo({ photo_key: "y", position: 2 })] };
  assert.deepEqual(sortedPhotos(unplaced).map((p) => p.photo_key), ["y", "x"]);
});

test("by default nothing is preselected except a decided photo, which starts with its stored decision and reason", () => {
  assert.deepEqual(initialChoices(record(), "micrograph"),
                   { "inat:4": { decision: "exclude", reason: "micrograph" } });
  assert.deepEqual(initialChoices(mixed(), "all", "none"), {});
  const kept = record({ photos: [photo({ status: "decided", flagged: true, in_tab: true, suggested_reason: "slip",
                                         decision: { decision: "keep_photo", reason: null, by: "S" } })] });
  assert.deepEqual(initialChoices(kept, "all"), { "inat:1": { decision: "keep", reason: null } });
  // An open photo never starts chosen, even with an earlier decision on file.
  const reopened = record({ photos: [photo({ status: "open", flagged: true, in_tab: true,
                                             decision: { decision: "exclude_photo", reason: "slip", by: "S" } })] });
  assert.deepEqual(initialChoices(reopened, "all"), {});
});

test("in suggest mode each open photo flagged in the tab starts as a suggested Exclude with its own reason", () => {
  assert.deepEqual(initialChoices(mixed(), "all", "suggest"), {
    p5: { decision: "exclude", reason: "habitat", suggested: true },
    p6: { decision: "exclude", reason: "slip", suggested: true },
    p7: { decision: "exclude", reason: "not_fungus", suggested: true },
  });
  // Flagged only for another tab (3), unflagged (2): nothing. Decided (4): its stored decision.
  assert.deepEqual(initialChoices(record(), "micrograph", "suggest"), {
    "inat:1": { decision: "exclude", reason: "micrograph", suggested: true },
    "inat:4": { decision: "exclude", reason: "micrograph" },
  });
  const keptBefore = record({ photos: [photo({ status: "decided", flagged: true, in_tab: true, suggested_reason: "slip",
                                               decision: { decision: "keep_photo", reason: null, by: "S" } })] });
  assert.deepEqual(initialChoices(keptBefore, "slip", "suggest"), { "inat:1": { decision: "keep", reason: null } });
  assert.equal(suggestedCount(mixed(), initialChoices(mixed(), "all", "suggest")), 3);
});

test("suggesting is the default; a viewer who turns it off is remembered", () => {
  const box = new Map<string, string>();
  const store = { getItem: (k: string) => box.get(k) ?? null, setItem: (k: string, v: string) => { box.set(k, v); } };
  assert.equal(readPreselectMode(() => store), "suggest");          // nothing stored: the default
  writePreselectMode(() => store, "none");
  assert.equal(readPreselectMode(() => store), "none");
  writePreselectMode(() => store, "suggest");
  assert.equal(readPreselectMode(() => store), "suggest");
  const broken = () => { throw new Error("SecurityError"); };
  assert.equal(readPreselectMode(broken), "suggest");
  assert.doesNotThrow(() => writePreselectMode(broken, "none"));
  assert.equal(readPreselectMode(() => null), "suggest");
});

test("switching the mode adds or drops only the suggestions, never the reviewer's own choices", () => {
  const mine: PhotoChoices = { p1: { decision: "keep", reason: null }, p6: { decision: "keep", reason: "slip" } };
  const on = withMode(mixed(), "all", "suggest", mine);
  assert.deepEqual(on, { ...mine, p5: { decision: "exclude", reason: "habitat", suggested: true },
                         p7: { decision: "exclude", reason: "not_fungus", suggested: true } });
  assert.deepEqual(withMode(mixed(), "all", "none", on), mine);
});

test("changing the tab keeps the reasons the reviewer picked", () => {
  // Picked "slip" for photo 1 under the micrograph tab, then opened the All tab.
  const picked = { "inat:1": "slip" };
  const inAll = record({ tab: "all", photos: record().photos.map((p) => ({ ...p, in_tab: p.flagged })) });
  assert.equal(initialChoices(inAll, "all", "suggest", picked)["inat:1"].reason, "slip");
  const one = inAll.photos.find((p) => p.photo_key === "inat:1")!;
  assert.equal(chooseDecision({}, one, "exclude", picked)["inat:1"].reason, "slip");
  assert.equal(bulkExcludeFlagged(inAll, "all", {}, picked)["inat:1"].reason, "slip");
});

test("on a suggestion Keep is one click and Exclude confirms it; both make it a choice", () => {
  const start = initialChoices(mixed(), "all", "suggest");
  const p6 = mixed().photos[2];
  assert.deepEqual(chooseDecision(start, p6, "keep").p6, { decision: "keep", reason: "slip" });
  assert.deepEqual(chooseDecision(start, p6, "exclude").p6, { decision: "exclude", reason: "slip" });
  // Once confirmed, a second click on Exclude clears it, as for any choice.
  assert.equal(chooseDecision(chooseDecision(start, p6, "exclude"), p6, "exclude").p6, undefined);
});

test("one click picks Keep or Exclude, a second click on the same one clears it", () => {
  const p5 = mixed().photos[1];
  let c = chooseDecision({}, p5, "keep");
  assert.deepEqual(c, { p5: { decision: "keep", reason: null } });
  c = chooseDecision(c, p5, "exclude");
  assert.deepEqual(c, { p5: { decision: "exclude", reason: "habitat" } });
  assert.deepEqual(chooseDecision(c, p5, "exclude"), {});
  // The label review's plain toggle is unchanged.
  assert.deepEqual(toggleChoice(toggleChoice({}, "a", "keep"), "a", "keep"), {});
});

test("an unflagged photo starts with no reason when excluded, and gets the one the reviewer picks", () => {
  const p1 = mixed().photos[0];
  let c = chooseDecision({}, p1, "exclude");
  assert.deepEqual(c.p1, { decision: "exclude", reason: null });
  c = setReason(c, "p1", "people");
  assert.deepEqual(c.p1, { decision: "exclude", reason: "people" });
  // Kept and excluded again, it comes back with that reason.
  assert.equal(chooseDecision(chooseDecision(c, p1, "keep"), p1, "exclude").p1.reason, "people");
  // A kept or unchosen photo has no reason to pick.
  assert.deepEqual(setReason({}, "p1", "slip"), {});
});

test("R steps through the reasons of an excluded photo, and picking a reason confirms a suggestion", () => {
  const start = initialChoices(mixed(), "all", "suggest");
  assert.deepEqual(cycleReason(start, "p6", REASONS).p6, { decision: "exclude", reason: "habitat" });
  assert.equal(cycleReason(start, "p7", REASONS).p7.reason, "other");
  assert.equal(cycleReason({ p7: { decision: "exclude", reason: "other" } }, "p7", REASONS).p7.reason, "micrograph");
  assert.equal(cycleReason({ p1: { decision: "exclude", reason: null } }, "p1", REASONS).p1.reason, "micrograph");
  const kept: PhotoChoices = { p1: { decision: "keep", reason: null } };
  assert.equal(cycleReason(kept, "p1", REASONS), kept);
  assert.deepEqual(setReason(start, "p5", "people").p5, { decision: "exclude", reason: "people" });
});

test("Keep all keeps every photo, flagged or not", () => {
  const all = bulkKeepAll(record());
  assert.deepEqual(Object.keys(all).sort(), ["inat:1", "inat:2", "inat:3", "inat:4"]);
  assert.ok(Object.values(all).every((c) => c.decision === "keep" && !c.suggested));
});

test("Exclude all flagged leaves out the tab's flagged photos, each with its own suggested reason", () => {
  assert.deepEqual(bulkExcludeFlagged(mixed(), "all", { p1: { decision: "keep", reason: null } }), {
    p1: { decision: "keep", reason: null },
    p5: { decision: "exclude", reason: "habitat" },
    p6: { decision: "exclude", reason: "slip" },
    p7: { decision: "exclude", reason: "not_fungus" },
  });
  const got = bulkExcludeFlagged(record(), "micrograph", { "inat:2": { decision: "keep", reason: null } });
  assert.deepEqual(Object.keys(got).sort(), ["inat:1", "inat:2", "inat:4"]);
  assert.deepEqual(Object.keys(bulkExcludeFlagged(record(), "all")).sort(), ["inat:1", "inat:3", "inat:4"]);
  // A reason already on a photo stays; suggestions become choices.
  const bulk = bulkExcludeFlagged(mixed(), "all", { p6: { decision: "exclude", reason: "people", suggested: true } });
  assert.deepEqual(bulk.p6, { decision: "exclude", reason: "people" });
});

test("saving sends only the chosen photos, each exclusion with its own reason", () => {
  const choices: PhotoChoices = {
    p1: { decision: "keep", reason: null },
    p5: { decision: "exclude", reason: "habitat" },
    p6: { decision: "exclude", reason: "slip", suggested: true },
    p7: { decision: "keep", reason: "not_fungus" },
  };
  assert.deepEqual(saveBody(mixed(), choices, "  slips and a scene ", REASONS), { body: { photos: [
    { photo_key: "p1", decision: "keep" },
    { photo_key: "p5", decision: "exclude", reason: "habitat" },
    { photo_key: "p6", decision: "exclude", reason: "slip" },
    { photo_key: "p7", decision: "keep" }], note: "slips and a scene" } });
  // Suggest mode: saving straight away sends every suggested exclusion.
  const got = saveBody(mixed(), initialChoices(mixed(), "all", "suggest"), undefined, REASONS);
  assert.ok("body" in got);
  assert.deepEqual(got.body.photos.map((p) => [p.photo_key, p.reason]),
                   [["p5", "habitat"], ["p6", "slip"], ["p7", "not_fungus"]]);
});

test("saving refuses an exclusion without a reason, naming the photo, and nothing chosen", () => {
  assert.deepEqual(saveBody(record(), {}), { error: "Keep or exclude at least one photo first." });
  const noReason: PhotoChoices = { p5: { decision: "exclude", reason: "habitat" }, p1: { decision: "exclude", reason: null } };
  assert.deepEqual(saveBody(mixed(), noReason), { error: "Pick why photo 1 is left out." });
  assert.equal(missingReason(mixed(), noReason)?.photo_key, "p1");
  assert.equal(missingReason(mixed(), { ...noReason, p1: { decision: "exclude", reason: " " } })?.photo_key, "p1");
  assert.equal(missingReason(mixed(), { p5: { decision: "exclude", reason: "habitat" } }), null);
  assert.deepEqual(saveBody(mixed(), { p6: { decision: "exclude", reason: "nope" } }, undefined, REASONS),
                   { error: "Pick a listed reason for photo 6." });
  // A choice for a photo that is not the record's is never sent.
  assert.ok("error" in saveBody(record(), { "inat:99": { decision: "keep", reason: null } }));
});

test("Save counts the chosen photos, and a saved card says how many were left out as suggested", () => {
  assert.equal(chosenCount(record(), {}), 0);
  assert.equal(chosenCount(record(), initialChoices(record(), "micrograph")), 1);
  const choices: PhotoChoices = {
    a: { decision: "exclude", reason: "slip", suggested: true },
    b: { decision: "exclude", reason: "slip" },
    c: { decision: "keep", reason: null }, d: { decision: "keep", reason: null }, e: { decision: "keep", reason: null },
  };
  const body = { photos: [{ photo_key: "a", decision: "exclude" as const, reason: "slip" },
                          { photo_key: "b", decision: "exclude" as const, reason: "slip" },
                          { photo_key: "c", decision: "keep" as const }, { photo_key: "d", decision: "keep" as const },
                          { photo_key: "e", decision: "keep" as const }] };
  assert.equal(savedLine(body, choices), "Saved: 2 left out (1 as suggested), 3 kept");
  assert.equal(savedLine(body, { ...choices, a: { decision: "exclude", reason: "slip" } }), "Saved: 2 left out, 3 kept");
  assert.deepEqual(decisionsOf(choices), { a: "exclude", b: "exclude", c: "keep", d: "keep", e: "keep" });
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

test("on the list: j/k move between records, arrows between photos, Shift+K keeps, x excludes, r cycles the reason", () => {
  assert.deepEqual(keyAction({ key: "j" }, list), { type: "record", step: 1 });
  assert.deepEqual(keyAction({ key: "k" }, list), { type: "record", step: -1 });
  assert.deepEqual(keyAction({ key: "ArrowRight" }, list), { type: "photo", step: 1 });
  assert.deepEqual(keyAction({ key: "ArrowLeft" }, list), { type: "photo", step: -1 });
  assert.deepEqual(keyAction({ key: "K" }, list), { type: "choose", choice: "keep" });
  assert.deepEqual(keyAction({ key: "x" }, list), { type: "choose", choice: "exclude" });
  assert.deepEqual(keyAction({ key: "X" }, list), { type: "choose", choice: "exclude" });
  assert.deepEqual(keyAction({ key: "r" }, list), { type: "reason" });
  assert.deepEqual(keyAction({ key: "R" }, list), { type: "reason" });
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

test("in the photo viewer: arrows move, K or k keeps, X excludes, R cycles the reason, Esc closes, nothing else", () => {
  const lb: KeyState = { typing: false, lightbox: true };
  assert.deepEqual(keyAction({ key: "Escape" }, lb), { type: "close" });
  assert.deepEqual(keyAction({ key: "ArrowLeft" }, lb), { type: "photo", step: -1 });
  assert.deepEqual(keyAction({ key: "ArrowRight" }, lb), { type: "photo", step: 1 });
  assert.deepEqual(keyAction({ key: "k" }, lb), { type: "choose", choice: "keep" });
  assert.deepEqual(keyAction({ key: "K" }, lb), { type: "choose", choice: "keep" });
  assert.deepEqual(keyAction({ key: "x" }, lb), { type: "choose", choice: "exclude" });
  assert.deepEqual(keyAction({ key: "R" }, lb), { type: "reason" });
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

// --- quick reasons (Steve, 2026-10-21) ----------------------------------------------------------

test("a quick reason leaves the record-view photo out for that reason, a kept or suggested one too", () => {
  const r = record();
  const kept = chooseDecision(initialChoices(r, "micrograph"), r.photos[1], "keep");
  for (const { code } of QUICK_REASONS) {
    assert.deepEqual(quickChoice(kept, "inat:1", code)["inat:1"], { decision: "exclude", reason: code });
  }
  const suggested = initialChoices(r, "micrograph", "suggest");
  assert.equal(suggested["inat:1"].suggested, true);
  assert.deepEqual(quickChoice(suggested, "inat:1", "micrograph")["inat:1"], { decision: "exclude", reason: "micrograph" },
                   "confirms the suggestion");
  const mine = quickChoice(kept, "inat:2", "spore_print");
  assert.equal(quickChoice(mine, "inat:2", "spore_print"), mine, "same reason: no change");
  const b = saveBody(r, mine, undefined, QUICK_REASONS.map((q) => q.code));
  assert.ok("body" in b && b.body.photos.some((p) => p.photo_key === "inat:2" && p.decision === "exclude" && p.reason === "spore_print"));
});

test("keys 1–5 are the quick reasons in the photo viewer only", () => {
  const lb: KeyState = { typing: false, lightbox: true };
  assert.deepEqual(keyAction({ key: "2" }, lb), { type: "quick", index: 1 });
  assert.equal(keyAction({ key: "2" }, { ...lb, lightbox: false }), null);
  assert.equal(keyAction({ key: "0" }, lb), null);
});
