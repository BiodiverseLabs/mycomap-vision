import { test } from "node:test";
import assert from "node:assert/strict";
import type { LabelReviewCard, LabelReviewPhoto } from "./api";
import { REVEAL_BUTTON, REVEAL_CONFIRM } from "./feedback";
import { decidedSummary, excludedCount, isMasked, LABEL_KEY_HELP, LABEL_REVIEW_ABOUT, OUTCOME_BUTTONS, labelKeyAction, maskedCard, outcomeHint,
         outcomeStates, outcomeWords, saveBody, savedLine, sortedPhotos, splitStoredNote, visionSaid,
         WHY_LEFT_OUT, WHY_NO_NAME, WHY_NO_REASON, WHY_NONE_LEFT_OUT } from "./labelReview";

const photo = (over: Partial<LabelReviewPhoto> = {}): LabelReviewPhoto => ({
  photo_key: "inat:1", position: 1, license_class: "cc", shown: "public",
  thumb: "https://img.example/1/medium.jpg", full: "https://img.example/1/large.jpg", excluded: false, ...over,
});

const REASONS = [
  { code: "other_organism", label: "Shows another organism" },
  { code: "no_specimen", label: "No usable specimen" },
  { code: "other", label: "Other" },
];
const CODES = REASONS.map((r) => r.code);

/** An open card as the server sends it: photos 2, 1, 3 out of order. */
const open = (over: Partial<LabelReviewCard> = {}): LabelReviewCard => ({
  item_id: 7, record_key: "102", label: "A x", link: "https://www.inaturalist.org/observations/102",
  status: "open", masked: true,
  photos: [photo({ photo_key: "inat:2", position: 2 }), photo({ photo_key: "inat:1", position: 1 }),
           photo({ photo_key: "inat:3", position: 3, shown: "private", thumb: "/api/feedback/photo/inat:3" })],
  ...over,
});

const SECRET = { rule: "M1 truth ranked >300",
                 evidence: { evidence: "Vision: B y 0.71; C z 0.40", in_v1: "val", v1_val_outcome: "miss" },
                 decision: { decision: "hard_example", value: null, photo_keys: [], note: null, by: "S", at: 1 } };

const decided = (over: Partial<LabelReviewCard> = {}): LabelReviewCard => ({
  ...open(), status: "decided", masked: false, ...SECRET,
  decision: { decision: "exclude_photos", value: null, photo_keys: ["inat:2"],
              note: "reason: other_organism; a slime mould", by: "Steve", at: 1760800000 },
  photos: [photo({ photo_key: "inat:1", position: 1 }), photo({ photo_key: "inat:2", position: 2, excluded: true })],
  ...over,
});

// --- the blind rule --------------------------------------------------------------------------

test("an open card shows the record, its label and photos only, even when the server sent Vision's answer", () => {
  const leaky = { ...open(), ...SECRET, model_flagged: true, score: 0.71 } as LabelReviewCard;
  const shown = maskedCard(leaky);
  assert.deepEqual(Object.keys(shown).sort(),
                   ["item_id", "label", "link", "masked", "photos", "record_key", "revealed", "status"]);
  assert.equal(shown.revealed, null);
  assert.equal(shown.label, "A x");
  assert.equal(shown.photos.length, 3);
  const text = JSON.stringify(shown);
  for (const word of ["Vision", "M1 truth", "v1_val_outcome", "model_flagged", "score", "hard_example"]) {
    assert.ok(!text.includes(word), word);
  }
});

test("an open card is masked even if the server forgot to mark it, and its photos carry no stored outcome", () => {
  const unmarked = { ...open({ masked: false }), ...SECRET } as LabelReviewCard;
  assert.ok(isMasked(unmarked));
  const shown = maskedCard({ ...unmarked, photos: [photo({ excluded: true, extra: "x" } as Partial<LabelReviewPhoto>)] });
  assert.equal(shown.rule, undefined);
  assert.equal(shown.evidence, undefined);
  assert.equal(shown.decision, undefined);
  assert.deepEqual(shown.photos, [{ photo_key: "inat:1", position: 1, license_class: "cc", shown: "public",
                                    thumb: "https://img.example/1/medium.jpg",
                                    full: "https://img.example/1/large.jpg", excluded: false }]);
  assert.equal(visionSaid(unmarked), null);
  assert.equal(decidedSummary(unmarked, REASONS), null);
});

test("every open card looks the same: the same fields whatever the server held back", () => {
  const flagged = maskedCard({ ...open(), ...SECRET } as LabelReviewCard);
  const blind = maskedCard(open({ item_id: 8, record_key: "101" }));
  assert.deepEqual(Object.keys(flagged).sort(), Object.keys(blind).sort());
  // Before anyone asks, no card carries an answer: revealed is null on both alike.
  assert.equal(flagged.revealed, null);
  assert.equal(blind.revealed, null);
});

test("a card this person chose to reveal keeps its own answer, and only that answer", () => {
  const answer = { available: true, top5: [["B y", 0.71], ["A x", 0.4]] as [string, number][], reason: null };
  const shown = maskedCard({ ...open(), ...SECRET,
                             revealed: { at: 1760800000, answer: { ...answer, rule: "M1 truth", seed: 3 }, blind: true },
                           } as unknown as LabelReviewCard);
  assert.deepEqual(shown.revealed, { at: 1760800000, answer });
  assert.equal(shown.rule, undefined);
  assert.equal(shown.evidence, undefined);
  for (const word of ["M1 truth", "seed", "blind", "hard_example"]) {
    assert.ok(!JSON.stringify(shown).includes(word), word);
  }
  const none = maskedCard(open({ revealed: { at: 2, answer: { available: false, top5: [], reason: "no photos" } } }));
  assert.deepEqual(none.revealed, { at: 2, answer: { available: false, top5: [], reason: "no photos" } });
});

test("the page's own words say nothing about Vision or why records are in the queue, but for the one reveal button", () => {
  for (const text of [LABEL_REVIEW_ABOUT, ...Object.values(OUTCOME_BUTTONS)]) {
    assert.ok(!/vision|model|flag|rank|blind|score/i.test(text), text);
  }
  // "Show Vision's answer" and its confirm must name Vision. They are the only exception: fixed
  // words, the same on every card, that say nothing else about Vision or why a record is here.
  for (const text of [REVEAL_BUTTON, REVEAL_CONFIRM]) {
    const rest = text.replace(/Vision's answer/g, "");
    assert.ok(!/vision|model|flag|rank|blind|score|sample/i.test(rest), text);
  }
});

test("a decided card shows what Vision said, worded for reviewers", () => {
  const card = decided({ rule: "blind sample (seed 3)" });
  assert.equal(maskedCard(card), card);
  const said = visionSaid(card)!;
  assert.equal(said.rule, "check sample");
  assert.deepEqual(said.rows[0], ["Evidence", "Vision: B y 0.71; C z 0.40"]);
  assert.ok(said.rows.some(([k]) => k === "In v1"));
});

// --- photos and outcomes ---------------------------------------------------------------------

test("photos show in position order, one without a position last", () => {
  assert.deepEqual(sortedPhotos(open().photos).map((p) => p.photo_key), ["inat:1", "inat:2", "inat:3"]);
  assert.deepEqual(sortedPhotos([photo({ photo_key: "x", position: null }), photo({ photo_key: "y", position: 2 })])
    .map((p) => p.photo_key), ["y", "x"]);
});

test("with nothing chosen, a hard example may be saved, a rename once a name is typed, leaving out never", () => {
  const s = outcomeStates(open(), {}, { reason: "other_organism", value: "" }, CODES);
  assert.deepEqual(s.exclude_photos, { enabled: false, why: WHY_NONE_LEFT_OUT });
  assert.deepEqual(s.hard_example, { enabled: true });
  assert.deepEqual(s.rename, { enabled: false, why: WHY_NO_NAME });
  assert.deepEqual(outcomeStates(open(), { "inat:1": "keep" }, { value: "  B y " }).rename, { enabled: true });
  assert.equal(outcomeHint(s), `${WHY_NONE_LEFT_OUT} ${WHY_NO_NAME}`);
});

test("once a photo is left out, only 'Leave out the chosen photos' may be saved, and it needs a known reason", () => {
  const choices = { "inat:2": "exclude", "inat:1": "keep" } as const;
  const s = outcomeStates(open(), choices, { reason: "other_organism", value: "B y" }, CODES);
  assert.deepEqual(s.exclude_photos, { enabled: true });
  assert.deepEqual(s.hard_example, { enabled: false, why: WHY_LEFT_OUT });
  assert.deepEqual(s.rename, { enabled: false, why: WHY_LEFT_OUT });
  assert.equal(outcomeHint(s), WHY_LEFT_OUT);
  assert.deepEqual(outcomeStates(open(), choices, { reason: "" }, CODES).exclude_photos,
                   { enabled: false, why: WHY_NO_REASON });
  assert.deepEqual(outcomeStates(open(), choices, { reason: "slime" }, CODES).exclude_photos,
                   { enabled: false, why: WHY_NO_REASON });
  // A choice for a photo that is not on the card does not count.
  assert.equal(excludedCount(open(), { "inat:99": "exclude" }), 0);
});

test("the saved body carries only the chosen photos, in order; the reason only to leave out, the name only to rename", () => {
  const card = open();
  const leave = saveBody(card, { "inat:3": "exclude", "inat:1": "keep", "inat:99": "exclude" }, "exclude_photos",
                         { reason: "no_specimen", value: "B y", note: "  slip  " }, CODES);
  assert.deepEqual(leave, { body: { outcome: "exclude_photos", reason: "no_specimen", note: "slip",
                                    photos: [{ photo_key: "inat:1", decision: "keep" },
                                             { photo_key: "inat:3", decision: "exclude" }] } });
  assert.deepEqual(saveBody(card, { "inat:2": "keep" }, "hard_example", { reason: "other_organism", value: "B y" }),
                   { body: { outcome: "hard_example", photos: [{ photo_key: "inat:2", decision: "keep" }] } });
  assert.deepEqual(saveBody(card, {}, "rename", { reason: "other_organism", value: "  B y  ", note: "" }),
                   { body: { outcome: "rename", photos: [], value: "B y" } });
});

test("invalid combinations are refused before anything is sent", () => {
  const card = open();
  const left = { "inat:1": "exclude" } as const;
  assert.deepEqual(saveBody(card, {}, "exclude_photos", { reason: "other_organism" }), { error: WHY_NONE_LEFT_OUT });
  assert.deepEqual(saveBody(card, left, "exclude_photos", {}), { error: WHY_NO_REASON });
  assert.deepEqual(saveBody(card, left, "hard_example"), { error: WHY_LEFT_OUT });
  assert.deepEqual(saveBody(card, left, "rename", { value: "B y" }), { error: WHY_LEFT_OUT });
  assert.deepEqual(saveBody(card, {}, "rename", { value: "   " }), { error: WHY_NO_NAME });
  assert.ok("error" in saveBody(card, {}, "maybe" as "rename"));
});

// --- the outcome in words --------------------------------------------------------------------

test("a saved card collapses to its outcome in words", () => {
  assert.equal(savedLine({ outcome: "exclude_photos", reason: "other_organism",
                           photos: [{ photo_key: "a", decision: "exclude" }, { photo_key: "b", decision: "keep" }] },
                         REASONS), "Saved: left out 1 photo (Shows another organism)");
  assert.equal(savedLine({ outcome: "hard_example", photos: [] }), "Saved: label and photos are right (a hard example)");
  assert.equal(savedLine({ outcome: "rename", photos: [], value: "B y" }), "Saved: renamed to B y");
  assert.equal(outcomeWords("exclude_photos", { left_out: 2, reason: "mystery" }), "left out 2 photos (mystery)");
});

test("a decided card says its stored outcome, the reviewer's own note and who decided", () => {
  assert.deepEqual(decidedSummary(decided(), REASONS),
                   { words: "left out 1 photo (Shows another organism)", note: "a slime mould", by: "Steve", at: 1760800000 });
  const renamed = decided({ decision: { decision: "rename", value: "B y", photo_keys: [], note: null, by: "S", at: 2 } });
  assert.equal(decidedSummary(renamed, REASONS)!.words, "renamed to B y");
  assert.deepEqual(splitStoredNote("reason: no_specimen"), { reason: "no_specimen", note: null });
  assert.deepEqual(splitStoredNote("just a note; with a semicolon"), { reason: null, note: "just a note; with a semicolon" });
  assert.deepEqual(splitStoredNote(null), { reason: null, note: null });
});

// --- keys ------------------------------------------------------------------------------------

test("the keys: j/k records, arrows photos, Shift+K keep, X exclude, E leave out, H hard example, R rename, ? help", () => {
  const list = { typing: false, lightbox: false };
  const on = (key: string) => labelKeyAction({ key }, list);
  assert.deepEqual(on("j"), { type: "record", step: 1 });
  assert.deepEqual(on("k"), { type: "record", step: -1 });
  assert.deepEqual(on("K"), { type: "choose", choice: "keep" });
  assert.deepEqual(on("x"), { type: "choose", choice: "exclude" });
  assert.deepEqual(on("ArrowRight"), { type: "photo", step: 1 });
  assert.deepEqual(on("ArrowLeft"), { type: "photo", step: -1 });
  assert.deepEqual(on("e"), { type: "outcome", outcome: "exclude_photos" });
  assert.deepEqual(on("H"), { type: "outcome", outcome: "hard_example" });
  assert.deepEqual(on("r"), { type: "rename" });
  assert.deepEqual(on("?"), { type: "help" });
  assert.equal(on("Enter"), null);                                   // three outcomes: no one "save"
  assert.equal(on("Escape"), null);
  assert.deepEqual(labelKeyAction({ key: "Escape" }, { ...list, help: true }), { type: "help" });
  assert.equal(labelKeyAction({ key: "e", ctrlKey: true }, list), null);
  assert.equal(LABEL_KEY_HELP.length, 9);
});

test("no key acts while typing, and the photo viewer has its own few", () => {
  for (const key of ["j", "k", "K", "x", "e", "h", "r", "?", "ArrowLeft"]) {
    assert.equal(labelKeyAction({ key }, { typing: true, lightbox: false }), null, key);
  }
  const lb = { typing: false, lightbox: true };
  assert.deepEqual(labelKeyAction({ key: "Escape" }, lb), { type: "close" });
  assert.deepEqual(labelKeyAction({ key: "k" }, lb), { type: "choose", choice: "keep" });
  assert.deepEqual(labelKeyAction({ key: "X" }, lb), { type: "choose", choice: "exclude" });
  assert.deepEqual(labelKeyAction({ key: "ArrowRight" }, lb), { type: "photo", step: 1 });
  for (const key of ["e", "h", "r", "j"]) assert.equal(labelKeyAction({ key }, lb), null, key);
});
