import { test } from "node:test";
import assert from "node:assert/strict";
import type { FlaggedTile } from "./api";
import { dueDecisions, initialMarks, markTouched, mergeMarks, pendingDecisions, quickMark, reclassify, setDecision,
         setReason, tileStatus, type Marks } from "./photoSummary";

// Steve, 2026-10-21: "on the decided tab I should have the ability to open and reclassify an
// image as needed when viewing an image full size". The grid keeps a decided tile locked; the
// full-screen viewer reclassifies it, and the change saves like a flip (latest decision wins).

const decided = (id: number, decision: "exclude_photo" | "keep_photo", reason: string | null = null) =>
  ({ item_id: id, status: "decided", decision: { decision, reason }, suggested_reason: "micrograph" }) as unknown as FlaggedTile;

const start = (): Marks => initialMarks([decided(1, "exclude_photo", "micrograph"), decided(2, "keep_photo")]);

test("the grid never changes a decided tile", () => {
  const m = start();
  assert.equal(setDecision(m, 1, "keep"), m);
  assert.equal(setReason(m, 1, "slip"), m);
  assert.equal(quickMark(m, 1, "slip"), m);
});

test("the viewer reclassifies a decided tile: a new reason, Include, or Exclude for a reason", () => {
  const m = start();
  const r1 = reclassify(m, 1, (x) => setReason(x, 1, "slip"));
  assert.deepEqual(r1[1], { decision: "exclude", reason: "slip", reclassified: true });
  const r2 = reclassify(m, 1, (x) => setDecision(x, 1, "keep"));
  assert.deepEqual(r2[1], { decision: "keep", reason: "micrograph", reclassified: true });
  const r3 = reclassify(m, 2, (x) => quickMark(x, 2, "uv"));
  assert.deepEqual(r3[2], { decision: "exclude", reason: "uv", reclassified: true });
  // a change that ends where the stored decision is changes nothing
  assert.equal(reclassify(m, 1, (x) => setReason(x, 1, "micrograph")), m);
  assert.equal(reclassify(m, 1, (x) => setDecision(setDecision(x, 1, "keep"), 1, "exclude")), m);   // flipped back
  // an open tile: the change itself, never marked reclassified
  const open = { 9: { decision: "exclude" as const, reason: "slip" } };
  assert.deepEqual(reclassify(open, 9, (x) => setDecision(x, 9, "keep"))[9], { decision: "keep", reason: "slip" });
});

test("a reclassified tile stays changed across page reloads of the visit and is saved as a flip", () => {
  const tiles = [decided(1, "exclude_photo", "micrograph"), decided(2, "keep_photo")];
  const edits = { 1: reclassify(start(), 1, (x) => setReason(x, 1, "slip"))[1] };
  const marks = mergeMarks(initialMarks(tiles), edits);
  assert.deepEqual(marks[1], { decision: "exclude", reason: "slip", reclassified: true });
  assert.equal(marks[2].readOnly, true);
  const states = markTouched({}, 1);
  assert.deepEqual(dueDecisions(tiles, marks, states),
                   [{ item_id: 1, decision: "exclude", reason: "slip", note: "summary: flipped" }]);
  assert.deepEqual(pendingDecisions(tiles, marks, states).map((d) => d.item_id), [1]);
  assert.equal(tileStatus(marks[1], states[1], true), "pending");
  assert.equal(tileStatus(marks[2], undefined, true), "decided");
  // an untouched decided tile is never sent
  assert.deepEqual(dueDecisions(tiles, mergeMarks(initialMarks(tiles), {}), {}), []);
});
