import { test } from "node:test";
import assert from "node:assert/strict";
import { photoLines, readOnlyNote } from "./feedbackReadOnly";

test("a read-only viewer is told so, and asked to sign in only when signed out", () => {
  assert.equal(readOnlyNote({ user: { id: "1", name: "A" }, read_only: false }), null);
  assert.equal(readOnlyNote(undefined), null);
  assert.deepEqual(readOnlyNote({ user: null, read_only: true }), {
    text: "Read-only: mycomap.org admins, project leaders and contributors sign in to review.", signIn: true });
  assert.equal(readOnlyNote({ user: { id: "9", name: null }, read_only: true })?.signIn, false);
});

test("a photo reads as its flags and its outcome, never who decided", () => {
  const flags = [{ tab: "micrograph", kind: "micrograph probe", p: 0.8, strength: 0.8 }];
  assert.deepEqual(photoLines({ flags, status: "open", decision: null }),
                   ["Flagged: micrograph probe (p 0.80)", "Open"]);
  assert.deepEqual(photoLines({ flags, status: "decided", decision: { decision: "exclude_photo", reason: "micrograph" } },
                              (c) => (c === "micrograph" ? "Micrograph" : c)),
                   ["Flagged: micrograph probe (p 0.80)", "Decided: Exclude (Micrograph)"]);
  assert.deepEqual(photoLines({ status: "decided", decision: { decision: "keep", reason: null } }),
                   ["Decided: Include"]);
  assert.deepEqual(photoLines({ suggested: [{ view: "gills", p: 0.9 }], status: "decided",
                                decision: { decision: "confirm_views", views: ["gills", "cap"] } }),
                   ["Suggested: gills (p 0.90)", "Decided: gills, cap"]);
});
