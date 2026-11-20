import { test } from "node:test";
import assert from "node:assert/strict";
import { photoLines, readOnlyNote } from "./feedbackReadOnly";

test("a visitor is asked to sign in to tag photos; an account below Contributor is told what is not theirs", () => {
  // Steve, 2026-11-20: any account works with photos; Taxonomy is for Contributors and up.
  assert.equal(readOnlyNote({ user: { id: "1", name: "A" }, read_only: false, taxonomy: true }), null);
  assert.equal(readOnlyNote({ user: { id: "1", name: "A" }, read_only: false }), null, "an older server");
  assert.equal(readOnlyNote(undefined), null);
  assert.deepEqual(readOnlyNote({ user: null, read_only: true }), {
    text: "Read-only: sign in with your mycomap.org account to tag photos.", signIn: true });
  const member = readOnlyNote({ user: { id: "9", name: null }, read_only: false, taxonomy: false });
  assert.equal(member?.signIn, false);
  assert.match(member!.text, /tag the photos anyone may see/);
  assert.match(member!.text, /Taxonomy queues are for mycomap\.org Contributors, Lab Contributors, Project Leaders and admins/);
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
