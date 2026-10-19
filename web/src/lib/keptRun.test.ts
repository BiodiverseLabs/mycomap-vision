import { test } from "node:test";
import assert from "node:assert/strict";
import { askedBy, runDetails, type Answered } from "./keptRun";

const when = (iso: string) => `[${iso}]`;
const answered = (over: Partial<Answered> = {}): Answered =>
  ({ at: "2026-10-19T16:29:31Z", by: "mycomap.org", answers_version: "v", current: true, ...over });

test("a kept answer says when it was run and who asked", () => {
  const d = runDetails(answered(), true, when);
  assert.equal(d.ran, "Run [2026-10-19T16:29:31Z] from mycomap.org.");
  assert.match(d.note!, /nothing was run again/);
  assert.match(d.note!, /haven't changed/);
});

test("an answer from an older reference set says a new run may differ", () => {
  const d = runDetails(answered({ current: false, by: "site" }), true, when);
  assert.equal(d.ran, "Run [2026-10-19T16:29:31Z] on this site.");
  assert.match(d.note!, /reference set has changed since/);
});

test("a fresh run says it is kept for next time", () => {
  const d = runDetails(answered({ by: "site" }), false, when);
  assert.equal(d.ran, "Run just now on this site.");
  assert.match(d.note!, /until someone runs it again/);
});

test("who asked, in words; unknown or missing says nothing", () => {
  assert.equal(askedBy("app"), "from the MycoMap app");
  assert.equal(askedBy("api"), "through the Vision API");
  assert.equal(askedBy(null), "");
  assert.equal(askedBy("someone-new"), "");
  assert.equal(runDetails(answered({ by: null }), true, when).ran, "Run [2026-10-19T16:29:31Z].");
});
