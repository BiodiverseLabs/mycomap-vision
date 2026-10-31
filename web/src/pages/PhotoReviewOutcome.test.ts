// The photo review's Outcome filter (Steve, 2026-10-30: "another dropdown at the top that would
// filter for things that were originally flagged but then changed to included or another
// category"): its options and counts, the query each list sends, and that it rests at "All
// outcomes" in Open.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { photoFlaggedQuery, photoReviewQuery, type PhotoOutcomeFilter } from "../lib/api";
import { OUTCOME_FILTERS, outcomeCount, outcomeInUse } from "../lib/photoReview";
import { OutcomeSelect } from "./PhotoReview";

const COUNTS = { as_flagged: 1234, included: 56, moved: 7, not_flagged: 0 };
const select = (status: "open" | "decided", outcome: PhotoOutcomeFilter = "all", counts: typeof COUNTS | null = COUNTS) =>
  renderToStaticMarkup(createElement(OutcomeSelect, { status, outcome, counts, onChange: () => {} }));

test("the options, in order, worded plainly, each with the sentence its hover shows", () => {
  assert.deepEqual(OUTCOME_FILTERS.map((o) => [o.code, o.label]), [
    ["all", "All outcomes"],
    ["as_flagged", "Kept as flagged"],
    ["included", "Changed to included"],
    ["moved", "Moved to another category"],
    ["not_flagged", "Not flagged for this tab"],
  ]);
  for (const o of OUTCOME_FILTERS) assert.ok(o.title.length > 10, o.code);
  // No "Not yet decided": that is Open.
  assert.ok(!OUTCOME_FILTERS.some((o) => /not yet/i.test(o.label)));
});

test("in Decided the select is labelled Outcome, live, and each option carries its count", () => {
  const h = select("decided");
  assert.match(h, /<span[^>]*>Outcome<\/span>/);
  assert.match(h, /data-testid="photo-review-outcome"/);
  assert.doesNotMatch(h, /disabled/);
  assert.match(h, />All outcomes<\/option>/);                       // no count on "All"
  assert.match(h, />Kept as flagged \(1,234\)<\/option>/);
  assert.match(h, />Changed to included \(56\)<\/option>/);
  assert.match(h, />Moved to another category \(7\)<\/option>/);
  assert.match(h, />Not flagged for this tab \(0\)<\/option>/);
  assert.match(select("decided", "moved"), /<option value="moved" title="[^"]*" selected="">/);
});

test("before the counts load, the options show no number", () => {
  assert.match(select("decided", "all", null), />Changed to included<\/option>/);
});

test("in Open it is disabled at All outcomes, with no counts, and says why", () => {
  const h = select("open", "moved");
  assert.match(h, /<select[^>]*disabled=""/);
  assert.match(h, /<option value="all" title="[^"]*" selected="">/);
  assert.doesNotMatch(h, /\(\d/);
  assert.match(h, /pick Decided/);
});

test("the outcome asked for is the pick in Decided, All in Open", () => {
  assert.equal(outcomeInUse("decided", "included"), "included");
  assert.equal(outcomeInUse("open", "included"), "all");
  assert.equal(outcomeInUse("decided", "all"), "all");
});

test("an option's count is what the list would show; nothing for All or before loading", () => {
  assert.equal(outcomeCount(COUNTS, "moved"), " (7)");
  assert.equal(outcomeCount(COUNTS, "not_flagged"), " (0)");
  assert.equal(outcomeCount(COUNTS, "all"), "");
  assert.equal(outcomeCount(undefined, "moved"), "");
  assert.equal(outcomeCount({}, "moved"), "");
});

test("both lists send the outcome only when one is picked, the other filters as before", () => {
  const base = { tab: "slip" as const, status: "decided" as const, sort: "p" as const, q: "x", band: "high" as const };
  assert.equal(photoFlaggedQuery({ ...base, offset: 150, limit: 150 }),
               "tab=slip&status=decided&sort=p&q=x&band=high&offset=150&limit=150");
  assert.equal(photoFlaggedQuery({ ...base, outcome: "all", offset: 150, limit: 150 }),
               "tab=slip&status=decided&sort=p&q=x&band=high&offset=150&limit=150");
  assert.equal(photoFlaggedQuery({ ...base, outcome: "moved", offset: 150, limit: 150 }),
               "tab=slip&status=decided&sort=p&q=x&band=high&offset=150&limit=150&outcome=moved");
  assert.equal(photoReviewQuery({ ...base, min_flagged: 2, outcome: "included", offset: 10 }),
               "tab=slip&status=decided&sort=p&q=x&min_flagged=2&band=high&offset=10&limit=10&outcome=included");
  assert.equal(photoReviewQuery({ ...base }),
               "tab=slip&status=decided&sort=p&q=x&min_flagged=1&band=high&offset=0&limit=10");
});
