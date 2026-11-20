// The photo review's Category box (Steve, 2026-11-20: "Within the decided tabs there should be a
// selectbox for Show only this category / other categories - showing only things that remain in
// the bucket or have been moved elsewhere"): its choices and counts, where it is shown, and the
// query each list sends.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { photoFlaggedQuery, photoReviewQuery, type PhotoShowFilter } from "../lib/api";
import { SHOW_FILTERS, showCount, showInUse, showsCategoryBox } from "../lib/photoReview";
import { CategorySelect } from "./PhotoReview";

const COUNTS = { stayed: 1840, left: 312 };
const box = (show: PhotoShowFilter = "all", counts: typeof COUNTS | null = COUNTS, off = false) =>
  renderToStaticMarkup(createElement(CategorySelect, { show, counts, onChange: () => {}, off }));

test("the choices, in order: all, still in this category, moved to other categories", () => {
  assert.deepEqual(SHOW_FILTERS.map((o) => [o.code, o.label]), [
    ["all", "All decided"],
    ["stayed", "Still in this category"],
    ["left", "Moved to other categories"],
  ]);
  for (const o of SHOW_FILTERS) assert.ok(o.title.length > 10, o.code);
});

test("the box is labelled Category, and each choice carries what it would show", () => {
  const h = box();
  assert.match(h, /<span[^>]*>Category<\/span>/);
  assert.match(h, /data-testid="photo-review-category"/);
  assert.match(h, />All decided<\/option>/);
  assert.match(h, />Still in this category \(1,840\)<\/option>/);
  assert.match(h, />Moved to other categories \(312\)<\/option>/);
  assert.match(box("left"), /<option value="left" title="[^"]*" selected="">/);
  assert.match(box("all", null), />Still in this category<\/option>/, "no number before counts load");
  assert.equal(showCount(COUNTS, "all"), "");
});

test("it is shown in Decided on one tab only, never in Open or on All", () => {
  assert.equal(showsCategoryBox("decided", "slip"), true);
  assert.equal(showsCategoryBox("open", "slip"), false);
  assert.equal(showsCategoryBox("decided", "all"), false);
});

test("the pick is asked for only where the box is shown, and not with High score, not in this tab", () => {
  assert.equal(showInUse("decided", "slip", "all", "stayed"), "stayed");
  assert.equal(showInUse("decided", "slip", "moved", "left"), "left", "it combines with Outcome");
  assert.equal(showInUse("open", "slip", "all", "stayed"), "all");
  assert.equal(showInUse("decided", "all", "all", "stayed"), "all");
  assert.equal(showInUse("decided", "slip", "high_not_in_tab", "stayed"), "all");
  const h = box("stayed", COUNTS, true);
  assert.match(h, /<select[^>]*disabled=""/);
  assert.match(h, /<option value="all" title="[^"]*" selected="">/);
});

test("both lists send show only when one is picked", () => {
  const base = { tab: "slip" as const, status: "decided" as const };
  assert.match(photoFlaggedQuery({ ...base, show: "stayed" }), /(^|&)show=stayed(&|$)/);
  assert.match(photoReviewQuery({ ...base, show: "left" }), /(^|&)show=left(&|$)/);
  assert.doesNotMatch(photoFlaggedQuery({ ...base, show: "all" }), /show=/);
  assert.doesNotMatch(photoReviewQuery(base), /show=/);
});
