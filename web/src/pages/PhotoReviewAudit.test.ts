// The rescoring audit's Outcome option on the page's select (lib/photoAudit.ts): offered on a
// scored tab's Decided Summary, with its count like the others; absent elsewhere.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { OUTCOME_FILTERS } from "../lib/photoReview";
import { outcomeOptions } from "../lib/photoAudit";
import { OutcomeSelect } from "./PhotoReview";

const COUNTS = { as_flagged: 1234, included: 56, moved: 7, not_flagged: 0, high_not_in_tab: 89 };
const select = (available: boolean, outcome: "all" | "high_not_in_tab" = "all") =>
  renderToStaticMarkup(createElement(OutcomeSelect, {
    status: "decided", outcome, counts: COUNTS, onChange: () => {},
    options: outcomeOptions(OUTCOME_FILTERS, available) }));

test("with an audit for the tab the select offers High score, not in this tab, with its count", () => {
  const h = select(true);
  assert.match(h, /<option value="high_not_in_tab" title="[^"]*may have missed[^"]*">High score, not in this tab \(89\)<\/option>/);
  assert.match(h, />Kept as flagged \(1,234\)<\/option>/);               // the others unchanged
  assert.match(select(true, "high_not_in_tab"), /<option value="high_not_in_tab" title="[^"]*" selected="">/);
});

test("without one the select is as it was", () => {
  const h = select(false);
  assert.doesNotMatch(h, /high_not_in_tab/);
  assert.equal(h, renderToStaticMarkup(createElement(OutcomeSelect, {
    status: "decided", outcome: "all", counts: COUNTS, onChange: () => {} })));
});
