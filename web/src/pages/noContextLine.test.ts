// The line above an answer that used no place or date says why (Steve, 2026-11-24). An
// observation that is one of Vision's DNA-checked reference records has a place and date,
// but they are left out with the record itself, because the range-and-season score is built
// from those records. "No place or date given" was wrong for it (iNat 101576358).
import { test } from "node:test";
import assert from "node:assert/strict";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { ContextLine } from "./Identify";
import { RunContext } from "./Run";
import { noContextWords } from "../lib/photoPlace";
import type { ContextUsed } from "../lib/api";

const NONE: ContextUsed = { latitude: null, longitude: null, observed_on: null, place_from: null, date_from: null };
const text = (h: string) => h.replace(/<[^>]+>/g, "").replace(/&#x27;/g, "'");

test("a reference record's link says its place and date were left out, not that none was given", () => {
  for (const placeModel of [true, false]) {
    const h = text(renderToStaticMarkup(createElement(ContextLine, { used: NONE, placeModel, excludedReference: true })));
    assert.doesNotMatch(h, /no place or date given|No place or date given/, `placeModel=${placeModel}`);
    assert.match(h, /DNA-checked records/);
    assert.match(h, /place and date (weren't|aren't) used/);
  }
});

test("a find with no place or date given still says so", () => {
  assert.match(text(renderToStaticMarkup(createElement(ContextLine, { used: NONE, placeModel: true }))),
               /^No place or date given or found in the photos, so models with a range-and-season score used the photos alone\.$/);
  assert.match(text(renderToStaticMarkup(createElement(ContextLine, { used: NONE, placeModel: false }))),
               /^From the photos alone \(no place or date given\)\.$/);
});

test("a kept run's page gives the same reason for a reference record", () => {
  const ref = text(renderToStaticMarkup(createElement(RunContext, { used: NONE, excludedReference: true })));
  assert.equal(ref, noContextWords(false, true));
  assert.doesNotMatch(ref, /no place or date given/);
  assert.equal(text(renderToStaticMarkup(createElement(RunContext, { used: NONE }))), "From the photos alone (no place or date given).");
});

test("a place that was used is shown, whatever the record", () => {
  const used: ContextUsed = { latitude: 40.1, longitude: -83, observed_on: "2026-10-11", place_from: "observation", date_from: "observation" };
  const h = text(renderToStaticMarkup(createElement(ContextLine, { used, placeModel: true, excludedReference: true })));
  assert.match(h, /near 40\.1, -83 \(from the observation\)/);
  assert.doesNotMatch(h, /DNA-checked/);
});
