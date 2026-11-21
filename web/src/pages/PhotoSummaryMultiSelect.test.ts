// Multiple select on the photo review's Summary (Steve, 2026-11-21: "a checkbox option here for
// "Multiple Select" if checked I would be able to click individual photos down through the list
// and then bulk update them to a category"): the box, the bar and a selected tile as drawn. The
// rules (what a click selects, what Apply sends) are in lib/photoSummary.test.ts.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import type { FlaggedTile } from "../lib/api";
import { MultiSelectBox } from "./PhotoReview";
import { BulkBar, SummaryTile } from "./PhotoSummary";

const REASONS = [{ code: "slip", label: "Field slip or label" }, { code: "culture", label: "Culture or mycelium" }];

const TILE: FlaggedTile = {
  item_id: 7, photo_key: "inat:1000", record_key: "100", label: "Amanita muscaria",
  link: "https://www.inaturalist.org/observations/100", flags: [], strength: 0.9,
  suggested_reason: "slip", status: "decided", decision: { decision: "exclude_photo", reason: "slip" }, shown: "public",
  thumb: "https://img.example/1/small.jpg", full: "https://img.example/1/large.jpg", position: 1,
  license_class: "open", license_code: "cc-by", owner_login: "alice", record_photos: 4,
};

const tile = (over: { selecting?: boolean; isSelected?: boolean } = {}) =>
  renderToStaticMarkup(createElement(SummaryTile, {
    tile: TILE, reasons: REASONS, mark: { decision: "exclude", reason: "slip", readOnly: true }, state: undefined,
    focused: false, tileRef: () => {}, manual: false, onFocus: () => {}, onOpen: () => {}, onFlip: () => {},
    onRetry: () => {}, onSave: () => {}, onLoaded: () => {}, onSelect: () => {}, ...over }));

test("the Multiple select box is a labelled checkbox, unticked until picked", () => {
  const off = renderToStaticMarkup(createElement(MultiSelectBox, { on: false, onChange: () => {} }));
  assert.match(off, /type="checkbox"/);
  assert.match(off, /data-testid="photo-review-multi"/);
  assert.match(off, />Multiple select<\/span>/);
  assert.doesNotMatch(off, /checked=""/);
  assert.match(renderToStaticMarkup(createElement(MultiSelectBox, { on: true, onChange: () => {} })), /checked=""/);
});

test("with it off a photo opens full screen; with it on the photo selects, and shows whether it is selected", () => {
  const off = tile();
  assert.match(off, /aria-label="Open [^"]+ full screen"/);
  assert.doesNotMatch(off, /photo-summary-select-mark/);
  assert.doesNotMatch(off, /aria-pressed/);
  const on = tile({ selecting: true });
  assert.match(on, /aria-label="Select [^"]+"/);
  assert.match(on, /aria-pressed="false"/);
  assert.match(on, /data-selected="false"/);
  assert.match(on, /photo-summary-select-mark/);
  const picked = tile({ selecting: true, isSelected: true });
  assert.match(picked, /aria-label="Unselect [^"]+"/);
  assert.match(picked, /aria-pressed="true"/);
  assert.match(picked, /data-selected="true"/);
  assert.match(picked, /ring-sky-500/);
});

test("the bar counts the selection, offers Include and every reason, and Apply waits for both", () => {
  const bar = (count: number) => renderToStaticMarkup(createElement(BulkBar, {
    count, reasons: REASONS, note: null, onApply: () => {}, onAll: () => {}, onClear: () => {} }));
  const none = bar(0);
  assert.match(none, /Click photos to select them; Shift-click selects a run\./);
  assert.match(none, /<option value="keep">Include \(no tag\)<\/option>/);
  assert.match(none, /<option value="exclude:slip">Exclude: Field slip or label<\/option>/);
  assert.match(none, /<option value="exclude:culture">Exclude: Culture or mycelium<\/option>/);
  assert.match(none, /<button[^>]*disabled=""[^>]*data-testid="photo-summary-bulk-apply"/);
  assert.match(none, />Select all on page</);
  const some = bar(12);
  assert.match(some, /12 photos selected/);
  assert.match(some, />Apply to 12</);
  // Nothing picked in "Set to" yet: Apply stays off even with photos selected.
  assert.match(some, /<button[^>]*disabled=""[^>]*data-testid="photo-summary-bulk-apply"/);
});
