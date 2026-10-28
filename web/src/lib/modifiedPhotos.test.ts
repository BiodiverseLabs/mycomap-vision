// Research > Excluded photos > Modified photos (Steve, 2026-10-27): the tab from its data alone,
// and the lines it says. The server's rules (which photos, the seal, who sees what, no map
// possible) are tested in tests/test_modified_photos.py.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Router } from "wouter";
import { cropLine, HELD_LINE, modifiedUrl, noMapLine, type ModifiedCard, type ModifiedPage } from "./excludedPhotos";
import { ModifiedPhotos } from "../components/ModifiedPhotos";
import { ExcludedPhotos } from "../pages/ExcludedPhotos";

const text = (html: string) => html.replace(/<[^>]+>/g, " ").replace(/&#x27;/g, "'").replace(/&amp;/g, "&").replace(/\s+/g, " ");
const MAP = { map: { crop: { x: 0, y: 0, w: 1, h: 1 }, grid: 2, map: [0, 85, 170, 255] },
              name: "Helvella sp. 'IN05'", rank: 1, top: 10, made_at: "2026-10-28T09:00:00Z" };

const card = (over: Partial<ModifiedCard> = {}): ModifiedCard => ({
  item_id: 1, photo_key: "inat:1", record_key: "129729182", name: "Helvella sp. 'IN05'",
  record_url: "https://www.inaturalist.org/observations/129729182", photo_url: "https://x/1.jpg",
  shown: "public", held: false, owner_login: "someone", license_code: "cc-by-nc", width: 40, height: 30,
  decided: false, decided_at: null, edited: false, map: null, no_map: null,
  crops: [{ n: 1, box: [0, 0, 20, 20], source: "detector", map: null, no_map: null },
          { n: 2, box: [20, 10, 40, 30], source: "reviewer", map: MAP, no_map: null }],
  ...over,
});

const page = (cards: ModifiedCard[]): ModifiedPage => ({
  total: cards.length, offset: 0, counts: { all: cards.length, edited: 1, detector: cards.length - 1 },
  filters: [{ code: "all", label: "All", count: cards.length }, { code: "edited", label: "Box edited by a reviewer", count: 1 },
            { code: "detector", label: "Detector's box", count: cards.length - 1 }],
  cards, include_open: true,
});

const render = (p: ModifiedPage) => renderToStaticMarkup(createElement(ModifiedPhotos, {
  page: p, filter: "all", withMaps: false, onFilter: () => {}, onWithMaps: () => {}, onPage: () => {} }));

test("each crop says whose box it is, and a map shows only where one is kept", () => {
  const html = render(page([card()]));
  const t = text(html);
  assert.match(t, /Crop 1 · detector's box/);
  assert.match(t, /Crop 2 · box edited by a reviewer/);
  assert.match(t, /Open · included by default/);
  assert.equal((html.match(/data-testid="text-map-coming"/g) ?? []).length, 2);     // the photo and crop 1
  assert.equal((html.match(/data-testid="text-map-line"/g) ?? []).length, 1);       // crop 2
  assert.match(t, /on its own it puts it first/);
});

test("a photo or crop no map can be made for says why instead of map coming", () => {
  const why = "no other DNA-verified record of Mycena corticola to compare with";
  const html = render(page([card({ no_map: why, crops: [{ n: 1, box: [0, 0, 20, 20], source: "detector", map: null, no_map: why }] })]));
  assert.equal((html.match(/data-testid="text-map-none"/g) ?? []).length, 2);
  assert.ok(!html.includes("text-map-coming"));
  assert.match(text(html), /No map: no other DNA-verified record of Mycena corticola to compare with\./);
  assert.equal(noMapLine("the record's name is not in the reference set."),
               "No map: the record's name is not in the reference set.");
});

test("the seal's line stands in for every map of a held record", () => {
  const html = render(page([card({ held: true })]));
  assert.equal((html.match(/data-testid="text-map-held"/g) ?? []).length, 3);
  assert.ok(text(html).includes(HELD_LINE.replace(/'/g, "'")));
  assert.ok(!html.includes("text-map-line"));
});

test("a photo not anyone may see is a placeholder: no image, crops or photographer", () => {
  const html = render(page([card({ shown: "hidden", photo_url: null, owner_login: null, crops: [] })]));
  assert.match(html, /data-testid="card-modified-hidden"/);
  assert.ok(!html.includes("<img") && !html.includes("figure-modified-crop") && !html.includes("someone"));
});

test("the page's address carries the filter, the page and the mapped-only switch", () => {
  assert.equal(modifiedUrl("edited", 12), "/api/research/excluded-photos/modified?filter=edited&offset=12&limit=12");
  assert.equal(modifiedUrl("all", 0, true), "/api/research/excluded-photos/modified?filter=all&offset=0&limit=12&maps=1");
  assert.equal(cropLine({ n: 3, source: "detector" }, false), "Crop · detector's box");
});

test("Excluded photos has a Modified photos tab beside the left-out photos", () => {
  const html = renderToStaticMarkup(createElement(QueryClientProvider, { client: new QueryClient() },
    createElement(Router, { ssrPath: "/research/excluded-photos", children: createElement(ExcludedPhotos, {
      me: { signin: "all", user: null }, category: "all", withMaps: false, onCategory: () => {}, onWithMaps: () => {},
      onPage: () => {}, view: "modified", onView: () => {}, modified: createElement("p", null, "the modified tab"), modifiedCount: 2429 }) })));
  assert.match(html, /data-testid="tab-excluded-modified"/);
  assert.match(html, /data-testid="section-excluded-modified"/);
  assert.match(text(html), /Modified photos 2,429/);
  assert.match(text(html), /the modified tab/);
});
