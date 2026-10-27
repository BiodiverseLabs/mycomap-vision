// Research > Excluded photos (Steve, 2026-10-26): the page from its data alone, and the lines it
// says. The server's rules (who sees what, the seal, the limits) are tested in
// tests/test_excluded_looked_api.py.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Router } from "wouter";
import {
  CATEGORIES, categoryLabel, HELD_LINE, listUrl, lookedUrl, rangeLine, rankLine, refusal,
  type ExcludedPage,
} from "./excludedPhotos";
import { researchBar, researchNav } from "./research";
import { ExcludedPhotos, PhotoDialog } from "../pages/ExcludedPhotos";

const render = (page?: ExcludedPage, extra: Partial<Parameters<typeof ExcludedPhotos>[0]> = {}) => renderToStaticMarkup(
  createElement(QueryClientProvider, { client: new QueryClient() }, createElement(Router, {
    ssrPath: "/research/excluded-photos",
    children: createElement(ExcludedPhotos, { me: { signin: "all", user: null }, page, category: "all", withMaps: false,
      onCategory: () => {}, onWithMaps: () => {}, onPage: () => {}, ...extra }),
  })));
const text = (html: string) => html.replace(/<[^>]+>/g, " ").replace(/&#x27;/g, "'").replace(/&amp;/g, "&").replace(/\s+/g, " ");

const PAGE: ExcludedPage = {
  categories: [{ code: "slip", label: "Field slips and labels", count: 16245 },
               { code: "micrograph", label: "Micrographs", count: 37073 }],
  total: 53318, offset: 0, as_of: "2026-10-26T12:00:00Z",
  tiles: [
    { photo_key: "inat:1", record_key: "369903305", name: "Hemimycena sp. 'CA01'", category: "slip", shown: "public",
      thumb: "https://x/small.jpg", photo_url: "https://x/large.jpg", owner_login: "benkeen", license_code: "cc0",
      record_url: "https://www.inaturalist.org/observations/369903305", maps: ["v1"] },
    { photo_key: "inat:2", record_key: "2", name: "Russula emetica", category: "micrograph", shown: "hidden",
      thumb: null, photo_url: null, owner_login: null, license_code: null, record_url: null, maps: [] },
  ],
};

test("the page sits in the Research section for everyone, before For researchers", () => {
  for (const nav of [researchNav(null), researchBar()]) {
    const hrefs = nav.map((n) => n.href);
    const at = hrefs.indexOf("/research/excluded-photos");
    assert.ok(at > 0 && hrefs[at + 1] === "/research/for-researchers", hrefs.join());
  }
});

test("every category is a reason a person can give, labelled for the page", () => {
  assert.equal(CATEGORIES.length, 14);
  assert.equal(categoryLabel("slip"), "Field slips and labels");
  assert.equal(categoryLabel("something-new"), "something-new");
});

test("a category's chip counts its photos, and All counts them all", () => {
  const t = text(render(PAGE));
  assert.match(t, /All 53,318/);
  assert.match(t, /Field slips and labels 16,245/);
  assert.match(t, /53,318 photos left out by people, in 2 kinds/);
  assert.match(t, /1–2 of 53,318/);
});

test("a photo not everyone may see is a placeholder, with no image", () => {
  const html = render(PAGE);
  assert.equal((html.match(/<img /g) ?? []).length, 1);
  assert.ok(html.includes("https://x/small.jpg") && !html.includes("https://x/large.jpg"));
  assert.match(text(html), /Not shown publicly/);
  assert.match(text(html), /never shown/);                 // who decided
});

test("a photo someone already mapped is marked on its tile", () => {
  assert.equal((render(PAGE).match(/> ?map<\/span>/g) ?? []).length, 1);
});

test("the page waits for its list, and says when it could not be read", () => {
  assert.match(text(render(undefined, { loading: true })), /Reading Human Feedback/);
  assert.match(text(render(undefined, { failed: true })), /could not be loaded/);
});

test("the photo alone puts the name first, at a number, or below the top", () => {
  assert.equal(rankLine(1), "first");
  assert.equal(rankLine(4), "at number 4");
  assert.equal(rankLine(null, 10), "not in its top 10");
});

test("each refusal is said in words", () => {
  assert.match(refusal(401, ""), /Sign in/);
  assert.equal(refusal(403, ""), HELD_LINE);
  assert.match(HELD_LINE, /sealed/);
  assert.match(refusal(409, "no other DNA-verified record of A x to compare with."), /no other DNA-verified record of A x to compare with\.$/);
  assert.match(refusal(429, ""), /try again in a minute/);
});

test("the addresses carry the category, page, map filter, photo and model", () => {
  assert.equal(listUrl("slip", 48, true), "/api/research/excluded-photos?category=slip&offset=48&limit=48&maps=1");
  assert.equal(lookedUrl("mo:12"), "/api/research/excluded-photos/looked?photo=mo%3A12&model=v1");
  assert.equal(rangeLine(0, 0, 0), "No photos");
});

test("a photo's map is v1's, as a spotlight: no Glow, no BioCLIP base, no side by side", () => {
  const html = renderToStaticMarkup(createElement(PhotoDialog, { t: PAGE.tiles[0], me: { signin: "all", user: null },
                                                                 onClose: () => {} }));
  const t = text(html);
  assert.match(t, /Spotlight strength/);
  assert.equal((html.match(/type="range"/g) ?? []).length, 1);
  assert.match(t, /Show where Vision looked/);
  assert.match(html, /data-testid="figure-excluded-v1"/);
  for (const gone of [/Glow/i, /BioCLIP/i, /base model/i, /Side by side/i, /Show as/, /button-compare-models/, /excluded-base/]) {
    assert.doesNotMatch(html, gone);
  }
});

test("drawings are one category: no separate microscopy or specimen drawings", () => {
  assert.equal(categoryLabel("drawing"), "Drawings");
  assert.deepEqual(CATEGORIES.filter((c) => /drawing/i.test(c.code + c.label)), [{ code: "drawing", label: "Drawings" }]);
  const page = { ...PAGE, categories: [...PAGE.categories, { code: "drawing", label: "Drawings", count: 453 }] };
  const t = text(render(page));
  assert.match(t, /Drawings 453/);
  assert.doesNotMatch(t, /Drawings? – (microscopy|specimen)/);
});
