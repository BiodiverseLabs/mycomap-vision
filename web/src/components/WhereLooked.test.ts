// "Where Vision looked" on an answer: the button only where the server has it on, and the
// panel from kept maps (rendered from their data alone).
import { test } from "node:test";
import assert from "node:assert/strict";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { LookedPanel, WhereLooked } from "./WhereLooked";
import type { IdentifyResult, LookedPhoto, Me } from "../lib/api";

const text = (h: string) => h.replace(/<[^>]+>/g, " ").replace(/&#x27;/g, "'").replace(/&amp;/g, "&")
  .replace(/&quot;/g, '"').replace(/\s+/g, " ");

const R = {
  per_photo: [0, 1].map((photo) => ({
    photo, overall_top: { species: { name: "Laetiporus sulphureus", position: 1, confidence: photo ? 0.93 : 0.77 } },
  })),
} as unknown as IdentifyResult;
const URLS = ["https://img/1.jpg", "https://img/2.jpg"];
const map = Array.from({ length: 16 }, (_, i) => i * 17);
const photo = (n: number, match: LookedPhoto["match"]): LookedPhoto =>
  ({ photo: n, crop: { x: 0, y: 0.125, w: 1, h: 0.75 }, grid: 4, map, match });
const MATCH: LookedPhoto["match"] = {
  pairs: [[0.375, 0.375, 0.625, 0.625]], crop: { x: 0, y: 0, w: 1, h: 1 },
  specimen: { observation_id: "305015970", name: "Laetiporus sulphureus", similarity: 0.858,
              photo_url: "https://img/s.jpg", photo_owner: "doubleblade", source: "inat",
              record_url: "https://www.inaturalist.org/observations/305015970" },
};
const READY = { status: "ready" as const, name: "Laetiporus sulphureus", made_at: "2026-10-26T15:00:00Z",
                photos: [photo(0, null), photo(1, MATCH)] };

function withMe(me: Me, el: ReturnType<typeof createElement>) {
  const client = new QueryClient();
  client.setQueryData(["me"], me);
  return renderToStaticMarkup(createElement(QueryClientProvider, { client }, el));
}

test("the button shows only where the server has Where Vision looked on", () => {
  const on = withMe({ signin: "all", user: null, features: { where_looked: true } },
                    createElement(WhereLooked, { runId: "AAAAAAAAAAAAAAAA", r: R, urls: URLS }));
  assert.match(text(on), /Show where Vision looked takes a few seconds/);
  const off = withMe({ signin: "all", user: null, features: { where_looked: false } },
                     createElement(WhereLooked, { runId: "AAAAAAAAAAAAAAAA", r: R, urls: URLS }));
  assert.equal(off, "");
  const older = withMe({ signin: "all", user: null },
                       createElement(WhereLooked, { runId: "AAAAAAAAAAAAAAAA", r: R, urls: URLS }));
  assert.equal(older, "", "a server without the feature says nothing about it");
});

test("the panel says what the map means and doesn't, and opens on the strongest photo", () => {
  const t = text(renderToStaticMarkup(createElement(LookedPanel, { view: READY, r: R, urls: URLS })));
  assert.match(t, /The bright parts of each photo are where the evidence for Laetiporus sulphureus was/);
  assert.match(t, /It shows where , not why/);
  assert.match(t, /On its own, photo 2 puts Laetiporus sulphureus first, at 93%/);
  assert.match(t, /Spotlight strength/);
  assert.match(t, /Clear: this photo's strongest evidence for the name/);
  assert.match(t, /Vision doesn't see this strip/);
  assert.match(t, /What this map does and doesn't mean/);
  assert.match(t, /hiding the brightest tenth of a photo cost Vision about six times as much score/);
  assert.match(t, /It doesn't show why/);
});

test("the panel has one look, Spotlight with its strength: no Glow, no style to choose", () => {
  const h = renderToStaticMarkup(createElement(LookedPanel, { view: READY, r: R, urls: URLS }));
  assert.doesNotMatch(h, /Glow/i);
  assert.doesNotMatch(text(h), /Show as/);
  assert.equal((h.match(/type="range"/g) ?? []).length, 1);
  assert.match(h, /aria-label="Strength of the map"/);
});

test("the match shows the specimen the link shows, credited, as similar patterns not proof", () => {
  const h = renderToStaticMarkup(createElement(LookedPanel, { view: READY, r: R, urls: URLS }));
  const t = text(h);
  assert.match(t, /What it matched Photo 2 beside the DNA-verified specimen it was closest to: Laetiporus sulphureus , 86% alike/);
  assert.match(t, /iNat 305015970 · © doubleblade/);
  assert.match(t, /Similar patterns, not proof/);
  assert.match(h, /src="https:\/\/img\/s.jpg"/);
  assert.match(h, /href="https:\/\/www.inaturalist.org\/observations\/305015970"/);
});

test("a photo with no specimen match shows its map without a match section", () => {
  const one = { ...READY, photos: [photo(0, null)] };
  const t = text(renderToStaticMarkup(createElement(LookedPanel, { view: one, r: R, urls: URLS })));
  assert.doesNotMatch(t, /What it matched/);
  assert.match(t, /On its own, photo 1 puts/);
});
