// The Identify page's empty state and photo guide, rendered from their data alone.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createElement, type FC } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { ExampleResult, GUIDE_ID, PhotoGuide, ViewsChecklist, ViewsRow, WhyVision } from "./IdentifyHome";
import { PHOTO_CHECKLIST } from "../lib/publicView";
import { exampleView } from "../lib/identifyHome";
import { EXAMPLE_RUN } from "../lib/exampleRun.fixture";
import type { HowGood, Stats } from "../lib/api";

const STATS = {
  records: 182926, records_north_america: 177390, label_conflicts: 0, inat_ok: 1, inat_missing: 0,
  photos: 700746, photos_by_license: {}, embedded: { "served-model": 687249 },
  permissions: { last_good_sync: null, last_attempt: null, fresh_for_showing: true },
  names: 19483, names_provisional: 11044, names_by_records: {},
  recent_week: { records: 365, from: "2026-10-19", through: "2026-10-25" },
} as Stats;
const tk = (n: number, top1: number, top3: number, top5: number, top10: number) => ({ n, top1, top3, top5, top10 });
const HOW_GOOD: HowGood = {
  run: { id: 169, backbone: "served-model", method: "m", cutoff: "2026-08-17", test_days: 0, n_test: 12874,
         n_reference: 155648 },
  top_k: { k: [1, 3, 5, 10], species: tk(12798, 0.5917, 0.7528, 0.8015, 0.8513),
           genus: tk(12865, 0.8575, 0.9389, 0.9569, 0.9733), family: tk(12759, 0.9059, 0.9628, 0.9755, 0.9853) },
};

const html = <P extends object>(el: FC<P>, props: P = {} as P) => renderToStaticMarkup(createElement(el, props));
const text = (h: string) => h.replace(/<[^>]+>/g, " ").replace(/&#x27;/g, "'").replace(/&amp;/g, "&")
  .replace(/\s+/g, " ");

test("Why MycoMap Vision shows the live numbers, benchmarks as top-1, 3, 5 and 10", () => {
  const t = text(html(WhyVision, { stats: STATS, howGood: HOW_GOOD }));
  assert.match(t, /177,390 DNA-verified North American records/);
  assert.match(t, /compared with their 687,249 photos/);
  assert.match(t, /365 North American records were DNA-verified in the week to 25 October 2026/);
  assert.match(t, /11,044 of 19,483 names are provisional: known so far only from DNA/);
  assert.match(t, /provisional \(57%\)/);
  assert.match(t, /Right name within the first 1, 3, 5 and 10, on the 12,874 newest records \(validated after 17 August 2026\)/);
  assert.match(t, /Top 1 Top 3 Top 5 Top 10 Species 59\.2% 75\.3% 80\.2% 85\.1% Genus 85\.8% 93\.9% 95\.7% 97\.3% Family 90\.6% 96\.3% 97\.5% 98\.5%/,
               "pct() rounds as on How it works (0.9755 is 97.5%)");
});

test("a card whose numbers the server doesn't send is left out, and with none the section is too", () => {
  assert.equal(html(WhyVision, {}), "", "loading, or the API down: nothing, never a made-up number");
  const statsOnly = html(WhyVision, { stats: STATS, howGood: { run: null } });
  assert.match(statsOnly, /why-references/);
  assert.match(statsOnly, /why-provisional/);
  assert.doesNotMatch(statsOnly, /why-tested/, "no test run, no benchmark card");
  const older = html(WhyVision, { stats: { ...STATS, names_provisional: undefined }, howGood: HOW_GOOD });
  assert.doesNotMatch(older, /why-provisional/, "an older server without provisional names");
});

test("the example result is labelled at the top and links to the real run", () => {
  const t = text(html(ExampleResult, { view: exampleView(EXAMPLE_RUN) }));
  assert.ok(t.trim().startsWith("Example result"), "the label comes first");
  const h = html(ExampleResult, { view: exampleView(EXAMPLE_RUN) });
  assert.match(h, /href="\/r\/AAAAAAAAAAAAAAAA"[^>]*data-testid="link-example-run"/);
  assert.match(t, /Open the full result/);
  assert.match(t, /iNaturalist 91443365 · photos Tom Norton \(CC BY\)/);
  assert.match(t, /Best answer from these 3 photos Laetiporus sulphureus 80%/);
  assert.match(t, /Species · Genus Laetiporus 100% · Family Laetiporaceae 100%/);
});

test("the example shows real photos only where the run's link may show them", () => {
  const h = html(ExampleResult, { view: exampleView(EXAMPLE_RUN) });
  const srcs = [...h.matchAll(/<img[^>]* src="([^"]+)"/g)].map((m) => m[1]);
  assert.deepEqual(srcs, ["https://img/1.jpg", "https://img/3.jpg", "https://img/s1.jpg", "https://img/s3.jpg",
                          "https://img/s4.jpg", "https://img/1.jpg", "https://img/3.jpg"],
                   "the find's shown photos, three specimens with photos, then each photo on its own");
  assert.match(text(h), /© owner1/, "each specimen photo credited to its photographer");
});

test("while the example comes, its place is kept; once it's known there is none, the place goes", () => {
  assert.match(html(ExampleResult, { view: null, loading: true }), /data-testid="example-loading"/);
  assert.doesNotMatch(html(ExampleResult, { view: null, loading: false }), /example-loading/);
  assert.doesNotMatch(html(ExampleResult, { view: exampleView(EXAMPLE_RUN), loading: true }), /example-loading/,
                      "never both the placeholder and the card");
});

test("the example's photos load with the page, not lazily", () => {
  assert.doesNotMatch(html(ExampleResult, { view: exampleView(EXAMPLE_RUN) }), /loading="lazy"/);
});

test("without an example run the card is left out and the list of what you get stays", () => {
  const h = html(ExampleResult, { view: null });
  assert.doesNotMatch(h, /<img/);
  assert.doesNotMatch(h, /link-example-run/);
  const t = text(h);
  for (const part of ["An answer at every rank", "The closest DNA-verified specimens", "How each photo scores on its own",
                      "provisional names"]) {
    assert.ok(t.includes(part), part);
  }
});

test("the photo guide draws the five views in the checklist's order, with its words", () => {
  const h = html(PhotoGuide);
  assert.match(h, new RegExp(`id="${GUIDE_ID}"`));
  const t = text(h);
  let at = 0;
  for (const c of PHOTO_CHECKLIST) {
    const i = t.indexOf(c.what, at);
    assert.ok(i >= at, `${c.what} in order`);
    assert.ok(t.includes(c.why), c.why);
    at = i;
  }
  assert.equal((h.match(/<svg(?![^>]*lucide)/g) ?? []).length, PHOTO_CHECKLIST.length, "one drawing per view");
});

test("the views under the drop zone each lead to the guide", () => {
  const h = html(ViewsRow);
  assert.equal((h.match(new RegExp(`href="#${GUIDE_ID}"`, "g")) ?? []).length, PHOTO_CHECKLIST.length);
  for (const c of PHOTO_CHECKLIST) assert.ok(text(h).includes(c.short), c.short);
});

test("with photos in, the checklist is open on a wide screen and folded to one line on a phone", () => {
  assert.match(html(ViewsChecklist, { wide: true }), /<details[^>]* open=""/);
  const phone = html(ViewsChecklist, { wide: false });
  assert.doesNotMatch(phone, /<details[^>]* open=""/);
  assert.match(text(phone), /What to photograph · 5 views/);
});
