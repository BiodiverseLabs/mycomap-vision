// How it works, rendered from its data alone (HowItWorks): the states a visitor can meet.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { HowItWorks, type HowItWorksData } from "./About";
import { PERMISSION_LINE, PHOTO_PERMISSIONS_URL } from "../lib/publicView";
import type { HowGood, ModelsInfo, SpeedDay, Stats } from "../lib/api";

const TODAY = "2026-10-22";

function days(n: number, last = TODAY): SpeedDay[] {
  const end = Date.parse(`${last}T00:00:00Z`);
  return Array.from({ length: n }, (_, i) => ({
    day: new Date(end - (n - 1 - i) * 86_400_000).toISOString().slice(0, 10),
    seconds_per_photo: i === n - 1 ? 2.28 : Number((1.2 + (i % 5) * 0.15).toFixed(2)),
  }));
}

const STATS = {
  records: 180000, records_north_america: 176655, label_conflicts: 3, inat_ok: 1, inat_missing: 0,
  photos: 700000, photos_by_license: { cc: 1 }, contributors_arr: 777, embedded: {},
  permissions: { last_good_sync: null, last_attempt: null, fresh_for_showing: true },
  names: 19406, names_provisional: 10992,
  names_by_records: { "1": 8091, "2": 2589, "3-5": 3200, "6-30": 4115, "31+": 1411 },
  projects: 834, photographers: 2675,
  recent_week: { records: 2946, from: "2026-10-06", through: "2026-10-12" },
} as Stats;

const BACKBONE = "bioclip-2-336-ft-20261018-171420";
const MODELS = {
  ready: [BACKBONE], methods: [], max_models: 1, default_method: "nearest+mean-records-balanced+place",
  backbones: [{ backbone: BACKBONE, spec: "", note: "", embedded_photos: 684135, photos_per_second: null,
                trained_through: "2026-08-17" }],
} as unknown as ModelsInfo;

const sc = (n: number, top1: number, top3: number, top5: number, top10: number) => ({ n, top1, top3, top5, top10 });
const HOW_GOOD: HowGood = {
  run: { id: 169, backbone: BACKBONE, method: "nearest+mean-records-balanced+place", cutoff: "2026-08-17",
         test_days: 0, n_test: 12874, n_reference: 155648 },
  top1: { species: 0.5917, genus: 0.8575, family: 0.9059 },
  top_k: { k: [1, 3, 5, 10], species: sc(12798, 0.5917, 0.7531, 0.8015, 0.8512),
           genus: sc(12865, 0.8575, 0.9389, 0.9569, 0.9733), family: sc(12759, 0.9059, 0.9628, 0.9755, 0.9853) },
  species_by_depth: [{ band: "0", ...sc(649, 0, 0, 0, 0) }, { band: "1-4", ...sc(1391, 0.199, 0.388, 0.463, 0.595) },
                     { band: "5-19", ...sc(2864, 0.502, 0.717, 0.799, 0.884) },
                     { band: "20-99", ...sc(5736, 0.73, 0.884, 0.921, 0.95) },
                     { band: "100+", ...sc(2158, 0.773, 0.912, 0.946, 0.967) }],
  first_photo: { n: 12798, first_photo_top1: 0.5309, all_photos_n: 12798, all_photos_top1: 0.5917 },
  inat: null,
};

const page = (over: Partial<HowItWorksData> = {}) => renderToStaticMarkup(createElement(HowItWorks, {
  stats: STATS, models: MODELS, howGood: HOW_GOOD, speed: days(30), today: TODAY, ...over }));
const text = (html: string) => html.replace(/<[^>]+>/g, " ").replace(/&#x27;/g, "'").replace(/&amp;/g, "&")
  .replace(/\s+/g, " ");

test("every section of the approved design is there, in its order", () => {
  const html = page();
  const order = ["hero-tiles", "section-identification", "section-dna", "section-provisional",
                 "section-how-good", "section-first-step", "section-newest-weeks", "section-standards",
                 "section-your-photos", "section-contributors", "section-becomes-reference",
                 "section-openness", "section-nonprofit"];
  const at = order.map((id) => html.indexOf(`data-testid="${id}"`));
  assert.ok(at.every((i) => i >= 0), `missing: ${order.filter((_, i) => at[i] < 0)}`);
  assert.deepEqual([...at].sort((a, b) => a - b), at);
  assert.equal((html.match(/data-testid="standard-photographers"/g) ?? []).length, 1);
  assert.match(text(html), /Eight rules every answer follows/);
});

test("the numbers are the API's, live", () => {
  const t = text(page());
  for (const n of ["176,655", "684,135", "19,406", "+2,946", "12 October 2026", "10,992 provisional (57%)",
                   "8,091", "53.1%", "59.2%", "85.8%", "90.6%", "75.3%", "97.3%", "19.9%", "1,391 tested",
                   "12,874", "155,648", "17 August 2026", "2,675", "834"]) {
    assert.ok(t.includes(n), n);
  }
  assert.match(t, /newest 12,798 test records/);
  assert.match(t, /More than half of the names are provisional species/);
});

test("with no speed recorded the chart says so and shows no figure", () => {
  const html = page({ speed: [] });
  assert.match(html, /data-testid="speed-chart" data-state="none"/);
  assert.match(text(html), /No day recorded yet/);
  assert.doesNotMatch(html, /Recording since|speed-latest|Clean measurement/);
});

test("with one day the day is shown with when recording began, and no line", () => {
  const html = page({ speed: [{ day: TODAY, seconds_per_photo: 1.24 }] });
  assert.match(html, /data-state="few"/);
  const t = text(html);
  assert.match(t, /1\.24 s per photo on 22 October 2026/);
  assert.match(t, /Recording since 22 October 2026\./);
  assert.match(t, /1\.24 s per photo, today/);
  assert.doesNotMatch(html, /speed-table|Clean measurement/, "no line from one day");
  assert.match(text(page({ speed: days(2) })), /The daily chart starts once there are 3 days/);
});

test("with thirty days the line is drawn with the clean measurement and the newest figure", () => {
  const html = page();
  assert.match(html, /data-state="chart"/);
  const t = text(html);
  assert.match(t, /Clean measurement, 1\.1 s/);
  assert.match(t, /2\.28 s today/);
  assert.match(t, /Recording since 23 September 2026\./);
  assert.match(t, /23 Sep/);
  assert.equal((html.match(/data-testid="speed-table"[^>]*>(.*?)<\/ul>/)?.[1].match(/<li>/g) ?? []).length, 30);
});

test("without iNaturalist on the same records nothing is said about it; with it, it is", () => {
  assert.doesNotMatch(page(), /iNaturalist&#x27;s computer vision|text-inat/);
  const html = page({ howGood: { ...HOW_GOOD, inat: { method: "combined-max", n_test: 12874,
    species_top1: 0.31, genus_top1: 0.7, family_top1: 0.8 } } });
  assert.match(text(html), /iNaturalist's computer vision, given the same photos and locations, was right first time for 31\.0% at species/);
});

test("photographers read only the two approved lines, each with its link, and no count", () => {
  const html = page();
  const t = text(html);
  assert.ok(t.includes(PERMISSION_LINE.control));
  assert.ok(t.includes(PERMISSION_LINE.learn));
  assert.equal((html.match(new RegExp(`href="${PHOTO_PERMISSIONS_URL.replace(/[?]/g, "\\?")}"`, "g")) ?? []).length, 3,
               "the standards card, the contributors box and the loop's box");
  assert.doesNotMatch(t, /with permission|licensed|licence|granted|declined/i);
  assert.ok(!t.includes("777"), "never how many photographers keep all rights");
});

test("BioCLIP 2 is named only in the small model credit", () => {
  const html = page();
  const t = text(html);
  assert.match(t, /Model: BioCLIP 2, 336 px, fine-tuned \(public \+ non-public training data\), closest specimens/);
  assert.match(t, /Built on BioCLIP 2 by Imageomics/);
  assert.equal((t.match(/BioCLIP/g) ?? []).length, 3, "the credit line, the card's title and its credit");
  assert.ok(t.includes("MycoMap Vision"));
});

test("step 3's answer is labelled an example; nothing else is made up while results are missing", () => {
  assert.match(text(page()), /Example answer, for illustration\./);
  const html = page({ howGood: { run: null }, stats: undefined, models: undefined, speed: undefined });
  const t = text(html);
  assert.match(t, /Results for this model are not published yet\./);
  assert.doesNotMatch(html, /chart-top-k|chart-by-depth|chart-first-photo|chart-names-by-records|card-model-in-use|text-model-credit/);
  assert.doesNotMatch(t, /176,655|\+2,946/);
  const loading = text(page({ howGood: undefined, loading: true }));
  assert.match(loading, /Loading the latest results/);
});

test("a run scored before the standard summary shows its top-1 tiles and no ladder", () => {
  const html = page({ howGood: { ...HOW_GOOD, top_k: null, species_by_depth: null } });
  assert.doesNotMatch(html, /chart-top-k|chart-by-depth/);
  assert.match(text(html), /59\.2%/);
  assert.match(text(html), /Top 3, 5 and 10 are not measured for this run yet/);
});
