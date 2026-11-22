// How it works, rendered from its data alone (HowItWorks): the states a visitor can meet.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { HowItWorks, type HowItWorksData } from "./About";
import { PERMISSION_LINE, PHOTO_PERMISSIONS_URL } from "../lib/publicView";
import type { HowGood, ModelsInfo, SpeedDay, Stats } from "../lib/api";
import { DAILY_V1_ONLY, DAILY_V2, fortyDays } from "../lib/dailyScores.fixture";

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
  assert.doesNotMatch(html, /Recording since|speed-latest|speed-rolling|Clean measurement/);
});

test("with one day the day is shown with when recording began, and no line", () => {
  const html = page({ speed: [{ day: TODAY, seconds_per_photo: 1.24 }] });
  assert.match(html, /data-state="few"/);
  const t = text(html);
  assert.match(t, /1\.24 s per photo on 22 October 2026/);
  assert.match(t, /Recording since 22 October 2026\./);
  assert.match(t, /1\.24 s per photo, today/);
  assert.doesNotMatch(html, /speed-table|speed-rolling|7-day average:|Clean measurement/, "no line, no average from one day");
  const two = text(page({ speed: [{ day: "2026-10-21", seconds_per_photo: 1.2 }, { day: TODAY, seconds_per_photo: 1.5 }] }));
  assert.match(two, /The daily chart starts once there are 3 days/);
  assert.match(two, /7-day average: 1\.35 s per photo/, "from 2 days, their 7-day average");
});

test("with thirty days the daily line is drawn with its 7-day average and the newest figure", () => {
  const html = page();
  assert.match(html, /data-state="chart"/);
  const t = text(html);
  assert.doesNotMatch(t, /Clean measurement|1\.1 s/, "no fixed reference figure");
  assert.match(html, /data-testid="speed-rolling-line"/);
  assert.match(html, /data-testid="speed-legend"[^]*Daily average[^]*7-day average/);
  assert.match(t, /22 Oct: 2\.28 s \(7-day average 1\.\d\d s\)/, "the table gives each day's average too");
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

test("BioCLIP 2 is named only in the small model credit's credit line", () => {
  const html = page();
  const t = text(html);
  assert.match(t, /Model: MycoMap Vision v1 \(public \+ non-public training data\), closest specimens/);
  assert.match(t, /MycoMap Vision is BioCLIP 2 \(Imageomics\) fine-tuned on DNA-verified records\./);
  assert.equal((t.match(/BioCLIP/g) ?? []).length, 1, "the credit line only");
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

test("How it works: the headline names its test, place and date included; the speed chart says when it is loading", () => {
  assert.match(text(page()), /Tested on the newest 12,874 DNA-verified finds \(validated after 17 August 2026\), identified from the 155,648 records verified before them, with each find's photos and its public place and date\./);
  const loading = text(page({ speed: undefined, loading: true }));
  assert.match(loading, /Loading the daily averages…/);
  assert.doesNotMatch(loading, /No day recorded yet/);
});

test("How it works: range and season and the photo-type tagger are already in, not 'still to come'", () => {
  const t = text(page());
  assert.doesNotMatch(t, /What's still to come/);
  assert.match(t, /Already in, and still to come/);
  assert.match(t, /Range and season Already in: when you give a place and date, Vision weighs each name/);
  assert.match(t, /Already in: micrographs and photos of data slips are set aside\. To come: recognising cap, underside, stem and more\./);
});

// Steve, 2026-11-06: "+154 this week" while he had validated far more. The weekly figure
// counts what the set holds; what was validated after the last nightly read waits for
// tonight, and both weekly counters now say how many.
test("records validated since the last nightly read are counted under both weekly figures", () => {
  const html = page({ stats: { ...STATS, waiting: { records: 1190, since: "2026-11-06T08:00:41.000Z", as_of: null } } });
  assert.match(html, /data-testid="hero-waiting"/);
  assert.match(html, /data-testid="weekly-waiting"/);
  const t = text(html);
  assert.equal(t.match(/\+1,190 more validated since, added in tonight's update/g)?.length, 2);
  assert.ok(t.includes("+2,946"), "the weekly figure itself is unchanged");
});

test("no waiting line when nothing is waiting, the count is missing, or the week has no figure", () => {
  for (const waiting of [undefined, null, { records: 0, since: null, as_of: null }]) {
    const html = page({ stats: { ...STATS, waiting } as Stats });
    assert.doesNotMatch(html, /hero-waiting|weekly-waiting|more validated since/);
  }
  const noWeek = page({ stats: { ...STATS, recent_week: null, waiting: { records: 5, since: null, as_of: null } } });
  assert.doesNotMatch(noWeek, /more validated since/);
});

// --- the v1 / v2 dropdown (Steve, 2026-11-22) -------------------------------------------------

const withoutPicker = (html: string) => html.replace(/<div[^>]*data-testid="model-picker"[^]*?<\/select><\/div>/, "");

test("the How good section has the model dropdown, on v1 while v2 is not available", () => {
  for (const daily of [undefined, DAILY_V1_ONLY]) {
    const html = page({ dailyScores: daily });
    assert.match(html, /data-testid="model-picker"/);
    assert.match(html, /<option value="v1" selected="">MycoMap Vision v1<\/option>/);
    assert.doesNotMatch(html, /value="v2"/, "no v2 to choose until the server offers it");
    assert.ok(html.indexOf('data-testid="model-picker"') > html.indexOf('data-testid="section-how-good"'));
  }
});

test("the dropdown opens on v2 once the server offers it, and shows v2's numbers", () => {
  const html = page({ dailyScores: DAILY_V2 });
  assert.match(html, /<option value="v2" selected="">MycoMap Vision v2<\/option>/);
  assert.match(html, /data-testid="scores-v2"/);
  assert.doesNotMatch(html, /chart-top-k|table-by-depth|text-model-credit/, "v1's run is not shown under v2");
  const t = text(html);
  assert.match(t, /MycoMap Vision v2 learned from every DNA-verified record up to 22 November 2026/);
  assert.ok(t.includes("63.0%"), "species overall");
  assert.match(t, /Likely between 58\.1% and 67\.7%/);
  assert.ok(t.includes("403") && t.includes("1,291"), "records and new photos since training");
  assert.match(t, /Last 7 days \(to 25 November 2026\): species 63\.0% \(n = 400\) ?, genus 86\.6% \(n = 403\)/);
  assert.match(t, /403 verified since, each tested once/);
});

test("every daily value in v2's view is written with its n", () => {
  const html = page({ dailyScores: DAILY_V2 });
  const chart = html.match(/data-testid="chart-daily"[^]*?<\/div><\/div><\/div>/)?.[0] ?? "";
  assert.ok(chart, "the daily chart");
  for (const n of ["n = 152", "n = 185", "n = 63, so far"]) assert.ok(text(html).includes(n), n);
  const table = html.match(/data-testid="table-daily"[^]*?<\/table>/)?.[0] ?? "";
  const cells = [...table.matchAll(/<td[^>]*>([^<]*)<\/td>/g)].map((m) => m[1]);
  const rates = cells.filter((c) => /%/.test(c));
  assert.equal(rates.length, 12, "species, genus, family and 7-day species for each of the three days");
  for (const c of rates) assert.match(c, /^\d+% \(n = [\d,]+\)$/, c);
  assert.doesNotMatch(table, /22 November 2026/, "a day with no records is not a row");
});

test("v1's view is the page as it was, whatever v2 has", () => {
  assert.equal(withoutPicker(page({ dailyScores: DAILY_V2, model: "v1" })), withoutPicker(page()));
  assert.equal(withoutPicker(page({ dailyScores: DAILY_V1_ONLY })), withoutPicker(page()));
  const t = text(page({ dailyScores: DAILY_V2, model: "v1" }));
  for (const n of ["59.2%", "85.8%", "90.6%", "12,874", "155,648", "17 August 2026"]) assert.ok(t.includes(n), n);
});

const part = (html: string, id: string) => {
  const at = html.indexOf(`data-testid="${id}"`);
  return at < 0 ? "" : html.slice(at, html.indexOf("<p ", at));
};
const subs = (html: string) => [...html.matchAll(/<small[^>]*>([^<]*)<\/small>/g)].map((m) => m[1]);

test("v2 is measured and said as v1 is: photos and public place and date, from the records before it", () => {
  const t = text(page({ dailyScores: DAILY_V2 }));
  assert.match(t, /Tested on the 403 DNA-verified finds validated since, none of them used in training, each identified from the records verified before it, with its photos and its public place and date\./);
  const v1 = text(page());
  assert.match(v1, /with each find's photos and its public place and date/, "v1's own words, unchanged");
});

test("v2 shows v1's breakdowns: family beside species and genus, the top-k ladder, and depth with too few said so", () => {
  const html = page({ dailyScores: DAILY_V2 });
  const t = text(html);
  assert.match(html, /data-testid="total-family"/);
  assert.match(t, /92\.8% Family right first time, 403 finds Likely between 90\.0% and 95\.0%/);
  assert.match(t, /The right name, within the first few/);
  const ladder = text(html.match(/data-testid="v2-top-k"[^]*?<\/table>/)?.[0] ?? "");
  for (const v of ["63.0%", "78.3%", "83.0%", "87.8%", "86.6%", "94.5%", "96.3%", "97.5%"]) assert.ok(ladder.includes(v), v);
  assert.match(t, /Every sequenced record makes it better/);
  const depth = text(part(html, "chart-v2-by-depth"));
  assert.match(depth, /0 records n = 21 too few yet/);
  assert.match(depth, /1–4 records n = 64 26\.6%/);
  assert.match(depth, /100\+ records n = 57 82\.5%/);
  assert.doesNotMatch(depth, /0 records n = 21 \d/, "no rate for a band with too few");
});

test("the daily chart draws the last 30 days and the table keeps every day", () => {
  const html = page({ dailyScores: fortyDays() });
  const drawn = subs(part(html, "chart-daily"));
  assert.equal(drawn.length, 30);
  assert.equal(drawn[0], "n = 110", "the first drawn day is the 11th of 40");
  assert.equal(drawn[29], "n = 139, so far");
  assert.match(text(html), /day by day, last 30 days/);
  const table = html.match(/data-testid="table-daily"[^]*?<\/table>/)?.[0] ?? "";
  assert.equal((table.match(/<tr>/g) ?? []).length, 41, "a header and all 40 days");
});
