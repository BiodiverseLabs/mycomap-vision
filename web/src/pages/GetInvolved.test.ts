// Get involved, rendered from its data alone (GetInvolved): the states a visitor can meet.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { GetInvolved, type GetInvolvedData } from "./GetInvolved";
import { PERMISSION_LINE, PHOTO_PERMISSIONS_URL } from "../lib/publicView";
import type { FeedbackSummary, HowGood, Stats } from "../lib/api";
import type { Outline } from "../lib/apiOutline";

const STATS = {
  records: 182877, records_north_america: 177341, label_conflicts: 0, inat_ok: 1, inat_missing: 0,
  photos: 700460, photos_by_license: { arr: 108312, nc: 455371, open: 136777 }, embedded: {},
  permissions: { last_good_sync: null, last_attempt: null, fresh_for_showing: true },
  names: 19467, names_provisional: 11040,
  names_by_records: { "1": 8131, "2": 2603, "3-5": 3206, "6-30": 4110, "31+": 1417 },
  projects: 836, photographers: 2676,
  recent_week: { records: 557, from: "2026-10-16", through: "2026-10-22" },
} as Stats;

const sc = (n: number, top1: number) => ({ n, top1, top3: null, top5: null, top10: null });
const HOW_GOOD: HowGood = {
  run: { id: 169, backbone: "bioclip-2-336-ft-20261018-171420", method: "nearest+mean-records-balanced+place",
         cutoff: "2026-08-17", test_days: 0, n_test: 12874, n_reference: 155648 },
  top1: { species: 0.5917, genus: 0.8575, family: 0.9059 },
  species_by_depth: [{ band: "0", ...sc(649, 0) }, { band: "1-4", ...sc(1391, 0.199) },
                     { band: "5-19", ...sc(2864, 0.502) }, { band: "20-99", ...sc(5736, 0.73) },
                     { band: "100+", ...sc(2158, 0.773) }],
};

const fq = (type: string, section: string, open: number, title: string, decided = 67590): FeedbackSummary => ({
  type, title, about: "", granularity: "record", model_flagged: false, section, blind: false, open, decided,
});
const FEEDBACK: FeedbackSummary[] = [
  fq("reference_label_doubt", "validation", 22, "Reference records in doubt"),
  fq("label_vs_own_dna", "validation", 0, "The record's own DNA names another taxon"),
  fq("non_specimen_photo", "cleanup", 19383, "Photos that show no specimen"),
  fq("crosspost_twin", "cleanup", 0, "One find posted to both iNaturalist and Mushroom Observer"),
  { ...fq("photo_view", "views", 4984, "Photo views", 1234),
    about: "What each photo shows: cap top, gills, stem and base, growing in place, several specimens… More." },
];

const ep = (method: string, path: string, summary = "x") =>
  ({ type: "endpoint" as const, method, path, status: "available" as const, summary });
const OUTLINE: Outline = {
  title: "MycoMap Vision API", version: "v1", base_url: "https://vision.mycomap.org/api/v1", page_url: "",
  summary: "", sections: [{ id: "identify", title: "Identify", blocks: [
    ep("POST", "/identify", "Identify one find from 1–10 photos."), ep("POST", "/identify/inat/{observation_id}"),
    ep("POST", "/identify/mo/{observation_id}"), ep("GET", "/taxon-counts"), ep("GET", "/models")] }],
};

const page = (over: Partial<GetInvolvedData> = {}) => renderToStaticMarkup(createElement(GetInvolved, {
  stats: STATS, howGood: HOW_GOOD, feedback: FEEDBACK, outline: OUTLINE, ...over }));
const text = (html: string) => html.replace(/<[^>]+>/g, " ").replace(/&#x27;/g, "'").replace(/&amp;/g, "&")
  .replace(/\s+/g, " ");

test("every section of the approved design is there, in its order", () => {
  const html = page();
  const order = ["hero-tiles", "section-ways", "section-photograph", "photo-checklist", "section-sequence",
                 "sequence-lanes", "chart-names-by-records", "chart-by-depth", "one-record-banner",
                 "section-projects", "project-dots", "section-review", "review-queues", "section-build",
                 "section-support"];
  const at = order.map((id) => html.indexOf(`data-testid="${id}"`));
  assert.ok(at.every((i) => i >= 0), `missing: ${order.filter((_, i) => at[i] < 0)}`);
  assert.deepEqual([...at].sort((a, b) => a - b), at);
  for (const id of ["photograph", "sequence", "projects", "review", "build", "support"]) {
    assert.match(html, new RegExp(`href="#${id}"`), `six ways lead to #${id}`);
    assert.match(html, new RegExp(`id="${id}"`));
  }
});

test("the numbers are the API's, live", () => {
  const html = page();
  const t = text(html);
  for (const n of ["177,341", "2,676", "836", "+557", "22 October 2026", "8,131", "2,603", "1,417",
                   "42%", "8,131 of 19,467 names", "20%", "77%", "12,874", "17 August 2026",
                   "22 open to browse", "19,383 open to browse", "4,984 open to browse"]) {
    assert.ok(t.includes(n), n);
  }
  assert.match(t, /5 endpoints available now/, "counted from the outline");
  assert.equal((html.match(/<i class="aspect-square rounded-full/g) ?? []).length, 836, "one dot per project");
  assert.match(t, /One dot per project\./);
  assert.match(t, /POST https:\/\/vision\.mycomap\.org\/api\/v1\/identify/);
});

test("review shows each group's open count and never a decided count or a reviewer", () => {
  const t = text(page());
  assert.match(t, /Validation 2 queues 22 open to browse The record's own DNA names another taxon Reference records in doubt/);
  assert.match(t, /Photo views 1 queue 4,984 open to browse What each photo shows: cap top, gills, stem and base…/);
  for (const decided of ["67,590", "67590", "1,234", "1234"]) assert.ok(!t.includes(decided), decided);
  assert.doesNotMatch(t, /decided by|reviewed by/i);
});

test("while loading, no number is shown and the charts say they are loading", () => {
  const html = page({ stats: undefined, howGood: undefined, feedback: undefined, outline: undefined, loading: true });
  const t = text(html);
  assert.match(t, /Loading the latest numbers…/);
  assert.doesNotMatch(html, /chart-names-by-records|chart-by-depth|one-record-banner|project-dots|review-queues|api-example/);
  assert.doesNotMatch(t, /\d{1,3},\d{3}|\d+%|undefined|NaN/);
});

test("with no data nothing is made up: each part says it isn't available, and the copy stays", () => {
  const html = page({ stats: undefined, howGood: { run: null }, feedback: [], outline: undefined });
  const t = text(html);
  assert.match(t, /Not available right now\./);
  assert.doesNotMatch(html, /chart-names-by-records|chart-by-depth|one-record-banner|project-dots|review-queues|api-example/);
  assert.doesNotMatch(t, /\d{1,3},\d{3}|\d+%|endpoints? available|open to browse|undefined|NaN| – /, "no number, and no '–' standing in for one");
  assert.match(t, /Pick the way that fits you/);
  assert.match(t, /Identify fungi from your own app or site\./);
  assert.match(t, /DNA sequences in 3–4 months/, "the sequencing steps are static copy");
  assert.match(html, /data-testid="hero-tiles"><\/div>/, "no tile without its number");
});

test("photographers read only the approved 'learn' line, with its link, and no count", () => {
  const html = page();
  const t = text(html);
  assert.ok(t.includes(PERMISSION_LINE.learn));
  assert.ok(!t.includes(PERMISSION_LINE.control));
  assert.equal((html.match(new RegExp(`href="${PHOTO_PERMISSIONS_URL.replace(/[?]/g, "\\?")}"`, "g")) ?? []).length, 1);
  assert.doesNotMatch(t, /with permission|licensed|licence|granted|declined/i);
  for (const n of ["108,312", "455,371", "136,777"]) assert.ok(!t.includes(n), n);
});

test("never mycomap.com, and every outside link is a real mycomap.org or iNaturalist page", () => {
  const html = page();
  assert.doesNotMatch(html, /mycomap\.com/i);
  const outside = [...html.matchAll(/href="(https?:[^"]+)"/g)].map((m) => m[1]);
  assert.deepEqual([...new Set(outside)].sort(), [
    "https://mycomap.org/join", "https://mycomap.org/network", "https://mycomap.org/profile?tab=photo-permission",
    "https://mycomap.org/projects", "https://mycomap.org/start-project", "https://mycomap.org/vision-keys",
    "https://www.inaturalist.org",
  ]);
  assert.doesNotMatch(text(html), /Zenodo|Hugging ?Face|v1-test|sealed/i);
  assert.match(text(html), /MycoMap Vision/);
});
