// The Research pages rendered from their data alone (Steve, 2026-10-25: the approved redesign):
// the states a visitor can meet, missing data included. Each page's view takes its feeds as
// props, so no server is needed.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createElement, type ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Router } from "wouter";
import { ApiError, type EarlyAnswersGroup, type HowGood, type ProspectiveModel, type RunReport, type ScoreRun,
         type Stats } from "../lib/api";
import type { Outline } from "../lib/apiOutline";
import type { ExperimentSummary } from "../lib/experiments";
import { BENCHMARKS, CITATIONS } from "../lib/research";
import { NO_FILTER } from "../lib/researchView";
import { CitationCard, InPageNav, ResearchBar, StatusPill } from "../components/ResearchKit";
import { ResearchLanding } from "./ResearchLanding";
import { Results, RESULTS_SECTIONS } from "./Results";
import { BenchmarkDetail, Benchmarks, DETAIL_SECTIONS } from "./Benchmarks";
import { Protocols } from "./Protocols";
import { Data } from "./Data";
import { Experiments } from "./Experiments";
import { ForResearchers } from "./ForResearchers";
import { ApiDocs } from "./ApiDocs";

const render = (el: ReactElement, path = "/research") => renderToStaticMarkup(
  createElement(QueryClientProvider, { client: new QueryClient() }, createElement(Router, { ssrPath: path, children: el })));
const text = (html: string) => html.replace(/<[^>]+>/g, " ").replace(/&#x27;/g, "'").replace(/&amp;/g, "&")
  .replace(/&quot;/g, '"').replace(/\s+/g, " ");
const count = (html: string, re: RegExp) => (html.match(re) ?? []).length;

const V1 = "bioclip-2-336-ft-20261018-171420";
const M = "nearest+mean-records-balanced+place";
const std = (top1: number, n = 12798) => ({ n, top1, top3: top1 + 0.15, top5: top1 + 0.2, top10: top1 + 0.25 });
const HOW_GOOD: HowGood = {
  run: { id: 169, backbone: V1, method: M, cutoff: "2026-08-17", test_days: 0, n_test: 12874, n_reference: 155648 },
  top1: { species: 0.5917, genus: 0.8575, family: 0.9059 },
  top_k: { k: [1, 3, 5, 10], species: std(0.5917), genus: std(0.7575, 12865), family: std(0.7059, 12759) },
  species_by_depth: [{ band: "0", ...std(0, 649) }, { band: "1-4", ...std(0.1991, 1391) }, { band: "5-19", ...std(0.5024, 2864) },
                     { band: "20-99", ...std(0.7301, 5736) }, { band: "100+", ...std(0.7734, 2158) }],
  first_photo: { n: 12798, first_photo_top1: 0.5309, all_photos_n: 12798, all_photos_top1: 0.5917 },
  inat: null,
};
const run = (id: number, method: string, backbone = V1, extra: Partial<ScoreRun> = {}): ScoreRun => ({
  id, comparison_id: "c", backbone, method, cutoff: "2026-08-17", test_days: 0, n_reference: 155648, n_test: 12874,
  record_set: "36fd876aa871", species_top1: 0.5, genus_top1: 0.85, family_top1: 0.9, species_top1_first_photo: 0.5,
  code_version: null, created_at: "2026-10-19T21:57:19+00:00", dataset_release: "v1", ...extra });
const RUNS = [run(169, M), run(168, "nearest+mean-records-balanced")];
const REPORT = {
  backbone: V1, method: M, first_photo_only: { standard: { species: { all: std(0.5309) } } },
  all_photos: { answer_key: "species-rank-2026-10-18", standard: { species: { all: std(0.5917), "0": std(0, 649), "1-4": std(0.199, 1391),
    "5-19": std(0.502, 2864), "20-99": std(0.73, 5736), "100+": std(0.773, 2158) }, genus: { all: std(0.8575, 12865) },
    family: { all: std(0.9059, 12759) } } },
  reference_species: { answer_key: "species-rank-2026-10-18", records: 155648, species: 16809,
                       bands: { "1-4": 11131, "5-19": 3710, "20-99": 1781, "100+": 187 } },
  ece: { species: { ece: 0.043, n: 12798 } }, errata: { file: "v1-errata-r4.tsv" },
} as unknown as RunReport;
const PHOTOS_ONLY = { ...REPORT, all_photos: { ...(REPORT.all_photos as object),
  standard: { ...((REPORT.all_photos as unknown as { standard: object }).standard), species: { all: std(0.5632) } } } } as unknown as RunReport;
const REPORTS = new Map<number, RunReport>([[169, REPORT], [168, PHOTOS_ONLY]]);
const PROSPECTIVE: ProspectiveModel[] = [{ backbone: "bioclip-2", method: "species-mean", predicted: 300, resolved: 20,
  family_top1: 0.55, genus_top1: 0.6, species_top1: 0.25, mean_species_confidence: 0.2 }];
const EARLY: EarlyAnswersGroup = {
  backbone: V1, method: M, context: "photos + place/date", answers: 574, scored: 0, awaiting_verdict: 531,
  asked_by: { "mycomap.org": 251, site: 323 }, left_out: { answered_after_verdict: 36 },
  answered: { from: null, through: null }, validated: { from: null, through: null }, top_k: {},
  species_by_true_species_reference_records: {}, inat_cv: { status: "not run" } };
const STATS = {
  records: 182926, records_north_america: 177390, label_conflicts: 0, inat_ok: 162358, inat_missing: 74, photos: 700744,
  photos_by_license: { arr: 108379, nc: 455517, open: 136848 }, embedded: { [V1]: 687249 }, names: 19483, names_provisional: 11044,
  names_by_records: { "1": 8146, "2": 2601, "3-5": 3207, "6-30": 4111, "31+": 1418 }, projects: 836, photographers: 2677,
  recent_week: { records: 365, from: "2026-10-19", through: "2026-10-25" },
  permissions: { last_good_sync: null, last_attempt: null, fresh_for_showing: true },
} as Stats;
const visitor = { signin: "identify" as const, user: null };
// The sign-in link says where to come back to (api.signInUrl reads the page's address).
(globalThis as { location?: unknown }).location = { pathname: "/research/experiments", search: "" };

test("status pills: a dot and a word; Sealed and members-only carry a lock instead", () => {
  const running = render(createElement(StatusPill, { status: "running" }));
  assert.match(running, /class="pill running"/);
  assert.match(running, /<i><\/i>Running/);
  const sealed = render(createElement(StatusPill, { status: "sealed" }));
  assert.match(sealed, /class="pill sealed"/);
  assert.match(sealed, /<svg[^>]*lucide-lock/);
  assert.match(text(sealed), /Sealed/);
  assert.doesNotMatch(sealed, /<i><\/i>/);
  assert.match(text(render(createElement(StatusPill, { status: "history" }))), /Development history/);
  assert.match(text(render(createElement(StatusPill, { status: "adopted" }))), /Adopted/);
});

test("the sticky in-page nav numbers each section and links to it, the first marked to start", () => {
  const html = render(createElement(InPageNav, { sections: RESULTS_SECTIONS }));
  assert.match(html, /class="pnav"/);
  assert.deepEqual([...html.matchAll(/href="#([a-z]+)"/g)].map((m) => m[1]),
                   ["headline", "two", "approaches", "depth", "scoreboard", "advance", "early", "models"]);
  assert.match(html, /href="#headline" class="on" aria-current="true"><i>1<\/i>Headline/);
  assert.equal(count(html, /class="on"/), 1);
  assert.deepEqual(DETAIL_SECTIONS.map((s) => s.label), ["Question", "Protocol", "Dataset", "Results", "History", "Links"]);
});

test("the Research bar shows Experiments, locked, to a visitor, and marks the page being read", () => {
  const html = render(createElement(ResearchBar, { me: visitor }), "/research/benchmarks/rare-species");
  assert.match(text(html), /Overview Results Benchmarks Evaluation protocols Reference data Experiments For researchers API/);
  assert.match(html, /href="\/research\/experiments" class="">Experiments<span class="lk" aria-label="members only">/);
  assert.match(html, /aria-current="page" href="\/research\/benchmarks" class="on"/);
});

test("the landing: live headline tiles, every finding card, the section map and the credit line", () => {
  const html = render(createElement(ResearchLanding, { me: visitor, howGood: HOW_GOOD, stats: STATS, runs: RUNS,
    reports: REPORTS, early: [EARLY] }));
  const t = text(html);
  assert.match(t, /59\.2% of new finds get the right species as the first answer/);
  assert.match(t, /177,390 DNA-verified North American records/);
  assert.match(t, /\+365 in the week to 25 Oct 2026/);
  for (const id of ["card-served", "card-two-models", "card-depth", "card-place", "card-early"]) assert.match(html, new RegExp(`data-testid="${id}"`), id);
  assert.match(t, /Place and date add 2\.85 points/, "computed from runs 169 and 168, not typed in");
  assert.match(t, /v1-open names 55\.3%/, "the open-reference number, labelled");
  assert.match(t, /66% of reference species have 1-4/);
  assert.match(html, /data-testid="map-experiments"/, "Experiments is on the map for a visitor too");
  assert.match(t, /For mycomap\.org members: sign in to read/);
  assert.match(t, /MycoMap Vision is BioCLIP 2 \(Imageomics\) fine-tuned on DNA-verified records\./);
  assert.doesNotMatch(t, /v1-test result|Zenodo|Hugging ?Face/i);
});

test("the landing with nothing read yet: no made-up numbers, the cards from committed sources stay", () => {
  const html = render(createElement(ResearchLanding, { me: visitor, reports: new Map() }));
  const t = text(html);
  assert.match(html, /data-testid="landing-no-results"/);
  assert.match(t, /could not be read just now/);
  assert.doesNotMatch(html, /data-testid="card-served"|data-testid="card-depth"|data-testid="card-early"/);
  assert.match(html, /data-testid="card-two-models"/, "twoModels.ts is a committed source");
  assert.match(t, /scored runs|Every scored run/);
});

test("Results: eight sections in the nav's order, the two models with both v1-open numbers labelled", () => {
  const html = render(createElement(Results, { me: visitor, howGood: HOW_GOOD, runs: RUNS, reports: REPORTS,
    prospective: { models: PROSPECTIVE, sealed_after: "2026-10-12T13:17:51+00:00" }, early: { groups: [EARLY], sealed_after: "2026-10-12" } }),
    "/research/results");
  const order = RESULTS_SECTIONS.map((s) => html.indexOf(`id="${s.id}"`));
  assert.ok(order.every((i) => i > 0), JSON.stringify(order));
  assert.deepEqual([...order].sort((a, b) => a - b), order);
  const t = text(html);
  assert.match(t, /Species against the openly licensed reference only \(the paper's headline setup\) 55\.3/);
  assert.match(t, /Species against the full reference 55\.9/);
  assert.match(t, /Using every photo of a find helps\. The first photo alone gets the species first 53\.1%/);
  assert.match(t, /honest to within 4\.3 points/);
  assert.match(t, /iNaturalist's computer vision on these records: not scored yet/);
  assert.match(t, /574 answers kept/);
  assert.match(html, /data-testid="permission-control"/, "an approved photographer line");
  assert.match(t, /Photographers stay in control of how their photos are used in MycoMap Vision\./);
});

test("Results with no feeds: every section is there and says it is reading or empty, never a number from nowhere", () => {
  const html = render(createElement(Results, { reports: new Map(), loading: false }), "/research/results");
  for (const id of ["results-no-headline", "approaches-empty", "depth-empty", "scoreboard-empty", "advance-empty", "early-empty"]) {
    assert.match(html, new RegExp(`data-testid="${id}"`), id);
  }
  assert.doesNotMatch(text(html), /59\.2|12,798 finds with/);
});

test("Benchmarks: the summary, the screen with FungiTastic BEiT-B (frozen), the fungi cards and every card", () => {
  const html = render(createElement(Benchmarks, { me: visitor, howGood: HOW_GOOD, runs: RUNS, prospective: PROSPECTIVE,
    early: [EARLY] }), "/research/benchmarks");
  const t = text(html);
  assert.match(t, /9 benchmarks/);
  assert.match(t, /Newest weeks · species top 1 59\.2%/);
  assert.match(t, /Advance predictions · resolved 20 of 300/);
  assert.match(html, /data-testid="label-dev">Development screen/);
  assert.match(t, /Which image model to build on\? BioCLIP 2, by 2\.1 points/);
  assert.match(t, /FungiTastic BEiT-B \(frozen\)/);
  assert.match(t, /ef3a8df/);
  for (const k of ["d", "c", "b1", "b2", "a", "fewshot"]) assert.match(html, new RegExp(`data-testid="fungi-card-${k}"`), k);
  assert.equal(count(html, /data-testid="label-interim"/g), 1, "only their recipe on our photos (E4) is interim");
  assert.match(t, /75\.28%/);
  assert.match(t, /6090354/);
  assert.match(t, /5792c17/);
  assert.doesNotMatch(t, /8147947/, "the cards cite the entry as it stands now");
  assert.equal(count(html, /data-testid="benchmark-card-/g), BENCHMARKS.length);
  assert.match(t, /Showing all 9 benchmarks/);
  assert.match(html, /data-testid="pill-sealed"/);
  for (const chip of ["chip-status-sealed", "chip-measures-danger", "chip-data-prev1"]) assert.match(html, new RegExp(chip), chip);
  assert.match(t, /Dangerous mistakes 1/);
});

test("Benchmarks, fungi cards: the head to head, the published models on our finds, cross-continent and few-shot, as the page shows them", () => {
  const html = render(createElement(Benchmarks, { me: visitor }), "/research/benchmarks");
  const part = (key: string) => {
    const at = html.indexOf(`data-testid="fungi-card-${key}"`);
    assert.ok(at >= 0, key);
    const next = html.indexOf('data-testid="fungi-card-', at + 10);
    return text(html.slice(at + 10, next < 0 ? undefined : next));
  };
  // (b2): the headline, both tables, the paired intervals with true minus signs, photo only said plainly.
  const b2 = part("b2");
  assert.match(b2, /77\.46% MycoMap Vision v1-open \(their training photos as reference\), top 1 per image on ClosedSet-Test · FungiTastic BEiT-B\/384: 75\.28% · per observation 88\.31% against 87\.42%/);
  assert.match(b2, /Both sides are photo only/);
  assert.match(b2, /ClosedSet-Test · 91,756 images, 48,361 observations/);
  assert.match(b2, /DNA-Test · 5,105 images, 2,041 observations/);
  assert.match(b2, /MycoMap Vision v1-open their training photos as reference 77\.46 89\.09 91\.81 94\.25 88\.31 96\.76 98\.21 99\.13/);
  assert.match(b2, /BioCLIP 2 \(base model, not fine-tuned\) the same reference 76\.36 88\.26 91\.04 93\.60 87\.61/);
  assert.match(b2, /FungiTastic Swin-B\/384 their classifier · all 91,832 images 74\.24 86\.33 89\.33 92\.02 – – – –/);
  assert.match(b2, /\+2\.18 \[1\.92, 2\.47\] \+0\.90 \[0\.63, 1\.17\] \+3\.13 \[1\.63, 4\.55\] −0\.69 \[−2\.79, 1\.32\]/);
  assert.match(b2, /What the fine-tune adds: MycoMap Vision v1-open − BioCLIP 2 \(base model, not fine-tuned\) \+1\.11 \[0\.93, 1\.28\] \+0\.71 \[0\.51, 0\.90\] \+4\.19 \[3\.22, 5\.20\] \+5\.05 \[3\.58, 6\.61\]/);
  // (a): the published models on 12,798 v1 val finds, with intervals under each top 1.
  const a = part("a");
  assert.match(a, /56\.3% MycoMap Vision v1, species top 1 on 12,798 v1 validation finds · FungiTastic BEiT-B\/384: 10\.7% · on the 2,411 finds of species it has a class for, 67\.6% against 56\.9%/);
  assert.match(a, /FungiTastic BEiT-B\/384 10\.7 9\.4–12\.4 14\.1 15\.2 16\.5 58\.6 56\.9 53\.1–61\.2/);
  assert.match(a, /FungiTastic ConvNeXt-B\/384 9\.6 8\.2–11\.4 13\.0 14\.4 15\.8 53\.9 51\.0 46\.7–56\.2/);
  assert.match(a, /^\S* ?Their published models on our finds v1 validation/);
  // (b1) and few-shot.
  assert.match(part("b1"), /43\.77% species top 1 per observation, on the 29,558 ClosedSet-Test observations of species we have/);
  const f = part("fewshot");
  assert.match(f, /46\.41% MycoMap Vision v1-open \(class centroid\), top 1 per image on FewShot-Test · FungiTastic paper: BioCLIP: 21\.8%/);
  assert.match(f, /FungiTastic paper: BioCLIP their Table 4, best centroid 21\.8 32\.6 – – – – – –/);
});

test("Benchmarks filtered: the count says so, a clear button shows, and an empty result offers to clear", () => {
  const sealed = render(createElement(Benchmarks, { filter: { ...NO_FILTER, status: ["sealed"] } }), "/research/benchmarks");
  assert.match(text(sealed), /Showing 1 of 9 benchmarks/);
  assert.equal(count(sealed, /data-testid="benchmark-card-/g), 1);
  assert.match(sealed, /data-testid="benchmark-card-sealed-test"/);
  assert.match(sealed, /data-testid="button-clear"/);
  assert.match(sealed, /aria-pressed="true"[^>]*data-testid="chip-status-sealed"/);
  const none = render(createElement(Benchmarks, { filter: { ...NO_FILTER, q: "nothing like this" } }), "/research/benchmarks");
  assert.match(none, /data-testid="benchmarks-none"/);
  assert.equal(count(none, /data-testid="benchmark-card-/g), 0);
});

test("Benchmarks before the feeds answer: cards say why they have no number; the sealed test says it is sealed", () => {
  const html = render(createElement(Benchmarks, { loading: false }), "/research/benchmarks");
  const t = text(html);
  assert.equal(count(html, /data-testid="card-figure"/g), 0);
  assert.match(t, /The latest number could not be read just now\./);
  assert.match(t, /Sealed\. The set closes once it holds at least 3,000 records/);
  assert.doesNotMatch(t, /Newest weeks · species top 1/);
});

test("a benchmark's page: question, protocol, dataset, results, history and links, from its entry and the run", () => {
  const b = BENCHMARKS.find((x) => x.slug === "rare-species")!;
  const html = render(createElement(BenchmarkDetail, { b, me: visitor, howGood: HOW_GOOD, runs: RUNS, reports: REPORTS }),
                      "/research/benchmarks/rare-species");
  for (const s of DETAIL_SECTIONS) assert.match(html, new RegExp(`id="${s.id}"`), s.id);
  const t = text(html);
  assert.match(t, /19\.9% species top 1 when the true species has 1-4 records/);
  assert.match(t, /66% of reference species have 1-4 records, but only 11% of finds/);
  assert.match(t, /from the 1-4 band \( ?19\.9% ?\) toward 5-19 \( ?50\.2% ?\)/);
  assert.match(t, /Answer key species-rank-2026-10-18/);
  assert.match(html, /data-testid="pipeline"/);
  assert.match(html, /href="https:\/\/arxiv\.org\/abs\/2408\.13632"/);
});

test("the sealed test's page shows no number at all", () => {
  const b = BENCHMARKS.find((x) => x.slug === "sealed-test")!;
  const html = render(createElement(BenchmarkDetail, { b, howGood: HOW_GOOD, runs: RUNS, reports: REPORTS }),
                      "/research/benchmarks/sealed-test");
  assert.doesNotMatch(html, /data-testid="detail-tiles"/);
  assert.match(html, /data-testid="detail-results-empty"/);
  assert.match(text(html), /No numbers are published before it is opened/);
  assert.doesNotMatch(text(html), /59\.2/);
});

test("Protocols: the erratum in use is r4 (650 records), r3 kept as history; the anatomy names its run", () => {
  const html = render(createElement(Protocols, { me: visitor, howGood: HOW_GOOD, runs: RUNS, reports: REPORTS }), "/research/protocols");
  const t = text(html);
  assert.match(t, /Erratum r4: 650 records left out/);
  assert.match(t, /Cross-posted twins 475/);
  assert.match(t, /Earlier revisions, kept as history: r3 \(175 records\)/);
  assert.doesNotMatch(t, /Open question/);
  assert.match(t, /#169 · 19 Oct 2026/);
  assert.match(t, /Nine parts of the standard summary/);
  assert.match(html, /data-testid="text-model-credit-line"/);
});

test("Reference data: the frozen release with copyable hashes, and the live set; without stats it says so", () => {
  const html = render(createElement(Data, { me: visitor, stats: STATS, howGood: HOW_GOOD }), "/research/data");
  const t = text(html);
  assert.match(t, /Rare is normal: 42% of names have a single record/);
  assert.match(t, /182,236 .*169,135 .*155,781 .*13,354/);
  assert.match(html, /data-testid="button-copy"/);
  assert.match(t, /Photos the identifier compares with \(MycoMap Vision v1\) 687,249/);
  assert.match(t, /erratum in use for scoring: v1-errata-r4\.tsv \(650 records\)/);
  const none = render(createElement(Data, { failed: true }), "/research/data");
  assert.match(none, /data-testid="data-no-stats"/);
  assert.doesNotMatch(none, /data-testid="section-live"/);
});

const ENTRIES: ExperimentSummary[] = [
  { title: "Backbone screen on v1", slug: "v1-backbone-screen", date: "2026-10-13", status: "done", reproducibility: "reproduced-on-v1",
    question: "q", headline: "On the 664 deciding queries, species top-1: BioCLIP 2 frozen 70.8%", verdict: "v", decision: "Supplement only",
    benchmark: "v1 seeded subset", split: "dev" },
  { title: "Range and season prior", slug: "range-prior", date: "2026-09-28", status: "adopted", reproducibility: "exploratory-pre-freeze",
    question: "q", headline: "Held-out test: species 48.3 -> 51.2%", verdict: "v", decision: "pending", benchmark: "heldout-2026-10-08", split: "dev" },
];

test("Experiments signed out: the header and a sign-in button, never an entry or a count", () => {
  const html = render(createElement(Experiments, { me: visitor, error: new ApiError(401, "sign in") }), "/research/experiments");
  assert.match(html, /data-testid="experiments-signed-out"/);
  assert.match(text(html), /Experiments are for mycomap\.org members\. Sign in to read them\./);
  assert.doesNotMatch(html, /data-testid="experiment-list"|data-testid="experiments-tiles"/);
});

test("Experiments for a member: counts, the strip, the filter bar and each entry with its excerpt", () => {
  const html = render(createElement(Experiments, { list: ENTRIES }), "/research/experiments");
  const t = text(html);
  assert.match(t, /2 entries in the registry/);
  assert.match(t, /1 rebuilt on dataset release v1/);
  assert.match(html, /data-testid="entries-by-day"/);
  assert.match(t, /Showing all 2 entries, newest first/);
  assert.ok(html.indexOf("experiment-v1-backbone-screen") < html.indexOf("experiment-range-prior"), "newest first");
  assert.match(t, /decision pending/);
  const filtered = render(createElement(Experiments, { list: ENTRIES, initial: { q: "", dataset: "pre", status: [], sets: [] } }));
  assert.match(text(filtered), /Showing 1 of 2 entries/);
  assert.doesNotMatch(filtered, /experiment-v1-backbone-screen/);
});

test("citation cards: text first, BibTeX on request, each with its copy button", () => {
  const html = render(createElement(ForResearchers, { me: visitor, howGood: HOW_GOOD, runs: RUNS, reports: REPORTS }),
                      "/research/for-researchers");
  for (const c of CITATIONS) assert.match(html, new RegExp(`data-testid="cite-${c.key}"`), c.key);
  assert.equal(count(html, /<pre/g), 0, "text shown first");
  assert.match(text(html), /MycoMap Vision v1-open open data only/);
  assert.match(html, /data-testid="number-to-beat"/);
  assert.match(text(html), /\/api\/scoreboard\/168/, "the number to beat is the photos-only run, read from the scoreboard");
  const bib = render(createElement(CitationCard, { c: CITATIONS[1], initial: "bibtex" }));
  assert.match(bib, /data-testid="bibtex-gu"/);
  assert.match(bib, /@article\{gu2025bioclip2,/);
  assert.match(bib, /aria-pressed="true">BibTeX/);
});

const OUTLINE: Outline = {
  title: "MycoMap Vision API", version: "v1", base_url: "https://vision.mycomap.org/api/v1", page_url: "", summary: "Identify fungi.",
  sections: [
    { id: "overview", title: "Overview", blocks: [{ type: "p", text: "Base URL: `x`." }, { type: "list", items: ["Never eat a mushroom on this."] }] },
    { id: "identify", title: "Identify", blocks: [
      { type: "endpoint", method: "POST", path: "/identify", status: "available", summary: "Identify one find.", auth: "Vision token",
        fields: [["photos", "file", "yes", "One find."]], request: "curl -X POST", response: "{}" },
      { type: "endpoint", method: "GET", path: "/later", status: "planned", summary: "Coming." }] },
    { id: "answer", title: "The answer", blocks: [{ type: "table", columns: ["Field", "What it is"], rows: [["`answer`", "The answer."]] }] },
    { id: "errors", title: "Errors", blocks: [{ type: "table", columns: ["Status", "code", "When"], rows: [["400", "`bad_request`", "Wrong."], ["503", "`busy`", "Busy."]] }] },
    { id: "limits", title: "Limits", blocks: [{ type: "list", items: ["20 photos a minute."] }] },
    { id: "rules", title: "Using the results", blocks: [{ type: "list", items: ["Call it MycoMap Vision."] }] },
  ],
};

test("the API page: endpoint count, quickstart, endpoint index, fields beside the code panel, answer and errors", () => {
  const html = render(createElement(ApiDocs, { me: visitor, outline: OUTLINE }), "/research/api");
  const t = text(html);
  assert.match(t, /v1 · 1 of 2 endpoints available/);
  assert.match(html, /data-testid="api-quickstart"/);
  assert.match(t, /Never eat a mushroom on this\./, "the overview's warning is the quickstart's callout");
  assert.match(html, /data-testid="api-endpoint-index"/);
  assert.match(html, /data-testid="endpoint-POST-\/identify"/);
  assert.match(html, /data-testid="code-panel"/);
  assert.match(html, /data-testid="field-list"/);
  assert.match(html, /class="st e4">400/);
  assert.match(html, /class="st e5">503/);
  assert.match(html, /data-testid="api-cards"/);
  assert.match(t, /Using it well/);
  assert.deepEqual([...html.matchAll(/<nav class="side"[^]*?<\/nav>/g)][0][0].match(/href="#([a-z]+)"/g)!.map((h) => h.slice(7, -1)),
                   ["quickstart", "overview", "endpoints", "identify", "answer", "errors", "limits"]);
  const failed = render(createElement(ApiDocs, { error: new Error("The API outline didn't load (502).") }), "/research/api");
  assert.match(failed, /data-testid="api-error"/);
});
