// The Research pages' logic (lib/researchView.ts), apart from React: the one top-k chart's axis,
// which runs each section reads, the benchmark cards' numbers and facets, the experiment
// filters, and the sticky nav. Fixtures are the live feeds' shapes (25 Oct 2026).
import { test } from "node:test";
import assert from "node:assert/strict";
import type { EarlyAnswersGroup, HowGood, ProspectiveModel, RunReport, ScoreRun } from "./api";
import { BENCHMARKS } from "./research";
import type { ExperimentSummary } from "./experiments";
import { activeSection, approachGroups, axisAt, benchmarkFacetCounts, cardFigure, cardMeta, earlyFigures,
         entriesByDay, entryCounts, excerpt, filterBenchmarks, filterEntries, filterFromSearch, filterToSearch,
         fineTuneVsBase, heatRows, heatStep, historyLines, isFiltered, modelParts, NO_EXPERIMENT_FILTER, NO_FILTER,
         pc, placeVsPhotos, prospectiveLead, pt, releaseGroups, scoreLines, scoreboardCounts, shortDay, statusBar,
         testSetsOf, toggle, topkAxis, type TopKRow } from "./researchView";

const V1 = "bioclip-2-336-ft-20261018-171420";
const V0 = "bioclip-2-ft-20261013-003428";
const SERVED_METHOD = "nearest+mean-records-balanced+place";

let nextId = 1;
const run = (o: Partial<ScoreRun>): ScoreRun => ({
  id: nextId++, comparison_id: "c", backbone: V1, method: "nearest", cutoff: "2026-08-17", test_days: 0,
  n_reference: 155648, n_test: 12874, record_set: "36fd876aa871", species_top1: 0.5, genus_top1: 0.8,
  family_top1: 0.9, species_top1_first_photo: 0.45, code_version: null, created_at: "2026-10-19T21:00:00+00:00",
  dataset_release: "v1", ...o });

const R169 = run({ id: 169, method: SERVED_METHOD, species_top1: 0.5917 });
const R168 = run({ id: 168, method: "nearest+mean-records-balanced", species_top1: 0.5632 });
const R160 = run({ id: 160, backbone: V0, method: SERVED_METHOD, species_top1: 0.5737 });
const B = { record_set: "d7bd133c0a25", n_test: 13354 };
const R166 = run({ id: 166, method: "nearest+mean", species_top1: 0.5726, ...B });
const R163 = run({ id: 163, backbone: "bioclip-2-336", method: "nearest+mean", species_top1: 0.5279, ...B });
const R145 = run({ id: 145, backbone: V0, method: "nearest+mean", species_top1: 0.5513, ...B, created_at: "2026-10-13T00:00:00+00:00" });
const OLD = run({ id: 139, backbone: "bioclip-2-ft-20261007-165400", method: "nearest+mean", record_set: "4ef7b0035bc9",
                  n_test: 1152, cutoff: "2026-09-10", test_days: 28, dataset_release: null, species_top1: 0.3836,
                  created_at: "2026-10-09T00:00:00+00:00" });
const INAT = run({ id: 140, backbone: "external:inat-cv", method: "combined-max", record_set: "4ef7b0035bc9",
                   n_test: 1152, cutoff: "2026-09-10", test_days: 28, dataset_release: null, species_top1: 0.2,
                   created_at: "2026-10-09T00:00:00+00:00" });
const RUNS = [R169, R168, R160, R166, R163, R145, OLD, INAT];

const std = (top1: number, n = 12798) => ({ n, top1, top3: top1 + 0.15, top5: top1 + 0.2, top10: top1 + 0.25 });
const report = (top1: number, bands = true): RunReport => ({
  backbone: "", method: "", first_photo_only: {} as RunReport["first_photo_only"],
  all_photos: { species: {}, genus: {}, family: {}, standard: {
    species: { all: std(top1), ...(bands ? { "0": std(0, 649), "1-4": std(top1 - 0.4, 1391), "5-19": std(top1 - 0.1, 2864),
                                            "20-99": std(top1 + 0.13, 5736), "100+": std(top1 + 0.17, 2158) } : {}) },
    genus: { all: std(0.85) }, family: { all: std(0.9) } } } as unknown as RunReport["all_photos"],
});

test("the top-k axis starts ten points under the lowest value and runs to 100%", () => {
  const row = (v: (number | null)[]): TopKRow => ({ key: "r", name: "r", values: { 1: v[0], 3: v[1], 5: v[2], 10: v[3] } });
  assert.deepEqual(topkAxis([row([0.592, 0.753, 0.802, 0.851])]), { lo: 0.5, hi: 1, ticks: [0.5, 0.7, 0.9] });
  assert.deepEqual(topkAxis([row([0.387, 0.527, 0.592, 0.673])]).lo, 0.3);
  assert.equal(topkAxis([row([0.401, 0.6, 0.7, 0.8])]).lo, 0.3, "a value on the edge gets a step of room");
  assert.equal(topkAxis([row([0, 0, 0, 0])]).lo, 0);
  assert.deepEqual(topkAxis([row([null, null, null, null])]), { lo: 0, hi: 1, ticks: [0, 0.2, 0.4, 0.6, 0.8, 1] }, "nothing measured");
  const axis = topkAxis([row([0.5, 0.6, 0.7, 0.8])]);
  assert.equal(axisAt(0.4, axis), 0);
  assert.equal(axisAt(1, axis), 100);
  assert.equal(axisAt(2, axis), 100, "clamped");
});

test("numbers read the same way everywhere: one decimal, a dash when missing, dates in UTC", () => {
  assert.equal(pt(0.5917), "59.2");
  assert.equal(pc(0.5917), "59.2%");
  assert.equal(pc(null), "–");
  assert.equal(pt(Number.NaN), "–");
  assert.equal(shortDay("2026-10-19T23:57:19+00:00"), "19 Oct 2026");
  assert.equal(shortDay("2026-08-17"), "17 Aug 2026");
  assert.equal(shortDay(null), "–");
});

test("the heatmap has six steps from under 15% to 75%+, white ink on the two darkest", () => {
  assert.equal(heatStep(0.016).bg, "#e9f1fc");
  assert.equal(heatStep(0.217).bg, "#cde2fb");
  assert.equal(heatStep(0.466).bg, "#3987e5");
  assert.equal(heatStep(0.707).bg, "#1c5cab");
  assert.equal(heatStep(0.751).bg, "#0d366b");
  assert.equal(heatStep(0.751).ink, "#ffffff");
  assert.equal(heatStep(0.466).ink, "#0b0b0b");
  assert.equal(heatStep(null).bg, "transparent");
});

test("rows name models as live: the version, then its training data and method below", () => {
  assert.deepEqual(modelParts(V1, "nearest+mean"),
    { name: "MycoMap Vision v1", detail: "public + non-public training data · closest specimens + species average" });
  assert.deepEqual(modelParts("bioclip-2-336-ft-20261020-173106"), { name: "MycoMap Vision v1-open", detail: "open data only" });
  assert.deepEqual(modelParts(V0, "ft-head"), { name: "MycoMap Vision v0", detail: "224 px, development · its own classifier" });
  assert.deepEqual(modelParts("bioclip-2-336", "nearest"),
    { name: "BioCLIP 2 (base model, not fine-tuned)", detail: "336 px · closest specimen" });
  assert.deepEqual(modelParts("external:inat-cv", "combined-max"),
    { name: "iNaturalist's computer vision", detail: "outside model · photos + location" });
});

test("the release comparisons come first with the site's model and method, then the rest", () => {
  const groups = releaseGroups(RUNS, [V1], SERVED_METHOD);
  assert.deepEqual(groups.map((g) => g[0].record_set), ["36fd876aa871", "d7bd133c0a25"]);
  assert.ok(!groups.flat().some((r) => r.dataset_release === null), "development history is not a release comparison");
});

test("approach charts: every approach where the site's method was scored; elsewhere the fine-tune against its own base only", () => {
  const charts = approachGroups(RUNS, [V1], SERVED_METHOD);
  assert.equal(charts.length, 2);
  assert.equal(charts[0].baseOnly, false);
  assert.deepEqual(charts[0].runs.map((r) => r.id).sort(), [160, 168, 169]);
  assert.equal(charts[1].baseOnly, true);
  assert.deepEqual(charts[1].runs.map((r) => r.id).sort(), [163, 166], "the v0 run on the same records stays in the scoreboard");
});

test("the landing's two findings pick their runs from the scoreboard, never by number", () => {
  assert.deepEqual(fineTuneVsBase(RUNS, [V1]), { tuned: R166, base: R163 });
  assert.deepEqual(placeVsPhotos(RUNS, [V1], SERVED_METHOD), { place: R169, photos: R168 });
  assert.equal(placeVsPhotos(RUNS, [V1], "nearest"), null, "a served method without place has no pair");
  assert.equal(fineTuneVsBase([R169, R168], [V1]), null, "no base model on the same finds");
});

test("scoreboard lines carry top 3/5/10 from the standard summary, and say '–' when a run predates it", () => {
  const reports = new Map<number, RunReport>([[169, report(0.5917)]]);
  const lines = scoreLines([R168, R169], reports, [V1], SERVED_METHOD);
  assert.equal(lines[0].run.id, 169, "best first");
  assert.ok(lines[0].highlight);
  assert.equal(lines[0].species[10], 0.5917 + 0.25);
  assert.deepEqual(lines[1].species, { 1: 0.5632, 3: null, 5: null, 10: null });
  assert.deepEqual(scoreboardCounts(RUNS), { runs: 8, comparisons: 3, release: 6 });
});

test("development history: one line per earlier test, the best of ours and iNaturalist's with location", () => {
  const [line] = historyLines([[OLD, INAT]]);
  assert.equal(line.best.id, 139);
  assert.equal(line.inat, 0.2);
  assert.equal(line.finds, 1152);
  assert.equal(line.approaches, 2);
});

test("the heatmap's rows come from runs with depth bands, best first, one highlighted", () => {
  const reports = new Map<number, RunReport>([[166, report(0.57)], [163, report(0.53)], [145, report(0.55, false)]]);
  const { rows, finds } = heatRows([R163, R166, R145], reports, [V1]);
  assert.deepEqual(rows.map((r) => r.key), ["166", "163", "145"]);
  assert.equal(rows.filter((r) => r.highlight).length, 1);
  assert.deepEqual(finds, [649, 1391, 2864, 5736, 2158]);
  assert.equal(rows[2].cells.every((c) => c === null), true, "a run without bands keeps empty cells");
});

const EARLY: EarlyAnswersGroup = {
  backbone: V1, method: SERVED_METHOD, context: "photos + place/date", answers: 574, scored: 0, awaiting_verdict: 531,
  asked_by: { "mycomap.org": 251, site: 323 }, left_out: { answered_after_verdict: 36 },
  answered: { from: null, through: null }, validated: { from: null, through: null }, top_k: {},
  species_by_true_species_reference_records: {}, inat_cv: { status: "not run" } };
const PROSPECTIVE: ProspectiveModel[] = [
  { backbone: "bioclip-2", method: "species-mean", predicted: 300, resolved: 20, family_top1: 0.55, genus_top1: 0.6,
    species_top1: 0.25, mean_species_confidence: 0.2 },
  { backbone: V0, method: "nearest", predicted: 378, resolved: 0, family_top1: null, genus_top1: null,
    species_top1: null, mean_species_confidence: null }];

test("live feeds: the answered-with-place group leads; the advance row with most resolved is quoted", () => {
  const e = earlyFigures([{ ...EARLY, context: "photos only", answers: 1501 }, EARLY])!;
  assert.deepEqual([e.answers, e.awaiting, e.afterVerdict, e.scored], [574, 531, 36, 0]);
  assert.deepEqual(e.askedBy, [{ who: "mycomap.org", n: 251 }, { who: "This site", n: 323 }]);
  assert.equal(earlyFigures([]), null);
  const lead = prospectiveLead(PROSPECTIVE)!;
  assert.equal(lead.row.resolved, 20);
  assert.equal(lead.predicted, 678);
  assert.equal(prospectiveLead(undefined), null);
});

const HOW_GOOD: HowGood = {
  run: { id: 169, backbone: V1, method: SERVED_METHOD, cutoff: "2026-08-17", test_days: 0, n_test: 12874, n_reference: 155648 },
  top_k: { k: [1, 3, 5, 10], species: std(0.5917) },
  species_by_depth: [{ band: "0", ...std(0, 649) }, { band: "1-4", ...std(0.199, 1391) }, { band: "100+", ...std(0.773, 2158) }],
};
const bench = (slug: string) => BENCHMARKS.find((b) => b.slug === slug)!;
const FEEDS = { howGood: HOW_GOOD, runs: RUNS, prospective: PROSPECTIVE, early: [EARLY] };

test("benchmark cards read their headline from the feeds", () => {
  const nw = cardFigure(bench("newest-weeks"), FEEDS);
  assert.ok(nw?.kind === "topk" && nw.big === "59.2%" && /MycoMap Vision v1, photos \+ place and date/.test(nw.caption));
  const rare = cardFigure(bench("rare-species"), FEEDS);
  assert.ok(rare?.kind === "depth" && rare.from === "19.9" && rare.to === "77.3%");
  const unknown = cardFigure(bench("unknown-species"), FEEDS);
  assert.ok(unknown?.kind === "count" && unknown.big === "649");
  const adv = cardFigure(bench("advance-predictions"), FEEDS);
  assert.ok(adv?.kind === "count" && adv.big === "20" && adv.of === "/ 300" && /Too few resolved/.test(adv.note));
  const early = cardFigure(bench("answered-before-verdict"), FEEDS);
  assert.ok(early?.kind === "count" && early.big === "574");
  assert.deepEqual(cardMeta(bench("newest-weeks"), FEEDS), { dataset: "v1 val", size: "12,874 finds", lastRun: "19 Oct 2026" });
  assert.deepEqual(cardMeta(bench("unknown-species"), FEEDS).size, "649 finds");
  assert.deepEqual(cardMeta(bench("advance-predictions"), FEEDS), { dataset: "live", size: "≤ 300 a night", lastRun: "nightly" });
});

test("with a feed missing, a card has no number rather than a made-up one; the sealed test never has one", () => {
  for (const b of BENCHMARKS) assert.equal(cardFigure(b, {}), null, b.slug);
  assert.equal(cardFigure(bench("sealed-test"), FEEDS), null);
  assert.deepEqual(cardMeta(bench("newest-weeks"), {}), { dataset: "v1 val", size: "–", lastRun: "–" });
  assert.equal(cardFigure(bench("rare-species"), { howGood: { ...HOW_GOOD, species_by_depth: [] } }), null);
});

test("benchmark filters: words anywhere, OR within a chip group, AND across groups; status or name order", () => {
  const all = filterBenchmarks(BENCHMARKS, NO_FILTER);
  assert.equal(all.length, BENCHMARKS.length);
  assert.deepEqual([...new Set(all.map((b) => b.status))], ["running", "partial", "sealed", "planned", "history"]);
  assert.ok(filterBenchmarks(BENCHMARKS, { ...NO_FILTER, q: "SEALED test" }).some((b) => b.slug === "sealed-test"));
  assert.deepEqual(filterBenchmarks(BENCHMARKS, { ...NO_FILTER, q: "v1-test" }).map((b) => b.slug), ["sealed-test"]);
  assert.deepEqual(filterBenchmarks(BENCHMARKS, { ...NO_FILTER, q: "inaturalist" }).map((b) => b.slug).sort(),
                   ["held-out", "sealed-test"]);
  assert.deepEqual(filterBenchmarks(BENCHMARKS, { ...NO_FILTER, status: ["sealed", "history"] }).map((b) => b.slug),
                   ["sealed-test", "held-out"]);
  assert.deepEqual(filterBenchmarks(BENCHMARKS, { ...NO_FILTER, measures: ["danger"] }).map((b) => b.slug), ["poisonous-mistakes"]);
  assert.deepEqual(filterBenchmarks(BENCHMARKS, { ...NO_FILTER, status: ["running"], data: ["live"] }).map((b) => b.slug),
                   ["rare-species", "advance-predictions", "answered-before-verdict"]);
  assert.deepEqual(filterBenchmarks(BENCHMARKS, { ...NO_FILTER, q: "no such benchmark" }), []);
  const byName = filterBenchmarks(BENCHMARKS, { ...NO_FILTER, sort: "name" }).map((b) => b.name);
  assert.deepEqual(byName, [...byName].sort((a, b) => a.localeCompare(b)));
});

test("chip counts come from the whole list, as the mockup's: running 4, accuracy 6, v1 validation 5", () => {
  const c = benchmarkFacetCounts();
  assert.deepEqual(c.status, { running: 4, partial: 1, sealed: 1, planned: 2, history: 1 });
  assert.equal(c.measures.accuracy, 6);
  assert.equal(c.measures.danger, 1);
  assert.deepEqual(c.data, { v1val: 5, sealed: 1, live: 4, prev1: 1 });
  assert.deepEqual(statusBar().map((s) => [s.status, s.n]), [["running", 4], ["partial", 1], ["sealed", 1], ["planned", 2], ["history", 1]]);
});

test("a filter round-trips through the address, unknown values dropped", () => {
  const f = { q: "rare", status: ["running" as const], measures: ["rare" as const, "danger" as const], data: ["v1val" as const], sort: "name" as const };
  assert.equal(filterToSearch(f), "?q=rare&status=running&measures=rare%2Cdanger&data=v1val&sort=name");
  assert.deepEqual(filterFromSearch(filterToSearch(f)), f);
  assert.deepEqual(filterFromSearch("?status=running,bogus&measures=nope&sort=weird"), { ...NO_FILTER, status: ["running"] });
  assert.equal(filterToSearch(NO_FILTER), "");
  assert.equal(isFiltered(NO_FILTER), false);
  assert.equal(isFiltered({ ...NO_FILTER, sort: "name" }), false, "sorting is not filtering");
  assert.deepEqual(toggle(["a"], "b"), ["a", "b"]);
  assert.deepEqual(toggle(["a", "b"], "a"), ["b"]);
});

const entry = (slug: string, date: string, status: ExperimentSummary["status"], extra: Partial<ExperimentSummary> = {}): ExperimentSummary => ({
  title: slug, slug, date, status, reproducibility: "exploratory-pre-freeze", question: "q", headline: "h", verdict: "v",
  decision: "pending", ...extra });
const ENTRIES = [
  entry("screen", "2026-10-13", "done", { reproducibility: "reproduced-on-v1", decision: "Supplement only",
        benchmark: "v1 seeded subset (seed 20261013): 664 val records excluding the former held-out", split: "dev", title: "Backbone screen" }),
  entry("ood", "2026-10-13", "done", { reproducibility: "reproduced-on-v1", benchmark: "ood-v1-2026-10-13: 1,500 photos", split: "test" }),
  entry("prior", "2026-09-28", "adopted", { benchmark: "heldout-2026-10-08 dev and test", split: "dev; test", decision: "Steve: yes",
        headline: "Held-out test: species 48.3 -> 51.2% (744 fixed, 454 broken), genus 80.1 -> 80.1%, family 86.8 -> 87.0%" }),
  entry("weeks", "2026-10-08", "dropped", { benchmark: "comparison 20261008-012435-4ef7b0 (1,152); 8,000-record sample", split: "sample; temporal" }),
];

test("experiment test sets come from each entry's benchmark and split; 'former held-out' is not the held-out set", () => {
  assert.deepEqual(testSetsOf(ENTRIES[0]), ["v1val"]);
  assert.deepEqual(testSetsOf(ENTRIES[1]), ["sealedsets"]);
  assert.deepEqual(testSetsOf(ENTRIES[2]), ["heldout"]);
  assert.deepEqual(testSetsOf(ENTRIES[3]), ["weeks", "sample"]);
});

test("experiment filters: dataset, status, test set and search together; newest first", () => {
  assert.deepEqual(filterEntries(ENTRIES, NO_EXPERIMENT_FILTER).map((e) => e.slug), ["screen", "ood", "weeks", "prior"],
                   "newest first; a day's entries by title");
  assert.deepEqual(filterEntries(ENTRIES, { ...NO_EXPERIMENT_FILTER, dataset: "v1" }).map((e) => e.slug), ["screen", "ood"]);
  assert.deepEqual(filterEntries(ENTRIES, { ...NO_EXPERIMENT_FILTER, dataset: "pre", status: ["adopted"] }).map((e) => e.slug), ["prior"]);
  assert.deepEqual(filterEntries(ENTRIES, { ...NO_EXPERIMENT_FILTER, sets: ["sample", "sealedsets"] }).map((e) => e.slug), ["ood", "weeks"]);
  assert.deepEqual(filterEntries(ENTRIES, { ...NO_EXPERIMENT_FILTER, q: "BACKBONE screen" }).map((e) => e.slug), ["screen"]);
  const c = entryCounts(ENTRIES);
  assert.deepEqual([c.all, c.v1, c.pre, c.pending], [4, 2, 2, 2]);
  assert.equal(c.status.done, 2);
  assert.deepEqual(entriesByDay(ENTRIES).map((d) => [d.day, d.statuses.length]),
                   [["2026-09-28", 1], ["2026-10-08", 1], ["2026-10-13", 2]]);
});

test("an entry's excerpt is its headline cut at a word, with an ellipsis", () => {
  assert.equal(excerpt("short"), "short");
  const cut = excerpt(ENTRIES[2].headline, 60);
  assert.ok(cut.endsWith("…") && cut.length <= 61, cut);
  assert.ok(!/\s…$/.test(cut));
  assert.ok(ENTRIES[2].headline.startsWith(cut.slice(0, -1)));
});

test("the in-page nav marks the last section whose top has passed the line under the bars", () => {
  const tops = [300, 900, null, 1500];
  assert.equal(activeSection(tops, 0), 0, "the first before any is reached");
  assert.equal(activeSection(tops, 800), 1);
  assert.equal(activeSection(tops, 1400), 3, "a missing section is skipped");
  assert.equal(activeSection(tops, 5000), 3);
  assert.equal(activeSection([], 100), 0);
});
