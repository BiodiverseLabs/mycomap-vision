// What the Research pages show, worked out apart from React so it can be tested (Steve,
// 2026-10-25: the approved Research redesign). Every number comes from the API or from a
// committed source named where it is used; a part the server doesn't send is left out (or
// said to be missing), never made up.

import type { HowGood, ProspectiveModel, RunReport, ScoreRun, TopK as ApiTopK } from "./api";
import { comparisons, groupRelease, incompleteRun, isExternal, standardOf, STANDARD_BANDS,
         type StandardScore } from "./compareView";
import type { EarlyAnswersGroup } from "./api";
import { BASE_PX, publicMethodLabel, visionVersion } from "./publicNames";
import { modelName } from "./publicView";
import { BENCHMARKS, DATA_SHORT, MEASURE_ORDER, DATA_ORDER, STATUS_ORDER, type Benchmark,
         type BenchmarkStatus, type DataSet, type Measure } from "./research";
import { datasetOf, type ExperimentStatus, type ExperimentSummary, EXPERIMENT_STATUSES,
         awaitsDecision } from "./experiments";

// --- numbers ------------------------------------------------------------------------------

/** A share as the Research pages write it, one decimal, no sign: 0.5917 -> "59.2". */
export const pt = (v: number | null | undefined, digits = 1): string =>
  v == null || !Number.isFinite(v) ? "–" : (100 * v).toFixed(digits);
/** The same with its percent sign: "59.2%". */
export const pc = (v: number | null | undefined, digits = 1): string =>
  v == null || !Number.isFinite(v) ? "–" : `${pt(v, digits)}%`;
/** A count with thousands separators: 12798 -> "12,798". */
export const nf = (n: number | null | undefined): string =>
  n == null || !Number.isFinite(n) ? "–" : n.toLocaleString("en-US");
/** The gap between two shares in points, two decimals: "4.47". */
export const pointsGap = (a: number, b: number): string => (100 * (a - b)).toFixed(2);

const MONTHS_SHORT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
/** A date as cards write it, in UTC: "19 Oct 2026". */
export function shortDay(iso: string | null | undefined): string {
  if (!iso) return "–";
  const d = new Date(iso.length === 10 ? `${iso}T00:00:00Z` : iso);
  if (Number.isNaN(d.getTime())) return "–";
  return `${d.getUTCDate()} ${MONTHS_SHORT[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
}

// --- the top-1/3/5/10 chart ---------------------------------------------------------------

export const KS = [1, 3, 5, 10] as const;
export type K = (typeof KS)[number];
/** One row of the top-k chart: shares 0-1, null where the run didn't measure that k. */
export interface TopKRow {
  key: string;
  name: string;
  detail?: string;
  /** How many finds the shares are of. */
  n?: number | null;
  values: Record<K, number | null>;
  /** The site's model and method, or the row the chart is about. */
  highlight?: boolean;
  /** An outside model. */
  outside?: boolean;
}

/** A top-k block (the API's {n, top1, top3, top5, top10}) as a chart row's values. */
export const topkValues = (t: Partial<StandardScore | ApiTopK> | null | undefined): Record<K, number | null> => ({
  1: t?.top1 ?? null, 3: t?.top3 ?? null, 5: t?.top5 ?? null, 10: t?.top10 ?? null,
});

export interface TopKAxis { lo: number; hi: number; ticks: number[] }

/** The chart's axis: from the lowest value rounded down to ten points (a step lower when a
 *  value would sit on the edge) to 100%, ticks every 20 points from there; 0-100 when nothing
 *  is measured. Shares, 0-1. */
export function topkAxis(rows: TopKRow[]): TopKAxis {
  const vals = rows.flatMap((r) => KS.map((k) => r.values[k])).filter((v): v is number => v != null);
  if (!vals.length) return { lo: 0, hi: 1, ticks: [0, 0.2, 0.4, 0.6, 0.8, 1] };
  const min = Math.min(...vals);
  let lo = Math.floor(min * 10) / 10;
  if (min - lo < 0.02 && lo > 0) lo -= 0.1;
  lo = Math.max(0, Math.round(lo * 10) / 10);
  const ticks: number[] = [];
  for (let t = lo; t <= 1.0001; t += 0.2) ticks.push(Math.round(t * 100) / 100);
  return { lo, hi: 1, ticks };
}

/** Where a share sits on the axis, in percent of the track (clamped). */
export function axisAt(v: number, axis: TopKAxis): number {
  const span = axis.hi - axis.lo || 1;
  return Math.min(100, Math.max(0, (100 * (v - axis.lo)) / span));
}

/** A tooltip's two lines, "Title|detail", as the chart's marks carry them. */
export function topkTip(row: TopKRow, k: K): string {
  const v = row.values[k];
  return `${row.name} · Top ${k}|${v == null ? "not measured" : `${pc(v)}${row.n ? ` of ${nf(row.n)} finds` : ""}`}`;
}

// --- reference-depth colours --------------------------------------------------------------

/** The depth bands' colours, light for few records to dark for many (the blue ramp of the
 *  heatmap and of the bars beside it). */
export const BAND_COLOR: Record<string, { bg: string; ink: string }> = {
  "0": { bg: "#cde2fb", ink: "#0b0b0b" },
  "1-4": { bg: "#86b6ef", ink: "#0b0b0b" },
  "5-19": { bg: "#3987e5", ink: "#0b0b0b" },
  "20-99": { bg: "#1c5cab", ink: "#ffffff" },
  "100+": { bg: "#0d366b", ink: "#ffffff" },
};

/** The heatmap's six steps (the approved mockup's scale): under 15%, then every 15 points to
 *  75% and over. Ink turns white on the two darkest. */
export const HEAT_STEPS = [
  { from: 0, bg: "#e9f1fc", ink: "#0b0b0b" }, { from: 0.15, bg: "#cde2fb", ink: "#0b0b0b" },
  { from: 0.30, bg: "#86b6ef", ink: "#0b0b0b" }, { from: 0.45, bg: "#3987e5", ink: "#0b0b0b" },
  { from: 0.60, bg: "#1c5cab", ink: "#ffffff" }, { from: 0.75, bg: "#0d366b", ink: "#ffffff" },
];
export function heatStep(v: number | null | undefined): { bg: string; ink: string } {
  if (v == null || !Number.isFinite(v)) return { bg: "transparent", ink: "#52514e" };
  let s = HEAT_STEPS[0];
  for (const step of HEAT_STEPS) if (v >= step.from - 1e-9) s = step;
  return { bg: s.bg, ink: s.ink };
}

// --- model names on research rows ---------------------------------------------------------

/** A run's model as a chart row names it: the model, then its training data, size and method
 *  on the line below ("MycoMap Vision v1" / "public + non-public training data · closest
 *  specimens + species average"). The live names (Steve, 2026-10-23): our fine-tunes by
 *  version; an unchanged base model as "BioCLIP 2 (base model, not fine-tuned)". */
export function modelParts(backbone: string, method?: string): { name: string; detail: string } {
  const how = method ? publicMethodLabel(method) : "";
  const join = (...parts: (string | null | undefined)[]) => parts.filter(Boolean).join(" · ");
  const v = visionVersion(backbone);
  if (v) return { name: v.name, detail: join(v.qualifier, how) };
  const full = modelName(backbone);
  const base = /^(.*\(base model, not fine-tuned\)), (\d+ px)$/.exec(full);
  if (base) return { name: base[1], detail: join(base[2], how) };
  if (isExternal(backbone)) return { name: full, detail: join("outside model", how) };
  const px = BASE_PX[backbone];
  return { name: full, detail: join(px ? `${px} px` : null, how) };
}

// --- scoreboard: what each Results section reads ------------------------------------------

const fineTuned = (backbone: string) => /-ft-/.test(backbone);

/** A comparison's runs as chart rows from their reports' standard summary (species, all
 *  photos), best top 1 first. Runs without the summary are left out and counted. */
export function comparisonRows(group: ScoreRun[], reports: Map<number, RunReport>, served: string[],
                               servedMethod: string, rank: "species" | "genus" | "family" = "species"):
    { rows: TopKRow[]; missing: number } {
  const rows: TopKRow[] = [];
  let missing = 0;
  for (const r of group) {
    const std = standardOf(reports.get(r.id))?.[rank]?.all;
    if (!std) { missing++; continue; }
    const p = modelParts(r.backbone, r.method);
    rows.push({ key: String(r.id), name: p.name, detail: p.detail, n: std.n, values: topkValues(std),
                highlight: served.includes(r.backbone) && r.method === servedMethod,
                outside: isExternal(r.backbone) });
  }
  rows.sort((a, b) => (b.values[1] ?? -1) - (a.values[1] ?? -1));
  return { rows, missing };
}

/** The release comparisons, the one scoring the site's model and method first, then those
 *  scoring the site's model at all, then the rest (newest first within each). */
export function releaseGroups(runs: ScoreRun[], served: string[], servedMethod: string): ScoreRun[][] {
  const groups = comparisons(runs.filter((r) => !incompleteRun(r))).filter((g) => groupRelease(g) !== null);
  const score = (g: ScoreRun[]) => g.some((r) => served.includes(r.backbone) && r.method === servedMethod) ? 0
    : g.some((r) => served.includes(r.backbone)) ? 1 : 2;
  return groups.map((g, i) => ({ g, i })).sort((a, b) => score(a.g) - score(b.g) || a.i - b.i).map(({ g }) => g);
}

/** The comparisons the "Approaches compared" charts show: the release comparisons that score
 *  the site's model. One that scores the site's method shows every approach on its finds; one
 *  that doesn't is the fine-tune against its own base model, so it keeps only those two models
 *  (an older model scored on the same records sits in the scoreboard, not in this chart). */
export function approachGroups(runs: ScoreRun[], served: string[], servedMethod: string):
    { runs: ScoreRun[]; baseOnly: boolean }[] {
  return releaseGroups(runs, served, servedMethod).filter((g) => g.some((r) => served.includes(r.backbone)))
    .map((g) => {
      if (g.some((r) => served.includes(r.backbone) && r.method === servedMethod)) return { runs: g, baseOnly: false };
      const bases = served.map((b) => b.replace(/-ft-.*$/, ""));
      return { runs: g.filter((r) => served.includes(r.backbone) || bases.includes(r.backbone)), baseOnly: true };
    });
}

/** The fine-tuned model against its own base model, unchanged, by the same method on the same
 *  finds (the landing's "Fine-tuning adds ..."): the newest release comparison holding both. */
export function fineTuneVsBase(runs: ScoreRun[], served: string[], method = "nearest+mean"):
    { tuned: ScoreRun; base: ScoreRun } | null {
  for (const g of releaseGroups(runs, served, method)) {
    const tuned = g.find((r) => served.includes(r.backbone) && fineTuned(r.backbone) && r.method === method);
    if (!tuned) continue;
    const baseName = tuned.backbone.replace(/-ft-.*$/, "");
    const base = g.find((r) => r.backbone === baseName && r.method === method);
    if (base) return { tuned, base };
  }
  return null;
}

/** The site's model with place and date against the same method from photos alone, on the
 *  same finds (the landing's "Place and date add ..."). Null when the served method uses no
 *  place, or either run is missing. */
export function placeVsPhotos(runs: ScoreRun[], served: string[], servedMethod: string):
    { place: ScoreRun; photos: ScoreRun } | null {
  if (!servedMethod.endsWith("+place")) return null;
  const photosMethod = servedMethod.slice(0, -"+place".length);
  for (const g of releaseGroups(runs, served, servedMethod)) {
    const place = g.find((r) => served.includes(r.backbone) && r.method === servedMethod);
    const photos = place && g.find((r) => r.backbone === place.backbone && r.method === photosMethod);
    if (place && photos) return { place, photos };
  }
  return null;
}

/** How many runs and comparisons the scoreboard holds, the incomplete one left out. */
export function scoreboardCounts(runs: ScoreRun[]): { runs: number; comparisons: number; release: number } {
  const ok = runs.filter((r) => !incompleteRun(r));
  return { runs: ok.length, comparisons: comparisons(ok).length,
           release: ok.filter((r) => r.dataset_release).length };
}

/** One line of the development-history table: a comparison before release v1, its best run of
 *  ours and iNaturalist's computer vision (with location) on the same records. */
export interface HistoryLine { key: string; measured: string; finds: number; references: number;
                               approaches: number; best: ScoreRun; inat: number | null }
export function historyLines(groups: ScoreRun[][]): HistoryLine[] {
  return groups.map((g) => {
    const ours = g.filter((r) => !isExternal(r.backbone));
    const best = [...(ours.length ? ours : g)].sort((a, b) => (b.species_top1 ?? -1) - (a.species_top1 ?? -1))[0];
    const inat = g.find((r) => r.backbone === "external:inat-cv" && r.method === "combined-max")
      ?? g.find((r) => r.backbone === "external:inat-cv");
    const newest = g.reduce((m, r) => (r.created_at > m ? r.created_at : m), "");
    return { key: `${g[0].record_set}|${g[0].n_test}`, measured: newest, finds: g[0].n_test,
             references: g[0].n_reference, approaches: g.length, best, inat: inat?.species_top1 ?? null };
  });
}

/** A scoreboard table row: species top 1/3/5/10 from the run's report when it carries the
 *  standard summary (else top 1 from the scoreboard and the rest "–"), genus and family top 1,
 *  the first photo alone, its finds and run id. */
export interface ScoreLine { run: ScoreRun; name: string; detail: string; species: Record<K, number | null>;
                             genus: number | null; family: number | null; firstPhoto: number | null;
                             finds: number; highlight: boolean }
export function scoreLines(group: ScoreRun[], reports: Map<number, RunReport>, served: string[],
                           servedMethod: string): ScoreLine[] {
  return group.map((r) => {
    const std = standardOf(reports.get(r.id));
    const p = modelParts(r.backbone, r.method);
    const sp = std?.species?.all;
    return { run: r, name: p.name, detail: p.detail,
             species: sp ? topkValues(sp) : { 1: r.species_top1, 3: null, 5: null, 10: null },
             genus: std?.genus?.all?.top1 ?? r.genus_top1, family: std?.family?.all?.top1 ?? r.family_top1,
             firstPhoto: r.species_top1_first_photo, finds: sp?.n ?? r.n_test,
             highlight: served.includes(r.backbone) && r.method === servedMethod };
  }).sort((a, b) => (b.species[1] ?? -1) - (a.species[1] ?? -1));
}

/** Species by the true species' reference records, one heatmap row per run with the standard
 *  summary: top 1 and top 5 per band, and the finds in each band (from the first row). */
export interface HeatRow { key: string; name: string; detail: string; highlight: boolean;
                           cells: ({ top1: number; top5: number; n: number } | null)[] }
export function heatRows(group: ScoreRun[], reports: Map<number, RunReport>, served: string[]):
    { rows: HeatRow[]; finds: (number | null)[] } {
  const rows: HeatRow[] = [];
  for (const r of group) {
    const std = standardOf(reports.get(r.id));
    if (!std) continue;
    const p = modelParts(r.backbone, r.method);
    rows.push({ key: String(r.id), name: p.name, detail: p.detail,
                highlight: served.includes(r.backbone) && /nearest\+mean/.test(r.method),
                cells: STANDARD_BANDS.map((b) => {
                  const c = std.species?.[b];
                  return c ? { top1: c.top1, top5: c.top5, n: c.n } : null;
                }) });
  }
  rows.sort((a, b) => weighted(b) - weighted(a));
  const firstHl = rows.findIndex((r) => r.highlight);
  rows.forEach((r, i) => { r.highlight = i === firstHl; });
  return { rows, finds: STANDARD_BANDS.map((_, i) => rows[0]?.cells[i]?.n ?? null) };
}
const weighted = (r: HeatRow) => r.cells.reduce((s, c) => s + (c ? c.top1 * c.n : 0), 0);

// --- live feeds -----------------------------------------------------------------------------

/** Answered before the DNA verdict, as the landing and Results show it: the group answered
 *  with place and date (the site's usual answer) when there is one, else the first. */
export function earlyFigures(groups: EarlyAnswersGroup[] | undefined): {
  answers: number; awaiting: number; afterVerdict: number; scored: number;
  askedBy: { who: string; n: number }[]; context: string; model: string;
} | null {
  if (!groups?.length) return null;
  const g = groups.find((x) => /place/.test(x.context)) ?? groups[0];
  const asked = Object.entries(g.asked_by ?? {}).map(([who, n]) => ({
    who: who === "site" ? "This site" : who, n: n as number }));
  return { answers: g.answers, awaiting: g.awaiting_verdict, afterVerdict: g.left_out?.answered_after_verdict ?? 0,
           scored: g.scored, askedBy: asked, context: g.context, model: modelParts(g.backbone).name };
}

/** The advance-prediction row with the most records resolved (the one worth quoting), and how
 *  many predictions all rows hold. */
export function prospectiveLead(models: ProspectiveModel[] | undefined):
    { row: ProspectiveModel; predicted: number; rows: number } | null {
  if (!models?.length) return null;
  const row = [...models].sort((a, b) => b.resolved - a.resolved || b.predicted - a.predicted)[0];
  return { row, predicted: models.reduce((s, m) => s + m.predicted, 0), rows: models.length };
}
/** Too few resolved to read a rate (the cards say so rather than show one). */
export const TOO_FEW_RESOLVED = 30;

// --- benchmarks: card numbers, facets and sort ----------------------------------------------

export interface BenchmarkFeeds {
  howGood?: HowGood;
  runs?: ScoreRun[];
  prospective?: ProspectiveModel[];
  early?: EarlyAnswersGroup[];
}

/** A benchmark card's number block, read from the feeds; null when it has none yet (the card
 *  then shows its `pending` words, or that the number is missing). */
export type CardFigure =
  | { kind: "topk"; big: string; caption: string; values: Record<K, number | null>; n: number | null }
  | { kind: "depth"; from: string; to: string; caption: string; bands: { band: string; top1: number; n: number }[] }
  | { kind: "count"; big: string; of?: string; caption: string; note: string };

export function cardFigure(b: Benchmark, f: BenchmarkFeeds): CardFigure | null {
  const h = b.headline;
  if (!h) return null;
  const hg = f.howGood;
  const served = hg?.run ? modelParts(hg.run.backbone, hg.run.method) : null;
  const withPlace = hg?.run?.method.endsWith("+place") ? "photos + place and date" : "photos only";
  switch (h.metric) {
    case "species-topk": {
      const s = hg?.top_k?.species;
      if (!s || s.top1 == null || !served) return null;
      return { kind: "topk", big: pc(s.top1), caption: `species top 1 · ${served.name}, ${withPlace}`,
               values: topkValues(s), n: s.n };
    }
    case "species-depth": {
      const bands = (hg?.species_by_depth ?? []).filter((d) => d.top1 != null)
        .map((d) => ({ band: d.band, top1: d.top1 as number, n: d.n }));
      const lo = bands.find((d) => d.band === "1-4");
      const hi = bands.find((d) => d.band === "100+");
      if (!lo || !hi) return null;
      return { kind: "depth", from: pt(lo.top1), to: pc(hi.top1), caption: "species top 1, 1-4 records against 100+", bands };
    }
    case "unknown-count": {
      const zero = hg?.species_by_depth?.find((d) => d.band === "0");
      if (!zero) return null;
      return { kind: "count", big: nf(zero.n), caption: "finds whose species has no reference record",
               note: "Planned: how well the top score separates known from unknown species (AUROC), and how often Vision holds back." };
    }
    case "prospective": {
      const lead = prospectiveLead(f.prospective);
      if (!lead) return null;
      const few = lead.row.resolved < TOO_FEW_RESOLVED;
      return { kind: "count", big: nf(lead.row.resolved), of: `/ ${nf(lead.row.predicted)}`,
               caption: `resolved so far (${modelParts(lead.row.backbone).name})`,
               note: few ? `Too few resolved to read a rate. Predictions saved before the name existed: ${nf(lead.predicted)} across ${lead.rows} model rows.`
                         : `Species top 1 ${pc(lead.row.species_top1)} on the ${nf(lead.row.resolved)} resolved.` };
    }
    case "early-answers": {
      const e = earlyFigures(f.early);
      if (!e) return null;
      return { kind: "count", big: nf(e.answers), caption: `answers kept · ${nf(e.awaiting)} awaiting their verdict`,
               note: e.scored ? `${nf(e.scored)} scored so far.`
                              : "0 scored so far. Top 1/3/5/10 for species strict, s.l. and complex appear as records turn green." };
    }
  }
}

/** A card's size and last-run lines: from its headline run when it has one, else its words. */
export function cardMeta(b: Benchmark, f: BenchmarkFeeds): { dataset: string; size: string; lastRun: string } {
  const dataset = DATA_SHORT[b.data[0]];
  const hg = f.howGood;
  const run = b.headline?.run === "served" ? hg?.run : null;
  const scored = run ? f.runs?.find((r) => r.id === run.id) : null;
  let size = b.size ?? "–";
  if (b.headline?.metric === "species-topk" && run) size = `${nf(run.n_test)} finds`;
  if (b.headline?.metric === "species-depth" && hg?.top_k?.species) size = `5 bands · ${nf(hg.top_k.species.n)}`;
  if (b.headline?.metric === "unknown-count") {
    const zero = hg?.species_by_depth?.find((d) => d.band === "0");
    if (zero) size = `${nf(zero.n)} finds`;
  }
  if (b.headline?.metric === "early-answers") {
    const e = earlyFigures(f.early);
    if (e) size = `${nf(e.answers)} answers`;
  }
  const lastRun = b.lastRun === "run" ? (scored ? shortDay(scored.created_at) : "–") : (b.lastRun ?? "–");
  return { dataset, size, lastRun };
}

export interface BenchmarkFilter { q: string; status: BenchmarkStatus[]; measures: Measure[]; data: DataSet[];
                                   sort: "status" | "name" }
export const NO_FILTER: BenchmarkFilter = { q: "", status: [], measures: [], data: [], sort: "status" };

/** Lower-case words, so a search matches whatever the case and order of its words. */
const words = (q: string) => q.toLowerCase().split(/\s+/).filter(Boolean);
const haystack = (b: Benchmark) => [b.name, b.longName, b.summary, b.whyItMatters, b.records, b.scoring,
  b.elsewhere, b.status, ...b.measures, ...b.data].join(" ").toLowerCase();

/** The benchmarks a filter keeps: every word of the search somewhere in its text; within a
 *  chip group any chosen value (OR), across groups all of them (AND). Sorted by status (the
 *  summary's order) then as listed, or by name. */
export function filterBenchmarks(list: Benchmark[], f: BenchmarkFilter): Benchmark[] {
  const q = words(f.q);
  const kept = list.filter((b) => (!f.status.length || f.status.includes(b.status))
    && (!f.measures.length || b.measures.some((m) => f.measures.includes(m)))
    && (!f.data.length || b.data.some((d) => f.data.includes(d)))
    && q.every((w) => haystack(b).includes(w)));
  const idx = new Map(list.map((b, i) => [b.slug, i]));
  return kept.sort(f.sort === "name" ? (a, b) => a.name.localeCompare(b.name)
    : (a, b) => STATUS_ORDER.indexOf(a.status) - STATUS_ORDER.indexOf(b.status) || idx.get(a.slug)! - idx.get(b.slug)!);
}

/** How many benchmarks carry each value, for the chips (of the whole list, so a count never
 *  changes as filters are chosen). */
export function benchmarkFacetCounts(list: Benchmark[] = BENCHMARKS) {
  const status = Object.fromEntries(STATUS_ORDER.map((s) => [s, list.filter((b) => b.status === s).length])) as Record<BenchmarkStatus, number>;
  const measures = Object.fromEntries(MEASURE_ORDER.map((m) => [m, list.filter((b) => b.measures.includes(m)).length])) as Record<Measure, number>;
  const data = Object.fromEntries(DATA_ORDER.map((d) => [d, list.filter((b) => b.data.includes(d)).length])) as Record<DataSet, number>;
  return { status, measures, data };
}

const LIST = (v: string | null) => (v ? v.split(",").filter(Boolean) : []);

/** A filter kept in the address (?q=&status=&measures=&data=&sort=), so a filtered list can be
 *  shared; unknown values are dropped. */
export function filterFromSearch(search: string): BenchmarkFilter {
  const p = new URLSearchParams(search);
  return {
    q: p.get("q") ?? "",
    status: LIST(p.get("status")).filter((s): s is BenchmarkStatus => (STATUS_ORDER as string[]).includes(s)),
    measures: LIST(p.get("measures")).filter((m): m is Measure => (MEASURE_ORDER as string[]).includes(m)),
    data: LIST(p.get("data")).filter((d): d is DataSet => (DATA_ORDER as string[]).includes(d)),
    sort: p.get("sort") === "name" ? "name" : "status",
  };
}
export function filterToSearch(f: BenchmarkFilter): string {
  const p = new URLSearchParams();
  if (f.q.trim()) p.set("q", f.q.trim());
  if (f.status.length) p.set("status", f.status.join(","));
  if (f.measures.length) p.set("measures", f.measures.join(","));
  if (f.data.length) p.set("data", f.data.join(","));
  if (f.sort === "name") p.set("sort", "name");
  const s = p.toString();
  return s ? `?${s}` : "";
}
export const isFiltered = (f: BenchmarkFilter) =>
  !!f.q.trim() || f.status.length > 0 || f.measures.length > 0 || f.data.length > 0;

/** Turn one chip on or off. */
export function toggle<T>(list: T[], v: T): T[] {
  return list.includes(v) ? list.filter((x) => x !== v) : [...list, v];
}

/** The summary's status bar and counts, in the summary's order, statuses with none left out. */
export function statusBar(list: Benchmark[] = BENCHMARKS): { status: BenchmarkStatus; n: number }[] {
  return STATUS_ORDER.map((s) => ({ status: s, n: list.filter((b) => b.status === s).length })).filter((x) => x.n > 0);
}

// --- experiments: facets ---------------------------------------------------------------------

/** Which test records an entry's numbers come from (the Test set filter), read from its
 *  benchmark and split fields. An entry can use several. */
export type TestSet = "v1val" | "sealedsets" | "heldout" | "weeks" | "sample";
export const TEST_SET_LABEL: Record<TestSet, string> = {
  v1val: "v1 validation", sealedsets: "Sealed side sets", heldout: "Held-out benchmark",
  weeks: "Newest weeks (live)", sample: "Development sample",
};
export const TEST_SETS = Object.keys(TEST_SET_LABEL) as TestSet[];

export function testSetsOf(e: ExperimentSummary): TestSet[] {
  const b = `${e.benchmark ?? ""}`.toLowerCase();
  const split = `${e.split ?? ""}`.toLowerCase();
  const out = new Set<TestSet>();
  if (/\bood-v1|sealed side/.test(b)) out.add("sealedsets");
  if (/(?<!former )held-?out/.test(b)) out.add("heldout");
  if (/\bv1 val\b|v1 seeded subset|dataset release v1 val|served v1 reference/.test(b)) out.add("v1val");
  if (/temporal/.test(split) || /\bcomparison 20\d{6}/.test(b)) out.add("weeks");
  if (/sample/.test(split) || /\bsample\b|random10000/.test(b)) out.add("sample");
  return TEST_SETS.filter((t) => out.has(t));
}

export interface ExperimentFilter { q: string; dataset: "all" | "v1" | "pre"; status: ExperimentStatus[]; sets: TestSet[] }
export const NO_EXPERIMENT_FILTER: ExperimentFilter = { q: "", dataset: "all", status: [], sets: [] };

const datasetKey = (e: ExperimentSummary): "v1" | "pre" => (datasetOf(e) === "pre-v1" ? "pre" : "v1");

/** The entries a filter keeps, newest first: dataset, any chosen status, any chosen test set,
 *  and every word of the search in its title, headline, question or verdict. */
export function filterEntries(list: ExperimentSummary[], f: ExperimentFilter): ExperimentSummary[] {
  const q = words(f.q);
  return list.filter((e) => (f.dataset === "all" || datasetKey(e) === f.dataset)
      && (!f.status.length || f.status.includes(e.status))
      && (!f.sets.length || testSetsOf(e).some((t) => f.sets.includes(t)))
      && q.every((w) => [e.title, e.headline, e.question, e.verdict, e.slug].join(" ").toLowerCase().includes(w)))
    .sort((a, b) => b.date.localeCompare(a.date) || a.title.localeCompare(b.title));
}

export function entryCounts(list: ExperimentSummary[]) {
  return {
    all: list.length,
    v1: list.filter((e) => datasetKey(e) === "v1").length,
    pre: list.filter((e) => datasetKey(e) === "pre").length,
    pending: list.filter(awaitsDecision).length,
    status: Object.fromEntries(EXPERIMENT_STATUSES.map((s) => [s, list.filter((e) => e.status === s).length])) as Record<ExperimentStatus, number>,
    sets: Object.fromEntries(TEST_SETS.map((t) => [t, list.filter((e) => testSetsOf(e).includes(t)).length])) as Record<TestSet, number>,
  };
}

/** An entry's line under its title: its headline, cut at a word near `max` characters. */
export function excerpt(text: string, max = 110): string {
  const t = text.replace(/\s+/g, " ").trim();
  if (t.length <= max) return t;
  const cut = t.slice(0, max);
  const at = cut.lastIndexOf(" ");
  return `${(at > max * 0.6 ? cut.slice(0, at) : cut).replace(/[\s,;:(]+$/, "")}…`;
}

/** Entries by day, oldest first, each day's entries in their status order (the strip). */
export function entriesByDay(list: ExperimentSummary[]): { day: string; statuses: ExperimentStatus[] }[] {
  const days = new Map<string, ExperimentStatus[]>();
  for (const e of list) days.set(e.date, [...(days.get(e.date) ?? []), e.status]);
  return [...days.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([day, statuses]) => ({ day, statuses }));
}

// --- the sticky in-page nav -----------------------------------------------------------------

/** The section being read: the last one whose top has passed the line `offset` px below the
 *  top of the window (the header, the Research bar and the in-page nav sit above it); the
 *  first while none has. Tops are document offsets, in the nav's order; a missing section is
 *  null and never chosen. */
export function activeSection(tops: (number | null)[], scrollY: number, offset = 170): number {
  let cur = 0;
  tops.forEach((t, i) => { if (t != null && t <= scrollY + offset) cur = i; });
  return cur;
}
