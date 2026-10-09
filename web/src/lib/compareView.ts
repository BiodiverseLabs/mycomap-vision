// What the Models page's "Approaches compared" charts show, worked out apart from React so
// it can be tested. Only models scored on the same records sit in one chart: a comparison
// is a set of scoreboard runs sharing record set, cutoff and number of test records.

import type { Rank, RunReport, ScoreRun } from "./api";

export const STANDARD_K = [1, 3, 5, 10] as const;
export const STANDARD_BANDS = ["0", "1-4", "5-19", "20-99", "100+"] as const;
export const SERVED_METHOD = "nearest";

export interface StandardScore { n: number; top1: number; top3: number; top5: number; top10: number }
/** evaluate.py's "standard" block (Steve's summary format): per rank, "all" and each band. */
export type Standard = Record<Rank, Record<string, StandardScore>>;

export function standardOf(report: RunReport | undefined): Standard | null {
  const s = (report?.all_photos as unknown as { standard?: Standard } | undefined)?.standard;
  return s && s.species ? s : null;
}

const newest = (g: ScoreRun[]) => g.reduce((m, r) => (r.created_at > m ? r.created_at : m), "");

/** Runs measured on one set of records, most recently measured set first. Several comparisons
 *  of the same records merge; each approach (backbone and method) keeps only its newest run. */
export function comparisons(runs: ScoreRun[]): ScoreRun[][] {
  const groups = new Map<string, Map<string, ScoreRun>>();
  for (const r of runs) {
    const key = `${r.record_set}|${r.cutoff}|${r.n_test}`;
    const g = groups.get(key) ?? new Map<string, ScoreRun>();
    const who = `${r.backbone}/${r.method}`;
    const had = g.get(who);
    if (!had || r.created_at > had.created_at || (r.created_at === had.created_at && r.id > had.id)) {
      g.set(who, r);
    }
    groups.set(key, g);
  }
  return [...groups.values()].map((g) => [...g.values()])
    .sort((a, b) => newest(b).localeCompare(newest(a)) || b[0].n_test - a[0].n_test);
}

/** The comparison shown first: the most recently measured one that scores the model the site
 *  serves next to at least one other approach; else any with two approaches; else the first. */
export function defaultComparison(groups: ScoreRun[][], servedBackbones: string[] = [],
                                  servedMethod: string = SERVED_METHOD): number {
  const serves = (g: ScoreRun[]) => g.some((r) => r.method === servedMethod
                                              && servedBackbones.includes(r.backbone));
  let i = groups.findIndex((g) => g.length >= 2 && serves(g));
  if (i < 0) i = groups.findIndex((g) => g.length >= 2);
  return i < 0 ? 0 : i;
}

export const isExternal = (backbone: string) => backbone.startsWith("external:");

export interface Bar {
  id: number;
  backbone: string;
  method: string;
  value: number | null;
  n: number | null;
  external: boolean;
  served: boolean;
}

/** One bar per run: its top-k at `rank` over all test records, best first. Top 1 and 5 come
 *  from every run; top 3 and 10 only from runs that carry the standard summary. */
export function bars(runs: ScoreRun[], reports: Map<number, RunReport>, rank: Rank, k: number,
                     servedBackbones: string[] = [], servedMethod: string = SERVED_METHOD): Bar[] {
  return runs.map((r) => {
    const rep = reports.get(r.id);
    const std = standardOf(rep)?.[rank]?.all;
    const old = rep?.all_photos?.[rank]?.all;
    let value: number | null = null;
    if (std) value = std[`top${k}` as keyof StandardScore] ?? null;
    else if (k === 1) value = old?.top1 ?? (r[`${rank}_top1` as keyof ScoreRun] as number | null);
    else if (k === 5) value = old?.top5 ?? null;
    return {
      id: r.id, backbone: r.backbone, method: r.method, value,
      n: std?.n ?? old?.n ?? null, external: isExternal(r.backbone),
      served: r.method === servedMethod && servedBackbones.includes(r.backbone),
    };
  }).sort((a, b) => (b.value ?? -1) - (a.value ?? -1));
}

/** The k values to offer: 1/3/5/10 once any run carries the standard summary (runs without
 *  it show "not measured" at 3 and 10), else the 1 and 5 every run has. */
export function kChoices(runs: ScoreRun[], reports: Map<number, RunReport>): number[] {
  return runs.some((r) => standardOf(reports.get(r.id))) ? [...STANDARD_K] : [1, 5];
}

export interface DepthRow { id: number; backbone: string; method: string; external: boolean;
                            cells: (StandardScore | null)[] }

/** Species by the true species' reference records, for runs with the standard summary. */
export function depthRows(runs: ScoreRun[], reports: Map<number, RunReport>,
                          rank: Rank = "species"): { rows: DepthRow[]; missing: number } {
  const rows: DepthRow[] = [];
  let missing = 0;
  for (const r of runs) {
    const std = standardOf(reports.get(r.id));
    if (!std) { missing++; continue; }
    rows.push({ id: r.id, backbone: r.backbone, method: r.method, external: isExternal(r.backbone),
                cells: STANDARD_BANDS.map((b) => std[rank]?.[b] ?? null) });
  }
  rows.sort((a, b) => (b.cells.reduce((s, c) => s + (c?.top1 ?? 0) * (c?.n ?? 0), 0))
                      - (a.cells.reduce((s, c) => s + (c?.top1 ?? 0) * (c?.n ?? 0), 0)));
  return { rows, missing };
}

/** Sequential ramp (one hue, light to dark) for a 0-1 share; ink flips to white on dark steps. */
const RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
              "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];
export function heat(v: number | null): { bg: string; ink: string } {
  if (v == null) return { bg: "transparent", ink: "#52514e" };
  const i = Math.min(RAMP.length - 1, Math.max(0, Math.round(v * (RAMP.length - 1))));
  return { bg: RAMP[i], ink: i >= 7 ? "#ffffff" : "#0b0b0b" };
}

// --- held-out benchmarks (mv benchmark-export / benchmark-import, aggregates only) ----------

export interface Rate { n: number; right?: number; rate: number | null; ci95?: [number, number];
                        ci95_observer_bootstrap?: [number, number] }
export interface PublishedBenchmark {
  benchmark: string;
  split: string;
  sealed: boolean;
  records: number;
  scored_records: number;
  imported_at: string;
  /** "exploratory-pre-freeze" until results come from a frozen dataset release. */
  reproducibility?: string;
  models: Record<string, { records: number } & Partial<Record<Rank, Partial<Record<"top1" | "top5", Rate>>>>>;
  species_by_reference_records: Record<string, Record<string, Partial<Record<Rank, Rate>>>>;
  summary?: { models?: Record<string, { rows?: Record<string, Partial<Record<string, Rate | null>>> }> } | null;
}

/** A held-out model key, 'backbone/method[@place][[size]]', as backbone and method. The place
 *  ("@org": the record's place from mycomap.org) stays on the method so labels can say so. */
export function splitModelKey(key: string): { backbone: string; method: string } {
  const i = key.indexOf("/");
  return i < 0 ? { backbone: key, method: "" } : { backbone: key.slice(0, i), method: key.slice(i + 1) };
}

export interface HeldoutBar { key: string; backbone: string; method: string; value: number | null;
                              ci: [number, number] | null; n: number | null; external: boolean }

/** k values to offer: 1/3/5/10 once any model's standard summary is stored ten deep (the
 *  others show "not measured" there), else the 1 and 5 every model has. */
export function heldoutKs(b: PublishedBenchmark): number[] {
  const rows = Object.values(b.summary?.models ?? {}).map((m) => m.rows?.["species strict"]);
  return rows.some((r) => r?.top10 != null) ? [...STANDARD_K] : [1, 5];
}

export function heldoutBars(b: PublishedBenchmark, rank: Rank, k: number): HeldoutBar[] {
  return Object.entries(b.models).map(([key, m]) => {
    const fromSummary = b.summary?.models?.[key]?.rows?.[`${rank} strict`]?.[`top${k}`] ?? null;
    const plain = k === 1 || k === 5 ? m[rank]?.[`top${k}` as "top1" | "top5"] ?? null : null;
    const r = fromSummary ?? plain;
    const { backbone, method } = splitModelKey(key);
    return { key, backbone, method, value: r?.rate ?? null,
             ci: r?.ci95_observer_bootstrap ?? r?.ci95 ?? null, n: r?.n ?? null,
             external: isExternal(backbone) };
  }).sort((a, z) => (z.value ?? -1) - (a.value ?? -1));
}

/** Species top-1 by the true species' reference records, one row per model, standard bands. */
export function heldoutDepth(b: PublishedBenchmark): { key: string; backbone: string; method: string;
                                                        cells: (Rate | null)[] }[] {
  return Object.entries(b.species_by_reference_records).map(([key, bands]) => ({
    key, ...splitModelKey(key),
    cells: STANDARD_BANDS.map((band) => bands[band]?.species ?? null),
  }));
}
