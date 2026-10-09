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

/** Runs measured on one set of records, newest set first. */
export function comparisons(runs: ScoreRun[]): ScoreRun[][] {
  const groups = new Map<string, ScoreRun[]>();
  for (const r of runs) {
    const key = `${r.record_set}|${r.cutoff}|${r.n_test}`;
    groups.set(key, [...(groups.get(key) ?? []), r]);
  }
  return [...groups.values()].sort((a, b) => (b[0].cutoff ?? "").localeCompare(a[0].cutoff ?? "")
    || b[0].n_test - a[0].n_test);
}

/** The comparison shown first: the newest one that compares at least two approaches. */
export function defaultComparison(groups: ScoreRun[][]): number {
  const i = groups.findIndex((g) => g.length >= 2);
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
                     servedBackbones: string[] = []): Bar[] {
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
      served: r.method === SERVED_METHOD && servedBackbones.includes(r.backbone),
    };
  }).sort((a, b) => (b.value ?? -1) - (a.value ?? -1));
}

/** The k values every run in the comparison can answer. */
export function kChoices(runs: ScoreRun[], reports: Map<number, RunReport>): number[] {
  const allStandard = runs.every((r) => standardOf(reports.get(r.id)));
  return allStandard ? [...STANDARD_K] : [1, 5];
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
