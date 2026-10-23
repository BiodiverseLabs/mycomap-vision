// The How it works page's numbers, shaped for its charts (Steve, 2026-10-22: the approved
// redesign). Every figure comes from the API as it is now (/api/stats, /api/models,
// /api/how-good, /api/speed/daily); a part the server doesn't send is left out, never made up.

import type { HowGood, ModelsInfo, SpeedDay, Stats, TopK } from "./api";
import { longDate, modelName } from "./publicView";

/** Fewer days than this are listed, not drawn as a line. */
export const MIN_CHART_DAYS = 3;
/** The rolling line's window, in calendar days, and its name in the legend and tooltip. */
export const ROLLING_DAYS = 7;
export const ROLLING_LABEL = "7-day average";

const SHORT_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
                      "Nov", "Dec"];
const DAY_MS = 86_400_000;

/** "2026-10-22" -> "22 Oct", as written (the server's US Eastern days, no time zone shift). */
export function shortDate(iso: string): string {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso);
  return m ? `${Number(m[3])} ${SHORT_MONTHS[Number(m[2]) - 1]}` : iso;
}

const dayNumber = (iso: string) => Date.parse(`${iso.slice(0, 10)}T00:00:00Z`) / DAY_MS;

/** Seconds per photo as the page writes them: "1.24 s". */
export const seconds = (v: number) => `${v.toFixed(2)} s`;

export const recordingSince = (firstDay: string) => `Recording since ${longDate(firstDay)}`;

/** The newest day's figure: "2.28 s today", or "2.28 s on 21 Oct" when today has none yet. */
export function latestLabel(latest: SpeedDay, today: string): string {
  return latest.day === today ? `${seconds(latest.seconds_per_photo)} today`
    : `${seconds(latest.seconds_per_photo)} on ${shortDate(latest.day)}`;
}

export interface SpeedPoint extends SpeedDay {
  x: number; y: number;
  /** The 7-day average ending this day, and where it sits (null before there are 2 days). */
  rolling: number | null; ry: number | null;
}
export type SpeedView =
  | { state: "none" }
  | { state: "few"; days: SpeedDay[]; since: string; latest: SpeedDay; rolling: number | null }
  | { state: "chart"; points: SpeedPoint[]; since: string; latest: SpeedPoint; yMax: number;
      yTicks: { v: number; y: number }[]; xTicks: { day: string; label: string; x: number }[];
      /** Whether the 7-day line is drawn (2 days or more). */
      hasRolling: boolean };

/** Each recorded day's 7-day average: the mean of that day's figure and those of the recorded
 *  days among the 6 calendar days before it (so at most 7 days, fewer where days are missing;
 *  a day without photos is skipped, never counted as 0). /api/speed/daily serves each day's
 *  average only, not how many photos it had, so this is an unweighted mean of daily averages:
 *  a quiet day weighs as much as a busy one. Empty until there are 2 days, so it never just
 *  repeats the one day there is. `days` oldest first, one per day. */
export function rollingAverage(days: SpeedDay[], window = ROLLING_DAYS): { day: string; seconds: number }[] {
  if (days.length < 2) return [];
  return days.map((d, i) => {
    const end = dayNumber(d.day);
    let sum = 0, n = 0;
    for (let j = i; j >= 0 && end - dayNumber(days[j].day) < window; j--) {
      sum += days[j].seconds_per_photo;
      n++;
    }
    return { day: d.day, seconds: sum / n };
  });
}

/** A step for about four to six gridlines up to `top`. */
function niceStep(top: number): number {
  for (const s of [0.25, 0.5, 1, 2, 5, 10, 20, 50]) if (top / s <= 6) return s;
  return 100;
}

/** What the speed chart draws from /api/speed/daily. x runs by date (a day without photos
 *  leaves its gap), y is seconds per photo from 0 (x and y are 0..1, y from the top). With
 *  fewer than MIN_CHART_DAYS days there is no line: the days are listed with when recording
 *  began (and, from 2 days, their 7-day average). With none, nothing is drawn. */
export function speedView(days: SpeedDay[] | null | undefined): SpeedView {
  const seen = new Set<string>();
  const ok = [...(days ?? [])]
    .filter((d) => /^\d{4}-\d{2}-\d{2}$/.test(d.day) && Number.isFinite(d.seconds_per_photo)
                   && d.seconds_per_photo >= 0)
    .sort((a, b) => a.day.localeCompare(b.day))
    .filter((d) => !seen.has(d.day) && !!seen.add(d.day));
  if (!ok.length) return { state: "none" };
  const since = ok[0].day;
  const rolling = rollingAverage(ok);
  if (ok.length < MIN_CHART_DAYS) {
    return { state: "few", days: ok, since, latest: ok[ok.length - 1],
             rolling: rolling.length ? rolling[rolling.length - 1].seconds : null };
  }
  const top = Math.max(...ok.map((d) => d.seconds_per_photo)) * 1.1 || 1;
  const step = niceStep(top);
  const yMax = Math.ceil(top / step - 1e-9) * step;
  const y = (v: number) => 1 - v / yMax;
  const first = dayNumber(since), span = dayNumber(ok[ok.length - 1].day) - first;
  const points = ok.map((d, i) => {
    const r = rolling[i]?.seconds ?? null;
    return { ...d, x: (dayNumber(d.day) - first) / span, y: y(d.seconds_per_photo),
             rolling: r, ry: r == null ? null : y(r) };
  });
  const yTicks = [];
  for (let v = 0; v <= yMax + 1e-9; v += step) yTicks.push({ v: Number(v.toFixed(2)), y: y(v) });
  // Up to five date labels, evenly spaced in time, always the first and the last day.
  const n = Math.min(4, span);
  const xTicks = [];
  for (let i = 0; i <= n; i++) {
    const at = first + Math.round((span * i) / n);
    const day = new Date(at * DAY_MS).toISOString().slice(0, 10);
    xTicks.push({ day, label: shortDate(day), x: (at - first) / span });
  }
  return { state: "chart", points, since, latest: points[points.length - 1], yMax, yTicks,
           xTicks, hasRolling: rolling.length > 0 };
}

/** The point the crosshair snaps to for a pointer at fraction `fx` across the plot. */
export function nearestPoint<T extends { x: number }>(points: T[], fx: number): T | null {
  let best: T | null = null;
  for (const p of points) if (!best || Math.abs(p.x - fx) < Math.abs(best.x - fx)) best = p;
  return best;
}

/** Names in the reference set by how many DNA-verified records each has (/api/stats
 *  names_by_records), in the server's bands, the single-record band first. */
const NAME_BANDS: [string, string][] = [["1", "1 record"], ["2", "2"], ["3-5", "3–5"],
                                        ["6-30", "6–30"], ["31+", "31+"]];
export function namesByRecords(s: Stats | undefined): { key: string; label: string; n: number;
                                                        share: number }[] | null {
  const b = s?.names_by_records;
  if (!b || !s?.names) return null;
  const rows = NAME_BANDS.filter(([k]) => typeof b[k] === "number").map(([key, label]) => ({
    key, label, n: b[key], share: b[key] / s.names }));
  return rows.length ? rows : null;
}

/** A reference-depth band as a label: "0" -> "0 records", "20-99" -> "20–99 records". */
export const depthLabel = (band: string) => `${band.replace("-", "–")} records`;

/** The top-k ladder's rows (species and genus), only for the k the run measured. */
export function topKRows(h: HowGood | undefined): { k: number; species: number | null;
                                                    genus: number | null }[] | null {
  const t = h?.top_k;
  if (!t?.species) return null;
  const at = (s: TopK | null | undefined, k: number) =>
    (s?.[`top${k}` as keyof TopK] as number | null | undefined) ?? null;
  const rows = t.k.map((k) => ({ k, species: at(t.species, k), genus: at(t.genus, k) }))
    .filter((r) => r.species != null || r.genus != null);
  return rows.length ? rows : null;
}

/** The model the site serves: the one the results are for when it is ready, else the first
 *  ready one. */
export function servedModel(m: ModelsInfo | undefined, prefer?: string | null) {
  const ready = m?.backbones.filter((b) => b.embedded_photos > 0 && m.ready.includes(b.backbone)) ?? [];
  return ready.find((b) => b.backbone === prefer) ?? ready[0] ?? null;
}

/** The small model credit, the only place How it works names the base model (Steve,
 *  2026-10-22, approved mockup): "BioCLIP 2, 336 px, fine-tuned" and its training data, and
 *  who made the base model. The product stays MycoMap Vision everywhere else. */
export function modelCredit(backbone: string): { name: string; title: string; detail: string | null;
                                                 madeBy: string | null } {
  const name = modelName(backbone);
  const cut = name.indexOf(" (");
  const fineTuned = /-ft-/.test(backbone);
  const madeBy = /^bioclip-2/.test(backbone)
    ? `Built on BioCLIP 2 by Imageomics${fineTuned ? ", fine-tuned on DNA-verified MycoMap records" : ""}.`
    : null;
  return { name, title: cut > 0 ? name.slice(0, cut) : name,
           detail: cut > 0 ? name.slice(cut + 1) : null, madeBy };
}

/** The US Eastern day of `now`, YYYY-MM-DD: the server counts speed days in Eastern time
 *  (speedlog.py), so "today" beside the newest figure must too. */
export function easternDay(now: Date): string {
  return new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York", year: "numeric",
                                            month: "2-digit", day: "2-digit" }).format(now);
}
