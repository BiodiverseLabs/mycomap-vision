// The How it works page's numbers, shaped for its charts (Steve, 2026-10-22: the approved
// redesign). Every figure comes from the API as it is now (/api/stats, /api/models,
// /api/how-good, /api/speed/daily); a part the server doesn't send is left out, never made up.

import type { HowGood, ModelsInfo, SpeedDay, Stats, TopK } from "./api";
import { longDate, modelName } from "./publicView";

/** The box's clean measurement of one photo's time (identify only, no other load: 16 GB,
 *  4 vCPU, the 336 px model, October 2026), drawn as the speed chart's reference line. A fact
 *  about the box, like the Danish atlas study's figures, not a live number. */
export const CLEAN_SECONDS = 1.1;
export const CLEAN_LABEL = `Clean measurement, ${CLEAN_SECONDS} s`;
/** Fewer days than this are listed, not drawn as a line. */
export const MIN_CHART_DAYS = 3;

const SHORT_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
                      "Nov", "Dec"];
const DAY_MS = 86_400_000;

/** "2026-10-22" -> "22 Oct", as written (UTC days, no time zone shift). */
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

export interface SpeedPoint extends SpeedDay { x: number; y: number }
export type SpeedView =
  | { state: "none" }
  | { state: "few"; days: SpeedDay[]; since: string; latest: SpeedDay }
  | { state: "chart"; points: SpeedPoint[]; since: string; latest: SpeedPoint; yMax: number;
      yTicks: { v: number; y: number }[]; xTicks: { day: string; label: string; x: number }[];
      ref: { v: number; y: number } };

/** A step for about four to six gridlines up to `top`. */
function niceStep(top: number): number {
  for (const s of [0.25, 0.5, 1, 2, 5, 10, 20, 50]) if (top / s <= 6) return s;
  return 100;
}

/** What the speed chart draws from /api/speed/daily. x runs by date (a day without photos
 *  leaves its gap), y is seconds per photo from 0 (x and y are 0..1, y from the top). With
 *  fewer than MIN_CHART_DAYS days there is no line: the days are listed with when recording
 *  began. With none, nothing is drawn. */
export function speedView(days: SpeedDay[] | null | undefined, clean = CLEAN_SECONDS): SpeedView {
  const ok = [...(days ?? [])]
    .filter((d) => /^\d{4}-\d{2}-\d{2}$/.test(d.day) && Number.isFinite(d.seconds_per_photo)
                   && d.seconds_per_photo >= 0)
    .sort((a, b) => a.day.localeCompare(b.day));
  if (!ok.length) return { state: "none" };
  const since = ok[0].day;
  if (ok.length < MIN_CHART_DAYS) return { state: "few", days: ok, since, latest: ok[ok.length - 1] };
  const top = Math.max(clean, ...ok.map((d) => d.seconds_per_photo)) * 1.1;
  const step = niceStep(top);
  const yMax = Math.ceil(top / step - 1e-9) * step;
  const y = (v: number) => 1 - v / yMax;
  const first = dayNumber(since), span = dayNumber(ok[ok.length - 1].day) - first;
  const points = ok.map((d) => ({ ...d, x: (dayNumber(d.day) - first) / span, y: y(d.seconds_per_photo) }));
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
           xTicks, ref: { v: clean, y: y(clean) } };
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
