// The v1 / v2 dropdown on How it works and Get involved (Steve, 2026-11-22): v1 shows the numbers
// those pages always showed (/api/how-good); a newer model shows its scores since its training
// cut, day by day (/api/daily-scores, daily_scores.py). The dropdown opens on the server's
// default: the newer model once it is available, else v1.

import { num, pct, type DailyScores, type ModelScores, type RankScore, type ScoreDay } from "./api";

export const V1 = "v1";

export interface ModelOption { id: string; label: string }

/** "MycoMap Vision v1", "MycoMap Vision v2", "MycoMap Vision v2-open": the site's names for its models. */
export const choiceLabel = (id: string) => `MycoMap Vision ${id}`;

/** The dropdown's choices: v1 always, then each newer model the server has scores for. */
export function modelOptions(d: DailyScores | null | undefined): ModelOption[] {
  const ids = [V1, ...(d?.available ?? []).filter((id) => id !== V1 && d?.models?.[id])];
  return [...new Set(ids)].map((id) => ({ id, label: choiceLabel(id) }));
}

/** Which model the dropdown opens on: the server's default when it is a choice, else v1. */
export function defaultModel(d: DailyScores | null | undefined): string {
  const want = d?.default;
  return want && modelOptions(d).some((o) => o.id === want) ? want : V1;
}

/** A newer model's scores, or null for v1 (or a model the server doesn't send). */
export function scoresFor(d: DailyScores | null | undefined, id: string): ModelScores | null {
  return id === V1 ? null : d?.models?.[id] ?? null;
}

/** "61.8% (n = 152)": a rate with how many records it is over, never one without the other. */
export function withN(s: RankScore | null | undefined, digits = 0): string {
  if (!s || !s.n || s.top1 == null) return `no records (n = 0)`;
  return `${pct(s.top1, digits)} (n = ${num(s.n)})`;
}

/** "between 58.0% and 65.4%": the 95% range, said plainly. */
export function rangeText(s: RankScore | null | undefined): string | null {
  return s?.ci95 ? `between ${pct(s.ci95[0], 1)} and ${pct(s.ci95[1], 1)}` : null;
}

/** The days with records scored, newest last; the newest `last` for the chart. */
export function scoredDays(m: ModelScores, last?: number): ScoreDay[] {
  const days = m.days.filter((d) => d.records > 0);
  return last ? days.slice(-last) : days;
}
