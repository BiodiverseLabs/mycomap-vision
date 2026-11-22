// /api/daily-scores as the server sends it, for the tests: v2 scored on three days after its
// training cut (made-up numbers), the same over forty days (for the chart's 30-day window), and
// the answer while only v1 is offered.
import type { DailyScores, ModelScores, RankScore, ScoreDay } from "./api";

const r = (right: number, n: number, ci95: [number, number] | null = null): RankScore =>
  ({ right, n, top1: n ? Number((right / n).toFixed(4)) : null, ci95 });

const day = (d: string, records: number, sp: [number, number], ge: [number, number],
             fa: [number, number], roll: [number, number, number, number], photos: number,
             complete = true): ScoreDay => ({
  day: d, records, photos, species: r(...sp), genus: r(...ge), family: r(...fa),
  rolling_7_days: { records: roll[0], species: r(roll[1], roll[0]), genus: r(roll[2], roll[0]),
                    family: r(roll[3], roll[0]) },
  complete,
});

const tk = (n: number, top1: number, top3: number, top5: number, top10: number) => ({ n, top1, top3, top5, top10 });

export const V2_SCORES: ModelScores = {
  id: "v2", backbone: "bioclip-2.5-h14-336-ft-20261122-164602", method: "nearest+mean-records-balanced+place",
  trained_through: "2026-11-22T08:24:03+00:00", context: "photos + place and date",
  answer_key: "species-rank-2026-10-18", as_of: "2026-11-25T08:10:00+00:00",
  records: 403, new_photos: 1291, with_place: 391, with_date: 403, min_band_records: 30,
  days: [
    day("2026-11-22", 0, [0, 0], [0, 0], [0, 0], [0, 0, 0, 0], 0),
    day("2026-11-23", 152, [94, 152], [131, 152], [140, 152], [152, 94, 131, 140], 488),
    day("2026-11-24", 187, [121, 185], [163, 187], [175, 187], [339, 215, 294, 315], 601),
    day("2026-11-25", 64, [37, 63], [55, 64], [59, 64], [403, 252, 349, 374], 202, false),
  ],
  cumulative: { from: "2026-11-22", through: "2026-11-25", records: 403, photos: 1291,
                species: r(252, 400, [0.5814, 0.6767]), genus: r(349, 403, [0.8297, 0.8941]),
                family: r(374, 403, [0.9003, 0.9496]) },
  rolling_7_days: { through: "2026-11-25", records: 403, species: r(252, 400), genus: r(349, 403),
                    family: r(374, 403) },
  top_k: { k: [1, 3, 5, 10], species: tk(400, 0.63, 0.7825, 0.83, 0.8775),
           genus: tk(403, 0.866, 0.9454, 0.9628, 0.9752), family: tk(403, 0.928, 0.9727, 0.9826, 0.99) },
  species_by_depth: [
    { band: "0", n: 21, top1: null, top3: null, top5: null, top10: null, too_few: true },
    { band: "1-4", n: 64, top1: 0.2656, top3: 0.4531, top5: 0.5156, top10: 0.625, too_few: false },
    { band: "5-19", n: 98, top1: 0.551, top3: 0.7449, top5: 0.8163, top10: 0.8878, too_few: false },
    { band: "20-99", n: 160, top1: 0.775, top3: 0.9125, top5: 0.9375, top10: 0.9625, too_few: false },
    { band: "100+", n: 57, top1: 0.8246, top3: 0.9474, top5: 0.9649, top10: 0.9825, too_few: false },
  ],
  left_out: { vision_assisted: 5, not_a_fungus_label: 1 },
};

export const DAILY_V2: DailyScores = { public: true, available: ["v1", "v2"], default: "v2", models: { v2: V2_SCORES } };

/** Forty days scored after the cut, day i with 100 + i finds (the chart shows the last 30). */
export function fortyDays(): DailyScores {
  const start = Date.parse("2026-11-23T00:00:00Z");
  const days = Array.from({ length: 40 }, (_, i) => {
    const d = new Date(start + i * 86_400_000).toISOString().slice(0, 10);
    const n = 100 + i;
    return day(d, n, [Math.round(n * 0.6), n], [Math.round(n * 0.85), n], [Math.round(n * 0.9), n],
               [n, Math.round(n * 0.6), Math.round(n * 0.85), Math.round(n * 0.9)], 3 * n, i < 39);
  });
  return { ...DAILY_V2, models: { v2: { ...V2_SCORES, days } } };
}

/** The server with the display off, or before v2 has a score: only v1. */
export const DAILY_V1_ONLY: DailyScores = { public: false, available: ["v1"], default: "v1", models: {} };
