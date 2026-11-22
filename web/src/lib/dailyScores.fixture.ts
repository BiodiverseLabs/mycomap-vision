// /api/daily-scores as the server sends it, for the tests: v2 scored on three days after its
// training cut (made-up numbers), and the answer while only v1 is offered.
import type { DailyScores, RankScore, ScoreDay } from "./api";

const r = (right: number, n: number, ci95: [number, number] | null = null): RankScore =>
  ({ right, n, top1: n ? Number((right / n).toFixed(4)) : null, ci95 });

const day = (d: string, records: number, sp: [number, number], ge: [number, number],
             roll: [number, number, number], photos: number, complete = true): ScoreDay => ({
  day: d, records, photos, species: r(...sp), genus: r(...ge),
  rolling_7_days: { records: roll[0], species: r(roll[1], roll[0]), genus: r(roll[2], roll[0]) }, complete,
});

export const DAILY_V2: DailyScores = {
  public: true, available: ["v1", "v2"], default: "v2",
  models: {
    v2: {
      id: "v2", backbone: "bioclip-2.5-h14-336-ft-20261122-164602", method: "nearest+mean-records-balanced+place",
      trained_through: "2026-11-22T08:24:03+00:00", context: "photos only", as_of: "2026-11-25T08:10:00+00:00",
      records: 403, new_photos: 1291,
      days: [
        day("2026-11-22", 0, [0, 0], [0, 0], [0, 0, 0], 0),
        day("2026-11-23", 152, [94, 152], [131, 152], [152, 94, 131], 488),
        day("2026-11-24", 187, [121, 185], [163, 187], [339, 215, 294], 601),
        day("2026-11-25", 64, [37, 63], [55, 64], [403, 252, 349], 202, false),
      ],
      cumulative: { from: "2026-11-22", through: "2026-11-25", records: 403, photos: 1291,
                    species: r(252, 400, [0.5814, 0.6767]), genus: r(349, 403, [0.8297, 0.8941]) },
      rolling_7_days: { through: "2026-11-25", records: 403, species: r(252, 400), genus: r(349, 403) },
      left_out: { vision_assisted: 5, not_a_fungus_label: 1 },
    },
  },
};

/** The server with the display off, or before v2 has a score: only v1. */
export const DAILY_V1_ONLY: DailyScores = { public: false, available: ["v1"], default: "v1", models: {} };
