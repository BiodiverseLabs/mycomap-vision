// The site's model and the paper's headline model, side by side (Steve, 2026-10-21/22), on
// dataset release v1's validation split. There is no scoreboard row for the public model on
// the server, so its figures ship here, copied from the private research repo's
// paper/o1/o1-336-c-full.json (commit e9df6e4, "O1 at 336"): tables.val and paired.val, both
// models scored with nearest+mean-records-balanced (photos only) against the full v1
// reference, record set 36fd876aa871, the answer key species-rank-2026-10-18, erratum r4,
// observer-clustered bootstrap (2,000 draws). Never v1-test numbers.

import { modelName, modelTooltip } from "./publicView";

export interface TopK { n: number; top1: number; top3: number; top5: number; top10: number }
export interface ModelFigures {
  backbone: string;
  role: string;
  species: TopK;
  genus: TopK;
  /** Species, by the true species' DNA-verified records in the reference: 0, 1-4, 5-19, 20-99, 100+. */
  depth: Record<string, TopK>;
}

export const TWO_MODELS = {
  release: "v1",
  split: "validation",
  testRecords: 12874,
  referenceRecords: 155648,
  method: "nearest+mean-records-balanced",
  source: "paper/o1/o1-336-c-full.json (research repo, e9df6e4)",
  models: [
    { backbone: "bioclip-2-336-ft-20261018-171420", role: "The model the site serves",
      species: { n: 12798, top1: 0.5632, top3: 0.7355, top5: 0.7887, top10: 0.8428 },
      genus: { n: 12865, top1: 0.854, top3: 0.9387, top5: 0.9562, top10: 0.971 },
      depth: {
        "0": { n: 649, top1: 0, top3: 0, top5: 0, top10: 0 },
        "1-4": { n: 1391, top1: 0.1855, top3: 0.3559, top5: 0.4421, top10: 0.5694 },
        "5-19": { n: 2864, top1: 0.4511, top3: 0.6858, top5: 0.7758, top10: 0.8691 },
        "20-99": { n: 5736, top1: 0.7019, top3: 0.8699, top5: 0.9118, top10: 0.9453 },
        "100+": { n: 2158, top1: 0.7563, top3: 0.9101, top5: 0.9393, top10: 0.9652 },
      } },
    { backbone: "bioclip-2-336-ft-20261020-173106", role: "The paper's headline model",
      species: { n: 12798, top1: 0.5593, top3: 0.7338, top5: 0.7852, top10: 0.8411 },
      genus: { n: 12865, top1: 0.8525, top3: 0.9378, top5: 0.9547, top10: 0.9712 },
      depth: {
        "0": { n: 649, top1: 0, top3: 0, top5: 0, top10: 0 },
        "1-4": { n: 1391, top1: 0.1804, top3: 0.3472, top5: 0.4184, top10: 0.5593 },
        "5-19": { n: 2864, top1: 0.449, top3: 0.6826, top5: 0.7709, top10: 0.8677 },
        "20-99": { n: 5736, top1: 0.6947, top3: 0.8694, top5: 0.9113, top10: 0.9453 },
        "100+": { n: 2158, top1: 0.7581, top3: 0.911, top5: 0.9416, top10: 0.9639 },
      } },
  ] as ModelFigures[],
  /** Open data only minus the site's model, species top 1, on the same finds. */
  paired: { n: 12798, observers: 752, diff: -0.0039, ci95: [-0.0077, -0.0001] as [number, number] },
};

/** A paired difference in points, one decimal more than the table: "−0.39 [−0.77, −0.01]". */
export function pointsDiff(diff: number, ci: [number, number]): string {
  const p = (v: number) => `${v < 0 ? "−" : "+"}${Math.abs(100 * v).toFixed(2)}`;
  return `${p(diff)} [${p(ci[0])}, ${p(ci[1])}]`;
}

/** Each model's label as everywhere on the site (publicView.modelName): "MycoMap Vision v1
 *  (public + non-public training data)", "MycoMap Vision v1-open (open data only)". */
export const twoModelLabel = (m: ModelFigures) => modelName(m.backbone);
/** Its tooltip: the image size and the internal id. */
export const twoModelTip = (m: ModelFigures) => modelTooltip(m.backbone);

/** The same two models answered from the openly licensed reference only (Steve, 2026-10-25):
 *  the paper's headline setup, where MycoMap Vision v1-open names 55.3% of species first. Copied
 *  from the research repo's paper/o1/o1-336-c-cc.json (commit e9df6e4): reference
 *  "public-trainable" (134,942 records), record set 8951e6f53b28, the same 12,874 val finds
 *  (12,798 with a species), erratum r4, observer-clustered bootstrap (2,000 draws). */
export const OPEN_REFERENCE = {
  referenceRecords: 134942,
  source: "paper/o1/o1-336-c-cc.json (research repo, e9df6e4)",
  species: {
    "bioclip-2-336-ft-20261018-171420": { n: 12798, top1: 0.5556, top3: 0.7285, top5: 0.7804, top10: 0.8341 },
    "bioclip-2-336-ft-20261020-173106": { n: 12798, top1: 0.5531, top3: 0.7261, top5: 0.7765, top10: 0.8329 },
  } as Record<string, TopK>,
  genus: {
    "bioclip-2-336-ft-20261018-171420": { n: 12865, top1: 0.8497, top3: 0.9373, top5: 0.9545, top10: 0.9698 },
    "bioclip-2-336-ft-20261020-173106": { n: 12865, top1: 0.849, top3: 0.9356, top5: 0.9528, top10: 0.9695 },
  } as Record<string, TopK>,
  /** v1-open minus v1, species top 1, on the open reference. */
  paired: { n: 12798, observers: 752, diff: -0.0025, ci95: [-0.0061, 0.001] as [number, number] },
};

/** The two references a v1-open number can come from, each labelled (Steve, 2026-10-25): the
 *  open reference is the paper's headline setup; the full one is what the site answers from. */
export const REFERENCE_LABEL = {
  open: "against the openly licensed reference only (the paper's headline setup)",
  full: "against the full reference",
} as const;

/** v1-open's species top 1 on each reference, for the places that quote it. */
export function openModelHeadline(): { open: TopK; full: TopK } {
  const open = OPEN_REFERENCE.species[TWO_MODELS.models[1].backbone];
  return { open, full: TWO_MODELS.models[1].species };
}
