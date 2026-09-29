// The Identify page's model picker: one scoring method plus two add-on switches,
// mapped onto the method names the API offers (GET /api/models `methods`, limited
// on the server box by MV_METHODS). Only offered methods are ever sent.
//
//   Trained classifier  linear   (+ nearest specimens: hybrid)
//   Nearest specimen    nearest
//   Species average     species-mean
//   ...each with "+prior" when place and date (range and season) are used.

export type Base = "linear" | "nearest" | "species-mean";

export const BASES: Base[] = ["linear", "nearest", "species-mean"];

export const BASE_LABEL: Record<Base, string> = {
  linear: "Trained classifier",
  nearest: "Nearest specimen",
  "species-mean": "Species average",
};

export interface Choice {
  base: Base;
  /** Blend in the nearest DNA-verified specimens (trained classifier only). */
  weighNearest: boolean;
  /** Use place and date: the range-and-season score. */
  usePlace: boolean;
}

/** The method name a choice asks for, whether or not it is offered. */
export function methodFor(c: Choice): string {
  const core = c.base === "linear" && c.weighNearest ? "hybrid" : c.base;
  return c.usePlace ? `${core}+prior` : core;
}

const variants = (base: Base): string[] =>
  base === "linear" ? ["linear", "hybrid", "linear+prior", "hybrid+prior"] : [base, `${base}+prior`];

/** Scoring methods the server offers at least one variant of, in menu order. */
export function offeredBases(offered: string[]): Base[] {
  return BASES.filter((b) => variants(b).some((m) => offered.includes(m)));
}

/**
 * The offered method closest to what was asked: the exact one, else the same
 * method with the place switch flipped, else with the nearest switch flipped,
 * else both. null when the scoring method isn't offered at all.
 */
export function resolveMethod(c: Choice, offered: string[]): string | null {
  const tries: Choice[] = [
    c,
    { ...c, usePlace: !c.usePlace },
    { ...c, weighNearest: !c.weighNearest },
    { ...c, weighNearest: !c.weighNearest, usePlace: !c.usePlace },
  ];
  return tries.map(methodFor).find((m) => offered.includes(m)) ?? null;
}

/** Which switches make sense for this scoring method on this server. */
export function switchesFor(base: Base, offered: string[]): { weighNearest: boolean; usePlace: boolean } {
  const v = variants(base).filter((m) => offered.includes(m));
  return {
    weighNearest: base === "linear" && v.some((m) => m.startsWith("hybrid")) && v.some((m) => m.startsWith("linear")),
    usePlace: v.some((m) => m.endsWith("+prior")) && v.some((m) => !m.endsWith("+prior")),
  };
}

/** The starting choice: the trained classifier with both add-ons, where offered. */
export function defaultChoice(offered: string[]): Choice | null {
  const base = offeredBases(offered)[0];
  return base ? { base, weighNearest: true, usePlace: true } : null;
}

/** Most models compared side by side in one identification. */
export const MAX_COMPARED = 2;
