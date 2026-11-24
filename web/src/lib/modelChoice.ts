// The Identify page's model picker: one scoring method plus two add-on switches,
// mapped onto the method names the API offers (GET /api/models `methods`, limited
// on the server box by MV_METHODS). Only offered methods are ever sent.
//
//   Trained classifier  linear   (+ nearest specimens: hybrid)
//   Nearest + average   nearest+mean  (the site's default, Steve 2026-10-09;
//                       with place and date: nearest+mean+place, Steve 2026-10-11)
//   ...by specimen      nearest+mean-records-balanced (+place): each species by its two
//                       best specimens, a cross-post counted once; the server's default
//                       from Steve's E32 decision (2026-10-18), so the page asks what
//                       mycomap.org's Validate Records asks and both share one kept answer
//   Nearest specimen    nearest
//   Species average     species-mean
//   ...the others with "+prior" when place and date (range and season) are used.

export type Base = "linear" | "nearest+mean-records-balanced" | "nearest+mean" | "nearest" | "species-mean";

export const BASES: Base[] = ["linear", "nearest+mean-records-balanced", "nearest+mean", "nearest", "species-mean"];

export const BASE_LABEL: Record<Base, string> = {
  linear: "Trained classifier",
  "nearest+mean-records-balanced": "Closest specimens + species average, by specimen",
  "nearest+mean": "Closest specimens + species average",
  nearest: "Closest specimen",
  "species-mean": "Species average",
};

export interface Choice {
  base: Base;
  /** Blend in the nearest DNA-verified specimens (trained classifier only). */
  weighNearest: boolean;
  /** Use place and date: the range-and-season score. */
  usePlace: boolean;
}

/** A method's place-and-date variant: nearest+mean's (and its by-specimen form's) is the
 *  boost prior "+place", the others' the range-and-season "+prior". */
const withPlace = (core: string) => (core.startsWith("nearest+mean") ? `${core}+place` : `${core}+prior`);

/** True for a method that uses place and date. */
export const usesPlaceAndDate = (method: string) => method.endsWith("+prior") || method.endsWith("+place");

/** The method name a choice asks for, whether or not it is offered. */
export function methodFor(c: Choice): string {
  const core = c.base === "linear" && c.weighNearest ? "hybrid" : c.base;
  return c.usePlace ? withPlace(core) : core;
}

const variants = (base: Base): string[] =>
  base === "linear" ? ["linear", "hybrid", "linear+prior", "hybrid+prior"] : [base, withPlace(base)];

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
    usePlace: v.some(usesPlaceAndDate) && v.some((m) => !usesPlaceAndDate(m)),
  };
}

/** The starting choice: the server's default method (GET /api/models `default_method`)
 *  when it offers it; else the first offered scoring method with both add-ons. */
export function defaultChoice(offered: string[], preferred?: string | null): Choice | null {
  if (preferred && offered.includes(preferred)) {
    const base = BASES.find((b) => variants(b).includes(preferred));
    if (base) {
      return { base, weighNearest: preferred.startsWith("hybrid"), usePlace: usesPlaceAndDate(preferred) };
    }
  }
  const base = offeredBases(offered)[0];
  return base ? { base, weighNearest: true, usePlace: true } : null;
}

/** Most models compared side by side in one identification. */
export const MAX_COMPARED = 2;

/** The model the page selects first (GET /api/models): the server's named default
 *  (`default_backbone`, MV_DEFAULT_BACKBONE) when it is ready; null while it is named but
 *  not ready, so the page selects none rather than another model in its place (the server
 *  refuses an identification that names no model then); with no named default, the first
 *  ready one, as before. */
export function defaultBackbone(m: { ready: string[]; default_backbone?: string | null } | undefined): string | null {
  if (!m) return null;
  if (m.default_backbone) return m.ready.includes(m.default_backbone) ? m.default_backbone : null;
  return m.ready[0] ?? null;
}

type SiteModels = { ready: string[]; default_backbone?: string | null; default_ready?: boolean;
                    site_default_backbone?: string | null; site_default_ready?: boolean };

/** The model the Identify page selects first: the site's default (`site_default_backbone`,
 *  MV_SITE_DEFAULT_BACKBONE) when the server names one, null while it isn't ready (never
 *  another model in its place); else defaultBackbone, as before. How it works keeps
 *  defaultBackbone (mycomap.org's model). */
export function siteDefaultBackbone(m: SiteModels | undefined): string | null {
  if (m?.site_default_backbone) return m.ready.includes(m.site_default_backbone) ? m.site_default_backbone : null;
  return defaultBackbone(m);
}

/** Whether the page's default model is named but not served (the page selects none and
 *  says so): the site's when it names one, else MV_DEFAULT_BACKBONE's. */
export function siteDefaultMissing(m: SiteModels | undefined): boolean {
  if (m?.site_default_backbone) return m.site_default_ready === false;
  return !!m?.default_backbone && m.default_ready === false;
}
