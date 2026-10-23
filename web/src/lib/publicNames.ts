/**
 * What visitors read for a model. Outside the Research section only "MycoMap Vision" and its
 * version, never the backbone it was built on (Steve, 2026-10-11); the Research section adds
 * each version's training data (publicView.modelName, api.modelLabel). Backbone and method
 * ids stay as they are in the API; this is display only.
 *
 * The versions (Steve, 2026-10-23): the site's 336 px fine-tune is "MycoMap Vision v1", the
 * paper's open-data one "MycoMap Vision v1-open", the earlier 224 px fine-tunes "v0" and
 * "v0-open", and any other fine-tune (448 px, samples, ...) "MycoMap Vision (development,
 * <size> px)". The image size is a detail (tooltips, qualifiers), except for development
 * models, which differ only by it.
 */

/** The fine-tunes trained only on openly licensed (Creative Commons, incl. non-commercial)
 *  photos (`--public-trainable-only`): the paper's 336 px headline model and the 224 px one.
 *  Every other fine-tune learned from all its training photos. */
export const OPEN_V1 = "bioclip-2-336-ft-20261020-173106";
export const PUBLIC_TRAINED = new Set([OPEN_V1, "bioclip-2-ft-20261014-003539"]);
/** The site's served model when the names were set (any internal 336 px fine-tune is v1). */
export const SERVED_V1 = "bioclip-2-336-ft-20261018-171420";

/** The image size each base model (models.ALIASES) reads, in px. */
export const BASE_PX: Record<string, number> = {
  "bioclip-2": 224, "bioclip-2-336": 336, "bioclip-2-448": 448, bioclip: 224,
  "dinov2-b14": 518, "dinov2-l14": 518, "dinov3-b16": 256, "dinov3-l16": 256,
  "dinov3-b16-512": 512, "dinov3-l16-512": 512, "siglip2-l16-384": 384, "eva02-l14-448": 448,
  "convnextv2-l": 384,
};

export interface VisionVersion {
  /** "MycoMap Vision v1", "MycoMap Vision (development, 448 px)". */
  name: string;
  /** What sets it apart, said in brackets after the name in Research: its training data,
   *  or "224 px, development" for v0. Null for a development model (its size is its name). */
  qualifier: string | null;
  /** The image size it reads, when its base is known. */
  px: number | null;
}

/** A fine-tune of ours (`<base>-ft-<YYYYMMDD>-<HHMMSS>` from finetune.py, or `<base>-ft-sample`)
 *  as the MycoMap Vision version it is; null for anything that is not one of our fine-tunes. */
export function visionVersion(backbone: string): VisionVersion | null {
  const ft = /^(.+)-ft-(\d{8}-\d+|sample)$/.exec(backbone);
  if (!ft) return null;
  const [, base, run] = ft;
  const px = BASE_PX[base] ?? null;
  const open = PUBLIC_TRAINED.has(backbone);
  if (run !== "sample" && base === "bioclip-2-336") {
    return { name: `MycoMap Vision v1${open ? "-open" : ""}`,
             qualifier: open ? "open data only" : "public + non-public training data", px };
  }
  if (run !== "sample" && base === "bioclip-2") {
    return { name: `MycoMap Vision v0${open ? "-open" : ""}`, qualifier: "224 px, development", px };
  }
  return { name: `MycoMap Vision (development${px ? `, ${px} px` : ""})`, qualifier: null, px };
}

/** The model as visitors outside Research read it: "MycoMap Vision v1", "MycoMap Vision v0",
 *  "MycoMap Vision (development, 448 px)"; plain "MycoMap Vision" for anything else of ours. */
export function publicModelName(backbone: string): string {
  if (backbone === "external:inat-cv") return "iNaturalist's computer vision";
  return visionVersion(backbone)?.name ?? "MycoMap Vision";
}

/** How a method scores, in plain words; "closest" as in "Closest DNA-verified specimens". */
export const PUBLIC_METHOD_LABEL: Record<string, string> = {
  nearest: "closest specimen",
  "species-mean": "species average",
  "nearest+mean": "closest specimens + species average",
  "nearest+mean+place": "closest specimens + species average + place and date",
  "nearest+mean-records-balanced": "closest specimens + species average, by specimen",
  "nearest+mean-records-balanced+place": "closest specimens + species average, by specimen + place and date",
  "nearest-mix": "closest specimen, photo votes",
  linear: "trained classifier",
  hybrid: "trained classifier + closest specimens",
  "linear+prior": "trained classifier + range and season",
  "hybrid+prior": "trained classifier + closest specimens + range and season",
  "nearest+prior": "closest specimen + range and season",
  "species-mean+prior": "species average + range and season",
};

export function publicMethodLabel(method: string): string {
  return PUBLIC_METHOD_LABEL[method] ?? method.replace(/\+/g, " + ");
}

/** Sentence case, for menus and the start of a line. */
export const capitalize = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
