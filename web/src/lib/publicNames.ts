/**
 * What visitors read for a model (Steve, 2026-10-11): only "MycoMap Vision", never the
 * backbone it was built on. The Research section keeps the scientific names
 * (publicView.modelName, api.modelLabel); everywhere else uses these. Backbone and method
 * ids stay as they are in the API; this is display only.
 */

const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
                "September", "October", "November", "December"];

/** "MycoMap Vision · October 2026 model" for a model trained on records up to that month. */
export function publicModelName(backbone: string): string {
  if (backbone === "external:inat-cv") return "iNaturalist's computer vision";
  const ft = /-ft-(\d{4})(\d{2})\d{2}-\d+$/.exec(backbone);
  if (ft) return `MycoMap Vision · ${MONTHS[Number(ft[2]) - 1]} ${ft[1]} model`;
  if (backbone.endsWith("-ft-sample")) return "MycoMap Vision · test model";
  return "MycoMap Vision";
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
