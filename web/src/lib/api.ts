// Types and calls for the MycoMap Vision API (`mv serve`, proxied at /api).

export type Rank = "family" | "genus" | "species";
export const RANKS: Rank[] = ["family", "genus", "species"];

export interface Candidate {
  name: string;
  score: number;
  confidence: number;
  reference_records: number;
}

export interface Specimen {
  observation_id: string;
  species: string;
  genus: string;
  family: string;
  similarity: number;
  matched_query_photo: number;
  photo_url: string | null;
  photo_owner: string | null;
  inat_url: string;
  species_url: string;
}

export interface IdentifyResult {
  model: { backbone: string; method: string };
  reference: { records: number; species: number; photos: number };
  photos: number;
  ranks: Record<Rank, Candidate[]>;
  specimens: Specimen[];
  hints: string[];
  species_url: string | null;
  confidence_note: string;
}

export interface Where {
  lat?: string;
  lng?: string;
  observedOn?: string;
}

export interface ContextUsed {
  latitude: number | null;
  longitude: number | null;
  observed_on: string | null;
  place_from: "entered" | "photo" | null;
  date_from: "entered" | "photo" | null;
}

export interface ModelsInfo {
  backbones: {
    backbone: string;
    spec: string;
    note: string;
    embedded_photos: number;
    photos_per_second: number | null;
  }[];
  methods: string[];
  ready: string[];
}

export interface Stats {
  records: number;
  records_north_america: number;
  label_conflicts: number;
  inat_ok: number;
  inat_missing: number;
  photos: number;
  photos_by_status: Record<string, number>;
  photos_by_license: Record<string, number>;
  photos_by_size: Record<string, number>;
  contributors_arr: number;
  embedded: Record<string, number>;
  names: number;
  names_by_records: Record<string, number>;
}

export interface ScoreRun {
  id: number;
  comparison_id: string;
  backbone: string;
  method: string;
  cutoff: string;
  test_days: number;
  n_reference: number;
  n_test: number;
  record_set: string;
  species_top1: number | null;
  genus_top1: number | null;
  family_top1: number | null;
  species_top1_first_photo: number | null;
  code_version: string | null;
  created_at: string;
}

export type BucketScores = Record<string, { n: number; top1: number; top5: number }>;
export interface RunReport {
  backbone: string;
  method: string;
  all_photos: Record<Rank, BucketScores>;
  first_photo_only: Record<Rank, BucketScores>;
  // External baselines (iNat) add these.
  species_names_inat_knows?: number;
  species_by_observed_year?: Record<string, { n: number; top1: number; top5: number } | null>;
}

const METHOD_LABEL: Record<string, string> = {
  nearest: "nearest specimen",
  "species-mean": "species average",
  "vision-max": "photo only, best photo",
  "combined-max": "with location, best photo",
  linear: "trained classifier",
  hybrid: "classifier + nearest specimen",
  "linear+prior": "trained classifier + range and season",
  "hybrid+prior": "classifier + nearest + range and season",
};

/** Human names for scoreboard rows: iNat's model and our methods in plain words. */
export function modelLabel(backbone: string, method: string): { name: string; how: string } {
  const name = backbone === "external:inat-cv" ? "iNat computer vision" : backbone;
  return { name, how: METHOD_LABEL[method] ?? method };
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(await errorText(res));
  return res.json() as Promise<T>;
}

async function errorText(res: Response): Promise<string> {
  try {
    const body = await res.json();
    return typeof body.detail === "string" ? body.detail : `HTTP ${res.status}`;
  } catch {
    return `HTTP ${res.status}`;
  }
}

export const api = {
  models: () => getJson<ModelsInfo>("/api/models"),
  stats: () => getJson<Stats>("/api/stats"),
  scoreboard: () => getJson<{ runs: ScoreRun[] }>("/api/scoreboard"),
  run: (id: number) => getJson<RunReport>(`/api/scoreboard/${id}`),
  async identify(photos: File[], models: string[], where: Where = {}):
      Promise<{ results: IdentifyResult[]; context_used: ContextUsed }> {
    const form = new FormData();
    photos.forEach((p) => form.append("photos", p));
    form.append("models", models.join(","));
    if (where.lat?.trim() && where.lng?.trim()) {
      form.append("lat", where.lat.trim());
      form.append("lng", where.lng.trim());
    }
    if (where.observedOn) form.append("observed_on", where.observedOn);
    const res = await fetch("/api/identify", { method: "POST", body: form });
    if (!res.ok) throw new Error(await errorText(res));
    return res.json();
  },
};

export const pct = (v: number | null | undefined, digits = 0) =>
  v == null ? "–" : `${(100 * v).toFixed(digits)}%`;

export const num = (v: number | null | undefined) => (v == null ? "–" : v.toLocaleString());

/** "backbone/method" key used by the API for one model. */
export const modelKey = (backbone: string, method: string) => `${backbone}/${method}`;
