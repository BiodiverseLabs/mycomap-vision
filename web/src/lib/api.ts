// Types and calls for the MycoMap Vision API (`mv serve`, proxied at /api).

import type { Answered } from "./keptRun";
import type { PublishedBenchmark } from "./compareView";
import type { Experiment, ExperimentSummary } from "./experiments";
import { modelName, sendWaitingInLine } from "./publicView";

export type Rank = "family" | "genus" | "species";
export const RANKS: Rank[] = ["family", "genus", "species"];

/** The place lane's far-find cue for one name (reserved in docs/api.md): present only when a
 *  place or date was given and a model with a range-and-season score ran. */
export interface PlaceCue {
  nearest_record_km?: number | null;
  in_season?: boolean | null;
  far_from_known?: boolean | null;
  /** The server's own wording, for other API clients; the page builds its lines (answer.placeCue). */
  note?: string | null;
}

/** Often confused with (docs/api.md §4): on a species name people often get wrong, what DNA
 *  said instead on mycomap.org's DNA-checked iNat finds called that name. Absent otherwise. */
export interface ConfusedWith {
  /** At most 3, most finds first. */
  names: { name: string; finds: number }[];
  /** DNA-checked finds called this name, and how many DNA said were another species. */
  called: number;
  wrong: number;
  /** The line to show under the name, worded by the server. */
  note: string;
}

export interface Candidate {
  name: string;
  score: number;
  /** null when the chance is withheld: the photos may not show a fungus (the gate). */
  confidence: number | null;
  reference_records: number;
  place?: PlaceCue;
  confused_with?: ConfusedWith;
}

export interface Specimen {
  observation_id: string;
  /** What the record is named, at the finest rank it has (its species, else genus or family). */
  name?: string;
  /** "" when the record is named only to genus or higher: then it has no species page. */
  species: string;
  genus: string;
  family: string;
  similarity: number;
  matched_query_photo: number;
  photo_url: string | null;
  photo_owner: string | null;
  /** True when the record has photos but none may be shown here (all rights reserved,
   *  and the photographer hasn't given permission). */
  photo_withheld: boolean;
  /** Where the record is from: "inat" or "mo" (Mushroom Observer). */
  source?: string;
  /** The record's id at its source (the MO observation number for "mo:<n>"). */
  source_id?: string;
  /** The record's page at its source (iNat or Mushroom Observer). */
  record_url?: string | null;
  /** The iNat page, for iNat records only (older servers send it for every record). */
  inat_url: string | null;
  species_url: string | null;
  /** Set (true) only in a signed-in person's own answer, on a record they collected. */
  yours?: boolean;
}

/** The signed-in person's own DNA-verified records among the references (Steve,
 *  2026-10-18), sent only to that person. Absent: signed out, or mycomap.org couldn't be read. */
export interface Yours {
  /** False: no iNaturalist or Mushroom Observer account connected on mycomap.org. */
  linked: boolean;
  /** How many of the person's records are in the reference. */
  records: number;
  /** The most alike first, at most 6. */
  specimens: Specimen[];
}

/** Where one photo, scored on its own, puts the answer from all the photos together. */
export interface OverallPlace {
  name: string;
  /** 1 = this photo alone agrees with the overall answer. */
  position: number;
  confidence: number | null;
}

/** The not-a-fungus gate: whether a photo, or the whole find, looks like a fungus. */
export type GateVerdict = "fungus" | "unsure" | "not_fungus";

export interface PhotoGate {
  verdict: GateVerdict;
  /** BioCLIP 2's zero-shot chance that the photo shows a fungus. */
  p_fungus: number;
  message: string | null;
  /** Set when a photo not judged a fungus looks like an in-scope class ("slime": a slime mould). */
  looks_like?: string;
}

export interface FindGate {
  verdict: GateVerdict;
  message: string | null;
  /** What a find not judged a fungus looks like ("slime": a slime mould), or null. */
  looks_like?: string | null;
  /** Photos (0-based) left out of the answer because they don't look like a fungus. */
  left_out: number[];
  photos: PhotoGate[];
}

/** What the model says from one of your photos on its own. */
export interface PhotoResult {
  /** Index into the photos sent (0-based). */
  photo: number;
  ranks: Record<Rank, Candidate[]>;
  overall_top: Record<Rank, OverallPlace | null>;
  specimens: Specimen[];
  /** Missing when the gate didn't run. */
  gate?: PhotoGate;
}

/** The names worth considering at one rank, and how often such a list held the right
 *  one in the model's newest test (likely.py). */
export interface LikelySet {
  coverage: number;
  /** The same, measured on records the list was not fitted on. */
  checked_coverage: number | null;
  names: { name: string; confidence: number; reference_records: number; place?: PlaceCue;
           confused_with?: ConfusedWith }[];
  /** True when more names qualified than are listed. */
  capped: boolean;
}

export interface IdentifyResult {
  model: { backbone: string; method: string };
  reference: { records: number; species: number; photos: number };
  photos: number;
  /** The answer from all photos together (each photo's scores averaged). */
  ranks: Record<Rank, Candidate[]>;
  /** Missing for a model not yet calibrated with likely sets. */
  likely?: Partial<Record<Rank, LikelySet>>;
  specimens: Specimen[];
  /** Only in a signed-in person's own answer; `specimens` then leaves their records out. */
  yours?: Yours;
  per_photo: PhotoResult[];
  hints: string[];
  species_url: string | null;
  confidence_note: string;
  /** null when the gate didn't run (a backbone it can't read, or no gate saved). */
  gate?: FindGate | null;
  /** True when no chance is stated anywhere: the gate isn't sure this is a fungus. */
  confidence_withheld?: boolean;
  /** Per rank, the top name from the photos alone when the place and date changed it
   *  (reserved for the place lane). */
  photo_only_top?: Partial<Record<Rank, string>>;
  /** Whether a place/date prior was applied, and by which method (place lane). */
  prior?: { applied: boolean; method: string };
}

export interface Where {
  lat?: string;
  lng?: string;
  observedOn?: string;
  /** Set while the place / date shown is the one read from photo N (1-based), not typed. */
  placeFromPhoto?: number;
  dateFromPhoto?: number;
  /** Set while the place shown is the phone's current location ("Use my current location"). */
  placeFromDevice?: boolean;
  /** Set while the place shown was picked on the map (components/LocationPicker.tsx). */
  placeFromMap?: boolean;
}

export interface ContextUsed {
  latitude: number | null;
  longitude: number | null;
  observed_on: string | null;
  place_from: "entered" | "photo" | "observation" | null;
  date_from: "entered" | "photo" | "observation" | null;
}

export interface ModelsInfo {
  backbones: {
    backbone: string;
    spec: string;
    note: string;
    embedded_photos: number;
    photos_per_second: number | null;
    /** A fine-tuned model learned from records validated up to this date (ISO). */
    trained_through?: string | null;
  }[];
  methods: string[];
  ready: string[];
  /** What an identification uses when no method is named (MV_DEFAULT_METHOD). */
  default_method?: string;
  /** Most models one identification may run on this server (null: no limit). */
  max_models?: number | null;
}

export interface Stats {
  records: number;
  records_north_america: number;
  label_conflicts: number;
  inat_ok: number;
  inat_missing: number;
  photos: number;
  photos_by_license: Record<string, number>;
  contributors_arr: number;
  embedded: Record<string, number>;
  permissions: {
    granted: number;
    withdrawn: number;
    last_good_sync: string | null;
    last_attempt: { attempted_at: string; ok: boolean; error: string | null } | null;
    fresh_for_showing: boolean;
  };
  names: number;
  /** Names that are temporary codes: species known from DNA, not yet formally described. */
  names_provisional?: number;
  names_by_records: Record<string, number>;
  /** Projects that marked the reference records green. */
  projects?: number;
  /** People whose photos are in the reference records. */
  photographers?: number;
  /** Records DNA-verified in the week to the newest validation (ISO dates). */
  recent_week?: { records: number; from: string; through: string } | null;
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
  /** The frozen dataset release the run was measured on; null for the live data (or a draft). */
  dataset_release?: string | null;
}

export type BucketScores = Record<string, { n: number; top1: number; top5: number }>;
export interface GroupScore { name: string; n: number; species_top1: number | null; genus_top1: number | null }
export type EvalResult = Record<Rank, BucketScores> & {
  groups?: { project: GroupScore[]; observer: GroupScore[] };
};
export interface RunReport {
  backbone: string;
  method: string;
  all_photos: EvalResult;
  first_photo_only: EvalResult;
  // External baselines (iNat) add these.
  species_names_inat_knows?: number;
  species_by_observed_year?: Record<string, { n: number; top1: number; top5: number } | null>;
}

const METHOD_LABEL: Record<string, string> = {
  nearest: "nearest specimen",
  "species-mean": "species average",
  "nearest+mean": "nearest + species average",
  "nearest+mean+place": "nearest + species average + place and date",
  "nearest+mean-records-balanced": "nearest + species average, by specimen",
  "nearest+mean-records-balanced+place": "nearest + species average, by specimen + place and date",
  "nearest-mix": "nearest specimen, photo votes",
  "vision-max": "photo only, best photo",
  "combined-max": "with location, best photo",
  linear: "trained classifier",
  hybrid: "classifier + nearest specimen",
  "linear+prior": "trained classifier + range and season",
  "hybrid+prior": "classifier + nearest + range and season",
  "nearest+prior": "nearest specimen + range and season",
  "species-mean+prior": "species average + range and season",
};

/** Human names for scoreboard rows: iNat's model and our methods in plain words. */
export function modelLabel(backbone: string, method: string): { name: string; how: string } {
  return { name: modelName(backbone), how: METHOD_LABEL[method] ?? method };
}

/** An API refusal, with its HTTP status (401: sign in). */
export class ApiError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
  }
}

/**
 * POST with upload progress, which fetch can't report. Answers a standard Response so the
 * caller treats it like fetch's. Falls back to fetch where XMLHttpRequest is missing.
 */
function postWithProgress(url: string, body: FormData, onSent?: (fraction: number) => void): Promise<Response> {
  if (typeof XMLHttpRequest === "undefined") return fetch(url, { method: "POST", body });
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.responseType = "blob";
    onSent?.(0);
    xhr.upload.onprogress = (e) => { if (e.lengthComputable && e.total) onSent?.(Math.min(1, e.loaded / e.total)); };
    xhr.upload.onload = () => onSent?.(1);
    xhr.onload = () => {
      const headers = new Headers();
      xhr.getAllResponseHeaders().trim().split(/[\r\n]+/).forEach((line) => {
        const i = line.indexOf(":");
        if (i > 0) headers.append(line.slice(0, i).trim(), line.slice(i + 1).trim());
      });
      // 204/304 can't carry a body in a Response.
      const empty = xhr.status === 204 || xhr.status === 304;
      resolve(new Response(empty ? null : xhr.response, { status: xhr.status, statusText: xhr.statusText, headers }));
    };
    xhr.onerror = () => reject(new TypeError("Network error: the photos couldn't be sent. Check your connection and try again."));
    xhr.ontimeout = xhr.onerror;
    xhr.send(body);
  });
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new ApiError(res.status, await errorText(res));
  return res.json() as Promise<T>;
}

/** POST JSON (or nothing) with the site session cookie, as the signed-in GETs send it. */
async function postJson<T>(url: string, body?: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    credentials: "same-origin",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) throw new ApiError(res.status, await errorText(res));
  return res.json() as Promise<T>;
}

async function errorText(res: Response): Promise<string> {
  if (res.status === 401) return "Sign in with your mycomap.org account to use this.";
  try {
    const body = await res.json();
    return typeof body.detail === "string" ? body.detail : `HTTP ${res.status}`;
  } catch {
    return `HTTP ${res.status}`;
  }
}

export interface ProspectiveModel {
  backbone: string;
  method: string;
  predicted: number;
  resolved: number;
  family_top1: number | null;
  genus_top1: number | null;
  species_top1: number | null;
  mean_species_confidence: number | null;
  /** Predictions whose records were first validated after the paper's sealed-test cut:
   *  counted, but how they fared stays hidden until the test is opened. */
  sealed?: number;
  model_version?: string | null;
  serving_release?: string | null;
}

export interface Me {
  signin: "off" | "identify" | "all";
  user: { id: string; name: string | null } | null;
}

// --- Human Feedback (/feedback): places where Vision and the reference data disagree ---------

/** Who may review, and where: admins see every item, others their projects' items. */
export interface FeedbackMe {
  user: { id: string; name: string | null };
  admin: boolean;
  /** mycomap.org's staff flag: BLAST links open the admin results form. */
  staff?: boolean;
  projects: string[];
}

/** Where a decision takes effect: a change on mycomap.org, in Vision, or a note only. */
export type FeedbackApplies = "org" | "vision" | "note";

export interface FeedbackOption {
  code: string;
  label: string;
  applies: FeedbackApplies;
  /** The decision needs a value, e.g. the name to rename to. */
  needs_value: boolean;
  /** The kind of Name Update this decision opens on mycomap.org (a whole-name decision), or
   *  null for one that opens none (feedback_org.py). */
  name_update?: "respelling" | "accepted_name" | "rename" | "same_taxon" | null;
}

export interface FeedbackType {
  type: string;
  title: string;
  about: string;
  shows: string[];
  granularity: string;
  model_flagged: boolean;
  /** The page section the queue belongs to: "cleanup" or "validation". */
  section: FeedbackSection;
  /** Reviewed blind: Vision's answer is hidden on open items unless the reviewer asks. */
  blind: boolean;
  options: FeedbackOption[];
}

/** The page's two parts: mechanical fixes a rule found, and is the record's label right. */
export type FeedbackSection = "cleanup" | "validation" | (string & {});

export interface FeedbackSummary {
  type: string;
  title: string;
  about: string;
  granularity: string;
  model_flagged: boolean;
  section: FeedbackSection;
  blind: boolean;
  open: number;
  decided: number;
  /** Photo queues only: the observations their open / fully decided photos are in. */
  records_open?: number;
  records_decided?: number;
}

export type FeedbackStatus = "open" | "decided";

/** One queue item. Some fields are for the analysis only and are never shown to reviewers. */
export interface FeedbackItem {
  id: number;
  item_type: string;
  subject: string;
  granularity: string;
  rule: string;
  model_flagged: boolean;
  /** A label check still open: Vision's answer, the rule and the outcome are withheld
   *  until the reviewer decides (the server's reviewer_view). */
  masked?: boolean;
  blind?: boolean;
  hidden_until_decided?: boolean;
  /** iNat numbers ("123") or "mo:45". */
  record_keys: string[];
  /** The records' public pages, absolute. */
  links: string[];
  current: string | null;
  proposed: string | null;
  evidence: Record<string, unknown>;
  projects: string[];
  records: number;
  status: FeedbackStatus;
  seed?: string;
  /** On a masked item: this person's own "Show Vision's answer", or null. */
  revealed?: Revealed | null;
  /** Epoch seconds (or ISO). */
  created_at: number | string;
  /** The row's chip for the latest decision's Name Update on mycomap.org (lists only). */
  org_chip?: OrgChip | null;
}

/** A whole-name decision's Name Update on mycomap.org, as Vision last read it. */
export type OrgRequestStatus = "not_sent" | "not_taken" | "searching" | "waiting" | "running" | "done"
  | "needs_attention" | "refused" | "declined" | "withdrawn";

export interface OrgChip {
  text: string;
  tone: "good" | "bad" | "busy" | "quiet";
}

export interface OrgStep {
  step: "sent" | "searched" | "waiting" | "done";
  state: "done" | "current" | "todo" | "stopped";
  label: string;
}

export interface OrgRequest {
  status: OrgRequestStatus;
  request_id: number | null;
  /** Name Update on mycomap.org, opened in a new tab. */
  url: string | null;
  note: string | null;
  totals: { total?: number; fixed?: number; set_aside?: number; problems?: number; to_fix?: number };
  declined_by: string | null;
  declined_reason: string | null;
  old_names: string[];
  new_name: string | null;
  /** Epoch seconds. */
  sent_at: number | null;
  checked_at: number | null;
  steps: OrgStep[];
  chip: OrgChip;
  can_send: boolean;
  can_check: boolean;
}

/** What saving a decision would ask mycomap.org for, before it is saved. */
export type OrgPreview = { opens: false } | {
  opens: true;
  kind: string;
  old_names: string[];
  new_name: string;
  warnings: string[];
  records: Record<string, number | null>;
};

export interface FeedbackDecision {
  id: number;
  decision: string;
  value: string | null;
  note: string | null;
  photo_keys: string[];
  applies: FeedbackApplies;
  by_user: string;
  by_name: string | null;
  at: number | string;
  applied: number | boolean;
}

export type Top5 = [string, number][];

/** Vision's answer for a blind item, shown when the reviewer chose to see it ("Show Vision's
 *  answer"); `reason` says why when it is not available. */
export interface RevealedAnswer {
  available: boolean;
  top5: Top5;
  reason: string | null;
}

/** This person's own reveal of a blind item: when, and what Vision answered. */
export interface Revealed {
  at: number | string;
  answer: RevealedAnswer;
}

export interface FeedbackRefresh {
  at: number | string;
  name_now: string | null;
  top5: Top5;
  still_disagrees: boolean | number | null;
}

/** Where a sequence's BLAST link opens for this viewer, chosen by the server: the admin
 *  results form for staff, the public job page otherwise. */
export interface BlastLink {
  href: string;
  kind: "admin" | "public";
}

export interface LiveSequence {
  id: number | string;
  category_id?: number | string | null;
  trash?: boolean | number | null;
  name: string | null;
  length: number | null;
  approved?: boolean | number | null;
  flags?: unknown;
  genbank?: string | null;
  url: string | null;
  blast: { job_id?: string | number | null; url?: string | null; results_url?: string | null;
           saved_at?: string | number | null } | null;
  blast_link?: BlastLink | null;
}

/** A record as mycomap.org holds it now (or `unavailable` when it did not answer). */
export interface LiveRecord {
  record_key: string;
  unavailable?: boolean;
  name?: string | null;
  source?: string | null;
  projects?: { id: string | number; status: string | null; name: string | null }[];
  links?: { validate?: string[]; name_update?: string | null };
  sequences?: LiveSequence[];
}

export interface FeedbackItemDetail extends FeedbackItem {
  decisions: FeedbackDecision[];
  refreshes: FeedbackRefresh[];
  live?: LiveRecord[];
  org_request?: OrgRequest | null;
}

export interface RefreshedRecord {
  record_key: string;
  name_now: string | null;
  vision: { available: boolean; reason?: string; model?: string; top5?: Top5 };
  still_disagrees: boolean | null;
}

export interface FeedbackDecisionBody {
  decision: string;
  value?: string;
  note?: string;
  photo_keys?: string[];
}

// --- The photo review: the "Photos that show no specimen" queue, a record at a time ---------

export type PhotoReviewTab = "micrograph" | "slip" | "habitat" | "people" | "gate" | "other" | "all";
export type PhotoReviewSort = "p" | "p_asc" | "flagged";

/** Why a rule flagged a photo: its tab, the probe or view, its p, and how strongly it counts. */
export interface PhotoFlag {
  tab: string;
  kind: string;
  p: number | null;
  strength: number;
}

/** One photo of a record, as this reviewer may see it: by its public URL, Vision's own copy
 *  (reviewers of the item only, `private`), or a placeholder (`hidden`, no URL at all). */
export interface ReviewPhoto {
  photo_key: string;
  position: number | null;
  license_class?: string;
  shown: "public" | "private" | "hidden";
  thumb?: string;
  full?: string;
  hidden_reason?: string;
  flagged: boolean;
  /** Flagged for the tab being shown (every flagged photo in "all"). */
  in_tab: boolean;
  item_id: number | null;
  flags: PhotoFlag[];
  /** The reason a flagged photo's exclusion starts at (its flag for the open tab, else its
   *  strongest); null for an unflagged photo. */
  suggested_reason: string | null;
  status: FeedbackStatus | null;
  decision: { decision: string; reason: string | null; by: string | null } | null;
}

export interface ReviewRecord {
  record_key: string;
  label: string;
  /** The record's public page. */
  link: string;
  tab: PhotoReviewTab;
  strongest: number;
  flagged_in_tab: number;
  photos: ReviewPhoto[];
}

export interface PhotoReviewTabCount {
  code: Exclude<PhotoReviewTab, "all">;
  title: string;
  open: number;
  decided: number;
}

export interface PhotoReviewPage {
  tabs: PhotoReviewTabCount[];
  total: number;
  reasons: { code: string; label: string }[];
  records: ReviewRecord[];
}

export interface PhotoReviewDecisionBody {
  photos: { photo_key: string; decision: "keep" | "exclude"; reason?: string }[];
  note?: string;
}

export interface PhotoReviewDecided {
  decided: number;
  added: number;
  record: ReviewRecord | null;
}

// --- The label review: the blind "Photos unlike the label" queue, a record at a time -------

export type LabelReviewOutcome = "exclude_photos" | "hard_example" | "rename";

/** One photo of a label-review card, shown by the same rule as the photo review's. */
export interface LabelReviewPhoto {
  photo_key: string;
  position: number | null;
  license_class?: string;
  shown: "public" | "private" | "hidden";
  thumb?: string;
  full?: string;
  hidden_reason?: string;
  /** Left out of Vision by the stored decision (always false on an open card). */
  excluded: boolean;
}

export interface LabelReviewDecision {
  decision: string;
  value: string | null;
  photo_keys: string[];
  note: string | null;
  by: string | null;
  at: number | string | null;
}

/** One record of the queue. An open card is masked: the record, its label and its photos
 *  only. `rule`, `evidence` and `decision` come only on a decided card. */
export interface LabelReviewCard {
  item_id: number;
  record_key: string;
  label: string;
  link: string;
  status: FeedbackStatus;
  masked: boolean;
  rule?: string | null;
  evidence?: Record<string, unknown>;
  decision?: LabelReviewDecision | null;
  /** On a masked card: this person's own "Show Vision's answer", or null. */
  revealed?: Revealed | null;
  photos: LabelReviewPhoto[];
}

export interface LabelReviewPage {
  total: number;
  reasons: { code: string; label: string }[];
  outcomes: { code: string; label: string; applies: FeedbackApplies; needs_value: boolean }[];
  records: LabelReviewCard[];
}

export interface LabelReviewDecisionBody {
  outcome: LabelReviewOutcome;
  photos: { photo_key: string; decision: "keep" | "exclude" }[];
  reason?: string;
  value?: string;
  note?: string;
}

/** The admins' change list, a TSV download (pending changes unless `all`). */
export const feedbackChangesUrl = (applies: "org" | "vision", all = false) =>
  `/api/feedback/changes.tsv?applies=${applies}&all=${all}`;

const fbItem = (id: number) => `/api/feedback/items/${encodeURIComponent(String(id))}`;

export const feedbackApi = {
  me: () => getJson<FeedbackMe>("/api/feedback/me"),
  catalogue: () => getJson<FeedbackType[]>("/api/feedback/catalogue"),
  summary: () => getJson<FeedbackSummary[]>("/api/feedback/summary"),
  items: (p: { type: string; status: FeedbackStatus; q?: string; offset?: number; limit?: number }) => {
    const qs = new URLSearchParams({ type: p.type, status: p.status, q: p.q ?? "",
                                     offset: String(p.offset ?? 0), limit: String(p.limit ?? 25) });
    return getJson<{ total: number; items: FeedbackItem[] }>(`/api/feedback/items?${qs}`);
  },
  item: (id: number) => getJson<FeedbackItemDetail>(fbItem(id)),
  decide: (id: number, body: FeedbackDecisionBody) =>
    postJson<FeedbackItemDetail>(`${fbItem(id)}/decision`, body),
  reopen: (id: number) => postJson<FeedbackItemDetail>(`${fbItem(id)}/reopen`),
  orgPreview: (id: number, body: { decision: string; value?: string }) =>
    postJson<OrgPreview>(`${fbItem(id)}/org-preview`, body),
  sendAgain: (id: number) => postJson<FeedbackItemDetail>(`${fbItem(id)}/org-request`),
  /** "Show Vision's answer" on a masked item (409 on anything else). Recorded server-side: a
   *  decision made after it is kept out of the blind check. */
  reveal: (id: number) => postJson<Revealed>(`${fbItem(id)}/reveal`),
  refresh: (id: number) =>
    postJson<{ item: FeedbackItemDetail; refreshed: RefreshedRecord[] }>(`${fbItem(id)}/refresh`),
  photoReview: (p: { tab: PhotoReviewTab; status: FeedbackStatus; sort?: PhotoReviewSort;
                     min_flagged?: number; q?: string; offset?: number; limit?: number }) => {
    const qs = new URLSearchParams({ tab: p.tab, status: p.status, sort: p.sort ?? "p", q: p.q ?? "",
                                     min_flagged: String(p.min_flagged ?? 1),
                                     offset: String(p.offset ?? 0), limit: String(p.limit ?? 10) });
    return getJson<PhotoReviewPage>(`/api/feedback/photo-review?${qs}`);
  },
  photoReviewDecide: (recordKey: string, body: PhotoReviewDecisionBody) =>
    postJson<PhotoReviewDecided>(
      `/api/feedback/photo-review/${encodeURIComponent(recordKey)}/decision`, body),
  labelReview: (p: { status: FeedbackStatus; q?: string; offset?: number; limit?: number }) => {
    const qs = new URLSearchParams({ status: p.status, q: p.q ?? "", offset: String(p.offset ?? 0),
                                     limit: String(p.limit ?? 10) });
    return getJson<LabelReviewPage>(`/api/feedback/label-review?${qs}`);
  },
  labelReviewDecide: (itemId: number, body: LabelReviewDecisionBody) =>
    postJson<{ record: LabelReviewCard | null }>(
      `/api/feedback/label-review/${encodeURIComponent(String(itemId))}/decision`, body),
};

/** mycomap.org signs people in here; come back to the page they were on. */
export const signInUrl = () =>
  `/auth/signin?returnTo=${encodeURIComponent(location.pathname + location.search)}`;

export const api = {
  me: () => getJson<Me>("/api/me"),
  /** Where to go next: an open page, because a gated one would sign you straight back in. */
  async signOut(): Promise<string> {
    const res = await fetch("/auth/signout", { method: "POST" });
    const body = await res.json().catch(() => ({}));
    return typeof body.next === "string" ? body.next : "/";
  },
  prospective: () => getJson<{ models: ProspectiveModel[]; sealed_after?: string | null }>("/api/prospective"),
  /** Held-out benchmark results imported on the server (aggregates only). */
  benchmarks: () => getJson<{ benchmarks: PublishedBenchmark[] }>("/api/benchmarks"),
  /** The experiment registry, for signed-in members only. */
  /** `skipped`: entries left out because they break the registry's rules (the server logs which). */
  experiments: () => getJson<{ experiments: ExperimentSummary[]; skipped?: number }>("/api/experiments"),
  experiment: (slug: string) => getJson<Experiment>(`/api/experiments/${encodeURIComponent(slug)}`),
  models: () => getJson<ModelsInfo>("/api/models"),
  stats: () => getJson<Stats>("/api/stats"),
  scoreboard: () => getJson<{ runs: ScoreRun[] }>("/api/scoreboard"),
  run: (id: number) => getJson<RunReport>(`/api/scoreboard/${id}`),
  /** `onWait(attempt, seconds)`: the server was busy and the photos wait in line.
   *  `onSent(fraction)`: how much of the photos has been sent, 0 to 1, for each try. */
  async identify(photos: File[], models: string[], where: Where = {},
                 onWait?: (attempt: number, seconds: number) => void,
                 onSent?: (fraction: number) => void):
      Promise<{ results: IdentifyResult[]; context_used: ContextUsed }> {
    const form = new FormData();
    photos.forEach((p) => form.append("photos", p));
    form.append("models", models.join(","));
    if (where.lat?.trim() && where.lng?.trim()) {
      form.append("lat", where.lat.trim());
      form.append("lng", where.lng.trim());
    }
    if (where.observedOn) form.append("observed_on", where.observedOn);
    const res = await sendWaitingInLine(
      () => postWithProgress("/api/identify", form, onSent), onWait);
    if (!res.ok) throw new ApiError(res.status, await errorText(res));
    return res.json();
  },
  /** An iNat or MO observation's own photos, place and date, read by Vision: the answer
   *  kept from an earlier run (here, mycomap.org or the app) unless `rerun`. */
  async identifyObservation(source: "inat" | "mo", id: string, models: string[],
                            onWait?: (attempt: number, seconds: number) => void, rerun = false):
      Promise<{ results: IdentifyResult[]; context_used: ContextUsed; observation: LinkedObservation;
                cached?: boolean; answered?: Answered }> {
    const res = await sendWaitingInLine(() => fetch("/api/identify/observation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source, id, models: models.join(","), ...(rerun ? { rerun: true } : {}) }),
    }), onWait);
    if (!res.ok) throw new ApiError(res.status, await errorText(res));
    return res.json();
  },
};

/** The observation a link pointed at (POST /api/identify/observation). */
export interface LinkedObservation {
  source: "inat" | "mo";
  id: string;
  url: string | null;
  /** Its photos at the source, and how many Vision read (the first ten it could). */
  photos: number;
  photos_used: number;
  name_at_source: string | null;
  rank_at_source: string | null;
  /** One per photo scored, in per_photo's order: its URL when the page may show it
   *  (Creative Commons or public domain), else null. */
  photos_shown: (string | null)[];
  /** It is one of Vision's DNA reference records, left out of its own answer. */
  excluded_reference: boolean;
  left_out_photos: number;
}

export const pct = (v: number | null | undefined, digits = 0) =>
  v == null ? "–" : `${(100 * v).toFixed(digits)}%`;

export const num = (v: number | null | undefined) => (v == null ? "–" : v.toLocaleString());

/** "backbone/method" key used by the API for one model. */
export const modelKey = (backbone: string, method: string) => `${backbone}/${method}`;
