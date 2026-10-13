// The experiment registry as the Research > Experiments pages show it (experiments/ in the
// private research repo, served to signed-in members by /api/experiments).

export const EXPERIMENT_STATUSES = ["planned", "running", "done", "adopted", "dropped"] as const;
export type ExperimentStatus = (typeof EXPERIMENT_STATUSES)[number];

export interface ExperimentSummary {
  title: string;
  slug: string;
  date: string;
  status: ExperimentStatus;
  /** "exploratory-pre-freeze", or "reproduced-on-<release>" (experiments/README.md in the private research repo). */
  reproducibility: string;
  question: string;
  headline: string;
  verdict: string;
  decision: string;
  branch?: string;
  commits?: string[];
  benchmark?: string;
  split?: string;
  model?: string;
  methods?: string[];
  related?: string[];
  dataset_release?: string;
  reference_hash?: string;
  code_commit?: string;
  reproduce_command?: string;
  /** On an exploratory entry that was re-run on a release: the re-run's slug, and (filled by
   *  the server from that entry) its release. */
  reproduced_by?: string;
  reproduced_on?: string;
}

export interface Experiment extends ExperimentSummary { markdown: string }

export const STATUS_WORDS: Record<ExperimentStatus, { label: string; tone: string }> = {
  planned: { label: "Planned", tone: "bg-gray-100 text-gray-700" },
  running: { label: "Running", tone: "bg-[#e8f1fb] text-[#1c5cab]" },
  done: { label: "Done", tone: "bg-[#f3eee6] text-[#7a5a3a]" },
  adopted: { label: "Adopted", tone: "bg-myco-green/10 text-myco-green" },
  dropped: { label: "Dropped", tone: "bg-gray-100 text-gray-500 line-through decoration-1" },
};

/** Newest first; one status and/or one reproducibility kind, or all when null. */
export function filterExperiments(list: ExperimentSummary[], status: ExperimentStatus | null,
                                  repro: ReproKind | null = null): ExperimentSummary[] {
  return list.filter((e) => (!status || e.status === status) && (!repro || reproKind(e) === repro))
    .sort((a, b) => b.date.localeCompare(a.date) || a.title.localeCompare(b.title));
}

/** How many entries have each status, for the filter's labels. */
export function statusCounts(list: ExperimentSummary[]): Record<ExperimentStatus, number> {
  const out = Object.fromEntries(EXPERIMENT_STATUSES.map((s) => [s, 0])) as Record<ExperimentStatus, number>;
  for (const e of list) if (e.status in out) out[e.status] += 1;
  return out;
}

/** How far a result can be rebuilt: before the dataset freeze every result is exploratory and
 *  its decision provisional; after it, a result names the release it was reproduced on. An
 *  exploratory entry that has been re-run on a release is method history: its own numbers stay
 *  exploratory, but nothing provisional hangs on them any more. */
export function reproducibilityWords(r: string, reproducedOn?: string):
    { label: string; note: string; provisional: boolean } {
  const m = /^reproduced-on-(.+)$/.exec(r);
  if (m) {
    return { label: `Reproducible on release ${m[1]}`, provisional: false,
             note: `Rebuilt from dataset release ${m[1]}: the release, code and command below give the same numbers.` };
  }
  if (reproducedOn) {
    return { label: `Method history · re-run on release ${reproducedOn}`, provisional: false,
             note: `Run on the live data before the dataset freeze. It has been re-run on dataset release ${reproducedOn}; `
               + "the re-run's numbers are the ones that count, and this entry stays as the record of how we got there." };
  }
  return { label: "Exploratory (before the dataset freeze)", provisional: true,
           note: "Run on the live data before the dataset freeze, so it can't be rebuilt exactly. "
             + "The result and any decision taken from it are provisional until re-run on dataset release v1." };
}

/** Where an entry stands on reproducibility, for the list's second filter. */
export const REPRO_KINDS = ["release", "history", "exploratory"] as const;
export type ReproKind = (typeof REPRO_KINDS)[number];
export const REPRO_KIND_WORDS: Record<ReproKind, string> = {
  release: "On a dataset release", history: "Method history", exploratory: "Exploratory",
};

export function reproKind(e: ExperimentSummary): ReproKind {
  if (/^reproduced-on-/.test(e.reproducibility)) return "release";
  return e.reproduced_on ? "history" : "exploratory";
}

export function reproCounts(list: ExperimentSummary[]): Record<ReproKind, number> {
  const out = { release: 0, history: 0, exploratory: 0 };
  for (const e of list) out[reproKind(e)] += 1;
  return out;
}

/** "pending" decisions are the ones waiting on Steve. */
export const awaitsDecision = (e: ExperimentSummary) => e.decision.trim().toLowerCase().startsWith("pending");
