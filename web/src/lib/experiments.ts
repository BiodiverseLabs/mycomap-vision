// The experiment registry as the Research > Experiments pages show it (docs/experiments in the
// Vision repo, served to signed-in members by /api/experiments).

export const EXPERIMENT_STATUSES = ["planned", "running", "done", "adopted", "dropped"] as const;
export type ExperimentStatus = (typeof EXPERIMENT_STATUSES)[number];

export interface ExperimentSummary {
  title: string;
  slug: string;
  date: string;
  status: ExperimentStatus;
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
}

export interface Experiment extends ExperimentSummary { markdown: string }

export const STATUS_WORDS: Record<ExperimentStatus, { label: string; tone: string }> = {
  planned: { label: "Planned", tone: "bg-gray-100 text-gray-700" },
  running: { label: "Running", tone: "bg-[#e8f1fb] text-[#1c5cab]" },
  done: { label: "Done", tone: "bg-[#f3eee6] text-[#7a5a3a]" },
  adopted: { label: "Adopted", tone: "bg-myco-green/10 text-myco-green" },
  dropped: { label: "Dropped", tone: "bg-gray-100 text-gray-500 line-through decoration-1" },
};

/** Newest first; one status, or all when `status` is null. */
export function filterExperiments(list: ExperimentSummary[],
                                  status: ExperimentStatus | null): ExperimentSummary[] {
  return list.filter((e) => !status || e.status === status)
    .sort((a, b) => b.date.localeCompare(a.date) || a.title.localeCompare(b.title));
}

/** How many entries have each status, for the filter's labels. */
export function statusCounts(list: ExperimentSummary[]): Record<ExperimentStatus, number> {
  const out = Object.fromEntries(EXPERIMENT_STATUSES.map((s) => [s, 0])) as Record<ExperimentStatus, number>;
  for (const e of list) if (e.status in out) out[e.status] += 1;
  return out;
}

/** "pending" decisions are the ones waiting on Steve. */
export const awaitsDecision = (e: ExperimentSummary) => e.decision.trim().toLowerCase().startsWith("pending");
