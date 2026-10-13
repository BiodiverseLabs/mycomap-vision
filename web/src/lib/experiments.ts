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

/** Newest first; one status and/or one dataset, or all when null. */
export function filterExperiments(list: ExperimentSummary[], status: ExperimentStatus | null,
                                  dataset: string | null = null): ExperimentSummary[] {
  return list.filter((e) => (!status || e.status === status) && (!dataset || datasetOf(e) === dataset))
    .sort((a, b) => b.date.localeCompare(a.date) || a.title.localeCompare(b.title));
}

/** How many entries have each status, for the filter's labels. */
export function statusCounts(list: ExperimentSummary[]): Record<ExperimentStatus, number> {
  const out = Object.fromEntries(EXPERIMENT_STATUSES.map((s) => [s, 0])) as Record<ExperimentStatus, number>;
  for (const e of list) if (e.status in out) out[e.status] += 1;
  return out;
}

/** The dataset an entry's numbers come from: a frozen release ("v1"), or the exploratory data
 *  used before release v1 was frozen ("pre-v1"). The page shows one dataset at a time, v1 first;
 *  pre-v1 entries stay as the record of how we got there. */
export const PRE_V1 = "pre-v1";

export function datasetOf(e: ExperimentSummary): string {
  const m = /^reproduced-on-(.+)$/.exec(e.reproducibility);
  return m ? m[1] : PRE_V1;
}

/** Datasets present, newest release first, pre-v1 last; each with its entry count. */
export function datasetChoices(list: ExperimentSummary[]): { id: string; label: string; n: number }[] {
  const counts = new Map<string, number>();
  for (const e of list) counts.set(datasetOf(e), (counts.get(datasetOf(e)) ?? 0) + 1);
  const releases = [...counts.keys()].filter((d) => d !== PRE_V1)
    .sort((a, b) => b.localeCompare(a, undefined, { numeric: true }));
  const out = releases.map((id) => ({ id, label: `Dataset release ${id}`, n: counts.get(id)! }));
  if (counts.has(PRE_V1)) out.push({ id: PRE_V1, label: "Pre-v1 exploratory dataset", n: counts.get(PRE_V1)! });
  return out;
}

/** The dataset shown first: the newest release with entries, else pre-v1. */
export const defaultDataset = (list: ExperimentSummary[]) => datasetChoices(list)[0]?.id ?? PRE_V1;

/** The label on an entry: which dataset its numbers come from. */
export function reproducibilityWords(r: string, reproducedOn?: string):
    { label: string; note: string; provisional: boolean } {
  const m = /^reproduced-on-(.+)$/.exec(r);
  if (m) {
    return { label: `Dataset release ${m[1]}`, provisional: false,
             note: `Rebuilt from dataset release ${m[1]}: the release, code and command below give the same numbers.` };
  }
  const base = "Run on the exploratory data used before dataset release v1 was frozen, so its numbers "
    + "can't be rebuilt exactly. It stays as the record of how we got here.";
  if (reproducedOn) {
    return { label: `Pre-v1 exploratory dataset · re-run on ${reproducedOn}`, provisional: true,
             note: `${base} Its re-run on dataset release ${reproducedOn} holds the numbers that count.` };
  }
  return { label: "Pre-v1 exploratory dataset", provisional: true,
           note: `${base} Decisions taken from it are re-checked on dataset release v1.` };
}

/** "pending" decisions are the ones waiting on Steve. */
export const awaitsDecision = (e: ExperimentSummary) => e.decision.trim().toLowerCase().startsWith("pending");
