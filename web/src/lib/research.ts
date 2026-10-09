// The site's menu and its Research section (Steve, 2026-10-09): the public sees the tool and
// how to help first; results, benchmarks, protocols, data and the paper sit under Research.

import type { Me } from "./api";
import { showsPaper } from "./publicView";

export interface NavItem { href: string; label: string; blurb?: string }

export const PUBLIC_NAV: NavItem[] = [
  { href: "/", label: "Identify" },
  { href: "/about", label: "How it works" },
  { href: "/get-involved", label: "Get involved" },
];

const RESEARCH_ITEMS: (NavItem & { members?: boolean })[] = [
  { href: "/research/results", label: "Results",
    blurb: "Every approach, ours and outside models, scored on the same DNA-verified records." },
  { href: "/research/benchmarks", label: "Benchmarks",
    blurb: "The named test sets: what each one measures and how." },
  { href: "/research/protocols", label: "Evaluation protocols",
    blurb: "How every result is scored, so numbers can be compared." },
  { href: "/research/data", label: "Reference data",
    blurb: "The DNA-verified records and photos the identifier learns from." },
  { href: "/research/experiments", label: "Experiments", members: true,
    blurb: "Every experiment: the question, everything tried, the numbers, the decision (members)." },
  { href: "/research/paper", label: "Paper in progress", members: true,
    blurb: "The living draft of the Mycologia paper (mycomap.org members)." },
  { href: "/research/for-researchers", label: "For researchers",
    blurb: "How to cite, and how to test your own model against ours." },
];

/** The Research menu for this visitor: the paper only for signed-in members. */
export function researchNav(me: Me | null | undefined): NavItem[] {
  return RESEARCH_ITEMS.filter((i) => !i.members || showsPaper(me));
}

/** Pages that moved under Research keep working at their old address. */
export const MOVED: Record<string, string> = {
  "/models": "/research/results",
  "/data": "/research/data",
  "/paper": "/research/paper",
};

export const isResearchPath = (path: string) => path === "/research" || path.startsWith("/research/");

export type BenchmarkStatus = "running" | "partial" | "planned";

export interface Benchmark {
  slug: string;
  name: string;
  status: BenchmarkStatus;
  /** One line for the list. */
  summary: string;
  whyItMatters: string;
  records: string;
  scoring: string;
  /** Where its results are shown, when they are. */
  results?: { href: string; label: string };
  /** The equivalent task in other fungal benchmarks, when there is one. */
  elsewhere?: string;
}

export const BENCHMARKS: Benchmark[] = [
  { slug: "newest-weeks", name: "Newest weeks of validations", status: "running",
    summary: "Identify the records validated in the last four weeks from the records validated before them.",
    whyItMatters: "It is how a new find meets the identifier: the answer can only come from records verified earlier. Run again with every new model, so results over time stay comparable.",
    records: "The newest 28 days of DNA-verified records against everything validated before them (latest: 1,152 records against 152,915).",
    scoring: "Top 1, 3, 5 and 10 at species, genus and family, on all photos of a find and on the first photo alone; species by how many reference records the true species has.",
    results: { href: "/research/results", label: "Results: newest weeks" },
    elsewhere: "Chronological shift in FungiTastic." },
  { slug: "held-out", name: "Held-out records", status: "running",
    summary: "DNA-verified records no model has seen, in training or as a reference.",
    whyItMatters: "The fairest single number for how the identifier does on new records, with 95% intervals and paired tests between approaches.",
    records: "13,145 North American records validated on MycoMap that never reached the identifier's data; 3,000 for development, 10,145 for testing. A development benchmark: the paper's test set will be a fresh set of about 1,000, sealed before any model sees it.",
    scoring: "The standard summary (see Evaluation protocols), names compared strictly, sensu lato and as species complexes; iNaturalist's computer vision on the same records.",
    results: { href: "/research/results", label: "Results: held-out records" },
    elsewhere: "The closed-set test in Danish Fungi 2020 and FungiTastic." },
  { slug: "advance-predictions", name: "Advance predictions", status: "running",
    summary: "Records with a sequence but no validation yet are identified now and scored when their DNA name arrives.",
    whyItMatters: "Nothing could have seen the answer: the prediction is saved before the name exists. A static benchmark can't offer this.",
    records: "Up to 300 records awaiting validation every night, scored as they turn green.",
    scoring: "Top-1 at species, genus and family, and how the stated confidence compares with how often it was right.",
    results: { href: "/research/results", label: "Results: advance predictions" } },
  { slug: "rare-species", name: "Rare species", status: "running",
    summary: "Accuracy by how many DNA-verified records the true species has: 0, 1-4, 5-19, 20-99, 100+.",
    whyItMatters: "Almost half the names have a single record. Accuracy depends more on this than on the method, and every sequenced record moves a species up a band.",
    records: "Every benchmark above, broken down by the true species' reference records.",
    scoring: "Species top-1 and top-5 per band, with the number of finds in each.",
    results: { href: "/research/results", label: "Results: by reference depth" },
    elsewhere: "Few-shot learning in FungiTastic and FungiCLEF 2025." },
  { slug: "unknown-species", name: "Unknown species", status: "planned",
    summary: "How the identifier behaves when the species is not in its references at all.",
    whyItMatters: "About one find in eight belongs to a species with no DNA-verified record yet. The identifier should say so rather than name a look-alike with confidence.",
    records: "Finds whose true species has no reference record, against those whose species does.",
    scoring: "How well the top score separates known from unknown species (AUROC, and true negatives at 95% true positives), and how often the identifier holds back.",
    elsewhere: "Open-set classification in FungiTastic and FungiCLEF." },
  { slug: "look-alikes", name: "Look-alikes and species complexes", status: "partial",
    summary: "Credit for a near miss where the name is a matter of taxonomy rather than a wrong identification.",
    whyItMatters: "Genus splits (Cortinarius, Inocybe) and species complexes mean a strict name match can count a sound answer as wrong. Shown beside strict scores, never instead of them.",
    records: "Every benchmark above.",
    scoring: "Species strict, sensu lato and complex (beta); genus strict and sensu lato; the most confused pairs.",
    results: { href: "/research/results", label: "Results" } },
  { slug: "poisonous-mistakes", name: "Poisonous mistakes", status: "planned",
    summary: "How often a deadly species is named as something else, and the reverse.",
    whyItMatters: "Not for deciding what to eat. Measuring dangerous mistakes separately keeps them from hiding in an average.",
    records: "Finds of Amanita sect. Phalloideae, Galerina, Lepiota, Cortinarius sect. Orellani and Gyromitra, and finds named as them.",
    scoring: "Error rates in each direction, and a cost-weighted score with a dangerous miss counted 100 times a harmless one.",
    elsewhere: "Cost-sensitive classification in FungiCLEF 2023 and 2024." },
];

export const STATUS_LABEL: Record<BenchmarkStatus, string> = {
  running: "Running", partial: "Partly measured", planned: "Planned",
};
