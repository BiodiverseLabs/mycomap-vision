// The site's menu and its Research section (Steve, 2026-10-09): the public sees the tool and
// how to help first; results, benchmarks, protocols and data sit under Research.

import type { Me } from "./api";
import { showsMembersPages } from "./publicView";
import { V1, utcDay, utcMinute } from "./datasetReleases";
import { isRunPath } from "./runLink";
import { FEEDBACK_PATH } from "./feedback";

export interface NavItem { href: string; label: string; blurb?: string }

/** The developer API's page (it was /api-docs, which still leads here). */
export const API_PATH = "/research/api";

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
  { href: "/research/for-researchers", label: "For researchers",
    blurb: "How to cite, and how to test your own model against ours." },
  // Last (Steve, 2026-10-12): the developer API was linked only from the footer.
  { href: API_PATH, label: "API",
    blurb: "Identify fungi from your own app or site: access, requests and the answer, field by field." },
];

/** The Research menu for this visitor: experiments only for signed-in members. */
export function researchNav(me: Me | null | undefined): NavItem[] {
  return RESEARCH_ITEMS.filter((i) => !i.members || showsMembersPages(me));
}

/** Pages that moved under Research keep working at their old address. The paper draft's
 *  page was taken off the site (Steve, 2026-10-12; /api/paper still serves members), so its
 *  addresses lead to Research. */
export const MOVED: Record<string, string> = {
  "/models": "/research/results",
  "/data": "/research/data",
  "/paper": "/research",
  "/research/paper": "/research",
  "/api-docs": API_PATH,
};

/** Pages a signed-out visitor may read when the site needs sign-in (signin.PUBLIC_PAGES, and
 *  signin.RESEARCH_PAGES: the old addresses of Results and Reference data), and How it works
 *  (Steve, 2026-10-22: the site's open explainer; the server already sends it, a share page). */
export const PUBLIC_PAGES = [API_PATH, "/api-docs", "/models", "/data", "/about"];

export const isResearchPath = (path: string) => path === "/research" || path.startsWith("/research/");

/** Whether this visitor must sign in before seeing `path`: only when every page needs it
 * (signin "all"), they aren't signed in, and it isn't one of PUBLIC_PAGES, the Research
 * section and Human Feedback (open to anyone, read-only for Feedback, Steve, 2026-10-22:
 * signin.research_path, signin.feedback_read_path; Experiments then asks
 * them to sign in to read the entries) or a run's own link (/r/<id>, open to whoever has
 * it: signin.run_path). The server already sends them to sign in on a full page load; this
 * covers moving on inside the site. Never for the sign-in pages themselves, or a page shown
 * at one would loop. */
export function mustSignIn(me: Me | null | undefined, path: string): boolean {
  return !!me && me.signin === "all" && !me.user && !PUBLIC_PAGES.includes(path)
    && !isResearchPath(path) && path !== FEEDBACK_PATH && !isRunPath(path) && !path.startsWith("/auth/");
}

/** "history": measured before dataset release v1 and kept as development history. */
export type BenchmarkStatus = "running" | "partial" | "planned" | "history";

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
  { slug: "newest-weeks", name: "Newest weeks: the dataset release v1 validation split", status: "running",
    summary: "Identify the newest validated records from the records validated before them, on a frozen dataset release.",
    whyItMatters: "It is how a new find meets the identifier: the answer can only come from records verified earlier. Every setting is chosen here, never on the sealed test, and anyone can rebuild the numbers from the release.",
    records: `Dataset release v1's validation split: the records first validated in the eight weeks after ${utcDay(V1.valCutoff)}, identified against the records validated before. Earlier runs on the newest four weeks of the live data are development history.`,
    scoring: "Top 1, 3, 5 and 10 at species, genus and family, on all photos of a find and on the first photo alone; species by how many reference records the true species has.",
    results: { href: "/research/results", label: "Results: dataset release v1" },
    elsewhere: "Chronological shift in FungiTastic." },
  { slug: "sealed-test", name: "Sealed test (v1-test)", status: "planned",
    summary: "Fresh DNA-verified records, validated after release v1 was cut, scored once after every choice is frozen.",
    whyItMatters: "The paper's numbers. Nothing is tuned on these records and no model output exists for them until the analysis plan is frozen, so they measure the identifier as a new find would meet it.",
    records: `North American records first validated after dataset release v1's label snapshot (${utcMinute(V1.labelSnapshot)}), kept apart from v1 by record and by photo, and never trained on. The set closes once it holds at least 3,000 records (expected around 26 October 2026) and is frozen around 30 October; no numbers are published before it is opened.`,
    scoring: "The standard summary with 95% intervals clustered by observer; iNaturalist's computer vision on the same photos and the same public place; tests declared before the records are opened.",
    elsewhere: "The sealed test sets of FungiCLEF." },
  { slug: "held-out", name: "Held-out records (development history)", status: "history",
    summary: "DNA-verified records that, before dataset release v1, no model had seen in training or as a reference.",
    whyItMatters: "The first large test with 95% intervals and paired tests between approaches. Its results guided the choices that dataset release v1 now re-checks.",
    records: "13,145 North American records validated on MycoMap that had not reached the identifier's data; 3,000 for development, 10,145 for testing. A development benchmark, never the paper's test set: these records are part of dataset release v1 (5,679 in its training split, 7,381 in its validation split, 85 left out by its rules), so they no longer form a separate held-out set: the training-split ones are trained on, and the validation-split ones are scored as part of v1's validation split.",
    scoring: "The standard summary (see Evaluation protocols), names compared strictly, sensu lato and as species complexes; iNaturalist's computer vision on part of the records.",
    results: { href: "/research/results", label: "Results: development history" },
    elsewhere: "The closed-set test in Danish Fungi 2020 and FungiTastic." },
  { slug: "advance-predictions", name: "Advance predictions", status: "running",
    summary: "Records with a sequence but no validation yet are identified now and scored when their DNA name arrives.",
    whyItMatters: "Nothing could have seen the answer: the prediction is saved before the name exists. A static benchmark can't offer this.",
    records: "Up to 300 records awaiting validation every night, scored as they turn green.",
    scoring: "Top-1 at species, genus and family, and how the stated confidence compares with how often it was right.",
    results: { href: "/research/results", label: "Results: advance predictions" } },
  { slug: "answered-before-verdict", name: "Answered before the DNA verdict", status: "running",
    summary: "Vision's answers about an observation, given before its DNA verdict (when mycomap.org auto-validates a project, or someone identifies it from a link), scored once the record turns green.",
    whyItMatters: "Every answer is kept as it was given, so the score is on records no model had trained on, and it grows with every project auto-validated.",
    records: "The first answer per observation and model, given before the record was first validated. Left out: answers given on or after the verdict, records Vision held in its reference (or a twin of one), and records validated with Vision's help (mycomap.org's DNA + MycoMap Vision rule), which would be circular.",
    scoring: "The standard summary: top 1/3/5/10 for species strict, s.l. and complex (beta), and genus; species by the true species' reference records.",
    results: { href: "/research/results", label: "Results: answered before the DNA verdict" } },
  { slug: "rare-species", name: "Rare species", status: "running",
    summary: "Accuracy by how many DNA-verified records the true species has: 0, 1-4, 5-19, 20-99, 100+.",
    whyItMatters: "Almost half the names have a single record. Accuracy depends more on this than on the method, and every sequenced record moves a species up a band.",
    records: "Every benchmark above, broken down by the true species' reference records.",
    scoring: "Species top-1 and top-5 per band, with the number of finds in each.",
    results: { href: "/research/results", label: "Results: by reference depth" },
    elsewhere: "Few-shot learning in FungiTastic and FungiCLEF 2025." },
  { slug: "unknown-species", name: "Unknown species", status: "planned",
    summary: "How the identifier behaves when the species is not in its references at all.",
    whyItMatters: "In dataset release v1's validation split, 666 finds belong to a species with no record in the training split. The identifier should say so rather than name a look-alike with confidence.",
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
  running: "Running", partial: "Partly measured", planned: "Planned", history: "Development history",
};
