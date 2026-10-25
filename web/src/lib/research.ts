// The site's menu and its Research section (Steve, 2026-10-09): the public sees the tool and
// how to help first; results, benchmarks, protocols and data sit under Research.

import type { Me } from "./api";
import { showsMembersPages } from "./publicView";
import { V1, utcDay, utcMinute } from "./datasetReleases";
import { isRunPath } from "./runLink";
import { FEEDBACK_PATH } from "./feedback";

export interface NavItem { href: string; label: string; blurb?: string; members?: boolean }

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

/** The Research bar under the header (the approved redesign): the overview, then every part.
 *  A members-only part is there for everyone, with its lock, so a visitor sees it exists (its
 *  page asks them to sign in; the entries stay members only). The header's Research menu keeps
 *  to researchNav. */
export function researchBar(_me?: Me | null): NavItem[] {
  return [{ href: "/research", label: "Overview" }, ...RESEARCH_ITEMS.map(({ href, label, blurb, members }) =>
    ({ href, label, blurb, members }))];
}

/** Whether a Research bar tab is the page being read: Overview only on /research itself,
 *  every other part on its own address and the pages under it. */
export function isBarTabActive(href: string, location: string): boolean {
  return href === "/research" ? location === "/research" : location === href || location.startsWith(`${href}/`);
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
 *  signin.RESEARCH_PAGES: the old addresses of Results and Reference data), How it works
 *  (Steve, 2026-10-22: the site's open explainer; the server already sends it, a share page)
 *  and Get involved (Steve, 2026-10-22: the approved redesign, a share page too; every feed it
 *  reads answers signed out, tests/test_get_involved.py). */
export const PUBLIC_PAGES = [API_PATH, "/api-docs", "/models", "/data", "/about", "/get-involved"];

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


// --- Benchmarks ---------------------------------------------------------------------------

/** "history": measured before dataset release v1 and kept as development history. "sealed"
 *  (Steve, 2026-10-25): the paper's test, scored once after the analysis freeze. */
export type BenchmarkStatus = "running" | "partial" | "sealed" | "planned" | "history";

export const STATUS_LABEL: Record<BenchmarkStatus, string> = {
  running: "Running", partial: "Partly measured", sealed: "Sealed", planned: "Planned",
  history: "Development history",
};
/** The order the summary bar, the filters and the default sort follow. */
export const STATUS_ORDER: BenchmarkStatus[] = ["running", "partial", "sealed", "planned", "history"];

/** What a benchmark measures (the Measures filter; Steve, 2026-10-25, the mockup's values). */
export type Measure = "accuracy" | "rare" | "unknown" | "generalisation" | "leakage" | "outside"
  | "live" | "danger";
export const MEASURE_LABEL: Record<Measure, string> = {
  accuracy: "Accuracy", rare: "Rare species", unknown: "Unknown species",
  generalisation: "Generalisation", leakage: "Leakage", outside: "Outside models",
  live: "Live / prospective", danger: "Dangerous mistakes",
};
export const MEASURE_ORDER = Object.keys(MEASURE_LABEL) as Measure[];

/** Which records a benchmark is scored on (the Data filter). */
export type DataSet = "v1val" | "sealed" | "live" | "prev1";
export const DATA_LABEL: Record<DataSet, string> = {
  v1val: "v1 validation", sealed: "Sealed test", live: "Live", prev1: "Pre-v1 held-out",
};
/** The same, short, for a card's Dataset line. */
export const DATA_SHORT: Record<DataSet, string> = {
  v1val: "v1 val", sealed: "sealed", live: "live", prev1: "pre-v1",
};
export const DATA_ORDER = Object.keys(DATA_LABEL) as DataSet[];

/** Where a benchmark's headline number comes from: a scoreboard run ("served": the run
 *  /api/how-good reports, the served model's newest open run) and the metric read from it, or
 *  a live feed. Nothing here is a number: the page reads it from the API. */
export type HeadlineMetric = "species-topk" | "species-depth" | "unknown-count" | "prospective"
  | "early-answers";
export interface Headline { run?: number | "served"; metric: HeadlineMetric }

export interface ProtocolStep { title: string; text: string }
export interface BenchmarkLink { href: string; title: string; text: string; members?: boolean }

export interface Benchmark {
  slug: string;
  /** The short name on cards and the page title. */
  name: string;
  /** The full name, when the short one leaves something out. */
  longName?: string;
  status: BenchmarkStatus;
  measures: Measure[];
  data: DataSet[];
  /** One line for the list. */
  summary: string;
  whyItMatters: string;
  records: string;
  scoring: string;
  /** Its size when no run gives it (a run's own count is read from the API). */
  size?: string;
  /** When it last ran: "run" for its headline run's date, else words ("nightly", "live"). */
  lastRun?: string;
  headline?: Headline;
  /** What its card says while it has no number. */
  pending?: { lead: string; text: string };
  /** The question in one line, for the detail page's heading. */
  question: string;
  protocol: ProtocolStep[];
  /** Where its results are shown, when they are. */
  results?: { href: string; label: string };
  /** The equivalent task in other fungal benchmarks, when there is one. */
  elsewhere?: string;
  links: BenchmarkLink[];
}

const FUNGITASTIC_URL = "https://arxiv.org/abs/2408.13632";

export const BENCHMARKS: Benchmark[] = [
  { slug: "newest-weeks", name: "Newest weeks", longName: "Newest weeks: the dataset release v1 validation split",
    status: "running", measures: ["accuracy", "generalisation", "outside"], data: ["v1val"],
    headline: { run: "served", metric: "species-topk" }, lastRun: "run",
    question: "How often does Vision name a new find right, from records verified before it?",
    summary: "Identify the newest validated records from the records validated before them, on a frozen dataset release.",
    whyItMatters: "It is how a new find meets the identifier: the answer can only come from records verified earlier. Every setting is chosen here, never on the sealed test, and anyone can rebuild the numbers from the release.",
    records: `Dataset release v1's validation split: the records first validated in the eight weeks after ${utcDay(V1.valCutoff)}, identified against the records validated before. Earlier runs on the newest four weeks of the live data are development history.`,
    scoring: "Top 1, 3, 5 and 10 at species, genus and family, on all photos of a find and on the first photo alone; species by how many reference records the true species has.",
    protocol: [
      { title: "Freeze the release", text: `Dataset release ${V1.id}, hashed; its erratum lists faulty records, never edits.` },
      { title: "Split by date", text: `Validated on or before ${utcDay(V1.valCutoff)} is the reference; after it, the test.` },
      { title: "Identify each find", text: "From the earlier records only, with every photo of the find." },
      { title: "Score one summary", text: "Top 1/3/5/10 at species, genus and family, by reference depth." },
    ],
    results: { href: "/research/results", label: "Results: dataset release v1" },
    elsewhere: "Chronological shift in FungiTastic.",
    links: [
      { href: "/research/results#scoreboard", title: "Results: scoreboard", text: "Every approach on these finds" },
      { href: "/research/protocols", title: "Evaluation protocols", text: "How every number is scored" },
      { href: "/research/data", title: "Reference data", text: "The release behind it" },
      { href: FUNGITASTIC_URL, title: "FungiTastic (Picek et al. 2025)", text: "Chronological shift elsewhere" },
    ] },
  { slug: "rare-species", name: "Rare species", status: "running", measures: ["rare"], data: ["v1val", "live"],
    headline: { run: "served", metric: "species-depth" }, lastRun: "run",
    question: "How much does a species' record count decide whether Vision can name it?",
    summary: "Accuracy by how many DNA-verified records the true species has: 0, 1-4, 5-19, 20-99, 100+.",
    whyItMatters: "Almost half the names have a single record. Accuracy depends more on this than on the method, and every sequenced record moves a species up a band.",
    records: "Every benchmark above, broken down by the true species' reference records.",
    scoring: "Species top 1, 3, 5 and 10 per band, with the number of finds in each.",
    protocol: [
      { title: "Count references", text: "The true species' DNA-verified records in the reference the identifier answered from." },
      { title: "Place it in a band", text: "0 · 1-4 · 5-19 · 20-99 · 100+" },
      { title: "Score top 1, 3, 5, 10", text: "Species strict, all photos of the find." },
      { title: "Report with n", text: "Finds per band beside every rate." },
    ],
    results: { href: "/research/results#depth", label: "Results: by reference depth" },
    elsewhere: "Few-shot learning in FungiTastic and FungiCLEF 2025.",
    links: [
      { href: "/research/results#depth", title: "Results: reference depth", text: "Every approach, band by band" },
      { href: "/research/protocols", title: "Evaluation protocols", text: "How bands and top-k are scored" },
      { href: "/research/experiments/v1-learning-curve", title: "Experiment: learning curve on v1",
        text: "What one more record is worth (members)", members: true },
      { href: FUNGITASTIC_URL, title: "FungiTastic (Picek et al. 2025)", text: "Few-shot task elsewhere" },
    ] },
  { slug: "advance-predictions", name: "Advance predictions", status: "running", measures: ["live", "accuracy"],
    data: ["live"], headline: { metric: "prospective" }, size: "≤ 300 a night", lastRun: "nightly",
    question: "Is Vision right about a find before anyone knows its DNA name?",
    summary: "Records with a sequence but no validation yet are identified now and scored when their DNA name arrives.",
    whyItMatters: "Nothing could have seen the answer: the prediction is saved before the name exists. A static benchmark can't offer this.",
    records: "Up to 300 records awaiting validation every night, scored as they turn green.",
    scoring: "Top-1 at species, genus and family, and how the stated confidence compares with how often it was right.",
    protocol: [
      { title: "Pick waiting records", text: "Sequenced, not yet validated: up to 300 a night." },
      { title: "Save the answer", text: "Before the DNA name exists, with its stated confidence." },
      { title: "Wait for green", text: "The record is validated in its MycoMap project." },
      { title: "Score it", text: "Against the DNA name; the sealed test's records are counted, not scored." },
    ],
    results: { href: "/research/results#advance", label: "Results: advance predictions" },
    links: [
      { href: "/research/results#advance", title: "Results: advance predictions", text: "Every model's saved answers" },
      { href: "/research/protocols", title: "Evaluation protocols", text: "How every number is scored" },
    ] },
  { slug: "answered-before-verdict", name: "Answered before the DNA verdict", status: "running",
    measures: ["live", "accuracy"], data: ["live"], headline: { metric: "early-answers" }, lastRun: "live",
    question: "When people ask Vision before the DNA verdict, how often was it right?",
    summary: "Vision's answers about an observation, given before its DNA verdict (when mycomap.org auto-validates a project, or someone identifies it from a link), scored once the record turns green.",
    whyItMatters: "Every answer is kept as it was given, so the score is on records no model had trained on, and it grows with every project auto-validated.",
    records: "The first answer per observation and model, given before the record was first validated. Left out: answers given on or after the verdict, records Vision held in its reference (or a twin of one), and records validated with Vision's help (mycomap.org's DNA + MycoMap Vision rule), which would be circular.",
    scoring: "The standard summary: top 1/3/5/10 for species strict, s.l. and complex (beta), and genus; species by the true species' reference records.",
    protocol: [
      { title: "Keep every answer", text: "Frozen as given, insert-only, ten deep per observation." },
      { title: "Wait for the verdict", text: "The record's DNA name, marked green." },
      { title: "Leave out the circular", text: "Answers after the verdict, references and Vision-assisted validations." },
      { title: "Score the first answer", text: "Top 1/3/5/10, strict, s.l. and complex." },
    ],
    results: { href: "/research/results#early", label: "Results: answered before the DNA verdict" },
    links: [
      { href: "/research/results#early", title: "Results: before the verdict", text: "Answers kept and scored" },
      { href: "/research/protocols", title: "Evaluation protocols", text: "How every number is scored" },
    ] },
  { slug: "look-alikes", name: "Look-alikes and species complexes", status: "partial", measures: ["accuracy"],
    data: ["v1val", "live"], size: "every benchmark",
    pending: { lead: "Strict scores are published everywhere.",
               text: "Sensu lato and complex (beta) columns are scored, and appear beside strict once the standard summary carries them." },
    question: "Is a near miss a wrong identification, or a matter of taxonomy?",
    summary: "Credit for a near miss where the name is a matter of taxonomy rather than a wrong identification.",
    whyItMatters: "Genus splits (Cortinarius, Inocybe) and species complexes mean a strict name match can count a sound answer as wrong. Shown beside strict scores, never instead of them.",
    records: "Every benchmark above.",
    scoring: "Species strict, sensu lato and complex (beta); genus strict and sensu lato; the most confused pairs.",
    protocol: [
      { title: "Score strict", text: "The answer names exactly the DNA-verified species." },
      { title: "Fold the splits", text: "Cortinarius s.l., Inocybe s.l. and gender endings count as one name." },
      { title: "Group complexes", text: "Species complexes, as a beta rule." },
      { title: "Report beside strict", text: "Never instead of it." },
    ],
    results: { href: "/research/results", label: "Results" },
    links: [
      { href: "/research/protocols", title: "Evaluation protocols", text: "Strict, s.l. and complex" },
      { href: "/research/results", title: "Results", text: "Strict scores for every approach" },
    ] },
  { slug: "sealed-test", name: "Sealed test (v1-test)", status: "sealed", measures: ["accuracy", "leakage", "outside"],
    data: ["sealed"], size: "≥ 3,000 records", lastRun: "not opened",
    pending: { lead: "Sealed.",
               text: "The set closes once it holds at least 3,000 records (expected around 26 October 2026) and is frozen around 30 October. No numbers are published before it is opened." },
    question: "How good is Vision on finds nothing was ever tuned on?",
    summary: "Fresh DNA-verified records, validated after release v1 was cut, scored once after every choice is frozen.",
    whyItMatters: "The paper's numbers. Nothing is tuned on these records and no model output exists for them until the analysis plan is frozen, so they measure the identifier as a new find would meet it.",
    records: `North American records first validated after dataset release v1's label snapshot (${utcMinute(V1.labelSnapshot)}), kept apart from v1 by record and by photo, and never trained on. The set closes once it holds at least 3,000 records (expected around 26 October 2026) and is frozen around 30 October; no numbers are published before it is opened.`,
    scoring: "The standard summary with 95% intervals clustered by observer; iNaturalist's computer vision on the same photos and the same public place; tests declared before the records are opened.",
    protocol: [
      { title: "Collect", text: `Records first validated after ${utcDay(V1.labelSnapshot)}, kept apart by record and photo.` },
      { title: "Close and freeze", text: "At 3,000 records or more; the analysis plan is frozen first." },
      { title: "Open once", text: "Every model, iNaturalist's included, on the same photos and public place." },
      { title: "Report", text: "The standard summary with observer-clustered 95% intervals." },
    ],
    elsewhere: "The sealed test sets of FungiCLEF.",
    links: [
      { href: "/research/protocols", title: "Evaluation protocols", text: "How the sealed test is kept clean" },
      { href: "/research/benchmarks/newest-weeks", title: "Newest weeks", text: "Where every setting is chosen" },
    ] },
  { slug: "unknown-species", name: "Unknown species", status: "planned", measures: ["unknown"], data: ["v1val"],
    headline: { run: "served", metric: "unknown-count" },
    pending: { lead: "Planned.",
               text: "How well the top score separates known from unknown species (AUROC), and how often Vision holds back." },
    question: "Does Vision say so when a find's species is not in its references at all?",
    summary: "How the identifier behaves when the species is not in its references at all.",
    whyItMatters: "Some finds in dataset release v1's validation split belong to a species with no record in the training split (the count is on this page, from the latest run). The identifier should say so rather than name a look-alike with confidence.",
    records: "Finds whose true species has no reference record, against those whose species does.",
    scoring: "How well the top score separates known from unknown species (AUROC, and true negatives at 95% true positives), and how often the identifier holds back.",
    protocol: [
      { title: "Find the unknowns", text: "Test finds whose species has no reference record." },
      { title: "Read the top score", text: "For every find, known and unknown alike." },
      { title: "Separate them", text: "AUROC, and true negatives at 95% true positives." },
      { title: "Count hold-backs", text: "How often the identifier declines to name a species." },
    ],
    elsewhere: "Open-set classification in FungiTastic and FungiCLEF.",
    links: [
      { href: "/research/benchmarks/rare-species", title: "Rare species", text: "The 0-record band, scored" },
      { href: FUNGITASTIC_URL, title: "FungiTastic (Picek et al. 2025)", text: "Open-set task elsewhere" },
    ] },
  { slug: "poisonous-mistakes", name: "Poisonous mistakes", status: "planned", measures: ["danger"], data: ["v1val"],
    size: "5 groups",
    pending: { lead: "Not for deciding what to eat.",
               text: "Error rates both ways for Amanita sect. Phalloideae, Galerina, Lepiota, Cortinarius sect. Orellani and Gyromitra, with a dangerous miss counted 100 times a harmless one." },
    question: "How often is a deadly species named as something else, and the reverse?",
    summary: "How often a deadly species is named as something else, and the reverse.",
    whyItMatters: "Not for deciding what to eat. Measuring dangerous mistakes separately keeps them from hiding in an average.",
    records: "Finds of Amanita sect. Phalloideae, Galerina, Lepiota, Cortinarius sect. Orellani and Gyromitra, and finds named as them.",
    scoring: "Error rates in each direction, and a cost-weighted score with a dangerous miss counted 100 times a harmless one.",
    protocol: [
      { title: "Mark the groups", text: "Five deadly groups, by their DNA names." },
      { title: "Both directions", text: "Deadly named as harmless, and harmless named as deadly." },
      { title: "Weigh the cost", text: "A dangerous miss counts 100 times a harmless one." },
      { title: "Report apart", text: "Never folded into an average." },
    ],
    elsewhere: "Cost-sensitive classification in FungiCLEF 2023 and 2024.",
    links: [
      { href: "/research/protocols", title: "Evaluation protocols", text: "How every number is scored" },
    ] },
  { slug: "held-out", name: "Held-out records", longName: "Held-out records (development history)",
    status: "history", measures: ["accuracy", "leakage"], data: ["prev1"], size: "13,145 records",
    lastRun: "before v1",
    pending: { lead: "Development history.",
               text: "These records are now part of release v1: 5,679 in its training split, 7,381 in its validation split, 85 left out by its rules. Its results guided the choices v1 re-checks." },
    question: "Before release v1, how did Vision do on records no model had seen?",
    summary: "DNA-verified records that, before dataset release v1, no model had seen in training or as a reference.",
    whyItMatters: "The first large test with 95% intervals and paired tests between approaches. Its results guided the choices that dataset release v1 now re-checks.",
    records: "13,145 North American records validated on MycoMap that had not reached the identifier's data; 3,000 for development, 10,145 for testing. A development benchmark, never the paper's test set: these records are part of dataset release v1 (5,679 in its training split, 7,381 in its validation split, 85 left out by its rules), so they no longer form a separate held-out set: the training-split ones are trained on, and the validation-split ones are scored as part of v1's validation split.",
    scoring: "The standard summary (see Evaluation protocols), names compared strictly, sensu lato and as species complexes; iNaturalist's computer vision on part of the records.",
    protocol: [
      { title: "Hold out", text: "13,145 records no model had seen, listed before any model ran." },
      { title: "Split", text: "3,000 for development, 10,145 for testing." },
      { title: "Score", text: "The standard summary, with 95% intervals and paired tests." },
      { title: "Fold into v1", text: "Now part of release v1's training and validation splits." },
    ],
    results: { href: "/research/results", label: "Results: development history" },
    elsewhere: "The closed-set test in Danish Fungi 2020 and FungiTastic.",
    links: [
      { href: "/research/results#scoreboard", title: "Results: development history", text: "Earlier tests, kept as they were" },
      { href: "/research/benchmarks/newest-weeks", title: "Newest weeks", text: "Where v1 re-checks its choices" },
    ] },
];

// --- Benchmarks summary: the base-model screen and the fungi-specific models ---------------

/** One row of a top-1/3/5/10 table quoted from a committed research entry (percent). `run`
 *  says when a row comes from a different run than the rest of its table. */
export interface ScreenRow { name: string; detail: string; top1: number; top3: number; top5: number; top10: number;
                             run?: string }

/** The base-model screen (Steve, 2026-10-25: "Development screen", top of the Benchmarks
 *  summary). Frozen base models, nearest + species average, photos only, on the 664 deciding
 *  queries of a seeded v1 val subset (2,096 species). Copied from the research repo's
 *  experiments/2026-10-13-v1-backbone-screen.md at origin/main ef3a8df: its 25 Oct addendum
 *  (comparison 20261025-163049-1a3990), one run for every row it lists, with FungiTastic BEiT-B
 *  (frozen) beside BioCLIP 2, BioCLIP and DINOv3-L 512. SigLIP 2 L 384 is not in the
 *  addendum's table: its row is the original 13 Oct run's (the re-run moved the rows it lists
 *  by at most 0.2 points) and says so. The screen's fine-tuned row (the 224 px v0 fine-tune) is
 *  left out: this table is base models only. Data-driven: a row is added only once committed. */
export const BASE_MODEL_SCREEN = {
  queries: 664,
  species: 2096,
  source: "research repo, experiments/2026-10-13-v1-backbone-screen.md, addendum of 25 Oct (origin/main ef3a8df)",
  rows: [
    { name: "BioCLIP 2", detail: "224 px · Imageomics", top1: 70.9, top3: 85.2, top5: 88.9, top10: 92.6 },
    { name: "FungiTastic BEiT-B (frozen)", detail: "384 px · fungus classifier's features, head not used",
      top1: 68.8, top3: 81.3, top5: 84.5, top10: 88.5 },
    { name: "BioCLIP", detail: "224 px · Imageomics, first version", top1: 58.3, top3: 72.6, top5: 78.0, top10: 83.4 },
    { name: "DINOv3 large", detail: "512 px · Meta, general purpose", top1: 54.7, top3: 70.2, top5: 76.0, top10: 80.1 },
    { name: "SigLIP 2 large", detail: "384 px · Google, image-text", top1: 38.7, top3: 52.7, top5: 59.2, top10: 67.3,
      run: "13 Oct run" },
  ] as ScreenRow[],
};

/** The best row's lead over the next, in points: the screen's headline ("BioCLIP 2, by 12.5
 *  points"). Null with fewer than two rows. */
export function screenLead(rows: ScreenRow[]): { best: ScreenRow; points: number } | null {
  if (rows.length < 2) return null;
  const sorted = [...rows].sort((a, b) => b.top1 - a.top1);
  return { best: sorted[0], points: Math.round((sorted[0].top1 - sorted[1].top1) * 10) / 10 };
}

/** The research entry the fungi-specific cards cite (Steve, 2026-10-25): E33, committed on
 *  the research repo's origin/main as 8147947. */
export const E33 = {
  path: "experiments/2026-10-23-fungitastic-own-test.md",
  slug: "fungitastic-own-test",
  commit: "8147947",
};

/** Cards in the Benchmarks summary's "Fungi-specific models" block that wait on numbers not
 *  yet committed to the research repo: (a) the published FungiTastic / Danish Fungi models on
 *  our North American v1 val records, as published. Off until E33 holds them; switch it on
 *  (and put the numbers in FUNGI_PENDING) once the entry is committed. */
export const SHOW_PENDING_FUNGI_CARDS = false;

export interface VersusRow { name: string; detail?: string; top1: number; ours?: boolean }
export interface FewShotRow { per: string; n: number; top1: number; top3: number; top5: number; top10: number }

export type FungiCard =
  | { kind: "replication"; key: "d"; title: string; interim: boolean; ours: number; theirs: number;
      ours3: number; theirs3: number; model: string; testSet: string; also: string; source: string }
  | { kind: "versus"; key: "c"; title: string; interim: boolean; rows: VersusRow[];
      finds: number; withSpecies: number; note: string; source: string }
  | { kind: "fewshot"; key: "fewshot"; title: string; interim: boolean; rows: FewShotRow[];
      note: string; source: string }
  | { kind: "pending"; key: "a"; title: string; interim: boolean; text: string; source: string };

const E33_SOURCE = `research repo, ${E33.path} (origin/main ${E33.commit})`;

/** The committed cards, numbers copied from E33 (research 8147947): (d) the replication, final
 *  there; (c) their recipe on our open photos and the few-shot rows, both INTERIM there and here. */
export const FUNGI_CARDS: FungiCard[] = [
  { kind: "replication", key: "d", title: "Replicating FungiTastic", interim: false,
    ours: 75.28, theirs: 75.3, ours3: 86.68, theirs3: 86.7, model: "their released BEiT-B/384",
    testSet: "their ClosedSet-Test, 91,832 images",
    also: "ViT-B/384: 74.59 / 86.39 against their 74.9 / 86.3.",
    source: E33_SOURCE },
  { kind: "versus", key: "c", title: "Their recipe, our open photos", interim: true,
    rows: [
      { name: "MycoMap Vision v1-open", detail: "open photos, open reference", top1: 55.31, ours: true },
      { name: "FungiTastic recipe", detail: "BEiT-B, 384 px · best epoch so far (13)", top1: 51.23 },
    ],
    finds: 12874, withSpecies: 12798,
    note: "Species top 1 on the same v1 validation finds; both learned only from the same openly licensed photos. Their run is still training, so its number can move.",
    source: E33_SOURCE },
  { kind: "fewshot", key: "fewshot", title: "Few-shot, on their own test set", interim: true,
    rows: [
      { per: "Per image", n: 1909, top1: 44.84, top3: 59.98, top5: 67.31, top10: 75.22 },
      { per: "Per observation", n: 999, top1: 47.65, top3: 63.46, top5: 70.27, top10: 77.98 },
    ],
    note: "MycoMap Vision v1-open with FungiTastic's FewShot-Train (7,819 images, 2,427 species) as the reference, on FewShot-Test (567 species).",
    source: E33_SOURCE },
];

/** Cards waiting on committed numbers; shown only with SHOW_PENDING_FUNGI_CARDS. */
export const FUNGI_PENDING: FungiCard[] = [
  { kind: "pending", key: "a", title: "Their published models on our records", interim: true,
    text: "The FungiTastic and Danish Fungi models as published, on the v1 validation finds. Numbers come with the E33 entry.",
    source: E33_SOURCE },
];

/** The cards the summary shows: the committed ones, and the pending ones only when switched on. */
export const fungiCards = (showPending: boolean = SHOW_PENDING_FUNGI_CARDS): FungiCard[] =>
  showPending ? [...FUNGI_CARDS, ...FUNGI_PENDING] : FUNGI_CARDS;

// --- Citations (For researchers, and the references under Evaluation protocols) ------------

export interface Citation { key: string; label: string; text: string; bibtex: string; href?: string }

/** The site, and the works the identifier builds on or is compared with, as the paper draft
 *  cites them (research repo, paper/draft.md, References). */
export const CITATIONS: Citation[] = [
  { key: "site", label: "MycoMap Vision (this site)",
    text: "MycoMap Vision. https://vision.mycomap.org, accessed (date). A paper describing the method and its tests is in preparation.",
    bibtex: "@misc{mycomapvision,\n  title  = {MycoMap Vision},\n  author = {{MycoMap}},\n  howpublished = {\\url{https://vision.mycomap.org}},\n  note   = {Accessed (date)}\n}" },
  { key: "gu", label: "Gu et al. 2025 · the base model", href: "https://arxiv.org/abs/2505.23883",
    text: "Gu J, Stevens S, Campolongo EG, et al. 2025. BioCLIP 2: emergent properties from scaling hierarchical contrastive learning. arXiv:2505.23883.",
    bibtex: "@article{gu2025bioclip2,\n  title   = {BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning},\n  author  = {Gu, Jianyang and Stevens, Samuel and Campolongo, Elizabeth G. and others},\n  journal = {arXiv preprint arXiv:2505.23883},\n  year    = {2025}\n}" },
  { key: "stevens", label: "Stevens et al. 2024", href: "https://doi.org/10.1109/CVPR52733.2024.01836",
    text: "Stevens S, Wu J, Thompson MJ, et al. 2024. BioCLIP: a vision foundation model for the tree of life. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. doi:10.1109/CVPR52733.2024.01836.",
    bibtex: "@inproceedings{stevens2024bioclip,\n  title     = {BioCLIP: A Vision Foundation Model for the Tree of Life},\n  author    = {Stevens, Samuel and Wu, Jiaman and Thompson, Matthew J. and others},\n  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},\n  year      = {2024},\n  doi       = {10.1109/CVPR52733.2024.01836}\n}" },
  { key: "picek22", label: "Picek et al. 2022", href: "https://doi.org/10.1109/WACV51458.2022.00334",
    text: "Picek L, Šulc M, Matas J, Jeppesen TS, Heilmann-Clausen J, Læssøe T, Frøslev T. 2022. Danish Fungi 2020 – not just another image recognition dataset. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. p. 1525–1535. doi:10.1109/WACV51458.2022.00334.",
    bibtex: "@inproceedings{picek2022df20,\n  title     = {Danish Fungi 2020 -- Not Just Another Image Recognition Dataset},\n  author    = {Picek, Lukas and others},\n  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},\n  pages     = {1525--1535},\n  year      = {2022},\n  doi       = {10.1109/WACV51458.2022.00334}\n}" },
  { key: "picek25", label: "Picek et al. 2025", href: FUNGITASTIC_URL,
    text: "Picek L, et al. 2025. FungiTastic: a multi-modal dataset and benchmark for image categorization. arXiv:2408.13632.",
    bibtex: "@article{picek2025fungitastic,\n  title   = {FungiTastic: A Multi-Modal Dataset and Benchmark for Image Categorization},\n  author  = {Picek, Lukas and others},\n  journal = {arXiv preprint arXiv:2408.13632},\n  year    = {2025}\n}" },
];
