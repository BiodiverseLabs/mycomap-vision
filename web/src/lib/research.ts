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

// --- Benchmarks summary: the base-model screen and results by test set --------------------

/** One row of a top-1/3/5/10 table quoted from a committed research entry (percent). `run`
 *  says when a row comes from a different run than the rest of its table. */
export interface ScreenRow { name: string; detail: string; top1: number; top3: number; top5: number; top10: number;
                             run?: string }

/** The base-model screen (Steve, 2026-10-25: "Development screen", top of the Benchmarks
 *  summary). Frozen base models, nearest + species average, photos only, on the 664 deciding
 *  queries of a seeded v1 val subset (2,096 species). Copied from the research repo's
 *  experiments/2026-10-13-v1-backbone-screen.md at origin/main ef3a8df: its 25 Oct addendum
 *  (comparison 20261025-163049-1a3990), one run for every row it lists, with FungiTastic BEiT-B
 *  (frozen) beside BioCLIP 2, BioCLIP and DINOv3-L 512. SigLIP 2 L 384 is left out (Steve,
 *  2026-10-25): it isn't in the addendum's run, and the table shows one run only. The screen's fine-tuned row (the 224 px v0 fine-tune) is
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
  ] as ScreenRow[],
};

/** The research entry most test-set results cite (Steve, 2026-10-25): E33, as it stands on
 *  the research repo's origin/main at 2e2576a (E4 marked FINAL, the entry done). */
export const E33 = {
  path: "experiments/2026-10-23-fungitastic-own-test.md",
  slug: "fungitastic-own-test",
  commit: "2e2576a",
};

/** The entry the published models on v1 val come from: its 25 Oct addendum (502719f) and the
 *  three models added on 26 Oct, as on origin/main at 5792c17. */
export const V1_PUBLISHED = {
  path: "experiments/2026-10-13-v1-published-models.md",
  slug: "v1-published-models",
  commit: "5792c17",
};

/** A metric column; columns sharing a `group` sit under one heading ("Per image"). */
export interface ResultColumn { group?: string; label: string }
/** A model's row: one value per column, exactly as the entry gives it (null where it gives
 *  none), the 95% interval under a value where the entry gives one, and the entry's precision
 *  (one number for the row, or one per column) when it isn't the test set's. */
export interface ResultRow { name: string; detail?: string; cells: (number | null)[];
                             cis?: ([number, number] | null)[]; digits?: number | number[] }
/** A paired difference in points with its 95% interval. */
export interface PairedDiff { d: number; lo: number; hi: number }

/** One test set under one condition (its reference, or what a model was trained on), with
 *  every model scored on it. Results by test set (Steve, 2026-10-26: "This is not a
 *  competition... we have a number of test sets and will have more"): a new test set or model
 *  is a new entry or row here, not new card code. */
export interface TestSetResults {
  key: string;
  /** The test set's name. */
  name: string;
  /** The card it shares with other entries (Steve, 2026-10-26: aggregated where it makes
   *  sense); entries with the same card render as its sections, in order. Default: `name`. */
  card?: string;
  /** The condition, when one applies: "reference: FungiTastic-Train". */
  condition?: string;
  /** What it measures and who made it, in one line. */
  about: string;
  size: string;
  /** Who made the test set (the card's pill), unless the results are interim. */
  maker: string;
  interim?: boolean;
  columns: ResultColumn[];
  /** The entry's precision for the values. */
  digits: number;
  rows: ResultRow[];
  diffs?: { columns: string[]; rows: { name: string; cells: PairedDiff[] }[] };
  /** A check that the scoring reproduces a published number. */
  checks?: string[];
  notes: string[];
  source: string;
}

const E33_SOURCE = `research repo, ${E33.path} (origin/main ${E33.commit})`;
const V1_PUBLISHED_SOURCE = `research repo, ${V1_PUBLISHED.path}, addendum of 25 Oct and rows added 26 Oct (origin/main ${V1_PUBLISHED.commit})`;

const TOP_K = ["Top 1", "3", "5", "10"];
const topK = (group: string): ResultColumn[] => TOP_K.map((label) => ({ group, label }));
const PER_IMAGE_AND_OBSERVATION = [...topK("Per image"), ...topK("Per observation")];
const NO_K = [null, null, null, null];
/** A row's per-image and per-observation top 1/3/5/10 (per image only when `obs` is null). */
const io = (image: (number | null)[], obs: (number | null)[] | null): (number | null)[] => [...image, ...(obs ?? NO_K)];

const VISION_V1_OPEN = "MycoMap Vision v1-open";
const VISION_V1 = "MycoMap Vision v1";
const BIOCLIP2_BASE = "BioCLIP 2 (base model, not fine-tuned)";
const BIOCLIP_BASE = "BioCLIP (base model, not fine-tuned)";
const FT_BEIT = "FungiTastic BEiT-B/384";
const FT_VIT = "FungiTastic ViT-B/384";
const FT_SWIN = "FungiTastic Swin-B/384";
const FT_EFF = "FungiTastic EfficientNetV2-B3/384";
const FT_CONVNEXT = "FungiTastic ConvNeXt-B/384";
const DF20 = "Danish Fungi 2020 ViT-L/384";
const E4 = "FungiTastic recipe: BEiT-B/384";

const FUNGITASTIC = "FungiTastic (Picek et al. 2025)";
const RETRIEVAL_ON_FT_TRAIN = "Retrieval models answer from FungiTastic-Train (433,499 images, all 2,829 classes) as the reference, with no training on FungiTastic; the classifiers were trained on those same images. Every model works from the photos alone: no place, date or other metadata. Per observation, retrieval scores and classifier logits are each averaged over the observation's photos.";
const PAIRED = "Paired on the same images; 95% intervals from an observation-clustered bootstrap (2,000 resamples).";
const ON_NA_REFERENCE = "The reference is MycoMap's North American v1 training split, so only the species it holds can be named: a test of transfer across continents. An answer counts when it equals the image's scientific name or its GBIF accepted species (exact names, no crosswalk). Photos only.";

/** Every test set with committed results, each number copied from its entry on the research
 *  repo's origin/main (E33 at 2e2576a; v1-published-models at 5792c17). None is the sealed
 *  test (v1-test): it has no results until it is opened. FungiTastic's images are CC BY-NC-SA,
 *  so only scores appear. */
export const TEST_SET_RESULTS: TestSetResults[] = [
  { key: "fungitastic-closed-ft-train", name: "FungiTastic ClosedSet-Test", condition: "reference: FungiTastic-Train",
    about: `${FUNGITASTIC}'s closed-set test: observations from the Atlas of Danish Fungi whose species all appear in its training set.`,
    size: "91,756 images · 48,361 observations · 2,336 species", maker: "FungiTastic",
    columns: PER_IMAGE_AND_OBSERVATION, digits: 2,
    rows: [
      { name: VISION_V1_OPEN, detail: "retrieval, FungiTastic-Train as reference", cells: io([77.46, 89.09, 91.81, 94.25], [88.31, 96.76, 98.21, 99.13]) },
      { name: VISION_V1, detail: "retrieval, FungiTastic-Train as reference", cells: io([77.54, 89.12, 91.81, 94.31], [88.29, 96.78, 98.22, 99.13]) },
      { name: BIOCLIP2_BASE, detail: "retrieval, FungiTastic-Train as reference", cells: io([76.36, 88.26, 91.04, 93.60], [87.61, 96.38, 97.86, 98.88]) },
      { name: FT_BEIT, detail: "classifier trained on FungiTastic-Train", cells: io([75.28, 86.68, 89.56, 92.26], [87.42, 96.01, 97.52, 98.66]) },
      { name: FT_VIT, detail: "classifier trained on FungiTastic-Train", cells: io([74.59, 86.39, 89.39, 92.10], [86.41, 95.47, 97.10, 98.34]) },
      { name: FT_SWIN, detail: "classifier · per image, all 91,832 images", cells: io([74.24, 86.33, 89.33, 92.02], null) },
      { name: FT_EFF, detail: "classifier · per image, all 91,832 images", cells: io([72.03, 84.67, 88.02, 91.22], null) },
      { name: FT_CONVNEXT, detail: "classifier · per image, all 91,832 images", cells: io([70.71, 83.83, 87.29, 90.74], null) },
      { name: DF20, detail: "classifier, 1,604 classes · per image, all 91,832 images", cells: io([69.15, 79.41, 81.70, 83.84], null) },
    ],
    diffs: { columns: ["Per image", "Per observation"], rows: [
      { name: `${VISION_V1_OPEN} − ${FT_BEIT}`, cells: [{ d: 2.18, lo: 1.92, hi: 2.47 }, { d: 0.90, lo: 0.63, hi: 1.17 }] },
      { name: `${VISION_V1_OPEN} − ${FT_VIT}`, cells: [{ d: 2.87, lo: 2.59, hi: 3.16 }, { d: 1.91, lo: 1.62, hi: 2.18 }] },
      { name: `${VISION_V1} − ${FT_BEIT}`, cells: [{ d: 2.27, lo: 2.00, hi: 2.54 }, { d: 0.87, lo: 0.60, hi: 1.15 }] },
      { name: `${VISION_V1_OPEN} − ${BIOCLIP2_BASE}`, cells: [{ d: 1.11, lo: 0.93, hi: 1.28 }, { d: 0.71, lo: 0.51, hi: 0.90 }] },
    ] },
    checks: [`Scoring check: ${FT_BEIT} as published scores 75.28 at top 1 and 86.68 at top 3 in this pipeline, against 75.3 and 86.7 in the FungiTastic paper (Table 8); ${FT_VIT} 74.59 / 86.39 against 74.9 / 86.3.`],
    notes: [RETRIEVAL_ON_FT_TRAIN,
      "76 images that are cut short are left out for every model (it does not move BEiT-B or ViT-B at two decimals); rows scored per image only use all 91,832. Danish Fungi 2020 ViT-L/384 has 1,604 classes: an image of a species it has no class for counts as a miss.",
      PAIRED],
    source: E33_SOURCE },
  { key: "fungitastic-closed-mycomap-ref", name: "FungiTastic ClosedSet-Test", condition: "reference: MycoMap North American records",
    about: `${FUNGITASTIC}'s closed-set test: observations from the Atlas of Danish Fungi whose species all appear in its training set.`,
    size: "91,756 images · 48,361 observations · 2,336 species", maker: "FungiTastic",
    columns: PER_IMAGE_AND_OBSERVATION, digits: 2,
    rows: [
      { name: VISION_V1_OPEN, detail: "the 979 species in its reference (open photos, 16,153 species): 54,841 images, 29,558 observations",
        cells: io([37.23, 55.61, 62.85, 72.09], [43.77, 63.07, 70.45, 79.49]) },
      { name: VISION_V1_OPEN, detail: "all 2,336 species", cells: io([22.25, 33.24, 37.56, 43.09], [26.75, 38.55, 43.06, 48.59]) },
      { name: VISION_V1, detail: "the 994 species in its reference (all photos, 16,959 species): 55,631 images, 29,905 observations",
        cells: io([37.18, 55.49, 62.98, 72.59], [43.86, 62.90, 70.46, 80.25]) },
      { name: VISION_V1, detail: "all 2,336 species", cells: io([22.54, 33.64, 38.18, 44.01], [27.12, 38.89, 43.57, 49.63]) },
    ],
    notes: [ON_NA_REFERENCE],
    source: E33_SOURCE },
  { key: "fungitastic-dna-ft-train", name: "FungiTastic DNA-Test", condition: "reference: FungiTastic-Train",
    about: `${FUNGITASTIC}'s DNA-verified test: observations whose species was confirmed by DNA sequencing.`,
    size: "5,105 images · 2,041 observations · 725 species", maker: "FungiTastic",
    columns: PER_IMAGE_AND_OBSERVATION, digits: 2,
    rows: [
      { name: VISION_V1_OPEN, detail: "retrieval, FungiTastic-Train as reference", cells: io([41.86, 60.63, 67.44, 74.46], [56.59, 78.83, 86.58, 92.60]) },
      { name: VISION_V1, detail: "retrieval, FungiTastic-Train as reference", cells: io([41.78, 61.25, 67.44, 74.55], [56.83, 79.27, 86.38, 92.41]) },
      { name: BIOCLIP2_BASE, detail: "retrieval, FungiTastic-Train as reference", cells: io([37.67, 56.28, 63.70, 71.36], [51.54, 73.35, 81.58, 88.49]) },
      { name: FT_BEIT, detail: "classifier trained on FungiTastic-Train", cells: io([38.73, 56.87, 63.82, 70.91], [57.28, 78.69, 85.84, 92.45]) },
      { name: FT_VIT, detail: "classifier · per image", cells: io([36.38, 53.50, 60.24, 67.82], null) },
      { name: FT_SWIN, detail: "classifier · per image", cells: io([37.96, 57.51, 63.66, 71.26], null) },
      { name: FT_EFF, detail: "classifier · per image", cells: io([35.55, 54.44, 61.27, 69.21], null) },
      { name: FT_CONVNEXT, detail: "classifier · per image", cells: io([34.01, 53.12, 60.00, 68.15], null) },
      { name: DF20, detail: "classifier, 1,604 classes · per image", cells: io([31.75, 41.49, 44.72, 48.33], null) },
    ],
    diffs: { columns: ["Per image", "Per observation"], rows: [
      { name: `${VISION_V1_OPEN} − ${FT_BEIT}`, cells: [{ d: 3.13, lo: 1.63, hi: 4.55 }, { d: -0.69, lo: -2.79, hi: 1.32 }] },
      { name: `${VISION_V1_OPEN} − ${FT_VIT}`, cells: [{ d: 5.48, lo: 4.01, hi: 6.87 }, { d: 3.77, lo: 1.57, hi: 5.88 }] },
      { name: `${VISION_V1} − ${FT_BEIT}`, cells: [{ d: 3.06, lo: 1.59, hi: 4.48 }, { d: -0.44, lo: -2.55, hi: 1.57 }] },
      { name: `${VISION_V1_OPEN} − ${BIOCLIP2_BASE}`, cells: [{ d: 4.19, lo: 3.22, hi: 5.20 }, { d: 5.05, lo: 3.58, hi: 6.61 }] },
    ] },
    notes: [RETRIEVAL_ON_FT_TRAIN, "Danish Fungi 2020 ViT-L/384 has 1,604 classes: an image of a species it has no class for counts as a miss.", PAIRED],
    source: E33_SOURCE },
  { key: "fungitastic-dna-mycomap-ref", name: "FungiTastic DNA-Test", condition: "reference: MycoMap North American records",
    about: `${FUNGITASTIC}'s DNA-verified test: observations whose species was confirmed by DNA sequencing.`,
    size: "5,105 images · 725 species", maker: "FungiTastic",
    columns: PER_IMAGE_AND_OBSERVATION, digits: 2,
    rows: [
      { name: VISION_V1_OPEN, detail: "the 383 species in its reference (open photos): 2,782 images, 1,100 observations",
        cells: io([16.10, 27.93, 34.51, 42.95], [23.73, 39.82, 46.64, 55.73]) },
      { name: VISION_V1_OPEN, detail: "all 725 species · per image", cells: io([8.78, 15.22, 18.81, 23.41], null) },
    ],
    notes: [ON_NA_REFERENCE],
    source: E33_SOURCE },
  { key: "fungitastic-fewshot", name: "FungiTastic FewShot-Test", condition: "reference: FungiTastic FewShot-Train (7,819 images, 2,427 species)",
    about: `${FUNGITASTIC}'s few-shot test: species with only a handful of observations to learn from.`,
    size: "1,909 images · 999 observations · 567 species", maker: "FungiTastic",
    columns: PER_IMAGE_AND_OBSERVATION, digits: 2,
    rows: [
      { name: BIOCLIP_BASE, detail: "class centroid, as published in the FungiTastic paper (Table 4)",
        cells: io([21.8, 32.6, null, null], null), digits: 1 },
      { name: BIOCLIP_BASE, detail: "class centroid", cells: io([24.99, 37.40, 43.84, 52.28], [29.83, 42.54, 48.25, 56.86]) },
      { name: BIOCLIP2_BASE, detail: "class centroid, 336 px", cells: io([43.11, 58.88, 66.21, 73.86], [47.25, 63.16, 68.97, 76.68]) },
      { name: VISION_V1_OPEN, detail: "class centroid", cells: io([46.41, 61.92, 67.78, 76.27], [49.75, 64.46, 70.87, 78.48]) },
      { name: VISION_V1, detail: "class centroid", cells: io([45.94, 61.92, 68.31, 77.06], [49.25, 63.96, 71.67, 79.98]) },
      { name: BIOCLIP_BASE, detail: "closest specimens + species average", cells: io([24.67, 36.14, 42.85, 52.49], [28.93, 41.24, 47.35, 56.26]) },
      { name: BIOCLIP2_BASE, detail: "closest specimens + species average, 336 px", cells: io([42.74, 57.52, 64.75, 73.02], [46.25, 62.46, 68.77, 75.18]) },
      { name: VISION_V1_OPEN, detail: "closest specimens + species average", cells: io([44.84, 59.98, 67.31, 75.22], [47.65, 63.46, 70.27, 77.98]) },
      { name: VISION_V1, detail: "closest specimens + species average", cells: io([45.31, 61.08, 68.05, 76.43], [48.35, 63.76, 70.77, 79.38]) },
    ],
    notes: [
      "The class centroid (each species' average image) is the protocol of the FungiTastic paper's Table 4; closest specimens + species average is MycoMap Vision's served method. Photos only.",
      `From the published 21.8 to 46.41 at top 1: the same BioCLIP in this pipeline gives 24.99 (protocol and resolution, about 3 points); ${BIOCLIP2_BASE} adds about 18; the MycoMap fine-tune on North American open photos about 3 more. No intervals yet: about ±2.2 points at 95% per model.`,
    ],
    source: E33_SOURCE },
  { key: "v1-val-as-released", name: "MycoMap v1 validation (12,798 finds)", condition: "models as released",
    about: "MycoMap's DNA-verified North American records first validated in the eight weeks after dataset release v1's cutoff: the development set every setting is chosen on.",
    size: "12,874 finds · 12,798 with a species · 2,411 of a species in FungiTastic's 2,829 classes", maker: "MycoMap v1",
    columns: [...topK("Species, 12,798 finds"), { group: "Genus, 12,865 finds", label: "Top 1" },
              { group: "Species in FungiTastic's classes, 2,411 finds", label: "Top 1" }],
    digits: 1,
    rows: [
      { name: VISION_V1, detail: "all photos, full reference", cells: [56.3, 73.6, 78.9, 84.3, 85.8, 67.6],
        cis: [[53.4, 59.2], null, null, null, [84.0, 87.5], [64.6, 70.4]] },
      { name: VISION_V1_OPEN, detail: "open photos, open reference", cells: [55.3, 72.6, 77.7, 83.3, 85.4, 66.8],
        cis: [[52.4, 58.2], null, null, null, [83.6, 87.0], [63.9, 69.5]] },
      { name: VISION_V1_OPEN, detail: "open photos, full reference", cells: [55.9, 73.4, 78.5, 84.1, 85.6, null],
        cis: [[53.0, 58.9], null, null, null, [83.8, 87.3], null] },
      { name: FT_BEIT, cells: [10.7, 14.1, 15.2, 16.5, 58.6, 56.9], cis: [[9.4, 12.4], null, null, null, [56.0, 61.4], [53.1, 61.2]] },
      { name: FT_SWIN, cells: [10.3, 13.7, 14.9, 16.2, 56.6, 54.7], cis: [[8.9, 12.1], null, null, null, null, [50.3, 59.7]] },
      { name: FT_EFF, cells: [10.2, 13.5, 14.7, 15.9, 56.2, 54.1], cis: [[8.8, 11.9], null, null, null, null, [50.1, 58.6]] },
      { name: FT_VIT, cells: [10.1, 13.4, 14.6, 15.8, 55.6, 53.5], cis: [[8.7, 11.7], null, null, null, [52.8, 58.5], [49.4, 57.9]] },
      { name: FT_CONVNEXT, cells: [9.6, 13.0, 14.4, 15.8, 53.9, 51.0], cis: [[8.2, 11.4], null, null, null, null, [46.7, 56.2]] },
      { name: DF20, detail: "1,604 classes", cells: [10.2, 12.9, 13.8, 14.6, 55.6, 52.6],
        cis: [[8.9, 11.6], null, null, null, [52.9, 58.3], [49.2, 55.9]] },
    ],
    notes: [
      "The published classifiers answer from the photos alone (each find's photos' logits averaged), the classifiers' class names matched to MycoMap's through the reviewed GBIF / Catalogue of Life crosswalk. Every model is credited by the same species-rank answer key (erratum r4); 95% intervals, clustered by observer, sit under the values the entry gives them for.",
      "10,387 of the 12,798 finds are of a species with no FungiTastic class. The last column is the 2,411 that have one (all five FungiTastic models share the 2,829 classes; Danish Fungi 2020 is scored on the same finds).",
    ],
    source: V1_PUBLISHED_SOURCE },
  { key: "v1-val-open-photos", name: "MycoMap v1 validation (12,798 finds)", condition: "training: open photos only (E4)",
    about: "The same finds, with the FungiTastic training recipe trained on v1's openly licensed photos: the photos MycoMap Vision v1-open learned from.",
    size: "12,874 finds · 12,798 with a species", maker: "MycoMap v1",
    columns: [...topK("Species"), { group: "Genus", label: "Top 1" }], digits: 2,
    rows: [
      { name: E4, detail: "trained on v1's open photos (E4) · final: best epoch 13", cells: [51.23, 67.18, 72.90, 79.03, 80.22] },
      { name: VISION_V1_OPEN, detail: "open photos, open reference", cells: [55.31, 72.61, 77.65, 83.29, 84.9], digits: [2, 2, 2, 2, 1] },
      { name: VISION_V1, detail: "all photos, full reference", cells: [56.32, 73.55, 78.87, 84.28, 85.4], digits: [2, 2, 2, 2, 1] },
    ],
    diffs: { columns: ["Species top 1", "3", "5", "10"], rows: [
      { name: `${E4} − ${VISION_V1_OPEN}`, cells: [{ d: -4.04, lo: -4.91, hi: -3.24 }, { d: -5.41, lo: -6.37, hi: -4.53 },
        { d: -4.71, lo: -5.60, hi: -3.92 }, { d: -4.22, lo: -4.97, hi: -3.59 }] },
    ] },
    notes: [
      "E4 (BEiT-B/384, Seesaw loss, the FungiTastic schedule) stopped by its declared epoch rule at epoch 20; the final model is its best epoch, 13. Every row is scored the same way: each find's photos averaged, the species-rank answer key.",
      "Paired on the same finds; 95% intervals from an observer-clustered bootstrap (752 observers, 2,000 resamples). The interval's own scoring gives MycoMap Vision v1-open 55.27 rather than 55.31; the 0.04-point difference is in its row, not E4's.",
    ],
    source: E33_SOURCE },
];

/** The cards, in order: entries grouped by `card` (else `name`), each card's entries in order. */
export function resultCards(sets: TestSetResults[] = TEST_SET_RESULTS): { title: string; sets: TestSetResults[] }[] {
  const cards: { title: string; sets: TestSetResults[] }[] = [];
  for (const s of sets) {
    const title = s.card ?? s.name;
    const at = cards.find((c) => c.title === title);
    if (at) at.sets.push(s); else cards.push({ title, sets: [s] });
  }
  return cards;
}

/** The precision a value is shown at: the row's, per column or whole, else the test set's. */
export const cellDigits = (s: TestSetResults, r: ResultRow, k: number): number =>
  Array.isArray(r.digits) ? r.digits[k] ?? s.digits : r.digits ?? s.digits;

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
