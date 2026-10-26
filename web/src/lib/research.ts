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

/** The best row's lead over the next, in points: the screen's headline ("BioCLIP 2, by 12.5
 *  points"). Null with fewer than two rows. */
export function screenLead(rows: ScreenRow[]): { best: ScreenRow; points: number } | null {
  if (rows.length < 2) return null;
  const sorted = [...rows].sort((a, b) => b.top1 - a.top1);
  return { best: sorted[0], points: Math.round((sorted[0].top1 - sorted[1].top1) * 10) / 10 };
}

/** The research entry most fungi-specific cards cite (Steve, 2026-10-25): E33, as it stands on
 *  the research repo's origin/main at 6090354 (the entry's last change; read at 5792c17). */
export const E33 = {
  path: "experiments/2026-10-23-fungitastic-own-test.md",
  slug: "fungitastic-own-test",
  commit: "6090354",
};

/** The entry card (a) cites: the published models on v1 val, its 25 Oct addendum (502719f)
 *  and the three models added on 26 Oct, as on origin/main at 5792c17. */
export const V1_PUBLISHED = {
  path: "experiments/2026-10-13-v1-published-models.md",
  slug: "v1-published-models",
  commit: "5792c17",
};

/** Top 1, 3, 5 and 10 in percent, exactly as the entry gives them; null where it gives none. */
export type TopK4 = readonly [number | null, number | null, number | null, number | null];

export interface VersusRow { name: string; detail?: string; top1: number; ours?: boolean }

/** A model's row in a per-image / per-observation table; `observation` is null when the entry
 *  scores it per image only. `digits`: the entry's precision when it isn't the card's. */
export interface PairRow { name: string; detail?: string; ours?: boolean; image: TopK4; observation: TopK4 | null;
                           digits?: number }
/** One test set's table. */
export interface PairTable { split: string; images: number; observations: number; rows: PairRow[] }
/** A paired difference in points with its 95% interval (absent when the entry gives none). */
export interface Diff { d: number; lo: number; hi: number }
export interface DiffRow { name: string; cells: Diff[] }

/** A row of card (a): species top-k and genus top 1 on every find with a species, and species
 *  top 1 on the finds whose species the published models have a class for. */
export interface PublishedRow { name: string; detail?: string; ours?: boolean; species: TopK4; ci: [number, number];
                                genus: number; shared: number; sharedCi: [number, number] }

/** A row of card (b1): one split, per image or per observation, all finds or the species we have. */
export interface CrossRow { split: string; per: "image" | "observation"; of: "all" | "ours"; n: number; topk: TopK4 }

interface CardBase { title: string; interim: boolean; tag?: string; source: string }

export type FungiCard =
  | CardBase & { kind: "replication"; key: "d"; ours: number; theirs: number;
      ours3: number; theirs3: number; model: string; testSet: string; also: string }
  | CardBase & { kind: "versus"; key: "c"; rows: VersusRow[]; finds: number; withSpecies: number; note: string }
  | CardBase & { kind: "pairs"; key: "b2" | "fewshot"; lead: string; tables: PairTable[];
      /** The bigstat: the first `ours` row's top 1 per image, beside this row's. */
      versus: string; diffHead?: string[]; diffs?: DiffRow[]; notes: string[] }
  | CardBase & { kind: "published"; key: "a"; finds: number; withSpecies: number; shared: number;
      rows: PublishedRow[]; notes: string[] }
  | CardBase & { kind: "cross"; key: "b1"; rows: CrossRow[]; classes: Record<string, string>;
      note: string };

/** Card (b1)'s headline: the first per-observation row on the species we have. */
export const crossHeadline = (rows: CrossRow[]): CrossRow | undefined =>
  rows.find((r) => r.per === "observation" && r.of === "ours");

/** Whether a card takes the block's full width (its table needs it). */
export const isWideCard = (c: FungiCard): boolean => c.kind === "pairs" || c.kind === "published";

const E33_SOURCE = `research repo, ${E33.path} (origin/main ${E33.commit})`;
const V1_PUBLISHED_SOURCE = `research repo, ${V1_PUBLISHED.path}, addendum of 25 Oct and rows added 26 Oct (origin/main ${V1_PUBLISHED.commit})`;

const FT_BEIT = "FungiTastic BEiT-B/384";
const FT_VIT = "FungiTastic ViT-B/384";
const FT_SWIN = "FungiTastic Swin-B/384";
const FT_EFF = "FungiTastic EfficientNetV2-B3/384";
const FT_CONVNEXT = "FungiTastic ConvNeXt-B/384";
const BIOCLIP2_BASE = "BioCLIP 2 (base model, not fine-tuned)";

const CLOSED_CLASSES = "979 of their 2,336 ClosedSet-Test classes";
const DNA_CLASSES = "383 of 725 DNA-Test classes";

/** The committed cards, every number copied from its entry on the research repo's origin/main
 *  (E33 at 6090354; v1-published-models at 5792c17). Nothing here is from the sealed test
 *  (v1-test): FungiTastic's own test sets, or v1 validation finds. FungiTastic's images are
 *  CC BY-NC-SA, so the cards carry scores only. (c), their recipe on our open photos, is still
 *  INTERIM in E33 and here. */
export const FUNGI_CARDS: FungiCard[] = [
  { kind: "replication", key: "d", title: "Replicating FungiTastic", interim: false, tag: "Replication",
    ours: 75.28, theirs: 75.3, ours3: 86.68, theirs3: 86.7, model: "their released BEiT-B/384",
    testSet: "their ClosedSet-Test, 91,832 images",
    also: "ViT-B/384: 74.59 / 86.39 against their 74.9 / 86.3. Their other released checkpoints, not in Table 8, score lower on the same images at top 1: Swin-B/384 74.24, EfficientNetV2-B3/384 72.03, ConvNeXt-B/384 70.71.",
    source: E33_SOURCE },
  { kind: "versus", key: "c", title: "Their recipe, our open photos", interim: true,
    rows: [
      { name: "MycoMap Vision v1-open", detail: "open photos, open reference", top1: 55.31, ours: true },
      { name: "FungiTastic recipe", detail: "BEiT-B, 384 px · best epoch so far (13)", top1: 51.23 },
    ],
    finds: 12874, withSpecies: 12798,
    note: "Species top 1 on the same v1 validation finds; both learned only from the same openly licensed photos. Their run is still training, so its number can move.",
    source: E33_SOURCE },
  { kind: "cross", key: "b1", title: "Our North American references, their Danish finds", interim: false,
    tag: "Their test sets",
    rows: [
      { split: "ClosedSet-Test", per: "image", of: "ours", n: 54841, topk: [37.23, 55.61, 62.85, 72.09] },
      { split: "ClosedSet-Test", per: "observation", of: "ours", n: 29558, topk: [43.77, 63.07, 70.45, 79.49] },
      { split: "ClosedSet-Test", per: "image", of: "all", n: 91756, topk: [22.25, 33.24, 37.56, 43.09] },
      { split: "ClosedSet-Test", per: "observation", of: "all", n: 48361, topk: [26.75, 38.55, 43.06, 48.59] },
      { split: "DNA-Test", per: "image", of: "ours", n: 2782, topk: [16.10, 27.93, 34.51, 42.95] },
      { split: "DNA-Test", per: "observation", of: "ours", n: 1100, topk: [23.73, 39.82, 46.64, 55.73] },
      { split: "DNA-Test", per: "image", of: "all", n: 5105, topk: [8.78, 15.22, 18.81, 23.41] },
    ],
    classes: { "ClosedSet-Test": CLOSED_CLASSES, "DNA-Test": DNA_CLASSES },
    note: "MycoMap Vision v1-open, photos only, answering from its own North American reference (v1 training split, open photos: 16,153 species). “Ours” means the species it has, the only ones it can name, so this tests transfer across continents, not a match to their 75.3%. MycoMap Vision v1 is level on the species it has: 37.18 per image, 43.86 per observation.",
    source: E33_SOURCE },
  { kind: "pairs", key: "b2", title: "Head to head on their test sets", interim: false, tag: "Their test sets",
    lead: "Their training photos (433,499 images, all 2,829 classes) are MycoMap Vision's reference, and it is not trained on FungiTastic at all: the models are as released, fine-tuned on North American open photos only. Both sides are photo only: no place, date or other metadata.",
    versus: FT_BEIT,
    tables: [
      { split: "ClosedSet-Test", images: 91756, observations: 48361, rows: [
        { name: "MycoMap Vision v1-open", detail: "their training photos as reference", ours: true,
          image: [77.46, 89.09, 91.81, 94.25], observation: [88.31, 96.76, 98.21, 99.13] },
        { name: "MycoMap Vision v1", detail: "the same reference", ours: true,
          image: [77.54, 89.12, 91.81, 94.31], observation: [88.29, 96.78, 98.22, 99.13] },
        { name: BIOCLIP2_BASE, detail: "the same reference",
          image: [76.36, 88.26, 91.04, 93.60], observation: [87.61, 96.38, 97.86, 98.88] },
        { name: FT_BEIT, detail: "their classifier, trained on those photos",
          image: [75.28, 86.68, 89.56, 92.26], observation: [87.42, 96.01, 97.52, 98.66] },
        { name: FT_VIT, detail: "their classifier",
          image: [74.59, 86.39, 89.39, 92.10], observation: [86.41, 95.47, 97.10, 98.34] },
        { name: FT_SWIN, detail: "their classifier · all 91,832 images",
          image: [74.24, 86.33, 89.33, 92.02], observation: null },
        { name: FT_EFF, detail: "their classifier · all 91,832 images",
          image: [72.03, 84.67, 88.02, 91.22], observation: null },
        { name: FT_CONVNEXT, detail: "their classifier · all 91,832 images",
          image: [70.71, 83.83, 87.29, 90.74], observation: null },
      ] },
      { split: "DNA-Test", images: 5105, observations: 2041, rows: [
        { name: "MycoMap Vision v1-open", detail: "their training photos as reference", ours: true,
          image: [41.86, 60.63, 67.44, 74.46], observation: [56.59, 78.83, 86.58, 92.60] },
        { name: "MycoMap Vision v1", detail: "the same reference", ours: true,
          image: [41.78, 61.25, 67.44, 74.55], observation: [56.83, 79.27, 86.38, 92.41] },
        { name: BIOCLIP2_BASE, detail: "the same reference",
          image: [37.67, 56.28, 63.70, 71.36], observation: [51.54, 73.35, 81.58, 88.49] },
        { name: FT_BEIT, detail: "their classifier",
          image: [38.73, 56.87, 63.82, 70.91], observation: [57.28, 78.69, 85.84, 92.45] },
        { name: FT_VIT, detail: "their classifier", image: [36.38, 53.50, 60.24, 67.82], observation: null },
        { name: FT_SWIN, detail: "their classifier", image: [37.96, 57.51, 63.66, 71.26], observation: null },
        { name: FT_EFF, detail: "their classifier", image: [35.55, 54.44, 61.27, 69.21], observation: null },
        { name: FT_CONVNEXT, detail: "their classifier", image: [34.01, 53.12, 60.00, 68.15], observation: null },
      ] },
    ],
    diffHead: ["ClosedSet-Test, per image", "per observation", "DNA-Test, per image", "per observation"],
    diffs: [
      { name: `MycoMap Vision v1-open − ${FT_BEIT}`, cells: [
        { d: 2.18, lo: 1.92, hi: 2.47 }, { d: 0.90, lo: 0.63, hi: 1.17 },
        { d: 3.13, lo: 1.63, hi: 4.55 }, { d: -0.69, lo: -2.79, hi: 1.32 }] },
      { name: `MycoMap Vision v1-open − ${FT_VIT}`, cells: [
        { d: 2.87, lo: 2.59, hi: 3.16 }, { d: 1.91, lo: 1.62, hi: 2.18 },
        { d: 5.48, lo: 4.01, hi: 6.87 }, { d: 3.77, lo: 1.57, hi: 5.88 }] },
      { name: `MycoMap Vision v1 − ${FT_BEIT}`, cells: [
        { d: 2.27, lo: 2.00, hi: 2.54 }, { d: 0.87, lo: 0.60, hi: 1.15 },
        { d: 3.06, lo: 1.59, hi: 4.48 }, { d: -0.44, lo: -2.55, hi: 1.57 }] },
      { name: `What the fine-tune adds: MycoMap Vision v1-open − ${BIOCLIP2_BASE}`, cells: [
        { d: 1.11, lo: 0.93, hi: 1.28 }, { d: 0.71, lo: 0.51, hi: 0.90 },
        { d: 4.19, lo: 3.22, hi: 5.20 }, { d: 5.05, lo: 3.58, hi: 6.61 }] },
    ],
    notes: [
      "Their classifiers are scored on exactly the same images; per observation, by averaging their logits over the observation's photos (their own rule). 76 ClosedSet-Test images that are cut short are left out for every model; it does not move BEiT-B or ViT-B at two decimals. Their other three checkpoints are scored per image only, on all 91,832 ClosedSet-Test images.",
      "Paired differences at top 1, in points, with 95% intervals from an observation-clustered bootstrap (2,000 resamples). Every ClosedSet-Test difference excludes zero; on DNA-Test, the smaller sequence-verified set, MycoMap Vision leads per image, and per observation it is level with BEiT-B.",
    ],
    source: E33_SOURCE },
  { kind: "published", key: "a", title: "Their published models on our finds", interim: false, tag: "v1 validation",
    finds: 12874, withSpecies: 12798, shared: 2411,
    rows: [
      { name: "MycoMap Vision v1", detail: "all photos, full reference", ours: true,
        species: [56.3, 73.6, 78.9, 84.3], ci: [53.4, 59.2], genus: 85.8, shared: 67.6, sharedCi: [64.6, 70.4] },
      { name: "MycoMap Vision v1-open", detail: "open photos, open reference", ours: true,
        species: [55.3, 72.6, 77.7, 83.3], ci: [52.4, 58.2], genus: 85.4, shared: 66.8, sharedCi: [63.9, 69.5] },
      { name: FT_BEIT, species: [10.7, 14.1, 15.2, 16.5], ci: [9.4, 12.4], genus: 58.6, shared: 56.9, sharedCi: [53.1, 61.2] },
      { name: FT_SWIN, species: [10.3, 13.7, 14.9, 16.2], ci: [8.9, 12.1], genus: 56.6, shared: 54.7, sharedCi: [50.3, 59.7] },
      { name: FT_EFF, species: [10.2, 13.5, 14.7, 15.9], ci: [8.8, 11.9], genus: 56.2, shared: 54.1, sharedCi: [50.1, 58.6] },
      { name: FT_VIT, species: [10.1, 13.4, 14.6, 15.8], ci: [8.7, 11.7], genus: 55.6, shared: 53.5, sharedCi: [49.4, 57.9] },
      { name: FT_CONVNEXT, species: [9.6, 13.0, 14.4, 15.8], ci: [8.2, 11.4], genus: 53.9, shared: 51.0, sharedCi: [46.7, 56.2] },
      { name: "Danish Fungi 2020 ViT-L/384", detail: "1,604 classes of its own",
        species: [10.2, 12.9, 13.8, 14.6], ci: [8.9, 11.6], genus: 55.6, shared: 52.6, sharedCi: [49.2, 55.9] },
    ],
    notes: [
      "Their models as released, from the photos alone (each find's photos' logits averaged), their class names matched to ours through the reviewed GBIF / Catalogue of Life crosswalk; every model credited by the same species-rank answer key, with 95% intervals clustered by observer (below each top 1). Four in five finds are species they have no class for; on the finds of species they do have, the gap is not only vocabulary.",
      "The last column is the 2,411 finds whose species is one of FungiTastic's 2,829 classes (all five FungiTastic models share them; Danish Fungi 2020 is scored on the same finds). Genus top 1 is on the 12,865 finds with a genus.",
    ],
    source: V1_PUBLISHED_SOURCE },
  { kind: "pairs", key: "fewshot", title: "Few-shot, on their own test set", interim: false, tag: "Their test sets",
    lead: "FungiTastic's FewShot-Train (7,819 images, 2,427 species) is the reference; FewShot-Test is scored (567 species). The class centroid (each species' average image) is the protocol of their Table 4; the served method is MycoMap Vision's closest specimens + species average.",
    versus: "FungiTastic paper: BioCLIP",
    tables: [
      { split: "FewShot-Test", images: 1909, observations: 999, rows: [
        { name: "MycoMap Vision v1-open", detail: "class centroid", ours: true,
          image: [46.41, 61.92, 67.78, 76.27], observation: [49.75, 64.46, 70.87, 78.48] },
        { name: "MycoMap Vision v1", detail: "class centroid", ours: true,
          image: [45.94, 61.92, 68.31, 77.06], observation: [49.25, 63.96, 71.67, 79.98] },
        { name: "MycoMap Vision v1-open", detail: "served method", ours: true,
          image: [44.84, 59.98, 67.31, 75.22], observation: [47.65, 63.46, 70.27, 77.98] },
        { name: "MycoMap Vision v1", detail: "served method", ours: true,
          image: [45.31, 61.08, 68.05, 76.43], observation: [48.35, 63.76, 70.77, 79.38] },
        { name: BIOCLIP2_BASE, detail: "class centroid, 336 px",
          image: [43.11, 58.88, 66.21, 73.86], observation: [47.25, 63.16, 68.97, 76.68] },
        { name: "BioCLIP (base model, not fine-tuned)", detail: "class centroid, our pipeline",
          image: [24.99, 37.40, 43.84, 52.28], observation: [29.83, 42.54, 48.25, 56.86] },
        { name: "FungiTastic paper: BioCLIP", detail: "their Table 4, best centroid",
          image: [21.8, 32.6, null, null], observation: null, digits: 1 },
      ] },
    ],
    notes: [
      "Where the gain over their Table 4 comes from: our pipeline with the original BioCLIP gives 24.99 against their 21.8 (protocol and resolution, about 3 points); BioCLIP 2 adds about 18; our fine-tune on North American open photos about 3 more, on species it never saw as Danish finds. No intervals yet: about ±2.2 points at 95% per model on 1,909 images.",
    ],
    source: E33_SOURCE },
];

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
