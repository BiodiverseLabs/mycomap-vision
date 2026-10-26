import { Link } from "wouter";
import { ArrowUpRight, CheckSquare, Clock, Code2, Database, Dna, FlaskConical, Hash, LineChart, Lock,
         Quote, ScanEye, Star, Target } from "lucide-react";
import type { EarlyAnswersGroup, HowGood, Me, RunReport, ScoreRun, Stats } from "@/lib/api";
import type { Outline } from "@/lib/apiOutline";
import type { ExperimentSummary } from "@/lib/experiments";
import { datasetOf } from "@/lib/experiments";
import { BENCHMARKS, CITATIONS, researchNav } from "@/lib/research";
import { utcDay, V1 } from "@/lib/datasetReleases";
import { MODEL_CREDIT_LINE } from "@/lib/publicView";
import { capitalize, publicMethodLabel } from "@/lib/publicNames";
import { standardOf } from "@/lib/compareView";
import { openModelHeadline, pointsDiff, TWO_MODELS } from "@/lib/twoModels";
import { earlyFigures, fineTuneVsBase, modelParts, nf, pc, placeVsPhotos, pointsGap, scoreboardCounts,
         shortDay, topkValues, type TopKRow } from "@/lib/researchView";
import { BandBars, ChartCard, More, PairedDiff, ResearchHero, ResearchShell, Section, SourceLine,
         StatusPill, Strip, Tile, Tiles, TopKChart } from "@/components/ResearchKit";
import { useEarly, useExperimentsIfMember, useHowGood, useMe, useOutline, useRunReports, useScoreboard,
         useStats } from "@/components/researchQueries";

// Research, the overview (Steve, 2026-10-25: the approved redesign): live headline numbers,
// the newest findings as cards, a map of the section, and how a number is made. Every number
// is read from a public feed (/api/how-good, /api/stats, /api/scoreboard, /api/early-answers,
// /api/docs/v1.json) or from a committed source named under it (twoModels.ts).

export interface LandingData {
  me?: Me | null;
  howGood?: HowGood;
  stats?: Stats;
  runs?: ScoreRun[];
  reports: Map<number, RunReport>;
  early?: EarlyAnswersGroup[];
  experiments?: ExperimentSummary[];
  outline?: Outline;
  loading?: boolean;
}

export function ResearchPage() {
  const me = useMe();
  const howGood = useHowGood();
  const stats = useStats();
  const board = useScoreboard();
  const early = useEarly();
  const experiments = useExperimentsIfMember();
  const outline = useOutline();
  const runs = board.data?.runs;
  const served = howGood.data?.run ? [howGood.data.run.backbone] : [];
  const method = howGood.data?.run?.method ?? "";
  const ft = runs ? fineTuneVsBase(runs, served) : null;
  const place = runs ? placeVsPhotos(runs, served, method) : null;
  const ids = [howGood.data?.run?.id, ft?.tuned.id, ft?.base.id, place?.place.id, place?.photos.id]
    .filter((x): x is number => x != null);
  const { reports } = useRunReports(ids);
  return (
    <ResearchLanding me={me.data} howGood={howGood.data} stats={stats.data} runs={runs} reports={reports}
                     early={early.data?.groups} experiments={experiments.data?.experiments}
                     outline={outline.data} loading={howGood.isLoading || stats.isLoading} />
  );
}

export function ResearchLanding(d: LandingData) {
  return (
    <ResearchShell me={d.me}>
      <Hero d={d} />
      <LatestResults d={d} />
      <SectionMap d={d} />
      <HowANumberIsMade />
    </ResearchShell>
  );
}

const servedRun = (d: LandingData) => (d.howGood?.run ? d.runs?.find((r) => r.id === d.howGood!.run!.id) : undefined);
const withPlace = (method?: string) => (method?.endsWith("+place") ? "photos + place and date" : "photos only");

function Hero({ d }: { d: LandingData }) {
  const h = d.howGood;
  const s = h?.top_k?.species;
  const g = h?.top_k?.genus;
  const run = h?.run;
  const model = run ? modelParts(run.backbone).name : null;
  const scored = servedRun(d);
  const week = d.stats?.recent_week;
  return (
    <ResearchHero title="Research" testId="landing-hero"
                  top={<span className="chip"><span className="dot" />Open to everyone · Updated nightly</span>}
                  lede="How MycoMap Vision is tested, on which records, and how it compares with other approaches. Every reference is a DNA-verified MycoMap record, and every number names its test.">
      <Tiles testId="landing-tiles">
        {s?.top1 != null && run ? (
          <>
            <Tile highlight value={pc(s.top1)} label="of new finds get the right species as the first answer"
                  note={`${model} · ${withPlace(run.method)} · ${nf(s.n)} finds`} />
            <Tile value={pc(s.top5)} label="have the right species among the first five" note="same test, top 5" />
            <Tile value={pc(g?.top1)} label="get the right genus first" note={`same test · ${nf(g?.n)} finds`} />
          </>
        ) : (
          <p className="small col-span-3" data-testid="landing-no-results">
            {d.loading ? "Reading the latest results…" : "The latest results could not be read just now."}
          </p>
        )}
        {d.stats ? (
          <Tile value={nf(d.stats.records_north_america)} label="DNA-verified North American records to learn from"
                note={`live reference set${week ? ` · +${nf(week.records)} in the week to ${shortDay(week.through)}` : ""}`} />
        ) : null}
      </Tiles>
      {run && (
        <div className="meta">
          <span>Test: <b>{scored?.dataset_release ? `dataset release ${scored.dataset_release}, validation split` : "the newest validations"}</b></span>
          <span><b>{nf(run.n_test)}</b> finds first validated after {utcDay(run.cutoff)}</span>
          <span>identified from the <b>{nf(run.n_reference)}</b> records validated before them</span>
        </div>
      )}
      <SourceLine parts={[run ? `/api/how-good (run ${run.id}${scored ? `, measured ${shortDay(scored.created_at)}` : ""})` : null,
                          d.stats ? "/api/stats" : null]} />
    </ResearchHero>
  );
}

function ResultCard({ when, tag, title, children, foot, more, testId }: {
  when: React.ReactNode; tag: React.ReactNode; title: React.ReactNode; children: React.ReactNode;
  foot: React.ReactNode; more: { href: string; label: string }; testId: string;
}) {
  return (
    <div className="rc" data-testid={testId}>
      <div className="when"><span>{when}</span>{typeof tag === "string" ? <span className="tag">{tag}</span> : tag}</div>
      <h3 className="font-display font-semibold text-[#4a3728]">{title}</h3>
      <div className="fig">{children}</div>
      <div className="foot">{foot}</div>
      <More href={more.href}>{more.label}</More>
    </div>
  );
}

function rowOf(key: string, name: string, detail: string | undefined, report: RunReport | undefined, highlight = false): TopKRow | null {
  const sp = standardOf(report)?.species?.all;
  return sp ? { key, name, detail, n: sp.n, values: topkValues(sp), highlight } : null;
}

function LatestResults({ d }: { d: LandingData }) {
  const h = d.howGood;
  const run = h?.run;
  const scored = servedRun(d);
  const tag = scored?.dataset_release ? `${scored.dataset_release} val` : "newest";
  const served = run ? [run.backbone] : [];
  const ft = d.runs ? fineTuneVsBase(d.runs, served) : null;
  const place = d.runs && run ? placeVsPhotos(d.runs, served, run.method) : null;
  const report = run ? d.reports.get(run.id) : undefined;
  const refSpecies = report?.reference_species;
  const rare = refSpecies?.bands?.["1-4"] != null && refSpecies.species
    ? Math.round((100 * refSpecies.bands["1-4"]) / refSpecies.species) : null;
  const [site, open] = TWO_MODELS.models;
  const heads = openModelHeadline();
  const e = earlyFigures(d.early);
  const ftRows = ft ? [rowOf("ft", modelParts(ft.tuned.backbone).name, undefined, d.reports.get(ft.tuned.id), true),
                       rowOf("base", modelParts(ft.base.backbone).name.replace(" (base model, not fine-tuned)", ""),
                             "base, not fine-tuned", d.reports.get(ft.base.id))].filter((r): r is TopKRow => !!r) : [];
  const placeRows = place ? [rowOf("place", "With place + date", undefined, d.reports.get(place.place.id), true),
                             rowOf("photos", "Photos only", undefined, d.reports.get(place.photos.id))].filter((r): r is TopKRow => !!r) : [];
  const ids = [run?.id, ...(ft ? [ft.tuned.id, ft.base.id] : []), ...(place ? [place.place.id, place.photos.id] : [])]
    .filter((x): x is number => x != null);
  return (
    <Section kicker="Latest results" title="What the newest tests say" testId="section-latest"
             lead="Each card is one finding, with the test it comes from. All are on dataset release v1's validation split unless the card says otherwise; nothing here comes from the sealed test.">
      <div className="rgrid">
        {h?.top_k?.species && run && (
          <ResultCard testId="card-served" when={scored ? shortDay(scored.created_at) : "Latest"} tag={tag}
                      title={`${modelParts(run.backbone).name} on the newest validations`}
                      foot={`Species top 1 / 3 / 5 / 10, ${withPlace(run.method)}, ${nf(h.top_k.species.n)} finds.`}
                      more={{ href: "/research/results", label: "Results" }}>
            <TopKChart variant="mini" rows={[{ key: "s", name: modelParts(run.backbone).name, n: h.top_k.species.n,
                                               values: topkValues(h.top_k.species) }]} />
          </ResultCard>
        )}
        <ResultCard testId="card-two-models" when="v1 val" tag="paired, by observer"
                    title={`Open data only costs ${Math.abs(100 * TWO_MODELS.paired.diff).toFixed(2)} points`}
                    foot={<>Species top 1, photos only, {" "}against the full reference. v1-open minus v1: {pointsDiff(TWO_MODELS.paired.diff, TWO_MODELS.paired.ci95)} points, {nf(TWO_MODELS.paired.n)} finds from {nf(TWO_MODELS.paired.observers)} observers. Against the open-data reference only (the paper's headline setup), v1-open names {pc(heads.open.top1)}.</>}
                    more={{ href: "/research/results#two", label: "The two models" }}>
          <div className="flex justify-between text-[13px] text-[#4a3728]">
            <span>v1 <b>{pc(site.species.top1)}</b></span><span>v1-open <b>{pc(open.species.top1)}</b></span>
          </div>
          <PairedDiff diff={TWO_MODELS.paired.diff} ci={TWO_MODELS.paired.ci95}
                      tip={`v1-open minus v1, species top 1|${pointsDiff(TWO_MODELS.paired.diff, TWO_MODELS.paired.ci95)} points`} />
        </ResultCard>
        {(h?.species_by_depth?.length ?? 0) > 0 && (
          <ResultCard testId="card-depth" when={scored ? shortDay(scored.created_at) : "Latest"} tag={tag}
                      title="Reference depth decides, more than method"
                      foot={<>Species top 1 by how many DNA-verified records the true species has (0 · 1-4 · 5-19 · 20-99 · 100+).{rare != null && ` ${rare}% of reference species have 1-4.`}</>}
                      more={{ href: "/research/benchmarks/rare-species", label: "Rare species benchmark" }}>
            <BandBars bands={(h!.species_by_depth ?? []).filter((b) => b.top1 != null).map((b) => ({ band: b.band, top1: b.top1 as number, n: b.n }))} />
          </ResultCard>
        )}
        {ft && ftRows.length === 2 && (
          <ResultCard testId="card-fine-tune" when={shortDay(ft.tuned.created_at)} tag={tag}
                      title={`Fine-tuning adds ${pointsGap(ftRows[0].values[1]!, ftRows[1].values[1]!)} points over the base model`}
                      foot={`${capitalize(publicMethodLabel(ft.tuned.method))}, photos only, both ${modelParts(ft.base.backbone).detail}, the same ${nf(ftRows[0].n)} finds.`}
                      more={{ href: "/research/results#approaches", label: "Approaches compared" }}>
            <TopKChart rows={ftRows} legend={false} labelWidth={110} numsWidth={150} testId="chart-fine-tune" />
          </ResultCard>
        )}
        {place && placeRows.length === 2 && (
          <ResultCard testId="card-place" when={shortDay(place.place.created_at)} tag={tag}
                      title={`Place and date add ${pointsGap(placeRows[0].values[1]!, placeRows[1].values[1]!)} points`}
                      foot={`${modelParts(place.place.backbone).name}, scored by specimen, ${nf(placeRows[0].n)} finds; place is the find's public place.`}
                      more={{ href: "/research/results#scoreboard", label: "Scoreboard" }}>
            <TopKChart rows={placeRows} legend={false} labelWidth={110} numsWidth={150} testId="chart-place" />
          </ResultCard>
        )}
        {e && (
          <ResultCard testId="card-early" when="Live" tag={<StatusPill status="running" />}
                      title="Answers kept before the DNA verdict"
                      foot="Scored against the DNA name once each record turns green. Never a model's training records."
                      more={{ href: "/research/results#early", label: "Answered before the verdict" }}>
            <div className="bigstat">{nf(e.answers)} <small>answers kept</small></div>
            <p className="small mt-2">{nf(e.awaiting)} awaiting their verdict · {nf(e.afterVerdict)} left out (answered after it) · {nf(e.scored)} scored so far.</p>
          </ResultCard>
        )}
      </div>
      <SourceLine label="Sources" parts={[ids.length ? `/api/scoreboard/${ids.join(", ")} (standard summary)` : null,
                                          `twoModels.ts (${TWO_MODELS.source})`, e ? "/api/early-answers" : null]} />
    </Section>
  );
}

function MapCard({ href, icon, brown, title, text, fact, members }: {
  href: string; icon: React.ReactNode; brown?: boolean; title: string; text: string; fact: React.ReactNode; members?: boolean;
}) {
  return (
    <Link href={href} className="sm" data-testid={`map-${href.split("/").pop() || "research"}`}>
      <span className={`ic ${brown ? "b" : ""}`}>{icon}</span>
      <span className="go"><ArrowUpRight className="h-4 w-4" /></span>
      <h3 className="font-display font-semibold text-[#4a3728]">{title}{members && <> <span className="pill members ml-1"><Lock className="h-[11px] w-[11px]" /> Members</span></>}</h3>
      <p>{text}</p>
      <div className="fact">{fact}</div>
    </Link>
  );
}

function SectionMap({ d }: { d: LandingData }) {
  const nav = researchNav(d.me);
  const has = (href: string) => nav.some((n) => n.href === href);
  const counts = d.runs ? scoreboardCounts(d.runs) : null;
  const running = BENCHMARKS.filter((b) => b.status === "running").length;
  const endpoints = d.outline?.sections.flatMap((s) => s.blocks).filter((b) => b.type === "endpoint") ?? [];
  const available = endpoints.filter((b) => b.type === "endpoint" && b.status === "available").length;
  const xs = d.experiments;
  const icon = "h-[18px] w-[18px]";
  return (
    <Section band kicker="Section map" title="Everything in Research" testId="section-map"
             lead="Start with Results for the numbers, Benchmarks for what each test measures, and Evaluation protocols for how every number is scored.">
      <div className="smap">
        <MapCard href="/research/results" icon={<LineChart className={icon} />} title="Results"
                 text="Every approach, MycoMap Vision and published models, scored on the same DNA-verified records."
                 fact={counts ? <><b>{nf(counts.runs)}</b>scored runs in {nf(counts.comparisons)} comparisons</> : "Every scored run"} />
        <MapCard href="/research/benchmarks" icon={<Star className={icon} />} title="Benchmarks"
                 text="The named test sets: what each measures, its records, and its latest number."
                 fact={<><b>{BENCHMARKS.length}</b>benchmarks · {running} running</>} />
        <MapCard href="/research/protocols" icon={<CheckSquare className={icon} />} title="Evaluation protocols"
                 text="How every result is scored, so numbers from different tests and months compare."
                 fact={<><b>4 × 3 × 5</b>top-k × ranks × depth bands</>} />
        <MapCard href="/research/data" icon={<Database className={icon} />} title="Reference data"
                 text="The frozen release behind every result, and the live reference set that grows nightly."
                 fact={<><b>{nf(V1.records.included)}</b>records in release {V1.id}</>} />
        <MapCard href="/research/experiments" icon={<FlaskConical className={icon} />} brown members title="Experiments"
                 text="The registry: every question, everything tried, the numbers and the decision."
                 fact={!has("/research/experiments") ? "For mycomap.org members: sign in to read"
                   : xs ? <><b>{nf(xs.length)}</b>entries · {nf(xs.filter((x) => datasetOf(x) !== "pre-v1").length)} rebuilt on v1</> : "Reading the registry…"} />
        <MapCard href="/research/excluded-photos" icon={<ScanEye className={icon} />} title="Excluded photos"
                 text="Photos people left out of Vision, by why, and where Vision looked on any of them."
                 fact="Micrographs, field slips, habitat and more" />
        <MapCard href="/research/for-researchers" icon={<Quote className={icon} />} title="For researchers"
                 text="How to cite, the model names, and how to score your own model on the same records."
                 fact={<><b>{CITATIONS.length}</b>citations, copy-ready</>} />
        <MapCard href="/research/api" icon={<Code2 className={icon} />} title="API"
                 text="Identify fungi from your own app or site: access, requests and the answer, field by field."
                 fact={endpoints.length ? <><b>{available}</b>of {endpoints.length} endpoints available</> : "The developer API"} />
        <div className="sm" style={{ background: "linear-gradient(160deg,#fff,#f6f9ef)" }} data-testid="map-built-on-dna">
          <span className="ic"><Dna className={icon} /></span>
          <h3 className="font-display font-semibold text-[#4a3728]">Built on DNA</h3>
          <p data-testid="text-model-credit-line">{MODEL_CREDIT_LINE}</p>
          <div className="fact">{d.stats ? <><b>{nf(d.stats.names)}</b>names{d.stats.names_provisional != null && `, ${nf(d.stats.names_provisional)} provisional`}</> : "DNA-verified names"}</div>
        </div>
      </div>
    </Section>
  );
}

function HowANumberIsMade() {
  const icon = "h-[18px] w-[18px]";
  return (
    <Section kicker="How a number is made" title="Frozen data, newest finds, one protocol" testId="section-how-made">
      <Strip steps={[
        { icon: <Hash className={icon} />, title: "Freeze a release", text: `Dataset release ${V1.id}: ${nf(V1.records.included)} records, hashed, never changed.` },
        { icon: <Clock className={icon} />, title: "Test on the newest", text: `Records first validated after ${utcDay(V1.valCutoff)}, identified only from earlier ones.` },
        { icon: <Target className={icon} />, title: "Score it one way", text: "Top 1, 3, 5 and 10 at species, genus and family, by reference depth." },
      ]} />
      <p className="text-sm text-muted-foreground leading-normal mt-2.5">
        The paper's sealed test (v1-test) is scored once, after every choice is frozen; no number from it appears on this site until then.{" "}
        <Link href="/research/protocols" className="text-[#4f7a16] underline">Evaluation protocols →</Link>
      </p>
    </Section>
  );
}

