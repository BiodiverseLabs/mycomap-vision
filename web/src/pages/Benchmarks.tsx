import { useEffect, useMemo, useState } from "react";
import { Link } from "wouter";
import { ArrowUpRight, Check, Lock } from "lucide-react";
import type { EarlyAnswersGroup, HowGood, Me, ProspectiveModel, RunReport, ScoreRun } from "@/lib/api";
import { comparisons, incompleteRun, standardOf, STANDARD_BANDS, companionBars } from "@/lib/compareView";
import { BASE_MODEL_SCREEN, BENCHMARKS, cellDigits, DATA_LABEL, DATA_ORDER, diffsHeading, MEASURE_LABEL, MEASURE_ORDER,
         resultCards, STATUS_LABEL, STATUS_ORDER, TEST_SET_RESULTS, valueCaption, type Benchmark, type BenchmarkStatus,
         type TestSetResults } from "@/lib/research";
import { benchmarkFacetCounts, cardFigure, cardMeta, earlyFigures, filterBenchmarks, filterFromSearch,
         filterToSearch, heatRows, historyLines, isFiltered, modelParts, NO_FILTER, nf, pc, prospectiveLead, pt,
         placeVsPhotos, approachGroups, shortDay, statusBar, toggle, topkValues, type BenchmarkFeeds,
         type BenchmarkFilter, type TopKRow } from "@/lib/researchView";
import { BandBars, ChartCard, ChipRow, DepthHeatmap, DepthShareBars, DevLabel, InPageNav, Interim, KV,
         Pipeline, ResearchHero, ResearchShell, SearchBox, Section, SourceLine, StatusPill, Tile, Tiles,
         TopKChart, ValueBars } from "@/components/ResearchKit";
import { useEarly, useHowGood, useMe, useProspective, useRunReports, useScoreboard } from "@/components/researchQueries";

// Research > Benchmarks and each benchmark's page (Steve, 2026-10-25: the approved redesign).
// The list's text and facets are lib/research.ts BENCHMARKS; every number on a card is read
// from the public feeds (/api/how-good, /api/scoreboard, /api/prospective, /api/early-answers).
// The summary's base-model screen and results by test set quote committed research entries.

export const STATUS_SWATCH: Record<BenchmarkStatus, string> = {
  running: "#7FAE34", partial: "#c27c2e", paused: "#a8968a", sealed: "#2f241b", planned: "#cfc3b6", history: "#e8e1d8",
};

export interface BenchmarksData extends BenchmarkFeeds { me?: Me | null; loading?: boolean }

function useFeeds() {
  const me = useMe();
  const howGood = useHowGood();
  const board = useScoreboard();
  const prospective = useProspective();
  const early = useEarly();
  return { me: me.data, howGood: howGood.data, runs: board.data?.runs, prospective: prospective.data?.models,
           early: early.data?.groups, loading: howGood.isLoading || board.isLoading };
}

const initialFilter = () => (typeof window === "undefined" ? NO_FILTER : filterFromSearch(window.location.search));

export function BenchmarksPage() {
  const d = useFeeds();
  const [filter, setFilter] = useState<BenchmarkFilter>(initialFilter);
  useEffect(() => {
    const url = `${window.location.pathname}${filterToSearch(filter)}${window.location.hash}`;
    window.history.replaceState(window.history.state, "", url);
  }, [filter]);
  return <Benchmarks {...d} filter={filter} onFilter={setFilter} />;
}

export function Benchmarks(d: BenchmarksData & { filter?: BenchmarkFilter; onFilter?: (f: BenchmarkFilter) => void }) {
  const [own, setOwn] = useState<BenchmarkFilter>(d.filter ?? NO_FILTER);
  const filter = d.filter ?? own;
  const setFilter = d.onFilter ?? setOwn;
  return (
    <ResearchShell me={d.me}>
      <ResearchHero crumbs={[{ href: "/research", label: "Research" }, { label: "Benchmarks" }]} title="Benchmarks"
                    lede="The named test sets. Each measures one thing, on DNA-verified records the identifier did not learn from, and is scored the same way every time.">
        <div className="glance" style={{ marginTop: 24 }}>
          <Glance d={d} />
          <BaseModelScreen />
        </div>
        <ResultsByTestSet />
      </ResearchHero>
      <Section testId="section-benchmark-list">
        <FilterBar filter={filter} onFilter={setFilter} />
        <BenchmarkGrid d={d} filter={filter} onFilter={setFilter} />
      </Section>
    </ResearchShell>
  );
}

function Glance({ d }: { d: BenchmarksData }) {
  const bar = statusBar();
  const total = BENCHMARKS.length;
  const s = d.howGood?.top_k?.species;
  const depth = d.howGood?.species_by_depth ?? [];
  const lo = depth.find((b) => b.band === "1-4"), hi = depth.find((b) => b.band === "100+");
  const run = d.howGood?.run;
  const lead = prospectiveLead(d.prospective, run?.backbone);
  const advance = BENCHMARKS.find((b) => b.slug === "advance-predictions");
  const e = earlyFigures(d.early);
  const lines: [string, string][] = [];
  if (s?.top1 != null) lines.push(["Newest weeks · species top 1", pc(s.top1)], ["Newest weeks · species top 5", pc(s.top5)]);
  if (lo?.top1 != null && hi?.top1 != null) lines.push(["Rare species · 1-4 → 100+ records", `${pt(lo.top1)} → ${pc(hi.top1)}`]);
  if (advance?.status === "paused") lines.push(["Advance predictions", advance.lastRun ?? STATUS_LABEL.paused]);
  else if (lead) lines.push(["Advance predictions · resolved", `${nf(lead.row.resolved)} of ${nf(lead.row.predicted)}`]);
  if (e) lines.push(["Answered before the verdict · kept", nf(e.answers)]);
  lines.push(["Sealed test", "opens once, after the freeze"]);
  return (
    <div className="min-w-0 rounded-[10px] border border-[#A87146]/20 bg-white p-5 flex flex-col" data-testid="benchmarks-glance">
      <h2 className="text-xs font-semibold uppercase tracking-[.08em] text-[#6b8f2a] mb-1">At a glance</h2>
      <div className="bigstat">{total} <small>benchmarks</small></div>
      <div className="mb-2.5 mt-3.5 flex h-3 gap-[2px]" data-testid="status-bar">
        {bar.map((x, i) => (
          <div key={x.status} data-tip={`${STATUS_LABEL[x.status]}|${x.n} benchmark${x.n === 1 ? "" : "s"}`}
               style={{ flex: x.n, background: STATUS_SWATCH[x.status], borderRadius: i === 0 ? "4px 0 0 4px" : i === bar.length - 1 ? "0 4px 4px 0" : undefined }} />
        ))}
      </div>
      <div className="legend" style={{ margin: 0 }}>
        {bar.map((x) => <span key={x.status}><i style={{ background: STATUS_SWATCH[x.status] }} />{STATUS_LABEL[x.status]} <b style={{ color: "var(--ink)" }}>{x.n}</b></span>)}
      </div>
      <div className="mt-4 border-t border-[#ece6de] pt-1.5">
        {lines.map(([k, v]) => <div key={k} className="gl"><span>{k}</span><b>{v}</b></div>)}
      </div>
      <p className="small mt-2.5">{run && s ? `Newest weeks: ${modelParts(run.backbone).name} on ${nf(s.n)} finds, ${run.method.endsWith("+place") ? "with each find's photos and its public place and date" : "photos only"}. ` : ""}Every setting is chosen on the validation split; the sealed test is opened once.</p>
      <SourceLine parts={["research.ts BENCHMARKS (statuses)", run ? "/api/how-good" : null, lead ? "/api/prospective" : null, e ? "/api/early-answers" : null]} />
    </div>
  );
}

/** The base-model screen (lib/research.ts BASE_MODEL_SCREEN), labelled a development screen. */
export function BaseModelScreen({ rows = BASE_MODEL_SCREEN.rows }: { rows?: typeof BASE_MODEL_SCREEN.rows }) {
  const chartRows: TopKRow[] = rows.map((r) => ({ key: r.name, name: r.name, detail: r.run ? `${r.detail} · ${r.run}` : r.detail, n: BASE_MODEL_SCREEN.queries,
    values: { 1: r.top1 / 100, 3: r.top3 / 100, 5: r.top5 / 100, 10: r.top10 / 100 } }));
  return (
    <div className="min-w-0 rounded-[10px] border border-[#A87146]/20 bg-white p-5" data-testid="base-model-screen">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs font-semibold uppercase tracking-[.08em] text-[#6b8f2a]">Base-model screen</p><DevLabel />
      </div>
      <h3 className="font-display font-semibold text-lg leading-snug text-[#4a3728] mt-1.5 mb-0.5">
        MycoMap v1 validation subset ({nf(BASE_MODEL_SCREEN.queries)} queries)
      </h3>
      <p className="small mb-3">Frozen base models (not fine-tuned), closest specimens + species average, photos only, on the same {nf(BASE_MODEL_SCREEN.queries)} deciding queries.</p>
      <TopKChart rows={chartRows} labelHead="Base model" trackHead="Species" labelWidth={170} numsWidth={160} testId="chart-base-screen" />
      <p className="small mt-2.5">{rows.some((r) => r.run) && <>Every row is one run except where it names its own (that run's rows sit within 0.2 points of this one). </>}A v1 validation subset whose references cover {nf(BASE_MODEL_SCREEN.species)} species, so each query has far fewer candidate species than on the full split: not comparable with the numbers above. A top 1 near 70% on {nf(BASE_MODEL_SCREEN.queries)} queries carries about ±3.5 points (95%).</p>
      <SourceLine parts={[BASE_MODEL_SCREEN.source, "comparison 20261025-163049-1a3990", "dataset release v1", "seed 20261013"]} />
    </div>
  );
}

/** "Results by test set": every model we evaluate, on each test set it has been scored on
 *  (lib/research.ts TEST_SET_RESULTS). One card per test set, or per group of them where the
 *  data aggregates them; nothing here is new card code per test set or model. */
export function ResultsByTestSet({ sets = TEST_SET_RESULTS }: { sets?: TestSetResults[] }) {
  const cards = resultCards(sets);
  if (!cards.length) return null;
  return (
    <div className="mt-3.5" data-testid="results-by-test-set">
      <h2 className="text-xs font-semibold uppercase tracking-[.08em] text-[#6b8f2a] mb-1">Results by test set</h2>
      <p className="small mb-2">Every model scored on each test set, from the committed research entries. The sealed test (v1-test) has no results until it is opened.</p>
      <div className="grid gap-3.5">
        {cards.map((c) => <ResultCard key={c.title} title={c.title} sets={c.sets} />)}
      </div>
    </div>
  );
}

const fx = (v: number | null, digits: number) => (v == null ? "–" : v.toFixed(digits));
/** A difference in points, signed, with a true minus sign. */
const sgn = (v: number) => `${v < 0 ? "−" : "+"}${Math.abs(v).toFixed(2)}`;
const neg = (v: number, digits = 2) => (v < 0 ? `−${Math.abs(v).toFixed(digits)}` : v.toFixed(digits));
const SUB = "block whitespace-nowrap text-[11.5px] text-[var(--muted)] font-normal";

function ResultCard({ title, sets }: { title: string; sets: TestSetResults[] }) {
  const first = sets[0];
  const sources = [...new Set(sets.map((s) => s.source))];
  return (
    <div className="min-w-0 rounded-[10px] border border-[#A87146]/20 bg-white px-[18px] py-4 flex flex-col" data-testid={`result-card-${first.key}`}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <b className="text-[#4a3728]">{title}</b>
        <span className="tag">{first.maker}</span>
      </div>
      <p className="small mt-1.5">{first.about}</p>
      <p className="small mt-0.5" style={{ color: "var(--muted)" }}>{first.size}</p>
      <p className="small mt-1.5" data-testid="value-caption"><b style={{ color: "var(--ink)" }}>{valueCaption(first)}</b></p>
      {sets.map((s) => <ResultSection key={s.key} s={s} lead={s === first ? undefined : s} first={first} />)}
      <div className="mt-auto"><SourceLine parts={sources} /></div>
    </div>
  );
}

/** One test set under one condition: its table, paired differences, checks and notes. A
 *  section after the card's first says what it measures and its size only where they differ. */
function ResultSection({ s, first, lead }: { s: TestSetResults; first: TestSetResults; lead?: TestSetResults }) {
  return (
    <div className="mt-4" data-testid={`result-set-${s.key}`}>
      <div className="flex flex-wrap items-center gap-2 border-t border-[#ece6de] pt-3">
        {s.condition && <span className="text-[13px] font-semibold text-[#4a3728]">{s.name !== first.name ? `${s.name} · ` : ""}{s.condition}</span>}
        {s.interim && <Interim>Interim · val only</Interim>}
      </div>
      {lead && lead.about !== first.about && <p className="small mt-1">{lead.about}</p>}
      {lead && lead.size !== first.size && <p className="small mt-0.5" style={{ color: "var(--muted)" }}>{lead.size}</p>}
      {s.checks?.map((c) => (
        <p key={c} className="small mt-2 flex gap-1.5" data-testid="scoring-check"><Check size={14} className="mt-[3px] shrink-0 text-[#6b8f2a]" />{c}</p>
      ))}
      <ResultTable s={s} />
      {s.diffs && (
        <div className="mt-3" data-testid={`diffs-${s.key}`}>
          <p className="small mb-1"><b style={{ color: "var(--ink)" }}>{diffsHeading(s)}</b></p>
          <div className="scrollx"><table className="dt [&_td]:!px-1.5 [&_th]:!px-1.5">
            <thead><tr><th className="l min-w-[220px]">Difference</th>{s.diffs.columns.map((h) => <th key={h}>{h}</th>)}</tr></thead>
            <tbody>{s.diffs.rows.map((r) => (
              <tr key={r.name}>
                <td className="l">{r.name}</td>
                {r.cells.map((x, k) => <td key={k} className="whitespace-nowrap">{sgn(x.d)}<small className={SUB}>[{neg(x.lo)}, {neg(x.hi)}]</small></td>)}
              </tr>
            ))}</tbody>
          </table></div>
        </div>
      )}
      {s.notes.map((n, i) => <p key={i} className="small mt-2">{n}</p>)}
    </div>
  );
}

/** The models' table: columns under their group headings, a value's 95% interval beneath it. */
function ResultTable({ s }: { s: TestSetResults }) {
  const spans: { group?: string; label: string; n: number }[] = [];
  for (const c of s.columns) {
    const last = spans[spans.length - 1];
    if (last && c.group && last.group === c.group) last.n++;
    else spans.push({ group: c.group, label: c.label, n: 1 });
  }
  const grouped = spans.some((g) => g.group);
  return (
    <div className="scrollx mt-2"><table className="dt [&_td]:!px-1.5 [&_th]:!px-1.5">
      <thead>
        <tr>
          <th className="l min-w-[170px]" rowSpan={grouped ? 2 : 1}>Model</th>
          {spans.map((g, i) => g.group
            ? <th key={i} colSpan={g.n} style={{ textAlign: "center" }}>{g.group}</th>
            : <th key={i} rowSpan={grouped ? 2 : 1}>{g.label}</th>)}
        </tr>
        {grouped && <tr>{s.columns.filter((c) => c.group).map((c, i) => <th key={i}>{c.label}</th>)}</tr>}
      </thead>
      <tbody>{s.rows.map((r, i) => (
        <tr key={i}>
          <td className="l">{r.name}{r.detail && <small>{r.detail}</small>}</td>
          {r.cells.map((v, k) => {
            const ci = r.cis?.[k];
            return (
              <td key={k} className={v == null ? "mut" : s.columns[k]?.label === "Top 1" ? "strong" : undefined}>
                {fx(v, cellDigits(s, r, k))}{ci && <small className={SUB}>{ci[0].toFixed(cellDigits(s, r, k))}–{ci[1].toFixed(cellDigits(s, r, k))}</small>}
              </td>
            );
          })}
        </tr>
      ))}</tbody>
    </table></div>
  );
}

function FilterBar({ filter, onFilter }: { filter: BenchmarkFilter; onFilter: (f: BenchmarkFilter) => void }) {
  const counts = useMemo(() => benchmarkFacetCounts(), []);
  return (
    <div className="fbar" data-testid="filter-bar">
      <div className="row"><SearchBox value={filter.q} onChange={(q) => onFilter({ ...filter, q })}
                                      placeholder="Search benchmarks: rare, sealed, iNaturalist…" label="Search benchmarks" /></div>
      <ChipRow label="Status" values={STATUS_ORDER} chosen={filter.status} counts={counts.status} words={STATUS_LABEL}
               onToggle={(v) => onFilter({ ...filter, status: toggle(filter.status, v) })} />
      <ChipRow label="Measures" values={MEASURE_ORDER} chosen={filter.measures} counts={counts.measures} words={MEASURE_LABEL}
               onToggle={(v) => onFilter({ ...filter, measures: toggle(filter.measures, v) })} />
      <ChipRow label="Data" values={DATA_ORDER} chosen={filter.data} counts={counts.data} words={DATA_LABEL}
               onToggle={(v) => onFilter({ ...filter, data: toggle(filter.data, v) })} />
    </div>
  );
}

function BenchmarkGrid({ d, filter, onFilter }: { d: BenchmarksData; filter: BenchmarkFilter; onFilter: (f: BenchmarkFilter) => void }) {
  const shown = filterBenchmarks(BENCHMARKS, filter);
  const clear = () => onFilter({ ...NO_FILTER, sort: filter.sort });
  return (
    <>
      <div className="fsum">
        <span data-testid="benchmarks-shown">{isFiltered(filter) ? `Showing ${shown.length} of ${BENCHMARKS.length} benchmarks` : `Showing all ${BENCHMARKS.length} benchmarks`}</span>
        <span className="flex items-center gap-2.5">
          {isFiltered(filter) && <button type="button" className="clear" onClick={clear} data-testid="button-clear">Clear filters</button>}
          <label>Sort <select value={filter.sort} onChange={(e) => onFilter({ ...filter, sort: e.target.value === "name" ? "name" : "status" })}
                              aria-label="Sort benchmarks">
            <option value="status">Status</option><option value="name">Name</option></select></label>
        </span>
      </div>
      <div className="bgrid" data-testid="benchmark-grid">
        {shown.map((b) => <BenchmarkCard key={b.slug} b={b} d={d} />)}
      </div>
      {!shown.length && (
        <p className="small py-8 text-center" data-testid="benchmarks-none">No benchmark matches. <button type="button" className="clear" onClick={clear}>Clear filters</button></p>
      )}
      <SourceLine label="Facets" parts={["research.ts BENCHMARKS: measures, data, size, lastRun, headline (run + metric)",
                                        "Numbers: /api/how-good, /api/scoreboard, /api/prospective, /api/early-answers"]} />
    </>
  );
}

export function BenchmarkCard({ b, d }: { b: Benchmark; d: BenchmarkFeeds & { loading?: boolean } }) {
  const fig = cardFigure(b, d);
  const meta = cardMeta(b, d);
  return (
    <Link href={`/research/benchmarks/${b.slug}`} className="bcard" data-testid={`benchmark-card-${b.slug}`}>
      <div className="top"><StatusPill status={b.status} /><span className="tag d ds">{DATA_LABEL[b.data[0]]}</span></div>
      <h3 className="font-display font-semibold text-[#4a3728]">{b.name}</h3>
      <p className="sum">{b.summary}</p>
      {fig ? <Figure fig={fig} /> : (
        <div className="num empty" data-testid="card-empty">
          <div className="msg">
            {b.pending ? <><b>{b.pending.lead}</b> {b.pending.text}</>
              : b.headline ? (d.loading ? "Reading the latest number…" : "The latest number could not be read just now.")
              : "Not measured yet."}
          </div>
        </div>
      )}
      <div className="meta"><div>Dataset<b>{meta.dataset}</b></div><div>Size<b>{meta.size}</b></div><div>Last run<b>{meta.lastRun}</b></div></div>
      <div className="tags">{b.measures.map((m) => <span key={m} className="tag">{MEASURE_LABEL[m]}</span>)}</div>
    </Link>
  );
}

function Figure({ fig }: { fig: NonNullable<ReturnType<typeof cardFigure>> }) {
  if (fig.kind === "topk") {
    return (
      <div className="num" data-testid="card-figure">
        <div className="big">{fig.big}<small>{fig.caption}</small></div>
        <TopKChart variant="mini" rows={[{ key: "x", name: "Species", n: fig.n, values: fig.values }]} testId="card-mini" />
      </div>
    );
  }
  if (fig.kind === "depth") {
    return (
      <div className="num" data-testid="card-figure">
        <div className="big">{fig.from}<span style={{ fontSize: 20, color: "var(--muted)" }}> → </span>{fig.to}<small>{fig.caption}</small></div>
        <BandBars bands={fig.bands} testId="card-bands" />
      </div>
    );
  }
  return (
    <div className="num" data-testid="card-figure">
      <div className="big">{fig.big}{fig.of && <small style={{ display: "inline", fontSize: 18, color: "var(--muted)" }}> {fig.of}</small>}<small>{fig.caption}</small></div>
      <div className="small">{fig.note}</div>
    </div>
  );
}

// --- one benchmark ---------------------------------------------------------------------------

export const DETAIL_SECTIONS = [
  { id: "question", label: "Question" }, { id: "protocol", label: "Protocol" }, { id: "dataset", label: "Dataset" },
  { id: "results", label: "Results" }, { id: "history", label: "History" }, { id: "links", label: "Links" },
];

export interface BenchmarkDetailData extends BenchmarksData { reports: Map<number, RunReport> }

export function BenchmarkPage({ slug }: { slug: string }) {
  const d = useFeeds();
  const b = BENCHMARKS.find((x) => x.slug === slug);
  const run = d.howGood?.run;
  const served = run ? [run.backbone] : [];
  const runs = d.runs ?? [];
  const place = run ? placeVsPhotos(runs, served, run.method) : null;
  const heat = approachGroups(runs, served, run?.method ?? "").flatMap((g) => g.runs);
  const hist = historyLines(comparisons(runs.filter((r) => !incompleteRun(r)))).slice(0, 8);
  const usesRun = b?.headline?.run != null;
  const ids = usesRun ? [run?.id, place?.photos.id, ...heat.map((r) => r.id), ...hist.map((h) => h.best.id)].filter((x): x is number => x != null) : [];
  const { reports } = useRunReports(ids);
  if (!b) return <ResearchShell me={d.me}><ResearchHero crumbs={[{ href: "/research", label: "Research" }, { href: "/research/benchmarks", label: "Benchmarks" }]} title="Benchmark not found" /></ResearchShell>;
  return <BenchmarkDetail b={b} {...d} reports={reports} />;
}

export function BenchmarkDetail({ b, ...d }: BenchmarkDetailData & { b: Benchmark }) {
  const run = d.howGood?.run;
  const report = run ? d.reports.get(run.id) : undefined;
  const scored = run ? d.runs?.find((r) => r.id === run.id) : undefined;
  const fromRun = b.headline?.run != null;
  return (
    <ResearchShell me={d.me}>
      <ResearchHero crumbs={[{ href: "/research", label: "Research" }, { href: "/research/benchmarks", label: "Benchmarks" }, { label: b.name }]}
                    title={b.longName ?? b.name} lede={b.summary} testId="benchmark-hero"
                    top={<div className="flex flex-wrap items-center gap-2"><StatusPill status={b.status} />
                      {b.data.map((x) => <span key={x} className="tag d">{DATA_LABEL[x]}</span>)}
                      {b.measures.map((m) => <span key={m} className="tag">{MEASURE_LABEL[m]}</span>)}</div>}>
        <DetailTiles b={b} d={d} />
        {fromRun && run && (
          <div className="meta">
            <span>{modelParts(run.backbone).name} · {run.method.endsWith("+place") ? "photos + place and date" : "photos only"}</span>
            {scored?.dataset_release && <span>Dataset release {scored.dataset_release}, validation split</span>}
            {scored && <span>Last run <b>{shortDay(scored.created_at)}</b> (run #{run.id})</span>}
          </div>
        )}
      </ResearchHero>
      <InPageNav sections={DETAIL_SECTIONS} />
      <Section id="question" kicker="1 · The question">
        <div className="split2">
          <div>
            <h2 className="font-display font-semibold text-[25px] sm:text-[30px] leading-tight text-[#4a3728] mb-2">{b.question}</h2>
            <p className="text-base">{b.whyItMatters}</p>
            <RareCallout b={b} h={d.howGood} />
          </div>
          <div className="min-w-0 rounded-[10px] border border-[#A87146]/20 bg-white p-5">
            <KV items={[
              ["Status", <StatusPill key="s" status={b.status} />],
              ["Measures", b.measures.map((m) => MEASURE_LABEL[m]).join(", ")],
              ["Records", b.records], ["Scoring", b.scoring],
              ...((report?.all_photos as unknown as { answer_key?: string } | undefined)?.answer_key && fromRun
                ? [["Answer key", <span key="k" className="mono">{(report!.all_photos as unknown as { answer_key: string }).answer_key}</span>] as [React.ReactNode, React.ReactNode]] : []),
              ...(b.elsewhere ? [["Elsewhere", b.elsewhere] as [React.ReactNode, React.ReactNode]] : []),
            ]} />
            <SourceLine parts={[`research.ts BENCHMARKS[${b.slug}]`, fromRun && run ? `/api/scoreboard/${run.id}` : null]} />
          </div>
        </div>
      </Section>
      <Section id="protocol" band kicker="2 · Protocol" title="The steps, the same every time it runs">
        <div className="mt-4"><Pipeline steps={b.protocol} icons={b.protocol.map((_, i) => <Check key={i} className="h-5 w-5" />)} /></div>
        <p className="text-sm text-muted-foreground leading-normal mt-2.5">Bands use the reference set the identifier answered from, not the live counts. <Link href="/research/protocols" className="text-[#4f7a16] underline">Evaluation protocols →</Link></p>
      </Section>
      <DatasetSection b={b} d={d} report={report} />
      <ResultsSection b={b} d={d} />
      <HistorySection b={b} d={d} />
      <Section id="links" band kicker="6 · Links">
        <div className="rlinks rl4" data-testid="benchmark-links">
          {b.links.map((l) => {
            const outside = l.href.startsWith("http");
            const inner = <><b>{l.title}{l.members && <Lock className="ml-1 inline h-3 w-3" aria-label="members only" />}{outside && <ArrowUpRight className="ml-0.5 inline h-3 w-3" />}</b>{l.text}</>;
            return outside ? <a key={l.href} href={l.href} target="_blank" rel="noreferrer">{inner}</a> : <Link key={l.href} href={l.href}>{inner}</Link>;
          })}
        </div>
      </Section>
    </ResearchShell>
  );
}

function DetailTiles({ b, d }: { b: Benchmark; d: BenchmarkDetailData }) {
  const h = d.howGood;
  const m = b.headline?.metric;
  if (m === "species-topk" && h?.top_k?.species) {
    const t = h.top_k;
    return (
      <Tiles testId="detail-tiles">
        <Tile highlight value={pc(t.species?.top1)} label="species top 1" note={`${nf(t.species?.n)} finds`} />
        <Tile value={pc(t.species?.top5)} label="species top 5" note="same finds" />
        <Tile value={pc(t.genus?.top1)} label="genus top 1" note={`${nf(t.genus?.n)} finds`} />
        <Tile value={pc(t.family?.top1)} label="family top 1" note={`${nf(t.family?.n)} finds`} />
      </Tiles>
    );
  }
  if (m === "species-depth" && h?.species_by_depth?.length) {
    const words: Record<string, string> = { "1-4": "species top 1 when the true species has 1-4 records", "5-19": "with 5-19 records", "20-99": "with 20-99 records", "100+": "with 100 or more" };
    return (
      <Tiles testId="detail-tiles">
        {h.species_by_depth.filter((x) => x.band !== "0").map((x, i) => (
          <Tile key={x.band} highlight={i === 0} value={pc(x.top1)} label={words[x.band] ?? `${x.band} records`} note={`${nf(x.n)} finds`} />
        ))}
      </Tiles>
    );
  }
  if (m === "unknown-count") {
    const zero = h?.species_by_depth?.find((x) => x.band === "0");
    return zero ? <Tiles testId="detail-tiles"><Tile highlight value={nf(zero.n)} label="finds whose species has no reference record" note={`of ${nf(h?.top_k?.species?.n)} finds with a species`} /></Tiles> : null;
  }
  if (m === "prospective") {
    const lead = prospectiveLead(d.prospective, h?.run?.backbone);
    return lead && b.status !== "paused" ? (
      <Tiles testId="detail-tiles">
        <Tile highlight value={nf(lead.predicted)} label="predictions saved before the name existed" note={`${lead.rows} model rows`} />
        <Tile value={nf(lead.row.resolved)} label="resolved so far" note={modelParts(lead.row.backbone).name} />
      </Tiles>
    ) : null;
  }
  if (m === "early-answers") {
    const e = earlyFigures(d.early);
    return e ? (
      <Tiles testId="detail-tiles">
        <Tile highlight value={nf(e.answers)} label="answers kept" note={e.context} />
        <Tile value={nf(e.awaiting)} label="awaiting their verdict" />
        <Tile value={nf(e.afterVerdict)} label="left out: after the verdict" />
        {e.sealed != null && <Tile value={nf(e.sealed)} label="counted, not scored: sealed test" />}
        <Tile value={nf(e.scored)} label="scored so far" />
      </Tiles>
    ) : null;
  }
  return null;
}

function RareCallout({ b, h }: { b: Benchmark; h?: HowGood }) {
  if (b.headline?.metric !== "species-depth") return null;
  const lo = h?.species_by_depth?.find((x) => x.band === "1-4"), mid = h?.species_by_depth?.find((x) => x.band === "5-19");
  if (lo?.top1 == null || mid?.top1 == null) return null;
  return (
    <div className="callout green mt-3.5" data-testid="rare-callout"><div>
      Every sequenced find of a rare species is the cheapest accuracy gain there is: one more record moves a species from the 1-4 band (<b style={{ color: "var(--ink)" }}>{pc(lo.top1)}</b>) toward 5-19 (<b style={{ color: "var(--ink)" }}>{pc(mid.top1)}</b>).
    </div></div>
  );
}

function DatasetSection({ b, d, report }: { b: Benchmark; d: BenchmarkDetailData; report?: RunReport }) {
  const std = standardOf(report);
  const fromRun = b.headline?.run != null;
  if (fromRun && std) {
    const finds = STANDARD_BANDS.map((x) => std.species?.[x]?.n ?? null);
    const bars = companionBars(report?.reference_species, finds);
    const sp = bars.species?.find((x) => x.band === "1-4");
    const fd = bars.finds.find((x) => x.band === "1-4");
    return (
      <Section id="dataset" kicker="3 · Dataset"
               title={sp && fd ? `${sp.share}% of reference species have 1-4 records, but only ${fd.share}% of finds` : "The records it is scored on"}>
        <ChartCard className="mt-3.5">
          <DepthShareBars species={bars.species} finds={bars.finds} speciesTotal={report?.reference_species?.species} />
          <SourceLine parts={[`/api/scoreboard/${d.howGood!.run!.id} · reference_species${report?.reference_species ? ` (answer key ${report.reference_species.answer_key}, ${nf(report.reference_species.records)} records)` : ""}`]} />
        </ChartCard>
      </Section>
    );
  }
  return (
    <Section id="dataset" kicker="3 · Dataset" title="The records it is scored on">
      <div className="min-w-0 rounded-[10px] border border-[#A87146]/20 bg-white p-5 mt-3.5">
        <KV items={[["Records", b.records], ["Size", cardMeta(b, d).size], ["Data", b.data.map((x) => DATA_LABEL[x]).join(", ")]]} />
        <SourceLine parts={[`research.ts BENCHMARKS[${b.slug}]`]} />
      </div>
    </Section>
  );
}

function ResultsSection({ b, d }: { b: Benchmark; d: BenchmarkDetailData }) {
  const h = d.howGood;
  const run = h?.run;
  const m = b.headline?.metric;
  const served = run ? [run.backbone] : [];
  let body: React.ReactNode = null;
  if (m === "species-topk" && run) {
    const std = standardOf(d.reports.get(run.id));
    const rows: TopKRow[] = (["species", "genus", "family"] as const).flatMap((rank) => {
      const t = std?.[rank]?.all ?? h?.top_k?.[rank];
      return t ? [{ key: rank, name: rank[0].toUpperCase() + rank.slice(1), detail: `${nf(t.n)} finds`, n: t.n, values: topkValues(t), highlight: rank === "species" }] : [];
    });
    body = (
      <ChartCard className="mt-3.5">
        <TopKChart rows={rows} labelHead="Rank" trackHead={`${modelParts(run.backbone).name}, ${run.method.endsWith("+place") ? "photos + place and date" : "photos only"}`} testId="chart-detail" />
        <SourceLine parts={[`/api/scoreboard/${run.id}`, "/api/how-good"]} />
      </ChartCard>
    );
  } else if (m === "species-depth" && run && h?.species_by_depth) {
    const place = d.runs ? placeVsPhotos(d.runs, served, run.method) : null;
    const photos = place ? standardOf(d.reports.get(place.photos.id)) : null;
    const rows: TopKRow[] = h.species_by_depth.filter((x) => x.band !== "0").map((x, i) => ({
      key: x.band, name: `${x.band} records`, detail: `${nf(x.n)} finds`, n: x.n, values: topkValues(x), highlight: i === 0 }));
    const table = (title: string, sub: string, bands: { band: string; n: number; t: Partial<Record<"top1" | "top3" | "top5" | "top10", number | null>> }[], all?: { n: number; t: Partial<Record<"top1" | "top3" | "top5" | "top10", number | null>> }) => (
      <div className="tcard"><div className="th"><b>{title}</b><span>{sub}</span></div><div className="scrollx"><table className="dt">
        <thead><tr><th className="l">True species has</th><th>Finds</th><th>Top 1</th><th>3</th><th>5</th><th>10</th></tr></thead>
        <tbody>
          {bands.map((x) => <tr key={x.band}><td className="l"><b>{x.band} records</b></td><td className="mut">{nf(x.n)}</td><td className="strong">{pt(x.t.top1)}</td><td>{pt(x.t.top3)}</td><td>{pt(x.t.top5)}</td><td>{pt(x.t.top10)}</td></tr>)}
          {all && <tr className="hl"><td className="l"><b>All finds</b></td><td className="mut">{nf(all.n)}</td><td className="strong">{pt(all.t.top1)}</td><td>{pt(all.t.top3)}</td><td>{pt(all.t.top5)}</td><td>{pt(all.t.top10)}</td></tr>}
        </tbody></table></div></div>
    );
    const groups = d.runs ? approachGroups(d.runs, served, run.method) : [];
    const heat = groups.map((g) => heatRows(g.runs, d.reports, served)).sort((a, z) => z.rows.length - a.rows.length)[0];
    const p = modelParts(run.backbone);
    body = (
      <>
        <ChartCard className="mt-3.5">
          <TopKChart rows={rows} labelHead="True species has" trackHead={`${p.name}, ${run.method.endsWith("+place") ? "photos + place and date" : "photos only"}`} testId="chart-detail" />
          <SourceLine parts={[`/api/how-good (run ${run.id})`]} />
        </ChartCard>
        <div className="two mt-3.5">
          {table(p.name, `${p.detail} · ${run.method.endsWith("+place") ? "photos + place and date" : "photos only"}`,
                 h.species_by_depth.map((x) => ({ band: x.band, n: x.n, t: x })), h.top_k?.species ? { n: h.top_k.species.n, t: h.top_k.species } : undefined)}
          {photos && table(p.name, `${p.detail} · photos only`, STANDARD_BANDS.map((x) => ({ band: x, n: photos.species[x]?.n ?? 0, t: photos.species[x] ?? {} })),
                           { n: photos.species.all.n, t: photos.species.all })}
        </div>
        {heat && heat.rows.length > 0 && (
          <ChartCard className="mt-3.5">
            <h3 className="font-display font-semibold text-[#4a3728] mb-2.5">Which method copes best with rare species · {nf(heat.finds.reduce<number>((s, n) => s + (n ?? 0), 0))} finds</h3>
            <DepthHeatmap rows={heat.rows} finds={heat.finds} bands={STANDARD_BANDS} />
            <SourceLine parts={[`/api/scoreboard/${heat.rows.map((r) => r.key).join(", ")}`]} />
          </ChartCard>
        )}
      </>
    );
  } else if (m === "prospective") {
    const rows = d.prospective ?? [];
    body = rows.length ? (
      <div className="mt-3.5"><ValueBars labelWidth={220} rows={rows.map((r) => {
        const p = modelParts(r.backbone, r.method);
        return { label: p.name, sub: p.detail, frac: r.predicted ? r.resolved / r.predicted : 0, text: `${nf(r.resolved)} of ${nf(r.predicted)} resolved`, color: "var(--s1)", tip: `${p.name}|${nf(r.resolved)} of ${nf(r.predicted)} resolved` };
      })} /><SourceLine parts={["/api/prospective"]} /></div>
    ) : null;
  } else if (m === "early-answers") {
    const e = earlyFigures(d.early);
    body = e ? <p className="mt-3.5 text-base">{nf(e.scored)} of {nf(e.answers)} answers scored so far; {nf(e.awaiting)} await their verdict. <Link href="/research/results#early" className="text-[#4f7a16] underline">The full table →</Link></p> : null;
  }
  return (
    <Section id="results" band kicker="4 · Results" title={m === "species-depth" ? "Species top 1 / 3 / 5 / 10, band by band" : "What it measures now"}>
      {body ?? (
        <div className="num empty mt-3.5 rounded-[10px] border border-[#A87146]/[.12] p-4" data-testid="detail-results-empty"
             style={{ background: "repeating-linear-gradient(135deg,#fbf9f6 0 10px,#f6f1ea 10px 20px)" }}>
          <div className="msg text-[13px] text-[#6b5a4a]">{b.pending ? <><b style={{ color: "var(--ink)" }}>{b.pending.lead}</b> {b.pending.text}</> : d.loading ? "Reading…" : "Not measured yet."}</div>
        </div>
      )}
    </Section>
  );
}

function HistorySection({ b, d }: { b: Benchmark; d: BenchmarkDetailData }) {
  if (b.headline?.run == null) {
    return (
      <Section id="history" kicker="5 · History" title="Every run">
        <p className="text-base">{b.lastRun ? `It runs ${b.lastRun === "live" ? "live, as answers are given" : b.lastRun === "nightly" ? "every night" : b.lastRun}.` : "Not run yet."} {b.results && <Link href={b.results.href} className="text-[#4f7a16] underline">{b.results.label} →</Link>}</p>
      </Section>
    );
  }
  const hist = historyLines(comparisons((d.runs ?? []).filter((r) => !incompleteRun(r)))).slice(0, 8);
  const depth = b.headline.metric === "species-depth";
  const bands = ["1-4", "5-19", "20-99", "100+"];
  const servedId = d.howGood?.run?.id;
  return (
    <Section id="history" kicker="5 · History" title="Every run of this breakdown"
             lead="Not a trend line: the models and the test records change between rows. Read each row on its own. One row per set of test records: the best MycoMap Vision row on it.">
      {!hist.length ? <p className="small">{d.loading ? "Reading…" : "No runs published yet."}</p> : (
        <div className="tcard" data-testid="benchmark-history"><div className="scrollx"><table className="dt">
          <thead><tr><th className="l">Measured</th><th className="l">Model · method</th><th className="l">Test records</th>
            {depth ? bands.map((x) => <th key={x}>{x}</th>) : <><th>Species top 1</th><th>Genus top 1</th></>}</tr></thead>
          <tbody>{hist.map((h) => {
            const p = modelParts(h.best.backbone, h.best.method);
            const std = standardOf(d.reports.get(h.best.id));
            return (
              <tr key={h.key} className={h.best.id === servedId ? "hl" : ""}>
                <td className="l"><span className="whitespace-nowrap">{shortDay(h.measured)}</span><small className="mono">run #{h.best.id}</small></td>
                <td className="l"><b>{p.name}</b><small>{p.detail}</small></td>
                <td className="l">{h.best.dataset_release ? `${h.best.dataset_release} val` : "live data"}<small>{nf(h.finds)} finds</small></td>
                {depth ? bands.map((x) => {
                  const v = std?.species?.[x]?.top1;
                  return <td key={x}>{v == null ? "–" : <div className="flex items-center justify-end gap-1.5">
                    <span className="inline-block h-1.5 w-[46px] overflow-hidden rounded-[3px] bg-[#ebe4da]"><b className="block h-full" style={{ width: `${100 * v}%`, background: "#3987e5" }} /></span>{pt(v)}</div>}</td>;
                }) : <><td className="strong">{pt(h.best.species_top1)}</td><td>{pt(h.best.genus_top1)}</td></>}
              </tr>
            );
          })}</tbody></table></div></div>
      )}
      <SourceLine parts={[`/api/scoreboard${depth ? " and each run's standard.species" : ""}`, depth ? "species top 1; – where a run predates the depth bands" : "top 1"]} />
    </Section>
  );
}

