import { useState } from "react";
import type { EarlyAnswersGroup, HowGood, Me, ModelsInfo, ProspectiveModel, RunReport, ScoreRun,
              SpeedDay } from "@/lib/api";
import { companionBars, scoreboardSections, standardOf, STANDARD_BANDS } from "@/lib/compareView";
import { utcDay } from "@/lib/datasetReleases";
import { MODEL_CREDIT_LINE, modelName } from "@/lib/publicView";
import { OPEN_REFERENCE, pointsDiff, REFERENCE_LABEL, TWO_MODELS, type TopK } from "@/lib/twoModels";
import { APPROACHES } from "@/lib/approaches";
import { approachGroups, comparisonRows, earlyFigures, heatRows, historyLines, modelParts, nf, pc, pt,
         releaseGroups, scoreLines, shortDay, topkValues, type TopKRow } from "@/lib/researchView";
import { ChartCard, DepthHeatmap, DepthShareBars, InPageNav, PairedDiff, ResearchHero, ResearchShell,
         Section, SourceLine, StatusPill, Tile, Tiles, TopKChart, ValueBars, KLegend } from "@/components/ResearchKit";
import { PermissionBox } from "@/components/Explainer";
import { AnsweredBeforeVerdictCard } from "@/components/AnsweredBeforeVerdict";
import { useEarly, useHowGood, useMe, useModels, useProspective, useRunReports, useScoreboard,
         useSpeed } from "@/components/researchQueries";

// Research > Results (Steve, 2026-10-25: the approved redesign). One page, eight sections
// behind a sticky in-page nav; every comparison scores its models on exactly the same finds.
// Numbers: /api/how-good, /api/scoreboard and its run reports (standard summary),
// /api/prospective, /api/early-answers, /api/models, /api/speed/daily; the two models from
// lib/twoModels.ts (a committed research-repo file: no v1-open scoreboard run exists).

export const RESULTS_SECTIONS = [
  { id: "headline", label: "Headline" }, { id: "two", label: "The two models" },
  { id: "approaches", label: "Approaches compared" }, { id: "depth", label: "Reference depth" },
  { id: "scoreboard", label: "Scoreboard" }, { id: "advance", label: "Advance predictions" },
  { id: "early", label: "Before the DNA verdict" }, { id: "models", label: "Models in use" },
];

export interface ResultsData {
  me?: Me | null;
  howGood?: HowGood;
  runs?: ScoreRun[];
  reports: Map<number, RunReport>;
  prospective?: { models: ProspectiveModel[]; sealed_after?: string | null };
  early?: { groups: EarlyAnswersGroup[]; sealed_after: string };
  models?: ModelsInfo;
  speed?: SpeedDay[];
  loading?: boolean;
}

export function ResultsPage() {
  const me = useMe();
  const howGood = useHowGood();
  const board = useScoreboard();
  const prospective = useProspective();
  const early = useEarly();
  const models = useModels();
  const speed = useSpeed();
  const runs = board.data?.runs ?? [];
  const served = howGood.data?.run ? [howGood.data.run.backbone] : [];
  const method = howGood.data?.run?.method ?? "";
  const ids = [howGood.data?.run?.id, ...releaseGroups(runs, served, method).flat().map((r) => r.id)]
    .filter((x): x is number => x != null);
  const { reports } = useRunReports(ids);
  return (
    <Results me={me.data} howGood={howGood.data} runs={board.data?.runs} reports={reports}
             prospective={prospective.data} early={early.data} models={models.data} speed={speed.data?.days}
             loading={board.isLoading || howGood.isLoading} />
  );
}

export function Results(d: ResultsData) {
  return (
    <ResearchShell me={d.me}>
      <Hero d={d} />
      <InPageNav sections={RESULTS_SECTIONS} />
      <Headline d={d} />
      <TwoModelsSection />
      <Approaches d={d} />
      <Depth d={d} />
      <Scoreboard d={d} />
      <Advance d={d} />
      <Early d={d} />
      <ModelsInUse d={d} />
    </ResearchShell>
  );
}

const servedOf = (d: ResultsData) => ({
  served: d.howGood?.run ? [d.howGood.run.backbone] : [], method: d.howGood?.run?.method ?? "",
});
const runOf = (d: ResultsData, id?: number) => (id == null ? undefined : d.runs?.find((r) => r.id === id));
const withPlace = (method?: string) => (method?.endsWith("+place") ? "photos + place and date" : "photos only");

function Hero({ d }: { d: ResultsData }) {
  const h = d.howGood;
  const s = h?.top_k?.species;
  const run = h?.run;
  const scored = runOf(d, run?.id);
  return (
    <ResearchHero crumbs={[{ href: "/research", label: "Research" }, { label: "Results" }]} title="Results"
                  lede="Every comparison scores its models on exactly the same records, identified only from the records validated before them. Top 1 means the first answer was right; top 10, that it was among the first ten.">
      {s && run ? (
        <>
          <Tiles testId="results-tiles">
            <Tile highlight value={pc(s.top1)} label="species top 1" note={`${modelParts(run.backbone).name} · ${withPlace(run.method)}`} />
            <Tile value={pc(s.top3)} label="species top 3" />
            <Tile value={pc(s.top5)} label="species top 5" />
            <Tile value={pc(s.top10)} label="species top 10" />
          </Tiles>
          <div className="meta">
            <span>Genus top 1 <b>{pc(h?.top_k?.genus?.top1)}</b> · top 5 <b>{pc(h?.top_k?.genus?.top5)}</b></span>
            <span>Family top 1 <b>{pc(h?.top_k?.family?.top1)}</b></span>
            <span>{scored?.dataset_release ? `Dataset release ${scored.dataset_release}, validation split` : "The newest validations"} · <b>{nf(s.n)}</b> finds with a species answer</span>
          </div>
          <SourceLine parts={[`/api/how-good → run ${run.id}`, scored ? `measured ${shortDay(scored.created_at)}` : null]} />
        </>
      ) : (
        <p className="small mt-4" data-testid="results-no-headline">
          {d.loading ? "Reading the latest results…" : "The headline results could not be read just now."}
        </p>
      )}
    </ResearchHero>
  );
}

function Headline({ d }: { d: ResultsData }) {
  const h = d.howGood;
  const run = h?.run;
  const report = run ? d.reports.get(run.id) : undefined;
  const std = standardOf(report);
  const first = (report?.first_photo_only as unknown as { standard?: ReturnType<typeof standardOf> } | undefined)?.standard;
  const rows: TopKRow[] = [];
  if (std) {
    rows.push({ key: "species", name: "Species", detail: `${nf(std.species.all.n)} finds`, n: std.species.all.n, values: topkValues(std.species.all), highlight: true });
    if (std.genus?.all) rows.push({ key: "genus", name: "Genus", detail: `${nf(std.genus.all.n)} finds`, n: std.genus.all.n, values: topkValues(std.genus.all) });
    if (std.family?.all) rows.push({ key: "family", name: "Family", detail: `${nf(std.family.all.n)} finds`, n: std.family.all.n, values: topkValues(std.family.all) });
    if (first?.species?.all) rows.push({ key: "first", name: "Species, first photo only", detail: "the first photo of each find", n: first.species.all.n, values: topkValues(first.species.all) });
  } else if (h?.top_k) {
    for (const rank of ["species", "genus", "family"] as const) {
      const t = h.top_k[rank];
      if (t) rows.push({ key: rank, name: rank[0].toUpperCase() + rank.slice(1), detail: `${nf(t.n)} finds`, n: t.n, values: topkValues(t), highlight: rank === "species" });
    }
  }
  const ece = (report as unknown as { ece?: { species?: { ece: number; n: number } } } | undefined)?.ece?.species;
  const fp = h?.first_photo;
  const errata = (report as unknown as { errata?: { file?: string } } | undefined)?.errata?.file?.match(/-(r\d+)\.tsv$/)?.[1];
  return (
    <Section id="headline" kicker="1 · Headline" title="The site's model, at every rank"
             lead={run ? `${modelParts(run.backbone).name} on ${runOf(d, run.id)?.dataset_release ? `dataset release ${runOf(d, run.id)!.dataset_release}'s validation split` : "the newest validations"}, with each find's photos${run.method.endsWith("+place") ? " and its public place and date" : ""}. The same summary is reported for every test on this site.` : undefined}>
      <ChartCard>
        <TopKChart rows={rows} labelHead="Rank" trackHead="Share right within the first k answers" testId="chart-headline" />
        <SourceLine parts={[run ? `/api/scoreboard/${run.id} · all_photos.standard${first ? ", first_photo_only.standard" : ""}` : null,
                            report?.all_photos && (report.all_photos as unknown as { answer_key?: string }).answer_key ? `answer key ${(report.all_photos as unknown as { answer_key: string }).answer_key}` : null,
                            errata ? `erratum ${errata}` : null]} />
      </ChartCard>
      <div className="three" style={{ marginTop: 14 }}>
        {fp && (
          <div className="callout green" data-testid="callout-first-photo"><div><b style={{ color: "var(--ink)" }}>Using every photo of a find helps.</b>{" "}
            The first photo alone gets the species first {pc(fp.first_photo_top1)} of the time; all the photos, {pc(fp.all_photos_top1)}.</div></div>
        )}
        {ece && (
          <div className="callout" data-testid="callout-calibration"><div><b style={{ color: "var(--ink)" }}>Stated confidence is honest to within {pt(ece.ece)} points</b>{" "}
            on average (expected calibration error, species, {nf(ece.n)} finds).</div></div>
        )}
        <div className="callout warm" data-testid="callout-inat"><div>
          {h?.inat?.species_top1 != null
            ? <><b style={{ color: "var(--ink)" }}>iNaturalist's computer vision on these records:</b> species top 1 {pc(h.inat.species_top1)}{h.inat.n_test ? ` of ${nf(h.inat.n_test)}` : ""}.</>
            : <><b style={{ color: "var(--ink)" }}>iNaturalist's computer vision on these records:</b> not scored yet. It will be scored on the sealed test, on the same photos and the same public place, after the analysis freeze.</>}
        </div></div>
      </div>
    </Section>
  );
}

function ModelTable({ title, swatch, sub, rows }: {
  title: string; swatch: string; sub: string; rows: { label: string; note?: string; t: TopK; hl?: boolean }[];
}) {
  return (
    <div className="tcard" data-testid="two-models-table">
      <div className="th"><b><span className="mr-1.5 inline-block h-[9px] w-[9px] rounded-sm" style={{ background: swatch }} />{title}</b><span>{sub}</span></div>
      <div className="scrollx">
        <table className="dt">
          <thead><tr><th className="l">Rank</th><th>Top 1</th><th>3</th><th>5</th><th>10</th><th>Finds</th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.label + (r.note ?? "")} className={r.hl ? "hl" : ""}>
                <td className="l"><b>{r.label}</b>{r.note && <small>{r.note}</small>}</td>
                <td className="strong">{pt(r.t.top1)}</td><td>{pt(r.t.top3)}</td><td>{pt(r.t.top5)}</td><td>{pt(r.t.top10)}</td>
                <td className="mut">{nf(r.t.n)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/** The site's model and the paper's headline model (lib/twoModels.ts). v1-open has two
 *  species numbers, each labelled with its reference (Steve, 2026-10-25). */
export function TwoModelsSection() {
  const [site, open] = TWO_MODELS.models;
  const p = TWO_MODELS.paired;
  const openRef = OPEN_REFERENCE.species[open.backbone];
  return (
    <Section id="two" band kicker="2 · The two models" title="What the site serves, and what the paper headlines"
             lead={<>Both are the same base model, fine-tuned the same way at 336 px on release v1's training split. They differ only in the photos they learned from. <span data-testid="text-model-credit-line">{MODEL_CREDIT_LINE}</span></>}>
      <div className="two">
        <ModelTable title="MycoMap Vision v1" swatch="var(--s1)" sub="public + non-public training data · the site's model"
                    rows={[{ label: "Species", note: REFERENCE_LABEL.full, t: site.species, hl: true },
                           { label: "Genus", note: REFERENCE_LABEL.full, t: site.genus }]} />
        <ModelTable title="MycoMap Vision v1-open" swatch="#8B4A1F" sub="open data only · the paper's headline model"
                    rows={[{ label: "Species", note: REFERENCE_LABEL.open, t: openRef, hl: true },
                           { label: "Species", note: REFERENCE_LABEL.full, t: open.species },
                           { label: "Genus", note: REFERENCE_LABEL.full, t: open.genus }]} />
      </div>
      <ChartCard className="mt-3.5">
        <div className="flex flex-wrap justify-between gap-2"><h3 className="font-display font-semibold text-[#4a3728]">Paired difference, species top 1 (v1-open minus v1)</h3><span className="small">points · 95% interval clustered by observer · full reference</span></div>
        <div className="max-w-[560px]"><PairedDiff diff={p.diff} ci={p.ci95} tip={`v1-open minus v1, species top 1|${pointsDiff(p.diff, p.ci95)} points`} /></div>
        <p className="small mt-1.5" data-testid="text-paired">
          {pointsDiff(p.diff, p.ci95)} points on the same {nf(p.n)} finds from {nf(p.observers)} observers, both against the full reference: learning only from openly licensed photos costs less than half a point.
          {" "}Against the openly licensed reference only: {pointsDiff(OPEN_REFERENCE.paired.diff, OPEN_REFERENCE.paired.ci95)} points.
        </p>
        <PermissionBox kind="control" className="mt-3" />
        <SourceLine parts={[`web/src/lib/twoModels.ts (research repo ${TWO_MODELS.source.replace(" (research repo, ", ", ").replace(")", "")}; ${OPEN_REFERENCE.source.replace(" (research repo, ", ", ").replace(")", "")})`,
                            "photos only, scored by specimen", "observer-clustered bootstrap, 2,000 draws"]} />
      </ChartCard>
    </Section>
  );
}

const groupTitle = (g: ScoreRun[], reports: Map<number, RunReport>) => {
  const rep = g.map((r) => reports.get(r.id)).find((x) => x);
  const errata = (rep as unknown as { errata?: { file?: string } } | undefined)?.errata?.file?.match(/-(r\d+)\.tsv$/)?.[1];
  const release = g.find((r) => r.dataset_release)?.dataset_release;
  return `${release ? `release ${release} validation split` : "newest validations"}${errata ? `, erratum ${errata}` : ", every record"}`;
};

function Approaches({ d }: { d: ResultsData }) {
  const { served, method } = servedOf(d);
  const groups = d.runs ? approachGroups(d.runs, served, method) : [];
  return (
    <Section id="approaches" kicker="3 · Approaches compared" title="Every approach, on exactly the same finds"
             lead="One chart per set of test records: a model is only ever compared with others measured on the same finds. Species strict, all photos of a find.">
      {!groups.length && <p className="small" data-testid="approaches-empty">{d.loading ? "Reading the scoreboard…" : "No comparison on a dataset release is published yet."}</p>}
      {groups.map(({ runs: g, baseOnly }, i) => {
        const { rows, missing } = comparisonRows(g, d.reports, served, method);
        const n = rows.find((r) => r.n)?.n ?? g[0].n_test;
        return (
          <ChartCard key={g[0].id} className={i ? "mt-3.5" : ""} testId="approach-chart">
            <div className="mb-2.5 flex flex-wrap justify-between gap-2">
              <h3 className="font-display font-semibold text-[#4a3728]">{String.fromCharCode(65 + i)} · {nf(n)} finds ({groupTitle(g, d.reports)}){baseOnly && ": fine-tuned against the base model"}</h3>
              {g.some((r) => r.method.endsWith("+place")) ? <span className="tag">place-and-date methods use the find's public place</span>
                : baseOnly ? <span className="tag">photos only</span> : null}
            </div>
            <TopKChart rows={rows} legend={false} labelHead="Approach" trackHead="Species, by specimen" testId={`chart-approaches-${i}`} />
            <KLegend extra={<span><i style={{ background: "var(--s1)", borderRadius: 2 }} />the site's model and method</span>} />
            {missing > 0 && <p className="small">{missing} approach{missing === 1 ? " was" : "es were"} scored before top 3 and top 10 were recorded and {missing === 1 ? "is" : "are"} in the scoreboard only.</p>}
            <SourceLine parts={[`/api/scoreboard/${g.map((r) => r.id).join(", ")}`, `record set ${g[0].record_set}`]} />
          </ChartCard>
        );
      })}
      <details className="tbl" style={{ marginTop: 16 }}>
        <summary>How each approach names a species, and its status ({APPROACHES.length} approaches)</summary>
        <div className="scrollx" style={{ marginTop: 8 }}>
          <table className="dt"><thead><tr><th className="l">Approach</th><th className="l">How it names a species</th><th className="l">Status</th></tr></thead>
            <tbody>{APPROACHES.map((a) => (
              <tr key={a.name}><td className="l"><b>{a.name}</b>{a.outside && <> <span className="tag d">outside</span></>}</td><td className="l">{a.how}</td><td className="l">{a.status}</td></tr>
            ))}</tbody></table>
        </div>
      </details>
    </Section>
  );
}

function Depth({ d }: { d: ResultsData }) {
  const { served, method } = servedOf(d);
  const run = d.howGood?.run;
  const report = run ? d.reports.get(run.id) : undefined;
  const std = standardOf(report);
  const finds = STANDARD_BANDS.map((b) => std?.species?.[b]?.n ?? null);
  const bars = companionBars(report?.reference_species, finds);
  const groups = d.runs ? approachGroups(d.runs, served, method) : [];
  const heats = groups.map((g) => ({ g: g.runs, ...heatRows(g.runs, d.reports, served) })).sort((a, b) => b.rows.length - a.rows.length);
  const best = heats[0];
  return (
    <Section id="depth" band kicker="4 · Reference depth" title="Most species are rare; most finds are of common species"
             lead="How many DNA-verified records the true species has explains accuracy better than any method. The bars show the two populations; the heatmap shows species top 1 in each band.">
      {std ? (
        <ChartCard>
          <DepthShareBars species={bars.species} finds={bars.finds} speciesTotal={report?.reference_species?.species} />
          <SourceLine parts={[`/api/scoreboard/${run!.id} · reference_species, standard.species bands`]} />
        </ChartCard>
      ) : <p className="small" data-testid="depth-empty">{d.loading ? "Reading…" : "The depth breakdown is not available just now."}</p>}
      {best && best.rows.length > 0 && (
        <ChartCard className="mt-3.5">
          <h3 className="font-display font-semibold text-[#4a3728] mb-2.5">Species top 1 by the true species' reference records · {nf(best.finds.reduce<number>((s, n) => s + (n ?? 0), 0))} finds</h3>
          <DepthHeatmap rows={best.rows} finds={best.finds} bands={STANDARD_BANDS} />
          <p className="small mt-2">A species with no reference can't be named by any approach that learns from references.</p>
          <SourceLine parts={[`/api/scoreboard/${best.rows.map((r) => r.key).join(", ")} · standard.species`, "photos only unless the row says place and date"]} />
        </ChartCard>
      )}
    </Section>
  );
}

function Scoreboard({ d }: { d: ResultsData }) {
  const { served, method } = servedOf(d);
  const runs = d.runs ?? [];
  const groups = releaseGroups(runs, served, method);
  const { history } = scoreboardSections(runs);
  const [pick, setPick] = useState(0);
  const g = groups[Math.min(pick, Math.max(0, groups.length - 1))];
  const lines = g ? scoreLines(g, d.reports, served, method) : [];
  const hist = historyLines(history);
  return (
    <Section id="scoreboard" kicker="5 · Scoreboard" title="Every published run"
             lead="One table per set of test records. Rows best first by species top 1; the site's model and method highlighted.">
      {!runs.length && <p className="small" data-testid="scoreboard-empty">{d.loading ? "Reading the scoreboard…" : "No tests published yet."}</p>}
      {groups.length > 0 && (
        <div className="mb-3 flex flex-wrap items-center gap-2.5">
          <div className="seg" role="tablist" aria-label="Test records">
            {groups.map((x, i) => (
              <button key={x[0].id} type="button" role="tab" aria-selected={i === pick} className={i === pick ? "on" : ""} onClick={() => setPick(i)}
                      data-testid={`scoreboard-tab-${i}`}>
                v1 val · {nf(x[0].n_test)} records
              </button>
            ))}
          </div>
          <span className="small">{nf(groups.flat().length)} runs on a release · {nf(history.flat().length)} in development history</span>
        </div>
      )}
      {g && (
        <div className="tcard" data-testid="scoreboard-table">
          <div className="th"><b>Dataset release {g[0].dataset_release}, validation split · {nf(g[0].n_test)} records validated after {utcDay(g[0].cutoff)}</b>
            <span>identified from {nf(g[0].n_reference)} earlier DNA-verified records</span></div>
          <div className="scrollx">
            <table className="dt">
              <thead><tr><th className="l">Model · method</th><th>Species top 1</th><th>3</th><th>5</th><th>10</th><th>Genus top 1</th><th>Family top 1</th><th>1st photo only</th><th>Finds</th><th>Run</th></tr></thead>
              <tbody>
                {lines.map((l) => (
                  <tr key={l.run.id} className={l.highlight ? "hl" : ""}>
                    <td className="l"><b>{l.name}</b><small>{l.detail}</small></td>
                    <td className="strong">{pt(l.species[1])}</td><td>{pt(l.species[3])}</td><td>{pt(l.species[5])}</td><td>{pt(l.species[10])}</td>
                    <td>{pt(l.genus)}</td><td>{pt(l.family)}</td><td className="mut">{pt(l.firstPhoto)}</td>
                    <td className="mut">{nf(l.finds)}</td><td className="mut mono">#{l.run.id}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {g && <SourceLine parts={["/api/scoreboard and /api/scoreboard/{id} · standard summary", "species strict; genus and family top 1"]} />}
      {hist.length > 0 && (
        <details className="tbl" style={{ marginTop: 16 }} data-testid="scoreboard-history">
          <summary>Development history: {hist.length} earlier tests on the live data, before release v1 was frozen</summary>
          <div className="scrollx" style={{ marginTop: 8 }}>
            <table className="dt">
              <thead><tr><th className="l">Measured</th><th>Test finds</th><th>References</th><th>Approaches</th><th className="l">Best of ours</th><th>Species top 1</th><th>iNat CV (with location)</th></tr></thead>
              <tbody>{hist.map((h) => {
                const p = modelParts(h.best.backbone, h.best.method);
                return (
                  <tr key={h.key}><td className="l">{shortDay(h.measured)}</td><td>{nf(h.finds)}</td><td>{nf(h.references)}</td><td>{h.approaches}</td>
                    <td className="l"><b>{p.name}</b><small>{p.detail}</small></td><td className="strong">{pt(h.best.species_top1)}</td><td>{pt(h.inat)}</td></tr>
                );
              })}</tbody>
            </table>
          </div>
          <p className="small mt-2">Top 1 only: these runs predate the standard top-1/3/5/10 summary. iNaturalist's computer vision on the 1,152 records of 8 October 2026 is left out: that was an incomplete development run, not a fair comparison.</p>
        </details>
      )}
    </Section>
  );
}

function Advance({ d }: { d: ResultsData }) {
  const rows = d.prospective?.models ?? [];
  const sealed = rows.reduce((n, r) => n + (r.sealed ?? 0), 0);
  const maxPredicted = Math.max(1, ...rows.map((r) => r.predicted));
  const resolvedMax = Math.max(...rows.map((r) => r.resolved), 0);
  return (
    <Section id="advance" band kicker="6 · Advance predictions" title="Predicted before the DNA name existed"
             lead="Records with a sequence but no validation yet are identified now; when they turn green, the saved answer is checked against the DNA name.">
      {!rows.length ? <p className="small" data-testid="advance-empty">{d.prospective ? "None yet." : "Reading…"}</p> : (
        <div className="tcard" data-testid="advance-table"><div className="scrollx"><table className="dt">
          <thead><tr><th className="l">Model · method</th><th>Predicted</th><th>Validated since</th><th>Species top 1</th><th>Genus</th><th>Family</th><th>Stated confidence</th></tr></thead>
          <tbody>{rows.map((r) => {
            const p = modelParts(r.backbone, r.method);
            return (
              <tr key={`${r.backbone}/${r.method}`}>
                <td className="l"><b>{p.name}</b><small>{p.detail}</small></td><td>{nf(r.predicted)}</td>
                <td><div className="flex items-center justify-end gap-2">
                  <span className="inline-block h-1.5 w-[70px] overflow-hidden rounded-[3px] bg-[#ebe4da]"><b className="block h-full" style={{ width: `${(100 * r.resolved) / maxPredicted}%`, background: "var(--s1)" }} /></span>{nf(r.resolved)}</div></td>
                <td className="strong">{r.species_top1 == null ? "–" : pc(r.species_top1)}</td><td>{r.genus_top1 == null ? "–" : pc(r.genus_top1)}</td>
                <td>{r.family_top1 == null ? "–" : pc(r.family_top1)}</td><td className="mut">{r.mean_species_confidence == null ? "–" : pc(r.mean_species_confidence)}</td>
              </tr>
            );
          })}</tbody></table></div></div>
      )}
      {rows.length > 0 && (
        <p className="small mt-2.5" data-testid="advance-note">
          Records validated since {shortDay(d.prospective?.sealed_after)} form the paper's sealed test: their predictions are counted{sealed ? ` (${nf(sealed)})` : ""}, not scored here, until that test is opened.
          {resolvedMax > 0 && resolvedMax < 30 && ` With ${nf(resolvedMax)} resolved, the one scored row is too small to read.`}
        </p>
      )}
      <SourceLine parts={["/api/prospective"]} />
    </Section>
  );
}

function Early({ d }: { d: ResultsData }) {
  const e = earlyFigures(d.early?.groups);
  const total = e ? e.askedBy.reduce((s, x) => s + x.n, 0) || 1 : 1;
  const anyScored = (d.early?.groups ?? []).some((g) => g.scored > 0);
  return (
    <Section id="early" kicker="7 · Answered before the DNA verdict" title="Answers kept as given, scored when the record turns green"
             lead="When mycomap.org auto-validates a project, or someone identifies an observation from its link, Vision answers before the record has its DNA verdict.">
      {!e ? <p className="small" data-testid="early-empty">{d.early ? "None yet." : "Reading…"}</p> : (
        <>
          <div className="four" data-testid="early-tiles">
            {([[e.answers, "answers kept"], [e.awaiting, "awaiting their verdict"], [e.afterVerdict, "left out: after the verdict"], [e.scored, "scored so far"]] as const).map(([v, l]) => (
              <div key={l} className="score"><div className="v">{nf(v)}</div><div className="l">{l}</div></div>
            ))}
          </div>
          <ChartCard className="mt-3.5">
            <div className="split2" style={{ alignItems: "center" }}>
              <div><h3 className="font-display font-semibold text-[#4a3728]">Who asked</h3>
                <div className="mt-2.5"><ValueBars labelWidth={120} rows={e.askedBy.map((a) => ({ label: a.who, frac: a.n / total, text: nf(a.n), color: "var(--s1)", tip: `${a.who}|${nf(a.n)} answers` }))} /></div></div>
              <div className="small">{anyScored ? "The scored answers, in full, are below." : "When the first records are scored, this section shows the standard summary: top 1/3/5/10 for species strict, s.l. and complex (beta), and genus, with species by reference depth."} Left out by rule: answers on or after the verdict, records Vision holds as references (or a twin of one), and records validated with Vision's help.</div>
            </div>
            <SourceLine parts={["/api/early-answers", `${e.model} · ${e.context}`]} />
          </ChartCard>
          {anyScored && <div className="mt-4"><AnsweredBeforeVerdictCard groups={d.early!.groups} sealedAfter={d.early!.sealed_after} /></div>}
        </>
      )}
    </Section>
  );
}

function SpeedSpark({ days }: { days: SpeedDay[] }) {
  const pts = days.slice(-14);
  if (pts.length < 2) return null;
  const max = Math.max(...pts.map((p) => p.seconds_per_photo)), min = Math.min(...pts.map((p) => p.seconds_per_photo));
  const x = (i: number) => 6 + (208 * i) / (pts.length - 1);
  const y = (v: number) => 46 - (36 * (v - min)) / (max - min || 1);
  return (
    <svg width="220" height="54" viewBox="0 0 220 54" className="max-w-full" role="img" aria-label="Seconds per photo, day by day" data-testid="speed-spark">
      <polyline fill="none" stroke="#b9a48f" strokeWidth="2" points={pts.map((p, i) => `${x(i)},${y(p.seconds_per_photo)}`).join(" ")} />
      {pts.map((p, i) => (
        <circle key={p.day} cx={x(i)} cy={y(p.seconds_per_photo)} r={i === pts.length - 1 ? 4 : 2.5} fill={i === pts.length - 1 ? "#7FAE34" : "#cfc3b6"}
                stroke="#fff" strokeWidth="1.5" data-tip={`${shortDay(p.day)}|${p.seconds_per_photo.toFixed(2)} s per photo`}>
          <title>{`${shortDay(p.day)}: ${p.seconds_per_photo.toFixed(2)} s per photo`}</title>
        </circle>
      ))}
    </svg>
  );
}

function ModelsInUse({ d }: { d: ResultsData }) {
  const run = d.howGood?.run;
  const served = run ? d.models?.backbones.find((b) => b.backbone === run.backbone) : undefined;
  const bases = (d.models?.backbones ?? []).filter((b) => !/-ft-/.test(b.backbone) && !b.backbone.startsWith("external:"));
  const last = d.speed?.[d.speed.length - 1];
  return (
    <Section id="models" band kicker="8 · Models in use" title="What answers on the site today">
      <div className="split2">
        {run ? (
          <div className="projcard" data-testid="served-model">
            <StatusPill status="running" label="Serving" />
            <h3 className="font-display font-semibold text-[#4a3728] mt-2.5 text-[22px]">{modelParts(run.backbone).name}</h3>
            <p className="small mb-3">{[modelParts(run.backbone).detail, MODEL_CREDIT_LINE].filter(Boolean).join(" · ")}</p>
            <dl className="kv">
              {served && <><dt>Reference photos</dt><dd>{nf(served.embedded_photos)}</dd></>}
              {served?.trained_through && <><dt>Learned from records validated up to</dt><dd>{utcDay(served.trained_through)}</dd></>}
              <dt>Method</dt><dd>{modelParts(run.backbone, run.method).detail.split(" · ").slice(-1)[0]}</dd>
              {last && <><dt>Speed</dt><dd>{last.seconds_per_photo.toFixed(2)} s per photo on {shortDay(last.day)} (daily average)</dd></>}
            </dl>
            {d.speed && <div className="mt-2.5"><SpeedSpark days={d.speed} /></div>}
            <SourceLine parts={["/api/models", d.speed ? "/api/speed/daily" : null, "/api/how-good"]} />
          </div>
        ) : <p className="small">The served model could not be read just now.</p>}
        <div className="min-w-0 rounded-[10px] border border-[#A87146]/20 bg-white p-5" data-testid="base-models">
          <h3 className="font-display font-semibold text-[#4a3728]">Base models on the server for comparisons</h3>
          <p className="small mb-2.5">Used in tests only, never to answer visitors. Each by its makers' name and image size.</p>
          <ul className="list-disc pl-[18px] text-[13.5px] leading-[1.7] sm:columns-2 sm:gap-6">
            {bases.map((b) => <li key={b.backbone}>{modelName(b.backbone).replace(" (base model, not fine-tuned),", ",")}</li>)}
          </ul>
          <SourceLine parts={["/api/models · backbones"]} />
        </div>
      </div>
    </Section>
  );
}
