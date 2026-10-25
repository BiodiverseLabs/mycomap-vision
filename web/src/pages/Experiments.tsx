import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "wouter";
import { Lock } from "lucide-react";
import { Markdown } from "@/components/Markdown";
import { api, ApiError, signInUrl, type Me } from "@/lib/api";
import { awaitsDecision, datasetOf, EXPERIMENT_STATUSES, PRE_V1, reproducibilityWords, STATUS_WORDS,
         type ExperimentStatus, type ExperimentSummary } from "@/lib/experiments";
import { entriesByDay, entryCounts, excerpt, filterEntries, NO_EXPERIMENT_FILTER, nf, shortDay, TEST_SET_LABEL,
         TEST_SETS, toggle, type ExperimentFilter } from "@/lib/researchView";
import { ChartCard, ChipRow, ResearchHero, ResearchShell, SearchBox, Section, SourceLine, StatusPill, Tile,
         Tiles } from "@/components/ResearchKit";
import { PRIMARY_BUTTON } from "@/components/Explainer";

// Research > Experiments (Steve, 2026-10-09): every experiment recorded so it can be reviewed.
// Members only: development numbers and unpublished results (experiments/README.md in the private research repo).

function Status({ s }: { s: ExperimentStatus }) {
  const w = STATUS_WORDS[s];
  return <span className={`rounded px-2 py-0.5 text-xs font-medium ${w.tone}`}>{w.label}</span>;
}

function Repro({ r, on }: { r: string; on?: string }) {
  const w = reproducibilityWords(r, on);
  return (
    <span title={w.note}
          className={`rounded px-2 py-0.5 text-xs ${w.provisional ? "border border-dashed border-[#A87146]/50 text-[#7a5a3a]"
                                                              : "bg-myco-green/10 text-myco-green"}`}>
      {w.label}
    </span>
  );
}

function SignInNote({ error }: { error: unknown }) {
  if (!error) return null;
  if (error instanceof ApiError && error.status === 401) {
    return (
      <p>
        Experiments are for mycomap.org members.{" "}
        <a href={signInUrl()} className="text-myco-green underline">Sign in</a> to read them.
      </p>
    );
  }
  return <p>The experiments could not be loaded.</p>;
}

const STATUS_DOT: Record<ExperimentStatus, React.CSSProperties> = {
  running: { background: "#3987e5" }, adopted: { background: "var(--s1)" }, done: { background: "#a87146" },
  dropped: { background: "#cfc3b6" }, planned: { background: "#fff", boxShadow: "inset 0 0 0 1.5px #a8968a" },
};
const STRIP_ORDER: ExperimentStatus[] = ["running", "adopted", "done", "dropped", "planned"];
const STATUS_NAMES = Object.fromEntries(EXPERIMENT_STATUSES.map((st) => [st, STATUS_WORDS[st].label])) as Record<ExperimentStatus, string>;
const DATASET_TABS: ["all" | "v1" | "pre", string][] = [["all", "All"], ["v1", "Release v1"], ["pre", "Pre-v1"]];

export interface ExperimentsData { me?: Me | null; list?: ExperimentSummary[]; skipped?: number; error?: unknown; loading?: boolean }

export function ExperimentsPage() {
  const me = useQuery({ queryKey: ["me"], queryFn: api.me, retry: false });
  const q = useQuery({ queryKey: ["experiments"], queryFn: api.experiments, retry: false });
  return <Experiments me={me.data} list={q.data?.experiments} skipped={q.data?.skipped} error={q.error} loading={q.isPending} />;
}

/** Research > Experiments (Steve, 2026-10-25: the approved redesign): the registry, members
 *  only. Signed out, the header and a sign-in button, never the entries (the server refuses
 *  them, and no count is published). Real titles and short excerpts; no names. */
export function Experiments(d: ExperimentsData & { initial?: ExperimentFilter }) {
  const [f, setF] = useState<ExperimentFilter>(d.initial ?? NO_EXPERIMENT_FILTER);
  const all = d.list ?? [];
  const c = entryCounts(all);
  const shown = filterEntries(all, f);
  const filtered = !!f.q.trim() || f.dataset !== "all" || f.status.length > 0 || f.sets.length > 0;
  const locked = d.error instanceof ApiError && d.error.status === 401;
  const tabCount = { all: c.all, v1: c.v1, pre: c.pre };
  return (
    <ResearchShell me={d.me}>
      <ResearchHero crumbs={[{ href: "/research", label: "Research" }, { label: "Experiments" }]} title="Experiments"
                    top={<StatusPill status="members" label="For mycomap.org members" />}
                    lede="Every experiment on MycoMap Vision: the question, everything tried, the numbers, and what was decided. Results come from dataset release v1, which anyone can rebuild; earlier entries on the pre-v1 data stay as the record of how we got here.">
        {all.length > 0 && (
          <Tiles testId="experiments-tiles">
            <Tile highlight value={nf(c.all)} label="entries in the registry" />
            <Tile value={nf(c.v1)} label="rebuilt on dataset release v1" note="same release, code and command give the same numbers" />
            <Tile value={nf(c.pre)} label="pre-v1 exploratory" note="kept as development history" />
            <Tile value={nf(c.pending)} label="decisions pending" />
          </Tiles>
        )}
        {all.length > 0 && <SourceLine parts={["/api/experiments (the research repo's experiments/*.md front matter, members only)"]} />}
      </ResearchHero>
      <Section testId="section-experiments">
        {d.loading && !d.error && <p className="small">Loading…</p>}
        {locked && (
          <div className="signedout" data-testid="experiments-signed-out">
            <span className="ic b" style={{ margin: 0, flex: "none" }}><Lock className="h-[18px] w-[18px]" /></span>
            <div>Experiments are for mycomap.org members.{" "}
              <a href={signInUrl()} className={`${PRIMARY_BUTTON} ml-1 !py-1.5`}>Sign in</a> to read them.</div>
          </div>
        )}
        {!!d.error && !locked && <p data-testid="experiments-error">The experiments could not be loaded.</p>}
        {(d.skipped ?? 0) > 0 && (
          <p className="text-xs text-[#7a5a3a] mb-3" data-testid="experiments-skipped">
            {d.skipped} {d.skipped === 1 ? "entry is" : "entries are"} not shown: still being written, or not yet meeting the registry's rules.
          </p>
        )}
        {all.length > 0 && (
          <>
            <ChartCard className="mb-3.5" testId="entries-by-day">
              <div className="flex flex-wrap justify-between gap-2"><h3 className="font-display font-semibold text-[#4a3728]">Entries by day</h3>
                <div className="legend" style={{ margin: 0 }}>{STRIP_ORDER.map((st) => <span key={st}><i style={{ ...STATUS_DOT[st], borderRadius: "50%" }} />{STATUS_NAMES[st]}</span>)}</div></div>
              <div className="tline mt-2.5">
                {entriesByDay(all).map((x) => (
                  <div key={x.day} className="tcol">
                    <div className="dots2">{[...x.statuses].sort((a, b) => STRIP_ORDER.indexOf(a) - STRIP_ORDER.indexOf(b)).map((st, i) => (
                      <i key={i} style={STATUS_DOT[st]} data-tip={`${shortDay(x.day)}|${STATUS_NAMES[st]}`} />))}</div>
                    <span>{shortDay(x.day).split(" ").slice(0, 2).join(" ")}</span>
                  </div>
                ))}
              </div>
              <p className="small mt-1.5">Release v1 was cut on 12 October 2026; entries from 13 October on are rebuilt on it.</p>
            </ChartCard>
            <div className="fbar" data-testid="experiments-filter">
              <div className="row">
                <SearchBox value={f.q} onChange={(q) => setF({ ...f, q })} placeholder="Search: place prior, fine-tune, not a fungus…" label="Search experiments" />
                <div className="seg ml-auto" role="group" aria-label="Dataset">
                  {DATASET_TABS.map(([v, l]) => (
                    <button key={v} type="button" className={f.dataset === v ? "on" : ""} aria-pressed={f.dataset === v}
                            onClick={() => setF({ ...f, dataset: v })} data-testid={`dataset-${v}`}>{l} {tabCount[v]}</button>
                  ))}
                </div>
              </div>
              <ChipRow label="Status" values={STRIP_ORDER} chosen={f.status} counts={c.status} words={STATUS_NAMES}
                       onToggle={(v) => setF({ ...f, status: toggle(f.status, v) })} />
              <ChipRow label="Test set" values={TEST_SETS} chosen={f.sets} counts={c.sets} words={TEST_SET_LABEL}
                       onToggle={(v) => setF({ ...f, sets: toggle(f.sets, v) })} />
            </div>
            <div className="fsum">
              <span data-testid="experiments-shown">{filtered ? `Showing ${shown.length} of ${all.length} entries, newest first` : `Showing all ${all.length} entries, newest first`}</span>
              {filtered && <button type="button" className="clear" onClick={() => setF(NO_EXPERIMENT_FILTER)}>Clear filters</button>}
            </div>
            <div className="xlist" data-testid="experiment-list">
              {shown.map((e) => (
                <Link key={e.slug} href={`/research/experiments/${e.slug}`} className="xrow" data-testid={`experiment-${e.slug}`}>
                  <span className="xd">{shortDay(e.date)}</span>
                  <span className="xm"><b>{e.title}</b><small>{excerpt(e.headline)}</small></span>
                  <span className="xs"><StatusPill status={e.status} />
                    <span className={`tag ${datasetOf(e) === PRE_V1 ? "" : "d"}`}>{datasetOf(e) === PRE_V1 ? "Pre-v1 exploratory" : `Release ${datasetOf(e)}`}</span>
                    {awaitsDecision(e) && <span className="tag" style={{ background: "#fff7e6", color: "#92400e" }}>decision pending</span>}</span>
                </Link>
              ))}
            </div>
            {!shown.length && <p className="small py-8 text-center" data-testid="experiments-none">No entry matches. <button type="button" className="clear" onClick={() => setF(NO_EXPERIMENT_FILTER)}>Clear filters</button></p>}
          </>
        )}
      </Section>
    </ResearchShell>
  );
}

export function ExperimentPage({ slug }: { slug: string }) {
  const q = useQuery({ queryKey: ["experiment", slug], queryFn: () => api.experiment(slug), retry: false });
  const e = q.data;
  const facts: [string, string | undefined][] = e ? [
    ["Question", e.question], ["Benchmark", e.benchmark], ["Split", e.split], ["Model", e.model],
    ["Methods", e.methods?.join(", ")], ["Code", [e.branch, e.commits?.join(", ")].filter(Boolean).join(" · ")],
    ["Decision", e.decision],
    ["Dataset release", e.dataset_release], ["Reference hash", e.reference_hash],
    ["Code commit", e.code_commit], ["Reproduce with", e.reproduce_command],
  ] : [];
  return (
    <ResearchShell>
      <ResearchHero crumbs={[{ href: "/research", label: "Research" }, { href: "/research/experiments", label: "Experiments" }, { label: e?.title ?? "Experiment" }]}
                    title={e?.title ?? "Experiment"} lede={e?.headline}
                    top={e ? <div className="flex flex-wrap items-center gap-2"><StatusPill status={e.status} /><StatusPill status="members" /></div> : undefined} />
      <article className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 max-w-3xl text-[#5c4a3a] leading-relaxed">
        {q.isPending && <p>Loading…</p>}
        {q.error instanceof ApiError && q.error.status === 404 && <p>No such experiment.</p>}
        {!(q.error instanceof ApiError && q.error.status === 404) && <SignInNote error={q.error} />}
        {e && (
          <>
            <p className="flex flex-wrap items-center gap-2 text-sm">
              <Status s={e.status} /> <Repro r={e.reproducibility} on={e.reproduced_on} />
              <span className="text-muted-foreground">{e.date}</span>
            </p>
            <p className="mt-2 text-sm text-muted-foreground">
              {reproducibilityWords(e.reproducibility, e.reproduced_on).note}
            </p>
            {e.reproduced_by && (
              <p className="mt-2 text-sm" data-testid="reproduced-by">
                Re-run on release {e.reproduced_on}:{" "}
                <Link href={`/research/experiments/${e.reproduced_by}`} className="text-myco-green underline">
                  {e.reproduced_by}
                </Link>
              </p>
            )}
            <dl className="mt-4 grid grid-cols-[8rem_1fr] gap-x-4 gap-y-1 text-sm">
              {facts.filter(([, v]) => v).map(([k, v]) => (
                <div key={k} className="contents">
                  <dt className="text-muted-foreground">{k}</dt>
                  <dd className="min-w-0 break-words">{v}</dd>
                </div>
              ))}
            </dl>
            <div className="mt-6"><Markdown>{e.markdown}</Markdown></div>
            {e.related && e.related.length > 0 && (
              <p className="mt-8 text-sm">
                Related:{" "}
                {e.related.map((r, i) => (
                  <span key={r}>
                    {i > 0 && ", "}
                    <Link href={`/research/experiments/${r}`} className="text-myco-green underline">{r}</Link>
                  </span>
                ))}
              </p>
            )}
          </>
        )}
        <p className="mt-8 text-sm"><Link href="/research/experiments" className="text-myco-green underline">All experiments</Link></p>
      </article>
    </ResearchShell>
  );
}
