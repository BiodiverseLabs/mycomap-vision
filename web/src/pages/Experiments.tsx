import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "wouter";
import { PageHeader } from "@/components/Layout";
import { Markdown } from "@/components/Markdown";
import { api, ApiError, signInUrl } from "@/lib/api";
import { awaitsDecision, EXPERIMENT_STATUSES, filterExperiments, STATUS_WORDS, statusCounts,
         type ExperimentStatus } from "@/lib/experiments";

// Research > Experiments (Steve, 2026-10-09): every experiment recorded so it can be reviewed.
// Members only: development numbers and unpublished results (docs/experiments/README.md).

function Status({ s }: { s: ExperimentStatus }) {
  const w = STATUS_WORDS[s];
  return <span className={`rounded px-2 py-0.5 text-xs font-medium ${w.tone}`}>{w.label}</span>;
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

export function ExperimentsPage() {
  const q = useQuery({ queryKey: ["experiments"], queryFn: api.experiments, retry: false });
  const [status, setStatus] = useState<ExperimentStatus | null>(null);
  const all = q.data?.experiments ?? [];
  const counts = statusCounts(all);
  const shown = filterExperiments(all, status);
  const chip = (on: boolean) => `rounded-full border px-3 py-1 text-sm ${
    on ? "border-myco-green bg-myco-green/10 text-myco-green" : "border-[#A87146]/30 bg-white text-gray-700 hover:bg-myco-green/5"}`;
  return (
    <>
      <PageHeader title="Experiments">
        Every experiment on MycoMap Vision: the question, everything tried, the numbers, and what
        was decided. Development results, for mycomap.org members.
      </PageHeader>
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 max-w-4xl space-y-4">
        {q.isPending && <p className="text-sm text-muted-foreground">Loading…</p>}
        <SignInNote error={q.error} />
        {all.length > 0 && (
          <div className="flex flex-wrap gap-2" role="group" aria-label="Filter by status">
            <button className={chip(status === null)} onClick={() => setStatus(null)}>All ({all.length})</button>
            {EXPERIMENT_STATUSES.map((s) => (
              <button key={s} className={chip(status === s)} onClick={() => setStatus(s)}
                      disabled={!counts[s]}>
                {STATUS_WORDS[s].label} ({counts[s]})
              </button>
            ))}
          </div>
        )}
        {shown.map((e) => (
          <Link key={e.slug} href={`/research/experiments/${e.slug}`}
                className="block rounded-lg border border-[#A87146]/20 bg-white p-4 hover:border-myco-green/50 hover:bg-myco-green/5">
            <span className="flex flex-wrap items-center gap-2">
              <span className="font-semibold text-[#4a3728]">{e.title}</span>
              <Status s={e.status} />
              {awaitsDecision(e) && <span className="text-xs text-[#7a5a3a]">decision pending</span>}
              <span className="text-xs text-muted-foreground ml-auto">{e.date}</span>
            </span>
            <p className="text-sm mt-1 text-[#5c4a3a]">{e.headline}</p>
            <p className="text-sm mt-1 text-muted-foreground">{e.verdict}</p>
          </Link>
        ))}
      </div>
    </>
  );
}

export function ExperimentPage({ slug }: { slug: string }) {
  const q = useQuery({ queryKey: ["experiment", slug], queryFn: () => api.experiment(slug), retry: false });
  const e = q.data;
  const facts: [string, string | undefined][] = e ? [
    ["Question", e.question], ["Benchmark", e.benchmark], ["Split", e.split], ["Model", e.model],
    ["Methods", e.methods?.join(", ")], ["Code", [e.branch, e.commits?.join(", ")].filter(Boolean).join(" · ")],
    ["Decision", e.decision],
  ] : [];
  return (
    <>
      <PageHeader title={e?.title ?? "Experiment"}>{e && e.headline}</PageHeader>
      <article className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 max-w-3xl text-[#5c4a3a] leading-relaxed">
        {q.isPending && <p>Loading…</p>}
        {q.error instanceof ApiError && q.error.status === 404 && <p>No such experiment.</p>}
        {!(q.error instanceof ApiError && q.error.status === 404) && <SignInNote error={q.error} />}
        {e && (
          <>
            <p className="flex flex-wrap items-center gap-2 text-sm">
              <Status s={e.status} /> <span className="text-muted-foreground">{e.date}</span>
            </p>
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
    </>
  );
}
