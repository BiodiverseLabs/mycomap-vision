import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Trophy } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PageHeader } from "@/components/Layout";
import { api, modelLabel, num, pct, RANKS, type RunReport, type ScoreRun } from "@/lib/api";
import { modelName } from "@/lib/publicView";
import { CompareApproaches } from "@/components/CompareApproaches";
import { TwoModels } from "@/components/TwoModels";
import { incompleteRun, releaseResultsPending, scoreboardSections } from "@/lib/compareView";
import { utcDay, V1 } from "@/lib/datasetReleases";

const BUCKET_ORDER = ["all", "names iNat knows", "novel (0 refs)", "1 ref", "2 refs", "3-5 refs",
                      "6-30 refs", "31+ refs"];

export function ModelsPage() {
  const board = useQuery({ queryKey: ["scoreboard"], queryFn: api.scoreboard });
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const benchmarks = useQuery({ queryKey: ["benchmarks"], queryFn: api.benchmarks });
  const [open, setOpen] = useState<number | null>(null);
  // iNaturalist's incomplete development run is left out everywhere here (compareView.incompleteRun).
  const runs = (board.data?.runs ?? []).filter((r) => !incompleteRun(r));
  const { release, history } = scoreboardSections(runs);
  // Without a release comparison yet, the newest development test leads, as before.
  const top = release.length ? release : history.slice(0, 1);
  const earlier = release.length ? history : history.slice(1);
  const leftOut = (board.data?.runs ?? []).some(incompleteRun);

  return (
    <>
      <PageHeader title="Results">
        Every comparison scores its models on exactly the same records, identified from the
        records validated before them: on dataset release {V1.id}, its validation split (first
        validated after {utcDay(V1.valCutoff)}); before the release, the newest weeks of the live
        data. Top-1 means the first answer was right.
      </PageHeader>
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {board.data && benchmarks.data
          && releaseResultsPending(board.data.runs, benchmarks.data.benchmarks) && (
          <p className="text-sm text-[#7a5a3a] max-w-3xl" data-testid="text-release-pending">
            Results on dataset release {V1.id} are not published here yet. Until they are, the tests
            below are development history from before the release was frozen.
          </p>
        )}
        <TwoModels />
        {board.data && runs.length > 0 && (
          <CompareApproaches runs={runs} served={models.data?.ready ?? []}
                             servedMethod={models.data?.default_method ?? "nearest"} />
        )}
        <section>
          <h2 className="font-display font-semibold text-2xl text-[#4a3728] mb-3">Scoreboard</h2>
          {board.isLoading && <p className="text-sm text-muted-foreground">Loading…</p>}
          {board.data && !board.data.runs.length && (
            <p className="text-sm text-muted-foreground">No tests published yet.</p>
          )}
          <div className="space-y-6">
            {top.map((g) => (
              <Comparison key={g[0].id} runs={g} open={open} setOpen={setOpen} />
            ))}
            {earlier.length > 0 && (
              <details className="rounded-lg border border-[#A87146]/20 p-4">
                <summary className="cursor-pointer text-sm text-muted-foreground">
                  Earlier tests ({earlier.length}), from development before dataset release {V1.id}, on the live data
                </summary>
                <div className="space-y-6 mt-4">
                  {earlier.map((g) => (
                    <Comparison key={g[0].id} runs={g} open={open} setOpen={setOpen} />
                  ))}
                  {leftOut && (
                    <p className="text-xs text-muted-foreground" data-testid="text-inat-incomplete">
                      iNaturalist's computer vision on the 1,152 records of the 8 October 2026 test is
                      not shown: that was an incomplete development run, not a fair comparison. It
                      will be scored on the sealed test after the analysis freeze.
                    </p>
                  )}
                </div>
              </details>
            )}
          </div>
        </section>

        <Prospective />

        <section>
          <h2 className="font-display font-semibold text-2xl text-[#4a3728] mb-3">Models in use</h2>
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {models.data?.backbones.filter((b) => b.embedded_photos).map((b) => (
              <Card key={b.backbone}>
                <CardContent className="pt-4">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-semibold text-[#4a3728]" title={b.backbone}>{modelName(b.backbone)}</span>
                    <Badge variant="default">{num(b.embedded_photos)} photos</Badge>
                  </div>
                  {b.note && <p className="text-sm text-muted-foreground mt-1">{b.note}</p>}
                </CardContent>
              </Card>
            ))}
          </div>
        </section>
      </div>
    </>
  );
}

function Comparison({ runs, open, setOpen }: {
  runs: ScoreRun[]; open: number | null; setOpen: (id: number | null) => void;
}) {
  const release = runs.find((r) => r.dataset_release)?.dataset_release;
  const ids = [...new Set(runs.map((r) => r.comparison_id))].join(", ");
  return (
              <Card>
                <CardHeader className="bg-[#f8f5f0] border-b border-[#A87146]/10 py-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <CardTitle className="text-base text-[#4a3728]">
                      {release ? `Dataset release ${release}, validation split: ` : "Test on "}
                      {num(runs[0].n_test)} records validated after {runs[0].cutoff}
                    </CardTitle>
                    <span className="text-xs text-muted-foreground" title={`comparison ${ids}, set ${runs[0].record_set}`}>
                      identified from {num(runs[0].n_reference)} earlier DNA-verified records
                    </span>
                  </div>
                </CardHeader>
                <CardContent className="p-0 overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="text-xs uppercase tracking-wider text-muted-foreground">
                      <tr className="border-b">
                        <th className="text-left font-medium px-4 py-2">Model</th>
                        <th className="text-right font-medium px-3 py-2">Species</th>
                        <th className="text-right font-medium px-3 py-2">Genus</th>
                        <th className="text-right font-medium px-3 py-2">Family</th>
                        <th className="text-right font-medium px-4 py-2" title="Species top-1 from the first photo only">
                          1st photo only
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {runs.map((r, i) => (
                        <RunRow key={r.id} r={r} best={i === 0} open={open === r.id}
                                onToggle={() => setOpen(open === r.id ? null : r.id)} />
                      ))}
                    </tbody>
                  </table>
                </CardContent>
              </Card>
  );
}

function Prospective() {
  const q = useQuery({ queryKey: ["prospective"], queryFn: api.prospective });
  const rows = q.data?.models ?? [];
  return (
    <section>
      <h2 className="font-display font-semibold text-2xl text-[#4a3728] mb-1">Advance predictions</h2>
      <p className="text-sm text-muted-foreground mb-3 max-w-3xl">
        Records with a sequence but no validation yet are identified now; when they turn green,
        the saved answer is checked against the DNA name. Only predictions made before the
        answer existed count.
      </p>
      {rows.some((r) => (r.sealed ?? 0) > 0) && (
        <p className="text-xs text-[#7a5a3a] mb-3 max-w-3xl" data-testid="prospective-sealed">
          Records validated since {q.data?.sealed_after?.slice(0, 10)} form the paper's sealed test:
          their {rows.reduce((n, r) => n + (r.sealed ?? 0), 0).toLocaleString()} predictions are
          counted but not scored here until that test is opened.
        </p>
      )}
      {q.isLoading ? (
        <p className="text-sm text-muted-foreground">Loading…</p>
      ) : !rows.length ? (
        <p className="text-sm text-muted-foreground">None yet.</p>
      ) : (
        <Card>
          <CardContent className="p-0 overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-xs uppercase tracking-wider text-muted-foreground">
                <tr className="border-b">
                  <th className="text-left font-medium px-4 py-2">Model</th>
                  <th className="text-right font-medium px-3 py-2">Predicted</th>
                  <th className="text-right font-medium px-3 py-2">Validated since</th>
                  <th className="text-right font-medium px-3 py-2">Species</th>
                  <th className="text-right font-medium px-3 py-2">Genus</th>
                  <th className="text-right font-medium px-3 py-2">Family</th>
                  <th className="text-right font-medium px-4 py-2" title="Average stated confidence in the top species; compare with the species column">
                    Stated confidence
                  </th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => {
                  const label = modelLabel(r.backbone, r.method);
                  return (
                    <tr key={`${r.backbone}/${r.method}`} className="border-b">
                      <td className="px-4 py-2">
                        <span className="font-medium">{label.name}</span>{" "}
                        <span className="text-muted-foreground">· {label.how}</span>
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums">{num(r.predicted)}</td>
                      <td className="px-3 py-2 text-right tabular-nums">{num(r.resolved)}</td>
                      <td className="px-3 py-2 text-right tabular-nums font-semibold">{pct(r.species_top1, 1)}</td>
                      <td className="px-3 py-2 text-right tabular-nums">{pct(r.genus_top1, 1)}</td>
                      <td className="px-3 py-2 text-right tabular-nums">{pct(r.family_top1, 1)}</td>
                      <td className="px-4 py-2 text-right tabular-nums text-muted-foreground">
                        {pct(r.mean_species_confidence, 1)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </CardContent>
        </Card>
      )}
    </section>
  );
}

function RunRow({ r, best, open, onToggle }: {
  r: ScoreRun; best: boolean; open: boolean; onToggle: () => void;
}) {
  const label = modelLabel(r.backbone, r.method);
  const external = r.backbone.startsWith("external:");
  return (
    <>
      <tr className={`border-b hover:bg-myco-green/5 cursor-pointer ${external ? "bg-[#fbf7f0]" : ""}`}
          onClick={onToggle}>
        <td className="px-4 py-2">
          <span className="font-medium">{label.name}</span>{" "}
          <span className="text-muted-foreground">· {label.how}</span>
          {best && <Trophy className="inline h-3.5 w-3.5 ml-1.5 text-myco-green" />}
          {external && <Badge variant="outline" className="ml-2 px-1.5 py-0 text-[10px] font-normal">baseline</Badge>}
        </td>
        <td className="px-3 py-2 text-right tabular-nums font-semibold">{pct(r.species_top1, 1)}</td>
        <td className="px-3 py-2 text-right tabular-nums">{pct(r.genus_top1, 1)}</td>
        <td className="px-3 py-2 text-right tabular-nums">{pct(r.family_top1, 1)}</td>
        <td className="px-4 py-2 text-right tabular-nums text-muted-foreground">{pct(r.species_top1_first_photo, 1)}</td>
      </tr>
      {open && (
        <tr className="border-b bg-[#faf9f7]">
          <td colSpan={5} className="px-4 py-3" title={`code ${r.code_version}`}>
            <RunDetail id={r.id} />
          </td>
        </tr>
      )}
    </>
  );
}

function RunDetail({ id }: { id: number }) {
  const q = useQuery<RunReport>({ queryKey: ["run", id], queryFn: () => api.run(id) });
  if (!q.data) return <p className="text-sm text-muted-foreground">Loading…</p>;
  const buckets = BUCKET_ORDER.filter((b) => q.data.all_photos.species[b]);
  return (
    <div className="overflow-x-auto">
      <p className="text-xs text-muted-foreground mb-2">
        Top-1 / top-5 by how many DNA-verified reference records the right species has. Novel
        species can't be right at species level; they show how the higher ranks cope.
      </p>
      <table className="text-xs">
        <thead>
          <tr className="text-muted-foreground">
            <th className="text-left pr-4 py-1 font-medium">Reference records</th>
            <th className="text-right pr-4 font-medium">Tests</th>
            {RANKS.map((rank) => (
              <th key={rank} className="text-right pr-4 font-medium capitalize">{rank}</th>
            ))}
            <th className="text-right font-medium">Species, 1st photo</th>
          </tr>
        </thead>
        <tbody>
          {buckets.map((b) => (
            <tr key={b}>
              <td className="pr-4 py-0.5">{b}</td>
              <td className="text-right pr-4 tabular-nums">{num(q.data.all_photos.species[b]?.n)}</td>
              {RANKS.map((rank) => {
                const s = q.data.all_photos[rank][b];
                return (
                  <td key={rank} className="text-right pr-4 tabular-nums">
                    {s ? `${pct(s.top1)} / ${pct(s.top5)}` : "–"}
                  </td>
                );
              })}
              <td className="text-right tabular-nums text-muted-foreground">
                {pct(q.data.first_photo_only.species[b]?.top1)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {q.data.all_photos.groups && (
        <div className="mt-4 grid gap-4 md:grid-cols-2">
          {(["project", "observer"] as const).filter((kind) => q.data.all_photos.groups![kind]).map((kind) => (
            <div key={kind}>
              <p className="text-xs text-muted-foreground mb-1">
                By {kind} (largest first; batches can be dominated by one {kind}):
              </p>
              <table className="text-xs w-full">
                <thead>
                  <tr className="text-muted-foreground">
                    <th className="text-left font-medium py-0.5 capitalize">{kind}</th>
                    <th className="text-right font-medium">Tests</th>
                    <th className="text-right font-medium">Species</th>
                    <th className="text-right font-medium">Genus</th>
                  </tr>
                </thead>
                <tbody>
                  {q.data.all_photos.groups![kind]!.map((g) => (
                    <tr key={g.name}>
                      <td className="py-0.5 pr-2 truncate max-w-[14rem]" title={g.name}>{g.name}</td>
                      <td className="text-right tabular-nums">{num(g.n)}</td>
                      <td className="text-right tabular-nums">{pct(g.species_top1)}</td>
                      <td className="text-right tabular-nums">{pct(g.genus_top1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        </div>
      )}
      {q.data.species_names_inat_knows != null && (
        <p className="text-xs text-muted-foreground mt-3">
          iNat has a taxon for {num(q.data.species_names_inat_knows)} of these test records' DNA
          names; it can't be right at species level for the others.
        </p>
      )}
      {q.data.species_by_observed_year && (
        <div className="mt-3">
          <p className="text-xs text-muted-foreground mb-1">
            Species top-1 by year observed (older photos are more likely to be in iNat's training
            data):
          </p>
          <div className="flex flex-wrap gap-2 text-xs">
            {Object.entries(q.data.species_by_observed_year).map(([year, s]) => (
              <span key={year} className="rounded border px-2 py-0.5 bg-white tabular-nums">
                {year}: {s ? `${pct(s.top1)} of ${s.n}` : "–"}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

