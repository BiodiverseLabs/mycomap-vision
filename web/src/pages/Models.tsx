import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Trophy } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PageHeader } from "@/components/Layout";
import { api, modelLabel, num, pct, RANKS, type RunReport, type ScoreRun } from "@/lib/api";

const BUCKET_ORDER = ["all", "names iNat knows", "novel (0 refs)", "1 ref", "2 refs", "3-5 refs",
                      "6-30 refs", "31+ refs"];

export function ModelsPage() {
  const board = useQuery({ queryKey: ["scoreboard"], queryFn: api.scoreboard });
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const [open, setOpen] = useState<number | null>(null);
  const groups = groupBy(board.data?.runs ?? [], (r) => r.comparison_id);
  const speed = Object.fromEntries(
    (models.data?.backbones ?? []).map((b) => [b.backbone, b.photos_per_second]),
  );

  return (
    <>
      <PageHeader title="Models">
        Every comparison scores its models on exactly the same records: those that turned green
        in the newest weeks, identified from the older ones. Top-1 means the first answer was
        right.
      </PageHeader>
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        <section>
          <h2 className="font-display text-2xl text-[#4a3728] mb-3">Scoreboard</h2>
          {board.isLoading && <p className="text-sm text-muted-foreground">Loading…</p>}
          {board.data && !board.data.runs.length && (
            <p className="text-sm text-muted-foreground">
              No comparisons yet. Run <code>mv compare --backbones a,b --methods nearest,species-mean</code>.
            </p>
          )}
          <div className="space-y-6">
            {groups.map(([cid, runs]) => (
              <Card key={cid}>
                <CardHeader className="bg-[#f8f5f0] border-b border-[#A87146]/10 py-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <CardTitle className="text-base text-[#4a3728]">Comparison {cid}</CardTitle>
                    <span className="text-xs text-muted-foreground">
                      test: green after {runs[0].cutoff} · {num(runs[0].n_test)} test /{" "}
                      {num(runs[0].n_reference)} reference records · set {runs[0].record_set}
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
                        <th className="text-right font-medium px-3 py-2" title="Species top-1 from the first photo only">
                          1st photo only
                        </th>
                        <th className="text-right font-medium px-3 py-2" title="Embedding speed on the laptop GPU">
                          Photos/s
                        </th>
                        <th className="text-right font-medium px-4 py-2">Code</th>
                      </tr>
                    </thead>
                    <tbody>
                      {runs.map((r, i) => (
                        <RunRow key={r.id} r={r} best={i === 0} open={open === r.id}
                                speed={speed[r.backbone] ?? null}
                                onToggle={() => setOpen(open === r.id ? null : r.id)} />
                      ))}
                    </tbody>
                  </table>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        <section>
          <h2 className="font-display text-2xl text-[#4a3728] mb-3">Backbones</h2>
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {models.data?.backbones.map((b) => (
              <Card key={b.backbone}>
                <CardContent className="pt-4">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-[#4a3728]">{b.backbone}</span>
                    <Badge variant={b.embedded_photos ? "default" : "outline"}>
                      {b.embedded_photos ? `${num(b.embedded_photos)} photos` : "not embedded"}
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground mt-1">{b.note}</p>
                  <p className="text-xs text-muted-foreground mt-2 break-all font-mono">{b.spec}</p>
                </CardContent>
              </Card>
            ))}
          </div>
          <p className="text-sm text-muted-foreground mt-3">
            Methods: {models.data?.methods.join(", ")}. Any timm or open_clip model can be added with{" "}
            <code>mv embed --backbone timm:&lt;name&gt;</code> and scored with <code>mv compare</code>.
          </p>
        </section>
      </div>
    </>
  );
}

function RunRow({ r, best, open, speed, onToggle }: {
  r: ScoreRun; best: boolean; open: boolean; speed: number | null; onToggle: () => void;
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
        <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">{pct(r.species_top1_first_photo, 1)}</td>
        <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
          {external ? "API" : speed == null ? "–" : Math.round(speed)}
        </td>
        <td className="px-4 py-2 text-right text-xs text-muted-foreground font-mono">{r.code_version}</td>
      </tr>
      {open && (
        <tr className="border-b bg-[#faf9f7]">
          <td colSpan={7} className="px-4 py-3">
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
          {(["project", "observer"] as const).map((kind) => (
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
                  {q.data.all_photos.groups![kind].map((g) => (
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

function groupBy<T>(items: T[], key: (t: T) => string): [string, T[]][] {
  const m = new Map<string, T[]>();
  for (const it of items) m.set(key(it), [...(m.get(key(it)) ?? []), it]);
  return [...m.entries()];
}
