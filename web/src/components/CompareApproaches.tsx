import { useMemo, useState } from "react";
import { useQueries, useQuery } from "@tanstack/react-query";
import { api, modelLabel, num, pct, type Rank, type RunReport, type ScoreRun } from "@/lib/api";
import { bars, comparisons, defaultComparison, depthRows, heat, heldoutBars, heldoutDepth,
         heldoutKs, kChoices, STANDARD_BANDS, type Bar, type PublishedBenchmark } from "@/lib/compareView";

// "Approaches compared" on the Results page (Research > Results) (Steve, 2026-10-09): every model and method,
// ours and outside ones, on the same records, by rank, top-k and reference depth.

const OURS = "#2a78d6";      // validated categorical slots 1 and 2 (dataviz palette)
const OUTSIDE = "#eb6834";
const RANK_LABEL: Record<Rank, string> = { species: "Species", genus: "Genus", family: "Family" };

interface Approach { name: string; how: string; trained: string; names: string; status: string }

/** How the approaches differ. Rows without results yet say so. */
const APPROACHES: Approach[] = [
  { name: "Nearest + species average", how: "Blends each photo's two best specimen matches with the species' average look",
    trained: "BioCLIP 2, last 4 of 24 blocks fine-tuned on our records", names: "Any species with one DNA-verified record; new records added nightly",
    status: "On the site (the default since October 2026); confirmed on held-out records" },
  { name: "Nearest specimen", how: "Finds the most similar DNA-verified specimens; each species scored by its best match to all your photos",
    trained: "Same model", names: "Same as above", status: "Tested; the site's method until October 2026" },
  { name: "Species average", how: "Compares your photos with each species' average photo", trained: "Same model",
    names: "Same as nearest specimen", status: "Tested" },
  { name: "Trained classifier (small)", how: "A classifier trained on top of finished image features",
    trained: "Only the classifier; the image model unchanged", names: "Species in its training list", status: "Tested: lost to nearest specimen" },
  { name: "iNaturalist computer vision", how: "iNat's own model; best score across your photos, with or without location",
    trained: "Community-identified iNat observations", names: "Species iNat has as taxa (few provisional names)",
    status: "Outside baseline; being measured on the newest test and held-out records" },
  { name: "Danish Fungi method on our records", how: "Whole network retrained as a classifier (BEiT, 384 px, rare-class loss), photo probabilities averaged, month prior",
    trained: "Our DNA-verified records", names: "Species in its training list; a new species needs a retrain",
    status: "Being built (replication of Picek et al.)" },
  { name: "Published Danish Fungi / FungiTastic models", how: "The published models, unchanged",
    trained: "Danish Fungal Atlas records (expert-checked)", names: "Mostly European species",
    status: "Tested on held-out development records: can name 13-19% of formally named North American records; behind on the species they share" },
];

export function CompareApproaches({ runs, served, servedMethod = "nearest" }: {
  runs: ScoreRun[]; served: string[]; servedMethod?: string;
}) {
  const groups = useMemo(() => comparisons(runs), [runs]);
  const [pick, setPick] = useState<number | null>(null);
  const idx = pick ?? defaultComparison(groups, served, servedMethod);
  const group = groups[idx] ?? [];
  const reportQs = useQueries({
    queries: group.map((r) => ({ queryKey: ["run", r.id], queryFn: () => api.run(r.id) })),
  });
  const reports = new Map<number, RunReport>();
  group.forEach((r, i) => { const d = reportQs[i]?.data; if (d) reports.set(r.id, d); });
  const loading = reportQs.some((q) => q.isLoading);

  return (
    <section data-testid="section-compare-approaches" className="space-y-6">
      <div>
        <h2 className="font-display font-semibold text-2xl text-[#4a3728] mb-1">Approaches compared</h2>
        <p className="text-sm text-muted-foreground max-w-3xl">
          Every way of naming a find that we have tried or are testing, including outside models,
          scored on exactly the same DNA-verified records. A model is only compared with others
          measured on the same records.
        </p>
      </div>
      <ApproachesTable />
      <HeldOut />
      <h3 className="font-semibold text-lg text-[#4a3728] pt-2">Newest weeks of validations</h3>
      {group.length > 0 && (
        <>
          <ComparisonPicker groups={groups} idx={idx} onPick={setPick} />
          {loading ? <p className="text-sm text-muted-foreground">Loading results…</p> : (
            <>
              <SideBySide group={group} reports={reports} served={served} servedMethod={servedMethod} />
              <DepthTable group={group} reports={reports} />
            </>
          )}
        </>
      )}
    </section>
  );
}

function ApproachesTable() {
  return (
    <div className="overflow-x-auto rounded-lg border border-[#A87146]/20">
      <table className="w-full text-sm">
        <thead className="text-xs uppercase tracking-wider text-muted-foreground bg-[#f8f5f0]">
          <tr>
            <th className="text-left font-medium px-3 py-2">Approach</th>
            <th className="text-left font-medium px-3 py-2">How it names a species</th>
            <th className="text-left font-medium px-3 py-2">Trained on</th>
            <th className="text-left font-medium px-3 py-2">Which species it can name</th>
            <th className="text-left font-medium px-3 py-2">Status</th>
          </tr>
        </thead>
        <tbody>
          {APPROACHES.map((a) => (
            <tr key={a.name} className="border-t align-top">
              <td className="px-3 py-2 font-medium text-[#4a3728]">{a.name}</td>
              <td className="px-3 py-2">{a.how}</td>
              <td className="px-3 py-2">{a.trained}</td>
              <td className="px-3 py-2">{a.names}</td>
              <td className="px-3 py-2 text-muted-foreground">{a.status}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ComparisonPicker({ groups, idx, onPick }: {
  groups: ScoreRun[][]; idx: number; onPick: (i: number) => void;
}) {
  return (
    <label className="flex flex-wrap items-center gap-2 text-sm">
      <span className="text-muted-foreground">Test records:</span>
      <select className="rounded-md border border-[#A87146]/30 bg-white px-2 py-1 max-w-full min-w-0"
              value={idx} onChange={(e) => onPick(Number(e.target.value))}>
        {groups.map((g, i) => (
          <option key={g[0].id} value={i}>
            {num(g[0].n_test)} records validated after {g[0].cutoff} ({g.length} approach
            {g.length === 1 ? "" : "es"}){i === 0 ? ", most recent" : ""}
          </option>
        ))}
      </select>
    </label>
  );
}

function Toggle<T extends string | number>({ options, value, onChange, label }: {
  options: T[]; value: T; onChange: (v: T) => void; label: (v: T) => string;
}) {
  return (
    <div className="inline-flex rounded-md border border-[#A87146]/30 overflow-hidden text-sm">
      {options.map((o) => (
        <button key={String(o)} onClick={() => onChange(o)}
                className={`px-3 py-1 ${o === value ? "bg-myco-green/10 text-myco-green font-medium"
                                                    : "bg-white text-gray-700 hover:bg-myco-green/5"}`}>
          {label(o)}
        </button>
      ))}
    </div>
  );
}

function SideBySide({ group, reports, served, servedMethod }: {
  group: ScoreRun[]; reports: Map<number, RunReport>; served: string[]; servedMethod: string;
}) {
  const ks = kChoices(group, reports);
  const [rank, setRank] = useState<Rank>("species");
  const [kPick, setK] = useState(1);
  const k = ks.includes(kPick) ? kPick : 1;
  const rows = bars(group, reports, rank, k, served, servedMethod);
  const [hover, setHover] = useState<number | null>(null);
  const anyOutside = rows.some((r) => r.external);
  return (
    <div className="rounded-lg border border-[#A87146]/20 bg-white p-4" data-testid="chart-side-by-side">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
        <h3 className="font-semibold text-[#4a3728]">
          {RANK_LABEL[rank]} right in the top {k === 1 ? "answer" : k}, all photos of a find
        </h3>
        <div className="flex flex-wrap gap-2">
          <Toggle options={["species", "genus", "family"] as Rank[]} value={rank} onChange={setRank}
                  label={(r) => RANK_LABEL[r]} />
          <Toggle options={ks} value={k} onChange={setK} label={(v) => `Top ${v}`} />
        </div>
      </div>
      <div className="flex gap-4 text-xs text-muted-foreground mb-2">
        <span className="inline-flex items-center gap-1.5"><Swatch c={OURS} /> MycoMap Vision</span>
        {anyOutside && <span className="inline-flex items-center gap-1.5"><Swatch c={OUTSIDE} /> Outside model</span>}
      </div>
      <div className="space-y-1.5">
        {rows.map((b) => <BarRow key={b.id} b={b} hover={hover === b.id} onHover={setHover} rank={rank} k={k} />)}
      </div>
      {rows.some((r) => r.value == null) && (
        <p className="text-xs text-muted-foreground mt-3">
          "Not measured": that approach was scored before top 3 and top 10 were recorded.
        </p>
      )}
    </div>
  );
}

function Swatch({ c }: { c: string }) {
  return <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: c }} />;
}

function BarRow({ b, hover, onHover, rank, k }: {
  b: Bar; hover: boolean; onHover: (id: number | null) => void; rank: Rank; k: number;
}) {
  const label = modelLabel(b.backbone, b.method);
  const w = b.value == null ? 0 : Math.max(0.5, 100 * b.value);
  return (
    <div className="grid grid-cols-[minmax(0,16rem)_1fr] items-center gap-3 relative"
         onMouseEnter={() => onHover(b.id)} onMouseLeave={() => onHover(null)}>
      <div className="text-sm leading-tight min-w-0">
        <span className="font-medium text-[#4a3728]">{label.name}</span>{" "}
        <span className="text-muted-foreground">· {label.how}</span>
        {b.served && <span className="ml-1.5 text-[10px] uppercase tracking-wide text-myco-green">on the site</span>}
      </div>
      <div className="flex items-center gap-2 h-6">
        <div className="h-[18px] rounded-r-[4px]" style={{ width: `${w}%`, background: b.external ? OUTSIDE : OURS,
             opacity: hover ? 1 : 0.9 }} />
        <span className="text-sm tabular-nums text-[#0b0b0b] whitespace-nowrap">
          {b.value == null ? "not measured" : pct(b.value, 1)}
        </span>
      </div>
      {hover && (
        <div role="tooltip" className="absolute right-0 -top-8 z-10 rounded-md border bg-white px-2 py-1 text-xs shadow">
          {label.name} · {label.how}: {RANK_LABEL[rank].toLowerCase()} top {k}{" "}
          {b.value == null ? "not measured" : pct(b.value, 1)}
          {b.n != null && ` of ${num(b.n)} records`}
        </div>
      )}
    </div>
  );
}

function DepthTable({ group, reports }: { group: ScoreRun[]; reports: Map<number, RunReport> }) {
  const { rows, missing } = depthRows(group, reports);
  const counts = STANDARD_BANDS.map((_, i) => rows[0]?.cells[i]?.n ?? null);
  return (
    <div className="rounded-lg border border-[#A87146]/20 bg-white p-4" data-testid="chart-depth">
      <h3 className="font-semibold text-[#4a3728]">Species right first time, by how many DNA-verified records the species has</h3>
      <p className="text-xs text-muted-foreground mb-3">
        Darker = more often right. A species with no reference record can't be named by any
        approach that learns from references.
      </p>
      {rows.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          These results were saved before the depth bands were recorded; they appear once the
          approaches are rescored.
        </p>
      ) : (
        <div className="overflow-x-auto">
          <table className="text-sm">
            <thead>
              <tr className="text-xs text-muted-foreground">
                <th className="text-left font-medium pr-4 py-1">Approach</th>
                {STANDARD_BANDS.map((band, i) => (
                  <th key={band} className="font-medium px-1 text-center min-w-[4.5rem]">
                    {band} records<br />
                    <span className="font-normal">{counts[i] != null ? `${num(counts[i])} finds` : ""}</span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const label = modelLabel(r.backbone, r.method);
                return (
                  <tr key={r.id}>
                    <td className="pr-4 py-0.5">
                      <span className="inline-flex items-center gap-1.5">
                        <Swatch c={r.external ? OUTSIDE : OURS} />
                        <span className="font-medium text-[#4a3728]">{label.name}</span>
                        <span className="text-muted-foreground">· {label.how}</span>
                      </span>
                    </td>
                    {r.cells.map((c, i) => {
                      const h = heat(c?.top1 ?? null);
                      return (
                        <td key={i} className="px-0.5 py-0.5">
                          <div className="rounded px-2 py-1 text-center tabular-nums"
                               style={{ background: h.bg, color: h.ink }}
                               title={c ? `${pct(c.top1, 1)} top 1, ${pct(c.top5, 1)} top 5, of ${num(c.n)} finds` : "no finds"}>
                            {c ? pct(c.top1, 0) : "–"}
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      {missing > 0 && rows.length > 0 && (
        <p className="text-xs text-muted-foreground mt-2">
          {missing} approach{missing === 1 ? " was" : "es were"} saved before the depth bands
          were recorded and {missing === 1 ? "is" : "are"} not shown here.
        </p>
      )}
    </div>
  );
}

/** A held-out model key's label: our wording for the method, and where the place came from. */
function heldoutLabel(backbone: string, method: string) {
  const [base, place] = method.split("@");
  const label = modelLabel(backbone, base);
  return { ...label, how: place ? `${label.how} (place from ${place === "org" ? "mycomap.org" : place})`
                                : label.how };
}

/** Held-out benchmarks: records no model saw in training or as references. */
function HeldOut() {
  const q = useQuery({ queryKey: ["benchmarks"], queryFn: api.benchmarks });
  const list = q.data?.benchmarks ?? [];
  const [pick, setPick] = useState(0);
  const [rank, setRank] = useState<Rank>("species");
  const [kPick, setK] = useState(1);
  const b: PublishedBenchmark | undefined = list[Math.min(pick, list.length - 1)];
  return (
    <div className="space-y-4" data-testid="chart-heldout">
      <h3 className="font-semibold text-lg text-[#4a3728]">Held-out records</h3>
      <p className="text-sm text-muted-foreground max-w-3xl">
        DNA-verified records that no model here saw, in training or as a reference. Ranges in
        brackets are 95% intervals. Results before the dataset freeze are exploratory: they will be
        re-run on dataset release v1.
      </p>
      {!b ? (
        <p className="text-sm text-muted-foreground">
          {q.isLoading ? "Loading…" : "Held-out results appear here once they are published."}
        </p>
      ) : (
        <HeldOutBody b={b} list={list} pick={pick} setPick={setPick} rank={rank} setRank={setRank}
                     kPick={kPick} setK={setK} />
      )}
    </div>
  );
}

function HeldOutBody({ b, list, pick, setPick, rank, setRank, kPick, setK }: {
  b: PublishedBenchmark; list: PublishedBenchmark[]; pick: number; setPick: (i: number) => void;
  rank: Rank; setRank: (r: Rank) => void; kPick: number; setK: (k: number) => void;
}) {
  const ks = heldoutKs(b);
  const k = ks.includes(kPick) ? kPick : 1;
  const rows = heldoutBars(b, rank, k);
  const depth = heldoutDepth(b);
  const anyOutside = rows.some((r) => r.external);
  return (
    <>
      <label className="flex flex-wrap items-center gap-2 text-sm">
        <span className="text-muted-foreground">Benchmark:</span>
        <select className="rounded-md border border-[#A87146]/30 bg-white px-2 py-1 max-w-full min-w-0" value={pick}
                onChange={(e) => setPick(Number(e.target.value))}>
          {list.map((x, i) => (
            <option key={`${x.benchmark}/${x.split}`} value={i}>
              {x.benchmark}, {x.split} split, {num(x.scored_records)} records
              {x.sealed ? " (sealed test)" : " (development benchmark"}
              {x.sealed ? "" : x.reproducibility?.startsWith("reproduced-on-")
                ? `, ${x.reproducibility.replace("reproduced-on-", "release ")})` : ", exploratory, pre-freeze)"}
            </option>
          ))}
        </select>
      </label>
      <div className="rounded-lg border border-[#A87146]/20 bg-white p-4">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
          <h4 className="font-semibold text-[#4a3728]">
            {RANK_LABEL[rank]} right in the top {k === 1 ? "answer" : k}
          </h4>
          <div className="flex flex-wrap gap-2">
            <Toggle options={["species", "genus", "family"] as Rank[]} value={rank} onChange={setRank}
                    label={(r) => RANK_LABEL[r]} />
            <Toggle options={ks} value={k} onChange={setK} label={(v) => `Top ${v}`} />
          </div>
        </div>
        <div className="flex gap-4 text-xs text-muted-foreground mb-2">
          <span className="inline-flex items-center gap-1.5"><Swatch c={OURS} /> MycoMap Vision</span>
          {anyOutside && <span className="inline-flex items-center gap-1.5"><Swatch c={OUTSIDE} /> Outside model</span>}
        </div>
        <div className="space-y-1.5">
          {rows.map((r) => {
            const label = heldoutLabel(r.backbone, r.method);
            return (
              <div key={r.key} className="grid grid-cols-[minmax(0,16rem)_1fr] items-center gap-3">
                <div className="text-sm leading-tight min-w-0">
                  <span className="font-medium text-[#4a3728]">{label.name}</span>{" "}
                  <span className="text-muted-foreground">· {label.how}</span>
                </div>
                <div className="flex items-center gap-2 h-6"
                     title={r.n != null ? `${pct(r.value, 1)} of ${num(r.n)} records` : undefined}>
                  <div className="h-[18px] rounded-r-[4px]"
                       style={{ width: `${r.value == null ? 0 : Math.max(0.5, 100 * r.value)}%`,
                                background: r.external ? OUTSIDE : OURS }} />
                  <span className="text-sm tabular-nums text-[#0b0b0b] whitespace-nowrap">
                    {r.value == null ? "not measured" : pct(r.value, 1)}
                    {r.ci && <span className="text-muted-foreground"> ({pct(r.ci[0], 1)}–{pct(r.ci[1], 1)})</span>}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
      {depth.length > 0 && (
        <div className="rounded-lg border border-[#A87146]/20 bg-white p-4 overflow-x-auto">
          <h4 className="font-semibold text-[#4a3728] mb-2">
            Species right first time, by how many DNA-verified records the species has
          </h4>
          <table className="text-sm">
            <thead>
              <tr className="text-xs text-muted-foreground">
                <th className="text-left font-medium pr-4 py-1">Approach</th>
                {STANDARD_BANDS.map((band, i) => (
                  <th key={band} className="font-medium px-1 text-center min-w-[4.5rem]">
                    {band} records<br />
                    <span className="font-normal">
                      {depth[0].cells[i] ? `${num(depth[0].cells[i]!.n)} finds` : ""}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {depth.map((row) => {
                const label = heldoutLabel(row.backbone, row.method);
                return (
                  <tr key={row.key}>
                    <td className="pr-4 py-0.5">
                      <span className="font-medium text-[#4a3728]">{label.name}</span>{" "}
                      <span className="text-muted-foreground">· {label.how}</span>
                    </td>
                    {row.cells.map((c, i) => {
                      const h = heat(c?.rate ?? null);
                      return (
                        <td key={i} className="px-0.5 py-0.5">
                          <div className="rounded px-2 py-1 text-center tabular-nums"
                               style={{ background: h.bg, color: h.ink }}
                               title={c ? `${pct(c.rate, 1)} of ${num(c.n)} finds` : "no finds"}>
                            {c ? pct(c.rate, 0) : "–"}
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
