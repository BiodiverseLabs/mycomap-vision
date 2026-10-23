import { num, pct } from "@/lib/api";
import { utcDay, V1 } from "@/lib/datasetReleases";
import { MODEL_CREDIT_LINE } from "@/lib/publicView";
import { pointsDiff, twoModelLabel, twoModelTip, TWO_MODELS, type TopK } from "@/lib/twoModels";
import { PhotoPermissionLine } from "@/components/PhotoPermissionLine";

// "The two models" on the Results page (Steve, 2026-10-22): what the site serves, what the
// paper headlines, and how far apart they are on the same records (lib/twoModels.ts).

const BANDS = ["0", "1-4", "5-19", "20-99", "100+"];

export function TwoModels() {
  const [site, open] = TWO_MODELS.models;
  const p = TWO_MODELS.paired;
  return (
    <section data-testid="section-two-models" className="space-y-3">
      <h2 className="font-display font-semibold text-2xl text-[#4a3728]">The two models</h2>
      <div className="text-sm text-[#5c4a3a] max-w-3xl space-y-2 leading-relaxed">
        <p data-testid="text-model-credit-line">{MODEL_CREDIT_LINE}</p>
        <p>
          Both read photos at 336 px and were fine-tuned the same way on dataset release{" "}
          {V1.id}'s training split. They differ only in the photos they learned from:
        </p>
        <ul className="list-disc pl-5 space-y-1">
          <li>
            <span className="font-medium">{twoModelLabel(site)}</span>: the model this site serves.
            It learned from every training photo, public and non-public.
          </li>
          <li>
            <span className="font-medium">{twoModelLabel(open)}</span>: the paper's headline model.
            It learned only from Creative Commons photos (527,842 of the 624,785), so anyone can
            rebuild its numbers.
          </li>
        </ul>
        <p><PhotoPermissionLine kind="control" /></p>
      </div>
      <div className="overflow-x-auto rounded-lg border border-[#A87146]/20 bg-white">
        <table className="w-full text-sm">
          <thead className="text-xs uppercase tracking-wider text-muted-foreground bg-[#f8f5f0]">
            <tr>
              <th className="text-left font-medium px-3 py-2">Model</th>
              <th className="text-left font-medium px-3 py-2">Rank</th>
              {[1, 3, 5, 10].map((k) => <th key={k} className="text-right font-medium px-3 py-2">Top {k}</th>)}
              <th className="text-right font-medium px-3 py-2">Finds</th>
            </tr>
          </thead>
          <tbody>
            {TWO_MODELS.models.flatMap((m) => (["species", "genus"] as const).map((rank, i) => (
              <tr key={`${m.backbone}/${rank}`} className={i === 0 ? "border-t" : ""}>
                <td className="px-3 py-1.5 font-medium text-[#4a3728]" title={twoModelTip(m)}>
                  {i === 0 ? <>{twoModelLabel(m)}<br /><span className="font-normal text-xs text-muted-foreground">{m.role}</span></> : null}
                </td>
                <td className="px-3 py-1.5 capitalize">{rank}</td>
                <Cells t={m[rank]} />
              </tr>
            )))}
          </tbody>
        </table>
      </div>
      <p className="text-sm text-[#5c4a3a] max-w-3xl">
        On the same {num(p.n)} finds from {num(p.observers)} observers, species top 1 of the open-data
        model minus the site's model: {pointsDiff(p.diff, p.ci95)} points (95% interval in brackets,
        clustered by observer).
      </p>
      <div className="overflow-x-auto rounded-lg border border-[#A87146]/20 bg-white">
        <table className="text-sm">
          <thead className="text-xs text-muted-foreground">
            <tr>
              <th className="text-left font-medium px-3 py-2">Species, top 1 / top 5, by the species' DNA-verified records</th>
              {BANDS.map((b) => (
                <th key={b} className="text-center font-medium px-2 py-2">
                  {b} records<br /><span className="font-normal">{num(site.depth[b].n)} finds</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {TWO_MODELS.models.map((m) => (
              <tr key={m.backbone} className="border-t">
                <td className="px-3 py-1.5" title={twoModelTip(m)}>{twoModelLabel(m)}</td>
                {BANDS.map((b) => (
                  <td key={b} className="px-2 py-1.5 text-center tabular-nums">
                    {pct(m.depth[b].top1, 1)} / {pct(m.depth[b].top5, 1)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-muted-foreground max-w-3xl">
        Dataset release {V1.id}, validation split: {num(TWO_MODELS.testRecords)} records first validated
        after {utcDay(V1.valCutoff)}, identified from the {num(TWO_MODELS.referenceRecords)} records
        validated before them (the erratum's records left out). Photos only, no place or date: both
        models scored by specimen (record-grouped scoring, see Approaches compared), against the same
        reference photos. Species strict; genus strict.
      </p>
    </section>
  );
}

function Cells({ t }: { t: TopK }) {
  return (
    <>
      {[t.top1, t.top3, t.top5, t.top10].map((v, i) => (
        <td key={i} className={`px-3 py-1.5 text-right tabular-nums ${i === 0 ? "font-semibold" : ""}`}>{pct(v, 1)}</td>
      ))}
      <td className="px-3 py-1.5 text-right tabular-nums text-muted-foreground">{num(t.n)}</td>
    </>
  );
}
