// The v1 / v2 dropdown and a newer model's scores since its training cut (lib/modelScores.ts),
// shared by How it works and Get involved. A newer model is measured as v1's numbers are (each
// find identified from the records verified before it, with its photos and its public place
// and date), and shown with the same breakdowns. Every daily value is written with its n: a
// day holds a hundred or two records, so a day's figure swings.

import { useId } from "react";
import { Bars, C, Cap, Card, H3, Ink, Small, Tile } from "@/components/Explainer";
import { num, pct, type ModelScores, type RankScore } from "@/lib/api";
import { depthLabel } from "@/lib/howItWorks";
import { longDate } from "@/lib/publicView";
import {
  CHART_DAYS, choiceLabel, depthRows, rangeText, scoredDays, topKLadder, withN, type ModelOption,
} from "@/lib/modelScores";

/** The dropdown. With one choice it still shows which model the numbers are for. */
export function ModelPicker({ options, value, onChange, testId = "model-picker" }: {
  options: ModelOption[]; value: string; onChange: (id: string) => void; testId?: string;
}) {
  const id = useId();
  return (
    <div className="mb-5 flex flex-wrap items-center gap-2 text-sm" data-testid={testId}>
      <label htmlFor={id} className="font-medium text-[#4a3728]">Numbers for</label>
      <select id={id} value={value} onChange={(e) => onChange(e.target.value)}
              className="rounded-md border border-[#A87146]/25 bg-white px-2.5 py-1.5 text-sm text-[#4a3728]">
        {options.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
      </select>
    </div>
  );
}

/** What the numbers were measured on, said as v1's How it works says it. */
function TestLine({ m }: { m: ModelScores }) {
  return (
    <Small className="mb-3">
      {choiceLabel(m.id)} learned from every DNA-verified record up to {longDate(m.trained_through)}.
      Tested on the <Ink>{num(m.records)}</Ink> DNA-verified finds validated since, none of them used
      in training, each identified from the records verified before it, with its photos and its
      public place and date. Each find is checked once, on its first answer.
    </Small>
  );
}

function RankTile({ s, rank, compact }: { s: RankScore; rank: string; compact?: boolean }) {
  const range = rangeText(s);
  return (
    <Tile value={pct(s.top1, 1)} testId={`total-${rank.toLowerCase()}`}
          label={compact ? `${rank}, right first time` : `${rank} right first time, ${num(s.n)} finds`}
          note={range ? `Likely ${range}` : undefined} />
  );
}

/** A newer model's figures. Full (How it works): species, genus and family right first time
 *  with their ranges, the records and new photos since training, the last 7 days, the top-k
 *  ladder, species by reference depth, and the last 30 days' chart with every day in a table.
 *  `compact` (Get involved): the totals, the depth bands and the chart. */
export function SinceTraining({ m, compact }: { m: ModelScores; compact?: boolean }) {
  const cum = m.cumulative;
  const roll = m.rolling_7_days;
  const chart = scoredDays(m, CHART_DAYS);
  return (
    <div data-testid={`scores-${m.id}`}>
      <TestLine m={m} />
      {cum && (
        <>
          <div className="grid grid-cols-3 gap-3 mb-2" data-testid="scores-totals">
            <RankTile s={cum.species} rank="Species" compact={compact} />
            <RankTile s={cum.genus} rank="Genus" compact={compact} />
            <RankTile s={cum.family} rank="Family" compact={compact} />
          </div>
          <Small className="mb-3">The first answer was right this often.</Small>
        </>
      )}
      {!compact && (
        <div className="grid grid-cols-2 gap-3 mb-4">
          <Tile value={num(m.records)} label={`finds verified since ${longDate(m.trained_through)}, all new to it`} testId="total-records" />
          <Tile value={num(m.new_photos)} label="new photos on those finds, none of them used in training" testId="total-photos" />
        </div>
      )}
      {compact && (
        <p className="text-sm mb-4" data-testid="scores-counts">
          <Ink>{num(m.records)}</Ink> finds verified since training, with <Ink>{num(m.new_photos)}</Ink> new photos.
        </p>
      )}
      {roll && (
        <p className="text-sm mb-4" data-testid="scores-rolling">
          Last 7 days (to {longDate(roll.through)}): species <Ink>{withN(roll.species, 1)}</Ink>,
          genus <Ink>{withN(roll.genus, 1)}</Ink>, family <Ink>{withN(roll.family, 1)}</Ink>.
        </p>
      )}
      {compact ? <DepthBands m={m} /> : (
        <div className="grid gap-8 md:grid-cols-2 items-start my-6">
          <Card>
            <H3>The right name, within the first few</H3>
            <TopK m={m} />
            <Cap>
              Species is the hard rank: many fungi are told apart only by DNA, and many names
              here are new, provisional ones. Genus is right far more often.
            </Cap>
          </Card>
          <Card>
            <H3>Every sequenced record makes it better</H3>
            <DepthBands m={m} />
          </Card>
        </div>
      )}
      {chart.length > 0 && (
        <div className="mt-4">
          <Small className="mb-3">
            Species right first time, day by day, last {CHART_DAYS} days (each day with how many finds it had)
          </Small>
          <Bars labelWidth={96} testId="chart-daily" rows={chart.map((d) => ({
            label: longDate(d.day).replace(/ \d{4}$/, ""),
            sub: `n = ${num(d.species.n)}${d.complete ? "" : ", so far"}`,
            frac: d.species.top1 ?? 0, text: pct(d.species.top1), color: C.s1,
            title: `${longDate(d.day)}: species ${withN(d.species, 1)}, genus ${withN(d.genus, 1)}, family ${withN(d.family, 1)}`,
          }))} />
          <Small className="mt-2">
            A day has only a hundred or two finds, so one day's figure moves a lot; the 7-day and
            overall figures are steadier.
          </Small>
        </div>
      )}
      {!compact && chart.length > 0 && <DailyTable m={m} />}
    </div>
  );
}

function TopK({ m }: { m: ModelScores }) {
  const rows = topKLadder(m);
  if (!rows) return <Small>Not measured yet.</Small>;
  const ks = rows.map((r) => r.k);
  const listed = ks.length > 1 ? `${ks.slice(0, -1).join(", ")} or ${ks[ks.length - 1]}` : `${ks[0]}`;
  return (
    <div data-testid="v2-top-k">
      <Small className="mb-3">How often the right name is in the top {listed} answers</Small>
      <div className="flex flex-wrap gap-4 text-[13px] mb-3">
        <span className="inline-flex items-center gap-1.5"><i className="h-3 w-3 rounded-[3px]" style={{ background: C.s1 }} />Species</span>
        <span className="inline-flex items-center gap-1.5"><i className="h-3 w-3 rounded-[3px]" style={{ background: C.s2 }} />Genus</span>
      </div>
      <table className="w-full border-collapse text-sm">
        <tbody>
          {rows.map((r) => (
            <tr key={r.k} className="border-t border-[#ece6de]">
              <th scope="row" className="w-[74px] py-[7px] text-left font-semibold text-[#4a3728] whitespace-nowrap">Top {r.k}</th>
              <td className="py-[7px]">
                <div className="flex flex-col gap-[3px]">
                  {([["Species", r.species, C.s1], ["Genus", r.genus, C.s2]] as const).map(([rank, v, color]) => v != null && (
                    <div key={rank} className="grid grid-cols-[1fr_52px] items-center gap-2"
                         title={`${rank}, right name in the top ${r.k}: ${pct(v, 1)}`}>
                      <span className="relative h-3">
                        <b className="absolute left-0 top-0 h-3 rounded-r" style={{ width: `${v * 100}%`, background: color }} />
                      </span>
                      <span className="text-right text-[13px] text-[#4a3728] tabular-nums">{pct(v, 1)}</span>
                    </div>
                  ))}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <Small className="mt-2">Over {num(m.top_k?.species?.n)} finds at species and {num(m.top_k?.genus?.n)} at genus.</Small>
    </div>
  );
}

function DepthBands({ m }: { m: ModelScores }) {
  const rows = depthRows(m);
  if (!rows) return <Small>Not measured yet.</Small>;
  const least = m.min_band_records ?? 30;
  return (
    <div data-testid="v2-by-depth">
      <Small className="mb-3">
        Species right first time, by how many DNA-verified records the species had when it was trained
      </Small>
      <Bars labelWidth={96} testId="chart-v2-by-depth" rows={rows.map((b, i) => ({
        label: depthLabel(b.band), sub: `n = ${num(b.n)}`,
        frac: b.top1 ?? 0, text: b.tooFew ? "too few yet" : pct(b.top1, 1),
        color: b.tooFew ? C.rest : C.ramp[Math.min(i, C.ramp.length - 1)],
        title: b.tooFew
          ? `${depthLabel(b.band)}: only ${num(b.n)} finds so far, too few to give a figure`
          : `${depthLabel(b.band)}: ${pct(b.top1, 1)} right first time (n = ${num(b.n)})`,
      }))} />
      <Cap>
        A species with no DNA-verified reference can't be named at species. A group with fewer than{" "}
        {least} finds says "too few yet" rather than a number that would swing.
      </Cap>
    </div>
  );
}

const TD = "px-2.5 py-1 border-b border-[#ece6de]";

function DailyTable({ m }: { m: ModelScores }) {
  const days = scoredDays(m).reverse();
  return (
    <details className="mt-3.5 text-[13px]">
      <summary className="cursor-pointer font-medium text-[#4a3728]">Every day as a table</summary>
      <div className="overflow-x-auto">
        <table className="mt-2 border-collapse tabular-nums" data-testid="table-daily">
          <thead>
            <tr>
              <th className={`${TD} text-left`}>Day verified</th>
              <th className={`${TD} text-right`}>Finds</th>
              <th className={`${TD} text-right`}>Species right</th>
              <th className={`${TD} text-right`}>Genus right</th>
              <th className={`${TD} text-right`}>Family right</th>
              <th className={`${TD} text-right`}>Species, last 7 days</th>
              <th className={`${TD} text-right`}>New photos</th>
            </tr>
          </thead>
          <tbody>
            {days.map((d) => (
              <tr key={d.day}>
                <td className={`${TD} text-left`}>{longDate(d.day)}{d.complete ? "" : " (so far)"}</td>
                <td className={`${TD} text-right`}>{num(d.records)}</td>
                <td className={`${TD} text-right`}>{withN(d.species)}</td>
                <td className={`${TD} text-right`}>{withN(d.genus)}</td>
                <td className={`${TD} text-right`}>{withN(d.family)}</td>
                <td className={`${TD} text-right`}>{withN(d.rolling_7_days.species)}</td>
                <td className={`${TD} text-right`}>{num(d.photos)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

/** Get involved's card for a newer model, in place of v1's depth chart. */
export function SinceTrainingCard({ m }: { m: ModelScores }) {
  return (
    <Card className="!rounded-xl">
      <H3>More records, more right answers</H3>
      <SinceTraining m={m} compact />
    </Card>
  );
}
