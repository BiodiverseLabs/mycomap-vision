// The v1 / v2 dropdown and a newer model's scores since its training cut (lib/modelScores.ts),
// shared by How it works and Get involved. Every daily value is written with its n: a day holds
// a hundred or two records, so a day's figure swings.

import { useId } from "react";
import { Bars, C, Card, H3, Ink, Small, Tile } from "@/components/Explainer";
import { num, pct, type ModelScores } from "@/lib/api";
import { longDate } from "@/lib/publicView";
import { choiceLabel, rangeText, scoredDays, withN, type ModelOption } from "@/lib/modelScores";

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

/** A newer model's figures: the cumulative total with its range, the last 7 days, the records
 *  and new photos since training, and the daily chart and table, each day with its n.
 *  `compact`: the chart and the totals only (Get involved). */
export function SinceTraining({ m, compact }: { m: ModelScores; compact?: boolean }) {
  const cum = m.cumulative;
  const roll = m.rolling_7_days;
  const name = choiceLabel(m.id);
  const chart = scoredDays(m, 14);
  return (
    <div data-testid={`scores-${m.id}`}>
      <Small className="mb-3">
        {name} learned from every DNA-verified record up to {longDate(m.trained_through)}. Every
        record verified since then is new to it: each one is identified once, from its own photos,
        and its first answer is checked against the DNA name.
      </Small>
      {cum && (
        <div className={`grid gap-3 mb-4 ${compact ? "grid-cols-2" : "grid-cols-2 lg:grid-cols-4"}`} data-testid="scores-totals">
          <Tile value={pct(cum.species.top1, 1)} label={`species right first time, all ${num(cum.species.n)} records since training`}
                note={rangeText(cum.species) ? `Likely ${rangeText(cum.species)}` : undefined} testId="total-species" />
          <Tile value={pct(cum.genus.top1, 1)} label={`genus right first time, all ${num(cum.genus.n)} records since training`}
                note={rangeText(cum.genus) ? `Likely ${rangeText(cum.genus)}` : undefined} testId="total-genus" />
          {!compact && <Tile value={num(m.records)} label={`records verified since ${longDate(m.trained_through)}, all new to it`} testId="total-records" />}
          {!compact && <Tile value={num(m.new_photos)} label="new photos on those records, none of them used in training" testId="total-photos" />}
        </div>
      )}
      {roll && (
        <p className="text-sm mb-4" data-testid="scores-rolling">
          Last 7 days (to {longDate(roll.through)}): species <Ink>{withN(roll.species, 1)}</Ink>,
          genus <Ink>{withN(roll.genus, 1)}</Ink>.
        </p>
      )}
      {compact && (
        <p className="text-sm mb-4" data-testid="scores-counts">
          <Ink>{num(m.records)}</Ink> records verified since training, with <Ink>{num(m.new_photos)}</Ink> new photos.
        </p>
      )}
      {chart.length > 0 && (
        <>
          <Small className="mb-3">Species right first time, day by day (each day with how many records it had)</Small>
          <Bars labelWidth={96} testId="chart-daily" rows={chart.map((d) => ({
            label: longDate(d.day).replace(/ \d{4}$/, ""),
            sub: `n = ${num(d.species.n)}${d.complete ? "" : ", so far"}`,
            frac: d.species.top1 ?? 0, text: pct(d.species.top1), color: C.s1,
            title: `${longDate(d.day)}: species ${withN(d.species, 1)}, genus ${withN(d.genus, 1)}`,
          }))} />
          <Small className="mt-2">
            A day has only a hundred or two records, so one day's figure moves a lot; the 7-day and
            overall figures are steadier.
          </Small>
        </>
      )}
      {!compact && chart.length > 0 && <DailyTable m={m} />}
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
              <th className={`${TD} text-right`}>Records</th>
              <th className={`${TD} text-right`}>Species right</th>
              <th className={`${TD} text-right`}>Genus right</th>
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
      <H3>Right first time, day by day</H3>
      <SinceTraining m={m} compact />
    </Card>
  );
}
