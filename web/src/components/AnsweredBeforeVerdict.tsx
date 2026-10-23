// Results: "Answered before the DNA verdict" (GET /api/early-answers; early_answers.py).
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { modelLabel, num, pct, type EarlyAnswersGroup } from "@/lib/api";
import { dayRange, DEPTH_BANDS, KS, leftOutLines, LEVELS, sealedCount } from "@/lib/earlyAnswers";

export function AnsweredBeforeVerdictCard({ groups, sealedAfter, loading = false }: {
  groups: EarlyAnswersGroup[]; sealedAfter?: string | null; loading?: boolean;
}) {
  const sealed = sealedCount(groups);
  return (
    <section data-testid="early-answers">
      <h2 className="font-display font-semibold text-2xl text-[#4a3728] mb-1">Answered before the DNA verdict</h2>
      <p className="text-sm text-muted-foreground mb-3 max-w-3xl">
        When mycomap.org auto-validates a project, or someone identifies an observation from its
        link, Vision answers before the record has its DNA verdict, and the answer is kept as it
        was. Once the record turns green, its first answer is scored against the DNA-validated
        name: records no model had trained on. Only answers given before the record was first
        validated count.
      </p>
      {sealed > 0 && (
        <p className="text-xs text-[#7a5a3a] mb-3 max-w-3xl" data-testid="early-answers-sealed">
          Records validated since {sealedAfter?.slice(0, 10)} form the paper's sealed test:
          their {sealed.toLocaleString("en-US")} answers are counted but not scored here until that
          test is opened.
        </p>
      )}
      {loading ? (
        <p className="text-sm text-muted-foreground">Loading…</p>
      ) : !groups.length ? (
        <p className="text-sm text-muted-foreground">None yet.</p>
      ) : (
        <div className="space-y-4">
          {groups.map((g) => <Group key={`${g.backbone}/${g.method}/${g.context}`} g={g} />)}
        </div>
      )}
    </section>
  );
}

function Group({ g }: { g: EarlyAnswersGroup }) {
  const label = modelLabel(g.backbone, g.method);
  const answered = dayRange(g.answered);
  const validated = dayRange(g.validated);
  return (
    <Card>
      <CardHeader className="bg-[#f8f5f0] border-b border-[#A87146]/10 py-3">
        <CardTitle className="text-base text-[#4a3728]">
          <span title={label.tip}>{label.name}</span>{" "}
          <span className="text-muted-foreground font-normal">· {label.how} · {g.context}</span>
        </CardTitle>
        <p className="text-xs text-muted-foreground" data-testid="early-answers-meta">
          {num(g.scored)} records scored
          {answered && <>, answered {answered}</>}
          {validated && <>, validated {validated}</>}
          ; {num(g.awaiting_verdict)} awaiting their verdict
        </p>
      </CardHeader>
      <CardContent className="p-0 overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-xs uppercase tracking-wider text-muted-foreground">
            <tr className="border-b">
              <th className="text-left font-medium px-4 py-2" />
              {KS.map((k) => <th key={k} className="text-right font-medium px-3 py-2">Top {k}</th>)}
              <th className="text-right font-medium px-4 py-2">n</th>
            </tr>
          </thead>
          <tbody>
            {LEVELS.map((level) => {
              const row = g.top_k[level];
              if (!row) return null;
              return (
                <tr key={level} className="border-b">
                  <td className="px-4 py-2">{level}</td>
                  {KS.map((k) => (
                    <td key={k} className="px-3 py-2 text-right tabular-nums">
                      {pct(row[`top${k}` as "top1"], 1)}
                    </td>
                  ))}
                  <td className="px-4 py-2 text-right tabular-nums text-muted-foreground"
                      title={row.n_top10 !== row.n ? `top 10 over the ${row.n_top10} answers kept ten deep` : undefined}>
                    {num(row.n)}{row.n_top10 !== row.n && <> ({num(row.n_top10)})</>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <table className="w-full text-sm mt-2">
          <caption className="text-left text-xs text-muted-foreground px-4 py-1">
            Species, by the true species' DNA-verified reference records
          </caption>
          <thead className="text-xs uppercase tracking-wider text-muted-foreground">
            <tr className="border-b">
              <th className="text-left font-medium px-4 py-2">Records</th>
              <th className="text-right font-medium px-3 py-2">Top 1</th>
              <th className="text-right font-medium px-3 py-2">Top 5</th>
              <th className="text-right font-medium px-4 py-2">n</th>
            </tr>
          </thead>
          <tbody>
            {DEPTH_BANDS.map((band) => {
              const row = g.species_by_true_species_reference_records[band];
              return (
                <tr key={band} className="border-b">
                  <td className="px-4 py-2">{band}</td>
                  <td className="px-3 py-2 text-right tabular-nums">{pct(row?.top1, 1)}</td>
                  <td className="px-3 py-2 text-right tabular-nums">{pct(row?.top5, 1)}</td>
                  <td className="px-4 py-2 text-right tabular-nums text-muted-foreground">{num(row?.n ?? 0)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="text-xs text-muted-foreground px-4 py-2" data-testid="early-answers-left-out">
          Left out: {leftOutLines(g).join("; ")}.
          {" "}iNaturalist's computer vision: {g.inat_cv.status === "not run" ? "not run on these records" : g.inat_cv.status}.
        </p>
      </CardContent>
    </Card>
  );
}
