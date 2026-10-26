import { Link } from "wouter";
import { BookOpen, Calendar, Check, Copy, FileText, Hash, Layers, ListChecks, Search,
         Target, Users } from "lucide-react";
import type { HowGood, Me, RunReport, ScoreRun } from "@/lib/api";
import { ERRATUM_REASON, utcMinute, V1, type ErratumReason } from "@/lib/datasetReleases";
import { CITATIONS, SEALED_TEST } from "@/lib/research";
import { MODEL_CREDIT_LINE } from "@/lib/publicView";
import { publicMethodLabel } from "@/lib/publicNames";
import { modelParts, nf, pc, shortDay } from "@/lib/researchView";
import { ChartCard, Pipeline, ResearchHero, ResearchShell, Section, SourceLine, Tile, Tiles,
         ValueBars } from "@/components/ResearchKit";
import { useHowGood, useMe, useRunReports, useScoreboard } from "@/components/researchQueries";

// Research > Evaluation protocols (Steve, 2026-10-25: the approved redesign): the six-step
// pipeline, the split on a time line, the standard summary, a number taken apart, the rules
// against leakage and the erratum in use (r4, Steve's answer 1; r3 kept as history).

export interface ProtocolsData { me?: Me | null; howGood?: HowGood; runs?: ScoreRun[]; reports: Map<number, RunReport> }

export function ProtocolsPage() {
  const me = useMe();
  const howGood = useHowGood();
  const board = useScoreboard();
  const { reports } = useRunReports(howGood.data?.run ? [howGood.data.run.id] : []);
  return <Protocols me={me.data} howGood={howGood.data} runs={board.data?.runs} reports={reports} />;
}

/** What is reported, every time: the nine parts of the standard summary. */
export const STANDARD_PARTS: [string, string][] = [
  ["Top 1, 3, 5 and 10", "The right name is the first answer, or among the first 3, 5 or 10. At species, genus and family; all photos of a find together, and the first photo alone."],
  ["Photos only, and with place and date", "Reported apart, so what the find's public place and date add is visible."],
  ["Species strict", "The answer names exactly the DNA-verified species. A temporary code counts like a published name."],
  ["Species sensu lato", "Also right when the genus was split recently (Cortinarius s.l., Inocybe s.l.) or only the gender ending differs. Beside strict, never instead."],
  ["Species complex (beta)", "Also right within the same species complex. Shown for context while the rule is tested."],
  ["By reference depth", "Species results by the true species' DNA-verified records: 0, 1-4, 5-19, 20-99, 100+."],
  ["Outside models, same records", "iNaturalist's computer vision and published models are scored on exactly the records MycoMap Vision is, or not shown."],
  ["Intervals and paired tests", "95% intervals (Wilson; observer-clustered bootstrap where shown) and McNemar tests when two approaches answer the same records."],
  ["Calibration", "Whether a stated 80% is right about 80% of the time, measured on records not used to set it."],
];

export function Protocols(d: ProtocolsData) {
  const icon = "h-5 w-5";
  return (
    <ResearchShell me={d.me}>
      <ResearchHero crumbs={[{ href: "/research", label: "Research" }, { label: "Evaluation protocols" }]} title="Evaluation protocols"
                    lede="How every MycoMap Vision result is scored, so results from different tests, models and months can be compared.">
        <Tiles>
          <Tile highlight value="4" label="top-k levels: 1, 3, 5 and 10" />
          <Tile value="3" label="ranks: species, genus, family" />
          <Tile value="5" label="reference-depth bands, 0 to 100+" />
          <Tile value="1" label="sealed test, scored once after the freeze" />
        </Tiles>
      </ResearchHero>
      <Section kicker="The pipeline" title="From a DNA-verified record to a published number" testId="section-pipeline"
               lead="Every result on this site goes through the same six steps. Nothing in steps 3 to 6 is ever tuned on the sealed test.">
        <Pipeline label={(i) => `${i + 1} · ${["Answer key", "Freeze", "Split by date", "Identify", "Score", "Report"][i]}`}
                  icons={[<BookOpen key={0} className={icon} />, <Hash key={1} className={icon} />, <Calendar key={2} className={icon} />,
                          <Search key={3} className={icon} />, <Target key={4} className={icon} />, <FileText key={5} className={icon} />]}
                  steps={[
                    { title: "DNA-validated name", text: "The observation's name on MycoMap, kept current from the DNA-validation review. Guest organisms left out." },
                    { title: "Dataset release", text: `Release ${V1.id}: ${nf(V1.records.included)} records, hashed; an erratum lists faulty records, never edits.` },
                    { title: "Train · val · sealed", text: `Validated before ${shortDay(V1.valCutoff)} trains; after, validates; after ${shortDay(V1.labelSnapshot)}, sealed.` },
                    { title: "From earlier records only", text: "A test record is never a reference or a training record." },
                    { title: "One summary", text: "Top 1/3/5/10 × species, genus, family × depth band." },
                    { title: "With its test", text: "Records and n, model and size, intervals, calibration." },
                  ]} />
        <ChartCard className="mt-3.5" testId="split-timeline">
          <h3 className="font-display font-semibold text-[#4a3728]">The split, on a time line</h3>
          <div className="tl mt-3">
            <div style={{ flex: V1.records.train, background: "#507a1d" }} data-tip={`Training split|${nf(V1.records.train)} records`}><b>{nf(V1.records.train)}</b>training split</div>
            <div style={{ flex: 30000, background: "#8B4A1F" }} data-tip={`Validation split|${nf(V1.records.val)} records`}><b>{nf(V1.records.val)}</b>validation</div>
            <div style={{ flex: 14000, background: "#2f241b" }} data-tip={`Sealed test|${nf(SEALED_TEST.floor)}–${nf(SEALED_TEST.target)} records`}><b>{nf(SEALED_TEST.floor)}–{nf(SEALED_TEST.target)}</b>sealed test</div>
          </div>
          <div className="tlx"><span>earlier validations</span><span>{shortDay(V1.valCutoff)}</span><span>{shortDay(V1.labelSnapshot)}, {utcMinute(V1.labelSnapshot).split(", ")[1]}</span><span>closes by {shortDay(SEALED_TEST.closeBy)}</span></div>
          <p className="small mt-2.5">Widths of the validation and sealed segments are not to scale (illustrative), so the labels fit; the counts are exact.</p>
          <SourceLine parts={["datasetReleases.ts (V1)", "research.ts BENCHMARKS[sealed-test]"]} />
        </ChartCard>
      </Section>
      <Section band kicker="What is reported, every time" title="Nine parts of the standard summary" testId="section-standard">
        <div className="checks three3">
          {STANDARD_PARTS.map(([t, x]) => <div key={t} className="chk"><div className="ci"><Check className="h-4 w-4" /></div><b>{t}</b>{x}</div>)}
        </div>
      </Section>
      <Anatomy d={d} />
      <Section band testId="section-leakage">
        <div className="split2">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[.08em] text-[#6b8f2a] mb-1.5">Keeping tests clean</p>
            <h2 className="font-display font-semibold text-[25px] sm:text-[30px] leading-tight text-[#4a3728] mb-2">Four rules against leakage</h2>
            <div className="ranks mt-3.5" style={{ gap: 10 }}>
              <Rule icon={<Layers className={icon} />} title="Never a reference">A test record is never a reference or training record.</Rule>
              <Rule icon={<ListChecks className={icon} />} title="Listed before any model sees them">Held-out sets are fixed first; the sealed test is scored once.</Rule>
              <Rule icon={<Users className={icon} />} title="Same observer, same day">Results broken down by whether the same observer recorded a reference on the same day.</Rule>
              <Rule icon={<Copy className={icon} />} title="Twins and duplicate photos">A find posted to both iNaturalist and Mushroom Observer counts once{twins(d) ? `; ${nf(twins(d))} twins are paired` : ""}.</Rule>
            </div>
          </div>
          <Erratum />
        </div>
      </Section>
      <Section testId="section-built-on">
        <div className="split2">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[.08em] text-[#6b8f2a] mb-1.5">What MycoMap Vision is built on</p>
            <h2 className="font-display font-semibold text-[25px] sm:text-[30px] leading-tight text-[#4a3728] mb-2" data-testid="text-model-credit-line">{MODEL_CREDIT_LINE}</h2>
            <p>The last 4 of the base model's 24 transformer blocks were trained for 2 epochs at 336 px on release {V1.id}'s training split: 155,781 DNA-verified records and 624,785 photos as cut (16,959 species once erratum r4 is applied), with classifiers at species (17,077 classes), genus and family. MycoMap Vision v1 learned from all of those photos; MycoMap Vision v1-open from the 527,842 Creative Commons photos among them. Earlier tests used the same recipe at 224 px: MycoMap Vision v0.</p>
            <p className="small mt-2">"BioCLIP 2 (base model, not fine-tuned)" in the results means the base model's own image vectors, unchanged, used to find the nearest DNA-verified records; not zero-shot naming from text prompts.</p>
            {/* The recipe: the served model's finetunes row on the box (bioclip-2-336-ft-20261018-171420:
                blocks_trained 4 of 24, 2 epochs, 155,781 records, 624,785 photos, 17,077 species); the
                open model's 527,842 photos: research repo paper/costs.csv (E23b). */}
          </div>
          <div className="min-w-0 rounded-[10px] border border-[#A87146]/20 bg-white p-5" data-testid="protocol-references">
            <h3 className="font-display font-semibold text-lg text-[#4a3728] mb-1">References</h3>
            <ul className="list-disc pl-[18px] text-sm leading-relaxed">
              {CITATIONS.filter((c) => c.key !== "site").map((c) => <li key={c.key} className="mb-1.5">{c.text}</li>)}
            </ul>
            <p className="small mt-2.5"><Link href="/research/for-researchers" className="text-[#4f7a16]">Copy citations and BibTeX →</Link></p>
          </div>
        </div>
      </Section>
    </ResearchShell>
  );
}

const twins = (d: ProtocolsData): number | null => {
  const r = d.howGood?.run ? d.reports.get(d.howGood.run.id) : undefined;
  const t = (r as unknown as { twins?: { records?: number }; answer_taxonomy?: { twins?: { records?: number } } } | undefined);
  return t?.answer_taxonomy?.twins?.records ?? t?.twins?.records ?? null;
};

function Rule({ icon, title, children }: { icon: React.ReactNode; title: string; children: React.ReactNode }) {
  return <div className="pi"><span className="ic">{icon}</span><div><b>{title}</b>{children}</div></div>;
}

/** The erratum in use, by reason, and the earlier revisions kept as history. */
export function Erratum() {
  const e = V1.erratum;
  if (!e) return null;
  const parts = (Object.entries(e.byReason) as [ErratumReason, number][]).filter(([, n]) => n > 0);
  const label: Record<ErratumReason, string> = { crosspost_twin: "Cross-posted twins",
    mo_twin_as_inat: "Mushroom Observer finds held as iNaturalist records", nonfungal_label: "Labels that name no fungus" };
  return (
    <div data-testid="protocols-erratum">
      <p className="text-xs font-semibold uppercase tracking-[.08em] text-[#6b8f2a] mb-1.5">The erratum in use</p>
      <h2 className="font-display font-semibold text-[25px] sm:text-[30px] leading-tight text-[#4a3728] mb-2">Erratum {e.id}: {nf(e.records)} records left out</h2>
      <ChartCard className="mt-3.5">
        <ValueBars labelWidth={190} rows={parts.map(([why, n]) => ({ label: label[why], frac: n / e.records, text: nf(n), color: "var(--o3)", tip: `${ERRATUM_REASON[why]}|${nf(n)} records` }))} />
        <p className="small mt-2.5">Left out of the references and of scoring; the release itself never changes. Every published release {V1.id} run is scored with erratum {e.id}.
          {V1.erratumHistory.length > 0 && <> Earlier revisions, kept as history: {V1.erratumHistory.map((h) => `${h.id} (${nf(h.records)} records)`).join(", ")}.</>}</p>
        <SourceLine parts={["datasetReleases.ts (V1.erratum)", "run reports' dataset_release.errata (v1-errata-r4.tsv)"]} />
      </ChartCard>
    </div>
  );
}

function Anatomy({ d }: { d: ProtocolsData }) {
  const run = d.howGood?.run;
  const s = d.howGood?.top_k?.species;
  if (!run || s?.top1 == null) return null;
  const scored = d.runs?.find((r) => r.id === run.id);
  const report = d.reports.get(run.id);
  const p = modelParts(run.backbone);
  const key = (report?.all_photos as unknown as { answer_key?: string } | undefined)?.answer_key;
  const errata = (report as unknown as { errata?: { file?: string } } | undefined)?.errata?.file?.match(/-(r\d+)\.tsv$/)?.[1];
  const how = publicMethodLabel(run.method);
  const an = (label: string, b: string, small?: string) => <div className="an"><span>{label}</span><b>{b}</b>{small && <small>{small}</small>}</div>;
  return (
    <Section kicker="No number without its test" title="Anatomy of a number" testId="section-anatomy"
             lead="Every accuracy figure on this site carries these six facts. Here is the headline one, taken apart.">
      <div className="anat">
        <div className="col l">
          {an("Model", p.name, [p.detail, "fine-tuned"].filter(Boolean).join(" · "))}
          {an("Method", how.replace(/^closest specimens \+ species average, /, ""), how.startsWith("closest specimens + species average") ? "closest specimens + species average" : undefined)}
          {an("Rank and k", "species strict · top 1", "the first answer was the DNA name")}
        </div>
        <div className="mid"><div className="n">{pc(s.top1)}</div><div className="c">species right first time</div></div>
        <div className="col r">
          {an("Records", `${nf(s.n)} finds`, `${scored?.dataset_release ? `release ${scored.dataset_release} validation split` : "the newest validations"}${errata ? `, erratum ${errata}` : ""}`)}
          {an("Reference", `${nf(run.n_reference)} earlier records`, `validated on or before ${shortDay(run.cutoff)}`)}
          {an("Run", `#${run.id}${scored ? ` · ${shortDay(scored.created_at)}` : ""}`, key ? `answer key ${key}` : undefined)}
        </div>
      </div>
      <SourceLine parts={[`/api/scoreboard/${run.id}`, "/api/how-good"]} />
    </Section>
  );
}

