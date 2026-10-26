import { Link } from "wouter";
import { Download, Search, Target } from "lucide-react";
import type { HowGood, Me, RunReport, ScoreRun } from "@/lib/api";
import { CITATIONS } from "@/lib/research";
import { shortHash, utcDay, V1 } from "@/lib/datasetReleases";
import { MODEL_CREDIT_LINE, V1_OPEN_DATA, V1_OPEN_PHOTOS } from "@/lib/publicView";
import { standardOf } from "@/lib/compareView";
import { modelParts, nf, placeVsPhotos, topkValues } from "@/lib/researchView";
import { CitationCard, HashChip, KV, ResearchHero, ResearchShell, Section, SourceLine, Strip,
         TopKChart } from "@/components/ResearchKit";
import { useHowGood, useMe, useRunReports, useScoreboard } from "@/components/researchQueries";

// Research > For researchers (Steve, 2026-10-25: the approved redesign): copy-ready citations
// (text or BibTeX), the model names as the live site uses them, and how to test a model on
// the same records. MycoMap Vision v1's number is read from the scoreboard (the site's model, photos
// only), never typed in.

export interface ForResearchersData { me?: Me | null; howGood?: HowGood; runs?: ScoreRun[]; reports: Map<number, RunReport> }

export function ForResearchersPage() {
  const me = useMe();
  const howGood = useHowGood();
  const board = useScoreboard();
  const run = howGood.data?.run;
  const pair = run && board.data ? placeVsPhotos(board.data.runs, [run.backbone], run.method) : null;
  const { reports } = useRunReports(pair ? [pair.photos.id] : run ? [run.id] : []);
  return <ForResearchers me={me.data} howGood={howGood.data} runs={board.data?.runs} reports={reports} />;
}

/** The model names a number can belong to (Steve, 2026-10-23: as live). */
export const MODEL_NAMES: { name: string; note: string; what: string; role: string }[] = [
  { name: "MycoMap Vision v1", note: "public + non-public training data",
    what: "The base model fine-tuned at 336 px on release v1's training split, all 624,785 photos", role: "The site's model" },
  { name: "MycoMap Vision v1-open", note: "open data only",
    what: `The same recipe on the ${V1_OPEN_PHOTOS.toLocaleString("en-US")} ${V1_OPEN_DATA}`, role: "The paper's headline model" },
  { name: "MycoMap Vision v0", note: "224 px, development", what: "The same recipe at 224 px", role: "Earlier tests" },
  { name: "BioCLIP 2 (base model, not fine-tuned)", note: "224 or 336 px",
    what: "The base model's own image vectors, unchanged", role: "Comparison only" },
];

export function ForResearchers(d: ForResearchersData) {
  const run = d.howGood?.run;
  const pair = run && d.runs ? placeVsPhotos(d.runs, [run.backbone], run.method) : null;
  const beat = pair ?? (run ? { photos: d.runs?.find((r) => r.id === run.id) } : null);
  const beatRun = beat?.photos;
  const sp = beatRun ? standardOf(d.reports.get(beatRun.id))?.species?.all : undefined;
  const icon = "h-[18px] w-[18px]";
  return (
    <ResearchShell me={d.me}>
      <ResearchHero crumbs={[{ href: "/research", label: "Research" }, { label: "For researchers" }]} title="For researchers"
                    lede="Citing MycoMap Vision, naming its models correctly, and testing your own model against DNA-verified records.">
        <div className="callout green mt-5 max-w-[760px]"><div><b style={{ color: "var(--ink)" }} data-testid="text-model-credit-line">{MODEL_CREDIT_LINE}</b>{" "}
          A paper describing the method and its tests is in preparation; until it is published, please cite the site and the base model.</div></div>
      </ResearchHero>
      <Section kicker="How to cite" title="Copy-ready citations" testId="section-cite">
        <div className="mt-4 grid gap-3">{CITATIONS.map((c) => <CitationCard key={c.key} c={c} />)}</div>
      </Section>
      <Section band kicker="Model names" title="Which model a number belongs to" testId="section-model-names">
        <div className="tcard mt-3.5"><div className="scrollx"><table className="dt">
          <thead><tr><th className="l">Name</th><th className="l">What it is</th><th className="l">Role</th></tr></thead>
          <tbody>{MODEL_NAMES.map((m) => (
            <tr key={m.name}><td className="l"><b>{m.name}</b><small>{m.note}</small></td><td className="l">{m.what}</td><td className="l">{m.role}</td></tr>
          ))}</tbody></table></div></div>
        <p className="small mt-2.5">Outside models keep their own names: iNaturalist's computer vision, the Danish Fungi 2020 and FungiTastic models.</p>
      </Section>
      <Section kicker="Score your model on the same records" title="Same records, same protocol, comparable numbers" testId="section-test-yours">
        <Strip steps={[
          { icon: <Download className={icon} />, title: `1 · Get release ${V1.id}`, text: "Its public variant (Creative Commons photos only) is planned with the paper." },
          { icon: <Search className={icon} />, title: "2 · Identify the validation split", text: "From the training split only, as a new find meets it." },
          { icon: <Target className={icon} />, title: "3 · Score the standard summary", text: "Top 1/3/5/10, three ranks, five depth bands." },
        ]} />
        <div className="two" style={{ marginTop: 16 }}>
          <div className="min-w-0 rounded-[10px] border border-[#A87146]/20 bg-white p-5">
            <h3 className="font-display font-semibold text-lg text-[#4a3728] mb-1">The release to quote</h3>
            <div className="mt-2"><KV labelWidth={120} items={[
              ["Release", `Dataset release ${V1.id} (frozen ${utcDay(V1.created)})`],
              ["Release hash", <HashChip key="h" value={V1.releaseHash} short={`${shortHash(V1.releaseHash)}…`} />],
              ["Validation split", `${nf(V1.records.val)} records first validated after ${utcDay(V1.valCutoff)}`],
            ]} /></div>
            <SourceLine parts={["datasetReleases.ts (V1)"]} />
          </div>
          <div className="min-w-0 rounded-[10px] border border-[#A87146]/20 bg-white p-5" data-testid="same-records-number">
            <h3 className="font-display font-semibold text-lg text-[#4a3728] mb-1">MycoMap Vision v1 on the same records</h3>
            {beatRun && sp ? (
              <>
                <p className="small mb-2">{modelParts(beatRun.backbone).name}, {beatRun.method.endsWith("+place") ? "photos + place and date" : "photos only, scored by specimen: the comparable number for a model that uses photos alone"}.</p>
                <TopKChart variant="mini" rows={[{ key: "beat", name: modelParts(beatRun.backbone).name, n: sp.n, values: topkValues(sp) }]} testId="chart-same-records" />
                <SourceLine parts={[`/api/scoreboard/${beatRun.id}`, `${nf(sp.n)} finds`]} />
              </>
            ) : <p className="small">MycoMap Vision v1's number is read from the scoreboard; it could not be read just now.</p>}
          </div>
        </div>
        <p className="text-sm text-muted-foreground leading-normal mt-2.5">
          If this would help your work, tell us at <a href="mailto:info@mycomap.org" className="text-[#4f7a16]">info@mycomap.org</a>.
          Where the references come from: community scientists' iNaturalist and Mushroom Observer observations, sequenced through MycoMap and validated in MycoMap projects (<Link href="/research/data" className="text-[#4f7a16]">Reference data</Link>).
        </p>
      </Section>
    </ResearchShell>
  );
}
