import { ArrowRight, Camera, FileCode2, Image as ImageIcon, Lock, ShieldCheck } from "lucide-react";
import type { HowGood, Me, Stats } from "@/lib/api";
import { shortHash, utcDay, V1, type DatasetRelease } from "@/lib/datasetReleases";
import { modelParts, nf, shortDay } from "@/lib/researchView";
import { ChartCard, HashChip, KV, Pipeline, ResearchHero, ResearchShell, Section, SourceLine, Tile,
         Tiles, ValueBars } from "@/components/ResearchKit";
import { PermissionBox } from "@/components/Explainer";
import { useHowGood, useMe, useStats } from "@/components/researchQueries";

// Research > Reference data (Steve, 2026-10-25: the approved redesign): the frozen release
// every result comes from (lib/datasetReleases.ts, from its MANIFEST), the live reference set
// (/api/stats, changes every night) and how anyone regenerates the images a model saw.

const LICENCE: { key: "open" | "nc" | "arr"; label: string; color: string; ink: string }[] = [
  { key: "open", label: "Open (CC0, CC-BY, CC-BY-SA)", color: "#507a1d", ink: "#fff" },
  { key: "nc", label: "Non-commercial (NC or ND)", color: "#98c052", ink: "#2f241b" },
  { key: "arr", label: "All rights reserved", color: "#d9d0c5", ink: "#2f241b" },
];

export interface DataData { me?: Me | null; stats?: Stats; howGood?: HowGood; loading?: boolean; failed?: boolean }

export function DataPage() {
  const me = useMe();
  const stats = useStats();
  const howGood = useHowGood();
  return <Data me={me.data} stats={stats.data} howGood={howGood.data} loading={stats.isLoading} failed={stats.isError} />;
}

export function Data(d: DataData) {
  const s = d.stats;
  return (
    <ResearchShell me={d.me}>
      <ResearchHero crumbs={[{ href: "/research", label: "Research" }, { label: "Reference data" }]} title="Reference data"
                    lede="Only records marked green (DNA-validated) in a MycoMap project are used, with every photo of each record from iNaturalist or Mushroom Observer.">
        {s ? (
          <>
            <Tiles testId="data-tiles">
              <Tile highlight value={nf(s.records_north_america)} label="DNA-verified North American records"
                    note={`${nf(s.records)} in all${s.recent_week ? ` · +${nf(s.recent_week.records)} in the week to ${shortDay(s.recent_week.through)}` : ""}`} />
              <Tile value={nf(s.names)} label="names" note={s.names_provisional != null ? `${nf(s.names_provisional)} provisional (not yet described)` : undefined} />
              <Tile value={nf(s.photos)} label="photos" note={`${(s.photos / Math.max(1, s.inat_ok)).toFixed(1)} per record`} />
              {s.projects != null && <Tile value={nf(s.projects)} label="MycoMap projects" note={s.photographers != null ? `${nf(s.photographers)} people photographed the finds` : undefined} />}
            </Tiles>
            <SourceLine parts={["/api/stats (live reference set, changes every night)"]} />
          </>
        ) : <p className="small mt-4" data-testid="data-no-stats">{d.failed ? "The live figures could not be read just now." : "Reading the live figures…"}</p>}
      </ResearchHero>
      <Release r={V1} />
      {s && <Live s={s} h={d.howGood} />}
      <GetTheImages />
    </ResearchShell>
  );
}

/** A 100% bar; a part's `text` replaces its share (for a part drawn wider than its count). */
function SplitBar({ parts, testId }: { parts: { n: number; color: string; ink: string; label: string; text?: string }[]; testId?: string }) {
  const total = parts.reduce((t, p) => t + p.n, 0) || 1;
  return (
    <div className="flex h-7 gap-[2px]" data-testid={testId} role="img"
         aria-label={parts.map((p) => `${p.label} ${Math.round((100 * p.n) / total)}%`).join(", ")}>
      {parts.map((p, i) => {
        const share = Math.round((100 * p.n) / total);
        return (
          <div key={p.label} data-tip={`${p.label}|${nf(p.n)} (${share}%)`}
               className={`flex items-center justify-center overflow-hidden whitespace-nowrap text-[11.5px] font-semibold ${i === 0 ? "rounded-l" : ""} ${i === parts.length - 1 ? "rounded-r" : ""}`}
               style={{ flex: p.n, background: p.color, color: p.ink }}>{p.text ?? (share >= 8 ? `${share}%` : "")}</div>
        );
      })}
    </div>
  );
}

function Licences({ counts, title }: { counts: Partial<Record<"open" | "nc" | "arr", number>>; title: string }) {
  return (
    <>
      <h3 className="font-display font-semibold text-[#4a3728]">{title}</h3>
      <div className="mt-3"><SplitBar parts={LICENCE.map((l) => ({ n: counts[l.key] ?? 0, color: l.color, ink: l.ink, label: l.label }))} /></div>
      <div className="legend" style={{ marginTop: 8 }}>{LICENCE.map((l) => <span key={l.key}><i style={{ background: l.color }} />{l.label}</span>)}</div>
    </>
  );
}

/** A frozen dataset release: the records and photos research numbers come from. */
export function Release({ r }: { r: DatasetRelease }) {
  const left = r.records.candidates - r.records.included;
  const mo = r.records.bySource.mo;
  return (
    <Section testId={`release-${r.id}`}>
      <div className="flex flex-wrap items-center gap-2.5">
        <p className="text-xs font-semibold uppercase tracking-[.08em] text-[#6b8f2a]">Dataset release {r.id}</p>
        <span className="pill sealed"><Lock className="h-[11px] w-[11px]" /> Frozen</span>
      </div>
      <h2 className="font-display font-semibold text-[25px] sm:text-[30px] leading-tight text-[#4a3728] mt-1.5 mb-2">The snapshot every research result comes from</h2>
      <p className="lead2">Labels as of {utcDay(r.labelSnapshot)}, cut {utcDay(r.created)}. It never changes, so anyone can rebuild a result from it with the code and the one command each result names.</p>
      <div className="funnel" data-testid="release-funnel">
        <div className="fb"><div className="l">Candidate records</div><div className="v">{nf(r.records.candidates)}</div><div className="bar"><b style={{ flex: 1, background: "#d9d0c5" }} /></div></div>
        <div className="ar"><ArrowRight className="h-[22px] w-[22px]" /></div>
        <div className="fb"><div className="l">Included</div><div className="v">{nf(r.records.included)}</div>
          <div className="bar"><b style={{ flex: r.records.included, background: "var(--o3)" }} /><b style={{ flex: left, background: "#efe9e1" }} /></div>
          <div className="l mt-1.5">{nf(left)} left out: no specimen photos, outside North America, a guest organism, no usable name</div></div>
        <div className="ar"><ArrowRight className="h-[22px] w-[22px]" /></div>
        <div className="fsplit">
          <div className="fb" style={{ padding: "10px 14px" }}><div className="l">Training split</div><div className="v" style={{ fontSize: 22 }}>{nf(r.records.train)}</div><div className="l">validated on or before {utcDay(r.valCutoff)}</div></div>
          <div className="fb" style={{ padding: "10px 14px", borderColor: "#c9a88a" }}><div className="l">Validation split</div><div className="v" style={{ fontSize: 22 }}>{nf(r.records.val)}</div><div className="l">first validated after {utcDay(r.valCutoff)}</div></div>
        </div>
      </div>
      <div className="three" style={{ marginTop: 14 }}>
        <ChartCard>
          <h3 className="font-display font-semibold text-[#4a3728]">Where the records come from</h3>
          <div className="mt-3"><SplitBar parts={[
            { n: r.records.bySource.inat, color: "#507a1d", ink: "#fff", label: "iNaturalist",
              text: `iNaturalist ${Math.round((100 * r.records.bySource.inat) / r.records.included)}%` },
            // Widened ×3 so the sliver shows; its share is not written on it, the counts under the bar are exact.
            { n: mo * 3, color: "#8B4A1F", ink: "#fff", label: "Mushroom Observer (shown ×3)", text: "" }]} /></div>
          <p className="small mt-2">{nf(r.records.bySource.inat)} iNaturalist · {nf(mo)} Mushroom Observer ({(100 * mo / r.records.included).toFixed(1)}%; bar widened ×3 to show)</p>
        </ChartCard>
        <ChartCard><Licences counts={r.photos.byLicence} title={`Photo licences, ${nf(r.photos.included)} photos`} /></ChartCard>
        <ChartCard testId="release-fingerprints">
          <h3 className="font-display font-semibold text-[#4a3728]">Fingerprints</h3>
          <div className="mt-2.5"><KV labelWidth={110} items={[
            ["Release hash", <HashChip key="r" value={r.releaseHash} short={`${shortHash(r.releaseHash)}…`} />],
            ["Reference hash", <HashChip key="f" value={r.referenceHash} short={`${shortHash(r.referenceHash)}…`} />],
            ["Cut by code", <span key="c" className="mono">{r.codeCommit.slice(0, 7)}</span>],
            // 16,959: species-level names in the training split after erratum r4 (research repo
            // paper/supplement/ED1-dataset.md), the count the paper uses.
            ["Names", `${nf(r.labels)} distinct labels, North America; 16,959 species in training (erratum r4)`],
            ["Public files", r.doi ? r.doi : "Creative Commons photos only, coming with the paper"],
          ]} /></div>
        </ChartCard>
      </div>
      <SourceLine parts={["web/src/lib/datasetReleases.ts (V1)", r.erratum ? `erratum in use for scoring: v1-errata-${r.erratum.id}.tsv (${nf(r.erratum.records)} records)` : null]} />
    </Section>
  );
}

function Live({ s, h }: { s: Stats; h?: HowGood }) {
  const rows = Object.entries(s.names_by_records);
  const max = Math.max(1, ...rows.map(([, n]) => n));
  const one = s.names_by_records["1"];
  const share = (n: number) => Math.round((100 * n) / Math.max(1, s.names));
  const served = h?.run?.backbone;
  const compared = served ? s.embedded[served] : undefined;
  return (
    <Section band kicker="Live reference set · changes every night" testId="section-live"
             title={one != null ? `Rare is normal: ${share(one)}% of names have a single record` : "The live reference set"}>
      <div className="two" style={{ marginTop: 14 }}>
        <ChartCard className="nbr">
          <h3 className="font-display font-semibold text-[#4a3728]">Names by number of DNA-verified records</h3>
          <div className="mt-3"><ValueBars labelWidth={92} rows={rows.map(([k, n]) => ({
            label: `${k} record${k === "1" ? "" : "s"}`, frac: n / max, strong: k === "1",
            text: `${nf(n)} · ${share(n)}%`, color: k === "1" ? "var(--s1)" : "var(--rest)", tip: `${k} record${k === "1" ? "" : "s"}|${nf(n)} names (${share(n)}%)` }))} /></div>
          <p className="small mt-2.5">Names with a single record still count: the identifier compares with each specimen, so one good match can win.</p>
        </ChartCard>
        <ChartCard>
          <Licences counts={s.photos_by_license as Partial<Record<"open" | "nc" | "arr", number>>} title={`Photo licences, live (${nf(s.photos)} photos)`} />
          <PermissionBox kind="control" className="mt-4" />
          <div className="mt-4 border-t border-[#ece6de] pt-3">
            {compared != null && served && <div className="gl"><span>Photos the identifier compares with ({modelParts(served).name})<small className="block text-[11.5px] text-[var(--muted)]">every DNA-verified record's photos, records validated since the release included; it learned from the training split only</small></span><b>{nf(compared)}</b></div>}
            <div className="gl"><span>Green records no longer on iNaturalist</span><b>{nf(s.inat_missing)}</b></div>
            <div className="gl"><span>Records held out with two different names</span><b>{nf(s.label_conflicts)}</b></div>
          </div>
        </ChartCard>
      </div>
      <SourceLine parts={["/api/stats", served ? "/api/how-good (the served model)" : null]} />
    </Section>
  );
}

/** How anyone gets exactly the images a model saw (Steve, 2026-10-09): originals are kept, each
 *  model input is derived by a published recipe and checked by its hash. CC photos only. */
function GetTheImages() {
  const icon = "h-5 w-5";
  return (
    <Section kicker="Get the images" title="Regenerate exactly what a model saw, and check it" testId="card-get-images"
             lead="Every original photo is kept. The image a model saw is made from its original by a fixed, published recipe, and each release lists a SHA-256 checksum of every original and every image made from it.">
      <Pipeline label={(i) => String(i + 1)}
                icons={[<Camera key={0} className={icon} />, <FileCode2 key={1} className={icon} />, <ImageIcon key={2} className={icon} />, <ShieldCheck key={3} className={icon} />]}
                steps={[
                  { title: "The original", text: "As posted to iNaturalist or Mushroom Observer." },
                  { title: "The recipe", text: "Resize and crop, size, JPEG quality, software and version." },
                  { title: "The model's input", text: "Rebuilt by anyone, byte for byte." },
                  { title: "Checksum matches", text: "SHA-256 against the release's list." },
                ]} />
      <p className="text-sm text-muted-foreground leading-normal mt-2.5">Only openly licensed (Creative Commons) photos can be fetched this way; photos whose owners reserve all rights are never redistributed. The download address and an example appear here with release {V1.id}'s public files.</p>
    </Section>
  );
}
