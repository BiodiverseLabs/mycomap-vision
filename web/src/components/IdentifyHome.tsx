import { useState } from "react";
import { ArrowRight, ArrowUpRight, BarChart3, Check, ChevronRight, CircleHelp, Dna } from "lucide-react";
import { H2, H3, Kicker, Sub, Wrap } from "@/components/Explainer";
import { VIEW_ART } from "@/components/ViewArt";
import { pct, RANKS, type HowGood, type Stats } from "@/lib/api";
import { whyNumbers, type ExampleView } from "@/lib/identifyHome";
import { PHOTO_CHECKLIST, SEQUENCING_URL } from "@/lib/publicView";

// The Identify page's empty state and photo guide (Steve, 2026-10-26: the approved redesign,
// .mockups/vision-identify-home), in How it works' and Get involved's look (Explainer.tsx).

export const GUIDE_ID = "what-to-photograph";

/** The five views as small drawings under the drop zone, each a link to the guide below. */
export function ViewsRow() {
  return (
    <div className="mt-4 flex flex-wrap justify-center gap-1 sm:gap-2.5" data-testid="row-views">
      {PHOTO_CHECKLIST.map((c, i) => (
        <a key={c.what} href={`#${GUIDE_ID}`} onClick={(e) => e.stopPropagation()}
           className="flex w-14 flex-col items-center gap-1 text-[10.5px] leading-tight text-[#5c4a3a] no-underline sm:w-16 sm:text-[11px]">
          <span className="flex h-9 w-9 items-center justify-center rounded-[10px] bg-[#f4efe8] p-1 sm:h-10 sm:w-10 [&_svg]:h-full [&_svg]:w-full"
                aria-hidden>
            {VIEW_ART[i]}
          </span>
          {c.short}
        </a>
      ))}
    </div>
  );
}

/** "Example result": one real run (lib/identifyHome.ts exampleView), its photos, answer and
 *  closest specimens as its own page shows them, with what each part is for. Without an
 *  example run, the list of what an answer holds stands alone. */
export function ExampleResult({ view, loading = false }: { view: ExampleView | null; loading?: boolean }) {
  const italic = (rank: string) => (rank !== "family" ? "sci" : "");
  const label = (rank: string) => rank.charAt(0).toUpperCase() + rank.slice(1);
  const mark = "flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[#4a3728] text-[11px] font-semibold text-white";
  return (
    <div className="mt-10 lg:mt-0" data-testid="example-result">
      <div className="mb-2.5 flex items-baseline justify-between gap-3">
        <p className="text-xs font-semibold uppercase tracking-[.08em] text-[#6b8f2a]">Example result</p>
        {view && (
          <a href={view.href} className="inline-flex items-center gap-1 text-[13px] font-medium text-[#4f7a16] no-underline hover:underline"
             data-testid="link-example-run">
            Open the full result <ArrowRight className="h-3.5 w-3.5" aria-hidden />
          </a>
        )}
      </div>
      {/* While it comes, its place is kept, so nothing below it moves when it arrives. */}
      {!view && loading && (
        <div className="h-[540px] rounded-[14px] sm:h-[600px] border border-[#A87146]/20 bg-white/60 motion-safe:animate-pulse"
             aria-hidden data-testid="example-loading" />
      )}
      {view && (
        <div className="overflow-hidden rounded-[14px] border border-[#A87146]/20 bg-white shadow-[0_10px_30px_-18px_rgba(74,55,40,.35)]">
          <div className="px-4 pb-3 pt-3.5">
            <div className="grid grid-cols-5 gap-1.5" data-testid="example-photos">
              {view.photos.slice(0, 5).map((u, i) => (
                <span key={u} className="relative aspect-square overflow-hidden rounded-md border border-[#A87146]/20 bg-muted">
                  <img src={u} alt={`Photo ${i + 1} of the example find`} decoding="async" className="h-full w-full object-cover" />
                  <span className="absolute bottom-0.5 left-0.5 rounded bg-black/60 px-1 text-[10px] text-white">{i + 1}</span>
                </span>
              ))}
            </div>
            <p className="mt-1.5 text-[11.5px] leading-snug text-muted-foreground" data-testid="example-credit">
              {view.source?.url
                ? <a href={view.source.url} target="_blank" rel="noreferrer" className="underline">{view.source.label}</a>
                : view.source?.label}
              {view.credit && <>{view.source ? " · " : ""}photos {view.credit}</>}
            </p>
          </div>
          <div className="border-y border-[#4f7a43]/15 bg-[#f1f6ec] px-4 py-3.5 text-[#4a3728]">
            <p className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[.08em] text-[#4f7a43]">
              <span className={mark} aria-hidden>1</span>Best answer from these {view.count} photos
            </p>
            <div className="mt-1 flex items-center justify-between gap-2.5">
              <span className={`${italic(view.headline.rank)} text-[21px] font-semibold leading-tight text-[#2f2419] sm:text-[24px]`}>
                {view.headline.name}
              </span>
              {view.headline.chance && (
                <span className="rounded-full bg-[#4f7a43] px-[11px] py-[3px] text-base font-semibold text-white tabular-nums">{view.headline.chance}</span>
              )}
            </div>
            <p className="mt-1 text-[13px]">
              {label(view.headline.rank)}
              {view.above.map((n) => (
                <span key={n.rank}> · {label(n.rank)} <span className={italic(n.rank)}>{n.name}</span>{n.chance && ` ${n.chance}`}</span>
              ))}
            </p>
            <div className="mt-2.5 grid gap-1.5" data-testid="example-ranks">
              {view.ranks.map((n) => (
                <div key={n.rank} className="grid grid-cols-[58px_1fr_40px] items-center gap-2 text-[12.5px]">
                  <span>{label(n.rank)}</span>
                  <span className="h-2 overflow-hidden rounded bg-white">
                    <b className="block h-full rounded bg-[#7FAE34]" style={{ width: `${Math.max(2, Math.round(n.value * 100))}%` }} />
                  </span>
                  <span className="text-right font-semibold tabular-nums">{n.chance}</span>
                </div>
              ))}
            </div>
          </div>
          {view.specimens.length > 0 && (
            <div className="px-4 pt-3">
              <p className="mb-1.5 flex items-center gap-2 text-xs text-muted-foreground">
                <span className={mark} aria-hidden>2</span>Closest DNA-verified specimens
              </p>
              <div className="grid grid-cols-3 gap-2" data-testid="example-specimens">
                {view.specimens.map((s) => (
                  <a key={s.observation_id} href={s.record_url ?? s.inat_url ?? undefined} target="_blank" rel="noreferrer"
                     className="overflow-hidden rounded-lg border border-[#A87146]/20 text-[11px] leading-tight text-[#5c4a3a] no-underline">
                    <img src={s.photo_url ?? ""} alt={s.name ?? s.species} decoding="async" className="block aspect-[4/3] w-full object-cover" />
                    <span className="block px-1.5 pb-1.5 pt-1">
                      <span className="sci line-clamp-2 block font-semibold text-[#4a3728]">{s.name ?? s.species}</span>
                      <span className="text-muted-foreground">{pct(s.similarity)} alike</span>
                      {s.photo_owner && <span className="block truncate text-[10px] text-muted-foreground">© {s.photo_owner}</span>}
                    </span>
                  </a>
                ))}
              </div>
            </div>
          )}
          {view.perPhoto.length > 1 && (
            <div className="px-4 pb-4 pt-3">
              <p className="mb-1.5 flex items-center gap-2 text-xs text-muted-foreground">
                <span className={mark} aria-hidden>3</span>
                <span>Each photo on its own: how it rates <span className={italic(view.headline.rank)}>{view.headline.name}</span></span>
              </p>
              <div className="flex flex-wrap gap-2" data-testid="example-per-photo">
                {view.perPhoto.map((p, i) => (
                  <span key={i} className="flex flex-col items-center gap-0.5">
                    <img src={p.url} alt="" decoding="async" className="h-10 w-10 rounded-md border border-[#A87146]/20 object-cover" />
                    <span className={`text-[11px] tabular-nums ${p.agrees ? "font-semibold text-[#5f8f1c]" : "text-muted-foreground"}`}>
                      {p.chance ?? "–"}
                    </span>
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
      <ol className="mt-4 grid list-none gap-2.5 p-0 text-sm leading-snug text-[#5c4a3a]" data-testid="list-what-you-get">
        <Callout n="1"><b className="font-semibold text-[#4a3728]">An answer at every rank</b>, each with its own calibrated chance. Often the genus is clear when the species isn't.</Callout>
        <Callout n="2"><b className="font-semibold text-[#4a3728]">The closest DNA-verified specimens</b>, with their photos, to compare by eye.</Callout>
        <Callout n="3"><b className="font-semibold text-[#4a3728]">How each photo scores on its own</b>, to see which ones agree.</Callout>
        <Callout n="+">Species known so far only from DNA, under provisional names; advice on what would firm it up; a link to keep or share the result.</Callout>
      </ol>
    </div>
  );
}

function Callout({ n, children }: { n: string; children: React.ReactNode }) {
  return (
    <li className="flex gap-2.5">
      <span className="mt-px flex h-[22px] w-[22px] shrink-0 items-center justify-center rounded-full bg-[#4a3728] text-xs font-semibold text-white" aria-hidden>{n}</span>
      <span>{children}</span>
    </li>
  );
}

/** The five views, drawn, with today's words (PHOTO_CHECKLIST). A side-scrolling row on a phone. */
export function PhotoGuide() {
  return (
    <Wrap id={GUIDE_ID} testId="section-guide">
      <Kicker>What to photograph</Kicker>
      <H2>Five views make a good identification</H2>
      <Sub>Vision uses every photo you add. The more of these it sees, the firmer the answer.</Sub>
      <div className="-mx-4 flex snap-x snap-mandatory gap-3.5 overflow-x-auto px-4 pb-1.5 sm:mx-0 sm:px-0 lg:grid lg:grid-cols-5 lg:overflow-visible"
           data-testid="guide-views">
        {PHOTO_CHECKLIST.map((c, i) => (
          <div key={c.what} className="w-[74%] shrink-0 snap-start overflow-hidden rounded-xl border border-[#A87146]/20 bg-white sm:w-[230px] lg:w-auto">
            <div className="flex aspect-[4/3] items-center justify-center bg-[#efe9e1]" aria-hidden>{VIEW_ART[i]}</div>
            <div className="px-3.5 pb-3.5 pt-3 text-sm leading-snug">
              <b className="mb-0.5 flex items-center gap-1.5 text-[15px] text-[#4a3728]">
                <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[#4a3728] text-[11px] font-semibold text-white">{i + 1}</span>
                {c.what}
              </b>
              {c.why}
            </div>
          </div>
        ))}
      </div>
      <div className="mt-4 flex flex-wrap gap-x-[18px] gap-y-2 text-sm">
        {["One find per identification", "Sharp, close and in daylight works best",
          "Habitat, microscopy and label photos are recognised and tagged"].map((t) => (
          <span key={t} className="inline-flex items-center gap-1.5"><Check className="h-4 w-4 text-[#4f7a16]" aria-hidden />{t}</span>
        ))}
      </div>
    </Wrap>
  );
}

/** With photos in: the five views as a checklist beside them. Open on a wide screen; on a
 *  phone, folded to one line under the photo box (it can still be opened). */
export function ViewsChecklist({ wide }: { wide: boolean }) {
  const [open, setOpen] = useState(wide);
  return (
    <details className="group mt-3.5 rounded-[14px] border border-[#A87146]/20 bg-[#faf9f7] p-3 lg:mt-[56px] lg:bg-white lg:p-[18px]"
             open={open} onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)} data-testid="list-photo-checklist">
      <summary className="flex cursor-pointer list-none items-center justify-between text-sm text-[#4a3728] [&::-webkit-details-marker]:hidden">
        <span>
          <span className="font-semibold">What to photograph</span>
          <span className="text-muted-foreground group-open:hidden"> · {PHOTO_CHECKLIST.length} views</span>
        </span>
        <ChevronRight className="h-4 w-4 text-muted-foreground transition-transform group-open:rotate-90" aria-hidden />
      </summary>
      <ul className="mt-3 grid list-none gap-2.5 p-0">
        {PHOTO_CHECKLIST.map((c, i) => (
          <li key={c.what} className="grid grid-cols-[44px_1fr] items-center gap-3 rounded-[10px] border border-[#A87146]/20 bg-white py-2 pl-2 pr-3 text-sm leading-snug">
            <span className="flex h-11 w-11 items-center justify-center rounded-lg bg-[#f4efe8] p-1 [&_svg]:h-full [&_svg]:w-full" aria-hidden>{VIEW_ART[i]}</span>
            <span><b className="block font-semibold text-[#4a3728]">{c.what}</b>{c.why}</span>
          </li>
        ))}
      </ul>
      <p className="mt-3 text-xs text-muted-foreground">One find per identification. Sharp, close and in daylight works best.</p>
    </details>
  );
}

/** "Why MycoMap Vision": three cards of live numbers. A card whose numbers aren't there is
 *  left out, and the section with it when none are. */
export function WhyVision({ stats, howGood }: { stats?: Stats | null; howGood?: HowGood | null }) {
  const w = whyNumbers(stats, howGood);
  if (!w.references && !w.provisional && !w.tested) return null;
  const ks = w.tested ? [...new Set(w.tested.rows.flatMap((r) => r.cells.map((c) => c.k)))].sort((a, b) => a - b) : [];
  const rankLabel = (r: (typeof RANKS)[number]) => r.charAt(0).toUpperCase() + r.slice(1);
  return (
    <Wrap band testId="section-why">
      <Kicker>Why MycoMap Vision</Kicker>
      <H2>Named by DNA, tested on finds it never saw</H2>
      <Sub>
        Every name Vision gives comes from a specimen whose DNA was sequenced, so it can name species
        that photo identifiers trained on community votes can't.
      </Sub>
      <div className="grid gap-4 lg:grid-cols-3">
        {w.references && (
          <WhyCard icon={<Dna className="h-[18px] w-[18px]" />} title="DNA-verified references" testId="why-references">
            <Big>{w.references.records}</Big>
            <p className="text-[13px] text-muted-foreground">DNA-verified North American records</p>
            {(w.references.photos || w.references.week) && (
              <p className="mt-1.5 text-sm">
                {w.references.photos && <>Your photos are compared with their {w.references.photos} photos. </>}
                {w.references.week}
              </p>
            )}
          </WhyCard>
        )}
        {w.provisional && (
          <WhyCard icon={<CircleHelp className="h-[18px] w-[18px]" />} brown title="Provisional species too" testId="why-provisional">
            <Big>{w.provisional.names}</Big>
            <p className="text-[13px] text-muted-foreground">
              of {w.provisional.total} names are provisional: known so far only from DNA
            </p>
            <div className="mb-1.5 mt-3 flex h-3.5 gap-0.5 overflow-hidden rounded" role="img"
                 aria-label={`${w.provisional.sharePct} of the names are provisional`}>
              <div className="bg-[#7FAE34]" style={{ width: `${w.provisional.share * 100}%` }} />
              <div className="flex-1 bg-[#d9d0c5]" />
            </div>
            <div className="flex flex-wrap gap-x-3.5 gap-y-1 text-xs">
              <span><i className="mr-1.5 inline-block h-2.5 w-2.5 rounded-sm bg-[#7FAE34] align-[-1px]" />provisional ({w.provisional.sharePct})</span>
              <span><i className="mr-1.5 inline-block h-2.5 w-2.5 rounded-sm bg-[#d9d0c5] align-[-1px]" />formally named</span>
            </div>
          </WhyCard>
        )}
        {w.tested && (
          <WhyCard icon={<BarChart3 className="h-[18px] w-[18px]" />} title="An answer at every rank" testId="why-tested">
            <p className="text-[13px] text-muted-foreground">
              Right name within the first {ks.join(", ").replace(/, (\d+)$/, " and $1")}, on the {w.tested.records} newest
              records{w.tested.after && <> (validated after {w.tested.after})</>}, which it never learned from
            </p>
            <table className="mt-2.5 w-full border-collapse text-[13px] tabular-nums" data-testid="table-top-k">
              <thead>
                <tr className="text-[11px] uppercase tracking-[.06em] text-muted-foreground">
                  <th className="pb-1.5 text-left font-semibold"><span className="sr-only">Rank</span></th>
                  {ks.map((k) => <th key={k} className="pb-1.5 text-right font-semibold">Top {k}</th>)}
                </tr>
              </thead>
              <tbody>
                {w.tested.rows.map((r) => (
                  <tr key={r.rank} className="border-t border-[#ece6de] text-[#4a3728]">
                    <td className="py-1.5 text-left">{rankLabel(r.rank)}</td>
                    {ks.map((k) => {
                      const c = r.cells.find((x) => x.k === k);
                      return <td key={k} className={`py-1.5 text-right ${k === ks[0] ? "font-bold" : ""}`}>{c?.value ?? "–"}</td>;
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </WhyCard>
        )}
      </div>
      <div className="mt-5 flex flex-wrap gap-2.5">
        <a className="inline-flex items-center gap-1.5 rounded-md border border-[#A87146]/20 bg-white px-4 py-2.5 text-sm font-medium text-[#4a3728]"
           href="/about">How it works and how good it is <ArrowRight className="h-4 w-4" aria-hidden /></a>
        <a className="inline-flex items-center gap-1.5 rounded-md border border-[#A87146]/20 bg-white px-4 py-2.5 text-sm font-medium text-[#4a3728]"
           href={SEQUENCING_URL} target="_blank" rel="noreferrer">Free DNA sequencing <ArrowUpRight className="h-4 w-4" aria-hidden /></a>
      </div>
    </Wrap>
  );
}

const Big = ({ children }: { children: React.ReactNode }) => (
  <div className="mb-0.5 mt-2.5 text-[34px] font-semibold leading-tight text-[#4a3728] tabular-nums">{children}</div>
);

function WhyCard({ icon, brown, title, testId, children }: {
  icon: React.ReactNode; brown?: boolean; title: string; testId: string; children: React.ReactNode;
}) {
  return (
    <div className="min-w-0 rounded-xl border border-[#A87146]/20 bg-white p-5" data-testid={testId}>
      <span className={`mb-2.5 flex h-9 w-9 items-center justify-center rounded-lg ${
        brown ? "bg-[#A36848]/[.12] text-[#7a4a2a]" : "bg-myco-green/[.14] text-[#4f7a16]"}`}>{icon}</span>
      <H3>{title}</H3>
      {children}
    </div>
  );
}
