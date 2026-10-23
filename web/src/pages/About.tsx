import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowRight, ArrowUpRight, BookOpen, Camera, ChartLine, Check, Dna, ExternalLink, EyeOff,
  Image as ImageIcon, Lock, MapPin, Moon, RefreshCw, Share2, ShieldCheck, Trash2, TrendingUp, Users,
} from "lucide-react";
import { SafetyNote } from "@/components/SafetyNote";
import { PhotoPermissionLine } from "@/components/PhotoPermissionLine";
import {
  Bars, C, Cap, Card, ExternalLinkText, FlowArrow, H2, H3, Icon, Ink, Kicker, PermissionBox, Small, Sub, Tile, Wrap,
} from "@/components/Explainer";
import {
  api, num, pct, type HowGood, type ModelsInfo, type SpeedDay, type Stats,
} from "@/lib/api";
import { publicMethodLabel, publicModelName } from "@/lib/publicNames";
import { JOIN_URL, longDate, SEQUENCING_URL, shareWords } from "@/lib/publicView";
import {
  depthLabel, latestLabel, modelCredit, namesByRecords, nearestPoint, recordingSince,
  seconds, servedModel, shortDate, speedView, topKRows, MIN_CHART_DAYS, ROLLING_LABEL,
  type SpeedView,
} from "@/lib/howItWorks";

const DANISH_STUDY_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC8779018/";

/** Everything the page shows, as the API sends it; undefined while loading or on failure. */
export interface HowItWorksData {
  stats?: Stats;
  models?: ModelsInfo;
  howGood?: HowGood;
  speed?: SpeedDay[];
  /** Today, UTC (YYYY-MM-DD), for "today" beside the newest speed figure. */
  today: string;
  loading?: boolean;
}

export function AboutPage() {
  const stats = useQuery({ queryKey: ["stats"], queryFn: api.stats });
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const howGood = useQuery({ queryKey: ["how-good"], queryFn: api.howGood });
  const speed = useQuery({ queryKey: ["speed-daily"], queryFn: api.speedDaily });
  return (
    <HowItWorks stats={stats.data} models={models.data} howGood={howGood.data}
                speed={speed.data?.days} today={new Date().toISOString().slice(0, 10)}
                loading={howGood.isLoading || models.isLoading} />
  );
}

/** The page, from its data alone (so every state can be rendered in a test). */
export function HowItWorks(d: HowItWorksData) {
  const s = d.stats;
  const model = servedModel(d.models, d.howGood?.run?.backbone);
  return (
    <div className="text-[#5c4a3a] leading-relaxed">
      <Hero s={s} referencePhotos={model?.embedded_photos} />
      <HowAnIdentificationWorks s={s} h={d.howGood} />
      <DnaNotVotes s={s} />
      <Provisional s={s} />
      <HowGoodSection h={d.howGood} loading={d.loading} speed={d.speed} today={d.today} />
      <FirstStepAndNewestWeeks h={d.howGood} />
      <Standards />
      <YourPhotos />
      <Contributors s={s} />
      <OpenAndComing model={model} />
      <Nonprofit />
    </div>
  );
}

/* ------------------------------------------------------------------ 1. hero */

function Hero({ s, referencePhotos }: { s?: Stats; referencePhotos?: number }) {
  const week = s?.recent_week && s.recent_week.records > 0 ? s.recent_week : null;
  return (
    <div className="bg-[#f8f5f0] border-b border-[#A87146]/10">
      <div className="mx-auto max-w-[1080px] px-4 sm:px-8 py-10">
        <span className="inline-flex items-center gap-1.5 rounded-full bg-myco-green/[.14] px-2.5 py-1 text-xs font-semibold uppercase tracking-[.06em] text-[#4f7a16]">
          <span className="h-2 w-2 shrink-0 rounded-full bg-[#7FAE34] shadow-[0_0_0_3px_hsl(82_53%_51%/.25)]" />
          DNA-verified references only · Updated daily
        </span>
        <h1 className="font-display font-semibold text-[34px] sm:text-[44px] leading-[1.1] text-[#4a3728] mt-3.5">
          How it works
        </h1>
        <p className="text-[17px] sm:text-xl max-w-[720px] mt-3">
          Identify a fungus from all your photos, compared with DNA-sequenced specimens, not
          community votes.
        </p>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mt-7" data-testid="hero-tiles">
          {s && <Tile value={num(s.records_north_america)} label="DNA-verified North American records" />}
          {referencePhotos ? <Tile value={num(referencePhotos)} label="reference photos your photos are compared with" /> : null}
          {s && <Tile value={num(s.names)} label="names in the reference set" />}
          {week && <Tile value={`+${num(week.records)}`}
                         label={`records DNA-verified in the week to ${longDate(week.through)}`} />}
        </div>
        <div className="flex flex-wrap gap-2.5 mt-6">
          <a href="/" className="inline-flex items-center gap-1.5 rounded-md bg-myco-green px-4 py-2.5 text-sm font-medium text-white">
            Identify a fungus <ArrowRight className="h-4 w-4" aria-hidden />
          </a>
          <a href="#how-good" className="inline-flex items-center rounded-md border border-[#A87146]/20 bg-white px-4 py-2.5 text-sm font-medium text-[#4a3728]">
            How good is it?
          </a>
        </div>
        <SafetyNote className="mt-7" />
      </div>
    </div>
  );
}

/* ------------------------------------------------- 2. how an identification works */

function HowAnIdentificationWorks({ s, h }: { s?: Stats; h?: HowGood }) {
  return (
    <Wrap testId="section-identification">
      <Kicker>How an identification works</Kicker>
      <H2>Every photo, every specimen, every rank</H2>
      <Sub>A mushroom often can't be named from its cap alone, so Vision uses all your photos at once.</Sub>
      <div className="grid md:grid-cols-[1fr_44px_1fr_44px_1fr] items-stretch">
        <Card>
          <StepTitle n={1}>You add all your photos</StepTitle>
          <div className="flex gap-2" aria-hidden>
            <Thumb label="Cap"><path d="M2 16C2 7 9 2 17 2s15 5 15 14z" fill="#b9875a" /><rect x="13" y="16" width="8" height="12" rx="2" fill="#e9dcc8" /></Thumb>
            <Thumb label="Underside"><path d="M2 8h30C32 14 25 18 17 18S2 14 2 8z" fill="#b9875a" /><path d="M6 10l3 5M11 10l2 6M17 10v7M23 10l-2 6M28 10l-3 5" stroke="#7a5536" strokeWidth="1.2" /><rect x="14" y="18" width="6" height="10" rx="2" fill="#e9dcc8" /></Thumb>
            <Thumb label="Stem"><path d="M8 6c0-3 18-3 18 0" stroke="#b9875a" strokeWidth="4" fill="none" /><rect x="14" y="6" width="6" height="18" rx="2" fill="#e9dcc8" /><ellipse cx="17" cy="26" rx="7" ry="3" fill="#d8c7ad" /></Thumb>
          </div>
          <p className="text-sm leading-normal mt-2.5">Adding an underside or stem photo can change the answer.</p>
        </Card>
        <FlowArrow />
        <Card>
          <StepTitle n={2}>Compared with every DNA-verified record</StepTitle>
          <div className="grid grid-cols-12 gap-[3px]" aria-hidden>
            {Array.from({ length: 48 }, (_, i) => (
              <i key={i} className="aspect-square rounded-sm"
                 style={i === 30 ? { background: C.s1, outline: "2px solid #4a3728", outlineOffset: 1 }
                   : [5, 18, 41].includes(i) ? { background: C.s1, opacity: 0.45 } : { background: "#e8e1d8" }} />
            ))}
          </div>
          <p className="text-sm leading-normal mt-2.5">
            Each photo is compared with every photo of every DNA-verified record. A species is
            scored by how well its <Ink>best-matching specimen</Ink> fits your photos as a set.
          </p>
        </Card>
        <FlowArrow />
        <Card>
          <StepTitle n={3}>An answer at every rank</StepTitle>
          <div className="flex flex-col gap-2" data-testid="example-answer"
               title="Example of how an answer reads; not a real identification">
            {([["Family", 0.94], ["Genus", 0.88], ["Species", 0.41]] as const).map(([rank, v]) => (
              <div key={rank} className="grid grid-cols-[64px_1fr_44px] items-center gap-2 text-[13px]">
                <span>{rank}</span>
                <span className="h-2.5 rounded-[3px] bg-[#efe9e1] overflow-hidden">
                  <b className="block h-full rounded-r-[3px]" style={{ width: `${v * 100}%`, background: C.s1 }} />
                </span>
                <span className="text-right font-semibold text-[#4a3728] tabular-nums">{pct(v)}</span>
              </div>
            ))}
          </div>
          <p className="text-sm leading-normal mt-2.5">
            Family, genus and species each get their own confidence, calibrated on the newest
            test records so it reads as a rough probability. When the genus is clear but the
            species is not, the answer says so and suggests what would help.
          </p>
          <Small className="mt-1.5">Example answer, for illustration.</Small>
        </Card>
      </div>

      <div className="grid gap-8 md:grid-cols-2 items-start mt-7">
        <FirstPhotoCard h={h} />
        <NamesByRecordsCard s={s} />
      </div>
    </Wrap>
  );
}

function StepTitle({ n, children }: { n: number; children: React.ReactNode }) {
  return (
    <H3 className="flex items-center !mb-2.5">
      <span className="inline-flex h-[26px] w-[26px] shrink-0 items-center justify-center rounded-full bg-[#4a3728] text-[13px] font-semibold text-white mr-2">{n}</span>
      {children}
    </H3>
  );
}

function Thumb({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="relative flex flex-1 aspect-square flex-col items-center justify-end rounded-md bg-[#efe9e1] px-0.5 py-1.5 text-[11px] font-medium text-[#4a3728]">
      <svg width="34" height="30" viewBox="0 0 34 30" className="absolute top-[18%] left-1/2 -translate-x-1/2">{children}</svg>
      {label}
    </div>
  );
}

function FirstPhotoCard({ h }: { h?: HowGood }) {
  const f = h?.first_photo;
  return (
    <Card>
      <H3>Every photo, not one</H3>
      {f && f.all_photos_top1 != null ? (
        <div data-testid="chart-first-photo">
          <Small className="mb-3">Species named right first time, newest {num(f.n)} test records</Small>
          <Bars labelWidth={130} rows={[
            { label: "First photo only", frac: f.first_photo_top1, text: pct(f.first_photo_top1, 1),
              color: C.rest, title: `First photo only: ${pct(f.first_photo_top1, 1)}` },
            { label: "All the photos", frac: f.all_photos_top1, text: pct(f.all_photos_top1, 1),
              color: C.s1, title: `All the photos of the find: ${pct(f.all_photos_top1, 1)}` },
          ]} />
          <Cap>Same records, same model: only the number of photos changes.</Cap>
        </div>
      ) : (
        <p className="text-sm">
          Each of your photos is compared with every photo of every DNA-verified record, and an
          underside or stem photo can change the answer.
        </p>
      )}
    </Card>
  );
}

function NamesByRecordsCard({ s }: { s?: Stats }) {
  const rows = namesByRecords(s);
  const top = rows ? Math.max(...rows.map((r) => r.n)) : 1;
  const single = rows?.find((r) => r.key === "1");
  return (
    <Card>
      <H3>Rare species are not drowned out</H3>
      {rows && (
        <>
          <Small className="mb-3">Names in the reference set, by how many DNA-verified records each has</Small>
          <Bars labelWidth={80} testId="chart-names-by-records" rows={rows.map((r) => ({
            label: r.label, frac: r.n / top, text: num(r.n), color: r.key === "1" ? C.s1 : C.rest,
            title: `${r.key === "1" ? r.label : `${r.label} records`}: ${num(r.n)} names (${pct(r.share)})`,
          }))} />
        </>
      )}
      <Cap>
        {single && s ? `${shareWords(single.n, s.names)} of the names rest on a single DNA-verified record. ` : ""}
        A species is scored by its closest specimen, not by how many records it has, so one
        known from one specimen competes on equal terms with one known from a thousand. The
        answer says when a match rests on a single specimen.
      </Cap>
    </Card>
  );
}

/* ------------------------------------------------------------ 3. DNA, not votes */

function DnaNotVotes({ s }: { s?: Stats }) {
  const week = s?.recent_week && s.recent_week.records > 0 ? s.recent_week : null;
  return (
    <Wrap band testId="section-dna">
      <Kicker>DNA-verified references only · Updated daily</Kicker>
      <H2>Learns from DNA, not from votes</H2>
      <Sub>Most photo identifiers learn from records named by community agreement, so their mistakes are built in.</Sub>
      <div className="grid gap-8 md:grid-cols-2 items-start">
        <div className="min-w-0">
          <div className="grid grid-cols-2 overflow-hidden rounded-[10px] border border-[#A87146]/20 bg-white">
            <div className="bg-[#f4f1ec] p-4 sm:p-[18px]">
              <Small className="mb-1.5 uppercase tracking-[.06em] font-semibold">Most identifiers</Small>
              <div className="mb-2.5"><Icon brown><Users className="h-5 w-5" aria-hidden /></Icon></div>
              <Ink>Community agreement</Ink>
              <p className="mt-1.5 text-sm leading-snug">A name wins by votes. Wrong names get learned too.</p>
            </div>
            <div className="p-4 sm:p-[18px]">
              <p className="mb-1.5 text-[13px] uppercase tracking-[.06em] font-semibold text-[#4f7a16]">MycoMap Vision</p>
              <div className="mb-2.5"><Icon><Dna className="h-5 w-5" aria-hidden /></Icon></div>
              <Ink>DNA sequencing</Ink>
              <p className="mt-1.5 text-sm leading-snug">Only records whose name is backed by DNA and marked green in a MycoMap project.</p>
            </div>
          </div>
          <Cap>No community identifications from iNaturalist or Mushroom Observer are used as labels.</Cap>
        </div>
        <Card>
          <H3 className="flex items-center gap-2"><Moon className="h-[18px] w-[18px] text-[#4f7a16]" aria-hidden />Every night</H3>
          <div className="mt-3 flex flex-col gap-3 text-[15px]">
            <div className="flex items-center gap-3"><Icon><b className="text-xl">+</b></Icon><span>Adds the records <Ink>validated that day</Ink></span></div>
            <div className="flex items-center gap-3"><Icon brown><b className="text-xl">−</b></Icon><span>Drops any record <Ink>no longer validated</Ink></span></div>
            <div className="flex items-center gap-3"><Icon><RefreshCw className="h-[18px] w-[18px]" aria-hidden /></Icon><span>Follows <Ink>renames on mycomap.org</Ink></span></div>
          </div>
          {week && (
            <div className="mt-4 flex flex-wrap items-baseline gap-x-2.5 border-t border-[#ece6de] pt-3" data-testid="weekly-counter">
              <span className="text-[28px] font-semibold text-[#4a3728] tabular-nums">+{num(week.records)}</span>
              <span className="text-[13px] text-muted-foreground">
                North American records DNA-verified in the week to {longDate(week.through)}
              </span>
            </div>
          )}
        </Card>
      </div>
    </Wrap>
  );
}

/* --------------------------------------------------------------- 4. provisional */

function Provisional({ s }: { s?: Stats }) {
  const p = s?.names_provisional;
  const share = s && p != null && s.names > 0 ? { p, n: s.names, f: p / s.names } : null;
  return (
    <Wrap testId="section-provisional">
      <div className="grid gap-8 md:grid-cols-2 items-center">
        <div className="min-w-0">
          <Kicker>Species no photo guide shows yet</Kicker>
          <H2>{share ? `${shareWords(share.p, share.n)} of the names are provisional species`
                     : "Many names are provisional species"}</H2>
          {share && (
            <>
              <div className="flex h-7 gap-0.5 mt-3.5 mb-2" role="img"
                   aria-label={`${num(share.p)} of ${num(share.n)} names (${pct(share.f)}) are provisional`}>
                <div className="h-full rounded-l" style={{ width: `${share.f * 100}%`, background: C.s1 }}
                     title={`Provisional species: ${num(share.p)} names (${pct(share.f)})`} />
                <div className="h-full rounded-r" style={{ width: `${(1 - share.f) * 100}%`, background: C.rest }}
                     title={`Other names: ${num(share.n - share.p)} (${pct(1 - share.f)})`} />
              </div>
              <div className="flex flex-wrap gap-x-4 gap-y-1 text-[13px]">
                <span className="inline-flex items-center gap-1.5"><i className="h-3 w-3 rounded-[3px]" style={{ background: C.s1 }} />{num(share.p)} provisional ({pct(share.f)})</span>
                <span className="inline-flex items-center gap-1.5"><i className="h-3 w-3 rounded-[3px]" style={{ background: C.rest }} />{num(share.n - share.p)} other names</span>
              </div>
              <Cap className="!mt-1">Of the {num(share.n)} names in the reference set.</Cap>
            </>
          )}
        </div>
        <Card>
          <div className="inline-block rounded-lg border border-dashed border-[#b9a48f] bg-white px-3.5 py-2 text-xl text-[#4a3728] font-species">
            <span className="sci">Russula</span> sp. 'IN01'
          </div>
          <ul className="mt-3.5 list-disc pl-[18px] text-[15px] leading-normal space-y-0.5">
            <li>A species told apart by its <Ink>DNA</Ink>, not formally described yet.</li>
            <li>Identifiers that learn from community names <Ink>can't suggest it</Ink>: there is no name to vote on.</li>
            <li>When Vision suggests one, your find may belong to a species still waiting to be described.</li>
            <li>The code is a working label, not a formal name. It changes when the species is published.</li>
          </ul>
        </Card>
      </div>
    </Wrap>
  );
}

/* --------------------------------------------------------------- 5. how good */

function HowGoodSection({ h, loading, speed, today }: {
  h?: HowGood; loading?: boolean; speed?: SpeedDay[]; today: string;
}) {
  const run = h?.run;
  const ladder = topKRows(h);
  return (
    <Wrap band id="how-good" testId="section-how-good">
      <Kicker>How good is it</Kicker>
      <H2>Right first time, on records it had never seen</H2>
      {!run || !h?.top1 ? (
        <p className="mb-7">{loading ? "Loading the latest results…" : "Results for this model are not published yet."}</p>
      ) : (
        <>
          <Sub>
            Tested on the newest <Ink>{num(run.n_test)}</Ink> DNA-verified records (validated
            after {longDate(run.cutoff)}), identified from the <Ink>{num(run.n_reference)}</Ink> records
            verified before them.
          </Sub>
          <div className="grid grid-cols-3 gap-2 sm:gap-3 mb-2">
            <Score rank="Species" value={h.top1.species} />
            <Score rank="Genus" value={h.top1.genus} />
            <Score rank="Family" value={h.top1.family} />
          </div>
          <Small className="mb-3">The first answer was right this often.</Small>
          {h.inat && (
            <p className="text-sm" data-testid="text-inat">
              {publicModelName("external:inat-cv")}, given the same photos
              {h.inat.method === "combined-max" ? " and locations" : ""}, was right first time for{" "}
              {pct(h.inat.species_top1, 1)} at species, {pct(h.inat.genus_top1, 1)} at genus and{" "}
              {pct(h.inat.family_top1, 1)} at family.
            </p>
          )}
          <div className="grid gap-8 md:grid-cols-2 items-start mt-7">
            <Card>
              <H3>The right name, within the first few</H3>
              {ladder ? <TopKLadder rows={ladder} /> : (
                <Small>Top 3, 5 and 10 are not measured for this run yet.</Small>
              )}
              <Cap>
                Species is the hard rank: many fungi are told apart only by DNA, and many names
                here are new, provisional ones. Genus is right far more often, and the answer
                tells you when the species is uncertain.
              </Cap>
            </Card>
            <DepthCard h={h} />
          </div>
        </>
      )}
      <SpeedCard days={speed} today={today} />
      {run && (
        <div data-testid="text-model-credit">
          <Small className="mt-[18px]">
            Model: {modelCredit(run.backbone).name}, {publicMethodLabel(run.method)}. Measured on
            the newest DNA-verified weeks; every test is on the{" "}
            <a className="underline" href="/research/results">Results page</a>.
          </Small>
        </div>
      )}
    </Wrap>
  );
}

function Score({ rank, value }: { rank: string; value: number | null }) {
  return (
    <div className="min-w-0 rounded-lg border border-[#A87146]/20 bg-[#faf9f7] px-1.5 py-3.5 sm:px-3 sm:py-[18px] text-center"
         title={`${rank}, first answer right: ${pct(value, 1)}`}>
      <div className="text-[26px] sm:text-[40px] font-semibold leading-tight text-[#4a3728] tabular-nums">{pct(value, 1)}</div>
      <div className="text-[11px] sm:text-xs uppercase tracking-[.04em] sm:tracking-[.08em] text-muted-foreground mt-1">{rank}</div>
      <div className="mx-2 mt-3 h-1.5 overflow-hidden rounded-[3px] bg-[#ebe4da]">
        <b className="block h-full" style={{ width: `${(value ?? 0) * 100}%`, background: C.s1 }} />
      </div>
    </div>
  );
}

function TopKLadder({ rows }: { rows: { k: number; species: number | null; genus: number | null }[] }) {
  const ks = rows.map((r) => r.k);
  const listed = ks.length > 1 ? `${ks.slice(0, -1).join(", ")} or ${ks[ks.length - 1]}` : `${ks[0]}`;
  return (
    <div data-testid="chart-top-k">
      <Small className="mb-3">How often the right name is in the top {listed} answers</Small>
      <div className="flex flex-wrap gap-4 text-[13px] mb-3">
        <span className="inline-flex items-center gap-1.5"><i className="h-3 w-3 rounded-[3px]" style={{ background: C.s1 }} />Species</span>
        <span className="inline-flex items-center gap-1.5"><i className="h-3 w-3 rounded-[3px]" style={{ background: C.s2 }} />Genus</span>
      </div>
      <table className="w-full border-collapse text-sm">
        <tbody>
          {rows.map((r) => (
            <tr key={r.k} className="border-t border-[#ece6de]">
              <td className="w-[74px] py-[7px] font-semibold text-[#4a3728] whitespace-nowrap">Top {r.k}</td>
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
    </div>
  );
}

function DepthCard({ h }: { h: HowGood }) {
  const bands = h.species_by_depth;
  const ks = h.top_k?.k ?? [1, 3, 5, 10];
  const all = h.top_k?.species;
  return (
    <Card>
      <H3>Every sequenced record makes it better</H3>
      {!bands ? (
        <Small>Accuracy by how many records each species has is not measured for this run yet.</Small>
      ) : (
        <>
          <Small className="mb-3">Species right first time, by how many DNA-verified records the species has</Small>
          <Bars labelWidth={96} testId="chart-by-depth" rows={bands.map((b, i) => ({
            label: depthLabel(b.band), sub: `${num(b.n)} tested`, frac: b.top1 ?? 0,
            text: b.band === "0" && !b.top1 ? `${pct(b.top1, 1)} · no reference yet` : pct(b.top1, 1),
            color: C.ramp[Math.min(i, C.ramp.length - 1)],
            title: `${depthLabel(b.band)}: ${pct(b.top1, 1)} right first time (${num(b.n)} test records)`,
          }))} />
          <div className="flex justify-between text-[11px] text-muted-foreground mt-1.5 mr-[4.5rem] ml-[108px]">
            <span>0%</span><span>50%</span><span>100%</span>
          </div>
          <Cap>
            A species with no DNA-verified reference yet can't be named at species. The more
            DNA-verified records a species has, the more often it is right first time.
          </Cap>
          <details className="mt-3.5 text-[13px]">
            <summary className="cursor-pointer font-medium text-[#4a3728]">
              Full numbers (top {ks.join(" / ")})
            </summary>
            <div className="overflow-x-auto">
              <table className="mt-2 border-collapse tabular-nums" data-testid="table-by-depth">
                <thead>
                  <tr>
                    <th className="px-2.5 py-1 text-left border-b border-[#ece6de]">Records of the species</th>
                    <th className="px-2.5 py-1 text-right border-b border-[#ece6de]">Test records</th>
                    {ks.map((k) => <th key={k} className="px-2.5 py-1 text-right border-b border-[#ece6de]">Top {k}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {[...bands, ...(all ? [{ ...all, band: "All" }] : [])].map((b) => (
                    <tr key={b.band}>
                      <td className="px-2.5 py-1 border-b border-[#ece6de]">{b.band.replace("-", "–")}</td>
                      <td className="px-2.5 py-1 text-right border-b border-[#ece6de]">{num(b.n)}</td>
                      {ks.map((k) => (
                        <td key={k} className="px-2.5 py-1 text-right border-b border-[#ece6de]">
                          {pct(b[`top${k}` as "top1"], 1)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        </>
      )}
    </Card>
  );
}

/* ------------------------------------------------------------- speed chart */

function SpeedCard({ days, today }: { days?: SpeedDay[]; today: string }) {
  const view = speedView(days);
  const latest = view.state === "none" ? null : view.latest;
  return (
    <Card className="mt-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <H3>How fast is it</H3>
          <Small>Average time per photo, day by day (seconds, UTC days)</Small>
        </div>
        {latest && (
          <div className="text-right" data-testid="speed-latest">
            <div className="text-[28px] font-semibold leading-tight text-[#4a3728] tabular-nums">{seconds(latest.seconds_per_photo)}</div>
            <div className="text-[13px] text-muted-foreground">
              per photo, {latest.day === today ? "today" : shortDate(latest.day)}
            </div>
          </div>
        )}
      </div>
      <div data-testid="speed-chart" data-state={view.state}>
        {view.state === "chart" && <SpeedLine view={view} today={today} />}
        {view.state === "few" && (
          <ul className="mt-4 space-y-1 text-sm tabular-nums">
            {view.days.map((d) => (
              <li key={d.day}><Ink>{seconds(d.seconds_per_photo)}</Ink> per photo on {longDate(d.day)}</li>
            ))}
            {view.rolling != null && (
              <li data-testid="speed-rolling">{ROLLING_LABEL}: <Ink>{seconds(view.rolling)}</Ink> per photo</li>
            )}
          </ul>
        )}
        {view.state === "none" && (
          <p className="mt-4 text-sm">
            No day recorded yet. Each day's average appears here once photos have been identified that day.
          </p>
        )}
      </div>
      <Cap>
        Lower is faster. Each point is one day's average across every photo identified that day;
        the {ROLLING_LABEL} is the mean of the days recorded in the 7 days to each day.
        {view.state !== "none" && <> {recordingSince(view.since)}.</>}
        {view.state === "few" && <> The daily chart starts once there are {MIN_CHART_DAYS} days.</>}
      </Cap>
    </Card>
  );
}

/** "0", "0.5 s", "1.0 s", "2.5 s": the y-axis labels. */
const tickLabel = (v: number) => (v === 0 ? "0" : `${Number.isInteger(v * 10) ? v.toFixed(1) : v.toFixed(2)} s`);

function SpeedLine({ view, today }: { view: Extract<SpeedView, { state: "chart" }>; today: string }) {
  const [hover, setHover] = useState<number | null>(null);
  const pts = view.points;
  const line = pts.map((p, i) => `${i ? "L" : "M"}${(p.x * 300).toFixed(2)} ${(p.y * 100).toFixed(2)}`).join(" ");
  const avg = pts.filter((p) => p.ry != null)
    .map((p, i) => `${i ? "L" : "M"}${(p.x * 300).toFixed(2)} ${(p.ry! * 100).toFixed(2)}`).join(" ");
  const shown = hover == null ? null : pts[hover];
  const move = (e: React.PointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const p = nearestPoint(pts, (e.clientX - r.left) / r.width);
    setHover(p ? pts.indexOf(p) : null);
  };
  return (
    <div className="relative mt-3.5">
      <div className="mb-3 flex flex-wrap gap-4 text-[13px]" data-testid="speed-legend">
        <span className="inline-flex items-center gap-1.5"><i className="h-0.5 w-4 rounded" style={{ background: C.s1 }} />Daily average</span>
        {view.hasRolling && (
          <span className="inline-flex items-center gap-1.5"><i className="h-[3px] w-4 rounded" style={{ background: C.s2 }} />{ROLLING_LABEL}</span>
        )}
      </div>
      <div className="relative ml-10 h-[220px]" onPointerMove={move} onPointerDown={move}
           onPointerLeave={() => setHover(null)} role="img"
           aria-label={`Average seconds per photo by day, ${pts.length} days from ${longDate(view.since)}; newest ${latestLabel(view.latest, today)}`}>
        <svg viewBox="0 0 300 100" preserveAspectRatio="none" className="absolute inset-0 h-full w-full overflow-visible" aria-hidden>
          {view.yTicks.map((t) => (
            <line key={t.v} x1="0" x2="300" y1={t.y * 100} y2={t.y * 100} stroke="#ece6de" strokeWidth="1" vectorEffect="non-scaling-stroke" />
          ))}
          <path d={`${line} L300 100 L0 100 Z`} fill={C.s1} fillOpacity=".10" />
          <path d={line} fill="none" stroke={C.s1} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
          {view.hasRolling && (
            <path d={avg} fill="none" stroke={C.s2} strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round"
                  vectorEffect="non-scaling-stroke" data-testid="speed-rolling-line" />
          )}
        </svg>
        {view.yTicks.map((t) => (
          <span key={t.v} className="absolute -left-10 w-8 -translate-y-1/2 text-right text-[11px] text-muted-foreground tabular-nums"
                style={{ top: `${t.y * 100}%` }}>{tickLabel(t.v)}</span>
        ))}
        <span className="absolute -right-[5px] h-2.5 w-2.5 -translate-y-1/2 rounded-full shadow-[0_0_0_2px_#fff]"
              style={{ top: `${view.latest.y * 100}%`, background: C.s1 }} />
        <span className="absolute right-2.5 -translate-y-[130%] rounded-sm bg-white/85 px-1 text-xs font-semibold text-[#4a3728]"
              style={{ top: `${view.latest.y * 100}%` }}>{latestLabel(view.latest, today)}</span>
        {shown && (
          <>
            <div className="pointer-events-none absolute inset-y-0 w-px bg-[#b9a48f]" style={{ left: `${shown.x * 100}%` }} />
            <div className="pointer-events-none absolute z-10 whitespace-nowrap rounded-md border border-[#A87146]/20 bg-white px-2.5 py-1.5 text-xs shadow-md"
                 style={{ left: `clamp(0px, calc(${shown.x * 100}% - 65px), calc(100% - 130px))`,
                          top: `max(0px, calc(${shown.y * 100}% - 82px))` }}>
              <span className="flex items-center gap-1.5">
                <i className="h-0.5 w-3 rounded" style={{ background: C.s1 }} />
                <b className="text-sm text-[#4a3728]">{seconds(shown.seconds_per_photo)} per photo</b>
              </span>
              {shown.rolling != null && (
                <span className="flex items-center gap-1.5">
                  <i className="h-[3px] w-3 rounded" style={{ background: C.s2 }} />
                  <span><b className="text-[#4a3728]">{seconds(shown.rolling)}</b> {ROLLING_LABEL}</span>
                </span>
              )}
              <span className="block text-muted-foreground">{longDate(shown.day)}</span>
            </div>
          </>
        )}
      </div>
      <div className="relative ml-10 mt-1.5 h-4 text-[11px] text-muted-foreground">
        {view.xTicks.map((t, i) => (
          <span key={t.day} className="absolute whitespace-nowrap"
                style={{ left: `${t.x * 100}%`,
                         transform: i === 0 ? "none" : i === view.xTicks.length - 1 ? "translateX(-100%)" : "translateX(-50%)" }}>
            {t.label}
          </span>
        ))}
      </div>
      <details className="mt-3 text-[13px]">
        <summary className="cursor-pointer font-medium text-[#4a3728]">Every day's figure</summary>
        <ul className="mt-2 sm:columns-2 tabular-nums" data-testid="speed-table">
          {[...pts].reverse().map((p) => (
            <li key={p.day}>
              {shortDate(p.day)}: {seconds(p.seconds_per_photo)}
              {p.rolling != null && <span className="text-muted-foreground"> ({ROLLING_LABEL} {seconds(p.rolling)})</span>}
            </li>
          ))}
        </ul>
      </details>
    </div>
  );
}

/* ------------------------------------------- 6 and 7. first step, newest weeks */

function FirstStepAndNewestWeeks({ h }: { h?: HowGood }) {
  const run = h?.run;
  return (
    <Wrap testId="section-first-step">
      <div className="grid gap-8 md:grid-cols-2 items-start">
        <div className="min-w-0">
          <Kicker>A first step, not the last word</Kicker>
          <H2>Vision suggests. You compare. DNA settles it.</H2>
          <div className="mt-4 flex flex-col gap-2.5">
            <Step color={C.ramp[0]} n={1} title="Vision suggests names">with how likely each is.</Step>
            <Step color={C.ramp[2]} n={2} title="You compare, photo by photo">with the DNA-verified specimens it shows.</Step>
            <Step color={C.ramp[4]} n={3} title="DNA settles it">
              MycoMap offers <ExternalLinkText href={SEQUENCING_URL}>free DNA sequencing</ExternalLinkText> for finds like yours.
            </Step>
          </div>
          <Cap>Vision learned from DNA-verified records, but its answer for your find comes from photos alone.</Cap>
        </div>
        <Card className="self-center">
          <H3>People checking make the difference</H3>
          <Small className="mb-3.5">Danish Fungal Atlas app, share of submissions named right</Small>
          <Bars labelWidth={150} testId="chart-danish" rows={[
            { label: "AI's first suggestion alone", frac: 0.69, text: "69%", color: C.rest,
              title: "AI's first suggestion alone: 69%" },
            { label: "People choosing, experts checking", frac: 0.93, text: "93%", color: C.s1,
              title: "People choosing among suggestions, experts checking: 93%" },
          ]} />
          <Cap>Source: <ExternalLinkText href={DANISH_STUDY_URL}>Picek et al., Sensors 2022</ExternalLinkText></Cap>
        </Card>
      </div>

      <div className="mt-12" data-testid="section-newest-weeks">
        <Kicker>Measured on the newest weeks</Kicker>
        <H2>Tested the way it meets a new find</H2>
        {run && (
          <>
            <div className="flex min-h-14 gap-0.5 overflow-hidden rounded-lg mt-[18px] mb-2 text-white" role="img"
                 aria-label={`${num(run.n_reference)} earlier records are the reference; the ${num(run.n_test)} newest are the test`}>
              <div className="flex min-w-0 flex-col justify-center bg-[#8a7560] px-3.5 py-2 text-[13px] leading-tight"
                   style={{ flex: `${run.n_reference} 1 0` }}
                   title={`Reference: ${num(run.n_reference)} records verified up to ${longDate(run.cutoff)}`}>
                <b className="text-base">{num(run.n_reference)}</b><span className="truncate">records it may use</span>
              </div>
              <div className="flex min-w-[130px] sm:min-w-[220px] flex-col justify-center px-3.5 py-2 text-[13px] leading-tight"
                   style={{ flex: `${run.n_test} 0 0`, background: C.ramp[3] }}
                   title={`Test: the ${num(run.n_test)} newest records`}>
                <b className="text-base">{num(run.n_test)}</b><span>newest records it is tested on</span>
              </div>
            </div>
            <div className="flex justify-between gap-3 text-xs text-muted-foreground">
              <span>Oldest DNA-verified records</span><span className="text-right">{longDate(run.cutoff)} → newest</span>
            </div>
          </>
        )}
        <Cap>
          Each model is tested on the records that turned green most recently, identified using
          only the records verified before them. Never a random split.{" "}
          <a className="underline" href="/research/results">Results</a> shows every test, broken
          down by how many reference records each species has.
        </Cap>
      </div>
    </Wrap>
  );
}

function Step({ color, n, title, children }: { color: string; n: number; title: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-3 rounded-[10px] border border-[#A87146]/20 bg-white p-3.5 text-sm leading-snug"
         style={{ borderLeft: `4px solid ${color}` }}>
      <Icon><b>{n}</b></Icon>
      <span><b className="block text-[#4a3728]">{title}</b>{children}</span>
    </div>
  );
}

/* ---------------------------------------------------------------- 8. standards */

const STANDARDS: { title: string; text: React.ReactNode; testId?: string }[] = [
  { title: "DNA-validated labels", text: "Labels come only from records DNA-validated (marked green) in a MycoMap project." },
  { title: "No community votes", text: "No community identifications from iNaturalist or Mushroom Observer are used as labels." },
  { title: "Renames, every night", text: "Names follow renames on mycomap.org every night, and records no longer validated drop out." },
  { title: "Newest weeks, not a random split", text: "Tested on the newest weeks of records, and ahead of the DNA on records still waiting for their sequence." },
  { title: "Every figure has its test", text: "Every accuracy figure comes with its test set and rank." },
  { title: "North America only", text: "A find from elsewhere is compared with North American species." },
  { title: "Photographers in control", text: <PhotoPermissionLine kind="control" />, testId: "standard-photographers" },
  { title: "Hidden stays hidden", text: "Locations hidden on iNaturalist stay hidden." },
];

function Standards() {
  return (
    <Wrap band testId="section-standards">
      <Kicker>Our standards</Kicker>
      <H2>Eight rules every answer follows</H2>
      <div className="mt-5 grid grid-cols-1 min-[340px]:grid-cols-2 lg:grid-cols-4 gap-2 sm:gap-3">
        {STANDARDS.map((s) => (
          <div key={s.title} data-testid={s.testId}
               className="min-w-0 rounded-[10px] border border-[#A87146]/20 bg-white p-3 sm:p-4 text-[13px] sm:text-sm leading-snug">
            <div className="mb-2.5 flex h-7 w-7 items-center justify-center rounded-full bg-myco-green/[.16] text-[#4f7a16]">
              <Check className="h-4 w-4" strokeWidth={2.6} aria-hidden />
            </div>
            <b className="block text-sm sm:text-[15px] text-[#4a3728] mb-0.5">{s.title}</b>
            {s.text}
          </div>
        ))}
      </div>
    </Wrap>
  );
}

/* --------------------------------------------------------------- 9. your photos */

const YOUR_PHOTOS: { icon: React.ReactNode; title: string; text: string }[] = [
  { icon: <Share2 className="h-[18px] w-[18px]" />, title: "Its own link", text: "Every identification gets one, so you can share it or come back to it. Vision keeps the answer." },
  { icon: <Lock className="h-[18px] w-[18px]" />, title: "Only people you give it to", text: "Search engines don't see it and nothing lists it." },
  { icon: <ImageIcon className="h-[18px] w-[18px]" />, title: "Smaller copies", text: "Photos you upload are kept at most 1024 pixels." },
  { icon: <EyeOff className="h-[18px] w-[18px]" />, title: "Hidden camera details removed", text: "Exact place, time and camera." },
  { icon: <MapPin className="h-[18px] w-[18px]" />, title: "Place to about 10 km", text: "Kept only to a tenth of a degree, as the answer shows it." },
  { icon: <Trash2 className="h-[18px] w-[18px]" />, title: "Stays until you delete it", text: "From its page. A social-media preview (the answer and your photos) can stay for a while after." },
  { icon: <ExternalLink className="h-[18px] w-[18px]" />, title: "iNaturalist and Mushroom Observer links", text: "Keep no photos, only where they are at the source." },
  { icon: <ShieldCheck className="h-[18px] w-[18px]" />, title: "Never used to train", text: "Your photos are never used to train Vision or as references." },
];

function YourPhotos() {
  return (
    <Wrap testId="section-your-photos">
      <Kicker>Your photos, and each result's own link</Kicker>
      <H2>Private by default, yours to delete</H2>
      <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {YOUR_PHOTOS.map((p) => (
          <div key={p.title} className="flex min-w-0 items-start gap-3 rounded-[10px] border border-[#A87146]/20 bg-white p-3.5 text-sm leading-snug">
            <Icon>{p.icon}</Icon>
            <span><b className="block text-[#4a3728]">{p.title}</b>{p.text}</span>
          </div>
        ))}
      </div>
    </Wrap>
  );
}

/* -------------------------------------------------------------- 10. contributors */

const LOOP: { icon: React.ReactNode; title: string; text: React.ReactNode; last?: boolean }[] = [
  { icon: <Camera className="h-[22px] w-[22px]" />, title: "Your find", text: "photographed and posted" },
  { icon: <Dna className="h-[22px] w-[22px]" />, title: "Sequenced",
    text: <ExternalLinkText href={SEQUENCING_URL}>How to get a find sequenced</ExternalLinkText> },
  { icon: <Check className="h-[22px] w-[22px]" strokeWidth={2.4} />, title: "Validated", text: "its DNA name marked green in a MycoMap project" },
  { icon: <Moon className="h-[22px] w-[22px]" />, title: "A reference, the next night", text: "with its photos", last: true },
];

function Contributors({ s }: { s?: Stats }) {
  return (
    <Wrap band testId="section-contributors">
      <Kicker>Built by contributors</Kicker>
      <H2>Made possible by people who found, photographed and sequenced these fungi</H2>
      {s && (
        <div className="mt-5 grid gap-3 sm:grid-cols-3">
          <Tile value={num(s.records_north_america)} label="DNA-verified North American records" />
          {s.photographers ? <Tile value={num(s.photographers)} label="people photographed them" /> : null}
          {s.projects ? <Tile value={num(s.projects)} label="MycoMap projects marked them green" /> : null}
        </div>
      )}
      <Cap>The photos come from iNaturalist and Mushroom Observer, where the finds were posted, and stay their photographers' own.</Cap>
      <PermissionBox kind="control" className="mt-[18px]" />

      <div className="mt-11" data-testid="section-becomes-reference">
        <H3 className="!text-[22px]">Your sequenced find becomes a reference</H3>
        <Small className="mb-[22px]">From then on, Vision can match the next find like it.</Small>
        <div className="relative grid grid-cols-2 gap-y-6 md:grid-cols-4">
          <div className="absolute left-[12.5%] right-[12.5%] top-7 hidden h-0.5 bg-[#cfe3ad] md:block" aria-hidden />
          {LOOP.map((l) => (
            <div key={l.title} className="relative px-2.5 text-center">
              <div className={`relative z-[1] mx-auto mb-2.5 flex h-14 w-14 items-center justify-center rounded-full border-2 ${
                l.last ? "text-white" : "bg-white text-[#4f7a16]"}`}
                   style={{ borderColor: C.s1, background: l.last ? C.s1 : undefined }}>{l.icon}</div>
              <b className="block text-[15px] text-[#4a3728]">{l.title}</b>
              <span className="block text-[13px] leading-snug">{l.text}</span>
            </div>
          ))}
        </div>
        <PermissionBox kind="learn" className="mt-6" />
      </div>
    </Wrap>
  );
}

/* ------------------------------------------------------- 11. open, still to come */

function OpenAndComing({ model }: { model: ReturnType<typeof servedModel> }) {
  const credit = model ? modelCredit(model.backbone) : null;
  return (
    <Wrap testId="section-openness">
      <div className="grid gap-8 md:grid-cols-2 items-start">
        <div className="min-w-0">
          <Kicker>Open about how it's made</Kicker>
          <H2>Every test, in the open</H2>
          <div className="mt-4 flex flex-col gap-2.5">
            <OpenItem icon={<ChartLine className="h-[18px] w-[18px]" />} title="Every test of every model">
              on the <a className="underline" href="/research/results">Results page</a>; how we test is in
              the <a className="underline" href="/research/protocols">evaluation protocols</a>.
            </OpenItem>
            <OpenItem icon={<ImageIcon className="h-[18px] w-[18px]" />} title="The records and photos it learns from">
              on the <a className="underline" href="/research/data">Reference data page</a>.
            </OpenItem>
            <OpenItem icon={<BookOpen className="h-[18px] w-[18px]" />} title="A paper is in preparation">
              Until it is out, please cite MycoMap Vision (vision.mycomap.org) with the date you used it.
            </OpenItem>
          </div>
        </div>
        {model && credit && (
          <div className="min-w-0 rounded-[10px] border border-[#A87146]/20 bg-[#faf9f7] p-5" data-testid="card-model-in-use">
            <Small className="mb-1 uppercase tracking-[.06em] font-semibold">Model in use</Small>
            <H3 className="!text-xl">{credit.title}</H3>
            {credit.detail && <Small>{credit.detail}</Small>}
            <div className="mt-4 grid grid-cols-2 gap-3">
              <div>
                <div className="text-[22px] font-semibold text-[#4a3728] tabular-nums">{num(model.embedded_photos)}</div>
                <Small>reference photos it compares yours with</Small>
              </div>
              {model.trained_through && (
                <div>
                  <div className="text-[22px] font-semibold text-[#4a3728]">{shortDate(model.trained_through)} {model.trained_through.slice(0, 4)}</div>
                  <Small>names learned from records validated up to</Small>
                </div>
              )}
            </div>
            {credit.madeBy && <Cap className="border-t border-[#ece6de] pt-2.5 !mt-4">{credit.madeBy}</Cap>}
          </div>
        )}
      </div>

      <div className="mt-12">
        <Kicker>What's still to come</Kicker>
        <div className="mt-2.5 grid gap-4 md:grid-cols-3">
          <Card><div className="mb-2.5"><Icon><ImageIcon className="h-[18px] w-[18px]" /></Icon></div><H3>Which part each photo shows</H3><p className="text-sm">Recognising cap, underside, stem and more.</p></Card>
          <Card><div className="mb-2.5"><Icon><MapPin className="h-[18px] w-[18px]" /></Icon></div><H3>Range and season</H3><p className="text-sm">Learned from DNA records.</p></Card>
          <Card><div className="mb-2.5"><Icon><TrendingUp className="h-[18px] w-[18px]" /></Icon></div><H3>A growing reference set</H3><p className="text-sm">Growing as more records are sequenced.</p></Card>
        </div>
      </div>
    </Wrap>
  );
}

function OpenItem({ icon, title, children }: { icon: React.ReactNode; title: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-3 rounded-[10px] border border-[#A87146]/20 bg-white p-3.5 text-sm leading-snug">
      <Icon>{icon}</Icon>
      <span><b className="block text-[#4a3728]">{title}</b>{children}</span>
    </div>
  );
}

/* ----------------------------------------------------------------- 12. nonprofit */

function Nonprofit() {
  return (
    <Wrap testId="section-nonprofit" className="!pt-0">
      <div className="grid items-center gap-6 rounded-[14px] border border-[#A87146]/20 bg-gradient-to-br from-[#f3f8ea] to-[#f8f5f0] p-5 md:grid-cols-[1.4fr_1fr] md:p-7">
        <div className="min-w-0">
          <Kicker>Free, from the nonprofit MycoMap</Kicker>
          <H2 className="!text-[22px] sm:!text-[26px]">Free to use. Built on records MycoMap's members and projects sequenced.</H2>
          <p className="mt-1.5 text-[15px]">
            MycoMap Vision is made by MycoMap, a nonprofit. To support the work, become a member
            or donate. Questions: <a className="underline" href="mailto:info@mycomap.org">info@mycomap.org</a>
          </p>
        </div>
        <div className="flex flex-col items-start gap-2.5">
          <a href={JOIN_URL} target="_blank" rel="noreferrer"
             className="inline-flex items-center gap-1.5 rounded-md bg-myco-green px-4 py-2.5 text-sm font-medium text-white">
            Become a member or donate <ArrowUpRight className="h-4 w-4" aria-hidden />
          </a>
          <Small>at mycomap.org/join</Small>
        </div>
      </div>
    </Wrap>
  );
}
