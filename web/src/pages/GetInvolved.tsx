import { useQuery } from "@tanstack/react-query";
import {
  ArrowDown, ArrowUp, ArrowUpRight, Camera, Check, Code, Dna, Eye, FlaskConical, Heart, ListChecks,
  Moon, Package, PenLine, Search, Users,
} from "lucide-react";
import {
  Bars, C, Cap, Card, FlowArrow, GHOST_BUTTON, H2, H3, Icon, Ink, Kicker, PermissionBox,
  PRIMARY_BUTTON, Small, Sub, Tile, Wrap,
} from "@/components/Explainer";
import { api, feedbackApi, num, pct, type FeedbackSummary, type HowGood, type Stats } from "@/lib/api";
import { fetchOutline, type Outline } from "@/lib/apiOutline";
import { FEEDBACK_PATH } from "@/lib/feedback";
import {
  apiCard, depthBars, PROJECTS_URL, projectDots, reviewGroups, singleRecordNames, START_PROJECT_URL,
  VISION_KEYS_URL,
} from "@/lib/getInvolved";
import { namesByRecords } from "@/lib/howItWorks";
import { JOIN_URL, longDate, PHOTO_CHECKLIST, SEQUENCING_URL } from "@/lib/publicView";
import { VIEW_ART } from "@/components/ViewArt";
import { API_PATH } from "@/lib/research";

// Get involved (Steve, 2026-10-22: the approved redesign, open to signed-out visitors): six
// ways to help, then each one in turn. Every number is the API's, live (lib/getInvolved.ts).

/** Everything the page shows, as the API sends it; undefined while loading or on failure. */
export interface GetInvolvedData {
  stats?: Stats;
  howGood?: HowGood;
  feedback?: FeedbackSummary[];
  outline?: Outline;
  loading?: boolean;
}

export function GetInvolvedPage() {
  const stats = useQuery({ queryKey: ["stats"], queryFn: api.stats });
  const howGood = useQuery({ queryKey: ["how-good"], queryFn: api.howGood });
  const feedback = useQuery({ queryKey: ["feedback", "summary"], queryFn: feedbackApi.summary });
  const outline = useQuery({ queryKey: ["api-outline"], queryFn: () => fetchOutline(), retry: 1 });
  return (
    <GetInvolved stats={stats.data} howGood={howGood.data} feedback={feedback.data} outline={outline.data}
                 loading={stats.isLoading || howGood.isLoading || feedback.isLoading} />
  );
}

/** The page, from its data alone (so every state can be rendered in a test). */
export function GetInvolved(d: GetInvolvedData) {
  return (
    <div className="text-[#5c4a3a] leading-relaxed">
      <Hero s={d.stats} />
      <SixWays />
      <Photograph />
      <Sequence s={d.stats} h={d.howGood} loading={d.loading} />
      <Projects s={d.stats} />
      <Review feedback={d.feedback} loading={d.loading} />
      <Build outline={d.outline} />
      <Support />
    </div>
  );
}

/* ------------------------------------------------------------------ pieces */

const SecNum = ({ n }: { n: number }) => (
  <span className="mr-1.5 inline-flex h-[22px] w-[22px] items-center justify-center rounded-full bg-[#6b8f2a] align-[1px] text-[11px] font-bold tracking-normal text-white">{n}</span>
);
const NowOpen = ({ children }: { children: React.ReactNode }) => (
  <span className="ml-2 rounded bg-[#7FAE34] px-1.5 py-0.5 align-[2px] text-[10px] font-bold uppercase tracking-[.08em] text-white">{children}</span>
);
const Out = ({ href, className, children }: { href: string; className: string; children: React.ReactNode }) => (
  <a href={href} target="_blank" rel="noreferrer" className={className}>
    {children} <ArrowUpRight className="h-4 w-4" aria-hidden />
  </a>
);
const ArrowOut = () => <ArrowUpRight className="inline h-3 w-3 ml-0.5 align-baseline" aria-hidden />;
/** What the page says in place of numbers that did not come: never a made-up one. */
const Missing = ({ loading }: { loading?: boolean }) => (
  <Small className="mt-2">{loading ? "Loading the latest numbers…" : "Not available right now."}</Small>
);

/* ------------------------------------------------------------------ hero */

function Hero({ s }: { s?: Stats }) {
  const week = s?.recent_week && s.recent_week.records > 0 ? s.recent_week : null;
  return (
    <div className="bg-[#f8f5f0] border-b border-[#A87146]/10">
      <div className="mx-auto max-w-[1080px] px-4 sm:px-8 py-10">
        <span className="inline-flex items-center gap-1.5 rounded-full bg-myco-green/[.14] px-2.5 py-1 text-xs font-semibold uppercase tracking-[.06em] text-[#4f7a16]">
          <span className="h-2 w-2 shrink-0 rounded-full bg-[#7FAE34] shadow-[0_0_0_3px_hsl(82_53%_51%/.25)]" />
          Free · From the nonprofit MycoMap
        </span>
        <h1 className="font-display font-semibold text-[34px] sm:text-[44px] leading-[1.1] text-[#4a3728] mt-3.5">
          Get involved
        </h1>
        <p className="text-[17px] sm:text-xl max-w-[720px] mt-3">
          MycoMap Vision is built from community scientists' finds, sequenced and checked by
          people. Every new record makes it better. Yours can be next.
        </p>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mt-7" data-testid="hero-tiles">
          {s && <Tile value={num(s.records_north_america)} label="DNA-verified North American records so far" />}
          {s?.photographers ? <Tile value={num(s.photographers)} label="people photographed them" /> : null}
          {s?.projects ? <Tile value={num(s.projects)} label="MycoMap projects have records in the reference set" /> : null}
          {week && <Tile value={`+${num(week.records)}`}
                         label={`records DNA-verified in the week to ${longDate(week.through)}`} />}
        </div>
        <div className="flex flex-wrap gap-2.5 mt-6">
          <Out href={SEQUENCING_URL} className={PRIMARY_BUTTON}>Get a find sequenced, free</Out>
          <Out href={PROJECTS_URL} className={GHOST_BUTTON}>Find a project</Out>
          <a href={FEEDBACK_PATH} className={GHOST_BUTTON}>Browse Human Feedback</a>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ six ways */

const WAYS: { href: string; icon: React.ReactNode; title: string; text: string; who: string;
              brown?: boolean; open?: boolean }[] = [
  { href: "#photograph", icon: <Camera className="h-[18px] w-[18px]" />, title: "Take photos that identify",
    text: "Five views of every find, posted to iNaturalist.", who: "Anyone with a camera" },
  { href: "#sequence", icon: <Dna className="h-[18px] w-[18px]" />, title: "Get your find DNA-sequenced",
    text: "Free through MycoMap. It becomes a reference the next night after it's validated.", who: "Free" },
  { href: "#projects", icon: <Users className="h-[18px] w-[18px]" />, title: "Join or start a project",
    text: "Collect, sequence and validate with others near you.", who: "Clubs, parks, surveys" },
  { href: "#review", icon: <ListChecks className="h-[18px] w-[18px]" />, title: "Review in Human Feedback",
    text: "See where Vision and the reference data disagree, and how people decided.",
    who: "Browse: anyone · Decide: reviewers", open: true },
  { href: "#build", icon: <Code className="h-[18px] w-[18px]" />, title: "Research and build",
    text: "Every test in the open, and an API to identify fungi from your own app.",
    who: "Researchers and developers", brown: true },
  { href: "#support", icon: <Heart className="h-[18px] w-[18px]" />, title: "Support the nonprofit",
    text: "Vision is free. Members and donors keep it that way.", who: "Members and donors", brown: true },
];

function SixWays() {
  return (
    <Wrap testId="section-ways">
      <Kicker>Six ways to help</Kicker>
      <H2>Pick the way that fits you</H2>
      <Sub>From one good photo set to a sequenced specimen, a decided review item or a donation.</Sub>
      <div className="grid gap-2.5 sm:gap-3.5 sm:grid-cols-2 lg:grid-cols-3">
        {WAYS.map((w, i) => (
          <a key={w.href} href={w.href}
             className={`flex min-w-0 flex-col rounded-xl border bg-white p-4 sm:p-5 no-underline ${
               w.open ? "border-myco-green/55 shadow-[0_0_0_3px_hsl(82_53%_51%/.10)]" : "border-[#A87146]/20"}`}>
            <div className="mb-3 flex items-center justify-between">
              <Icon brown={w.brown}>{w.icon}</Icon>
              {w.open && <NowOpen>Now open</NowOpen>}
            </div>
            <span className="text-xs font-semibold tracking-[.06em] text-muted-foreground">{String(i + 1).padStart(2, "0")}</span>
            <H3>{w.title}</H3>
            <p className="flex-1 text-sm leading-normal">{w.text}</p>
            <span className={`mt-3.5 self-start rounded-full px-2.5 py-0.5 text-xs font-semibold ${
              w.brown ? "bg-[#A36848]/[.12] text-[#7a4a2a]" : "bg-myco-green/[.12] text-[#4f7a16]"}`}>{w.who}</span>
          </a>
        ))}
      </div>
    </Wrap>
  );
}

/* ------------------------------------------------------------------ 1. photograph */

function StripStep({ icon, title, children }: { icon: React.ReactNode; title: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="flex h-full min-w-0 items-center gap-3 rounded-[10px] border border-[#A87146]/20 bg-white px-3.5 py-3 text-sm leading-snug">
      <Icon>{icon}</Icon>
      <span><b className="block text-[#4a3728]">{title}</b>{children}</span>
    </div>
  );
}

function Photograph() {
  return (
    <Wrap band id="photograph" testId="section-photograph">
      <Kicker><SecNum n={1} />Photograph</Kicker>
      <H2>Five photos that identify</H2>
      <Sub>A good set helps you, Vision and everyone who checks your find later. Photograph each find from every side.</Sub>
      <div className="grid grid-cols-2 gap-2 min-[560px]:grid-cols-3 min-[560px]:gap-3 min-[900px]:grid-cols-5" data-testid="photo-checklist">
        {PHOTO_CHECKLIST.map((c, i) => {
          const last = i === PHOTO_CHECKLIST.length - 1;
          return (
            <div key={c.what} className={`min-w-0 overflow-hidden rounded-[10px] border border-[#A87146]/20 bg-white ${last ? "col-span-2 min-[560px]:col-span-1" : ""}`}>
              <div className={`relative flex items-center justify-center bg-[#efe9e1] ${last ? "aspect-[8/3] min-[560px]:aspect-[4/3]" : "aspect-[4/3]"}`} aria-hidden>
                <span className="absolute left-2 top-2 flex h-[22px] w-[22px] items-center justify-center rounded-full bg-white text-xs font-bold text-[#4a3728] shadow-sm">{i + 1}</span>
                {VIEW_ART[i]}
              </div>
              <div className="px-3 pb-3 pt-2.5 text-[13px] leading-snug">
                <b className="mb-0.5 block text-sm text-[#4a3728]">{c.what}</b>{c.why}
              </div>
            </div>
          );
        })}
      </div>
      <div className="mt-[22px] grid items-center md:grid-cols-[1fr_32px_1fr_32px_1fr]">
        <StripStep icon={<Camera className="h-[18px] w-[18px]" />} title="Photograph every side">the five views above</StripStep>
        <FlowArrow small />
        <StripStep icon={<ArrowUp className="h-[18px] w-[18px]" />}
                   title={<>Post them to <a className="underline" href="https://www.inaturalist.org" target="_blank" rel="noreferrer">iNaturalist<ArrowOut /></a></>}>
          the photos of MycoMap's records come from there
        </StripStep>
        <FlowArrow small />
        <StripStep icon={<Search className="h-[18px] w-[18px]" />}
                   title={<><a className="underline" href="/">Identify your find</a> here</>}>
          from all of them at once
        </StripStep>
      </div>
      <PermissionBox kind="learn" className="mt-[18px]" />
    </Wrap>
  );
}

/* ------------------------------------------------------------------ 2. sequence */

/** mycomap.org's free sequencing, as its network page sets it out (static copy; the results
 *  are said without naming where they appear). */
const ORG_STEPS: { icon: React.ReactNode; title: string; text: string }[] = [
  { icon: <Users className="h-5 w-5" />, title: "Join your regional project", text: "MycoMap's iNaturalist project for your region" },
  { icon: <Camera className="h-5 w-5" />, title: "Collect and document", text: "photos, plus a field data slip with its unique ID" },
  { icon: <Package className="h-5 w-5" />, title: "Dry and ship", text: "dried at 140°F or below, mailed to the Mycota Lab in Michigan" },
  { icon: <Dna className="h-5 w-5" />, title: "Get results", text: "DNA sequences in 3–4 months" },
];
const VISION_STEPS: { icon: React.ReactNode; step: string; title: string; text: string }[] = [
  { icon: <Check className="h-5 w-5" strokeWidth={2.4} />, step: "Validated", title: "Its DNA name, marked green", text: "in a MycoMap project" },
  { icon: <Moon className="h-5 w-5" />, step: "The next night", title: "Joins the reference set", text: "with your photos" },
  { icon: <Search className="h-5 w-5" />, step: "From then on", title: "Names the next find like it", text: "for everyone who identifies here" },
];

function Lane({ org, pill, note, cols, children }: {
  org?: boolean; pill: string; note: string; cols: 3 | 4; children: React.ReactNode;
}) {
  return (
    <div className="min-w-0 rounded-xl border border-[#A87146]/20 bg-white px-[18px] pb-5 pt-[18px]">
      <div className="mb-4 flex flex-wrap items-center gap-2.5 text-xs font-semibold uppercase tracking-[.08em]">
        <span className={`whitespace-nowrap rounded-full px-2.5 py-0.5 ${org ? "bg-[#A36848]/[.12] text-[#7a4a2a]" : "bg-myco-green/[.14] text-[#4f7a16]"}`}>{pill}</span>
        <span className="text-[13px] font-medium normal-case tracking-normal text-muted-foreground">{note}</span>
      </div>
      <div className={`relative grid gap-y-3.5 ${cols === 4 ? "md:grid-cols-4" : "md:grid-cols-3"}`}>
        <div className={`absolute top-6 hidden h-0.5 md:block ${cols === 4 ? "left-[12.5%] right-[12.5%] bg-[#e2cdbd]" : "left-[16.6%] right-[16.6%] bg-[#cfe3ad]"}`} aria-hidden />
        {children}
      </div>
    </div>
  );
}

function LaneStep({ org, end, icon, step, title, text }: {
  org?: boolean; end?: boolean; icon: React.ReactNode; step: string; title: string; text: string;
}) {
  return (
    <div className="relative grid grid-cols-[44px_1fr] gap-x-3 md:block md:px-2.5 md:text-center">
      <div className={`relative z-[1] row-span-3 flex h-10 w-10 items-center justify-center rounded-full border-2 md:mx-auto md:mb-2.5 md:h-12 md:w-12 ${
        org ? "border-[#b07b58] bg-white text-[#7a4a2a]" : end ? "text-white" : "bg-white text-[#4f7a16]"}`}
           style={org ? undefined : { borderColor: C.s1, background: end ? C.s1 : undefined }}>{icon}</div>
      <span className="text-[11px] font-bold uppercase tracking-[.08em] text-muted-foreground">{step}</span>
      <b className="my-0.5 block text-[15px] leading-tight text-[#4a3728]">{title}</b>
      <span className="block text-[13px] leading-snug">{text}</span>
    </div>
  );
}

function Sequence({ s, h, loading }: { s?: Stats; h?: HowGood; loading?: boolean }) {
  const one = singleRecordNames(s);
  return (
    <Wrap id="sequence" testId="section-sequence">
      <Kicker><SecNum n={2} />Sequence</Kicker>
      <H2>Your find, sequenced free. Then it teaches Vision.</H2>
      <Sub>
        MycoMap offers <a className="underline" href={SEQUENCING_URL} target="_blank" rel="noreferrer">free DNA
        sequencing<ArrowOut /></a>. Once your find's DNA name is validated in a MycoMap project, it
        joins Vision's reference set the next night, with your photos.
      </Sub>

      <div className="flex flex-col gap-3.5" data-testid="sequence-lanes">
        <Lane org pill="On mycomap.org" note="Free sequencing, in four steps" cols={4}>
          {ORG_STEPS.map((st, i) => <LaneStep key={st.title} org icon={st.icon} step={`Step ${i + 1}`} title={st.title} text={st.text} />)}
        </Lane>
        <div className="-my-1 flex items-center justify-center gap-2 text-xs text-muted-foreground">
          <ArrowDown className="h-[18px] w-[18px] text-[#b9a48f]" aria-hidden />then, in MycoMap Vision
        </div>
        <Lane pill="In MycoMap Vision" note="Your sequenced find becomes a reference" cols={3}>
          {VISION_STEPS.map((st, i) => <LaneStep key={st.title} end={i === VISION_STEPS.length - 1} {...st} />)}
        </Lane>
      </div>

      <div className="mt-10 grid gap-8 md:grid-cols-2 items-start">
        <NamesCard s={s} loading={loading} />
        <DepthCard h={h} loading={loading} />
      </div>

      {one && (
        <div className="mt-5 grid items-center gap-x-6 gap-y-3 rounded-[14px] border border-[#A87146]/20 bg-white p-5 md:grid-cols-[auto_1fr_auto] md:p-7"
             data-testid="one-record-banner">
          <div className="text-[44px] font-semibold leading-none text-[#4a3728] tabular-nums">{pct(one.share)}</div>
          <div className="min-w-0">
            <b className="text-[17px] text-[#4a3728]">of the names have just one DNA-verified record.</b><br />
            <span className="text-[15px]">
              {num(one.n)} of {num(one.names)} names. Your sequenced find could be its second.
              Names with more records are named right more often.
            </span>
          </div>
          <Out href={SEQUENCING_URL} className={`${PRIMARY_BUTTON} justify-self-start`}>How to get a find sequenced</Out>
        </div>
      )}
    </Wrap>
  );
}

const TH = "px-2.5 py-1 border-b border-[#ece6de]";

function NamesCard({ s, loading }: { s?: Stats; loading?: boolean }) {
  const rows = namesByRecords(s);
  const top = rows ? Math.max(...rows.map((r) => r.n)) : 1;
  return (
    <Card className="!rounded-xl">
      <H3>Where one more record helps most</H3>
      <Small className="mb-4">Names in the reference set, by how many DNA-verified records each has.</Small>
      {!rows || !s ? <Missing loading={loading} /> : (
        <>
          <Bars labelWidth={92} testId="chart-names-by-records" rows={rows.map((r) => {
            const one = r.key === "1";
            return {
              label: one ? <Ink>{r.label}</Ink> : r.label, frac: r.n / top, text: num(r.n),
              color: one ? C.s1 : C.rest,
              title: `${one ? r.label : `${r.label} records`}: ${num(r.n)} names · ${pct(r.share)}${one ? ` of ${num(s.names)}` : ""}`,
            };
          })} />
          <details className="mt-3.5 text-[13px]">
            <summary className="cursor-pointer font-medium text-[#4a3728]">Show as a table</summary>
            <div className="overflow-x-auto">
              <table className="mt-2 border-collapse tabular-nums" data-testid="table-names-by-records">
                <thead>
                  <tr>
                    <th className={`${TH} text-left`}>Records per name</th>
                    <th className={`${TH} text-right`}>Names</th>
                    <th className={`${TH} text-right`}>Share</th>
                  </tr>
                </thead>
                <tbody>
                  {[...rows.map((r) => ({ k: r.key.replace("-", "–"), n: r.n, share: r.share })),
                    { k: "All", n: s.names, share: 1 }].map((r) => (
                    <tr key={r.k}>
                      <td className={`${TH} text-left`}>{r.k}</td>
                      <td className={`${TH} text-right`}>{num(r.n)}</td>
                      <td className={`${TH} text-right`}>{pct(r.share)}</td>
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

function DepthCard({ h, loading }: { h?: HowGood; loading?: boolean }) {
  const rows = depthBars(h);
  const run = h?.run;
  return (
    <Card className="!rounded-xl">
      <H3>More records, more right answers</H3>
      <Small className="mb-4">Species named right first time, by how many DNA-verified records Vision has of that species.</Small>
      {!rows || !run ? <Missing loading={loading} /> : (
        <>
          <Bars labelWidth={92} testId="chart-by-depth" rows={rows.map((r) => ({
            label: r.none ? <>{r.label} <small className="text-[11px] text-muted-foreground">none yet</small></> : r.label,
            frac: r.top1, text: pct(r.top1), color: C.ramp[1],
            title: r.none
              ? `No reference yet: ${pct(r.top1, 1)} · ${num(r.n)} test records: a species with no DNA-verified reference can't be named`
              : `${r.label} records: ${pct(r.top1, 1)} right first time · ${num(r.n)} test records`,
          }))} />
          <div className="mr-[4.5rem] ml-[104px] mt-1.5 flex justify-between text-[11px] text-muted-foreground" aria-hidden>
            <span>0%</span><span>50%</span><span>100%</span>
          </div>
          <Cap>
            Tested on the newest {num(run.n_test)} finds (validated after {longDate(run.cutoff)}), each
            identified from the records verified before it{run.method.endsWith("+place") ? ", with its photos and its public place and date" : ", from its photos alone"}.{" "}
            <a className="underline" href="/about#how-good">More in How it works</a>.
          </Cap>
        </>
      )}
    </Card>
  );
}

/* ------------------------------------------------------------------ 3. projects */

const CYCLE: [string, React.ReactNode][] = [
  ["Collect", <Camera key="c" className="h-3.5 w-3.5" />],
  ["Sequence", <Dna key="s" className="h-3.5 w-3.5" />],
  ["Validate", <Check key="v" className="h-3.5 w-3.5" strokeWidth={2.6} />],
];

function Projects({ s }: { s?: Stats }) {
  const grid = projectDots(s?.projects);
  return (
    <Wrap band id="projects" testId="section-projects">
      <div className={`grid gap-8 items-center ${grid ? "md:grid-cols-2" : ""}`}>
        <div className="min-w-0">
          <Kicker><SecNum n={3} />Projects</Kicker>
          <H2>Join or start a MycoMap project</H2>
          <p className="text-base sm:text-[17px]">
            Projects collect, sequence and validate the records Vision learns from. Find one near
            you, or start one for your club, park or survey.
          </p>
          <div className="mt-4 flex flex-wrap items-center gap-2" data-testid="project-cycle">
            {CYCLE.map(([label, icon], i) => (
              <span key={label} className="inline-flex items-center gap-2">
                {i > 0 && <span className="text-[#b9a48f]" aria-hidden>→</span>}
                <span className="inline-flex items-center gap-2 rounded-full border border-[#A87146]/20 bg-white py-1.5 pl-2 pr-3.5 text-sm font-medium text-[#4a3728]">
                  <i className="inline-flex h-[26px] w-[26px] items-center justify-center rounded-full bg-myco-green/[.14] text-[#4f7a16]">{icon}</i>
                  {label}
                </span>
              </span>
            ))}
          </div>
          <div className="mt-[18px] flex flex-wrap gap-2.5">
            <Out href={PROJECTS_URL} className={PRIMARY_BUTTON}>Find a project</Out>
            <Out href={START_PROJECT_URL} className={GHOST_BUTTON}>Start a project</Out>
          </div>
        </div>
        {grid && s?.projects ? (
          <div className="min-w-0 rounded-[14px] border border-[#A87146]/20 bg-gradient-to-br from-white to-[#f6f9ef] p-5 sm:p-6"
               data-testid="project-dots">
            <div className="text-[44px] sm:text-[56px] font-semibold leading-none text-[#4a3728] tabular-nums">{num(s.projects)}</div>
            <div className="mt-1 text-[15px] text-[#4a3728]">projects have records in the reference set so far</div>
            <div className="mt-4 grid gap-[3px]" style={{ gridTemplateColumns: "repeat(44, minmax(0, 1fr))" }}
                 role="img" aria-label={`${num(s.projects)} projects`}>
              {Array.from({ length: grid.dots }, (_, i) => (
                <i key={i} className="aspect-square rounded-full opacity-85" style={{ background: C.s1 }} />
              ))}
            </div>
            <Cap>{grid.per === 1 ? "One dot per project." : `One dot per ${num(grid.per)} projects.`}</Cap>
          </div>
        ) : null}
      </div>
    </Wrap>
  );
}

/* ------------------------------------------------------------------ 4. review */

function Review({ feedback, loading }: { feedback?: FeedbackSummary[]; loading?: boolean }) {
  const groups = reviewGroups(feedback);
  return (
    <Wrap id="review" testId="section-review">
      <Kicker><SecNum n={4} />Review <NowOpen>Now open to everyone</NowOpen></Kicker>
      <H2>See how people check Vision's work</H2>
      <Sub>
        Human Feedback lists the places where Vision and the reference data disagree. People decide
        each one, and the decision takes effect where it belongs.
      </Sub>
      <div className="grid gap-3 md:grid-cols-2">
        <WhoCard icon={<Eye className="h-[18px] w-[18px]" />} title="Browse: anyone"
                 items={["Every section, queue and the Summary", "Each item's photos, record and outcome",
                         "Read-only: look as much as you like"]} />
        <WhoCard brown icon={<PenLine className="h-[18px] w-[18px]" />} title="Decide: reviewers"
                 items={["mycomap.org admins, project leaders and contributors",
                         "Contributors review their own projects' items", "Every decision is recorded with its date"]} />
      </div>
      {groups ? (
        <div className="mt-3.5 grid gap-3 md:grid-cols-3" data-testid="review-queues">
          {groups.map((g) => (
            <div key={g.code} className="min-w-0 rounded-xl border border-[#A87146]/20 bg-[#fbf9f6] p-4" data-testid={`queue-group-${g.code}`}>
              <div className="flex items-baseline justify-between gap-2">
                <b className="text-[15px] text-[#4a3728]">{g.title}</b>
                <span className="whitespace-nowrap text-xs text-muted-foreground">{g.queues} {g.queues === 1 ? "queue" : "queues"}</span>
              </div>
              <div className="mt-2 text-[28px] font-semibold leading-tight text-[#4a3728] tabular-nums">
                {num(g.open)} <small className="text-[13px] font-medium text-muted-foreground">open to browse</small>
              </div>
              <ul className="mt-2.5 text-[13px] leading-snug">
                {g.lines.map((l) => <li key={l} className="border-t border-[#ece6de] py-[5px]">{l}</li>)}
                {g.more > 0 && <li className="border-t border-[#ece6de] py-[5px] text-muted-foreground">+ {g.more} more</li>}
              </ul>
            </div>
          ))}
        </div>
      ) : <Missing loading={loading} />}
      <div className="mt-3.5 flex flex-wrap items-center gap-2 text-[13px]">
        <span className="text-muted-foreground">Each decision takes effect</span>
        {["on mycomap.org", "in Vision", "or as a note"].map((a) => (
          <span key={a} className="inline-flex items-center rounded-md border border-[#A87146]/20 bg-white px-2.5 py-0.5 text-[#4a3728]">{a}</span>
        ))}
      </div>
      <div className="mt-[18px] flex flex-wrap items-center gap-2.5">
        <a href={FEEDBACK_PATH} className={PRIMARY_BUTTON}>Browse Human Feedback</a>
        <span className="text-[13px] text-muted-foreground">No sign-in needed to look.</span>
      </div>
    </Wrap>
  );
}

function WhoCard({ icon, title, items, brown }: { icon: React.ReactNode; title: string; items: string[]; brown?: boolean }) {
  return (
    <div className="min-w-0 rounded-xl border border-[#A87146]/20 border-t-4 bg-white p-[18px]"
         style={{ borderTopColor: brown ? C.s2 : C.s1 }}>
      <div className="mb-2 flex items-center gap-2.5"><Icon brown={brown}>{icon}</Icon><H3 className="!mb-0">{title}</H3></div>
      <ul className="mt-2 text-sm leading-snug">
        {items.map((t) => (
          <li key={t} className="my-[5px] flex gap-2">
            <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-[#c9b9a6]" aria-hidden />{t}
          </li>
        ))}
      </ul>
    </div>
  );
}

/* ------------------------------------------------------------------ 5. research and build */

const RESEARCH_LINKS: { href: string; title: string; text: string }[] = [
  { href: "/research/results", title: "Results", text: "every approach, scored on the same records" },
  { href: "/research/benchmarks", title: "Benchmarks", text: "the named test sets" },
  { href: "/research/protocols", title: "Evaluation protocols", text: "how every result is scored" },
  { href: "/research/data", title: "Reference data", text: "the records and photos it learns from" },
];

function Build({ outline }: { outline?: Outline }) {
  const card = apiCard(outline);
  return (
    <Wrap band id="build" testId="section-build">
      <Kicker><SecNum n={5} />Research and build</Kicker>
      <H2>Test it, cite it, build on it</H2>
      <div className="mt-[18px] grid gap-8 md:grid-cols-2 items-start">
        <Card>
          <div className="mb-2.5"><Icon><FlaskConical className="h-[18px] w-[18px]" /></Icon></div>
          <H3>For researchers</H3>
          <p className="text-sm">How each model is tested, the results and the reference data, all in the Research section.</p>
          <div className="mt-3 grid gap-2 sm:grid-cols-2">
            {RESEARCH_LINKS.map((l) => (
              <a key={l.href} href={l.href} className="block rounded-lg border border-[#A87146]/20 bg-white px-3 py-2.5 text-[13px] leading-snug no-underline">
                <b className="block text-sm text-[#4a3728]">{l.title}</b>{l.text}
              </a>
            ))}
          </div>
          <Cap>
            <a className="underline" href="/research/for-researchers">For researchers</a>: how to cite, and how
            to score your own model on the same records. Questions or collaborations:{" "}
            <a className="underline" href="mailto:info@mycomap.org">info@mycomap.org</a>
          </Cap>
        </Card>
        <Card>
          <div className="mb-2.5"><Icon brown><Code className="h-[18px] w-[18px]" /></Icon></div>
          <H3>For developers: the MycoMap Vision API</H3>
          <p className="text-sm" data-testid="api-endpoints">
            {card && card.available > 0
              ? <><Ink>{num(card.available)} {card.available === 1 ? "endpoint" : "endpoints"}</Ink> available now. </>
              : null}
            Identify fungi from your own app or site.
          </p>
          {card && card.highlights.length > 0 && (
            <div className="mt-3.5 flex flex-col gap-2 text-sm">
              {card.highlights.map((e) => (
                <div key={e.text} className="flex items-start gap-2.5">
                  <code className={`mt-0.5 shrink-0 rounded px-1.5 py-0.5 text-[11px] font-bold ${
                    e.method === "GET" ? "bg-[#e0f2fe] text-[#075985]" : "bg-[#d1fae5] text-[#065f46]"}`}>{e.method}</code>
                  <span>{e.text}</span>
                </div>
              ))}
            </div>
          )}
          {card?.identifyUrl && (
            <pre className="mt-3.5 overflow-x-auto whitespace-pre rounded-lg bg-[#2f241b] p-3.5 text-[12.5px] leading-normal text-[#f8f5f0]"
                 data-testid="api-example">
              <span className="text-[#b6dc76]">POST</span> {card.identifyUrl}{"\n"}Authorization: Bearer vision_…   <span className="text-[#c2ab92]"># your Vision token</span>
            </pre>
          )}
          <div className="mt-3.5 flex flex-wrap gap-2.5">
            <a href={API_PATH} className={PRIMARY_BUTTON}>Read the API docs</a>
            <Out href={VISION_KEYS_URL} className={GHOST_BUTTON}>Request a Vision token</Out>
          </div>
        </Card>
      </div>
    </Wrap>
  );
}

/* ------------------------------------------------------------------ 6. support */

function Support() {
  return (
    <Wrap id="support" testId="section-support">
      <div className="grid items-center gap-6 rounded-[14px] border border-[#A87146]/20 bg-gradient-to-br from-[#f3f8ea] to-[#f8f5f0] p-5 md:grid-cols-[1.4fr_1fr] md:p-7">
        <div className="min-w-0">
          <Kicker><SecNum n={6} />Free, from the nonprofit MycoMap</Kicker>
          <H2 className="!text-[22px] sm:!text-[26px]">Vision is free. Help keep it that way.</H2>
          <p className="mt-1.5 text-[15px]">
            MycoMap Vision is made by MycoMap, a nonprofit. To support the work, become a member
            or donate. Questions: <a className="underline" href="mailto:info@mycomap.org">info@mycomap.org</a>
          </p>
        </div>
        <div className="flex flex-col items-start gap-2.5">
          <Out href={JOIN_URL} className={PRIMARY_BUTTON}>Become a member or donate</Out>
          <Small>at mycomap.org/join</Small>
        </div>
      </div>
    </Wrap>
  );
}
