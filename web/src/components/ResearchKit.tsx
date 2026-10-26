import { useEffect, useRef, useState } from "react";
import { Link, useLocation } from "wouter";
import { useQuery } from "@tanstack/react-query";
import { ArrowRight, ArrowUpRight, Check, Copy, Layers, Lock, Search } from "lucide-react";
import { api, type Me } from "@/lib/api";
import { isBarTabActive, researchBar, STATUS_LABEL, type BenchmarkStatus, type Citation,
         type ProtocolStep } from "@/lib/research";
import { activeSection, axisAt, BAND_COLOR, heatStep, HEAT_STEPS, KS, nf, pt, topkAxis, topkTip,
         type HeatRow, type K, type TopKRow } from "@/lib/researchView";
import { STATUS_WORDS, type ExperimentStatus } from "@/lib/experiments";
import { Tile } from "@/components/Explainer";

// The Research section's shared pieces (Steve, 2026-10-25: the approved redesign). Section
// bands, headings, cards and tiles are Explainer's; these add the Research bar, the hero, the
// sticky in-page nav, source lines, status pills, the filter bar, the one top-1/3/5/10 chart,
// the depth heatmap and bars, the paired-difference mark, citation cards and copy buttons.
// Styles: research.css, scoped to the .rs wrapper ResearchShell puts round each page.

/** Every Research page sits in this: the .rs scope, the Research bar, and one hover tooltip for
 *  any mark carrying data-tip="Title|detail" (text set with textContent, never as HTML). */
export function ResearchShell({ me, children }: { me?: Me | null; children: React.ReactNode }) {
  const tip = useRef<HTMLDivElement>(null);
  const move = (e: React.PointerEvent) => {
    const el = (e.target as HTMLElement).closest?.("[data-tip]") as HTMLElement | null;
    const box = tip.current;
    if (!box) return;
    if (!el) { box.style.display = "none"; return; }
    const [title, detail] = (el.getAttribute("data-tip") ?? "").split("|");
    (box.firstChild as HTMLElement).textContent = title;
    (box.lastChild as HTMLElement).textContent = detail ?? "";
    box.style.display = "block";
    const x = Math.min(e.clientX + 14, window.innerWidth - box.offsetWidth - 8);
    box.style.left = `${Math.max(8, x)}px`;
    box.style.top = `${e.clientY - 52}px`;
  };
  return (
    <div className="rs" onPointerMove={move} onPointerLeave={() => { if (tip.current) tip.current.style.display = "none"; }}>
      <ResearchBar me={me} />
      {children}
      <div className="ctip" ref={tip} role="tooltip"><b /><span /></div>
    </div>
  );
}

/** The bar under the header: the overview and every part this visitor may open. */
export function ResearchBar({ me }: { me?: Me | null }) {
  const [location] = useLocation();
  const q = useQuery({ queryKey: ["me"], queryFn: api.me, retry: false, enabled: me === undefined });
  const who = me === undefined ? q.data : me;
  return (
    <div className="rbar" data-testid="research-bar">
      <div className="mx-auto flex h-[46px] max-w-[1280px] items-center gap-[10px] px-4 sm:gap-[18px] sm:px-8">
        <span className="rt"><i><Layers className="h-[13px] w-[13px]" /></i><span>Research</span></span>
        <nav className="rtabs" aria-label="Research">
          {researchBar(who).map((n) => (
            <Link key={n.href} href={n.href} className={isBarTabActive(n.href, location) ? "on" : ""}
                  aria-current={isBarTabActive(n.href, location) ? "page" : undefined}>
              {n.label}
              {n.members && <span className="lk" aria-label="members only"><Lock className="h-3 w-3" /></span>}
            </Link>
          ))}
        </nav>
      </div>
    </div>
  );
}

/** A Research page's head band: breadcrumbs, title, lede, then tiles or anything else. */
export function ResearchHero({ crumbs, title, lede, top, children, testId }: {
  crumbs?: { href?: string; label: string }[]; title: string; lede?: React.ReactNode;
  top?: React.ReactNode; children?: React.ReactNode; testId?: string;
}) {
  return (
    <div className="hero r" data-testid={testId ?? "research-hero"}>
      <div className="inner">
        {crumbs && (
          <div className="crumbs">
            {crumbs.map((c, i) => (
              <span key={i}>
                {i > 0 && " › "}
                {c.href ? <Link href={c.href}>{c.label}</Link> : c.label}
              </span>
            ))}
          </div>
        )}
        {top}
        <h1 className="font-display font-semibold" style={top ? { marginTop: 12 } : undefined}>{title}</h1>
        {lede && <p className="lede">{lede}</p>}
        {children}
      </div>
    </div>
  );
}

/** Four (or fewer) tiles in the hero. */
export function Tiles({ children, testId }: { children: React.ReactNode; testId?: string }) {
  return <div className="tiles" data-testid={testId}>{children}</div>;
}
export { Tile };

/** The line under every number block: where it comes from. The first part is labelled. */
export function SourceLine({ parts, label = "Source", testId }: {
  parts: React.ReactNode[]; label?: string; testId?: string;
}) {
  const shown = parts.filter((p) => p != null && p !== false && p !== "");
  if (!shown.length) return null;
  return (
    <div className="src" data-testid={testId ?? "source-line"}>
      {shown.map((p, i) => <span key={i}>{i === 0 && <b>{label} </b>}{p}</span>)}
    </div>
  );
}

export type PillStatus = BenchmarkStatus | ExperimentStatus | "members";
const PILL_LABEL: Record<PillStatus, string> = {
  ...STATUS_LABEL, adopted: STATUS_WORDS.adopted.label, done: STATUS_WORDS.done.label,
  dropped: STATUS_WORDS.dropped.label, members: "Members",
};
/** A status, as a pill: dot and word, a lock for sealed and members-only. */
export function StatusPill({ status, label }: { status: PillStatus; label?: string }) {
  const locked = status === "sealed" || status === "members";
  return (
    <span className={`pill ${status}`} data-testid={`pill-${status}`}>
      {locked ? <Lock className="h-[11px] w-[11px]" aria-hidden /> : <i />}
      {label ?? PILL_LABEL[status]}
    </span>
  );
}

/** Interim and development labels on a number that may still move. */
export const Interim = ({ children = "Interim" }: { children?: React.ReactNode }) =>
  <span className="interim" data-testid="label-interim">{children}</span>;
export const DevLabel = ({ children = "Development screen" }: { children?: React.ReactNode }) =>
  <span className="devlab" data-testid="label-dev">{children}</span>;

export interface PageSection { id: string; label: string }

/** The sticky in-page nav under the Research bar (Results, detail pages): numbered links to
 *  the page's sections, the one being read marked. */
export function InPageNav({ sections }: { sections: PageSection[] }) {
  const [cur, setCur] = useState(0);
  useEffect(() => {
    const on = () => setCur(activeSection(sections.map((s) => {
      const el = document.getElementById(s.id);
      return el ? el.getBoundingClientRect().top + window.scrollY : null;
    }), window.scrollY));
    on();
    window.addEventListener("scroll", on, { passive: true });
    return () => window.removeEventListener("scroll", on);
  }, [sections]);
  return (
    <nav className="pnav" aria-label="On this page" data-testid="in-page-nav">
      <div className="inner">
        {sections.map((s, i) => (
          <a key={s.id} href={`#${s.id}`} className={i === cur ? "on" : ""} aria-current={i === cur ? "true" : undefined}>
            <i>{i + 1}</i>{s.label}
          </a>
        ))}
      </div>
    </nav>
  );
}

/** A Research section: Explainer's band, its kicker numbered when the page has an in-page nav. */
export function Section({ id, band, kicker, title, lead, children, testId }: {
  id?: string; band?: boolean; kicker?: string; title?: React.ReactNode; lead?: React.ReactNode;
  children?: React.ReactNode; testId?: string;
}) {
  return (
    <section id={id} data-testid={testId ?? (id ? `section-${id}` : undefined)}
             className={`scroll-mt-[160px] ${band ? "bg-[#fbf9f6] border-y border-[#A87146]/[.12]" : ""}`}>
      <div className="mx-auto max-w-[1080px] px-4 sm:px-8 py-[30px] sm:py-10">
        {kicker && <p className="text-xs font-semibold uppercase tracking-[.08em] text-[#6b8f2a] mb-1.5">{kicker}</p>}
        {title && <h2 className="font-display font-semibold text-[25px] sm:text-[30px] leading-tight text-[#4a3728] mb-2">{title}</h2>}
        {lead && <p className="lead2">{lead}</p>}
        {children}
      </div>
    </section>
  );
}

export function ChartCard({ children, className = "", testId }: { children: React.ReactNode; className?: string; testId?: string }) {
  return <div className={`chartcard min-w-0 ${className}`} data-testid={testId}>{children}</div>;
}

// --- the one top-1/3/5/10 chart -------------------------------------------------------------

const K_COLOR: Record<K, string> = { 1: "var(--k1)", 3: "var(--k3)", 5: "var(--k5)", 10: "var(--k10)" };

export function KLegend({ extra }: { extra?: React.ReactNode }) {
  return (
    <div className="klegend">
      {KS.map((k) => <span key={k}><i style={{ background: K_COLOR[k] }} />Top {k}</span>)}
      {extra}
    </div>
  );
}

/** Top 1, 3, 5 and 10 for each row: a dot-range on one track per row with the four values
 *  written beside it ("range", the default), or four small bars ("mini", for cards). Every
 *  value is printed, so no colour stands alone. */
export function TopKChart({ rows, variant = "range", labelHead = "", trackHead = "", legend = true,
                            labelWidth, numsWidth, testId = "chart-topk" }: {
  rows: TopKRow[]; variant?: "range" | "mini"; labelHead?: string; trackHead?: string; legend?: boolean;
  labelWidth?: number; numsWidth?: number; testId?: string;
}) {
  if (!rows.length) return <p className="small" data-testid={`${testId}-empty`}>Not measured yet.</p>;
  if (variant === "mini") {
    const r = rows[0];
    return (
      <div className="mk" data-testid={testId}>
        {KS.map((k) => (
          <div key={k} data-tip={topkTip(r, k)}>
            <span>{pt(r.values[k])}</span>
            <b style={{ height: `${Math.max(2, 46 * (r.values[k] ?? 0))}px`, background: K_COLOR[k] }} />
            <em>{k}</em>
          </div>
        ))}
      </div>
    );
  }
  const axis = topkAxis(rows);
  const style = { ...(labelWidth ? { "--dlw": `${labelWidth}px` } : {}), ...(numsWidth ? { "--dnw": `${numsWidth}px` } : {}) } as React.CSSProperties;
  return (
    <div data-testid={testId}>
      {legend && <KLegend />}
      <div className="dr" style={style}>
        <div className="drh"><div>{labelHead}</div><div>{trackHead}</div>
          <div className="nums"><span>Top 1</span><span>3</span><span>5</span><span>10</span></div></div>
        {rows.map((r) => {
          const vals = KS.map((k) => r.values[k]).filter((v): v is number => v != null);
          const lo = vals.length ? axisAt(Math.min(...vals), axis) : 0;
          const hi = vals.length ? axisAt(Math.max(...vals), axis) : 0;
          return (
            <div key={r.key} className={`drr ${r.highlight ? "hl" : ""} ${r.outside ? "out" : ""}`} data-testid="topk-row">
              <div className="nm">{r.name}{r.detail && <small>{r.detail}</small>}</div>
              <div className="trk">
                {axis.ticks.map((t) => <i key={t} className="gl" style={{ left: `${axisAt(t, axis)}%` }} />)}
                {vals.length > 0 && <i className="ln" style={{ left: `${lo}%`, width: `${hi - lo}%` }} />}
                {[...KS].reverse().map((k) => r.values[k] != null && (
                  <i key={k} className="d" style={{ left: `${axisAt(r.values[k]!, axis)}%`, background: K_COLOR[k] }}
                     data-tip={topkTip(r, k)} />
                ))}
              </div>
              <div className="nums">
                {KS.map((k) => <span key={k} data-k={`${k}:`}>{pt(r.values[k])}</span>)}
              </div>
            </div>
          );
        })}
        <div className="draxis"><div /><div className="ax">
          {axis.ticks.map((t) => <span key={t} style={{ left: `${axisAt(t, axis)}%` }}>{Math.round(100 * t)}%</span>)}
        </div><div /></div>
      </div>
    </div>
  );
}

/** Species top 1 by depth band, as five small bars (cards). */
export function BandBars({ bands, testId = "chart-bands" }: {
  bands: { band: string; top1: number; n?: number }[]; testId?: string;
}) {
  return (
    <div className="mk bands" data-testid={testId}>
      {bands.map((b) => (
        <div key={b.band} data-tip={`Species top 1, true species with ${b.band} records|${pt(b.top1)}%${b.n ? ` of ${nf(b.n)} finds` : ""}`}>
          <span>{pt(b.top1)}</span>
          <b style={{ height: `${Math.max(2, 46 * b.top1)}px`, background: BAND_COLOR[b.band]?.bg ?? "#3987e5" }} />
          <em>{b.band}</em>
        </div>
      ))}
    </div>
  );
}

/** Two 100% bars on the depth bands: the reference's species, and the test finds. */
export function DepthShareBars({ species, finds, speciesTotal }: {
  species: { band: string; n: number }[] | null; finds: { band: string; n: number }[]; speciesTotal?: number;
}) {
  const bar = (label: string, segs: { band: string; n: number }[], unit: string) => {
    const total = segs.reduce((s, x) => s + x.n, 0) || 1;
    const shown = segs.filter((s) => s.n > 0);
    return (
      <div data-testid={`depth-bar-${unit}`}>
        <p className="mb-1 mt-3 text-[13px] font-semibold text-[#4a3728] first:mt-0">{label}</p>
        <div className="flex h-[26px] gap-[2px]" role="img"
             aria-label={`${label}: ${shown.map((s) => `${s.band} records ${Math.round((100 * s.n) / total)}%`).join(", ")}`}>
          {shown.map((s, i) => {
            const share = Math.round((100 * s.n) / total);
            return (
              <div key={s.band} data-tip={`${s.band} records|${nf(s.n)} ${unit} (${share}%)`}
                   className={`flex min-w-[2px] items-center justify-center text-[11px] font-semibold ${i === 0 ? "rounded-l" : ""} ${i === shown.length - 1 ? "rounded-r" : ""}`}
                   style={{ flex: s.n, background: BAND_COLOR[s.band].bg, color: BAND_COLOR[s.band].ink }}>
                {share >= 7 ? `${share}%` : ""}
              </div>
            );
          })}
        </div>
      </div>
    );
  };
  return (
    <div data-testid="chart-depth-shares">
      {species && bar(`Species in the reference set, by the answer key's names${speciesTotal ? ` (${nf(speciesTotal)})` : ""}`, species, "species")}
      {bar(`Test finds (${nf(finds.reduce((s, x) => s + x.n, 0))})`, finds, "finds")}
      <div className="legend" style={{ marginTop: 10 }}>
        {Object.entries(BAND_COLOR).map(([band, c]) => <span key={band}><i style={{ background: c.bg }} />{band} records</span>)}
      </div>
    </div>
  );
}

/** Species top 1 (top 5 beneath) by the true species' reference records, one row per approach. */
export function DepthHeatmap({ rows, finds, bands, testId = "chart-heatmap" }: {
  rows: HeatRow[]; finds: (number | null)[]; bands: readonly string[]; testId?: string;
}) {
  if (!rows.length) return <p className="small" data-testid={`${testId}-empty`}>These results were saved before the depth bands were recorded.</p>;
  const total = finds.reduce<number>((s, n) => s + (n ?? 0), 0);
  return (
    <div data-testid={testId}>
      <div className="scrollx">
        <table className="hm">
          <thead><tr><th className="l">Approach</th>
            {bands.map((b, i) => (
              <th key={b}>{b} records{finds[i] != null && <small>{nf(finds[i])} finds{total ? ` · ${Math.round((100 * finds[i]!) / total)}%` : ""}</small>}</th>
            ))}</tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.key} className={r.highlight ? "hl" : ""}>
                <td className="l">{r.name}<small>{r.detail}</small></td>
                {r.cells.map((c, i) => {
                  const h = heatStep(c?.top1);
                  return (
                    <td key={i} className="c" style={{ background: h.bg, color: h.ink }}
                        data-tip={c ? `${r.name} · ${bands[i]} records|top 1 ${pt(c.top1)}% · top 5 ${pt(c.top5)}% · ${nf(c.n)} finds` : undefined}>
                      {c ? <>{pt(c.top1)}<small>top 5 {pt(c.top5)}</small></> : "–"}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="hscale">Species right first time:
        {HEAT_STEPS.map((s) => <i key={s.bg} style={{ background: s.bg }} />)}
        <span>&lt;15% · 15 · 30 · 45 · 60 · 75%+</span>
      </div>
    </div>
  );
}

/** A paired difference in points with its 95% interval, on a ±2 point scale. */
export function PairedDiff({ diff, ci, tip }: { diff: number; ci: [number, number]; tip: string }) {
  const span = Math.max(0.02, Math.ceil(Math.max(Math.abs(ci[0]), Math.abs(ci[1])) * 100) / 100);
  const at = (v: number) => `${50 + (50 * v) / span}%`;
  const ticks = [-span, -span / 2, 0, span / 2, span];
  const sign = (v: number) => `${v < 0 ? "−" : v > 0 ? "+" : ""}${Math.abs(100 * v).toFixed(1)}`;
  return (
    <div className="forest" data-tip={tip} data-testid="chart-paired-diff">
      <i className="zero" style={{ left: "50%" }} />
      <i className="ci" style={{ left: at(ci[0]), width: `${(50 * (ci[1] - ci[0])) / span}%` }} />
      <i className="pt" style={{ left: at(diff) }} />
      {ticks.map((t) => <span key={t} className="tk" style={{ left: at(t) }}>{sign(t)}</span>)}
    </div>
  );
}

/** Horizontal bars with the value at each bar's end (the mockup's .bars). */
export function ValueBars({ rows, labelWidth = 120, testId }: {
  rows: { label: React.ReactNode; sub?: string; frac: number; text: string; color: string; tip?: string; strong?: boolean }[];
  labelWidth?: number; testId?: string;
}) {
  return (
    <div className="bars" data-testid={testId}>
      {rows.map((r, i) => {
        const at = `${Math.min(1, Math.max(0, r.frac)) * 75}%`;
        return (
          <div key={i} className="br" style={{ "--lw": `${labelWidth}px` } as React.CSSProperties} data-tip={r.tip}>
            <span className="lab">{r.strong ? <b style={{ color: "var(--ink)" }}>{r.label}</b> : r.label}{r.sub && <small>{r.sub}</small>}</span>
            <span className="track"><span className="fill" style={{ width: at, background: r.color }} /><span className="val" style={{ left: at }}>{r.text}</span></span>
          </div>
        );
      })}
    </div>
  );
}

/** A numbered pipeline of steps (protocols, benchmark pages, the image recipe). */
export function Pipeline({ steps, label = (i) => `Step ${i + 1}`, icons }: {
  steps: ProtocolStep[]; label?: (i: number) => string; icons?: React.ReactNode[];
}) {
  const cols = steps.length;
  return (
    <div className="lane vis" data-testid="pipeline">
      <div className={`pipe p${cols}`} style={cols === 6 ? undefined : undefined}>
        {steps.map((s, i) => (
          <div key={i} className={`pn ${i === cols - 1 ? "end" : ""}`}>
            <div className="o">{icons?.[i] ?? <Check className="h-5 w-5" />}</div>
            <span className="st">{label(i)}</span>
            <b>{s.title}</b>
            <span>{s.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Three linked steps in a row (the landing's "How a number is made", For researchers). */
export function Strip({ steps }: { steps: { icon: React.ReactNode; title: string; text: React.ReactNode }[] }) {
  return (
    <div className="strip">
      {steps.map((s, i) => (
        <StripCell key={i} last={i === steps.length - 1} icon={s.icon} title={s.title} text={s.text} />
      ))}
    </div>
  );
}
function StripCell({ icon, title, text, last }: { icon: React.ReactNode; title: string; text: React.ReactNode; last: boolean }) {
  return (
    <>
      <div className="sx"><span className="ic" style={{ margin: 0, flex: "none" }}>{icon}</span><div><b>{title}</b>{text}</div></div>
      {!last && <div className="ar"><ArrowRight className="h-[22px] w-[22px]" /></div>}
    </>
  );
}

// --- copy, citations --------------------------------------------------------------------------

/** Copies `text`; says "Copied" for a moment. */
export function CopyButton({ text, label = "Copy", testId }: { text: string; label?: string; testId?: string }) {
  const [done, setDone] = useState(false);
  return (
    <button type="button" className={`copy ${done ? "done" : ""}`} data-testid={testId ?? "button-copy"}
            onClick={() => {
              navigator.clipboard?.writeText(text).catch(() => {});
              setDone(true);
              setTimeout(() => setDone(false), 1400);
            }}>
      {done ? "Copied" : <><Copy className="h-3 w-3" /> {label}</>}
    </button>
  );
}

/** One citation, as text or BibTeX, with a copy button for the form shown. */
export function CitationCard({ c, initial = "text" }: { c: Citation; initial?: "text" | "bibtex" }) {
  const [form, setForm] = useState<"text" | "bibtex">(initial);
  return (
    <div className="cite" data-testid={`cite-${c.key}`}>
      <div className="ch">
        <b>{c.label}</b>
        <span className="flex items-center gap-2">
          <span className="seg" role="group" aria-label="Citation format">
            <button type="button" className={form === "text" ? "on" : ""} aria-pressed={form === "text"} onClick={() => setForm("text")}>Text</button>
            <button type="button" className={form === "bibtex" ? "on" : ""} aria-pressed={form === "bibtex"} onClick={() => setForm("bibtex")}>BibTeX</button>
          </span>
          <CopyButton text={form === "text" ? c.text : c.bibtex} testId={`copy-${c.key}`} />
        </span>
      </div>
      {form === "text"
        ? <div className="tx">{c.text}{c.href && <> <a href={c.href} target="_blank" rel="noreferrer" className="underline">Link<ArrowUpRight className="ml-0.5 inline h-3 w-3" /></a></>}</div>
        : <pre data-testid={`bibtex-${c.key}`}>{c.bibtex}</pre>}
    </div>
  );
}

/** A short code with its copy button (hashes, the base URL). */
export function HashChip({ value, short }: { value: string; short?: string }) {
  return (
    <span className="inline-flex flex-wrap items-center gap-2">
      <span className="hash" title={value}>{short ?? value}</span>
      <CopyButton text={value} />
    </span>
  );
}

/** The search box of a filter bar. */
export function SearchBox({ value, onChange, placeholder, label }: {
  value: string; onChange: (v: string) => void; placeholder: string; label: string;
}) {
  return (
    <label className="search">
      <Search className="h-4 w-4 text-muted-foreground" aria-hidden />
      <input type="search" value={value} placeholder={placeholder} aria-label={label}
             onChange={(e) => onChange(e.target.value)} />
    </label>
  );
}

/** One row of chips in a filter bar: label, then a chip per value with its count. */
export function ChipRow<T extends string>({ label, values, chosen, counts, words, onToggle, swatch }: {
  label: string; values: T[]; chosen: T[]; counts: Record<T, number>; words: Record<T, string>;
  onToggle: (v: T) => void; swatch?: (v: T) => string | undefined;
}) {
  return (
    <div className="row" role="group" aria-label={label}>
      <span className="lbl">{label}</span>
      {values.map((v) => (
        <button key={v} type="button" className={`fchip ${chosen.includes(v) ? "on" : ""}`}
                aria-pressed={chosen.includes(v)} disabled={!counts[v] && !chosen.includes(v)}
                onClick={() => onToggle(v)} data-testid={`chip-${label.toLowerCase().replace(/\s+/g, "-")}-${v}`}>
          {swatch?.(v) && <span className="sw" style={{ background: swatch(v) }} />}
          {words[v]} <span className="c">{counts[v] ?? 0}</span>
        </button>
      ))}
    </div>
  );
}

/** A small "more" link at the foot of a card. */
export function More({ href, children }: { href: string; children: React.ReactNode }) {
  return <Link href={href} className="more">{children} →</Link>;
}

/** A dl of label and value pairs. */
export function KV({ items, labelWidth }: { items: [React.ReactNode, React.ReactNode][]; labelWidth?: number }) {
  return (
    <dl className="kv" style={labelWidth ? { gridTemplateColumns: `${labelWidth}px 1fr` } : undefined}>
      {items.map(([k, v], i) => <Pair key={i} k={k} v={v} />)}
    </dl>
  );
}
const Pair = ({ k, v }: { k: React.ReactNode; v: React.ReactNode }) => <><dt>{k}</dt><dd>{v}</dd></>;
