import { ArrowRight, ArrowUpRight } from "lucide-react";
import { PhotoPermissionLine } from "@/components/PhotoPermissionLine";

// The pieces of the explainer pages (How it works and Get involved, Steve, 2026-10-22: the
// approved redesigns), so both read as one site: section bands, headings, cards, tiles and
// the bar chart.

// Chart marks (dataviz validate_palette.js on the white surface: species and genus pass the
// colour-blind and normal-vision separation checks; every value is also written beside its
// bar). REST is the de-emphasis grey for the bar that is not the point; the ramp is one hue,
// light to dark, for an ordered scale.
export const C = {
  s1: "#7FAE34", s2: "#8B4A1F", rest: "#d9d0c5",
  ramp: ["#98c052", "#7fab38", "#67922a", "#507a1d", "#3b5f13"],
};

export function Wrap({ band, id, testId, className = "", children }: {
  band?: boolean; id?: string; testId?: string; className?: string; children: React.ReactNode;
}) {
  return (
    <section id={id} data-testid={testId}
             className={`scroll-mt-16 ${band ? "bg-[#fbf9f6] border-y border-[#A87146]/[.12]" : ""}`}>
      <div className={`mx-auto max-w-[1080px] px-4 sm:px-8 py-10 sm:py-14 ${className}`}>{children}</div>
    </section>
  );
}

export const Kicker = ({ children }: { children: React.ReactNode }) => (
  <p className="text-xs font-semibold uppercase tracking-[.08em] text-[#6b8f2a] mb-1.5">{children}</p>
);
export const H2 = ({ children, className = "" }: { children: React.ReactNode; className?: string }) => (
  <h2 className={`font-display font-semibold text-[25px] sm:text-[30px] leading-tight text-[#4a3728] mb-2 ${className}`}>
    {children}
  </h2>
);
export const H3 = ({ children, className = "" }: { children: React.ReactNode; className?: string }) => (
  <h3 className={`font-display font-semibold text-lg leading-snug text-[#4a3728] mb-1 ${className}`}>{children}</h3>
);
export const Sub = ({ children }: { children: React.ReactNode }) => (
  <p className="text-base sm:text-[17px] max-w-[720px] mb-7">{children}</p>
);
export const Small = ({ children, className = "" }: { children: React.ReactNode; className?: string }) => (
  <p className={`text-[13px] text-muted-foreground ${className}`}>{children}</p>
);
export const Cap = ({ children, className = "" }: { children: React.ReactNode; className?: string }) => (
  <p className={`text-sm text-muted-foreground leading-normal mt-2.5 ${className}`}>{children}</p>
);
export const Card = ({ children, className = "" }: { children: React.ReactNode; className?: string }) => (
  <div className={`min-w-0 rounded-[10px] border border-[#A87146]/20 bg-white p-5 ${className}`}>{children}</div>
);
export const Ink = ({ children }: { children: React.ReactNode }) => (
  <b className="font-semibold text-[#4a3728]">{children}</b>
);
export const Icon = ({ children, brown }: { children: React.ReactNode; brown?: boolean }) => (
  <span className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ${
    brown ? "bg-[#A36848]/[.12] text-[#7a4a2a]" : "bg-myco-green/[.14] text-[#4f7a16]"}`}>
    {children}
  </span>
);

export function ExternalLinkText({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a className="underline" href={href} target="_blank" rel="noreferrer">
      {children}
      <ArrowUpRight className="inline h-3 w-3 ml-0.5 align-baseline" aria-hidden />
    </a>
  );
}

export function Tile({ value, label }: { value: string; label: string }) {
  return (
    <div className="min-w-0 rounded-lg border border-[#A87146]/20 bg-white px-4 py-3.5">
      <div className="text-2xl sm:text-[30px] font-semibold leading-tight text-[#4a3728] tabular-nums">{value}</div>
      <div className="text-[13px] leading-snug text-muted-foreground mt-1">{label}</div>
    </div>
  );
}

/** The hero's buttons: the green one, and the white ones beside it. */
export const PRIMARY_BUTTON = "inline-flex items-center gap-1.5 rounded-md bg-myco-green px-4 py-2.5 text-sm font-medium text-white";
export const GHOST_BUTTON = "inline-flex items-center gap-1.5 rounded-md border border-[#A87146]/20 bg-white px-4 py-2.5 text-sm font-medium text-[#4a3728]";

/** The arrow between steps: across on a wide screen, down on a phone. */
export function FlowArrow({ small }: { small?: boolean }) {
  return (
    <div className={`flex ${small ? "h-[26px]" : "h-9"} md:h-auto items-center justify-center text-[#b9a48f]`} aria-hidden>
      <ArrowRight className={`${small ? "h-5 w-5" : "h-7 w-7"} rotate-90 md:rotate-0`} />
    </div>
  );
}

export interface BarRow { label: React.ReactNode; sub?: string; frac: number; text: string; color: string;
                          title: string }

/** Horizontal bars, each value written at its bar's end (room is kept for it, so nothing
 *  runs past the card at phone width). */
export function Bars({ rows, labelWidth, testId }: { rows: BarRow[]; labelWidth: number; testId?: string }) {
  return (
    <div className="flex flex-col gap-2.5" data-testid={testId}>
      {rows.map((r, i) => {
        const at = `${Math.min(1, Math.max(0, r.frac)) * 100}%`;
        return (
          <div key={i} className="grid items-center gap-3 text-sm" title={r.title}
               style={{ gridTemplateColumns: `${labelWidth}px minmax(0,1fr)` }}>
            <span className="leading-tight">
              {r.label}
              {r.sub && <small className="block text-xs text-muted-foreground">{r.sub}</small>}
            </span>
            <span className="relative h-6 mr-[4.5rem]">
              <span className="absolute left-0 top-0.5 h-5 rounded-r min-w-[2px]"
                    style={{ width: at, background: r.color }} />
              <span className="absolute top-0 whitespace-nowrap pl-2 text-[13px] font-semibold leading-6 text-[#4a3728] tabular-nums"
                    style={{ left: at }}>{r.text}</span>
            </span>
          </div>
        );
      })}
    </div>
  );
}

/** One of the two approved lines to photographers, boxed with a green edge. */
export function PermissionBox({ kind, className = "" }: { kind: "control" | "learn"; className?: string }) {
  return (
    <div className={`rounded-lg border border-[#A87146]/20 border-l-4 bg-white px-4 py-3 text-[15px] text-[#4a3728] [&_a]:font-semibold [&_a]:text-[#4f7a16] [&_a]:no-underline ${className}`}
         style={{ borderLeftColor: C.s1 }} data-testid={`permission-${kind}`}>
      <PhotoPermissionLine kind={kind} />
    </div>
  );
}
