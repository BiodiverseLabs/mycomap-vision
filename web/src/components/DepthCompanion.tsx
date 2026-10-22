import { companionBars, type ReferenceSpecies, type Segment } from "../lib/compareView";

// Above the "species right first time" heatmap (Steve, 2026-10-22): its columns count test
// finds, and most finds are of common species, so a column holds far fewer finds than its
// share of species. Two 100% bars on the heatmap's own bands show both. Colours: the
// heatmap's blue ramp (compareView.heat), light for few records to dark for many.

const STEP: Record<string, { bg: string; ink: string }> = {
  "0": { bg: "#cde2fb", ink: "#0b0b0b" },
  "1-4": { bg: "#86b6ef", ink: "#0b0b0b" },
  "5-19": { bg: "#3987e5", ink: "#0b0b0b" },
  "20-99": { bg: "#1c5cab", ink: "#ffffff" },
  "100+": { bg: "#0d366b", ink: "#ffffff" },
};
const fmt = (n: number) => n.toLocaleString("en-US");

export function DepthCompanion({ species, finds }: {
  species: ReferenceSpecies | null | undefined; finds: (number | null)[];
}) {
  const bars = companionBars(species, finds);
  if (!bars.species || !bars.finds.length) return null;
  return (
    <div className="mb-4 max-w-3xl space-y-2" data-testid="chart-depth-companion">
      <Bar label={`Species in the reference set (${fmt(species!.species)})`} segs={bars.species} unit="species" />
      <Bar label={`Test finds (${fmt(bars.finds.reduce((s, x) => s + x.n, 0))})`} segs={bars.finds} unit="finds" />
      <p className="text-xs text-muted-foreground">
        Most species are rare, but most finds are of common species, so the rare-species columns
        below hold far fewer finds than their share of species.
      </p>
    </div>
  );
}

function Bar({ label, segs, unit }: { label: string; segs: Segment[]; unit: string }) {
  return (
    <div data-testid={`companion-${unit}`}>
      <p className="text-xs font-medium text-[#4a3728] mb-0.5">{label}</p>
      <div className="flex h-6 w-full gap-[2px]" role="img"
           aria-label={`${label}: ${segs.map((s) => `${s.band} records ${s.share}%`).join(", ")}`}>
        {segs.filter((s) => s.n > 0).map((s, i, all) => (
          <div key={s.band} title={`${s.band} records: ${fmt(s.n)} ${unit} (${s.share}%)`}
               className={`flex min-w-[2px] items-center justify-center overflow-hidden text-[11px] tabular-nums ${
                 i === 0 ? "rounded-l-[4px]" : ""} ${i === all.length - 1 ? "rounded-r-[4px]" : ""}`}
               style={{ width: `${(100 * s.n) / Math.max(1, all.reduce((t, x) => t + x.n, 0))}%`,
                        background: STEP[s.band].bg, color: STEP[s.band].ink }}>
            {s.share >= 7 ? `${s.share}%` : ""}
          </div>
        ))}
      </div>
      <p className="mt-0.5 text-[11px] text-muted-foreground tabular-nums">
        {segs.map((s) => `${s.band}: ${s.share}%`).join(" · ")}
      </p>
    </div>
  );
}
