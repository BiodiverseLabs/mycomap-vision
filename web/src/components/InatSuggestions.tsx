import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ChevronRight, Loader2 } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { inatApi } from "@/lib/api";
import { fullRows, GIVE_UP_MS, inatState, pollInterval, settledView, topRows, type InatView } from "@/lib/inatSuggestions";

const italic = (rank: string) => (rank === "species" || rank === "genus" ? "sci" : "");

/** iNaturalist's suggestions, below MycoMap Vision's answer: names Vision doesn't give, and
 *  "may not be a fungus" when iNat is confident. `initial` came with Vision's answer; with
 *  `visionFailed` there was none, so nothing shows until the server says how it stands
 *  (off, or no iNat account: nothing at all). */
export function InatSuggestions({ initial, visionFailed = false }: {
  initial: InatView | null | undefined; visionFailed?: boolean;
}) {
  const ref = initial?.pending ? initial.ref : null;
  const [startedAt] = useState(() => Date.now());
  // Re-renders once at the end of the wait, so a still-pending check disappears.
  const [, setWaited] = useState(false);
  useEffect(() => {
    if (ref == null) return;
    const t = setTimeout(() => setWaited(true), GIVE_UP_MS + 50);
    return () => clearTimeout(t);
  }, [ref]);
  const asked = useQuery({
    queryKey: ["inat-suggestions", ref, visionFailed],
    queryFn: () => inatApi.now(ref!, visionFailed),
    enabled: ref != null,
    retry: false,
    refetchInterval: (q) => pollInterval(q.state.data?.inat ?? initial, startedAt, Date.now()),
  });
  // A failed check shows nothing; a pending one past the wait shows nothing.
  const latest = ref == null ? initial : asked.isError ? null : asked.data ? asked.data.inat
    : visionFailed ? null : initial;
  const view = settledView(latest, startedAt, Date.now());
  const state = inatState(view);
  if (state === "none" || !view) return null;
  if (view.pending) {
    return (
      <p className="mt-4 flex items-center gap-2 text-sm text-muted-foreground" data-testid="text-inat-checking">
        <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> Checking iNaturalist…
      </p>
    );
  }
  return (
    <Card className="mt-6" data-testid="section-inat-suggestions">
      <CardContent className="space-y-3 px-4 py-4 sm:px-6">
        {view.not_fungus && (
          <p className="text-sm font-medium text-[#8a4b1f]" data-testid="text-inat-not-fungus">
            {view.not_fungus.label}
          </p>
        )}
        {view.lines.length > 0 && (
          <ul className="space-y-1.5">
            {view.lines.map((l) => (
              <li key={l.taxon_id} className="flex flex-wrap items-baseline gap-x-2 text-sm"
                  data-testid="row-inat-suggestion">
                <span className={`font-medium text-[#4a3728] ${italic(l.rank)}`}>{l.name}</span>
                <span className="text-muted-foreground">{l.label}</span>
              </li>
            ))}
          </ul>
        )}
        {view.full && view.full.length > 0 && <FullResponse view={view} />}
      </CardContent>
    </Card>
  );
}

function FullResponse({ view }: { view: Extract<InatView, { lines: unknown }> }) {
  const [open, setOpen] = useState(false);
  const [all, setAll] = useState(false);
  const every = fullRows(view.full);
  const top = topRows(every);
  const rows = all ? every : top;
  const photos = new Set(every.map((r) => r.photo)).size;
  return (
    <div>
      <button type="button" onClick={() => setOpen(!open)} aria-expanded={open}
              className="flex w-full items-center justify-between rounded-md border border-[#A87146]/20 px-3 py-2 text-sm text-[#5c4a3a] hover:bg-[#faf9f7]"
              data-testid="button-inat-full">
        <span>Full iNaturalist response</span>
        <ChevronRight className={`h-4 w-4 transition-transform ${open ? "rotate-90" : ""}`} aria-hidden />
      </button>
      {open && (
        <div className="mt-2 overflow-x-auto" data-testid="table-inat-full">
          <table className="w-full text-left text-xs">
            <thead className="text-muted-foreground">
              <tr>
                {photos > 1 && <th className="py-1 pr-3 font-normal">Photo</th>}
                <th className="py-1 pr-3 font-normal">Name</th>
                <th className="py-1 pr-3 font-normal">Rank</th>
                <th className="py-1 pr-3 font-normal">Score</th>
                <th className="py-1 font-normal">Seen nearby</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={`${r.photo}-${i}`} className="border-t border-[#A87146]/10">
                  {photos > 1 && <td className="py-1 pr-3">{r.photo}</td>}
                  <td className={`py-1 pr-3 ${italic(r.rank)}`}>{r.name}</td>
                  <td className="py-1 pr-3">{r.rank}</td>
                  <td className="py-1 pr-3">{r.score}</td>
                  <td className="py-1">{r.nearby}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {every.length > top.length && (
            <button type="button" onClick={() => setAll(!all)}
                    className="mt-2 text-xs underline decoration-dotted underline-offset-2 hover:text-myco-green"
                    data-testid="button-inat-full-all">
              {all ? "Show the top rows only" : `Show all ${every.length}`}
            </button>
          )}
          <p className="mt-2 text-xs text-muted-foreground">
            iNaturalist's computer vision, asked with your linked iNaturalist account. Not used by MycoMap Vision.
          </p>
        </div>
      )}
    </div>
  );
}
