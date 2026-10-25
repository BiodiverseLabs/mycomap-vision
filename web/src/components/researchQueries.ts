import { useQueries, useQuery } from "@tanstack/react-query";
import { api, type RunReport } from "@/lib/api";
import { fetchOutline } from "@/lib/apiOutline";
import { showsMembersPages } from "@/lib/publicView";

// The public feeds the Research pages read (each open to a signed-out visitor:
// tests/test_research_feeds.py), with the same query keys the rest of the site uses so a
// visitor moving between pages reads each once.

export const RESEARCH_FEEDS = ["/api/how-good", "/api/stats", "/api/scoreboard", "/api/prospective",
                               "/api/early-answers", "/api/models", "/api/speed/daily", "/api/docs/v1.json"] as const;

export const useMe = () => useQuery({ queryKey: ["me"], queryFn: api.me, retry: false });
export const useHowGood = () => useQuery({ queryKey: ["how-good"], queryFn: api.howGood });
export const useStats = () => useQuery({ queryKey: ["stats"], queryFn: api.stats });
export const useScoreboard = () => useQuery({ queryKey: ["scoreboard"], queryFn: api.scoreboard });
export const useProspective = () => useQuery({ queryKey: ["prospective"], queryFn: api.prospective });
export const useEarly = () => useQuery({ queryKey: ["early-answers"], queryFn: api.earlyAnswers });
export const useModels = () => useQuery({ queryKey: ["models"], queryFn: api.models });
export const useSpeed = () => useQuery({ queryKey: ["speed-daily"], queryFn: api.speedDaily });
export const useOutline = () => useQuery({ queryKey: ["api-outline"], queryFn: () => fetchOutline(), retry: 1 });

/** The experiment registry, asked for only when this visitor may read it (members). */
export function useExperimentsIfMember() {
  const me = useMe();
  const member = showsMembersPages(me.data);
  return useQuery({ queryKey: ["experiments"], queryFn: api.experiments, retry: false, enabled: member });
}

/** Several runs' reports (/api/scoreboard/{id}), as a map once each arrives. */
export function useRunReports(ids: number[]): { reports: Map<number, RunReport>; loading: boolean } {
  const unique = [...new Set(ids)];
  const qs = useQueries({ queries: unique.map((id) => ({ queryKey: ["run", id], queryFn: () => api.run(id) })) });
  const reports = new Map<number, RunReport>();
  unique.forEach((id, i) => { const d = qs[i]?.data; if (d) reports.set(id, d); });
  return { reports, loading: qs.some((q) => q.isLoading) };
}
