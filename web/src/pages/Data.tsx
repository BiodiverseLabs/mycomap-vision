import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PageHeader } from "@/components/Layout";
import { api, num, pct } from "@/lib/api";
import { modelName } from "@/lib/publicView";

const LICENSE_LABEL: Record<string, string> = {
  open: "Open",
  nc: "Non-commercial",
  arr: "All rights reserved",
};

export function DataPage() {
  const q = useQuery({ queryKey: ["stats"], queryFn: api.stats, refetchInterval: 30_000 });
  const s = q.data;
  return (
    <>
      <PageHeader title="The reference data">
        Only records marked green (DNA-validated) in a MycoMap project are used, with every photo
        of each record from iNaturalist.
      </PageHeader>
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {q.isError && <p className="text-destructive text-sm">The API isn't answering.</p>}
        {s && (
          <>
            <div className="grid gap-4 grid-cols-2 lg:grid-cols-3">
              <Tile label="DNA-validated records" value={num(s.records)}
                    sub={`${num(s.records_north_america)} in North America`} />
              <Tile label="Names" value={num(s.names)} sub="North America, no label conflicts" />
              <Tile label="Photos" value={num(s.photos)}
                    sub={`${(s.photos / Math.max(1, s.inat_ok)).toFixed(1)} per record`} />
            </div>

            <div className="grid gap-6 lg:grid-cols-2">
              <Card>
                <CardHeader><CardTitle className="text-base">Names by number of DNA-verified records</CardTitle></CardHeader>
                <CardContent>
                  <Bars data={Object.fromEntries(Object.entries(s.names_by_records).map(([k, v]) => [`${k} record${k === "1" ? "" : "s"}`, v]))}
                        total={s.names} />
                  <p className="text-xs text-muted-foreground mt-3">
                    Names with a single record still count: the identifier compares with each
                    specimen, so one good match can win.
                  </p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader><CardTitle className="text-base">Photo licenses</CardTitle></CardHeader>
                <CardContent>
                  <Bars data={Object.fromEntries(Object.entries(s.photos_by_license).map(([k, v]) => [LICENSE_LABEL[k] ?? k, v]))}
                        total={s.photos} />
                  <p className="text-xs text-muted-foreground mt-3">
                    Open: CC0, CC-BY, CC-BY-SA. Non-commercial: CC licences with NC or ND.{" "}
                    {num(s.contributors_arr)} photographers have all-rights-reserved photos in
                    the set; their photos are used for training only while permission is sought,
                    and shown only once they grant it.
                  </p>
                  <p className="text-xs text-muted-foreground mt-2" data-testid="text-permissions">
                    Permission from mycomap.org: {num(s.permissions.granted)} granted,{" "}
                    {num(s.permissions.withdrawn)} declined or withdrawn (their all-rights-reserved
                    photos are left out).{" "}
                    {s.permissions.last_good_sync
                      ? `Last read ${new Date(s.permissions.last_good_sync).toLocaleString()}${s.permissions.fresh_for_showing ? "" : " (too long ago: all-rights-reserved photos are hidden until it is read again)"}.`
                      : "Not read yet: all-rights-reserved photos are hidden."}
                  </p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader><CardTitle className="text-base">Photos the identifier compares with</CardTitle></CardHeader>
                <CardContent>
                  {Object.keys(s.embedded).length ? (
                    <Bars data={Object.fromEntries(Object.entries(s.embedded).map(([b, n]) => [modelName(b), n]))}
                          total={s.photos} />
                  ) : (
                    <p className="text-sm text-muted-foreground">None yet.</p>
                  )}
                  <p className="text-xs text-muted-foreground mt-3">
                    The rest are photos of records outside North America, or that iNaturalist no
                    longer serves. New records join every night.
                  </p>
                </CardContent>
              </Card>
            </div>
            <p className="text-xs text-muted-foreground">
              {num(s.inat_missing)} green records are no longer on iNaturalist (deleted or private).{" "}
              {num(s.label_conflicts)} records carry two different names on mycomap.org and are
              held out until resolved.
            </p>
          </>
        )}
      </div>
    </>
  );
}

function Tile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <Card>
      <CardContent className="pt-5">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{label}</div>
        <div className="text-2xl font-semibold text-[#4a3728] tabular-nums mt-1">{value}</div>
        {sub && <div className="text-xs text-muted-foreground mt-1">{sub}</div>}
      </CardContent>
    </Card>
  );
}

function Bars({ data, total }: { data: Record<string, number>; total: number }) {
  return (
    <div className="space-y-2">
      {Object.entries(data).map(([k, v]) => (
        <div key={k} className="grid grid-cols-[9.5rem_minmax(0,1fr)_auto] items-center gap-3 text-sm">
          <span className="truncate text-[#5c4a3a]" title={k}>{k}</span>
          <div className="h-2 rounded-full bg-muted overflow-hidden">
            <div className="h-full rounded-full bg-myco-green" style={{ width: `${(100 * v) / Math.max(1, total)}%` }} />
          </div>
          <span className="tabular-nums text-xs text-muted-foreground w-28 text-right">
            {num(v)} · {pct(v / Math.max(1, total))}
          </span>
        </div>
      ))}
    </div>
  );
}
