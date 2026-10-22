import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PageHeader } from "@/components/Layout";
import { api, num, pct } from "@/lib/api";
import { modelName } from "@/lib/publicView";
import { PhotoPermissionLine } from "@/components/PhotoPermissionLine";
import { erratumReasons, shortHash, V1, type DatasetRelease } from "@/lib/datasetReleases";

const LICENSE_LABEL: Record<string, string> = {
  open: "Open",
  nc: "Non-commercial",
  arr: "All rights reserved",
};

/** Names that are temporary codes (species known from DNA, not yet described); absent from
 *  a server older than the count. */
const provisional = (s: { names_provisional?: number }): number | null => s.names_provisional ?? null;

export function DataPage() {
  const q = useQuery({ queryKey: ["stats"], queryFn: api.stats, refetchInterval: 30_000 });
  const s = q.data;
  return (
    <>
      <PageHeader title="The reference data">
        Only records marked green (DNA-validated) in a MycoMap project are used, with every photo
        of each record from iNaturalist or Mushroom Observer.
      </PageHeader>
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        <ReleaseCard r={V1} />
        {q.isError && <p className="text-destructive text-sm">The API isn't answering.</p>}
        {s && (
          <>
            <h2 className="font-semibold text-lg text-[#4a3728]" data-testid="heading-live">
              Live reference set <span className="text-sm font-normal text-muted-foreground">(changes every night)</span>
            </h2>
            <div className="grid gap-4 grid-cols-2 lg:grid-cols-3">
              <Tile label="DNA-validated records" value={num(s.records)}
                    sub={`${num(s.records_north_america)} in North America`} />
              <Tile label="Names" value={num(s.names)}
                    sub={provisional(s) != null
                      ? `North America, no label conflicts; ${num(provisional(s))} provisional (not yet described)`
                      : "North America, no label conflicts"} />
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
                    the set.
                  </p>
                  <p className="text-xs text-muted-foreground mt-2" data-testid="text-permissions">
                    <PhotoPermissionLine kind="control" />
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
        <GetTheImages />
      </div>
    </>
  );
}

const day = (iso: string) => new Date(iso).toLocaleDateString("en-US",
  { year: "numeric", month: "long", day: "numeric", timeZone: "UTC" });

/** A frozen dataset release: the records and photos research numbers come from. It never
 *  changes; a correction makes the next release. */
function ReleaseCard({ r }: { r: DatasetRelease }) {
  const l = r.photos.byLicence;
  return (
    <Card data-testid={`card-release-${r.id}`}>
      <CardHeader>
        <CardTitle className="text-base">Dataset release {r.id} (frozen)</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-sm text-[#5c4a3a]">
        <p>
          The fixed snapshot every research result now comes from: labels as of {day(r.labelSnapshot)},
          cut {day(r.created)}. It never changes, so anyone can rebuild a result from it with the
          code and the one command each result names.
        </p>
        <div className="grid gap-4 grid-cols-2 lg:grid-cols-4">
          <Tile label="Records" value={num(r.records.included)}
                sub={`${num(r.records.bySource.inat)} iNaturalist, ${num(r.records.bySource.mo)} Mushroom Observer`} />
          <Tile label="Training split" value={num(r.records.train)}
                sub={`validation split ${num(r.records.val)}: first validated after ${day(r.valCutoff)}`} />
          <Tile label="Photos" value={num(r.photos.included)}
                sub={`open ${pct(l.open / r.photos.included)}, non-commercial ${pct(l.nc / r.photos.included)}, all rights reserved ${pct(l.arr / r.photos.included)}`} />
          <Tile label="Names" value={num(r.labels)} sub="North America" />
        </div>
        <dl className="grid grid-cols-[9rem_1fr] gap-x-4 gap-y-1 text-xs">
          <dt className="text-muted-foreground">Release hash</dt>
          <dd className="font-mono break-all" title={r.releaseHash}>{shortHash(r.releaseHash)}…</dd>
          <dt className="text-muted-foreground">Reference hash</dt>
          <dd className="font-mono break-all" title={r.referenceHash}>{shortHash(r.referenceHash)}…</dd>
          <dt className="text-muted-foreground">Cut by code</dt>
          <dd className="font-mono break-all">{r.codeCommit.slice(0, 7)}</dd>
          <dt className="text-muted-foreground">Public files</dt>
          <dd>{r.doi ? <a href={`https://doi.org/${r.doi}`} className="text-myco-green underline">{r.doi}</a>
                     : "Creative Commons photos only, coming with the paper"}</dd>
        </dl>
        <p className="text-xs text-muted-foreground">
          {num(r.records.candidates - r.records.included)} of the {num(r.records.candidates)} candidate
          records were left out, each with a recorded reason (no photos of the specimen, outside North
          America, a guest organism, no usable name). Results from before this release are kept as
          development history.
        </p>
        {r.erratum && (
          <p className="text-xs text-muted-foreground" data-testid={`text-erratum-${r.id}`}>
            Erratum {r.erratum.id}: {num(r.erratum.records)} records found faulty after the cut are left
            out of the references and of scoring ({erratumReasons(r.erratum)}). The release itself is
            unchanged.
          </p>
        )}
      </CardContent>
    </Card>
  );
}

/** How anyone gets exactly the images a model saw (Steve, 2026-10-09): originals are kept, each
 *  model input is derived by a published recipe and checked by its hash. CC photos only. Release
 *  v1 lists the recipe and hashes (`mv dataset derive`); the public address comes with its
 *  Creative Commons files, so until then this says so. */
function GetTheImages() {
  return (
    <Card data-testid="card-get-images">
      <CardHeader><CardTitle className="text-base">Get the images</CardTitle></CardHeader>
      <CardContent className="space-y-2 text-sm text-[#5c4a3a]">
        <p>
          Every original photo is kept. The image a model actually saw is made from its original by
          a fixed, published recipe: how it is resized and cropped, the size, the JPEG quality, and
          the software and version that did it. Each dataset release lists that recipe and a
          checksum (SHA-256) of every original and of every image made from it.
        </p>
        <p>
          So anyone can regenerate exactly the images behind a published result and check them: run
          the recipe on the original, and the checksum must match the one in the release.
        </p>
        <p>
          Only openly licensed (Creative Commons) photos can be fetched this way. Photos whose
          owners reserve all rights are never redistributed.
        </p>
        <p className="text-muted-foreground">
          Dataset release v1 lists the recipe and checksums. The download address and an example
          will appear here with its public (Creative Commons) files.
        </p>
      </CardContent>
    </Card>
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
