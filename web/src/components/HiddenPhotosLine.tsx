import { ArrowUpRight } from "lucide-react";
import { hiddenPhotosLine, type ObservationSource } from "@/lib/observationLink";

/** Under a set of an observation's photos: which aren't shown here (not Creative Commons),
 *  and a link to see them at the source, in a new tab. Nothing when every photo shows. */
export function HiddenPhotosLine({ urls, observation, testId }: {
  urls: string[];
  observation: { source: ObservationSource; url: string | null } | null | undefined;
  testId: string;
}) {
  const line = hiddenPhotosLine(urls, observation);
  if (!line) return null;
  return (
    <p className="text-xs text-muted-foreground" data-testid={testId}>
      {line.note}.{" "}
      <a href={line.href} target="_blank" rel="noreferrer"
         className="inline-flex items-center gap-0.5 font-medium text-[#4a3728] underline">
        {line.link}<ArrowUpRight className="h-3 w-3" aria-hidden />
      </a>
    </p>
  );
}
