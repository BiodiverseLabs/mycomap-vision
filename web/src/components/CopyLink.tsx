import { useState } from "react";
import { Check, Link2 } from "lucide-react";
import { Button } from "@/components/ui/button";

/** Copies this result's own link (/r/<id>). Only people it is given to can open it. */
export function CopyLink({ id }: { id: string }) {
  const [copied, setCopied] = useState(false);
  const url = `${window.location.origin}/r/${id}`;
  const copy = async () => {
    try { await navigator.clipboard.writeText(url); } catch { /* the address bar has it too */ }
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <Button type="button" variant="outline" size="sm" onClick={copy} data-testid="button-copy-link"
            title={`${url}\nOnly people you give this link to can open it.`}>
      {copied ? <Check /> : <Link2 />} {copied ? "Link copied" : "Copy link"}
    </Button>
  );
}
