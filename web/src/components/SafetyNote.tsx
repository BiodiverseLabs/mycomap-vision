import { TriangleAlert } from "lucide-react";
import { SAFETY_LINE } from "@/lib/publicView";

/** The safety line, set apart so nobody misses it: with every answer and on How it works. */
export function SafetyNote({ className = "" }: { className?: string }) {
  return (
    <div className={`flex gap-3 rounded-lg border border-amber-300 bg-amber-50 p-4 text-sm text-[#4a3728] ${className}`}
         role="note" data-testid="note-safety">
      <TriangleAlert className="h-5 w-5 shrink-0 text-amber-600" aria-hidden />
      <p>
        <strong>{SAFETY_LINE}</strong> Vision gives no edibility information, and a match from
        photos is not a DNA-confirmed answer for your find.
      </p>
    </div>
  );
}
