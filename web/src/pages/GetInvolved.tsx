import { ArrowUpRight } from "lucide-react";
import { PageHeader } from "@/components/Layout";

// Minimal page so the menu item works (Steve, 2026-10-09). Its copy belongs to the site-copy
// work (feat/site-standards-copy), which may replace this page.

const WAYS: { title: string; text: string; href: string; link: string }[] = [
  { title: "Get your finds sequenced",
    text: "Every DNA-verified record makes the identifier better for that species. MycoMap sequences fungal collections for free through its network.",
    href: "https://mycomap.org/network", link: "Free sequencing on mycomap.org" },
  { title: "Let us use your photos",
    text: "If your iNaturalist photos are all rights reserved, you can allow MycoMap Vision to use them, and withdraw at any time.",
    href: "https://mycomap.org/profile", link: "Your profile, AI photo permission" },
  { title: "Help validate records",
    text: "Records become references once a MycoMap validation project accepts their DNA name.",
    href: "https://mycomap.org/projects", link: "Validation projects" },
];

export function GetInvolvedPage() {
  return (
    <>
      <PageHeader title="Get involved">
        MycoMap Vision is built from community scientists' finds, sequenced and checked by people.
      </PageHeader>
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 grid gap-4 md:grid-cols-3 max-w-5xl">
        {WAYS.map((w) => (
          <div key={w.title} className="rounded-lg border border-[#A87146]/20 bg-white p-4 flex flex-col">
            <h2 className="font-semibold text-[#4a3728]">{w.title}</h2>
            <p className="text-sm text-[#5c4a3a] mt-1 flex-1">{w.text}</p>
            <a href={w.href} className="mt-3 inline-flex items-center gap-1 text-sm text-myco-green underline">
              {w.link} <ArrowUpRight className="h-3.5 w-3.5" />
            </a>
          </div>
        ))}
      </div>
    </>
  );
}
