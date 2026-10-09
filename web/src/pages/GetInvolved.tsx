import { PageHeader } from "@/components/Layout";
import { GetInvolvedSections } from "@/components/GetInvolvedSections";

export function GetInvolvedPage() {
  return (
    <>
      <PageHeader title="Get involved">
        MycoMap Vision is built from community scientists' finds, sequenced and checked by people.
      </PageHeader>
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 max-w-4xl">
        <GetInvolvedSections />
      </div>
    </>
  );
}
