import { useQuery } from "@tanstack/react-query";
import { PageHeader } from "@/components/Layout";
import { Markdown } from "@/components/Markdown";
import { api, ApiError, signInUrl } from "@/lib/api";

// The working draft of the Vision paper (docs/paper/draft.md), for signed-in mycomap.org
// members only. Display only: comments come by email (Steve, 2026-10-09).


export function PaperPage() {
  const paper = useQuery({ queryKey: ["paper"], queryFn: api.paper, retry: false });
  const signedOut = paper.error instanceof ApiError && paper.error.status === 401;
  return (
    <>
      <PageHeader title="Paper in Progress">
        A living draft of the MycoMap Vision paper, planned for Mycologia. Numbers change as
        tests are rerun. Comments and suggestions to{" "}
        <a href="mailto:info@mycomap.org" className="text-myco-green underline">
          info@mycomap.org
        </a>
        .
      </PageHeader>
      <article
        data-testid="paper-draft"
        className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 max-w-3xl text-[#5c4a3a] leading-relaxed"
      >
        {paper.isPending && <p>Loading the draft...</p>}
        {signedOut && (
          <p>
            The draft is for mycomap.org members.{" "}
            <a href={signInUrl()} className="text-myco-green underline">
              Sign in
            </a>{" "}
            to read it.
          </p>
        )}
        {paper.error && !signedOut && <p>The draft could not be loaded ({paper.error.message}).</p>}
        {paper.data && (
          <Markdown>{paper.data.markdown}</Markdown>
        )}
      </article>
    </>
  );
}
