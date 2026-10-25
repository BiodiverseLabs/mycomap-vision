import { useQuery } from "@tanstack/react-query";
import { Redirect, Route, Switch, useLocation } from "wouter";
import { Layout, PageHeader } from "@/components/Layout";
import { IdentifyPage } from "@/pages/Identify";
import { ResultsPage } from "@/pages/Results";
import { DataPage } from "@/pages/Data";
import { AboutPage } from "@/pages/About";
import { GetInvolvedPage } from "@/pages/GetInvolved";
import { ApiDocsPage } from "@/pages/ApiDocs";
import { ResearchPage } from "@/pages/ResearchLanding";
import { BenchmarkPage, BenchmarksPage } from "@/pages/Benchmarks";
import { ProtocolsPage } from "@/pages/Protocols";
import { ForResearchersPage } from "@/pages/ForResearchers";
import { ExperimentPage, ExperimentsPage } from "@/pages/Experiments";
import { FeedbackPage } from "@/pages/Feedback";
import { RunPage } from "@/pages/Run";
import { liveRuns } from "@/lib/runLink";
import { api, signInUrl } from "@/lib/api";
import { API_PATH, MOVED, mustSignIn } from "@/lib/research";
import { FEEDBACK_PATH } from "@/lib/feedback";

export function App() {
  const [location] = useLocation();
  const me = useQuery({ queryKey: ["me"], queryFn: api.me, retry: false });
  // A signed-out visitor on the public API page who moves on to any other page signs in
  // first, as a full page load would make them (signin.PUBLIC_PAGES).
  if (mustSignIn(me.data, location)) {
    window.location.assign(signInUrl());
    return null;
  }
  return (
    <Layout>
      <Switch>
        <Route path="/" component={IdentifyPage} />
        {/* A run's own link (lib/runLink.ts). The tab that just ran it stays on Identify. */}
        <Route path="/r/:id">{(params) => liveRuns.has(params.id) ? <IdentifyPage /> : <RunPage id={params.id} />}</Route>
        <Route path="/about" component={AboutPage} />
        <Route path="/get-involved" component={GetInvolvedPage} />
        <Route path="/research" component={ResearchPage} />
        <Route path="/research/results" component={ResultsPage} />
        <Route path="/research/benchmarks" component={BenchmarksPage} />
        <Route path="/research/benchmarks/:slug">
          {(params) => <BenchmarkPage slug={params.slug} />}
        </Route>
        <Route path="/research/protocols" component={ProtocolsPage} />
        <Route path="/research/data" component={DataPage} />
        <Route path="/research/experiments" component={ExperimentsPage} />
        <Route path="/research/experiments/:slug">
          {(params) => <ExperimentPage slug={params.slug} />}
        </Route>
        <Route path="/research/for-researchers" component={ForResearchersPage} />
        <Route path={API_PATH} component={ApiDocsPage} />
        <Route path={FEEDBACK_PATH} component={FeedbackPage} />
        {Object.entries(MOVED).map(([from, to]) => (
          <Route key={from} path={from}><Redirect to={to} replace /></Route>
        ))}
        <Route>
          <PageHeader title="Page not found" />
        </Route>
      </Switch>
    </Layout>
  );
}
