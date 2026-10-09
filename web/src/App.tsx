import { Redirect, Route, Switch } from "wouter";
import { Layout, PageHeader } from "@/components/Layout";
import { IdentifyPage } from "@/pages/Identify";
import { ModelsPage } from "@/pages/Models";
import { DataPage } from "@/pages/Data";
import { AboutPage } from "@/pages/About";
import { PaperPage } from "@/pages/Paper";
import { GetInvolvedPage } from "@/pages/GetInvolved";
import { BenchmarkPage, BenchmarksPage, ForResearchersPage, ProtocolsPage,
         ResearchPage } from "@/pages/Research";
import { ExperimentPage, ExperimentsPage } from "@/pages/Experiments";
import { MOVED } from "@/lib/research";

export function App() {
  return (
    <Layout>
      <Switch>
        <Route path="/" component={IdentifyPage} />
        <Route path="/about" component={AboutPage} />
        <Route path="/get-involved" component={GetInvolvedPage} />
        <Route path="/research" component={ResearchPage} />
        <Route path="/research/results" component={ModelsPage} />
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
        <Route path="/research/paper" component={PaperPage} />
        <Route path="/research/for-researchers" component={ForResearchersPage} />
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
