import { Route, Switch } from "wouter";
import { Layout, PageHeader } from "@/components/Layout";
import { IdentifyPage } from "@/pages/Identify";
import { ModelsPage } from "@/pages/Models";
import { DataPage } from "@/pages/Data";
import { AboutPage } from "@/pages/About";
import { PaperPage } from "@/pages/Paper";

export function App() {
  return (
    <Layout>
      <Switch>
        <Route path="/" component={IdentifyPage} />
        <Route path="/models" component={ModelsPage} />
        <Route path="/data" component={DataPage} />
        <Route path="/about" component={AboutPage} />
        <Route path="/paper" component={PaperPage} />
        <Route>
          <PageHeader title="Page not found" />
        </Route>
      </Switch>
    </Layout>
  );
}
