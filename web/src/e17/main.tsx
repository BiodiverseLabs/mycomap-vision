// The E17 review page's entry (vite.e17.config.ts). Nothing of the site's App or shell.
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
import "./e17.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
