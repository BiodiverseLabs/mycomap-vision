import path from "node:path";
import react from "@vitejs/plugin-react";
import autoprefixer from "autoprefixer";
import tailwindcss from "tailwindcss";
import { defineConfig } from "vite";

// The E17 blind review page (src/e17/), built on its own: none of the site's App, shell,
// fonts or /api/me. The server writes the page itself (e17review.page_html) from this
// build's manifest and serves the files only behind the review token, from
// dist/e17review/assets/. deploy.sh builds it after the site, into dist-next/e17review.
export default defineConfig({
  plugins: [react()],
  base: "./",
  publicDir: false,
  css: {
    postcss: {
      plugins: [tailwindcss({ config: path.resolve(import.meta.dirname, "tailwind.e17.config.ts") }), autoprefixer()],
    },
  },
  build: {
    outDir: "dist/e17review",
    emptyOutDir: true,
    manifest: true,
    modulePreload: { polyfill: false },
    assetsInlineLimit: 0,
    rollupOptions: {
      input: { e17: path.resolve(import.meta.dirname, "src/e17/main.tsx") },
      output: {
        entryFileNames: "assets/e17-[hash].js",
        chunkFileNames: "assets/e17-[hash].js",
        assetFileNames: "assets/e17-[hash][extname]",
      },
    },
  },
});
