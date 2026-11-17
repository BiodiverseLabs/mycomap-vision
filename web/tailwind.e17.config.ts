import type { Config } from "tailwindcss";

// The E17 review page's own Tailwind (vite.e17.config.ts): only its files, no site tokens,
// no web fonts.
export default {
  content: ["./src/e17/**/*.{ts,tsx}"],
  theme: { extend: {} },
  plugins: [],
} satisfies Config;
