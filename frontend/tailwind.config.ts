import type { Config } from "tailwindcss";

// Strictly monochrome — clean black / white / grey. No color.
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0a0a0a",
        surface: "#111112",
        surface2: "#161617",
        raised: "#1c1c1e",
        border: "#262628",
        border2: "#333336",
        text: "#ededed",
        muted: "#8a8a8f",
        faint: "#5a5a5f",
        white: "#fafafa",
      },
      fontFamily: {
        sans: ["'Inter'", "system-ui", "sans-serif"],
        mono: ["'JetBrains Mono'", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
export default config;
