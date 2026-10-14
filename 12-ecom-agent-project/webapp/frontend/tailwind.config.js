/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#F5F6F8",            // app background
        card: "#FFFFFF",
        line: "#E6E8EB",              // hairline borders
        ink: "#15171A",               // headings
        body: "#3A3F45",              // body text
        muted: "#6B7280",             // secondary text
        faint: "#8A9099",             // captions / mono ids
        brand: { DEFAULT: "#2563EB", 600: "#1D4ED8", 50: "#EFF6FF" }, // store blue (chart_gen primary)
        primary: { DEFAULT: "#1E3A5F", dark: "#11233A", light: "#A3AEBD" },
        graphite: "#535353",
        ash: "#878787",
        frost: "#F8FAFC",
        slate: { DEFAULT: "#475569" },
        mist: "#e8edf5",
        azure: "#01579B",
        indigo: { DEFAULT: "#1E40AF" },
        skyTint: "#EFF6FF",
        sky: { DEFAULT: "#3B82F6" },
        navy: "#183865",
        cobalt: "#1565C0",
        amber: { DEFAULT: "#A3660A" },
        linkBlue: "#1976D2",
        charcoal: "#303030",
        violet: { DEFAULT: "#8B5CF6" },
        cyan: { DEFAULT: "#06B6D4" },
        ok: "#16A34A",
        flag: {
          crit: "#DC2626", critbg: "#FDE7E7",
          high: "#B45309", highbg: "#FBEFD9",
          med: "#5B6470",  medbg: "#EFF0F2",
          low: "#15803D",  lowbg: "#E7F4EC",
        },
      },
      fontFamily: {
        sans: ["'Plus Jakarta Sans'", "Inter", "ui-sans-serif", "system-ui",
               "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
        mono: ["'JetBrains Mono'", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      // Layered, low-opacity shadows tuned to the ink-blue neutral for a soft,
      // premium elevation ramp. `card` and `pop` keep their original names so
      // existing usages keep working (pop is now a touch richer).
      boxShadow: {
        xs: "0 1px 2px rgba(16,24,40,0.04)",
        sm: "0 1px 2px rgba(16,24,40,0.04), 0 1px 3px rgba(16,24,40,0.06)",
        card: "0 1px 2px rgba(16,24,40,0.05), 0 4px 10px -2px rgba(16,24,40,0.07)",
        "card-hover": "0 2px 6px rgba(16,24,40,0.06), 0 12px 28px -8px rgba(16,24,40,0.14)",
        md: "0 4px 12px -2px rgba(16,24,40,0.08), 0 2px 6px -2px rgba(16,24,40,0.06)",
        pop: "0 4px 12px -2px rgba(16,24,40,0.08), 0 14px 34px -8px rgba(16,24,40,0.16)",
        lg: "0 12px 40px -8px rgba(16,24,40,0.20)",
        header: "0 1px 0 rgba(16,24,40,0.04), 0 4px 14px -8px rgba(16,24,40,0.10)",
      },
      borderRadius: { xl2: "12px", xl3: "16px" },
      transitionTimingFunction: {
        premium: "cubic-bezier(0.22, 0.61, 0.36, 1)",
      },
      keyframes: {
        "fade-in": { from: { opacity: "0" }, to: { opacity: "1" } },
        "fade-in-up": {
          from: { opacity: "0", transform: "translateY(6px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        "scale-in": {
          from: { opacity: "0", transform: "scale(0.97)" },
          to: { opacity: "1", transform: "scale(1)" },
        },
        shimmer: {
          from: { backgroundPosition: "-200% 0" },
          to: { backgroundPosition: "200% 0" },
        },
      },
      animation: {
        "fade-in": "fade-in 0.2s ease-out both",
        "fade-in-up": "fade-in-up 0.24s cubic-bezier(0.22,0.61,0.36,1) both",
        "scale-in": "scale-in 0.16s cubic-bezier(0.22,0.61,0.36,1) both",
        shimmer: "shimmer 1.4s linear infinite",
      },
    },
  },
  plugins: [],
};
