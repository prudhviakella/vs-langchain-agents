/** The reference app's palette, minus the colours only its removed screens used. */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#F5F6F8", card: "#FFFFFF", line: "#E6E8EB", ink: "#15171A",
        body: "#3A3F45", muted: "#6B7280", faint: "#8A9099",
        primary: { DEFAULT: "#1E3A5F", dark: "#11233A", light: "#A3AEBD" },
        skyTint: "#EFF6FF", ok: "#16A34A", bad: "#DC2626",
      },
      fontFamily: { mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"] },
      borderRadius: { xl2: "14px" },
      boxShadow: { card: "0 1px 2px rgba(16,24,40,.06), 0 1px 3px rgba(16,24,40,.08)" },
    },
  },
  plugins: [],
};
