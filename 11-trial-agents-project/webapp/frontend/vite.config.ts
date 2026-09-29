// Dev server: /api goes to the local backend, as the load balancer routes it in AWS.
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://localhost:8000" } },
  // The NVL graph chunk (~1.8 MB) is lazy-loaded — fetched only when a graph
  // answer's Data tab opens; the main bundle stays ~230 KB.
  build: { chunkSizeWarningLimit: 2000 },
  test: { environment: "jsdom", setupFiles: ["./src/test/setup.ts"], globals: true },
});
