// Vite: dev server on :5173 with /api proxied to the local backend (:8000) — the same
// single-origin layout CloudFront gives in AWS (/* -> S3, /api/* -> the backend).
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: false } },
  },
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 5000 },
});
