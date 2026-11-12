import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server pinned to 5173: the backend's CORS default
// (CORS_ORIGINS in .env.example) already allows exactly this origin.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
  preview: { port: 4173 },
});
