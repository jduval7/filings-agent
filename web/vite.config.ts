import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Dev: `npm run dev` on :5173 proxies the API on :8000 (`make serve`).
// Prod: `make web` builds into dist/, which FastAPI serves at /.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/analyses": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test-setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"], // e2e/ is Playwright's (make web-e2e)
  },
});
