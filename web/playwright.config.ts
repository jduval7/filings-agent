import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { defineConfig } from "@playwright/test";

// Browser end-to-end: the real server (real graph, cached models) on its own port, real Chromium.
// `make web-e2e` (needs AWS_PROFILE; ~$0 once the cache is warm). Not in CI: it needs Bedrock.
const PORT = Number(process.env.E2E_PORT ?? 8123);
// The test's runs go in a throwaway folder, not the demo's run list (filings and LLM cache are shared).
const STATE = mkdtempSync(join(tmpdir(), "filings-e2e-"));

export default defineConfig({
  testDir: "e2e",
  timeout: 180_000,
  expect: { timeout: 60_000 },
  workers: 1,
  reporter: [["list"]],
  use: { baseURL: `http://127.0.0.1:${PORT}`, viewport: { width: 1440, height: 900 }, trace: "retain-on-failure" },
  webServer: {
    command: `cd .. && FILINGS_STATE_DIR=${STATE} uv run filings-agent serve --port ${PORT}`,
    url: `http://127.0.0.1:${PORT}/health`,
    reuseExistingServer: false,
    timeout: 60_000,
  },
});
