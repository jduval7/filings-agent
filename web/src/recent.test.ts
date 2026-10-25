import { describe, expect, it } from "vitest";
import { groupRuns } from "./recent";
import type { RecentRun } from "./types";

const run = (thread_id: string, tickers: string[], created_at: string): RecentRun => ({
  thread_id,
  tickers,
  created_at,
  status: "published",
  waiting: 0,
});

describe("groupRuns", () => {
  it("keeps one row per set of companies, newest run on top, older ones behind it", () => {
    const groups = groupRuns([
      run("ge-1", ["GE"], "2026-10-23T10:00:00Z"),
      run("nvda-1", ["NVDA"], "2026-10-23T11:00:00Z"),
      run("ge-2", ["GE"], "2026-10-23T12:00:00Z"),
      run("both", ["NVDA", "GE"], "2026-10-23T09:00:00Z"),
    ]);
    expect(groups.map((g) => [g.key, g.latest.thread_id, g.earlier.map((r) => r.thread_id)])).toEqual([
      ["GE", "ge-2", ["ge-1"]],
      ["NVDA", "nvda-1", []],
      ["GE · NVDA", "both", []],
    ]);
  });

  it("is empty for no runs", () => {
    expect(groupRuns([])).toEqual([]);
  });
});
