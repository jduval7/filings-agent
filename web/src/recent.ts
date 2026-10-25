import type { RecentRun } from "./types";

/** One row per set of companies: the latest run, with earlier runs of the same companies behind
 *  it. Re-running GE adds to GE's row instead of pushing everything else down the list. */
export interface RunGroup {
  key: string;
  latest: RecentRun;
  earlier: RecentRun[];
}

export function groupRuns(runs: RecentRun[]): RunGroup[] {
  const newestFirst = [...runs].sort((a, b) => b.created_at.localeCompare(a.created_at));
  const groups = new Map<string, RunGroup>();
  for (const run of newestFirst) {
    const key = [...run.tickers].sort().join(" · ");
    const group = groups.get(key);
    if (group) group.earlier.push(run);
    else groups.set(key, { key, latest: run, earlier: [] });
  }
  return [...groups.values()];
}
