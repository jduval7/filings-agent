import type { ProgressEvent } from "../types";

/** A run's progress, built from its event stream (live, or a recorded run replayed). */
export interface Progress {
  branches: number;
  done: number;
  failed: number;
  toReview: number;
  autoApproved: number;
  researched: number;
  last: string;
  /** every fan-out branch (one per statement in each 10-K), as announced by `planned` */
  tasks: string[];
  finished: Record<string, "done" | "failed">;
  /** the second fan-out: one reconcile branch per material change, and where it went */
  changes: { id: string; lane: "auto" | "review" }[];
}

export const START: Progress = {
  branches: 0,
  done: 0,
  failed: 0,
  toReview: 0,
  autoApproved: 0,
  researched: 0,
  last: "",
  tasks: [],
  finished: {},
  changes: [],
};

export function step(p: Progress, e: ProgressEvent): Progress {
  switch (e.event) {
    case "planned": {
      const tasks = Array.isArray(e.tasks) ? e.tasks.map(String) : [];
      return { ...p, branches: Number(e.branches), tasks, last: `reading ${String(e.branches)} statements` };
    }
    case "branch_done":
      return {
        ...p,
        done: p.done + 1,
        finished: { ...p.finished, [String(e.task_id)]: "done" },
        last: `read ${branchLabel(String(e.task_id))}`,
      };
    case "branch_failed":
      return {
        ...p,
        failed: p.failed + 1,
        finished: { ...p.finished, [String(e.task_id)]: "failed" },
        last: `failed ${branchLabel(String(e.task_id))}`,
      };
    case "wave1_done":
      return { ...p, last: `${String(e.discrepancies)} restated figures, ${String(e.material)} material` };
    case "reconciled": // the event carries no cause: the UI shows only the lane
      return e.lane === "auto"
        ? {
            ...p,
            autoApproved: p.autoApproved + 1,
            changes: [...p.changes, { id: String(e.id), lane: "auto" }],
            last: `${String(e.id)}: confirmed by code`,
          }
        : {
            ...p,
            toReview: p.toReview + 1,
            changes: [...p.changes, { id: String(e.id), lane: "review" }],
            last: `${String(e.id)}: needs a human`,
          };
    case "researched":
      return { ...p, researched: p.researched + 1, last: `researched ${String(e.id)}` };
    case "memo":
      return { ...p, last: "memo written" };
    case "published":
      return { ...p, last: "published" };
    default:
      return p;
  }
}

const STATEMENTS: Record<string, string> = {
  income_statement: "Income statement",
  balance_sheet: "Balance sheet",
  cash_flow_statement: "Cash flows",
};

/** "0000040545-24-000014:income_statement" -> filing "0000040545-24-000014", statement "income_statement" */
export function splitTask(taskId: string): { filing: string; statement: string } {
  const i = taskId.lastIndexOf(":");
  return { filing: taskId.slice(0, i), statement: taskId.slice(i + 1) };
}

/** The year in an SEC accession number (0000040545-24-000014 was filed in 2024). */
export function filedYear(accession: string): string {
  const yy = accession.split("-")[1] ?? "";
  return yy.length === 2 ? `20${yy}` : accession;
}

export function branchLabel(taskId: string): string {
  const { filing, statement } = splitTask(taskId);
  return `${STATEMENTS[statement] ?? statement}, 10-K filed ${filedYear(filing)}`;
}

/** One row per 10-K, one tile per statement: the fan-out, filling in as branches finish. */
export function BranchGrid({ progress }: { progress: Progress }) {
  const filings = [...new Set(progress.tasks.map((t) => splitTask(t).filing))].sort();
  const statements = Object.keys(STATEMENTS);
  if (!filings.length) return null;
  return (
    <div className="branch-grid" role="table" aria-label="fan-out: one branch per statement">
      <div className="branch-row head" role="row">
        <span role="columnheader">10-K</span>
        {statements.map((s) => (
          <span key={s} role="columnheader">
            {STATEMENTS[s]}
          </span>
        ))}
      </div>
      {filings.map((f) => (
        <div className="branch-row" role="row" key={f}>
          <span role="rowheader" className="mono small">
            filed {filedYear(f)}
          </span>
          {statements.map((s) => {
            const id = `${f}:${s}`;
            const planned = progress.tasks.includes(id);
            const state = progress.finished[id] ?? (planned ? "pending" : "none");
            return (
              <span key={s} role="cell" className={`tile ${state}`} title={planned ? branchLabel(id) : "not in this 10-K"}>
                {state === "done" ? "read" : state === "failed" ? "failed" : state === "pending" ? "reading…" : ""}
              </span>
            );
          })}
        </div>
      ))}
    </div>
  );
}

/** The progress panel: bar, counts, the fan-out grid and the latest step. */
export function ProgressPanel({ progress }: { progress: Progress }) {
  return (
    <section className="progress panel">
      <div
        className={`bar${progress.branches ? "" : " indeterminate"}`}
        role="progressbar"
        aria-label="run progress"
        aria-valuemin={0}
        aria-valuemax={progress.branches || undefined}
        aria-valuenow={progress.branches ? progress.done : undefined}
      >
        <div style={{ width: `${progress.branches ? (100 * progress.done) / progress.branches : 30}%` }} />
      </div>
      {progress.branches > 0 && (
        <p>
          <strong>
            {progress.done}/{progress.branches}
          </strong>{" "}
          statements read · {progress.autoApproved} confirmed by code · {progress.toReview} for you ·{" "}
          {progress.researched} researched
        </p>
      )}
      <BranchGrid progress={progress} />
      {progress.changes.length > 0 && (
        <div className="changes">
          <div className="eyebrow">Second fan-out: one reconcile branch per restated figure</div>
          <ul>
            {progress.changes.map((c) => (
              <li key={c.id} className={`chip ${c.lane === "auto" ? "good" : "warn"}`}>
                <span className="mono">{c.id}</span> {c.lane === "auto" ? "code" : "for you"}
              </li>
            ))}
          </ul>
        </div>
      )}
      <p className="muted small" aria-live="polite">
        {progress.last || "starting…"}
      </p>
    </section>
  );
}
