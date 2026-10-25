// Mirrors the API (src/filings_agent/api/app.py, graph/runner.py summarize, steps/card.py).

export const CAUSES = [
  "stock_split",
  "discontinued_operations",
  "accounting_change",
  "reclassification",
] as const;
export type Cause = (typeof CAUSES)[number];

export type Fit = "fits" | "possible" | "ruled_out" | "no_evidence";

export interface ResearchFinding {
  stance: "supports" | "contradicts";
  quote: string;
  source: string;
  read: string;
  unverified_reading: string;
}

export interface Candidate {
  cause: Cause;
  fit: Fit;
  why: string;
  evidence: string[];
  research?: ResearchFinding[];
  research_not_shown?: number;
}

export interface ResearchRound {
  round: number;
  hint: string;
  looked_at: string;
  open_questions: string;
  findings: number;
  discarded_unverified: number;
  error: string | null;
}

/** One pending change. Deliberately carries no agent answer: the card never shows a favorite. */
export interface Card {
  id: string;
  change: string;
  values: [string, string][];
  changed_in: string[];
  candidates: Candidate[];
  other_causes?: (ResearchFinding & { name: string })[];
  research_rounds?: ResearchRound[];
  research_rounds_left?: number;
  /** How to decide, in words (for API clients; the UI has its own controls). */
  decide?: Record<string, string>;
}

export interface ResolutionSummary {
  /** null while the change waits on a human: the agent's answer never reaches the browser */
  cause: string | null;
  verified: boolean;
  needs_review: boolean;
  lane: "auto" | "review" | null;
}

export interface DecisionIn {
  action: "accept" | "use_original" | "escalate" | "research";
  note?: string;
  correct_cause?: Cause;
}

/** A decision as the server stored it: who made it and when are stamped server-side. */
export interface Decision extends DecisionIn {
  actor?: string;
  decided_at?: string;
}

export type RunStatus =
  | "running"
  | "resuming"
  | "awaiting_review"
  | "published"
  | "done"
  | "interrupted"
  | "not_found"
  | "failed"
  | "running_elsewhere";

export interface Analysis {
  thread_id: string;
  status: RunStatus;
  tickers: string[];
  branches: number;
  /** statements already read: after a crash, what a resume won't read again */
  branches_done: number;
  facts: number;
  rejected: number;
  errors: Record<string, string>;
  cost_usd: number;
  /** cost of the calls actually made (cached answers count $0 here) */
  paid_usd: number;
  cached_branches: number;
  discrepancies: number;
  resolutions: Record<string, ResolutionSummary>;
  awaiting_review: Card[];
  decisions: Record<string, Decision>;
  research_rounds: number;
  /** `table` is what the memo was written from: judges check the prose against it */
  memo: { table: string; markdown: string; verified: boolean; unverified: string[]; attempts: number } | null;
  publication: { published_at: string; facts: number; resolutions: number; newly_published: boolean } | null;
  /** The latest invocation's LangSmith root run (null until one has started in this server). */
  last_run_id: string | null;
  /** every invocation of this thread (start, review, resume), each with its LangSmith root run */
  runs: { run_id: string; kind: "start" | "review" | "resume"; started_at: string }[];
  /** Identifies the cards on screen; sent back with the review so stale tabs get a 409. */
  review_token: string | null;
  /** Why the last answer was asked again (partial or invalid review), per change id. */
  review_problems: Record<string, string>;
}

export interface RecentRun {
  thread_id: string;
  tickers: string[];
  created_at: string;
  status: RunStatus;
  waiting: number;
}

export interface ProgressEvent {
  event: string;
  [key: string]: unknown;
}

/** Demo mode (scripts/demo.py): one scene per decision point, each opened in a known state. */
export interface DemoScene {
  id: string;
  title: string;
  shows: string;
  decision: string;
  kind: "run" | "replay" | "case" | "links";
  case: string | null;
  thread_id: string | null;
  status: RunStatus | null;
  /** the LangSmith pages for this scene: its own traces, your workspace, or public pages */
  langsmith: {
    label: string;
    point_at: string;
    links: { text: string; url: string; where: "this run" | "your workspace" | "public" | "Studio" }[];
  }[];
}

export interface DemoInfo {
  notice: string;
  /** "connected": the scenes' runs are traced and monitored in LangSmith */
  langsmith: "connected" | "offline";
  scenes: DemoScene[];
}

/** One answer published for a case (a row of the ledger's append-only history). */
export interface CaseRow {
  seq?: number;
  thread_id: string;
  filing_set: string;
  cause: string;
  decision: string;
  actor: string | null;
  decided_at: string | null;
  recorded_at?: string;
  updated_at?: string;
  later_escalations?: number;
}

export interface CaseHistory {
  discrepancy_id: string;
  current: CaseRow[];
  history: CaseRow[];
}
