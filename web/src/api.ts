import type { Analysis, CaseHistory, DecisionIn, DemoInfo, ProgressEvent, RecentRun } from "./types";

/** fetch, signing in once if the server asks (401: API tokens are configured). The token is
 *  exchanged for an HttpOnly session cookie, which EventSource (no custom headers) also sends;
 *  the page never stores the token itself. */
async function call(url: string, init?: RequestInit): Promise<Response> {
  const resp = await fetch(url, init);
  if (resp.status !== 401) return resp;
  const token = window.prompt("API token for filings-agent");
  if (!token) return resp;
  const session = await fetch("/auth/session", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ token }),
  });
  return session.ok ? fetch(url, init) : resp;
}

async function json<T>(resp: Response): Promise<T> {
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`${resp.status}: ${body}`);
  }
  return (await resp.json()) as T;
}

/** `fresh`: re-analyze, a new run even when one with the same inputs exists. */
export async function startAnalysis(
  tickers: string[],
  fresh = false,
): Promise<{ thread_id: string }> {
  return json(
    await call("/analyses", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(fresh ? { tickers, fresh } : { tickers }),
    }),
  );
}

export async function getAnalysis(threadId: string): Promise<Analysis> {
  return json(await call(`/analyses/${threadId}`));
}

/** The LangSmith project page, or null when tracing is off: "open trace" links use it. */
export async function getProjectUrl(): Promise<string | null> {
  try {
    const meta = await json<{ langsmith_project_url: string | null }>(await call("/meta"));
    return meta.langsmith_project_url;
  } catch {
    return null; // a missing link must never break the page
  }
}

export async function listAnalyses(): Promise<RecentRun[]> {
  return json(await call("/analyses"));
}

/** `reviewToken` identifies the cards on screen: if they changed (another tab, a research
 *  round), the server answers 409 instead of applying decisions made on old evidence. */
export async function submitReview(
  threadId: string,
  decisions: Record<string, DecisionIn>,
  reviewToken: string | null,
): Promise<void> {
  await json(
    await call(`/analyses/${threadId}/review`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ decisions, review_token: reviewToken }),
    }),
  );
}

/** Live progress (Server-Sent Events). Calls onEnd once, when the run stops. Returns a closer. */
export function followProgress(
  threadId: string,
  onEvent: (e: ProgressEvent) => void,
  onEnd: () => void,
): () => void {
  const source = new EventSource(`/analyses/${threadId}/events`);
  let ended = false;
  const end = () => {
    if (!ended) {
      ended = true;
      source.close();
      onEnd();
    }
  };
  source.onmessage = (m) => onEvent(JSON.parse(m.data) as ProgressEvent);
  for (const name of ["planned", "branch_done", "branch_failed", "wave1_done", "reconciled", "researched", "memo", "published"]) {
    source.addEventListener(name, (m) => onEvent(JSON.parse((m as MessageEvent).data) as ProgressEvent));
  }
  source.addEventListener("end", (m) => {
    onEvent(JSON.parse((m as MessageEvent).data) as ProgressEvent);
    end();
  });
  source.addEventListener("status", end); // the run isn't live in this process: already finished
  source.onerror = end;
  return () => {
    ended = true;
    source.close();
  };
}

/** Continue a run that stopped part-way (crash, outage): from its last checkpoint. */
export async function resumeAnalysis(threadId: string): Promise<void> {
  await json(await call(`/analyses/${threadId}/resume`, { method: "POST" }));
}

/** One case's decision history from the ledger, and the answer that stands. */
export async function getCase(discrepancyId: string): Promise<CaseHistory> {
  return json(await call(`/cases/${encodeURIComponent(discrepancyId)}`));
}

/** The demo scenes, or null when the server isn't in demo mode. */
export async function getDemo(): Promise<DemoInfo | null> {
  try {
    const resp = await call("/demo");
    return resp.ok ? ((await resp.json()) as DemoInfo) : null;
  } catch {
    return null;
  }
}

/** A new run of a demo scene's recipe (from the recordings), after it was clicked through. */
export async function resetScene(sceneId: string): Promise<{ thread_id: string }> {
  return json(await call(`/demo/${sceneId}/reset`, { method: "POST" }));
}

/** A recorded run's progress events, replayed at reading pace. Calls onEnd once. */
export function followReplay(sceneId: string, onEvent: (e: ProgressEvent) => void, onEnd: () => void): () => void {
  const source = new EventSource(`/demo/replay/${sceneId}`);
  let ended = false;
  const handle = (m: MessageEvent) => {
    const e = JSON.parse(String(m.data)) as ProgressEvent;
    onEvent(e);
    if (e.event === "end" && !ended) {
      ended = true;
      source.close();
      onEnd();
    }
  };
  for (const name of ["planned", "branch_done", "branch_failed", "wave1_done", "reconciled", "researched", "memo", "published", "end"]) {
    source.addEventListener(name, (m) => handle(m as MessageEvent));
  }
  source.onerror = () => {
    if (!ended) {
      ended = true;
      source.close();
      onEnd();
    }
  };
  return () => {
    ended = true;
    source.close();
  };
}
