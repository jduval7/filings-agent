import { useCallback, useEffect, useState } from "react";
import { getDemo, listAnalyses, startAnalysis } from "./api";
import { AnalysisView } from "./components/AnalysisView";
import { SceneView } from "./components/DemoViews";
import { Icon } from "./components/Icon";
import { groupRuns } from "./recent";
import { applyTheme, initialTheme, type Theme } from "./theme";
import type { DemoInfo, RecentRun } from "./types";

const current = () => decodeURIComponent(window.location.hash.replace(/^#\/?/, "")) || null;

const STATUS_TONE: Record<string, string> = {
  awaiting_review: "warn",
  published: "good",
  done: "good",
  running: "info",
  resuming: "info",
  failed: "bad",
  interrupted: "bad",
};

function RunLink({ run, active }: { run: RecentRun; active: boolean }) {
  return (
    <a href={`#/${run.thread_id}`} className={active ? "active" : ""} aria-current={active ? "page" : undefined}>
      <strong>{run.tickers.join(" · ")}</strong>
      <span className={`pill ${STATUS_TONE[run.status] ?? "neutral"}`}>
        {run.status === "awaiting_review" ? `${run.waiting} to review` : run.status.replaceAll("_", " ")}
      </span>
      <span className="when">
        {new Date(run.created_at).toLocaleString()} · <span className="mono">{run.thread_id}</span>
      </span>
    </a>
  );
}

export function App() {
  const [threadId, setThreadId] = useState<string | null>(current);
  const [demo, setDemo] = useState<DemoInfo | null>(null);
  const refreshDemo = useCallback(() => {
    void getDemo().then(setDemo);
  }, []);
  useEffect(refreshDemo, [threadId, refreshDemo]);
  // "#/scene/<id>" opens a demo scene; anything else is a run's thread id
  const sceneId = threadId?.startsWith("scene/") ? threadId.slice("scene/".length) : null;
  const sceneIndex = demo && sceneId ? demo.scenes.findIndex((s) => s.id === sceneId) : -1;
  const scene = demo && sceneIndex >= 0 ? demo.scenes[sceneIndex] : undefined;
  const [recent, setRecent] = useState<RecentRun[] | null>(null);
  const [tickers, setTickers] = useState("GE");
  const [error, setError] = useState<string | null>(null);
  // Company groups whose earlier runs are shown (folded by default).
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const toggle = (key: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  const [theme, setTheme] = useState<Theme>(initialTheme);
  // Same tickers + same filings + same settings replays the earlier analysis; this starts anew.
  const [fresh, setFresh] = useState(false);

  useEffect(() => applyTheme(theme), [theme]);

  useEffect(() => {
    const onHash = () => setThreadId(current());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const refreshRecent = useCallback(() => {
    listAnalyses().then(setRecent).catch((e: unknown) => setError(String(e)));
  }, []);
  useEffect(refreshRecent, [threadId, refreshRecent]);

  async function start(e: React.FormEvent) {
    e.preventDefault();
    const list = tickers.split(/[\s,]+/).filter(Boolean);
    if (!list.length) return;
    try {
      const { thread_id } = await startAnalysis(list, fresh);
      window.location.hash = `/${thread_id}`;
    } catch (err) {
      setError(String(err));
    }
  }

  const runsPanel = (
    <>
          <div className="eyebrow">New analysis</div>
          <form onSubmit={(e) => void start(e)}>
            <label htmlFor="tickers" className="sr-only">
              Tickers
            </label>
            <div className="row">
              <input id="tickers" value={tickers} onChange={(e) => setTickers(e.target.value)} placeholder="Tickers, e.g. GE, NVDA" />
              <button type="submit" className="primary">
                Analyse
              </button>
            </div>
            <label className="muted small">
              <input type="checkbox" checked={fresh} onChange={(e) => setFresh(e.target.checked)} /> Re-analyze
              (new run, even if one exists)
            </label>
          </form>
          {error && <p className="error small">{error}</p>}
          <div className="eyebrow">Recent</div>
          {recent === null && <p className="muted small">Loading…</p>}
          <ul className="recent">
            {groupRuns(recent ?? []).map((g) => {
              const open = expanded.has(g.key) || g.earlier.some((r) => r.thread_id === threadId);
              return (
                <li key={g.key}>
                  <RunLink run={g.latest} active={g.latest.thread_id === threadId} />
                  {g.earlier.length > 0 && (
                    <button
                      type="button"
                      className="earlier-toggle"
                      aria-expanded={open}
                      onClick={() => toggle(g.key)}
                    >
                      {open ? "Hide" : "Show"} {g.earlier.length} earlier {g.earlier.length === 1 ? "run" : "runs"}
                    </button>
                  )}
                  {open && (
                    <ul className="recent earlier">
                      {g.earlier.map((r) => (
                        <li key={r.thread_id}>
                          <RunLink run={r} active={r.thread_id === threadId} />
                        </li>
                      ))}
                    </ul>
                  )}
                </li>
              );
            })}
          </ul>
    </>
  );

  return (
    <>
      <header className="topbar">
        <a className="brand" href="#/" style={{ color: "inherit", textDecoration: "none" }}>
          <span className="brand-mark">
            <Icon name="shield" size={16} />
          </span>
          <span>
            <h1>Restatement Review</h1>
            <p>SEC 10-K figures · verified evidence · human decisions</p>
          </span>
        </a>
        <span className="spacer" />
        <span className="env">Demo · SEC EDGAR data</span>
        <button
          className="icon-button"
          aria-label={`switch to ${theme === "dark" ? "light" : "dark"} mode`}
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
        >
          <Icon name={theme === "dark" ? "sun" : "moon"} size={15} />
        </button>
      </header>
      <div className="layout">
        <aside>
          {demo && (
            <>
              <div className="eyebrow">Demo</div>
              <p className="muted small demo-notice">
                {demo.notice}{" "}
                <span className={`chip ${demo.langsmith === "connected" ? "good" : "neutral"}`}>
                  LangSmith {demo.langsmith}
                </span>
              </p>
              <ol className="scenes">
                {demo.scenes.map((sc, i) => (
                  <li key={sc.id}>
                    <a
                      href={`#/scene/${sc.id}`}
                      className={sc.id === sceneId ? "active" : ""}
                      aria-current={sc.id === sceneId ? "page" : undefined}
                    >
                      <span className="scene-number num">{i + 1}</span>
                      <span>{sc.title}</span>
                      {sc.status && (
                        <span className={`pill ${STATUS_TONE[sc.status] ?? "neutral"}`}>{sc.status.replaceAll("_", " ")}</span>
                      )}
                    </a>
                  </li>
                ))}
              </ol>
            </>
          )}
          {demo ? (
            <details className="other-runs">
              <summary className="eyebrow">Other runs</summary>
              {runsPanel}
            </details>
          ) : (
            runsPanel
          )}
        </aside>
        <main>
          {scene ? (
            <SceneView
              key={`${scene.id}-${scene.thread_id ?? ""}`}
              scene={scene}
              number={sceneIndex + 1}
              onReset={refreshDemo}
              onStatus={refreshDemo}
            />
          ) : sceneId ? (
            <p className="muted">Loading the demo…</p>
          ) : threadId ? (
            <AnalysisView threadId={threadId} onStatus={refreshRecent} />
          ) : (
            <div className="empty">
              <h2>Which restated figures need a human?</h2>
              <p className="muted">
                Each 10-K repeats earlier years, sometimes with different numbers. Code finds every change
                and confirms what it can prove; you decide the rest from verified evidence.
              </p>
              <p className="muted">
                Try <strong>NVDA</strong> (stock splits: confirmed by code, nothing to review) or{" "}
                <strong>GE</strong> (spin-offs and an accounting change: every change needs your call).
              </p>
            </div>
          )}
        </main>
      </div>
    </>
  );
}
