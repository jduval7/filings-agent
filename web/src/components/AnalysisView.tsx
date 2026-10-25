import { useCallback, useEffect, useRef, useState } from "react";
import Markdown from "react-markdown";
import { followProgress, getAnalysis, getProjectUrl, resumeAnalysis, submitReview } from "../api";
import { problem, toReview, type Draft } from "../decisions";
import { titleOf } from "../format";
import type { Analysis } from "../types";

export { step } from "./Progress";
import { Icon, type IconName } from "./Icon";
import { ProgressPanel, START, step, type Progress } from "./Progress";
import { ReviewCard } from "./ReviewCard";

const STATUS: Record<string, { tone: string; icon: IconName }> = {
  running: { tone: "info", icon: "half" },
  resuming: { tone: "info", icon: "half" },
  awaiting_review: { tone: "warn", icon: "flag" },
  published: { tone: "good", icon: "shield" },
  done: { tone: "good", icon: "check" },
  interrupted: { tone: "bad", icon: "cross" },
  failed: { tone: "bad", icon: "cross" },
  not_found: { tone: "neutral", icon: "dash" },
};

const LIVE = new Set(["running", "resuming"]);

/** Turns an API error ("409: {"detail": ...}") into a sentence for the reviewer. */
export function humanError(e: unknown): string {
  const text = String(e instanceof Error ? e.message : e);
  const m = /^(\d{3}): (.*)$/s.exec(text);
  if (!m) return text;
  const [, code, body] = m;
  try {
    const detail = (JSON.parse(body ?? "") as { detail?: unknown }).detail;
    if (code === "409" && typeof detail === "string") return detail;
    if (detail && typeof detail === "object") {
      const d = detail as Record<string, unknown>;
      const parts = Object.entries((d.invalid_decisions ?? {}) as Record<string, unknown>).map(
        ([k, v]) => `${k}: ${String(v)}`,
      );
      if (Array.isArray(d.missing_decisions) && d.missing_decisions.length)
        parts.push(`no decision for ${d.missing_decisions.join(", ")}`);
      if (parts.length) return parts.join("; ");
    }
    return typeof detail === "string" ? detail : text;
  } catch {
    return text;
  }
}

interface Props {
  threadId: string;
  /** Called when the run's status changes (the sidebar refreshes its list). */
  onStatus?: (status: string) => void;
}

export function AnalysisView({ threadId, onStatus }: Props) {
  const [a, setA] = useState<Analysis | null>(null);
  const [progress, setProgress] = useState<Progress>(START);
  const [drafts, setDrafts] = useState<Record<string, Draft>>({});
  const [loadError, setLoadError] = useState<string | null>(null); // nothing to show at all
  const [notice, setNotice] = useState<string | null>(null); // inline: the page stays
  const [runError, setRunError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [resuming, setResuming] = useState(false);
  // statements already read when Resume was clicked: what the resume didn't read again
  const [resumedFrom, setResumedFrom] = useState<number | null>(null);
  const [projectUrl, setProjectUrl] = useState<string | null>(null);
  const [reconnect, setReconnect] = useState(0);
  const shown = useRef(threadId); // the run on screen: late responses for another run are dropped
  shown.current = threadId;
  const loaded = useRef(false);

  useEffect(() => {
    void getProjectUrl().then(setProjectUrl);
  }, []);

  const refresh = useCallback(async (): Promise<Analysis | null> => {
    try {
      const next = await getAnalysis(threadId);
      if (shown.current !== threadId) return null;
      setA(next);
      loaded.current = true;
      setLoadError(null);
      return next;
    } catch (e) {
      if (shown.current !== threadId) return null;
      if (loaded.current) setNotice(`Couldn't refresh: ${humanError(e)}`);
      else setLoadError(humanError(e));
      return null;
    }
  }, [threadId]);

  const follow = useCallback(() => {
    return followProgress(
      threadId,
      (e) => {
        if (e.event === "end" && e.status === "failed") setRunError(String(e.error ?? "the run failed"));
        setProgress((p) => step(p, e));
      },
      () =>
        void refresh().then((next) => {
          // the stream closed but the run is still going (a network blip): follow it again
          if (next && LIVE.has(next.status)) setTimeout(() => setReconnect((n) => n + 1), 1500);
        }),
    );
  }, [threadId, refresh]);

  useEffect(() => {
    setA(null);
    setDrafts({});
    setLoadError(null);
    setNotice(null);
    setRunError(null);
    setProgress(START);
    loaded.current = false;
    void refresh();
  }, [refresh]);

  const live = a ? LIVE.has(a.status) : false;
  useEffect(() => (live ? follow() : undefined), [live, follow, reconnect]);

  const status = a?.status;
  useEffect(() => {
    if (status) onStatus?.(status);
  }, [status, onStatus]);

  if (loadError) return <p className="error">{loadError}</p>;
  if (!a) return <p className="muted">Loading…</p>;

  const cards = a.awaiting_review;
  const review = toReview(cards, drafts);
  const undecided = cards.filter((c) => problem(c, drafts[c.id]) !== null);
  const decided = cards.length - undecided.length;
  const byCode = Object.entries(a.resolutions).filter(([, r]) => !r.needs_review);
  const serverProblems = Object.entries(a.review_problems ?? {});

  async function submit() {
    if (!review || !a) return;
    setSubmitting(true);
    setNotice(null);
    try {
      await submitReview(threadId, review, a.review_token);
      setDrafts({});
      setRunError(null);
      setProgress({ ...START, last: "resuming with your decisions…" });
      setA((prev) => (prev ? { ...prev, status: "resuming", awaiting_review: [] } : prev));
    } catch (e) {
      setNotice(humanError(e)); // keep the page and the drafts
      void refresh(); // e.g. the cards changed in another tab: show the current ones
    } finally {
      setSubmitting(false);
    }
  }

  async function resume() {
    setResuming(true);
    setNotice(null);
    try {
      setResumedFrom(a?.branches_done ?? null);
      await resumeAnalysis(threadId);
      setRunError(null);
      setProgress({ ...START, last: "resuming from the last checkpoint…" });
      setA((prev) => (prev ? { ...prev, status: "resuming" } : prev));
    } catch (e) {
      setNotice(humanError(e));
      void refresh();
    } finally {
      setResuming(false);
    }
  }

  const autoApproved = Object.values(a.resolutions).filter((r) => r.lane === "auto").length;
  const reviewed = Object.values(a.resolutions).filter((r) => r.needs_review).length;
  const tone = STATUS[a.status] ?? { tone: "info", icon: "half" as IconName };

  return (
    <div className="analysis">
      <header className="analysis-head">
        <h2>{a.tickers.join(" · ")}</h2>
        <span className={`pill ${tone.tone}`}>
          <Icon name={tone.icon} size={12} /> {a.status.replaceAll("_", " ")}
        </span>
        <span className="mono small">run {a.thread_id}</span>
        {projectUrl && a.last_run_id && (
          <a className="trace-link" href={`${projectUrl}/r/${a.last_run_id}`} target="_blank" rel="noreferrer">
            <Icon name="search" size={12} /> Open trace in LangSmith
          </a>
        )}
      </header>

      <div className="kpis">
        <Kpi label="Figures read" value={a.facts} sub={`${a.branches} statements`} />
        <Kpi label="Restated" value={a.discrepancies} sub="vs earlier 10-Ks" />
        <Kpi label="Confirmed by code" value={autoApproved} sub="no human needed" />
        <Kpi
          label="Needs your call"
          value={a.status === "awaiting_review" ? cards.length : reviewed}
          sub={a.status === "awaiting_review" ? "waiting now" : "reviewed"}
          attention={a.status === "awaiting_review"}
        />
        <Kpi
          label="Paid for model calls"
          value={`$${a.paid_usd.toFixed(2)}`}
          sub={`${a.cached_branches} answers from cache (first run cost $${a.cost_usd.toFixed(2)})`}
        />
      </div>

      {live && <ProgressPanel progress={progress} />}

      {resumedFrom !== null && !live && a.status !== "interrupted" && (
        <p className="notice good">
          <Icon name="check" size={13} /> Resumed from the last checkpoint: {resumedFrom} of {a.branches} statements were
          already read and weren't read again; the other {a.branches - resumedFrom} were read on resume. Paid: $
          {a.paid_usd.toFixed(2)}.
        </p>
      )}

      {a.status === "interrupted" && (
        <section className="stopped panel">
          <div>
            <strong>
              This run stopped part-way: {a.branches_done} of {a.branches} statements were read before it
              stopped.
            </strong>
            <p className="muted small">
              Every finished step is checkpointed and every model answer is cached, so resuming continues from
              the last checkpoint and nothing already done is paid for again.
            </p>
          </div>
          <button className="primary" disabled={resuming} onClick={() => void resume()}>
            {resuming ? "Resuming…" : "Resume run"}
          </button>
        </section>
      )}

      {runError && (
        <p className="notice bad" role="alert">
          <Icon name="cross" size={13} /> The run failed: {runError}
        </p>
      )}
      {notice && (
        <p className="notice warn" role="alert">
          <Icon name="flag" size={13} /> {notice}
        </p>
      )}

      {a.status === "awaiting_review" && (
        <section>
          <div className="section-head">
            <h2>
              {cards.length === 1 ? "1 change needs" : `${cards.length} changes need`} your call
            </h2>
          </div>
          <p className="lede">
            Code couldn't confirm these. For each, pick the cause the evidence supports, escalate if none
            fit, or ask for research. The agent's own answer is hidden on purpose: you judge the evidence,
            not a proposal.
          </p>
          {serverProblems.length > 0 && (
            <div className="notice warn" role="alert">
              <Icon name="flag" size={13} /> Your last answer couldn't be applied:{" "}
              {serverProblems.map(([k, v]) => `${k}: ${v}`).join("; ")}
            </div>
          )}
          {cards.map((c) => (
            <ReviewCard
              key={c.id}
              card={c}
              draft={drafts[c.id]}
              onChange={(d) => setDrafts((all) => ({ ...all, [c.id]: d }))}
            />
          ))}
          <div className="submit-bar panel">
            <span>
              <span className="count num">
                {decided} of {cards.length} decided
              </span>
              <span
                className="bar"
                role="progressbar"
                aria-label="decisions"
                aria-valuemin={0}
                aria-valuemax={cards.length}
                aria-valuenow={decided}
              >
                <div style={{ width: `${(100 * decided) / Math.max(cards.length, 1)}%` }} />
              </span>
              {undecided.length > 0 && undecided.length <= 3 && (
                <span className="muted small"> · still open: {undecided.map((c) => titleOf(c.id)).join(", ")}</span>
              )}
            </span>
            <button className="primary" disabled={!review || submitting} onClick={() => void submit()}>
              {submitting ? "Submitting…" : "Submit decisions"}
            </button>
          </div>
        </section>
      )}

      {byCode.length > 0 && (
        <details className="by-code panel">
          <summary>{byCode.length} decided by code (no human needed)</summary>
          <ul>
            {byCode.map(([id, r]) => (
              <li key={id}>
                <span className="mono">{id}</span>:{" "}
                {r.lane === "auto"
                  ? `${(r.cause ?? "").replaceAll("_", " ")}, confirmed by code (ratio and quote match)`
                  : "under 0.5%, auto-accepted as immaterial"}
              </li>
            ))}
          </ul>
        </details>
      )}

      {a.publication && (
        <section>
          <div className="section-head">
            <h2>Published</h2>
          </div>
          <div className="published panel">
            <span className="big good">
              <Icon name="shield" size={18} />
            </span>
            <div>
              <strong>
                {a.publication.facts} figures and {a.publication.resolutions} explanations published, exactly
                once
              </strong>
              <div className="muted small">
                {new Date(a.publication.published_at).toLocaleString()} · run {a.thread_id} · every quote
                verified against its filing
              </div>
            </div>
          </div>
          {Object.keys(a.decisions).length > 0 && (
            <table className="decisions">
              <thead>
                <tr>
                  <th>Change</th>
                  <th>Decision</th>
                  <th>Cause</th>
                  <th>Note</th>
                  <th>Reviewer</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(a.decisions).map(([id, d]) => (
                  <tr key={id} className={d.action === "escalate" ? "escalated" : ""}>
                    <td className="mono">{id}</td>
                    <td>{d.action.replaceAll("_", " ")}</td>
                    <td>{d.correct_cause?.replaceAll("_", " ") ?? "unresolved"}</td>
                    <td>{d.note}</td>
                    <td>{d.actor ?? ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {a.memo && (
            <div className="memo panel">
              <Markdown>{a.memo.markdown}</Markdown>
            </div>
          )}
        </section>
      )}
    </div>
  );
}

function Kpi(props: { label: string; value: number | string; sub: string; attention?: boolean }) {
  return (
    <div className={`kpi${props.attention ? " attention" : ""}`}>
      <div className="label">{props.label}</div>
      <div className="value">{props.value}</div>
      <div className="sub">{props.sub}</div>
    </div>
  );
}
