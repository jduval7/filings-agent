import { useEffect, useState } from "react";
import { followReplay, getCase, resetScene } from "../api";
import type { CaseHistory, DemoScene, ProgressEvent } from "../types";
import { AnalysisView } from "./AnalysisView";
import { Icon } from "./Icon";
import { ProgressPanel, START, step } from "./Progress";

/** A demo scene: what it shows and the decision behind it, then the scene itself. */
export function SceneView(props: { scene: DemoScene; number: number; onReset: () => void; onStatus?: () => void }) {
  const { scene, number } = props;
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function reset() {
    setBusy(true);
    setError(null);
    try {
      await resetScene(scene.id);
      props.onReset();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="scene">
      <header className="scene-head panel">
        <div className="scene-title">
          <span className="scene-number num">{number}</span>
          <div>
            <h2>{scene.title}</h2>
            <p>{scene.shows}</p>
            <p className="decision">
              <span className="eyebrow">Decision</span> {scene.decision}
            </p>
          </div>
        </div>
        {(scene.kind === "run" || scene.kind === "case") && (
          <button className="ghost" disabled={busy} onClick={() => void reset()} title="A new run of this scene, from the recordings">
            {busy ? "Resetting…" : "Reset scene"}
          </button>
        )}
      </header>
      {error && <p className="notice bad">{error}</p>}
      {scene.langsmith.length > 0 && <InLangSmith items={scene.langsmith} />}
      {scene.kind === "run" && scene.thread_id && <AnalysisView key={scene.thread_id} threadId={scene.thread_id} onStatus={props.onStatus} />}
      {scene.kind === "replay" && <ReplayView sceneId={scene.id} />}
      {scene.kind === "case" && scene.case && <CaseView key={scene.thread_id ?? ""} discrepancyId={scene.case} />}
    </div>
  );
}

/** Where to look in LangSmith for this scene, and what to point at there. */
function InLangSmith({ items }: { items: DemoScene["langsmith"] }) {
  return (
    <section className="in-langsmith panel" aria-label="In LangSmith">
      <div className="eyebrow">In LangSmith</div>
      <ul>
        {items.map((it) => (
          <li key={it.label}>
            <div className="ls-head">
              <strong>{it.label}</strong>
              {it.links.length ? (
                it.links.map((l) => (
                  <a key={l.url} href={l.url} target="_blank" rel="noreferrer" className="ls-link">
                    <Icon name="search" size={12} /> {l.text}
                    <span className={`chip ${l.where === "public" ? "neutral" : "info"}`}>{l.where}</span>
                  </a>
                ))
              ) : (
                <span className="muted small">connect LangSmith to open (a key in .env, then restart make demo)</span>
              )}
            </div>
            <p className="muted small">{it.point_at}</p>
          </li>
        ))}
      </ul>
    </section>
  );
}

/** A recorded run's progress, replayed at reading pace, clearly labelled as a replay. */
export function ReplayView({ sceneId }: { sceneId: string }) {
  const [progress, setProgress] = useState(START);
  const [playing, setPlaying] = useState(false);
  const [ended, setEnded] = useState<ProgressEvent | null>(null);
  const [run, setRun] = useState(0);

  useEffect(() => {
    if (run === 0) return undefined;
    setProgress(START);
    setEnded(null);
    setPlaying(true);
    return followReplay(
      sceneId,
      (e) => {
        if (e.event === "end") setEnded(e);
        setProgress((p) => step(p, e));
      },
      () => setPlaying(false),
    );
  }, [sceneId, run]);

  return (
    <div className="replay">
      <p className="notice info">
        <Icon name="half" size={13} /> Replay of a recorded run of GE, at reading speed. The run itself took about a
        second because every model answer came from the cache; the order and timing of steps here are the recorded
        events, slowed down.
      </p>
      <button className="primary" disabled={playing} onClick={() => setRun((n) => n + 1)}>
        {playing ? "Replaying…" : run ? "Replay again" : "Play the fan-out"}
      </button>
      {run > 0 && <ProgressPanel progress={progress} />}
      {ended && (
        <p className="muted">
          The run stopped for review: {progress.toReview} changes need a person, {progress.autoApproved} were confirmed by
          code. <a href="#/scene/review">Open it in scene 3</a>.
        </p>
      )}
    </div>
  );
}

/** One case's decision history: every answer ever published, and the one that stands. */
export function CaseView({ discrepancyId }: { discrepancyId: string }) {
  const [c, setC] = useState<CaseHistory | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    getCase(discrepancyId).then(setC).catch((e: unknown) => setError(String(e)));
  }, [discrepancyId]);
  if (error) return <p className="error">{error}</p>;
  if (!c) return <p className="muted">Loading…</p>;
  const stands = c.current[0];
  return (
    <div className="case">
      <h3 className="mono">{c.discrepancy_id}</h3>
      {stands && (
        <div className="published panel">
          <span className="big good">
            <Icon name="shield" size={18} />
          </span>
          <div>
            <strong>
              Stands: {stands.cause.replaceAll("_", " ")} ({stands.decision.replaceAll("_", " ")}
              {stands.actor ? ` by ${stands.actor}` : ""})
            </strong>
            {Boolean(stands.later_escalations) && (
              <div className="small warn-text">
                Escalated {stands.later_escalations} time{stands.later_escalations === 1 ? "" : "s"} since. Shown, not
                applied: only a person's pick replaces a person's pick.
              </div>
            )}
          </div>
        </div>
      )}
      <table className="decisions">
        <thead>
          <tr>
            <th>#</th>
            <th>Run</th>
            <th>Decision</th>
            <th>Cause</th>
            <th>Reviewer</th>
            <th>Decided</th>
          </tr>
        </thead>
        <tbody>
          {c.history.map((h, i) => (
            <tr key={`${h.thread_id}-${i}`} className={h.decision === "escalate" ? "escalated" : ""}>
              <td className="num">{i + 1}</td>
              <td className="mono">{h.thread_id}</td>
              <td>{h.decision.replaceAll("_", " ")}</td>
              <td>{h.cause.replaceAll("_", " ")}</td>
              <td>{h.actor ?? "automatic"}</td>
              <td>{h.decided_at ? new Date(h.decided_at).toLocaleString() : ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted small">
        Every run's answer is kept (append-only). The one that stands: a person's pick beats an escalation, which beats an
        automatic answer; among equals, the newest. A new 10-K that restates the year again is a new case.
      </p>
    </div>
  );
}
