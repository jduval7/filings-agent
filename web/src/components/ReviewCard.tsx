import { emptyDraft, overridesRule, problem, type Draft } from "../decisions";
import { formatAmount, titleOf } from "../format";
import { CAUSES, type Card, type Cause, type Fit, type ResearchFinding } from "../types";
import { Icon, type IconName } from "./Icon";

const FIT: Record<Fit, { label: string; tone: string; icon: IconName }> = {
  fits: { label: "fits", tone: "good", icon: "check" },
  possible: { label: "possible", tone: "neutral", icon: "half" },
  ruled_out: { label: "ruled out", tone: "bad", icon: "cross" },
  no_evidence: { label: "no evidence", tone: "chip muted", icon: "dash" },
};

const label = (cause: string) => cause.replaceAll("_", " ");
const host = (url: string) => {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
};

function Finding({ f }: { f: ResearchFinding }) {
  const supports = f.stance === "supports";
  return (
    <div className="citation research">
      <q>{f.quote}</q>
      <div className="meta">
        <span className={`chip ${supports ? "good" : "bad"}`}>
          <Icon name={supports ? "check" : "cross"} size={11} /> research: {f.stance}
        </span>
        <span>
          <Icon name="doc" size={12} />{" "}
          <a href={f.source} target="_blank" rel="noreferrer" className="source">
            {host(f.source)}
          </a>
        </span>
        <span>read {f.read}</span>
        <span>
          <Icon name="check" size={11} /> quote verified in source
        </span>
      </div>
      {/* A verified quote can still come with a wrong conclusion: the reading stays collapsed. */}
      <details>
        <summary>investigator's reading (unverified)</summary>
        <p>{f.unverified_reading}</p>
      </details>
    </div>
  );
}

interface Props {
  card: Card;
  draft: Draft | undefined;
  onChange: (draft: Draft) => void;
}

/** One pending change. Every candidate cause with its evidence and code's verdict; the agent's
 *  own answer is never shown (reviewers anchor on AI proposals). */
export function ReviewCard({ card, draft, onChange }: Props) {
  const d = draft ?? emptyDraft();
  const set = (patch: Partial<Draft>) => onChange({ ...d, ...patch });
  const roundsLeft = card.research_rounds_left ?? 0;
  const warning = overridesRule(card, d);
  const missing = problem(card, draft);
  const metric = card.id.split(":")[1] ?? "";
  const first = card.values[0]?.[1];
  const last = card.values[card.values.length - 1]?.[1];
  const pct = first && last && Number(first) !== 0 ? (Number(last) - Number(first)) / Math.abs(Number(first)) : null;

  return (
    <article className="card panel" aria-label={card.id}>
      <div className="card-head">
        <div>
          <h3>{titleOf(card.id)}</h3>
          <p className="change">
            {first !== undefined ? formatAmount(first, metric) : ""} <span className="muted">→</span>{" "}
            {last !== undefined ? formatAmount(last, metric) : ""}
            {pct !== null && (
              <span className="delta">
                {pct > 0 ? "+" : ""}
                {(100 * pct).toFixed(1)}%
              </span>
            )}
          </p>
          <div className="filings">
            {card.values.map(([accn, v]) => (
              <span key={accn}>
                <span className="mono">{accn}</span> {formatAmount(v, metric)}
                {card.changed_in.includes(accn) ? " · changed here" : ""}
              </span>
            ))}
          </div>
        </div>
        <span className="pill warn">
          <Icon name="flag" size={12} /> needs your call
        </span>
      </div>

      <table className="causes">
        <thead>
          <tr>
            <th>Possible cause</th>
            <th>Fits the numbers?</th>
            <th>Why (code's check)</th>
          </tr>
        </thead>
        <tbody>
          {card.candidates.map((c) => (
            <tr key={c.cause}>
              <td>{label(c.cause)}</td>
              <td>
                <span className={`chip ${FIT[c.fit].tone}`}>
                  <Icon name={FIT[c.fit].icon} size={11} />
                  {FIT[c.fit].label}
                </span>
              </td>
              <td>{c.why.replaceAll("_", " ")}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="evidence">
        {card.candidates
          .filter((c) => c.evidence.length || c.research?.length)
          .map((c) => (
            <section key={c.cause}>
              <h4>{label(c.cause)}: evidence</h4>
              {c.evidence.map((q) => (
                <div className="citation" key={q}>
                  <q>{q}</q>
                  <div className="meta">
                    <span>
                      <Icon name="doc" size={12} /> SEC 10-K excerpt
                    </span>
                    <span>
                      <Icon name="check" size={11} /> from the filing that changed the number
                    </span>
                  </div>
                </div>
              ))}
              {c.research?.map((f) => <Finding key={f.quote} f={f} />)}
              {c.research_not_shown ? (
                <p className="hint">{c.research_not_shown} more findings not shown</p>
              ) : null}
            </section>
          ))}
        {card.other_causes?.map((o) => (
          <section key={o.quote}>
            <h4>other: {o.name}</h4>
            <Finding f={o} />
          </section>
        ))}
        {card.research_rounds?.map((r) => (
          <div key={r.round} className="round">
            <p className="muted">
              <Icon name="search" size={12} /> Research round {r.round}
              {r.hint ? ` (hint: ${r.hint})` : ""}: {r.findings} verified findings,{" "}
              {r.discarded_unverified} discarded as unverifiable.{" "}
              {r.error ? <span className="error">Failed: {r.error}</span> : null}
            </p>
            {(r.looked_at || r.open_questions) && (
              <details>
                <summary>where the investigators looked, and what's still open</summary>
                <p>{r.looked_at}</p>
                {r.open_questions && (
                  <p>
                    <strong>Still open:</strong> {r.open_questions}
                  </p>
                )}
              </details>
            )}
          </div>
        ))}
      </div>

      <fieldset className="decide" style={{ border: "1px solid var(--line)" }}>
        <legend className="sr-only">Your call</legend>
        <div className="decide-title">Your call</div>
        <div className="segmented" role="radiogroup" aria-label="action">
          {(
            [
              ["decide", "Pick the cause"],
              ["escalate", "Escalate"],
              ["research", `Research further (${roundsLeft} left)`],
            ] as const
          ).map(([mode, text]) => (
            <label key={mode}>
              <input
                type="radio"
                name={`${card.id}-mode`}
                checked={d.mode === mode}
                disabled={mode === "research" && roundsLeft <= 0}
                onChange={() => set({ mode })}
              />
              {text}
            </label>
          ))}
        </div>

        {d.mode === "decide" && (
          <>
            <div className="causes-pick" role="radiogroup" aria-label="cause">
              {CAUSES.map((cause: Cause) => (
                <label key={cause}>
                  <input
                    type="radio"
                    name={`${card.id}-cause`}
                    checked={d.cause === cause}
                    onChange={() => set({ cause })}
                  />
                  {label(cause)}
                </label>
              ))}
            </div>
            {warning && (
              <p className="notice warn">
                <Icon name="flag" size={13} /> Code rules this out: {warning.replaceAll("_", " ")}
              </p>
            )}
            <label className="check">
              <input
                type="checkbox"
                checked={d.keepOriginal}
                onChange={(e) => set({ keepOriginal: e.target.checked })}
              />
              keep the originally reported figure (default: use the restated one)
            </label>
          </>
        )}

        <textarea
          placeholder={
            d.mode === "research"
              ? "Optional hint for the research, e.g. check the 8-K for the spin-off"
              : d.mode === "escalate"
                ? "Why none of these fit"
                : "Note for the audit record (optional)"
          }
          value={d.note}
          onChange={(e) => set({ note: e.target.value })}
        />
        {missing && missing !== "not decided" && <p className="hint">{missing}</p>}
      </fieldset>
    </article>
  );
}
