"""Online checks + annotation queue: watching production runs, not just test sets.

After every real run (CLI or API), cheap deterministic checks are posted as feedback on the run's
trace (so LangSmith can chart and filter them), and any run that needs a human goes to the
"filings-agent: review" annotation queue with a rubric for the reviewer.

A run is routed to the queue when it: paused for review, had branch errors, has a restatement
whose cause is unknown, or produced a memo whose numbers failed verification.

Monitoring must never break the thing it monitors: every call here is best-effort.
"""

# LangSmith client methods are partially untyped; this module is a boundary that accepts it.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from typing import Any

# The default queue; the demo routes its runs to its own (LANGSMITH_REVIEW_QUEUE, read per call).
QUEUE_NAME = "filings-agent: review"
QUEUE_INSTRUCTIONS = (
    "Runs land here when the agent needs a human: paused for review, branch errors, an unknown "
    "restatement cause, or a memo that failed number verification. Open the trace, check the "
    "reconcile spans (cause + quoted evidence) and the memo, then score the rubric. If a cause is "
    "wrong, make it a test: open that change's `reconcile` span, Add to Dataset -> "
    'filings-agent/reconcile-qa, and set the reference output to {"cause": "<the right cause>"}. '
    "`make import-qa` then adds it to the regression suite the CI gate runs."
)
RUBRIC_ITEMS: list[dict[str, Any]] = [
    {
        "feedback_key": "human.causes_correct",
        "description": "1 if every restatement cause is right (check the quoted evidence), else 0.",
    },
    {
        "feedback_key": "human.memo_acceptable",
        "description": "1 if you'd send the memo to an analyst as-is, else 0.",
    },
    {
        "feedback_key": "human.root_cause",
        "description": "If something's wrong, which step: extract, reconcile, memo, or data.",
    },
]
# A score type per rubric key: LangSmith only offers a key in the queue's form once it has a
# definition (the same fix as the calibration queues, evals/calibration_queue.py).
STEPS = ("extract", "reconcile", "memo", "data")
FEEDBACK_CONFIGS: dict[str, dict[str, Any]] = {
    "human.causes_correct": {
        "type": "categorical",
        "categories": [{"value": 1, "label": "right"}, {"value": 0, "label": "wrong"}],
    },
    "human.memo_acceptable": {
        "type": "categorical",
        "categories": [{"value": 1, "label": "send as-is"}, {"value": 0, "label": "needs work"}],
    },
    "human.root_cause": {
        "type": "categorical",
        "categories": [{"value": i, "label": step} for i, step in enumerate(STEPS)],
    },
}
FINISHED = frozenset({"published", "done"})
_NS = uuid.UUID("3c8e2a91-7f4d-4b6a-8e1c-5d2f9a0b7e64")


def checks(summary: dict[str, Any]) -> dict[str, float]:
    """Deterministic health checks from a run summary (no LLM, no cost)."""
    resolutions: dict[str, Any] = summary.get("resolutions") or {}
    memo: dict[str, Any] = summary.get("memo") or {}
    return {
        "online.branch_errors": float(len(summary.get("errors") or {})),
        "online.unknown_causes": float(
            sum(1 for r in resolutions.values() if r.get("cause") == "unknown")
        ),
        "online.unverified_quotes": float(
            sum(1 for r in resolutions.values() if not r.get("verified", True))
        ),
        "online.memo_verified": float(bool(memo.get("verified"))) if memo else -1.0,
        "online.awaiting_review": float(summary.get("status") == "awaiting_review"),
        "online.cost_usd": float(summary.get("cost_usd") or 0.0),
    }


def review_scores(summary: dict[str, Any]) -> tuple[dict[str, float], list[dict[str, str]]]:
    """What the reviewers did on a finished run, as numbers LangSmith can chart over time.

    review.agent_agreement is the reviewer's pick vs the agent's (hidden) answer, over the cases
    where the reviewer picked a cause: human-measured accuracy on exactly the cases that needed a
    human. It leaves out escalations (often the hardest cases), so it reads high: read it with
    review.picked (its denominator) and review.escalated. Escalations and research rounds show
    where the agent's evidence wasn't enough; auto_approved shows how much review triage saved.
    """
    resolutions: dict[str, Any] = summary.get("resolutions") or {}
    decisions: dict[str, Any] = summary.get("decisions") or {}
    detail: list[dict[str, str]] = []
    agree: list[bool] = []
    for k, d in sorted(decisions.items()):
        res: dict[str, Any] = resolutions.get(k) or {}
        agent = str(res.get("cause", ""))
        picked = str(d.get("correct_cause", ""))
        if picked:
            agree.append(picked == agent)
        detail.append({"id": k, "action": str(d.get("action")), "picked": picked, "agent": agent})
    scores: dict[str, float] = {
        "review.decisions": float(len(decisions)),
        "review.escalated": float(sum(d.get("action") == "escalate" for d in decisions.values())),
        "review.research_rounds": float(summary.get("research_rounds") or 0),
        "review.auto_approved": float(
            sum(1 for r in resolutions.values() if r.get("lane") == "auto")
        ),
    }
    scores["review.picked"] = float(len(agree))
    if agree:
        scores["review.agent_agreement"] = sum(agree) / len(agree)
    return scores, detail


def post_review(client: Any, run_id: str, summary: dict[str, Any], session_id: Any) -> None:
    """Reviewer decisions as feedback on the finished run's trace (idempotent ids). Source "api":
    the SDK accepts only "api" and "model" (found live; "app" was rejected)."""
    scores, detail = review_scores(summary)
    comment = json.dumps(detail)
    for key, value in scores.items():
        client.create_feedback(
            run_id,
            key=key,
            score=value,
            trace_id=run_id,
            session_id=session_id,
            feedback_id=uuid.uuid5(_NS, f"{run_id}|{key}"),
            feedback_source_type="api",  # human decisions, submitted through the API/CLI
            comment=comment,
        )


def needs_human(c: dict[str, float]) -> list[str]:
    reasons: list[str] = []
    if c["online.awaiting_review"]:
        reasons.append("awaiting review")
    if c["online.branch_errors"] > 0:
        reasons.append(f"{int(c['online.branch_errors'])} branch error(s)")
    if c["online.unknown_causes"] > 0:
        reasons.append(f"{int(c['online.unknown_causes'])} unknown cause(s)")
    if c["online.memo_verified"] == 0.0:
        reasons.append("memo numbers not verified")
    return reasons


def ensure_feedback_configs(client: Any) -> None:
    existing = {str(f.feedback_key): f for f in client.list_feedback_configs()}
    for key, config in FEEDBACK_CONFIGS.items():
        if key not in existing:
            client.create_feedback_config(key, feedback_config=config)
        elif dict(existing[key].feedback_config) != config:
            client.update_feedback_config(key, feedback_config=config)


def ensure_queue(client: Any) -> Any:
    ensure_feedback_configs(client)
    name = os.environ.get("LANGSMITH_REVIEW_QUEUE", QUEUE_NAME)
    for q in client.list_annotation_queues(name=name):
        return q
    return client.create_annotation_queue(
        name=name,
        description="Agent runs that need a human",
        rubric_instructions=QUEUE_INSTRUCTIONS,
        rubric_items=RUBRIC_ITEMS,
    )


def post(client: Any, run_id: str, summary: dict[str, Any], project_name: str) -> list[str]:
    """Post check feedback on the run's trace; route to the queue if a human is needed."""
    c = checks(summary)
    session_id = client.read_project(project_name=project_name).id
    for key, value in c.items():
        client.create_feedback(
            run_id,
            key=key,
            score=value,
            trace_id=run_id,
            session_id=session_id,
            feedback_id=uuid.uuid5(
                _NS, f"{run_id}|{key}"
            ),  # idempotent: re-posting can't duplicate
            feedback_source_type="model",
            comment="automatic online check",
        )
    # finished: "published" (with a ledger) or "done" (without). Found live: checking only "done"
    # meant production runs, which publish, never posted reviewer feedback.
    finished = summary.get("status") in FINISHED
    if finished and (summary.get("decisions") or summary.get("resolutions")):
        post_review(client, run_id, summary, session_id)
    reasons = needs_human(c)
    if reasons:
        queue = ensure_queue(client)
        _add_to_queue_when_ingested(client, queue.id, run_id)
        client.create_feedback(
            run_id,
            key="online.routed_to_review",
            score=1,
            trace_id=run_id,
            session_id=session_id,
            comment="; ".join(reasons),
            feedback_id=uuid.uuid5(_NS, f"{run_id}|routed"),
        )
    return reasons


def _add_to_queue_when_ingested(
    client: Any, queue_id: Any, run_id: str, attempts: int = 6, wait_s: float = 2.0
) -> None:
    """Traces upload in the background; right after a run the server may not know it yet (404).
    Retry with backoff instead of losing the routing."""
    for i in range(attempts):
        try:
            client.add_runs_to_annotation_queue(queue_id, run_ids=[run_id])
            return
        except Exception as e:
            if "not found" not in str(e).lower() or i == attempts - 1:
                raise
            time.sleep(wait_s * (i + 1))


def _flush_traces() -> None:
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers

        wait_for_all_tracers()
    except Exception as e:  # best effort: monitoring must never break the run, but say so
        print(f"[online] trace flush failed: {type(e).__name__}: {e}", file=sys.stderr)


def monitor(summary: dict[str, Any], client: Any | None = None) -> list[str]:
    """Best-effort: called after every real run. Never raises."""
    run_id = summary.get("run_id")
    if not run_id or os.environ.get("LANGSMITH_TRACING", "").lower() not in {"1", "true"}:
        return []
    try:
        _flush_traces()  # the run must exist server-side before feedback/queue calls
        if client is None:
            from langsmith import Client

            client = Client()
        return post(
            client, str(run_id), summary, os.environ.get("LANGSMITH_PROJECT", "filings-agent")
        )
    except Exception as e:  # monitoring must never break the run it monitors
        print(f"[online] checks not posted: {type(e).__name__}: {e}", file=sys.stderr)
        return []
