# LangGraph ships without type stubs; this module is the boundary that accepts that.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
"""The analysis graph.

  plan --Send xN--> extract --> find_discrepancies --Send xM--> reconcile --> review --> memo
                                        (immaterial: auto-resolved, no LLM)   interrupt()   |
                                                                                  publish <-+

reconcile ends with triage (code, steps/triage.py): answers whose cause, quote and number are
all confirmed by an accounting rule can skip review when `auto_approve` is on.

review shows a card per change (steps/card.py). On a hard case the reviewer can ask for research
instead of deciding: review --Send--> research --> review, with the findings added to the card.
Research runs one bounded investigation per candidate cause (research/agent.py). Bounded by
`max_research_rounds` per case.

Orchestration lives in code, not in a prompt: the fan-out size comes from the data, retries and
failure handling are node policies, and the LLM only ever does the one job in `extract`.

Dependencies (filing store, extractor) are injected through LangGraph's runtime context, so the
same graph runs with Bedrock in prod and with fakes in tests: swapping the extractor is config.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_config, get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy, Send, interrupt

from filings_agent.datasets import STATEMENT_METRICS
from filings_agent.domain import FilingRef, Metric
from filings_agent.graph.state import (
    AnalysisInput,
    AnalysisState,
    Decision,
    ExtractInput,
    ReconcileInput,
    ResearchInput,
    ResearchRequest,
)
from filings_agent.llm.transient import TRANSIENT_CODES as TRANSIENT_CODES  # re-exported
from filings_agent.llm.transient import is_transient as is_transient  # re-exported
from filings_agent.research.findings import ResearchResult
from filings_agent.sec.gold import find_discrepancies
from filings_agent.sec.sections import locate_statements
from filings_agent.steps.card import CANDIDATES, build_card, sentences_for
from filings_agent.steps.extract import Extractor, ExtractTask
from filings_agent.steps.memo import MemoWriter, build_table, table_text
from filings_agent.steps.reconcile import (
    Cause,
    Reconciler,
    ReconcileTask,
    Resolution,
    auto_resolution,
    evidence_for,
    is_material,
)
from filings_agent.steps.triage import apply_triage
from filings_agent.storage.ledger import Correction, Decided, Ledger, filing_set

# Part of every run's identity (graph/runner.py: run_identity). Bump it when a change to the graph
# or its steps changes what a run would publish, so the same request starts a new analysis
# instead of replaying one made by the old code.
GRAPH_VERSION = 1


class Researching(Protocol):
    def research(self, task: ReconcileTask, hint: str, round_: int) -> ResearchResult: ...


class FilingStore(Protocol):
    def filings(self, ticker: str) -> list[FilingRef]: ...
    def markdown(self, filing: FilingRef) -> str: ...


@dataclass(frozen=True)
class Deps:
    store: FilingStore
    extractor: Extractor
    metrics: tuple[Metric, ...]
    reconciler: Reconciler | None = None  # None: wave 2 only auto-resolves, never calls an LLM
    ledger: Ledger | None = None  # None: publish is a no-op (tests of earlier stages)
    memo_writer: MemoWriter | None = None  # None: no memo
    rep: int = 0  # cache-key repetition: a new value forces fresh LLM calls (demos, variance)
    auto_approve: bool = False  # skip review for answers triage fully confirms
    researcher: Researching | None = None  # None: the card offers no "research" action
    max_research_rounds: int = 2  # per case; after that the reviewer must decide or escalate


# Fallback dependencies for servers that can't put Python objects in the runtime context
# (LangGraph Studio / `langgraph dev` pass JSON). Set by build_graph(default_deps=...).
_default_deps: Callable[[], Deps] | None = None


def _deps(runtime: Runtime[Deps]) -> Deps:
    ctx = cast(object, runtime.context)  # typed as Deps; a server may hand us None or a dict
    if isinstance(ctx, Deps):
        return ctx
    if _default_deps is None:
        raise RuntimeError("no Deps in the runtime context and no default_deps configured")
    return _default_deps()


# ---- nodes ------------------------------------------------------------------------------------


def plan(state: AnalysisState, runtime: Runtime[Deps]) -> dict[str, Any]:
    deps = _deps(runtime)
    tasks: list[ExtractTask] = []
    for ticker in state.get("tickers", []):
        for filing in deps.store.filings(ticker):
            sections = locate_statements(deps.store.markdown(filing))
            for kind, section in sections.items():
                metrics = tuple(m for m in STATEMENT_METRICS[kind] if m in deps.metrics)
                if not metrics:
                    continue
                tasks.append(
                    ExtractTask(
                        task_id=f"{filing.accession_no}:{kind.value}",
                        ticker=ticker,
                        accession_no=filing.accession_no,
                        fiscal_year_end=filing.period_of_report,
                        statement=kind.value,
                        section_text=section.text,
                        metrics=metrics,
                    )
                )
    # The task ids let a UI lay out the whole fan-out before any branch finishes.
    get_stream_writer()(
        {"event": "planned", "branches": len(tasks), "tasks": [t.task_id for t in tasks]}
    )
    return {"tasks": tasks}


def fan_out(state: AnalysisState) -> list[Send]:
    """One branch per statement. The number of branches is decided by the data, not the code."""
    return [Send("extract", ExtractInput(task=t)) for t in state.get("tasks", [])]


def extract(inp: ExtractInput, runtime: Runtime[Deps]) -> dict[str, Any]:
    """Two failure classes, handled differently on purpose:

    - transient (throttling/outage): re-raised so RetryPolicy retries with backoff. If it keeps
      failing, the run stops; finished branches are already checkpointed, so a resume re-runs
      only the unfinished ones.
    - anything else (bad model output, a parsing bug): recorded for this branch; siblings
      carry on. SpendLimitExceeded is a BaseException, so it is never caught here.

    (LangGraph 1.2's node `error_handler` would be the natural fit, but it doesn't fire for
    `Send` branches; see DECISIONS.md.)
    """
    task = inp["task"]
    try:
        deps = _deps(runtime)
        res = deps.extractor.extract(task, rep=deps.rep)
    except Exception as e:
        if is_transient(e):
            raise
        msg = f"{type(e).__name__}: {e}"
        get_stream_writer()({"event": "branch_failed", "task_id": task.task_id, "error": msg})
        return {"errors": {task.task_id: msg}}
    out: dict[str, Any] = {
        "facts": {f.id: f for f in res.facts},
        "rejected": {task.task_id: res.rejected} if res.rejected else {},
        "usage": {task.task_id: res.usage} if res.usage else {},
    }
    if res.error:
        out["errors"] = {task.task_id: res.error}
    get_stream_writer()(
        {
            "event": "branch_done",
            "task_id": task.task_id,
            "facts": len(res.facts),
            "rejected": len(res.rejected),
            "error": res.error,
        }
    )
    return out


def find_discrepancies_node(state: AnalysisState, runtime: Runtime[Deps]) -> dict[str, Any]:
    """Plain code, no LLM. Decides wave 2's fan-out from wave 1's results."""
    deps = _deps(runtime)
    filings = {f.accession_no: f for t in state.get("tickers", []) for f in deps.store.filings(t)}
    facts = list(state.get("facts", {}).values())
    found = find_discrepancies(facts, {a: f.filed for a, f in filings.items()})
    resolutions: dict[str, Resolution] = {}
    tasks: list[ReconcileTask] = []
    md_cache: dict[str, str] = {}

    def markdown_of(accession_no: str) -> str:
        if accession_no not in md_cache:
            md_cache[accession_no] = deps.store.markdown(filings[accession_no])
        return md_cache[accession_no]

    for d in found:
        if not is_material(d) or deps.reconciler is None:
            resolutions[d.id] = auto_resolution(d.id, d)
            continue
        tasks.append(ReconcileTask(d.id, d, evidence_for(d, markdown_of)))
    get_stream_writer()(
        {
            "event": "wave1_done",
            "facts": len(facts),
            "errors": len(state.get("errors", {})),
            "discrepancies": len(found),
            "material": len(tasks),
        }
    )
    return {
        "discrepancies": {d.id: d for d in found},
        "resolutions": resolutions,
        "reconcile_tasks": tasks,
    }


def fan_out_reconcile(state: AnalysisState) -> list[Send] | str:
    tasks = state.get("reconcile_tasks", [])
    return [Send("reconcile", ReconcileInput(task=t)) for t in tasks] or "review"


def reconcile(inp: ReconcileInput, runtime: Runtime[Deps]) -> dict[str, Any]:
    task = inp["task"]
    deps = _deps(runtime)
    reconciler = deps.reconciler
    if reconciler is None:  # only routed here when one is configured; not an assert (-O strips it)
        raise RuntimeError("reconcile reached without a reconciler configured")
    try:
        res = reconciler.reconcile(task, rep=deps.rep)
    except Exception as e:
        if is_transient(e):
            raise
        d = task.discrepancy
        res = Resolution(
            task.discrepancy_id, Cause.UNKNOWN,
            f"Reconcile failed: {type(e).__name__}: {e}", "", True, False,
            d.values[-1][1], d.values[-1][0], True, None, str(e),
        )  # fmt: skip
    res = apply_triage(task.discrepancy, res, deps.auto_approve)
    lane = res.triage.lane if res.triage else "review"
    get_stream_writer()(
        # No cause in the event: it streams to the review UI, which never sees the agent's answer
        {"event": "reconciled", "id": task.discrepancy_id, "lane": lane}
    )
    out: dict[str, Any] = {"resolutions": {task.discrepancy_id: res}}
    if res.usage:
        out["reconcile_usage"] = {task.discrepancy_id: res.usage}
    return out


def research_results(state: AnalysisState, discrepancy_id: str) -> list[ResearchResult]:
    found = state.get("research", {})
    return sorted(
        (r for k, r in found.items() if k.split("#r")[0] == discrepancy_id), key=lambda r: r.round
    )


ACTIONS = frozenset({"accept", "use_original", "escalate", "research"})
PICKABLE = frozenset(c.value for c in CANDIDATES)


def review_problems(
    answer: object, pending: set[str], rounds_left: dict[str, int]
) -> dict[str, str]:
    """Why a review answer can't be applied, per change id ({} = fine). Partial answers are never
    treated as approvals; an unknown action is never treated as "use the original" (it was)."""
    if not isinstance(answer, dict):
        return {"*": "the answer must map each change id to a decision"}
    decisions = cast(dict[str, Any], answer)
    problems = {k: "no decision" for k in sorted(pending - set(decisions))}
    for k, d in decisions.items():
        if k not in pending:
            problems[k] = "not waiting for review"
            continue
        fields = cast(dict[str, Any], d) if isinstance(d, dict) else {}
        action, cause = fields.get("action"), fields.get("correct_cause")
        if action not in ACTIONS:
            problems[k] = f"unknown action {action!r} (use one of {sorted(ACTIONS)})"
        elif action == "research" and rounds_left.get(k, 0) <= 0:
            problems[k] = "no research rounds left: decide or escalate"
        elif cause is not None and cause not in PICKABLE:
            problems[k] = f"unknown cause {cause!r} (use one of {sorted(PICKABLE)})"
    return problems


def review(state: AnalysisState, runtime: Runtime[Deps]) -> dict[str, Any]:
    """Human-in-the-loop. No side effects before interrupt(): on resume this node re-runs from
    the top, so anything above interrupt() would happen twice.

    Each pending change is shown as a review card (steps/card.py): every candidate cause with
    its evidence and whether it fits the numbers. The agent's own answer is deliberately left
    off, so the reviewer judges the evidence instead of approving a proposal."""
    pending = {
        k: r
        for k, r in state.get("resolutions", {}).items()
        if r.needs_review and k not in state.get("decisions", {})
    }
    if not pending:
        return {}
    deps = _deps(runtime)
    discrepancies = state.get("discrepancies", {})
    evidence = {t.discrepancy_id: t.evidence for t in state.get("reconcile_tasks", [])}
    done = {k: research_results(state, k) for k in pending}
    rounds_left = {
        k: (deps.max_research_rounds - len(done[k])) if deps.researcher else 0 for k in pending
    }
    payload: dict[str, Any] = {
        "type": "review",
        "pending": [
            build_card(discrepancies[k], evidence.get(k, ()), done[k], rounds_left[k])
            for k in sorted(pending)
        ],
    }
    # One interrupt() per pass through this node. An invalid answer isn't raised (LangGraph keeps
    # the resume value, so raising replayed the same bad answer on every later resume and locked
    # the run, found by review) and isn't re-asked in a loop here (the LangGraph docs: no
    # `while True` validation loops around interrupt(), use a conditional edge). The problems go
    # into state and `after_review` routes back to this node, which asks again showing them.
    asked = state.get("review_problems") or {}
    answer = interrupt({**payload, "problems": asked} if asked else payload)
    problems = review_problems(answer, set(pending), rounds_left)
    if problems:
        return {"review_problems": problems}
    decisions = cast(dict[str, Decision], answer)
    research = {k for k in pending if decisions[k]["action"] == "research"}
    requests = {
        f"{k}#r{len(done[k]) + 1}": ResearchRequest(
            discrepancy_id=k, hint=decisions[k].get("note", ""), round=len(done[k]) + 1
        )
        for k in sorted(research)
    }
    return {
        "decisions": {k: decisions[k] for k in pending if k not in research},
        "research_requests": requests,
        "review_problems": {},
    }


def after_review(state: AnalysisState) -> list[Send] | str:
    """An invalid answer goes back to review, which asks again with the problems. Requested
    research rounds that haven't run yet go to the research agent; then the cards come back to
    the reviewer. Otherwise on to the memo."""
    if state.get("review_problems"):
        return "review"
    tasks = {t.discrepancy_id: t for t in state.get("reconcile_tasks", [])}
    done = state.get("research", {})
    todo = [
        Send("research", ResearchInput(task=tasks[r["discrepancy_id"]], request=r))
        for key, r in sorted(state.get("research_requests", {}).items())
        if key not in done
    ]
    return todo or "memo"


def research(inp: ResearchInput, runtime: Runtime[Deps]) -> dict[str, Any]:
    """One reviewer-requested research round for one case. Transient errors retry (node
    policy); anything else is recorded on the result so the card can show it."""
    task, req = inp["task"], inp["request"]
    researcher = _deps(runtime).researcher
    key = f"{req['discrepancy_id']}#r{req['round']}"
    try:
        if researcher is None:
            raise RuntimeError("research requested but no researcher is configured")
        res = researcher.research(task, req["hint"], req["round"])
    except Exception as e:
        if is_transient(e):
            raise
        res = ResearchResult(
            req["discrepancy_id"], req["round"], req["hint"], (), 0, "", "", None,
            f"{type(e).__name__}: {e}",
        )  # fmt: skip
    get_stream_writer()(
        {"event": "researched", "id": key, "findings": len(res.findings), "error": res.error}
    )
    out: dict[str, Any] = {"research": {key: res}}
    if res.usage:
        out["research_usage"] = {key: res.usage}
    return out


def _apply(resolution: Resolution, decision: Decision | None, state: AnalysisState) -> Resolution:
    if decision is None:
        return resolution
    if decision["action"] == "escalate":  # published now, flagged: the cause stays open
        note = decision.get("note", "")
        return replace(
            resolution, cause=Cause.UNKNOWN, explanation=f"Escalated by reviewer. {note}".strip()
        )
    picked = decision.get("correct_cause")
    if picked and picked != resolution.cause.value:
        # The reviewer's pick is what gets published, with the card's evidence for THAT cause,
        # so the stored explanation and quote never argue for the agent's (rejected) cause.
        tasks = {t.discrepancy_id: t for t in state.get("reconcile_tasks", [])}
        task = tasks.get(resolution.discrepancy_id)
        quotes = sentences_for(Cause(picked), task.evidence if task else ())
        note = decision.get("note", "")
        resolution = replace(
            resolution,
            cause=Cause(picked),
            explanation=f"Cause picked by reviewer. {note}".strip(),
            evidence_quote=quotes[0] if quotes else "",
        )
    if decision["action"] == "accept":
        return resolution
    d = state.get("discrepancies", {})[resolution.discrepancy_id]
    oldest_accn, oldest_value = d.values[0]
    return replace(resolution, use_value=oldest_value, use_accession=oldest_accn)


def _final_resolutions(state: AnalysisState) -> dict[str, Resolution]:
    decisions = state.get("decisions", {})
    return {k: _apply(r, decisions.get(k), state) for k, r in state.get("resolutions", {}).items()}


def memo(state: AnalysisState, runtime: Runtime[Deps]) -> dict[str, Any]:
    """Pure (an LLM call, no side effects): safe to re-run on resume."""
    deps = _deps(runtime)
    writer = deps.memo_writer
    if writer is None or state.get("memo"):
        return {}
    filed = {
        f.accession_no: f.filed for t in state.get("tickers", []) for f in deps.store.filings(t)
    }
    facts = list(state.get("facts", {}).values())
    rows, changes = build_table(facts, _final_resolutions(state), filed)
    m = writer.write(rows, changes, rep=deps.rep)
    get_stream_writer()({"event": "memo", "verified": m.verified, "attempts": m.attempts})
    return {
        "memo": {
            # the table it was written from, next to the prose: a judge (ours, or LangSmith's
            # online evaluator) checks "every claim is supported by the table" against this
            "table": table_text(rows, changes),
            "markdown": m.markdown,
            "verified": m.verified,
            "unverified": m.unverified,
            "attempts": m.attempts,
        },
        "memo_usage": m.usage,
    }


def publish(state: AnalysisState, runtime: Runtime[Deps]) -> dict[str, Any]:
    """The only side effect, and it happens once per thread (see Ledger.publish)."""
    ledger = _deps(runtime).ledger
    if ledger is None:
        return {}
    thread_id = str(get_config().get("configurable", {}).get("thread_id", "no-thread"))
    decisions = state.get("decisions", {})
    final = _final_resolutions(state)
    discrepancies = state.get("discrepancies", {})
    answers = [
        Decided(
            r,
            decisions[k]["action"] if k in decisions else "auto",
            filing_set(a for a, _ in discrepancies[k].values) if k in discrepancies else "",
            decisions[k].get("actor") if k in decisions else None,
            decisions[k].get("decided_at") if k in decisions else None,
        )
        for k, r in final.items()
    ]
    m = state.get("memo")
    memo_arg = (str(m["markdown"]), bool(m["verified"])) if m else None
    # The root run id is consumed by the root run and not inherited by nodes, so the runner
    # also passes it in `configurable` for feedback to target the right trace.
    run_id = get_config().get("configurable", {}).get("root_run_id")
    tasks = {t.discrepancy_id: t for t in state.get("reconcile_tasks", [])}
    resolutions = state.get("resolutions", {})
    wanted = {k: d.get("correct_cause", "") for k, d in decisions.items()}
    corrections = [
        Correction(
            discrepancy_id=k,
            model_cause=resolutions[k].cause.value,
            correct_cause=wanted[k],
            note=d.get("note", ""),
            task_json=json.dumps(
                {
                    "discrepancy": tasks[k].discrepancy.model_dump(mode="json"),
                    "evidence": list(tasks[k].evidence),
                }  # fmt: skip
            ),
            run_id=str(run_id) if run_id else None,
        )
        for k, d in decisions.items()
        if wanted[k] and k in tasks and wanted[k] != resolutions[k].cause.value
    ]
    pub = ledger.publish(thread_id, state.get("facts", {}).values(), answers, memo_arg, corrections)
    get_stream_writer()({"event": "published", **pub.__dict__})
    return {"publication": dict(pub.__dict__)}


# ---- graph ------------------------------------------------------------------------------------


DEFAULT_RETRY = RetryPolicy(max_attempts=4, initial_interval=1.0, retry_on=is_transient)


def build_graph(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    retry: RetryPolicy = DEFAULT_RETRY,
    default_deps: Callable[[], Deps] | None = None,
    typed_context: bool = True,
) -> CompiledStateGraph[AnalysisState, Deps, AnalysisInput, AnalysisState]:
    """`typed_context=False` for servers (Studio) that would try to build Deps from JSON."""
    global _default_deps
    if default_deps is not None:
        _default_deps = default_deps
    g = StateGraph(
        AnalysisState, context_schema=Deps if typed_context else None, input_schema=AnalysisInput
    )
    g.add_node("plan", plan)
    g.add_node(  # pyright: ignore[reportCallIssue]
        "extract",
        extract,  # pyright: ignore[reportArgumentType]
        input_schema=ExtractInput,
        retry_policy=retry,
        # No node `timeout`: LangGraph only supports it on async nodes. The Bedrock client's
        # read_timeout (120s) bounds each call instead, and its error goes through the retry policy.
    )
    g.add_node("find_discrepancies", find_discrepancies_node)
    g.add_node(  # pyright: ignore[reportCallIssue]
        "reconcile",
        reconcile,  # pyright: ignore[reportArgumentType]
        input_schema=ReconcileInput,
        retry_policy=retry,
    )
    g.add_node("review", review)
    g.add_node(  # pyright: ignore[reportCallIssue]
        "research",
        research,  # pyright: ignore[reportArgumentType]
        input_schema=ResearchInput,
        retry_policy=retry,
    )
    g.add_node("memo", memo)
    g.add_node("publish", publish)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", fan_out, ["extract"])
    g.add_edge("extract", "find_discrepancies")
    g.add_conditional_edges("find_discrepancies", fan_out_reconcile, ["reconcile", "review"])
    g.add_edge("reconcile", "review")
    g.add_conditional_edges("review", after_review, ["review", "research", "memo"])
    g.add_edge("research", "review")
    g.add_edge("memo", "publish")
    g.add_edge("publish", END)
    return g.compile(checkpointer=checkpointer)
