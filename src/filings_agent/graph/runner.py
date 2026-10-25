"""Run the graph on real filings: local filing store, SQLite checkpoints, streamed progress.

The thread id doubles as the idempotency key: the same request maps to the same thread, so
re-running it resumes or no-ops instead of paying for the fan-out again. "The same request" means
the same inputs: the tickers, the filings on disk for them, every model and prompt that shapes the
result, and the graph version (`run_identity`). A new 10-K or a new reconcile model is a new
analysis, not a replay of the old one (found by review: it used to be tickers + extractor only).

A run holds a lease on its thread (storage/leases.py) while it executes, so two processes (two
server workers, or the CLI and the server) can't run the same thread at once.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from filings_agent.config import AppConfig
from filings_agent.domain import FilingRef
from filings_agent.graph.build import GRAPH_VERSION, Deps, build_graph
from filings_agent.graph.state import AnalysisInput
from filings_agent.llm.cache import StepCache
from filings_agent.llm.models import chat_model
from filings_agent.steps.extract import LLMStatementExtractor
from filings_agent.steps.memo import MemoWriter
from filings_agent.storage.leases import Leases
from filings_agent.storage.ledger import Ledger


class LocalFilingStore:
    """Filings already fetched by `build-data` (data/gold/filings.jsonl + data/filings/*.md)."""

    def __init__(self, data_dir: Path) -> None:
        self._dir = data_dir
        rows = (data_dir / "gold" / "filings.jsonl").read_text().splitlines()
        self._filings = [FilingRef.model_validate_json(r) for r in rows]

    def filings(self, ticker: str) -> list[FilingRef]:
        return [f for f in self._filings if f.ticker == ticker]

    def markdown(self, filing: FilingRef) -> str:
        return (self._dir / "filings" / f"{filing.accession_no}.md").read_text()


def filings_on_disk(cfg: AppConfig, tickers: list[str]) -> list[str]:
    """Accession numbers the run would read. None yet (no `build-data`): an empty list."""
    path = cfg.data_dir / "gold" / "filings.jsonl"
    if not path.exists():
        return []
    wanted = set(tickers)
    refs = (FilingRef.model_validate_json(r) for r in path.read_text().splitlines() if r)
    return sorted(f.accession_no for f in refs if f.ticker in wanted)


def run_identity(cfg: AppConfig, tickers: list[str], model: str, prompt: str) -> dict[str, Any]:
    """Everything that changes what a run publishes. Ticker order doesn't."""
    return {
        "tickers": sorted(tickers),
        "filings": filings_on_disk(cfg, tickers),
        "extract": {"model": model, "prompt": prompt},
        "reconcile": {
            "model": cfg.roles.reconciler,
            "prompt": cfg.prompts.reconcile,
            "strategy": cfg.reconcile.strategy,
            "agent_version": cfg.reconcile.agent_version,
            "auto_approve": cfg.reconcile.auto_approve,
        },
        "research": cfg.research.model_dump() if cfg.research.enabled else None,
        "memo": cfg.roles.memo_writer,
        "metrics": sorted(m.value for m in cfg.metrics),
        "graph": GRAPH_VERSION,
    }


def request_key(identity: dict[str, Any]) -> str:
    """Same inputs -> same key -> same thread."""
    raw = json.dumps(identity, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# Our types that live in checkpoints. LangGraph refuses to deserialize unregistered types
# (in a future version), which stops a tampered checkpoint from instantiating arbitrary classes.
CHECKPOINT_TYPES = [
    ("filings_agent.steps.extract", "ExtractTask"),
    ("filings_agent.steps.extract", "Rejection"),
    ("filings_agent.steps.extract", "Usage"),
    ("filings_agent.steps.reconcile", "ReconcileTask"),
    ("filings_agent.steps.reconcile", "Resolution"),
    ("filings_agent.steps.reconcile", "Cause"),
    ("filings_agent.steps.triage", "Triage"),
    ("filings_agent.research.findings", "ResearchResult"),
    ("filings_agent.research.findings", "VerifiedFinding"),
    ("filings_agent.steps.triage", "Check"),
    ("filings_agent.steps.triage", "CheckStatus"),
    ("filings_agent.domain", "ExtractedFact"),
    ("filings_agent.domain", "Discrepancy"),
    ("filings_agent.domain", "Metric"),
]


def serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)


@contextmanager
def checkpointer(cfg: AppConfig) -> Generator[SqliteSaver]:
    path = cfg.state("checkpoints.sqlite")
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    try:
        yield SqliteSaver(conn, serde=serializer())
    finally:
        conn.close()


def make_researcher(cfg: AppConfig) -> Any:
    """Built lazily: the deep-agent stack is only imported when research is enabled."""
    from filings_agent.reconcile_agent.factory import filings_by_ticker
    from filings_agent.research.agent import Researcher
    from filings_agent.research.sources import LiveFetcher

    def cik_of(ticker: str) -> int | None:
        # Every company, incl. reconcile-only ones. It read the extraction companies' list once,
        # so 9 of 14 research companies had no CIK: "this company's filings" searched everyone's,
        # and citations read /edgar/data/0/ (found 2026-10-23, DECISIONS.md).
        refs = list(filings_by_ticker(cfg).get(ticker, {}).values())
        return refs[0].cik if refs else None

    name = cfg.research.model
    return Researcher(
        name,
        cfg.model(name),
        chat_model(cfg, name),
        StepCache(cfg.data_dir / "cache" / "llm"),
        # The SEC contact is read when research first reaches sec.gov, not at startup: building
        # the app must not need it (found when CI ran without SEC_USER_AGENT).
        LiveFetcher(cfg.sec.user_agent, cfg.data_dir / "cache" / "research"),
        cik_of,
        lambda accession_no: (cfg.data_dir / "filings" / f"{accession_no}.md").read_text(),
    )


_ledgers: dict[Path, Ledger] = {}
_ledgers_lock = threading.Lock()


def shared_ledger(path: Path) -> Ledger:
    """One connection per ledger file per process (thread-safe: Ledger serializes its writes).
    Opening one per run and never closing it leaked a connection per run in the server."""
    with _ledgers_lock:
        if path not in _ledgers:
            _ledgers[path] = Ledger(path)
        return _ledgers[path]


def make_deps(
    cfg: AppConfig, model: str, prompt: str, reconciler_model: str | None = None, rep: int = 0
) -> Deps:
    extractor = LLMStatementExtractor(
        model,
        cfg.model(model),
        chat_model(cfg, model),
        StepCache(cfg.data_dir / "cache" / "llm"),
        prompt,
    )
    from filings_agent.reconcile_agent.factory import make_reconciler

    reconciler = make_reconciler(cfg, reconciler_model)
    return Deps(
        store=LocalFilingStore(cfg.data_dir),
        extractor=extractor,
        metrics=tuple(cfg.metrics),
        reconciler=reconciler,
        ledger=shared_ledger(cfg.state("ledger.sqlite")),
        rep=rep,
        auto_approve=cfg.reconcile.auto_approve,
        researcher=make_researcher(cfg) if cfg.research.enabled else None,
        max_research_rounds=cfg.research.max_rounds,
        memo_writer=MemoWriter(
            cfg.roles.memo_writer,
            cfg.model(cfg.roles.memo_writer),
            chat_model(cfg, cfg.roles.memo_writer),
            # cached like every other step: a crash after the memo doesn't pay for it twice
            StepCache(cfg.data_dir / "cache" / "llm"),
        ),
    )


def run_analysis(
    cfg: AppConfig,
    tickers: list[str],
    model: str,
    prompt: str,
    thread_id: str | None = None,
    resume: bool = False,
    concurrency: int = 8,
    decisions: dict[str, Any] | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    rep: int = 0,
    lease_owner: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Start (or resume) a run. With `decisions`, resume a run paused for human review.

    Shared by the CLI (prints events) and the API (pushes them to SSE subscribers). The API takes
    the thread's lease before it answers the request and passes its `lease_owner` in; the run
    re-enters that lease. Raises ThreadBusy if another process holds it."""
    emit: Callable[[dict[str, Any]], None] = on_event or _print_event
    # --fresh (rep) is a new request: its own thread, so it never merges into an earlier run's state
    thread = thread_id or request_key(run_identity(cfg, tickers, model, prompt)) + (
        f"-r{rep}" if rep else ""
    )
    # the LangSmith root run id for this invocation (the API picks it, so it can link the trace
    # even when the run fails)
    root = uuid.UUID(run_id) if run_id else uuid.uuid4()
    config: RunnableConfig = {
        "configurable": {"thread_id": thread, "root_run_id": str(root)},
        "run_id": root,
        "max_concurrency": concurrency,
        "run_name": "filings-analysis",
        "tags": ["filings-agent", f"model:{model}", f"prompt:{prompt}"],
        "metadata": {"tickers": tickers, "model": model, "prompt_version": prompt},
    }
    deps = make_deps(cfg, model, prompt, rep=rep)
    t0 = time.monotonic()
    leases = Leases(cfg.state("leases.sqlite"))
    with leases.hold(thread, lease_owner or uuid.uuid4().hex), checkpointer(cfg) as cp:
        graph = build_graph(cp)
        inp: AnalysisInput | Command[Any] | None
        if decisions is not None:
            inp = Command(resume=decisions)
        elif resume:
            inp = None
        else:
            existing = graph.get_state(config)
            if existing.next:
                raise ValueError(
                    f"thread {thread} is unfinished: continue it with --resume, or `review` it"
                )
            if existing.values:
                # Already ran. Running new input on it would re-pay the work, keep the old
                # review decisions (keyed merge) and then publish nothing (once per thread):
                # new results silently discarded. Return what it published instead.
                summary = summarize(thread, existing)
                summary["replayed"] = True
                return summary
            inp = AnalysisInput(tickers=tickers)
        events = graph.stream(  # pyright: ignore[reportUnknownMemberType]
            inp,
            config,
            context=deps,
            stream_mode=["custom"],
            # "sync": each step's checkpoint is persisted before the next step starts. Note
            # (measured, LangGraph 1.2.16): writes of branches in the SAME fan-out step only
            # reach the checkpointer when the step ends, so a hard kill (SIGKILL) mid-fan-out
            # re-runs those branches on resume. The content-addressed StepCache (written
            # atomically per LLM call) makes those re-runs free: nothing is paid twice.
            # See DECISIONS.md and scripts/crash_rehearsal.py.
            durability="sync",
        )
        for mode, chunk in events:
            if mode == "custom":
                emit(cast(dict[str, Any], chunk))
        snapshot = graph.get_state(config)
    summary = summarize(thread, snapshot)
    summary["wall_s"] = round(time.monotonic() - t0, 2)
    summary["run_id"] = str(root)
    return summary


def _print_event(event: dict[str, Any]) -> None:
    print(json.dumps(event, default=str), flush=True)


def summarize(thread: str, snapshot: Any) -> dict[str, Any]:
    state = cast(dict[str, Any], snapshot.values)
    pending = [i.value for t in snapshot.tasks for i in t.interrupts]
    usage = {
        **state.get("usage", {}),
        **state.get("reconcile_usage", {}),
        **state.get("research_usage", {}),
    }
    if state.get("memo_usage"):
        usage["memo"] = state["memo_usage"]
    waiting_ids: set[str] = {str(c["id"]) for c in pending[0]["pending"]} if pending else set()
    if pending:
        status = "awaiting_review"
    elif state.get("publication"):
        status = "published"
    elif snapshot.next:
        status = "interrupted"  # stopped mid-run (crash/outage): resumable
    elif state:
        status = "done"
    else:
        status = "not_found"
    return {
        "thread_id": thread,
        "status": status,
        "tickers": state.get("tickers", []),
        "branches": len(state.get("tasks", [])),
        # statements already read (each finished extract branch records its usage): after a crash
        # this is what a resume won't read again
        "branches_done": len(state.get("usage", {})),
        "facts": len(state.get("facts", {})),
        "rejected": sum(len(v) for v in state.get("rejected", {}).values()),
        "errors": state.get("errors", {}),
        "cost_usd": round(sum(u.cost_usd for u in usage.values()), 4),
        # what this run's calls cost now: a cached answer reports its original cost above, $0 here
        "paid_usd": round(sum(u.cost_usd for u in usage.values() if not u.cached), 4),
        "cached_branches": sum(1 for u in usage.values() if u.cached),
        "discrepancies": len(state.get("discrepancies", {})),
        # The agent's cause stays hidden while a change waits on a human: the CLI, the API and the
        # UI all read this (the CLI printed it until review found it).
        "resolutions": {
            k: {
                "cause": None if k in waiting_ids else r.cause.value,
                "verified": r.verified,
                "needs_review": r.needs_review,
                "lane": r.triage.lane if r.triage else None,
            }
            for k, r in state.get("resolutions", {}).items()
        },
        "awaiting_review": pending[0]["pending"] if pending else [],
        # why the last answer was asked again (an invalid or partial review), if it was
        "review_problems": pending[0].get("problems", {}) if pending else {},
        # what the reviewer did (the agent's cause above is its own, hidden from the card)
        "decisions": dict(state.get("decisions", {})),
        "research_rounds": len(state.get("research", {})),
        "memo": state.get("memo"),
        "publication": state.get("publication"),
    }


def get_status(cfg: AppConfig, thread_id: str) -> dict[str, Any]:
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    with checkpointer(cfg) as cp:
        snapshot = build_graph(cp).get_state(config)
    return summarize(thread_id, snapshot)


def pending_cards(cfg: AppConfig, thread_id: str) -> list[dict[str, Any]]:
    """The review cards a paused thread is waiting on."""
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    with checkpointer(cfg) as cp:
        snapshot = build_graph(cp).get_state(config)
    return [p for t in snapshot.tasks for i in t.interrupts for p in i.value["pending"]]


def pending_review(cfg: AppConfig, thread_id: str) -> list[str]:
    """Ids of the changes a paused thread is waiting on."""
    return [str(c["id"]) for c in pending_cards(cfg, thread_id)]
