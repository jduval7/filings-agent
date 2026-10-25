"""Crash recovery and idempotency with a real SQLite checkpointer (fakes for the LLM)."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pytest
from botocore.exceptions import ClientError
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import RetryPolicy

from filings_agent.graph.build import Deps, build_graph, is_transient
from filings_agent.graph.runner import serializer
from filings_agent.steps.extract import ExtractResult, ExtractTask
from tests.support.graph_doubles import METRICS, FakeExtractor, FakeStore, throttle_error

FAST = RetryPolicy(max_attempts=1, retry_on=is_transient)


class CrashingExtractor(FakeExtractor):
    """Works for the first `ok` calls, then the 'service' goes down for good."""

    def __init__(self, ok: int) -> None:
        super().__init__()
        self.ok = ok
        self.succeeded = 0

    def extract(self, task: ExtractTask, rep: int = 0) -> ExtractResult:
        if self.succeeded >= self.ok:
            raise throttle_error()
        self.succeeded += 1
        return super().extract(task, rep)


@contextmanager
def _saver(tmp: Path) -> Generator[SqliteSaver]:
    conn = sqlite3.connect(str(tmp / "cp.sqlite"), check_same_thread=False)
    try:
        yield SqliteSaver(conn, serde=serializer())  # same allowlist as production
    finally:
        conn.close()


def _config(thread: str) -> RunnableConfig:
    # max_concurrency=1 makes "which branches finished before the crash" deterministic.
    return {"configurable": {"thread_id": thread}, "max_concurrency": 1}


def _invoke(graph: Any, inp: Any, deps: Deps, thread: str) -> dict[str, Any]:
    return cast(dict[str, Any], graph.invoke(inp, _config(thread), context=deps))


def test_resume_after_crash_reruns_only_unfinished_branches(tmp_path: Path) -> None:
    with _saver(tmp_path) as cp:
        graph = build_graph(cp, retry=FAST)

        crashing = CrashingExtractor(ok=4)
        with pytest.raises(ClientError):
            _invoke(graph, {"tickers": ["AAA"]}, Deps(FakeStore(), crashing, METRICS), "t1")
        assert crashing.succeeded == 4  # 4 of 6 branches finished before the outage

        healthy = FakeExtractor()
        out = _invoke(graph, None, Deps(FakeStore(), healthy, METRICS), "t1")  # resume

    assert healthy.calls == 2  # only the 2 unfinished branches ran again
    assert len(out["facts"]) == 6
    assert out.get("errors", {}) == {}


def test_running_the_same_request_twice_adds_no_duplicates(tmp_path: Path) -> None:
    with _saver(tmp_path) as cp:
        graph = build_graph(cp, retry=FAST)
        deps = Deps(FakeStore(), FakeExtractor(), METRICS)
        first = _invoke(graph, {"tickers": ["AAA"]}, deps, "t2")
        second = _invoke(graph, {"tickers": ["AAA"]}, deps, "t2")

    assert first["facts"].keys() == second["facts"].keys()
    assert len(second["facts"]) == 6
