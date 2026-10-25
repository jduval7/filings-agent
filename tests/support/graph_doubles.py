"""Test doubles for running the analysis graph without network or LLMs: a filing store,
extractors, a reconciler, and a review environment with a real SQLite checkpointer."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from botocore.exceptions import ClientError
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver

from filings_agent.domain import ExtractedFact, FilingRef, Metric
from filings_agent.graph.build import Deps, build_graph
from filings_agent.graph.runner import serializer
from filings_agent.steps.extract import ExtractResult, ExtractTask
from filings_agent.steps.reconcile import Cause, ReconcileTask, Resolution
from filings_agent.storage.ledger import Ledger

ROWS = "\n".join(f"| Line {i} | {1000 + i:,} |  | {900 + i:,} |  |" for i in range(8))


FILING_MD = (
    "**Consolidated Statements of Income**\n(In millions)\n"
    f"|  | 2025 |  | 2024 |  |\n{ROWS}\n| Net income | $ | 5,000 | $ | 4,000 |\n\n"
    "**Consolidated Balance Sheets**\n(In millions)\n"
    f"|  | 2025 |  | 2024 |  |\n{ROWS}\n| Total assets | $ | 9,000 | $ | 8,000 |\n\n"
    "**Consolidated Statements of Cash Flows**\n(In millions)\n"
    f"|  | 2025 |  | 2024 |  |\n{ROWS}\n| Net cash from operating activities | 7 |  | 6 |  |\n"
)


def _filing(ticker: str, n: int) -> FilingRef:
    return FilingRef(
        ticker=ticker,
        cik=1,
        accession_no=f"000000000{n}-25-00000{n}",
        form="10-K",
        filed=date(2025, 2, 1),
        period_of_report=date(2024, 12, 31),
    )


class FakeStore:
    def __init__(self, per_ticker: int = 2) -> None:
        self._n = per_ticker

    def filings(self, ticker: str) -> list[FilingRef]:
        return [_filing(ticker, i) for i in range(self._n)]

    def markdown(self, filing: FilingRef) -> str:
        return FILING_MD


def throttle_error() -> ClientError:
    return ClientError({"Error": {"Code": "ThrottlingException", "Message": "slow"}}, "Converse")


class FakeExtractor:
    """One fact per task. Can be told to fail specific statements in specific ways."""

    name = "fake"

    def __init__(self, fail: dict[str, BaseException] | None = None, flaky: int = 0) -> None:
        self.fail = fail or {}
        self.flaky = flaky  # first N calls raise a throttling error, then succeed
        self.calls = 0

    def extract(self, task: ExtractTask, rep: int = 0) -> ExtractResult:
        self.calls += 1
        if self.calls <= self.flaky:
            raise throttle_error()
        if task.statement in self.fail:
            raise self.fail[task.statement]
        fact = ExtractedFact(
            ticker=task.ticker,
            metric=task.metrics[0],
            fiscal_year=2024,
            period_end=date(2024, 12, 31),
            value=Decimal(1),
            accession_no=task.accession_no,
            statement=task.statement,
            row_label="x",
            printed_value="1",
            scale="units",
            model="fake",
        )
        return ExtractResult(task_id=task.task_id, facts=[fact])


METRICS = (Metric.REVENUE, Metric.CASH, Metric.OPERATING_CASH_FLOW)


OLD, NEW = "0000000001-24-000001", "0000000001-25-000001"


class TwoFilingStore:
    def filings(self, ticker: str) -> list[FilingRef]:
        return [
            FilingRef(ticker=ticker, cik=1, accession_no=OLD, form="10-K",
                      filed=date(2024, 2, 1), period_of_report=date(2023, 12, 31)),
            FilingRef(ticker=ticker, cik=1, accession_no=NEW, form="10-K",
                      filed=date(2025, 2, 1), period_of_report=date(2024, 12, 31)),
        ]  # fmt: skip

    def markdown(self, filing: FilingRef) -> str:
        return FILING_MD + "\nPrior periods were recast to reflect discontinued operations.\n"


class RestatingExtractor:
    """FY2023 revenue: 100 in the old filing, 60 (recast) in the new one. Shares: x10 split.
    Cash: a 0.1% revision (immaterial)."""

    name = "restating"
    VALUES = {
        (Metric.REVENUE, OLD): 100, (Metric.REVENUE, NEW): 60,
        (Metric.CASH, OLD): 1000, (Metric.CASH, NEW): 1001,
        (Metric.OPERATING_CASH_FLOW, OLD): 5, (Metric.OPERATING_CASH_FLOW, NEW): 5,
    }  # fmt: skip

    def extract(self, task: ExtractTask, rep: int = 0) -> ExtractResult:
        metric = task.metrics[0]
        fact = ExtractedFact(
            ticker=task.ticker, metric=metric, fiscal_year=2023, period_end=date(2023, 12, 31),
            value=Decimal(self.VALUES[(metric, task.accession_no)]),
            accession_no=task.accession_no, statement=task.statement, row_label="x",
            printed_value="x", scale="units", model="fake",
        )  # fmt: skip
        return ExtractResult(task_id=task.task_id, facts=[fact])


class FakeReconciler:
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def reconcile(self, task: ReconcileTask, rep: int = 0) -> Resolution:
        self.calls += 1
        accn, value = task.discrepancy.values[-1]
        return Resolution(
            task.discrepancy_id, Cause.DISCONTINUED_OPERATIONS, "recast", "recast", True, True,
            value, accn, True,
        )  # fmt: skip


@contextmanager
def review_env(tmp: Path) -> Generator[tuple[Any, Deps, FakeReconciler, Ledger]]:
    conn = sqlite3.connect(str(tmp / "cp.sqlite"), check_same_thread=False)
    ledger = Ledger(tmp / "ledger.sqlite")
    rec = FakeReconciler()
    deps = Deps(TwoFilingStore(), RestatingExtractor(), METRICS, reconciler=rec, ledger=ledger)
    try:
        yield build_graph(SqliteSaver(conn, serde=serializer())), deps, rec, ledger
    finally:
        conn.close()
        ledger.close()


CFG: RunnableConfig = {"configurable": {"thread_id": "review-1"}}


def invoke(graph: Any, inp: Any, deps: Deps) -> dict[str, Any]:
    return cast(dict[str, Any], graph.invoke(inp, CFG, context=deps))
