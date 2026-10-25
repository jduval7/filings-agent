"""Wave-1 graph behaviour with fakes: no network, no LLM."""

from __future__ import annotations

from typing import Any, cast

import pytest
from botocore.exceptions import ClientError
from langgraph.types import RetryPolicy

from filings_agent.graph.build import Deps, build_graph, is_transient
from filings_agent.llm.spend_guard import SpendLimitExceeded
from tests.support.graph_doubles import METRICS, FakeExtractor, FakeStore


def _run(extractor: FakeExtractor, tickers: list[str]) -> dict[str, Any]:
    graph = build_graph(
        retry=RetryPolicy(max_attempts=4, initial_interval=0.01, retry_on=is_transient)
    )
    deps = Deps(store=FakeStore(), extractor=extractor, metrics=METRICS)
    return cast(dict[str, Any], graph.invoke({"tickers": tickers}, context=deps))  # pyright: ignore[reportUnknownMemberType]


def test_fan_out_one_branch_per_filing_statement() -> None:
    ex = FakeExtractor()
    out = _run(ex, ["AAA", "BBB"])
    # 2 tickers x 2 filings x 3 statements
    assert len(out["tasks"]) == 12
    assert ex.calls == 12
    assert len(out["facts"]) == 12
    assert out.get("errors", {}) == {}


def test_transient_errors_are_retried() -> None:
    ex = FakeExtractor(flaky=2)
    out = _run(ex, ["AAA"])
    assert len(out["facts"]) == 6
    assert out.get("errors", {}) == {}
    assert ex.calls == 6 + 2


def test_failing_branch_is_isolated() -> None:
    ex = FakeExtractor(fail={"balance_sheet": ValueError("boom")})
    out = _run(ex, ["AAA"])
    assert len(out["facts"]) == 4  # income + cash flow for 2 filings
    assert len(out["errors"]) == 2
    assert all("ValueError: boom" in e for e in out["errors"].values())


def test_persistent_outage_stops_the_run_for_a_later_resume() -> None:
    ex = FakeExtractor(flaky=100)  # throttled on every attempt
    with pytest.raises(ClientError):
        _run(ex, ["AAA"])


def test_spend_cap_stops_the_whole_run() -> None:
    ex = FakeExtractor(fail={"cash_flow_statement": SpendLimitExceeded("cap")})
    with pytest.raises(SpendLimitExceeded):
        _run(ex, ["AAA"])
