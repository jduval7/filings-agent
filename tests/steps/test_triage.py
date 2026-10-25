"""Triage: the quote proves the evidence exists; these rules check it explains THIS number."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from filings_agent.domain import Discrepancy, Metric
from filings_agent.graph.build import Deps
from filings_agent.steps.reconcile import Cause, ReconcileTask, Resolution
from filings_agent.steps.triage import (
    CheckStatus,
    apply_triage,
    split_ratio,
    stated_ratios,
    triage,
)
from tests.support.graph_doubles import NEW, OLD, RestatingExtractor, invoke, review_env

A, B, C = "0000000001-23-000001", "0000000001-24-000001", "0000000001-25-000001"


def disc(metric: Metric, *values: int) -> Discrepancy:
    accns = (A, B, C)[: len(values)]
    return Discrepancy(
        ticker="AAA",
        metric=metric,
        fiscal_year=2022,
        period_end=date(2022, 12, 31),
        values=tuple((a, Decimal(v)) for a, v in zip(accns, values, strict=True)),
    )


def res(d: Discrepancy, cause: Cause, quote: str = "x", verified: bool = True) -> Resolution:
    accn, value = d.values[-1]
    return Resolution(d.id, cause, "e", quote, True, verified, value, accn, True)


def statuses(d: Discrepancy, r: Resolution) -> dict[str, CheckStatus]:
    return {c.name: c.status for c in triage(d, r).checks}


def test_exact_split_named_in_the_quote_is_the_easy_lane() -> None:
    d = disc(Metric.DILUTED_SHARES, 2_490, 24_940)  # NVDA-like, x10.016 is within tolerance
    r = res(d, Cause.STOCK_SPLIT, "In June 2024, we executed a ten-for-one stock split.")
    assert triage(d, r).lane == "auto"


def test_ge_net_income_spin_off_answer_is_contradicted() -> None:
    """The motivating case: real quote, but a spin-off can't move total net income."""
    d = disc(Metric.NET_INCOME, 225, 339)
    r = res(d, Cause.DISCONTINUED_OPERATIONS, "the spin-off of GE HealthCare")
    t = triage(d, r)
    assert t.lane == "review"
    assert statuses(d, r)["cause_fits_metric"] is CheckStatus.CONTRADICTED


@pytest.mark.parametrize(
    ("metric", "values", "cause", "quote", "check", "status"),
    [
        # a split can't move dollar amounts
        (Metric.REVENUE, (100, 1000), Cause.STOCK_SPLIT, "10-for-1", "cause_fits_metric",
         CheckStatus.CONTRADICTED),
        # x1.37 is not a split ratio
        (Metric.DILUTED_SHARES, (100, 137), Cause.STOCK_SPLIT, "10-for-1", "split_ratio",
         CheckStatus.CONTRADICTED),
        # the quote names a different split than the one that moved this number
        (Metric.DILUTED_SHARES, (100, 1000), Cause.STOCK_SPLIT, "a 4-for-1 split",
         "quote_names_ratio", CheckStatus.CONTRADICTED),
        # right ratio, but the quote doesn't tie it to the number: a human looks
        (Metric.DILUTED_SHARES, (100, 1000), Cause.STOCK_SPLIT, "a stock split",
         "quote_names_ratio", CheckStatus.NOT_COVERED),
        # spin-offs do move revenue, but no rule proves this one did
        (Metric.REVENUE, (100, 76), Cause.DISCONTINUED_OPERATIONS, "spin-off", "fingerprint",
         CheckStatus.NOT_COVERED),
        (Metric.NET_INCOME, (100, 90), Cause.ACCOUNTING_CHANGE, "adopted ASU 2018-12",
         "fingerprint", CheckStatus.NOT_COVERED),
    ],
)  # fmt: skip
def test_everything_not_confirmed_goes_to_review(
    metric: Metric,
    values: tuple[int, ...],
    cause: Cause,
    quote: str,
    check: str,
    status: CheckStatus,
) -> None:
    d = disc(metric, *values)
    r = res(d, cause, quote)
    assert statuses(d, r)[check] is status
    assert triage(d, r).lane == "review"


def test_unverified_or_unknown_answers_are_reviewed() -> None:
    d = disc(Metric.DILUTED_SHARES, 100, 1000)
    assert triage(d, res(d, Cause.STOCK_SPLIT, "ten-for-one", verified=False)).lane == "review"
    assert triage(d, res(d, Cause.UNKNOWN, "")).lane == "review"


def test_a_number_changed_by_two_splits_matches_either() -> None:
    d = disc(Metric.DILUTED_SHARES, 100, 400, 4000)  # 4-for-1, then 10-for-1
    assert triage(d, res(d, Cause.STOCK_SPLIT, "ten-for-one stock split")).lane == "auto"


def test_split_ratio_tolerance_and_reverse_splits() -> None:
    assert split_ratio(Decimal("10.0047")) == 10  # AVGO weighted-average shares
    assert split_ratio(Decimal("19.9922")) == 20  # AMZN
    assert split_ratio(Decimal("0.05")) == -20  # 1-for-20 reverse split
    assert split_ratio(Decimal("1.37")) is None
    assert split_ratio(Decimal("1.0")) is None


def test_stated_ratios_reads_filing_typography() -> None:
    assert stated_ratios("a 20 \\-for\\-1 stock split") == {20}  # escaped markdown hyphens
    assert stated_ratios("a ten‑for‑one split") == {10}  # non-breaking hyphens
    assert stated_ratios("a 4:1 split") == {4}
    assert stated_ratios("a 1-for-20 reverse split") == {-20}
    assert stated_ratios("revenue grew 10% in fiscal 2024") == set()


def test_auto_approve_off_still_reviews_but_records_the_lane() -> None:
    d = disc(Metric.DILUTED_SHARES, 100, 1000)
    r = res(d, Cause.STOCK_SPLIT, "ten-for-one")
    off = apply_triage(d, r, auto_approve=False)
    on = apply_triage(d, r, auto_approve=True)
    assert off.needs_review and off.triage and off.triage.lane == "auto"
    assert not on.needs_review


class SplitExtractor(RestatingExtractor):
    VALUES = {
        **RestatingExtractor.VALUES,
        (Metric.DILUTED_SHARES, OLD): 100,
        (Metric.DILUTED_SHARES, NEW): 1000,
    }


class SplitReconciler:
    name = "fake-split"

    def reconcile(self, task: ReconcileTask, rep: int = 0) -> Resolution:
        accn, value = task.discrepancy.values[-1]
        return Resolution(
            task.discrepancy_id, Cause.STOCK_SPLIT, "split", "a ten-for-one stock split", True,
            True, value, accn, True,
        )  # fmt: skip


@contextmanager
def _split_env(tmp: Path, auto_approve: bool) -> Generator[tuple[Any, Deps]]:
    with review_env(tmp) as (graph, deps, _rec, _ledger):
        yield (
            graph,
            replace(
                deps,
                extractor=SplitExtractor(),
                metrics=(Metric.DILUTED_SHARES, Metric.CASH, Metric.OPERATING_CASH_FLOW),
                reconciler=SplitReconciler(),
                auto_approve=auto_approve,
            ),
        )


def test_confirmed_split_skips_review_and_publishes(tmp_path: Path) -> None:
    with _split_env(tmp_path, auto_approve=True) as (graph, deps):
        out = invoke(graph, {"tickers": ["AAA"]}, deps)
        assert "__interrupt__" not in out
        r = out["resolutions"]["AAA:diluted_shares:FY2023"]
        assert r.triage.lane == "auto" and not r.needs_review
        assert out["publication"]["newly_published"] is True


def test_without_auto_approve_the_split_is_reviewed(tmp_path: Path) -> None:
    with _split_env(tmp_path, auto_approve=False) as (graph, deps):
        out = invoke(graph, {"tickers": ["AAA"]}, deps)
        (intr,) = out["__interrupt__"]
        (card,) = intr.value["pending"]
        assert card["id"] == "AAA:diluted_shares:FY2023"


def test_every_rule_says_where_it_comes_from() -> None:
    """A rule that rules out a cause with confidence must cite its source or say it's reasoning."""
    from filings_agent.steps.triage import CANNOT_MOVE

    for cause, rules in CANNOT_MOVE.items():
        for metric, reason in rules.items():
            assert "ASC " in reason or "reasoning:" in reason, (cause, metric)
