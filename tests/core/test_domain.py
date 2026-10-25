"""Domain types: periods and discrepancies."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from filings_agent.domain import Discrepancy, Metric, Period


def test_period_kind_and_key() -> None:
    p = Period(start=date(2024, 1, 1), end=date(2024, 12, 31))
    assert p.kind.value == "duration" and p.key() == "2024-01-01..2024-12-31"
    assert Period(start=None, end=date(2024, 12, 31)).kind.value == "instant"


def test_discrepancy_ratio_with_zero_oldest() -> None:
    d = Discrepancy(
        ticker="X",
        metric=Metric.REVENUE,
        fiscal_year=2024,
        period_end=date(2024, 12, 31),
        values=(("A", Decimal(0)), ("B", Decimal(5))),
    )
    assert d.ratio == 0
