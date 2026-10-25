"""Memo number formatting and the unverified banner."""

from __future__ import annotations

from decimal import Decimal

import pytest

from filings_agent.domain import Metric
from filings_agent.steps.memo import MemoOutput, MemoSection, display, pct, render


@pytest.mark.parametrize(
    ("value", "metric", "expected"),
    [
        (Decimal(215_938_000_000), Metric.REVENUE, "$215.9B"),
        (Decimal(336_000_000), Metric.NET_INCOME, "$336M"),
        (Decimal(-6_337_000_000), Metric.NET_INCOME, "-$6.3B"),
        (Decimal(24_514_000_000), Metric.DILUTED_SHARES, "24.51B shares"),
        (Decimal(610_000_000), Metric.DILUTED_SHARES, "610M shares"),
    ],
)
def test_display_formats(value: Decimal, metric: Metric, expected: str) -> None:
    assert display(value, metric) == expected


def test_pct_refuses_meaningless_comparisons() -> None:
    assert pct(Decimal(150), Decimal(100)) == "+50.0%"
    assert pct(Decimal(-150), Decimal(-100)) == "-50.0%"  # a bigger loss
    assert pct(Decimal(10), Decimal(-5)) is None  # across zero
    assert pct(Decimal(10), Decimal(0)) is None


def test_render_banner_only_when_unverified() -> None:
    out = MemoOutput(title="T", summary="S", sections=[MemoSection(heading="H", body="B")],
                     caveats="C")  # fmt: skip
    assert "NOT VERIFIED" not in render(out, True, [])
    bad = render(out, False, ["$216B"])
    assert "NOT VERIFIED" in bad and "$216B" in bad
