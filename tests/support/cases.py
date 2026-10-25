"""Real-shaped cases shared across tests (GE net income FY2022: the case that drove the
review card)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from filings_agent.domain import Discrepancy, Metric

# Shapes of the real GE 2024 10-K text (markdown escapes and a heading mid-sentence included).
GE_EVIDENCE = (
    "On January 3, 2023, we completed the separation of the HealthCare business from GE through "
    "the spin\\-off of GE HealthCare Technologies Inc\\. The historical results of GE HealthCare "
    "are now reported as discontinued operations\\.",
    "Genworth has been closed to new business since 2004\\. ## On January 1, 2023, we adopted "
    "ASU No\\. 2018\\-12, # Financial Services – Insurance (Topic 944): Targeted Improvements to "
    "the Accounting for Long\\-Duration Contracts\\.",
)


GE_NET_INCOME = Discrepancy(
    ticker="GE",
    metric=Metric.NET_INCOME,
    fiscal_year=2022,
    period_end=date(2022, 12, 31),
    values=(("0000040545-23-000001", Decimal(225)), ("0000040545-24-000027", Decimal(339))),
)
