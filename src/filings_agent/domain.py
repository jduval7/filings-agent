"""Core data shapes shared by the data layer, the graph, and the evals.

A *fact* is one number from one filing: (company, metric, period, filing) -> value.
The same (company, metric, period) can appear in several filings with different values
(stock splits, spin-off recasts). That is a *discrepancy*, not an error, so the filing's
accession number is part of a fact's identity.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class Metric(StrEnum):
    REVENUE = "revenue"
    OPERATING_INCOME = "operating_income"
    NET_INCOME = "net_income"
    LONG_TERM_DEBT = "long_term_debt"
    CASH = "cash"
    DILUTED_SHARES = "diluted_shares"
    OPERATING_CASH_FLOW = "operating_cash_flow"


class PeriodKind(StrEnum):
    DURATION = "duration"  # a flow over a fiscal year (income statement)
    INSTANT = "instant"  # a balance at a fiscal year-end (balance sheet)


class Unit(StrEnum):
    USD = "USD"
    SHARES = "shares"


@dataclass(frozen=True)
class MetricSpec:
    metric: Metric
    kind: PeriodKind
    unit: Unit
    definition: str
    # XBRL concepts in priority order. Used ONLY to build the answer key, never by the agent.
    concepts: tuple[str, ...]


METRIC_SPECS: dict[Metric, MetricSpec] = {
    s.metric: s
    for s in (
        MetricSpec(
            Metric.REVENUE,
            PeriodKind.DURATION,
            Unit.USD,
            "Total revenues for the fiscal year as shown on the income statement.",
            (
                "Revenues",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "SalesRevenueNet",
            ),
        ),
        MetricSpec(
            Metric.OPERATING_INCOME,
            PeriodKind.DURATION,
            Unit.USD,
            "Operating income (loss) for the fiscal year. Not every company reports it.",
            ("OperatingIncomeLoss",),
        ),
        MetricSpec(
            Metric.NET_INCOME,
            PeriodKind.DURATION,
            Unit.USD,
            "Net income (loss) attributable to the company (excludes noncontrolling interests).",
            ("NetIncomeLoss",),
        ),
        MetricSpec(
            Metric.LONG_TERM_DEBT,
            PeriodKind.INSTANT,
            Unit.USD,
            "Long-term debt, noncurrent portion, as shown on the balance sheet.",
            ("LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"),
        ),
        MetricSpec(
            Metric.CASH,
            PeriodKind.INSTANT,
            Unit.USD,
            "Cash and cash equivalents at fiscal year-end, as shown on the balance sheet. If the "
            "balance sheet only shows a combined line including restricted cash (GE), use that.",
            (
                "CashAndCashEquivalentsAtCarryingValue",
                "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
            ),
        ),
        MetricSpec(
            Metric.DILUTED_SHARES,
            PeriodKind.DURATION,
            Unit.SHARES,
            "Weighted-average diluted shares outstanding for the fiscal year.",
            ("WeightedAverageNumberOfDilutedSharesOutstanding",),
        ),
        MetricSpec(
            Metric.OPERATING_CASH_FLOW,
            PeriodKind.DURATION,
            Unit.USD,
            "Net cash provided by (used in) operating activities for the fiscal year.",
            (
                "NetCashProvidedByUsedInOperatingActivities",
                "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
            ),
        ),
    )
}


class Period(BaseModel):
    """A fiscal year (duration) or a fiscal year-end (instant). Keyed on dates, not labels,
    because fiscal-year labels differ between companies."""

    model_config = ConfigDict(frozen=True)

    start: date | None
    end: date

    @property
    def kind(self) -> PeriodKind:
        return PeriodKind.INSTANT if self.start is None else PeriodKind.DURATION

    @property
    def fiscal_year(self) -> int:
        return fiscal_year_of(self.end)

    def key(self) -> str:
        return f"{self.start.isoformat() if self.start else ''}..{self.end.isoformat()}"


class FilingRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    cik: int
    accession_no: str
    form: str
    filed: date
    period_of_report: date


def fiscal_year_of(end: date) -> int:
    """Fiscal-year label from a period end date, robust to 52/53-week years.

    NVIDIA FY2026 ends 2026-01-25, J&J FY2022 ends 2023-01-01. Shifting back 15 days maps both
    to the label the company itself uses. Keying on the label (not the exact date) means an
    extractor that reads "2024" from a column header, without the exact day, still lands on the
    same fact.
    """
    return (end - timedelta(days=15)).year


def fact_id(ticker: str, metric: Metric, fiscal_year: int, accession_no: str) -> str:
    """Stable identity for a fact. Same inputs -> same id, which makes writes idempotent.

    In one filing, a metric has exactly one value per fiscal year.
    """
    raw = f"{ticker}|{metric.value}|{fiscal_year}|{accession_no}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class GoldFact(BaseModel):
    """One answer-key value from SEC XBRL company facts."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    metric: Metric
    period: Period
    value: Decimal
    accession_no: str
    concept: str

    @property
    def fiscal_year(self) -> int:
        return self.period.fiscal_year

    @property
    def period_end(self) -> date:
        return self.period.end

    @property
    def id(self) -> str:
        return fact_id(self.ticker, self.metric, self.period.fiscal_year, self.accession_no)


class ExtractedFact(BaseModel):
    """One number the agent read from a filing, with enough provenance to check it by hand."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    metric: Metric
    fiscal_year: int
    period_end: date
    value: Decimal
    accession_no: str
    statement: str
    row_label: str
    printed_value: str
    scale: str
    model: str

    @property
    def id(self) -> str:
        return fact_id(self.ticker, self.metric, self.fiscal_year, self.accession_no)


class Discrepancy(BaseModel):
    """The same company/metric/period reported with different values in different filings."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    metric: Metric
    fiscal_year: int
    period_end: date
    # accession_no -> value, oldest filing first
    values: tuple[tuple[str, Decimal], ...]

    @property
    def id(self) -> str:
        return f"{self.ticker}:{self.metric.value}:FY{self.fiscal_year}"

    @property
    def ratio(self) -> Decimal:
        """Newest value / oldest value. ~4 or ~10 smells like a stock split."""
        oldest, newest = self.values[0][1], self.values[-1][1]
        return newest / oldest if oldest != 0 else Decimal(0)
