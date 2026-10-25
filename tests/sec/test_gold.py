"""Edge cases across steps: the small branches that tend to break in production."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

from filings_agent.domain import FilingRef, Metric, fact_id
from filings_agent.sec.gold import build_gold, find_discrepancies

A_OLD = "0000000001-23-000001"


A_NEW = "0000000001-24-000001"


def _filing(accn: str, fy_end: date, filed: date) -> FilingRef:
    return FilingRef(
        ticker="TST", cik=1, accession_no=accn, form="10-K", filed=filed, period_of_report=fy_end
    )


def _fact(
    val: int, accn: str, end: str, start: str | None = None, form: str = "10-K"
) -> dict[str, Any]:
    x: dict[str, Any] = {"val": val, "accn": accn, "end": end, "form": form}
    if start:
        x["start"] = start
    return x


def _company_facts() -> dict[str, Any]:
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            _fact(100, A_OLD, "2022-12-31", "2022-01-01"),
                            _fact(90, A_OLD, "2021-12-31", "2021-01-01"),
                            _fact(25, A_OLD, "2022-12-31", "2022-10-01"),  # quarter: ignored
                            _fact(60, A_NEW, "2022-12-31", "2022-01-01"),  # recast
                            _fact(120, A_NEW, "2023-12-31", "2023-01-01"),
                            _fact(999, A_NEW, "2023-12-31", "2023-01-01", form="10-Q"),
                        ]
                    }
                },
                "CashAndCashEquivalentsAtCarryingValue": {
                    "units": {
                        "USD": [
                            _fact(5, A_NEW, "2023-12-31"),
                            _fact(4, A_NEW, "2022-12-31"),
                            _fact(3, A_NEW, "2021-12-31"),  # cash-flow opening balance: not on BS
                        ]
                    }
                },
            }
        }
    }


FILINGS = [
    _filing(A_OLD, date(2022, 12, 31), date(2023, 2, 1)),
    _filing(A_NEW, date(2023, 12, 31), date(2024, 2, 1)),
]


def test_gold_keeps_annual_10k_facts_per_filing() -> None:
    res = build_gold("TST", _company_facts(), FILINGS, [Metric.REVENUE, Metric.CASH])
    rev = {(g.accession_no, g.period.end): g.value for g in res.facts if g.metric is Metric.REVENUE}
    assert rev == {
        (A_OLD, date(2022, 12, 31)): Decimal(100),
        (A_OLD, date(2021, 12, 31)): Decimal(90),
        (A_NEW, date(2022, 12, 31)): Decimal(60),
        (A_NEW, date(2023, 12, 31)): Decimal(120),
    }


def test_gold_balance_sheet_limited_to_two_year_ends() -> None:
    res = build_gold("TST", _company_facts(), FILINGS, [Metric.REVENUE, Metric.CASH])
    cash_ends = sorted(g.period.end for g in res.facts if g.metric is Metric.CASH)
    assert cash_ends == [date(2022, 12, 31), date(2023, 12, 31)]


def test_missing_concept_is_reported_not_invented() -> None:
    res = build_gold("TST", _company_facts(), FILINGS, [Metric.OPERATING_INCOME])
    assert res.facts == []
    assert {(a, m) for _, a, m in res.missing} == {
        (A_OLD, Metric.OPERATING_INCOME),
        (A_NEW, Metric.OPERATING_INCOME),
    }


def test_discrepancy_found_and_ordered_by_filing_date() -> None:
    res = build_gold("TST", _company_facts(), FILINGS, [Metric.REVENUE])
    filed = {f.accession_no: f.filed for f in FILINGS}
    (d,) = find_discrepancies(res.facts, filed)
    assert d.period_end == date(2022, 12, 31)
    assert d.values == ((A_OLD, Decimal(100)), (A_NEW, Decimal(60)))
    assert d.ratio == Decimal("0.6")


def test_fact_id_is_stable_and_distinguishes_filings() -> None:
    a = fact_id("TST", Metric.REVENUE, 2022, A_OLD)
    assert a == fact_id("TST", Metric.REVENUE, 2022, A_OLD)
    assert a != fact_id("TST", Metric.REVENUE, 2022, A_NEW)


def test_fiscal_year_label_handles_52_53_week_years() -> None:
    from filings_agent.domain import fiscal_year_of

    assert fiscal_year_of(date(2026, 1, 25)) == 2026  # NVIDIA FY2026
    assert fiscal_year_of(date(2023, 1, 1)) == 2022  # J&J FY2022
    assert fiscal_year_of(date(2025, 9, 27)) == 2025  # Apple FY2025


def test_gold_skips_conflicting_values_within_one_filing() -> None:
    facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        {"val": 1, "accn": "A", "end": "2024-12-31", "start": "2024-01-01", "form": "10-K"},
        {"val": 2, "accn": "A", "end": "2024-12-31", "start": "2024-01-01", "form": "10-K"},
    ]}}}}}  # fmt: skip
    ref = FilingRef(ticker="X", cik=1, accession_no="A", form="10-K", filed=date(2025, 2, 1),
                    period_of_report=date(2024, 12, 31))  # fmt: skip
    res = build_gold("X", facts, [ref], [Metric.REVENUE])
    assert res.facts == [] and "conflicting" in res.warnings[0]


def test_gold_ignores_facts_after_the_report_period() -> None:
    facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        {"val": 1, "accn": "A", "end": "2025-12-31", "start": "2025-01-01", "form": "10-K"},
    ]}}}}}  # fmt: skip
    ref = FilingRef(ticker="X", cik=1, accession_no="A", form="10-K", filed=date(2025, 2, 1),
                    period_of_report=date(2024, 12, 31))  # fmt: skip
    assert build_gold("X", facts, [ref], [Metric.REVENUE]).facts == []
