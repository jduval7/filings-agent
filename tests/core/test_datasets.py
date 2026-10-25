"""Dataset builders: what counts as an expected fact, an abstention, or a material discrepancy."""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

from filings_agent.config import AppConfig
from filings_agent.datasets import (
    ExtractExample,
    build_extract_dataset,
    build_reconcile_dataset,
    is_answerable,
    summarize,
    summarize_reconcile,
    write_jsonl,
)
from filings_agent.domain import Discrepancy, FilingRef, GoldFact, Metric, Period
from filings_agent.sec.sections import StatementKind

ROWS = "\n".join(f"| Line {i} | {1000 + i:,} |  | {900 + i:,} |  |" for i in range(8))


MD = (
    "**Consolidated Statements of Income**\n(In millions)\n"
    f"|  | 2024 |  | 2023 |  |\n{ROWS}\n| Total revenues | $ | 5,000 | $ | 4,000 |\n"
    "| Net income | $ | 700 | $ | 600 |\n\n"
    "**Consolidated Balance Sheets**\n(In millions)\n"
    f"|  | 2024 |  | 2023 |  |\n{ROWS}\n| Cash and cash equivalents | 300 |  | 250 |  |\n"
    "| Total assets | $ | 9,000 | $ | 8,000 |\n"
)


FILING = FilingRef(
    ticker="GE", cik=1, accession_no="A1", form="10-K", filed=date(2025, 2, 1),
    period_of_report=date(2024, 12, 31),
)  # fmt: skip


def _gold(metric: Metric, value: int, start: date | None, end: date) -> GoldFact:
    return GoldFact(
        ticker="GE", metric=metric, period=Period(start=start, end=end),
        value=Decimal(value), accession_no="A1", concept="x",
    )  # fmt: skip


def test_answerable_needs_two_significant_digits() -> None:
    assert not is_answerable(Decimal(5_000_000), "| x | 5 |")  # "5" alone is too ambiguous
    assert is_answerable(Decimal(-321_000_000), "| Other | (321) |")


def test_extract_examples_expected_facts_and_abstentions(cfg: AppConfig) -> None:
    y24, y23 = (date(2024, 1, 1), date(2024, 12, 31)), (date(2023, 1, 1), date(2023, 12, 31))
    gold = [
        _gold(Metric.REVENUE, 5_000_000_000, *y24),
        _gold(Metric.REVENUE, 4_000_000_000, *y23),
        _gold(Metric.NET_INCOME, 700_000_000, *y24),
        _gold(Metric.DILUTED_SHARES, 1_234_000_000, *y24),  # not printed on the statement
        _gold(Metric.CASH, 300_000_000, None, date(2024, 12, 31)),
    ]
    examples = build_extract_dataset(cfg, [FILING], gold, {"A1": MD})
    by_kind = {e.statement: e for e in examples}
    assert set(by_kind) == {StatementKind.INCOME, StatementKind.BALANCE}  # no cash-flow section
    inc = by_kind[StatementKind.INCOME]
    assert inc.example_id == "A1:income_statement" and inc.split == "dev"
    assert {(e.metric, e.period_end.year) for e in inc.expected} == {
        (Metric.REVENUE, 2024), (Metric.REVENUE, 2023), (Metric.NET_INCOME, 2024),
    }  # fmt: skip
    # operating income has no gold at all; diluted shares has gold but isn't on the statement
    assert set(inc.expect_absent) == {Metric.OPERATING_INCOME, Metric.DILUTED_SHARES}
    bal = by_kind[StatementKind.BALANCE]
    assert [e.metric for e in bal.expected] == [Metric.CASH]
    assert bal.expect_absent == [Metric.LONG_TERM_DEBT]


def test_reconcile_examples_flag_materiality(cfg: AppConfig) -> None:
    def d(ticker: str, a: int, b: int) -> Discrepancy:
        return Discrepancy(
            ticker=ticker, metric=Metric.REVENUE, fiscal_year=2023,
            period_end=date(2023, 12, 31), values=(("A1", Decimal(a)), ("A2", Decimal(b))),
        )  # fmt: skip

    filed = {"A1": date(2024, 2, 1), "A2": date(2025, 2, 1)}
    rows = build_reconcile_dataset(cfg, [d("GE", 100, 60), d("WMT", 9481, 9482)], filed)
    assert [(r.ticker, r.material, r.split) for r in rows] == [
        ("GE", True, "dev"), ("WMT", False, "test"),
    ]  # fmt: skip
    assert rows[0].ratio == Decimal("0.6") and rows[0].values[1] == (
        "A2",
        date(2025, 2, 1),
        Decimal(60),
    )
    assert rows[0].cause is None  # labels are added separately, after reading the filing
    assert summarize_reconcile(rows) == (
        "reconcile examples: {('dev', 'material'): 1, ('test', 'immaterial'): 1}"
    )


def test_write_jsonl_round_trips(cfg: AppConfig, tmp_path: Path) -> None:
    examples = build_extract_dataset(cfg, [FILING], [], {"A1": MD})
    path = tmp_path / "x" / "extract.jsonl"
    write_jsonl(path, examples)
    back = [ExtractExample.model_validate_json(line) for line in path.read_text().splitlines()]
    assert back == examples
    assert "expect 'not reported'" in summarize(examples)
    assert json.loads(path.read_text().splitlines()[0])["ticker"] == "GE"
