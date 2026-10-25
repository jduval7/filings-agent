"""Shared data and helpers for the eval tests: a tiny extraction example, model outputs,
a labelled split, and monkeypatch helpers."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from filings_agent.datasets import ExpectedFact, ExtractExample
from filings_agent.domain import Discrepancy, Metric
from filings_agent.sec.sections import Scale, StatementKind
from filings_agent.steps.extract import ExtractionOutput, ReportedValue
from tests.support.fakes import ScriptedChat

SECTION = (
    "**Consolidated Statements of Income** (In millions)\n"
    "| Total revenues | 5,000 |  | 4,000 |\n| Net income | 700 |  | 600 |\n"
)


def extract_example(split: str = "dev", accn: str = "A1") -> ExtractExample:
    return ExtractExample(
        example_id=f"{accn}:income_statement", split=split, ticker="GE", accession_no=accn,
        fiscal_year_end=date(2024, 12, 31), statement=StatementKind.INCOME, section_text=SECTION,
        metrics=[Metric.REVENUE, Metric.OPERATING_INCOME],
        expected=[
            ExpectedFact(metric=Metric.REVENUE, period_end=date(2024, 12, 31), value=Decimal(5e9)),
            ExpectedFact(metric=Metric.REVENUE, period_end=date(2023, 12, 31), value=Decimal(4e9)),
        ],
        expect_absent=[Metric.OPERATING_INCOME],
    )  # fmt: skip


def reported_value(printed: str, year: int) -> ReportedValue:
    return ReportedValue.model_validate(
        {
            "metric": "revenue",
            "row_label": "Total revenues",
            "printed_value": printed,
            "period_end": date(year, 12, 31),
            "scale": "millions",
        }  # fmt: skip
    )


PERFECT = ExtractionOutput(values=[reported_value("5,000", 2024), reported_value("4,000", 2023)],
                           not_reported=[Metric.OPERATING_INCOME])  # fmt: skip


WRONG_SCALE = ExtractionOutput(
    values=[
        reported_value("5,000", 2024).model_copy(update={"scale": Scale.THOUSANDS}),
        reported_value("4,000", 2023),
    ]
)


def patch_chat(monkeypatch: pytest.MonkeyPatch, module: Any, script: list[Any]) -> ScriptedChat:
    chat = ScriptedChat(script=script)

    def fake_chat_model(*_: Any, **__: Any) -> ScriptedChat:
        return chat

    monkeypatch.setattr(module, "chat_model", fake_chat_model)
    return chat


def returns(value: Any) -> Any:
    def fn(*_: Any, **__: Any) -> Any:
        return value

    return fn


def score_map(res: Any) -> dict[str, float]:
    """EvaluationResults -> {key: score} (score None -> 0.0)."""
    return {r.key: float(r.score or 0.0) for r in res.get("results", [])}


D = Discrepancy(
    ticker="NVDA", metric=Metric.DILUTED_SHARES, fiscal_year=2024, period_end=date(2024, 1, 28),
    values=(("N1", Decimal(2_494_000_000)), ("N2", Decimal(24_940_000_000))),
)  # fmt: skip


EVIDENCE = ("In June 2024, we executed a ten-for-one stock split of our common stock.",)
