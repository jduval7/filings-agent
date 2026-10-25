"""Extraction step with a fake model: parsing, scaling, verification, caching, scoring."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from filings_agent.config import ModelSpec
from filings_agent.domain import Metric
from filings_agent.evals.scoring import micro, score_extraction
from filings_agent.llm.cache import StepCache
from filings_agent.steps.extract import (
    ExtractionOutput,
    ExtractTask,
    LLMStatementExtractor,
    ReportedValue,
    parse_printed,
)

STATEMENT = (
    "**CONSOLIDATED STATEMENTS OF OPERATIONS**\n"
    "(In millions, except number of shares, which are reflected in thousands)\n"
    "|  | September 27, 2025 |  | September 28, 2024 |  |\n"
    "| Total net sales | 416,161 |  | 391,035 |  |\n"
    "| Net income | $ | 112,010 | $ | 93,736 |\n"
    "| Other income/(expense), net | (321) |  | 269 |  |\n"
    "| Diluted | 15,004,697 |  | 15,408,095 |  |\n"
)


class FakeStructuredModel(BaseChatModel):
    """Returns a fixed ExtractionOutput the way `with_structured_output(include_raw=True)` does."""

    output: ExtractionOutput | None
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - unused
        raise NotImplementedError

    def with_structured_output(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        def run(_: Any) -> dict[str, Any]:
            self.calls += 1
            msg = AIMessage(
                content="",
                usage_metadata={"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100},
            )
            return {"raw": msg, "parsed": self.output, "parsing_error": None}

        return RunnableLambda(run)


def _rv(metric: Metric, label: str, printed: str, end: date, scale: str) -> ReportedValue:
    return ReportedValue.model_validate(
        {
            "metric": metric,
            "row_label": label,
            "printed_value": printed,
            "period_end": end,
            "scale": scale,
        }
    )


TASK = ExtractTask(
    task_id="t1",
    ticker="AAPL",
    accession_no="0000320193-25-000079",
    fiscal_year_end=date(2025, 9, 27),
    statement="income_statement",
    section_text=STATEMENT,
    metrics=(Metric.REVENUE, Metric.NET_INCOME, Metric.DILUTED_SHARES, Metric.OPERATING_INCOME),
)


SPEC = ModelSpec(provider="bedrock", id="fake", price=(1.0, 5.0))


def _extractor(out: ExtractionOutput | None, tmp: Path) -> tuple[LLMStatementExtractor, Any]:
    fake = FakeStructuredModel(output=out)
    return LLMStatementExtractor("fake", SPEC, fake, StepCache(tmp), "v2"), fake


def test_parse_printed_formats() -> None:
    assert parse_printed("215,938") == Decimal(215938)
    assert parse_printed("$ 1,234") == Decimal(1234)
    assert parse_printed("(321)") == Decimal(-321)
    assert parse_printed("(41") == Decimal(-41)  # closing paren lives in the next cell
    assert parse_printed("1,609.4") == Decimal("1609.4")
    assert parse_printed("—") is None


def test_scales_applied_per_row_and_negatives_kept(tmp_path: Path) -> None:
    out = ExtractionOutput(
        values=[
            _rv(Metric.REVENUE, "Total net sales", "416,161", date(2025, 9, 27), "millions"),
            _rv(Metric.DILUTED_SHARES, "Diluted", "15,004,697", date(2025, 9, 27), "thousands"),
        ],
        not_reported=[Metric.OPERATING_INCOME],
    )
    ex, _ = _extractor(out, tmp_path)
    res = ex.extract(TASK)
    values = {f.metric: f.value for f in res.facts}
    assert values[Metric.REVENUE] == Decimal(416_161_000_000)
    assert values[Metric.DILUTED_SHARES] == Decimal(15_004_697_000)
    assert res.not_reported == [Metric.OPERATING_INCOME]
    assert res.usage is not None and res.usage.cost_usd == (1000 * 1.0 + 100 * 5.0) / 1e6


def test_values_not_in_the_text_are_rejected(tmp_path: Path) -> None:
    out = ExtractionOutput(
        values=[
            _rv(Metric.NET_INCOME, "Net income", "112,999", date(2025, 9, 27), "millions"),
            _rv(Metric.REVENUE, "Revenue (made up)", "416,161", date(2025, 9, 27), "millions"),
        ]
    )
    ex, _ = _extractor(out, tmp_path)
    res = ex.extract(TASK)
    assert res.facts == []
    reasons = sorted(r.reason.split()[0] for r in res.rejected)
    assert reasons == ["printed", "row"]


def test_schema_failure_is_reported_not_raised(tmp_path: Path) -> None:
    ex, _ = _extractor(None, tmp_path)
    res = ex.extract(TASK)
    assert res.error is not None and res.facts == []


def test_cache_returns_original_cost_and_skips_the_call(tmp_path: Path) -> None:
    out = ExtractionOutput(
        values=[_rv(Metric.REVENUE, "Total net sales", "416,161", date(2025, 9, 27), "millions")]
    )
    ex, fake = _extractor(out, tmp_path)
    first = ex.extract(TASK)
    second = ex.extract(TASK)
    third = ex.extract(TASK, rep=1)  # a new repetition is a real call
    assert fake.calls == 2
    assert second.usage is not None and second.usage.cached
    assert first.usage is not None and second.usage.cost_usd == first.usage.cost_usd
    assert third.usage is not None and not third.usage.cached


def test_scoring_outcomes() -> None:
    expected = [
        {"metric": "revenue", "period_end": "2025-09-27", "value": "416161000000"},
        {"metric": "revenue", "period_end": "2024-09-28", "value": "391035000000"},
        {"metric": "net_income", "period_end": "2025-09-27", "value": "112010000000"},
    ]
    predicted = [
        {"metric": "revenue", "fiscal_year": 2025, "value": "416161000000"},  # correct
        {"metric": "revenue", "fiscal_year": 2024, "value": "391035000"},  # wrong scale
        {"metric": "operating_income", "fiscal_year": 2025, "value": "1"},  # should abstain
    ]
    s = score_extraction(expected, ["operating_income"], predicted)
    assert (s.correct, s.wrong_value, s.missing, s.hallucinated) == (1, 1, 1, 1)
    assert s.precision == 1 / 3 and s.recall == 1 / 3
    assert s.abstain_correct == 0
    m = micro([s])
    assert m["abstain_accuracy"] == 0.0


def test_failed_output_gets_no_abstain_credit() -> None:
    s = score_extraction([], ["operating_income"], [], schema_error=True)
    assert s.abstain_correct == 0 and s.precision == 0.0
