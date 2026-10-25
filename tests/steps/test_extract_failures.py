"""Extraction failure handling: model errors, unrequested metrics, unparseable values,
trace metadata, and verification."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError

from filings_agent.config import ModelSpec
from filings_agent.domain import Metric
from filings_agent.llm.cache import StepCache
from filings_agent.steps.extract import (
    ExtractionOutput,
    ExtractTask,
    LLMStatementExtractor,
    ReportedValue,
    verify,
)
from tests.support.fakes import ScriptedChat, client_error

TASK = ExtractTask(
    "t", "AAPL", "A1", date(2025, 9, 27), "income_statement",
    "| Total net sales | 416,161 |  | 391,035 |", (Metric.REVENUE,),
)  # fmt: skip


SPEC = ModelSpec(provider="bedrock", id="fake", price=(1.0, 5.0))


def _rv(**kw: Any) -> ReportedValue:
    base: dict[str, Any] = {
        "metric": Metric.REVENUE, "row_label": "Total net sales", "printed_value": "416,161",
        "period_end": date(2025, 9, 27), "scale": "millions",
    }  # fmt: skip
    return ReportedValue.model_validate(base | kw)


def test_extract_model_error_recorded_throttle_raised(tmp_path: Path) -> None:
    bad_output = ScriptedChat(script=[client_error("ValidationException")])
    ex = LLMStatementExtractor("f", SPEC, bad_output, StepCache(tmp_path), "v2")
    res = ex.extract(TASK)
    assert res.error == "model_error:ValidationException" and res.facts == []
    throttled = ScriptedChat(script=[client_error("ThrottlingException")])
    ex2 = LLMStatementExtractor("f", SPEC, throttled, StepCache(tmp_path / "b"), "v2")
    with pytest.raises(ClientError):
        ex2.extract(TASK)


def test_extract_ignores_unrequested_metrics_and_duplicates(tmp_path: Path) -> None:
    out = ExtractionOutput(
        values=[_rv(), _rv(), _rv(metric=Metric.CASH)],  # dup + a metric nobody asked for
        not_reported=[Metric.CASH, Metric.REVENUE],
    )
    ex = LLMStatementExtractor("f", SPEC, ScriptedChat(script=[out]), StepCache(tmp_path), "v2")
    res = ex.extract(TASK)
    assert len(res.facts) == 1 and res.facts[0].value == Decimal(416_161_000_000)
    assert res.not_reported == [Metric.REVENUE]  # CASH wasn't requested


def test_extract_rejects_unparseable_value(tmp_path: Path) -> None:
    task = ExtractTask(
        "t", "X", "A1", date(2025, 1, 1), "s", "| Revenue | — | 5 |", (Metric.REVENUE,)
    )
    out = ExtractionOutput(values=[_rv(row_label="Revenue", printed_value="—")])
    ex = LLMStatementExtractor("f", SPEC, ScriptedChat(script=[out]), StepCache(tmp_path), "v2")
    (rej,) = ex.extract(task).rejected
    assert "unparseable" in rej.reason


def test_extract_attaches_trace_metadata(tmp_path: Path) -> None:
    chat = ScriptedChat(script=[ExtractionOutput(values=[])])
    LLMStatementExtractor("gpt-x", SPEC, chat, StepCache(tmp_path), "v2").extract(TASK, rep=3)
    cfg = chat.seen_configs[0]
    assert cfg["run_name"] == "extract AAPL income_statement"
    assert cfg["metadata"] == {
        "step": "extract", "ticker": "AAPL", "accession_no": "A1", "statement": "income_statement",
        "model": "gpt-x", "prompt_version": "v2", "rep": 3,
    }  # fmt: skip


def test_verify_requires_both_number_and_label() -> None:
    assert verify(_rv(), TASK.section_text) is None
    assert "printed value" in str(verify(_rv(printed_value="999,999"), TASK.section_text))
    assert "row label" in str(verify(_rv(row_label="Revenue, net"), TASK.section_text))
    assert "printed value" in str(verify(_rv(printed_value="$ "), TASK.section_text))
