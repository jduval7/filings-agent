""""""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from filings_agent.config import ModelSpec
from filings_agent.domain import ExtractedFact, Metric
from filings_agent.steps.memo import (
    MemoOutput,
    MemoSection,
    MemoWriter,
    allowed_numbers,
    build_table,
    unverified_numbers,
)
from filings_agent.steps.reconcile import Cause, Resolution


def _fact(fy: int, value: int, accn: str, metric: Metric = Metric.REVENUE) -> ExtractedFact:
    return ExtractedFact(
        ticker="NVDA", metric=metric, fiscal_year=fy, period_end=date(fy, 1, 28),
        value=Decimal(value), accession_no=accn, statement="income_statement",
        row_label="Revenue", printed_value="x", scale="millions", model="fake",
    )  # fmt: skip


FACTS = [
    _fact(2025, 130_497_000_000, "A2"),
    _fact(2026, 215_938_000_000, "A2"),
    _fact(2024, 2_494_000_000, "A1", Metric.DILUTED_SHARES),
    _fact(2024, 24_940_000_000, "A2", Metric.DILUTED_SHARES),
]


def test_table_and_changes_are_computed_in_code() -> None:
    rows, changes = build_table(FACTS, {})
    displays = {(r.metric, r.fiscal_year): r.display for r in rows}
    assert displays[(Metric.REVENUE, 2026)] == "$215.9B"
    assert displays[(Metric.DILUTED_SHARES, 2024)] == "24.94B shares"  # newest filing wins
    assert "NVDA revenue FY2026 vs FY2025: +65.5%" in changes


def test_no_comparison_across_bases() -> None:
    # FY2023 shares only exist in the old (pre-split) filing; FY2024 in the new one.
    facts = [
        _fact(2023, 2_507_000_000, "A1", Metric.DILUTED_SHARES),
        _fact(2024, 24_940_000_000, "A2", Metric.DILUTED_SHARES),
    ]
    rows, changes = build_table(facts, {})
    assert [r.fiscal_year for r in rows] == [2024]  # only the latest filing's basis
    assert changes == []  # no "+895%" from mixing pre- and post-split counts


def test_human_decision_to_use_original_flows_into_the_table() -> None:
    res = Resolution(
        "NVDA:diluted_shares:FY2024", Cause.STOCK_SPLIT, "split", "q", True, True,
        Decimal(2_494_000_000), "A1", True,
    )  # fmt: skip
    rows, _ = build_table(FACTS, {res.discrepancy_id: res})
    shares = next(r for r in rows if r.metric is Metric.DILUTED_SHARES)
    assert shares.display == "2.49B shares" and shares.changed_by == "stock_split"


def test_checker_flags_invented_and_misrounded_numbers() -> None:
    rows, changes = build_table(FACTS, {})
    allowed = allowed_numbers(rows, changes)
    ok = "NVDA revenue reached $215.9B in FY2026, up +65.5% on two strong years."
    assert unverified_numbers(ok, allowed) == []
    bad = "Revenue was $216B, up 66% and margins hit 75%."
    assert unverified_numbers(bad, allowed) == ["$216B", "66%", "75%"]


class ScriptedModel(BaseChatModel):
    """Returns the next scripted MemoOutput each call."""

    script: list[MemoOutput]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def with_structured_output(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        def run(_: Any) -> dict[str, Any]:
            out = self.script[min(self.calls, len(self.script) - 1)]
            self.calls += 1
            usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
            return {"raw": AIMessage(content="", usage_metadata=usage), "parsed": out}

        return RunnableLambda(run)


def _memo(text: str) -> MemoOutput:
    return MemoOutput(
        title="NVDA", summary=text, sections=[MemoSection(heading="NVDA", body="Growth.")],
        caveats="Shares were restated for a stock split.",
    )  # fmt: skip


SPEC = ModelSpec(provider="bedrock", id="fake", price=(1.0, 5.0))


def test_writer_retries_with_feedback_then_verifies() -> None:
    model = ScriptedModel(script=[_memo("Revenue hit $216B."), _memo("Revenue hit $215.9B.")])
    rows, changes = build_table(FACTS, {})
    memo = MemoWriter("fake", SPEC, model).write(rows, changes)
    assert memo.verified and memo.attempts == 2 and model.calls == 2
    assert "NOT VERIFIED" not in memo.markdown


def test_writer_ships_banner_when_it_never_verifies() -> None:
    model = ScriptedModel(script=[_memo("Revenue hit $216B.")])
    rows, changes = build_table(FACTS, {})
    memo = MemoWriter("fake", SPEC, model).write(rows, changes)
    assert not memo.verified and memo.unverified == ["$216B"]
    assert "NOT VERIFIED" in memo.markdown
