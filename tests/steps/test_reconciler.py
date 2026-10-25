"""LLMReconciler with a scripted model, plus the deterministic helpers around it."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from filings_agent.config import ModelSpec
from filings_agent.domain import Discrepancy, Metric
from filings_agent.llm.cache import StepCache
from filings_agent.steps.reconcile import (
    PROMPTS,
    Cause,
    LLMReconciler,
    ReconcileOutput,
    ReconcileTask,
    auto_resolution,
    build_messages,
    change_filings,
    evidence_for,
    spread,
)
from tests.support.fakes import ScriptedChat, client_error

EVIDENCE = (
    "In June 2024, we executed a ten\\-for\\-one stock split of our common stock\\. All share "
    "amounts have been retroactively adjusted\\.",
)


D = Discrepancy(
    ticker="NVDA", metric=Metric.DILUTED_SHARES, fiscal_year=2024, period_end=date(2024, 1, 28),
    values=(("A1", Decimal(2_494_000_000)), ("A2", Decimal(24_940_000_000))),
)  # fmt: skip


TASK = ReconcileTask(D.id, D, EVIDENCE)


SPEC = ModelSpec(provider="bedrock", id="fake", price=(1.0, 5.0))


def _rec(script: list[object], tmp: Path, prompt: str = "v2") -> tuple[LLMReconciler, ScriptedChat]:
    chat = ScriptedChat(script=script)
    return LLMReconciler("fake", SPEC, chat, StepCache(tmp), prompt), chat


def _out(cause: Cause, quote: str) -> ReconcileOutput:
    return ReconcileOutput(cause=cause, explanation="because", evidence_quote=quote)


def test_verified_answer_uses_newest_value_and_needs_review(tmp_path: Path) -> None:
    rec, _ = _rec([_out(Cause.STOCK_SPLIT, "we executed a ten-for-one stock split")], tmp_path)
    r = rec.reconcile(TASK)
    assert (r.cause, r.verified, r.material, r.needs_review) == (
        Cause.STOCK_SPLIT,
        True,
        True,
        True,
    )
    assert (r.use_accession, r.use_value) == ("A2", Decimal(24_940_000_000))
    assert r.usage is not None and r.usage.cost_usd == pytest.approx((1000 + 500) / 1e6)


def test_invented_quote_is_downgraded_to_unknown(tmp_path: Path) -> None:
    rec, _ = _rec([_out(Cause.STOCK_SPLIT, "we executed a five-for-one stock split")], tmp_path)
    r = rec.reconcile(TASK)
    assert r.cause is Cause.UNKNOWN and not r.verified
    assert r.explanation.startswith("Unverified claim:")


def test_unknown_needs_no_quote(tmp_path: Path) -> None:
    rec, _ = _rec([_out(Cause.UNKNOWN, "")], tmp_path)
    r = rec.reconcile(TASK)
    assert r.cause is Cause.UNKNOWN and r.verified  # an honest "don't know" isn't a violation


def test_schema_failure_becomes_unknown_needing_review(tmp_path: Path) -> None:
    rec, _ = _rec([None], tmp_path)
    r = rec.reconcile(TASK)
    assert r.cause is Cause.UNKNOWN and r.needs_review and r.error


def test_model_error_is_an_outcome_but_throttling_is_raised(tmp_path: Path) -> None:
    rec, _ = _rec([client_error("ModelErrorException")], tmp_path)
    r = rec.reconcile(TASK)
    assert r.error == "model_error:ModelErrorException" and r.cause is Cause.UNKNOWN
    rec2, _ = _rec([client_error("ThrottlingException")], tmp_path / "b")
    with pytest.raises(ClientError):
        rec2.reconcile(TASK)  # the graph's RetryPolicy handles this one


def test_results_are_cached_per_prompt_and_rep(tmp_path: Path) -> None:
    rec, chat = _rec([_out(Cause.STOCK_SPLIT, "we executed a ten-for-one stock split")], tmp_path)
    rec.reconcile(TASK)
    again = rec.reconcile(TASK)
    assert chat.calls == 1 and again.usage is not None and again.usage.cached
    rec.reconcile(TASK, rep=1)
    assert chat.calls == 2
    other_prompt, chat2 = _rec([_out(Cause.UNKNOWN, "")], tmp_path, prompt="v1")
    other_prompt.reconcile(TASK)
    assert chat2.calls == 1  # different prompt version -> different cache key


def test_trace_metadata_is_attached(tmp_path: Path) -> None:
    rec, chat = _rec([_out(Cause.UNKNOWN, "")], tmp_path)
    rec.reconcile(TASK)
    cfg = chat.seen_configs[0]
    assert cfg["run_name"] == "reconcile NVDA:diluted_shares:FY2024"
    assert cfg["metadata"]["prompt_version"] == "v2" and "step:reconcile" in cfg["tags"]


def test_prompt_and_messages_carry_the_facts() -> None:
    system, human = build_messages(TASK, "v2")
    assert system.content == PROMPTS["v2"]
    text = str(human.content)
    assert "2,494,000,000" in text and "24,940,000,000" in text and "10.0000" in text
    assert "ten\\-for\\-one" in text


def test_spread_change_filings_and_auto_resolution() -> None:
    d = Discrepancy(
        ticker="GE", metric=Metric.NET_INCOME, fiscal_year=2022, period_end=date(2022, 12, 31),
        values=(("A1", Decimal(225)), ("A2", Decimal(339)), ("A3", Decimal(339))),
    )  # fmt: skip
    assert spread(d) == Decimal(114) / Decimal(339)
    assert change_filings(d) == ["A2"]  # A3 repeated A2's value: it didn't change anything
    auto = auto_resolution(d.id, d)
    assert auto.cause is Cause.IMMATERIAL_REVISION and not auto.needs_review
    assert auto.use_accession == "A3"


def test_evidence_comes_from_every_filing_that_changed_the_number() -> None:
    d = Discrepancy(
        ticker="GE", metric=Metric.REVENUE, fiscal_year=2022, period_end=date(2022, 12, 31),
        values=(("A1", Decimal(76)), ("A2", Decimal(58)), ("A3", Decimal(29))),
    )  # fmt: skip
    filler = "Some unrelated words about the business and its markets. " * 20
    docs = {
        "A1": filler,
        "A2": filler + "We completed the spin-off of the HealthCare business." + filler,
        "A3": filler + "We adopted ASU 2018-12 on a retrospective basis this year." + filler,
    }
    ev = evidence_for(d, docs.__getitem__)
    joined = " ".join(ev)
    assert "spin-off" in joined and "ASU 2018-12" in joined  # both change points represented
