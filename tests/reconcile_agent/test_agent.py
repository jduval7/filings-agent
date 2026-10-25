"""Reconcile agent, retrieval and tools, tested the way the LangChain docs recommend: a
GenericFakeChatModel subclass whose bind_tools returns itself, scripted turn by turn, and fake
embeddings. No network, no LLM."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolCall

from filings_agent.config import ModelSpec
from filings_agent.domain import Discrepancy, FilingRef, Metric
from filings_agent.llm.cache import StepCache
from filings_agent.reconcile_agent.agent import AgentReconciler, question
from filings_agent.reconcile_agent.retrieval import (
    CachedEmbeddings,
    FilingIndex,
    chunk_filing,
    is_table_chunk,
)
from filings_agent.steps.reconcile import Cause, ReconcileTask

OLD, NEW = "0001045810-24-000029", "0001045810-25-000023"


SPLIT = "In June 2024, we executed a ten-for-one stock split of our common stock."


FILLER = "Our business includes data center platforms and gaming products sold worldwide. " * 30


DOCS = {OLD: FILLER, NEW: FILLER + "\n\n" + SPLIT + "\n\n" + FILLER}


FILINGS = {
    OLD: FilingRef(
        ticker="NVDA",
        cik=1,
        accession_no=OLD,
        form="10-K",
        filed=date(2024, 2, 21),
        period_of_report=date(2024, 1, 28),
    ),
    NEW: FilingRef(
        ticker="NVDA",
        cik=1,
        accession_no=NEW,
        form="10-K",
        filed=date(2025, 2, 26),
        period_of_report=date(2025, 1, 26),
    ),
}


D = Discrepancy(
    ticker="NVDA",
    metric=Metric.DILUTED_SHARES,
    fiscal_year=2024,
    period_end=date(2024, 1, 28),
    values=((OLD, Decimal(2_494_000_000)), (NEW, Decimal(24_940_000_000))),
)


TASK = ReconcileTask(D.id, D, ())


SPEC = ModelSpec(provider="bedrock", id="fake-agent-model", price=(1.0, 5.0))


class FakeToolCallingModel(GenericFakeChatModel):
    """From the LangChain unit-testing docs: tool binding is a no-op for the fake."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        return self


def _turns(*messages: AIMessage) -> Iterator[AIMessage]:
    return iter(messages)


def _call(name: str, args: dict[str, Any], i: int) -> AIMessage:
    return AIMessage(content="", tool_calls=[ToolCall(name=name, args=args, id=f"call_{i}")])


def _answer(cause: str, quote: str, i: int = 9) -> AIMessage:
    return _call(
        "AgentAnswer",
        {
            "cause": cause,
            "explanation": "A 10-for-1 split.",
            "evidence_quote": quote,
            "passage_id": "",
        },
        i,
    )


def _index(
    tmp: Path,
    mode: str = "hybrid",
    embeddings: Embeddings | None = None,
    extra: dict[str, list[str]] | None = None,
) -> FilingIndex:
    emb = embeddings or CachedEmbeddings(DeterministicFakeEmbedding(size=32), tmp / "emb", "fake")
    return FilingIndex(
        DOCS.__getitem__, emb, chunk_size=400, chunk_overlap=50, mode=mode, extra_chunks=extra
    )


def _reconciler(tmp: Path, *turns: AIMessage, index: FilingIndex | None = None) -> AgentReconciler:
    model = FakeToolCallingModel(messages=_turns(*turns))
    return AgentReconciler(
        "fake", SPEC, model, StepCache(tmp / "cache"), index or _index(tmp), lambda _t: FILINGS
    )


def test_chunking_drops_numeric_tables_and_keeps_ids() -> None:
    table = "| Revenue | 1,000 | 2,000 |\n" * 30
    docs = chunk_filing(FILLER + "\n\n" + table + "\n\n" + SPLIT, NEW, 400, 50)
    assert all(not is_table_chunk(d.page_content) for d in docs)
    assert any("ten-for-one" in d.page_content for d in docs)
    assert all(str(d.id).startswith(f"{NEW}#") for d in docs)


def test_embeddings_are_cached_on_disk(tmp_path: Path) -> None:
    class Counting(DeterministicFakeEmbedding):
        calls: int = 0

        def embed_query(self, text: str) -> list[float]:
            self.calls += 1
            return super().embed_query(text)

    base = Counting(size=8)
    cached = CachedEmbeddings(base, tmp_path, "m")
    cached.embed_documents(["a", "b", "a"])
    cached.embed_documents(["a", "b"])
    assert base.calls == 2  # each distinct text embedded once, ever


def test_hybrid_search_finds_exact_terms_vector_alone_may_miss(tmp_path: Path) -> None:
    hybrid = _index(tmp_path, "hybrid").search("ten-for-one stock split", NEW, k=2)
    assert "ten-for-one" in hybrid[0].text
    assert all(p.accession_no == NEW for p in hybrid)


def test_index_version_changes_with_retrieval_settings(tmp_path: Path) -> None:
    a = _index(tmp_path, "vector").version
    assert a != _index(tmp_path, "hybrid").version
    assert a != _index(tmp_path, "vector", extra={NEW: ["injected"]}).version


def test_happy_path_uses_tools_and_verifies_the_quote(tmp_path: Path) -> None:
    rec = _reconciler(
        tmp_path,
        _call("fact_history", {}, 1),
        _call("search_filing", {"query": "stock split", "accession_no": NEW}, 2),
        _answer("stock_split", "we executed a ten-for-one stock split"),
    )
    out = rec.investigate(TASK)
    assert out.resolution.cause is Cause.STOCK_SPLIT and out.resolution.verified
    assert [c.name for c in out.tool_calls] == ["fact_history", "search_filing", "AgentAnswer"]
    assert out.tool_calls[1].accession_no == NEW  # searched the filing where it changed
    assert any("ten-for-one" in p.text for p in out.passages)


def test_quote_not_in_retrieved_text_is_downgraded(tmp_path: Path) -> None:
    rec = _reconciler(
        tmp_path,
        _call("search_filing", {"query": "split", "accession_no": NEW}, 1),
        _answer("stock_split", "we executed a five-for-one stock split"),  # invented
    )
    res = rec.reconcile(TASK)
    assert res.cause is Cause.UNKNOWN and not res.verified


def test_answering_without_searching_cannot_be_verified(tmp_path: Path) -> None:
    rec = _reconciler(tmp_path, _answer("stock_split", "we executed a ten-for-one stock split"))
    res = rec.reconcile(TASK)
    assert (
        res.cause is Cause.UNKNOWN
    )  # no retrieval, no evidence: background knowledge isn't enough


def test_no_structured_answer_is_unknown_with_error(tmp_path: Path) -> None:
    rec = _reconciler(tmp_path, AIMessage(content="I think it was a split."))
    res = rec.reconcile(TASK)
    assert res.cause is Cause.UNKNOWN and "no structured answer" in (res.error or "")


def test_unknown_filing_is_reported_to_the_model_not_raised(tmp_path: Path) -> None:
    rec = _reconciler(
        tmp_path,
        _call("search_filing", {"query": "split", "accession_no": "nope"}, 1),
        _call("search_filing", {"query": "split", "accession_no": NEW}, 2),
        _answer("stock_split", "we executed a ten-for-one stock split", 3),
    )
    out = rec.investigate(TASK)
    assert out.resolution.cause is Cause.STOCK_SPLIT  # recovered after the tool's error message


def test_results_are_cached_per_rep(tmp_path: Path) -> None:
    turns = (
        _call("search_filing", {"query": "split", "accession_no": NEW}, 1),
        _answer("stock_split", "we executed a ten-for-one stock split"),
    )
    rec = _reconciler(tmp_path, *turns)
    first = rec.reconcile(TASK)
    again = rec.reconcile(TASK)  # the fake's script is exhausted: this must come from cache
    assert again.cause is first.cause and again.usage is not None and again.usage.cached


def test_question_hides_which_filing_changed() -> None:
    q = question(TASK)
    assert "2,494,000,000" in q and "24,940,000,000" in q
    assert OLD not in q and NEW not in q  # finding the filing is the agent's job (fact_history)


def test_factory_builds_the_configured_strategy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from filings_agent.reconcile_agent.factory import make_reconciler
    from filings_agent.steps.reconcile import LLMReconciler
    from tests.conftest import make_cfg

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    cfg = make_cfg(tmp_path)
    assert isinstance(make_reconciler(cfg, "nova-micro", "pipeline", "v2"), LLMReconciler)
    agent = make_reconciler(cfg, "nova-micro", "agent", "v1", index=_index(tmp_path))
    assert isinstance(agent, AgentReconciler) and agent.name == "reconcile-agent/nova-micro/v1"


def test_changing_the_harness_version_invalidates_cached_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from filings_agent.reconcile_agent import agent as agent_mod

    turns = (
        _call("search_filing", {"query": "split", "accession_no": NEW}, 1),
        _answer("stock_split", "we executed a ten-for-one stock split"),
    )
    _reconciler(tmp_path, *turns).reconcile(TASK)  # populate the cache
    monkeypatch.setattr(agent_mod, "HARNESS", "something-else")
    fresh = _reconciler(tmp_path, *turns).reconcile(TASK)
    assert (
        fresh.usage is not None and not fresh.usage.cached
    )  # a new harness never reuses old results


def test_invalid_structured_answer_is_retried_not_fatal(tmp_path: Path) -> None:
    rec = _reconciler(
        tmp_path,
        _call("search_filing", {"query": "split", "accession_no": NEW}, 1),
        _call("AgentAnswer", {"cause": "a-split-ish-thing", "explanation": "?"}, 2),  # invalid
        _answer("stock_split", "we executed a ten-for-one stock split", 3),
    )
    res = rec.reconcile(TASK)
    assert res.cause is Cause.STOCK_SPLIT and res.verified  # the parse error went back to the model
