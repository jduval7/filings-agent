"""Review and publish: material changes pause for a human, decisions publish exactly once,
the memo runs after review."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from filings_agent.config import ModelSpec
from filings_agent.graph.build import Deps, build_graph
from filings_agent.steps.memo import MemoOutput, MemoSection, MemoWriter
from filings_agent.steps.reconcile import Cause
from filings_agent.storage.ledger import Ledger
from tests.support.fakes import ScriptedChat
from tests.support.graph_doubles import (
    METRICS,
    OLD,
    FakeReconciler,
    RestatingExtractor,
    TwoFilingStore,
    invoke,
    review_env,
)


def test_material_change_pauses_for_review_and_immaterial_is_auto(tmp_path: Path) -> None:
    with review_env(tmp_path) as (graph, deps, rec, ledger):
        out = invoke(graph, {"tickers": ["AAA"]}, deps)
        (intr,) = out["__interrupt__"]
        pending = {p["id"] for p in intr.value["pending"]}
        assert pending == {"AAA:revenue:FY2023"}  # the 0.1% cash revision didn't need a human
        assert out["resolutions"]["AAA:cash:FY2023"].cause is Cause.IMMATERIAL_REVISION
        assert rec.calls == 1
        assert ledger.counts()["publications"] == 0  # nothing published before the decision


def test_decision_publishes_exactly_once(tmp_path: Path) -> None:
    with review_env(tmp_path) as (graph, deps, _rec, ledger):
        invoke(graph, {"tickers": ["AAA"]}, deps)
        decision = {"AAA:revenue:FY2023": {"action": "use_original", "note": "compare as filed"}}
        out = invoke(graph, Command(resume=decision), deps)
        assert out["publication"]["newly_published"] is True
        counts = ledger.counts()
        assert counts["publications"] == 1 and counts["resolutions"] == 2

        # Replaying the request on the same thread must not publish again.
        again = invoke(graph, {"tickers": ["AAA"]}, deps)
        assert ledger.counts()["publications"] == 1
        if "publication" in again and "__interrupt__" not in again:
            assert again["publication"]["newly_published"] is False

    conn = sqlite3.connect(str(tmp_path / "ledger.sqlite"))
    row = conn.execute(
        "SELECT use_value, use_accession, decision FROM resolutions WHERE discrepancy_id = ?",
        ("AAA:revenue:FY2023",),
    ).fetchone()
    assert row == ("100", OLD, "use_original")  # the human's choice is what got stored


def test_memo_node_runs_after_review_and_is_published(tmp_path: Path) -> None:
    # No invented figures: the checker would (rightly) flag "$60M" since the fake revenue is $60.
    memo = MemoOutput(title="AAA", summary="Revenue for FY2023 was restated in the latest report.",
                      sections=[MemoSection(heading="AAA", body="Restated.")],
                      caveats="Revenue was restated.")  # fmt: skip
    writer = MemoWriter("fake", ModelSpec(provider="bedrock", id="f", price=(1, 1)),
                        ScriptedChat(script=[memo]))  # fmt: skip
    ledger = Ledger(tmp_path / "l.sqlite")
    deps = Deps(TwoFilingStore(), RestatingExtractor(), METRICS, reconciler=FakeReconciler(),
                ledger=ledger, memo_writer=writer)  # fmt: skip
    graph = build_graph(InMemorySaver())
    cfg: RunnableConfig = {"configurable": {"thread_id": "memo"}}
    graph.invoke({"tickers": ["AAA"]}, cfg, context=deps)  # pyright: ignore[reportUnknownMemberType]
    resume = Command(resume={"AAA:revenue:FY2023": {"action": "accept"}})
    out = cast(dict[str, Any], graph.invoke(resume, cfg, context=deps))  # pyright: ignore[reportUnknownMemberType]
    assert out["memo"]["verified"] is True and out["memo"]["attempts"] == 1
    assert out["memo_usage"].cost_usd > 0
    assert ledger.counts()["memos"] == 1
