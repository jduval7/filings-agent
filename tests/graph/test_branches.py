# LangGraph ships without type stubs (graph.invoke is partially untyped).
# pyright: reportUnknownMemberType=false

"""Graph branches not covered elsewhere: failures inside wave 2, incomplete reviews, no LLM."""

from __future__ import annotations

from typing import Any, cast

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from filings_agent.config import load_config
from filings_agent.graph.build import Deps, build_graph
from filings_agent.steps.reconcile import Cause, ReconcileTask, Resolution
from tests.support.graph_doubles import METRICS, RestatingExtractor, TwoFilingStore


class BrokenReconciler:
    name = "broken"

    def reconcile(self, task: ReconcileTask, rep: int = 0) -> Resolution:
        raise ValueError("parser bug")


CFG: RunnableConfig = {"configurable": {"thread_id": "g"}}


def _invoke(deps: Deps, inp: Any) -> dict[str, Any]:
    graph = build_graph(InMemorySaver())
    return cast(dict[str, Any], graph.invoke(inp, CFG, context=deps))


def test_without_a_reconciler_everything_is_auto_resolved_and_nothing_pauses() -> None:
    deps = Deps(TwoFilingStore(), RestatingExtractor(), METRICS, reconciler=None)
    out = _invoke(deps, {"tickers": ["AAA"]})
    assert "__interrupt__" not in out
    assert {r.cause for r in out["resolutions"].values()} == {Cause.IMMATERIAL_REVISION}


def test_a_crashing_reconciler_becomes_unknown_needing_review() -> None:
    deps = Deps(TwoFilingStore(), RestatingExtractor(), METRICS, reconciler=BrokenReconciler())
    out = _invoke(deps, {"tickers": ["AAA"]})
    res = out["resolutions"]["AAA:revenue:FY2023"]
    assert res.cause is Cause.UNKNOWN and res.needs_review and "parser bug" in (res.error or "")
    assert out["__interrupt__"]  # still goes to a human


def test_incomplete_review_is_rejected_not_treated_as_approval() -> None:
    from tests.support.graph_doubles import FakeReconciler

    graph = build_graph(InMemorySaver())
    deps = Deps(TwoFilingStore(), RestatingExtractor(), METRICS, reconciler=FakeReconciler())
    graph.invoke({"tickers": ["AAA"]}, CFG, context=deps)
    out = graph.invoke(Command(resume={"some-other-id": {"action": "accept"}}), CFG, context=deps)
    problems = out["__interrupt__"][0].value["problems"]  # asked again, with the reasons
    assert problems == {
        "AAA:revenue:FY2023": "no decision",
        "some-other-id": "not waiting for review",
    }
    assert "publication" not in out

    # ...and the run recovers: a valid answer goes through (it used to stay locked)
    bad = graph.invoke(
        Command(resume={"AAA:revenue:FY2023": {"action": "approve"}}), CFG, context=deps
    )
    assert "unknown action" in bad["__interrupt__"][0].value["problems"]["AAA:revenue:FY2023"]
    good = graph.invoke(
        Command(resume={"AAA:revenue:FY2023": {"action": "escalate"}}), CFG, context=deps
    )
    assert (
        "__interrupt__" not in good
        and good["decisions"]["AAA:revenue:FY2023"]["action"] == "escalate"
    )


def test_an_invalid_review_goes_back_through_a_conditional_edge_not_a_loop() -> None:
    # The LangGraph docs: no `while True` validation loops around interrupt(); route back with a
    # conditional edge. So a refused answer ends the review step, and a new review step (one
    # interrupt, showing the problems) is what's waiting.
    from tests.support.graph_doubles import FakeReconciler

    graph = build_graph(InMemorySaver())
    deps = Deps(TwoFilingStore(), RestatingExtractor(), METRICS, reconciler=FakeReconciler())
    graph.invoke({"tickers": ["AAA"]}, CFG, context=deps)
    first = graph.get_state(CFG)
    graph.invoke(Command(resume={"AAA:revenue:FY2023": {"action": "approve"}}), CFG, context=deps)
    again = graph.get_state(CFG)
    assert again.next == ("review",)
    assert first.metadata and again.metadata
    assert again.metadata.get("step", 0) > first.metadata.get("step", 0)  # a new step, not a loop
    assert "unknown action" in again.values["review_problems"]["AAA:revenue:FY2023"]
    (task,) = again.tasks
    assert len(task.interrupts) == 1
    assert "unknown action" in task.interrupts[0].value["problems"]["AAA:revenue:FY2023"]

    graph.invoke(Command(resume={"AAA:revenue:FY2023": {"action": "escalate"}}), CFG, context=deps)
    assert graph.get_state(CFG).values["review_problems"] == {}  # a valid answer clears them


def test_no_ledger_means_publish_is_a_no_op() -> None:
    deps = Deps(TwoFilingStore(), RestatingExtractor(), METRICS)  # no ledger, no memo writer
    out = _invoke(deps, {"tickers": ["AAA"]})
    assert "publication" not in out and "memo" not in out


def test_studio_mode_uses_default_deps_when_context_is_empty() -> None:
    # LangGraph Studio / `langgraph dev` passes JSON context, never our Deps object.
    deps = Deps(TwoFilingStore(), RestatingExtractor(), METRICS)
    graph = build_graph(InMemorySaver(), default_deps=lambda: deps, typed_context=False)
    out = cast(dict[str, Any], graph.invoke({"tickers": ["AAA"]}, CFG))
    assert len(out["facts"]) == 6


def test_studio_entry_point_compiles() -> None:
    from filings_agent.graph import studio

    assert set(studio.graph.get_graph().nodes) >= {"plan", "extract", "reconcile", "review", "memo"}


def test_studio_builds_production_deps_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Studio passes JSON, not Python objects: the graph falls back to deps built from config."""
    from filings_agent.graph import runner, studio

    built: list[tuple[str, str]] = []

    def fake_make_deps(cfg: Any, model: str, prompt: str) -> str:
        built.append((model, prompt))
        return "deps"

    monkeypatch.setattr(runner, "make_deps", fake_make_deps)
    studio._deps.cache_clear()  # pyright: ignore[reportPrivateUsage]
    assert studio._deps() == "deps"  # pyright: ignore[reportPrivateUsage]
    cfg = load_config()
    assert built == [(cfg.roles.extractor, cfg.prompts.extract)]
    studio._deps.cache_clear()  # pyright: ignore[reportPrivateUsage]


def test_reconcile_without_a_reconciler_fails_loudly() -> None:
    from types import SimpleNamespace

    from filings_agent.graph.build import reconcile
    from filings_agent.steps.reconcile import ReconcileTask
    from tests.support.cases import GE_EVIDENCE, GE_NET_INCOME

    deps = Deps(TwoFilingStore(), RestatingExtractor(), METRICS, reconciler=None)
    task = ReconcileTask(GE_NET_INCOME.id, GE_NET_INCOME, GE_EVIDENCE)
    with pytest.raises(RuntimeError, match="without a reconciler"):
        reconcile({"task": task}, cast(Any, SimpleNamespace(context=deps)))
