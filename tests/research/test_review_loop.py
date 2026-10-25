"""Reviewer-triggered research in the graph: a round, then the card again; bounded rounds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.types import Command

from tests.support.graph_doubles import invoke, review_env
from tests.support.research_doubles import FakeResearcher

REVENUE = "AAA:revenue:FY2023"


def _pending(out: dict[str, Any]) -> dict[str, Any]:
    (card,) = out["__interrupt__"][0].value["pending"]
    return card


def test_research_round_then_decide(tmp_path: Path) -> None:
    from dataclasses import replace

    with review_env(tmp_path) as (graph, deps, _rec, ledger):
        researcher = FakeResearcher()
        deps = replace(deps, researcher=researcher, max_research_rounds=2)
        card = _pending(invoke(graph, {"tickers": ["AAA"]}, deps))
        assert card["research_rounds_left"] == 2

        out = invoke(
            graph, Command(resume={REVENUE: {"action": "research", "note": "check 8-Ks"}}), deps
        )
        card = _pending(out)  # the same case comes back, with findings
        assert researcher.calls == [(REVENUE, "check 8-Ks", 1)]
        assert card["other_causes"][0]["name"] == "error correction"
        assert card["research_rounds_left"] == 1
        assert ledger.counts()["publications"] == 0  # nothing published while researching

        out = invoke(graph, Command(resume={REVENUE: {"action": "escalate"}}), deps)
        assert out["publication"]["newly_published"] is True
        assert len(researcher.calls) == 1  # the finished round didn't run again


def test_research_is_bounded(tmp_path: Path) -> None:
    from dataclasses import replace

    with review_env(tmp_path) as (graph, deps, _rec, _ledger):
        deps = replace(deps, researcher=FakeResearcher(), max_research_rounds=1)
        invoke(graph, {"tickers": ["AAA"]}, deps)
        out = invoke(graph, Command(resume={REVENUE: {"action": "research"}}), deps)
        assert _pending(out)["research_rounds_left"] == 0
        again = invoke(graph, Command(resume={REVENUE: {"action": "research"}}), deps)
        assert "no research rounds" in again["__interrupt__"][0].value["problems"][REVENUE]
        done = invoke(graph, Command(resume={REVENUE: {"action": "escalate"}}), deps)
        assert done["publication"]["newly_published"] is True  # recovers, not locked


def test_without_a_researcher_research_is_refused(tmp_path: Path) -> None:
    with review_env(tmp_path) as (graph, deps, _rec, _ledger):
        card = _pending(invoke(graph, {"tickers": ["AAA"]}, deps))
        assert card["research_rounds_left"] == 0
        out = invoke(graph, Command(resume={REVENUE: {"action": "research"}}), deps)
        assert "no research rounds" in out["__interrupt__"][0].value["problems"][REVENUE]
