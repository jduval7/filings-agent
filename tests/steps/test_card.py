"""Review card: every candidate cause with its evidence and fit; never the agent's favorite."""

from __future__ import annotations

import sqlite3
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from langgraph.types import Command

from filings_agent.domain import Discrepancy, Metric
from filings_agent.steps.card import build_card, render_card, sentences_for
from filings_agent.steps.reconcile import Cause
from tests.support.cases import GE_EVIDENCE, GE_NET_INCOME
from tests.support.graph_doubles import FakeReconciler, invoke, review_env
from tests.support.research_doubles import result, vf


def fits(card: dict[str, object]) -> dict[str, str]:
    return {c["cause"]: c["fit"] for c in card["candidates"]}  # type: ignore[index,union-attr]


def test_ge_card_shows_the_spin_off_but_rules_it_out() -> None:
    card = build_card(GE_NET_INCOME, GE_EVIDENCE)
    assert fits(card) == {
        "stock_split": "ruled_out",
        "discontinued_operations": "ruled_out",  # mentioned in the filing, and still ruled out
        "accounting_change": "possible",
        "reclassification": "no_evidence",
    }
    (disc,) = [c for c in card["candidates"] if c["cause"] == "discontinued_operations"]
    assert "net income" in disc["why"] and disc["evidence"]  # the lesson, next to the evidence


def test_card_has_no_favorite() -> None:
    """Fixed order and no agent answer: nothing on the card says which cause the agent chose."""
    card = build_card(GE_NET_INCOME, GE_EVIDENCE)
    assert [c["cause"] for c in card["candidates"]] == [
        "stock_split",
        "discontinued_operations",
        "accounting_change",
        "reclassification",
    ]
    assert not {"cause", "explanation", "verified", "lane"} & set(card)


def test_split_card_fits_when_the_ratio_and_the_text_agree() -> None:
    d = Discrepancy(
        ticker="NVDA",
        metric=Metric.DILUTED_SHARES,
        fiscal_year=2023,
        period_end=date(2023, 1, 29),
        values=(("a", Decimal(2_507)), ("b", Decimal(25_070))),
    )
    card = build_card(d, ("In May 2024, we announced a ten\\-for\\-one stock split.",))
    assert fits(card)["stock_split"] == "fits"


def test_quotes_are_whole_clean_sentences() -> None:
    (asu,) = sentences_for(Cause.ACCOUNTING_CHANGE, GE_EVIDENCE)
    assert asu.startswith("On January 1, 2023, we adopted ASU No. 2018-12, Financial Services")
    assert "\\" not in asu and "#" not in asu and "Genworth" not in asu


def test_render_card_is_readable_markdown() -> None:
    text = render_card(build_card(GE_NET_INCOME, GE_EVIDENCE))
    assert "net_income FY2022: 225 -> 339 (+50.7%)" in text
    assert "| discontinued_operations | ✗ ruled out |" in text


REVENUE = "AAA:revenue:FY2023"


def _published(tmp: Path) -> tuple[str, str, str]:
    conn = sqlite3.connect(str(tmp / "ledger.sqlite"))
    row = conn.execute(
        "SELECT cause, decision, explanation FROM resolutions WHERE discrepancy_id = ?", (REVENUE,)
    ).fetchone()
    return row


def test_review_payload_is_cards_without_the_agents_answer(tmp_path: Path) -> None:
    with review_env(tmp_path) as (graph, deps, _rec, _ledger):
        out = invoke(graph, {"tickers": ["AAA"]}, deps)
        (card,) = out["__interrupt__"][0].value["pending"]
        assert card["id"] == REVENUE and len(card["candidates"]) == 4
        assert not {"cause", "explanation", "evidence", "verified"} & set(card)


def test_escalate_publishes_the_number_flagged_as_unresolved(tmp_path: Path) -> None:
    with review_env(tmp_path) as (graph, deps, _rec, _ledger):
        invoke(graph, {"tickers": ["AAA"]}, deps)
        decision = {REVENUE: {"action": "escalate", "note": "none of these fit"}}
        out = invoke(graph, Command(resume=decision), deps)
        assert out["publication"]["newly_published"] is True
    cause, action, explanation = _published(tmp_path)
    assert (cause, action) == (Cause.UNKNOWN.value, "escalate")
    assert "none of these fit" in explanation


def test_the_reviewers_pick_is_published_with_its_own_evidence(tmp_path: Path) -> None:
    assert FakeReconciler  # the fake agent answers discontinued_operations
    with review_env(tmp_path) as (graph, deps, _rec, ledger):
        invoke(graph, {"tickers": ["AAA"]}, deps)
        decision = {REVENUE: {"action": "accept", "correct_cause": "reclassification"}}
        invoke(graph, Command(resume=decision), deps)
        assert ledger.counts()["corrections"] == 1  # disagreement with the agent: a new test
    cause, _action, explanation = _published(tmp_path)
    assert cause == "reclassification" and explanation.startswith("Cause picked by reviewer")


def test_card_eval_flags_a_card_that_hides_the_true_cause(monkeypatch: Any) -> None:
    from filings_agent.evals import reconcile_local
    from filings_agent.evals.reconcile_local import LabeledTask, card_eval
    from filings_agent.steps.reconcile import ReconcileTask

    task = ReconcileTask(GE_NET_INCOME.id, GE_NET_INCOME, GE_EVIDENCE)
    cases = [LabeledTask(task, Cause.ACCOUNTING_CHANGE)]

    def fake_load(cfg: object, split: str) -> list[LabeledTask]:
        return cases

    monkeypatch.setattr(reconcile_local, "load_tasks", fake_load)
    assert card_eval(cast(Any, None), "dev")["true_cause_hidden"] == []

    cases = [LabeledTask(task, Cause.DISCONTINUED_OPERATIONS)]  # a label the card rules out
    assert card_eval(cast(Any, None), "dev")["true_cause_hidden"] == [GE_NET_INCOME.id]


def test_research_is_shown_but_never_overrides_a_rule() -> None:
    card = build_card(
        GE_NET_INCOME,
        GE_EVIDENCE,
        [result(vf("discontinued_operations"), vf("reclassification"), vf("other", other="error"))],
        research_rounds_left=1,
    )
    fit = {c["cause"]: c["fit"] for c in card["candidates"]}
    assert fit["discontinued_operations"] == "ruled_out"  # research can't rescue a ruled-out cause
    assert fit["reclassification"] == "possible"  # was "no evidence"; research found support
    assert card["other_causes"][0]["name"] == "error"
    assert card["research_rounds"][0]["discarded_unverified"] == 1
    text = render_card(card)
    assert "other: error" in text and "1 rounds left" in text


def test_card_caps_findings_per_cause_and_keeps_both_sides() -> None:
    many = [vf("accounting_change") for _ in range(5)] + [vf("accounting_change", "contradicts")]
    card = build_card(GE_NET_INCOME, GE_EVIDENCE, [result(*many)])
    (acc,) = [c for c in card["candidates"] if c["cause"] == "accounting_change"]
    assert [f["stance"] for f in acc["research"]] == ["supports", "contradicts"]
    assert acc["research_not_shown"] == 4
    assert "unverified_reading" in acc["research"][0] and "note" not in acc["research"][0]
