"""The research eval: per-case scoring and hard-case selection."""

from __future__ import annotations

from typing import Any, cast

import pytest

from filings_agent.steps.reconcile import ReconcileTask
from tests.support.research_doubles import vf


def test_research_eval_scores_one_case() -> None:
    from dataclasses import asdict

    from filings_agent.evals.research_experiments import score

    out = {
        "findings": [
            asdict(vf("accounting_change")),
            asdict(vf("discontinued_operations")),
            asdict(vf("accounting_change", "contradicts")),
        ],
        "dropped": 1,
        "card_fits": {"accounting_change": "possible", "discontinued_operations": "ruled_out"},
        "cost_usd": 0.9,
    }
    s = score(out, "accounting_change")
    assert s["true_cause_supported"] == 1.0 and s["true_cause_contradicted"] == 1.0
    assert s["wrong_causes_supported"] == 1.0 and s["true_cause_open_on_card"] == 1.0
    assert s["discarded_rate"] == 0.25


def test_hard_cases_skip_splits_and_duplicates_but_keep_ge_net_income(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import date
    from decimal import Decimal

    from filings_agent.domain import Discrepancy, Metric
    from filings_agent.evals import research_experiments as rx
    from filings_agent.evals.reconcile_local import LabeledTask
    from filings_agent.steps.reconcile import Cause

    def task(ticker: str, metric: Metric, fy: int, cause: Cause, suffix: str = "") -> LabeledTask:
        d = Discrepancy(
            ticker=ticker, metric=metric, fiscal_year=fy, period_end=date(fy, 12, 31),
            values=(("a", Decimal(1)), ("b", Decimal(2))),
        )  # fmt: skip
        return LabeledTask(ReconcileTask(d.id + suffix, d, ()), cause)

    tasks = [
        task("GE", Metric.NET_INCOME, 2021, Cause.ACCOUNTING_CHANGE),
        task("GE", Metric.NET_INCOME, 2022, Cause.ACCOUNTING_CHANGE),
        task("GE", Metric.NET_INCOME, 2022, Cause.ACCOUNTING_CHANGE, "#corr"),
        task("NVDA", Metric.DILUTED_SHARES, 2023, Cause.STOCK_SPLIT),
    ]

    def fake_load(cfg: object, split: str) -> list[LabeledTask]:
        return tasks if split == "dev" else []

    monkeypatch.setattr(rx, "load_tasks", fake_load)
    picked = [t.task.discrepancy_id for t in rx.hard_cases(cast(Any, None))]
    assert picked == ["GE:net_income:FY2022"]


def _cases() -> list[Any]:
    from filings_agent.evals.reconcile_local import LabeledTask
    from filings_agent.steps.reconcile import Cause
    from tests.support.cases import GE_EVIDENCE, GE_NET_INCOME

    return [
        LabeledTask(
            ReconcileTask(GE_NET_INCOME.id, GE_NET_INCOME, GE_EVIDENCE), Cause.ACCOUNTING_CHANGE
        )
    ]


class OneFindingResearcher:
    name = "research/m/fanout-test"

    def research(self, task: ReconcileTask, hint: str, round_: int) -> Any:
        from filings_agent.research.findings import ResearchResult
        from filings_agent.steps.extract import Usage

        usage = Usage("m", 10, 5, 0.5, 2.0, False)
        return ResearchResult(
            task.discrepancy_id, round_, hint, (vf("accounting_change"),), 0, "", "", usage
        )


def test_dataset_upload_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    from filings_agent.evals import research_experiments as rx
    from tests.support.langsmith_doubles import FakeClient

    def cases(cfg: object, which: str = "hard") -> list[Any]:
        return _cases()

    monkeypatch.setattr(rx, "hard_cases", cases)
    client = FakeClient()
    name, n = rx.upload_research_dataset(cast(Any, None), cast(Any, client))
    assert (name, n) == ("filings-agent/research-hard", 1)
    rx.upload_research_dataset(cast(Any, None), cast(Any, client))
    assert len(client.created) == 1  # second upload adds nothing


def test_experiment_runs_scores_and_summarizes(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from filings_agent.config import load_config
    from filings_agent.evals import research_experiments as rx
    from filings_agent.graph import runner
    from tests.support.langsmith_doubles import EvaluatingClient

    (case,) = _cases()
    example = {
        "inputs": {
            "discrepancy": case.task.discrepancy.model_dump(mode="json"),
            "evidence": list(case.task.evidence),
        },
        "outputs": {"cause": "accounting_change"},
    }
    client = EvaluatingClient({"filings-agent/research-hard": [example]})

    def uploaded(cfg: object, c: object, which: str = "hard") -> tuple[str, int]:
        return "filings-agent/research-hard", 1

    def researcher(cfg: object) -> OneFindingResearcher:
        return OneFindingResearcher()

    monkeypatch.setattr(rx, "upload_research_dataset", uploaded)
    monkeypatch.setattr(runner, "make_researcher", researcher)
    out = rx.run_research_experiment(load_config(), cast(Any, client))
    (scores,) = out["per_case"].values()
    assert scores["true_cause_supported"] == 1.0 and scores["cost_usd"] == 0.5
    assert client.calls[0]["metadata"]["harness"] == "research/m/fanout-test"

    run = SimpleNamespace(outputs=rx.make_target(OneFindingResearcher())(example["inputs"]))
    ex = SimpleNamespace(outputs=example["outputs"])
    per_run = {
        r.key: r.score
        for r in rx.research_evaluator(cast(Any, run), cast(Any, ex)).get("results", [])
    }
    assert per_run["true_cause_open_on_card"] == 1.0
    summary = rx.research_summary([cast(Any, run)], [cast(Any, ex)]).get("results", [])
    assert {r.key for r in summary} >= {"true_cause_supported_rate", "total_cost_usd"}
