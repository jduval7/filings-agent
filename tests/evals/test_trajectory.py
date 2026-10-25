"""Trajectory eval: the path through the graph, not just the answer."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from filings_agent.config import load_config
from filings_agent.evals import trajectory as tr
from filings_agent.graph.build import Deps
from tests.support.graph_doubles import (
    METRICS,
    FakeExtractor,
    FakeReconciler,
    FakeStore,
    RestatingExtractor,
    TwoFilingStore,
)


def test_clean_company_skips_reconcile_and_review_pause() -> None:
    deps = Deps(FakeStore(), FakeExtractor(), METRICS)
    _, path, review = tr.run_with_trajectory(deps, ["AAA"])
    assert not review and tr.dedupe(path) == tr.expected_path(False)
    assert path.count("extract") == 6  # the fan-out is visible in the raw path


def test_restated_company_goes_through_reconcile_and_human_review() -> None:
    deps = Deps(TwoFilingStore(), RestatingExtractor(), METRICS, reconciler=FakeReconciler())
    state, path, review = tr.run_with_trajectory(deps, ["AAA"])
    assert review and tr.dedupe(path) == tr.expected_path(True)
    assert state["decisions"]["AAA:revenue:FY2023"]["action"] == "accept"


def test_e2e_evaluators_score_path_review_and_restatements() -> None:
    evals = {f.__name__: f for f in tr.e2e_evaluators()}
    ref = SimpleNamespace(
        outputs={
            "path": tr.expected_path(True),
            "review": True,
            "material": ["A:revenue:FY2023", "B:cash:FY2024"],
        }
    )
    run = SimpleNamespace(
        outputs={
            "path": tr.expected_path(False),
            "review": False,
            "material": ["A:revenue:FY2023", "C:x:FY2022"],
        }
    )
    assert evals["trajectory"](run, ref).score == 0
    assert evals["review_gate"](run, ref).score == 0
    res = {r.key: r.score for r in evals["restatements"](run, ref)["results"]}
    assert res == {"restatement_recall": 0.5, "restatement_precision": 0.5}


def test_e2e_experiment_runs_the_graph_and_scores_the_path(tmp_path: Path) -> None:
    from filings_agent.evals import trajectory as tj
    from tests.support.graph_doubles import (
        METRICS,
        FakeReconciler,
        RestatingExtractor,
        TwoFilingStore,
    )
    from tests.support.langsmith_doubles import EvaluatingClient, FakeClient

    deps = Deps(TwoFilingStore(), RestatingExtractor(), METRICS, reconciler=FakeReconciler())
    target = tj.make_e2e_target(lambda: deps)
    out = target({"tickers": ["AAA"]})
    assert out["review"] is True and out["material"] == ["AAA:revenue:FY2023"]

    ex = {
        "inputs": {"tickers": ["AAA"]},
        "outputs": {"path": out["path"], "material": out["material"], "review": True},
    }
    client = EvaluatingClient({"filings-agent/e2e-dev": [ex]})
    assert tj.run_e2e_experiment(load_config(), client, lambda: deps) == "exp-123"

    cfg = load_config().model_copy(update={"data_dir": tmp_path})
    (tmp_path / "gold").mkdir()
    (tmp_path / "gold" / "gold.jsonl").write_text("")
    (tmp_path / "gold" / "filings.jsonl").write_text("")
    fake = FakeClient()
    assert tj.upload_e2e_dataset(cfg, fake) == len(tj.SCENARIOS)
    tj.upload_e2e_dataset(cfg, fake)
    assert len(fake.updated) == len(tj.SCENARIOS)  # second upload updates, never duplicates
