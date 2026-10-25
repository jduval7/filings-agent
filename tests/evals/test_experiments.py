"""LangSmith datasets and experiments: idempotent uploads, targets, evaluators, sweeps."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from filings_agent.config import AppConfig
from filings_agent.evals import extract_experiments as ee
from filings_agent.evals import reconcile_experiments as re_
from filings_agent.evals import reconcile_local as rl
from filings_agent.steps.reconcile import Cause, ReconcileOutput, ReconcileTask
from tests.support.eval_data import (
    EVIDENCE,
    PERFECT,
    SECTION,
    D,
    extract_example,
    patch_chat,
    score_map,
)
from tests.support.fakes import ScriptedChat
from tests.support.langsmith_doubles import EvaluatingClient, FakeClient


def test_upload_extract_datasets_is_idempotent(data: AppConfig) -> None:
    client = FakeClient()
    counts = ee.upload_extract_datasets(data, cast(Any, client))
    assert counts == {"filings-agent/extract-dev": 1, "filings-agent/extract-test": 1}
    ee.upload_extract_datasets(data, cast(Any, client))
    assert len(client.created) == 2 and len(client.updated) == 2  # second run updates in place
    (ex,) = client.datasets["filings-agent/extract-dev"].values()
    assert ex["outputs"]["expect_absent"] == ["operating_income"]
    assert ex["metadata"] == {"ticker": "GE", "statement": "income_statement", "split": "dev"}


def _run(outputs: dict[str, Any]) -> Any:
    return SimpleNamespace(outputs=outputs)


def test_extract_target_and_evaluators(data: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    from filings_agent.llm.cache import StepCache
    from filings_agent.steps.extract import LLMStatementExtractor

    chat = ScriptedChat(script=[PERFECT])
    extractor = LLMStatementExtractor("m", data.model("nova-micro"), chat,
                                      StepCache(data.data_dir / "c"), "v2")  # fmt: skip
    ex = extract_example()
    inputs = {"task_id": ex.example_id, "ticker": "GE", "accession_no": "A1",
              "fiscal_year_end": "2024-12-31", "statement": "income_statement",
              "section_text": SECTION, "metrics": ["revenue", "operating_income"]}  # fmt: skip
    out = ee.make_target(extractor, rep=0)(inputs)
    assert {f["fiscal_year"] for f in out["facts"]} == {2023, 2024} and out["error"] is None
    example = _run({"expected": [e.model_dump(mode="json") for e in ex.expected],
                    "expect_absent": ["operating_income"]})  # fmt: skip
    results = score_map(ee.extraction_evaluator(_run(out), example))
    assert results["f1"] == 1.0 and results["hallucinated"] == 0 and results["cost_usd"] > 0
    summary = score_map(ee.summary_evaluator([_run(out)], [example]))
    assert summary["micro_f1"] == 1.0 and "latency_p95_s" in summary
    empty = score_map(ee.extraction_evaluator(_run({}), example))
    assert empty["schema_error"] == 1  # a crashed run counts as a failure, not a pass


def test_sweep_table_sorts_by_quality_then_cost(tmp_path: Path) -> None:
    rows = [
        {"model": "cheap", "prompt_version": "v2", "rep": 0, "f1": 0.9, "precision": 0.9,
         "recall": 0.9, "abstain_accuracy": 1.0, "schema_error_rate": 0.0, "cost_usd": 0.01,
         "latency_p50_s": 1.0, "latency_p95_s": 2.0},
        {"model": "best", "prompt_version": "v2", "rep": 0, "f1": 1.0, "precision": 1.0,
         "recall": 1.0, "abstain_accuracy": 1.0, "schema_error_rate": 0.0, "cost_usd": 0.5,
         "latency_p50_s": None, "latency_p95_s": None},
    ]  # fmt: skip
    table = ee.write_sweep_table(rows, tmp_path / "t.md")
    lines = table.splitlines()
    assert lines[2].startswith("| best") and lines[3].startswith("| cheap")
    assert (tmp_path / "t.md").read_text() == table


def test_reconcile_upload_target_and_evaluators(
    cfg: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from filings_agent.llm.cache import StepCache
    from filings_agent.steps.reconcile import LLMReconciler

    tasks = [rl.LabeledTask(ReconcileTask(D.id, D, EVIDENCE), Cause.STOCK_SPLIT)]

    def fake_load_tasks(_cfg: AppConfig, split: str) -> list[rl.LabeledTask]:
        return tasks if split == "dev" else []

    monkeypatch.setattr(re_, "load_tasks", fake_load_tasks)
    client = FakeClient()
    assert re_.upload_reconcile_datasets(cfg, cast(Any, client)) == {
        "filings-agent/reconcile-dev": 1, "filings-agent/reconcile-test": 0,
    }  # fmt: skip
    (ex,) = client.datasets["filings-agent/reconcile-dev"].values()
    assert ex["outputs"]["cause"] == "stock_split" and ex["metadata"]["source"] == "label"
    assert ex["outputs"]["change_filings"] == ["N2"]  # agent evaluators need these

    good = ReconcileOutput(cause=Cause.STOCK_SPLIT, explanation="x",
                           evidence_quote="we executed a ten-for-one stock split")  # fmt: skip
    rec = LLMReconciler("m", cfg.model("nova-micro"), ScriptedChat(script=[good]),
                        StepCache(cfg.data_dir / "c"), "v2")  # fmt: skip
    out = re_.make_target(rec, rep=0)(ex["inputs"])
    assert out["cause"] == "stock_split" and out["verified"] is True
    ref = _run(ex["outputs"])
    scores = score_map(re_.reconcile_evaluator(_run(out), ref))
    assert scores["cause_correct"] == 1 and scores["unknown"] == 0
    summ = score_map(re_.reconcile_summary([_run(out)], [ref]))
    assert summ["cause_accuracy"] == 1.0


def test_extract_sweep_runs_reps_records_metadata_and_writes_files(
    cfg: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_chat(monkeypatch, ee, [PERFECT])
    ex = extract_example()
    inputs = {
        "task_id": ex.example_id,
        "ticker": "GE",
        "accession_no": "A1",
        "fiscal_year_end": "2024-12-31",
        "statement": "income_statement",
        "section_text": ex.section_text,
        "metrics": ["revenue", "operating_income"],
    }
    outputs = {
        "expected": [e.model_dump(mode="json") for e in ex.expected],
        "expect_absent": ["operating_income"],
    }
    dataset = [{"inputs": inputs, "outputs": outputs}]
    client = EvaluatingClient({"filings-agent/extract-dev": dataset})
    rows = ee.run_extract_sweep(cfg, cast(Any, client), ["nova-micro"], "dev", "v2", 2, 4)
    assert [r["rep"] for r in rows] == [0, 1] and all(r["f1"] == 1.0 for r in rows)
    meta = client.calls[0]["metadata"]
    assert meta["model"] == "nova-micro" and meta["prompt_version"] == "v2" and meta["rep"] == 0
    assert client.calls[0]["experiment_prefix"] == "extract-nova-micro-v2-r0"
    (written,) = (cfg.data_dir / "sweeps").glob("extract-*.jsonl")
    assert len(written.read_text().splitlines()) == 2


def test_reconcile_sweep(cfg: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    good = ReconcileOutput(cause=Cause.STOCK_SPLIT, explanation="x",
                           evidence_quote="we executed a ten-for-one stock split")  # fmt: skip
    from filings_agent.reconcile_agent import factory

    patch_chat(monkeypatch, factory, [good])  # reconcilers are built by the factory
    example = {"inputs": {"discrepancy": D.model_dump(mode="json"), "evidence": list(EVIDENCE)},
               "outputs": {"cause": "stock_split"}}  # fmt: skip
    client = EvaluatingClient({"filings-agent/reconcile-test": [example]})
    (row,) = re_.run_reconcile_sweep(cfg, cast(Any, client), ["nova-micro"], "test", "v2", 1, 2)
    assert row["cause_accuracy"] == 1.0 and row["split"] == "test" and row["n"] == 1
    assert row["experiment"] == "exp-123"


def test_corrections_are_uploaded_with_reviewer_source(
    cfg: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:

    corr = rl.LabeledTask(ReconcileTask(f"{D.id}#corr", D, EVIDENCE), Cause.STOCK_SPLIT)

    def fake_load(_cfg: AppConfig, split: str) -> list[rl.LabeledTask]:
        return [corr] if split == "dev" else []

    monkeypatch.setattr(re_, "load_tasks", fake_load)
    client = FakeClient()
    re_.upload_reconcile_datasets(cfg, cast(Any, client))
    (ex,) = client.datasets["filings-agent/reconcile-dev"].values()
    assert ex["metadata"]["source"] == "reviewer"
