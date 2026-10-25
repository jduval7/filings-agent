"""Local extraction scoring: splits, caching, wrong scale, whole-filing baseline."""

from __future__ import annotations

import pytest

from filings_agent.config import AppConfig
from filings_agent.evals import local
from tests.support.eval_data import PERFECT, WRONG_SCALE, patch_chat


def test_load_examples_filters_by_split(data: AppConfig) -> None:
    assert [e.accession_no for e in local.load_examples(data, "dev")] == ["A1"]
    assert [e.accession_no for e in local.load_examples(data, "test")] == ["B1"]


def test_score_model_perfect_then_cached(data: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    chat = patch_chat(monkeypatch, local, [PERFECT])
    (rep,) = local.score_model(data, "nova-micro", "v2", "dev", reps=1)
    assert rep.metrics["f1"] == 1.0 and rep.metrics["abstain_accuracy"] == 1.0
    assert rep.new_calls == 1 and rep.error_kinds == {}
    (again,) = local.score_model(data, "nova-micro", "v2", "dev", reps=1)
    assert again.new_calls == 0 and chat.calls == 1  # cached: free to re-score
    assert again.cost_usd == rep.cost_usd  # ...and reports the original cost


def test_score_model_counts_wrong_scale_and_errors(
    data: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_chat(monkeypatch, local, [WRONG_SCALE, None])
    reps = local.score_model(data, "nova-micro", "v2", "dev", reps=2)
    # the 1000x-off value is now rejected by the scale check, so it's missing, not wrong
    assert reps[0].metrics["precision"] == 1.0 and reps[0].metrics["recall"] == 0.5
    assert reps[1].error_kinds == {"parse": 1} and reps[1].metrics["schema_error_rate"] == 1.0
    summary = local.summarize_reps("nova-micro", "v2", reps)
    assert summary["reps"] == 2 and summary["f1_sd"] > 0
    assert summary["error_kinds"] == {"parse": 1}
    assert "| nova-micro | v2 | 2 |" in local.table([summary])


def test_whole_filing_baseline_scores_against_each_statement(
    data: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = patch_chat(monkeypatch, local, [PERFECT])
    (fe,) = local.load_filing_examples(data, "dev")
    assert fe.task.section_text.startswith("intro") and fe.task.task_id == "A1:whole_filing"
    (rep,) = local.score_whole_filing(data, "nova-micro", "v2", "dev", reps=1)
    assert rep.metrics["f1"] == 1.0 and chat.calls == 1
