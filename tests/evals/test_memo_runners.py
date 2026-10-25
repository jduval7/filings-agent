"""Memo eval runners with a fake writer and judge: local scoring, pairwise comparison, LangSmith
datasets, sweeps and the comparative experiment."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from filings_agent.config import AppConfig
from filings_agent.evals import memo_eval as me
from filings_agent.steps.extract import Usage
from filings_agent.steps.judge import JudgeResult, PairwiseResult, Rubric
from filings_agent.steps.memo import Memo, Row
from tests.support.langsmith_doubles import EvaluatingClient, FakeClient

USAGE = Usage("m", 10, 5, 0.01, 1.0, False)


class FakeWriter:
    def __init__(self, name: str) -> None:
        self.model_name = name

    def write(self, rows: list[Row], changes: list[str]) -> Memo:
        return Memo(f"memo by {self.model_name}", True, [], 1, USAGE)


class FakeJudge:
    model_name = "judge-x"

    def grade(self, rubric: Rubric, material: str, output: str) -> JudgeResult:
        names = rubric.names()
        return JudgeResult(
            rubric.name, dict.fromkeys(names, True), dict.fromkeys(names, "ok"), USAGE
        )

    def compare(self, material: str, a: str, b: str) -> PairwiseResult:
        return PairwiseResult("A", True, ("a is clearer", "a is clearer"), USAGE)


@pytest.fixture
def memo_cfg(cfg: AppConfig, monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    example = me.MemoExample(example_id="GE+NVDA", split="dev", group=["GE", "NVDA"], rows=[],
                             changes=["GE revenue FY2021 recast"])  # fmt: skip
    me.write_memo_dataset(cfg.data_dir / "datasets" / "memo.jsonl", [example])

    def writer(cfg: AppConfig, model: str) -> FakeWriter:
        return FakeWriter(model)

    def judge(cfg: AppConfig, model: str | None = None) -> FakeJudge:
        return FakeJudge()

    monkeypatch.setattr(me, "make_writer", writer)
    monkeypatch.setattr(me, "make_judge", judge)
    return cfg


def test_score_memo_reports_verification_and_every_criterion(memo_cfg: AppConfig) -> None:
    row = me.score_memo(memo_cfg, "claude-haiku-4.5", "dev", judge_model="judge-x")
    assert row["n"] == 1 and row["numbers_verified"] == 1.0 and row["judge_errors"] == 0
    assert all(row[f"judge.{c}"] == 1.0 for c in me.MEMO_RUBRIC.names())


def test_a_judge_error_is_no_verdict_not_a_failed_memo(
    memo_cfg: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review finding: an errored judge scored every criterion as failed."""
    example = me.MemoExample(example_id="KO", split="dev", group=["KO"], rows=[], changes=["KO x"])
    path = memo_cfg.data_dir / "datasets" / "memo.jsonl"
    me.write_memo_dataset(path, [*me.load_memo_examples(memo_cfg, "dev"), example])

    class Flaky(FakeJudge):
        def grade(self, rubric: Rubric, material: str, output: str) -> JudgeResult:
            if "KO" in material:
                return JudgeResult(rubric.name, {}, {}, USAGE, error="no structured output")
            return super().grade(rubric, material, output)

    def judge(cfg: AppConfig, model: str | None = None) -> Flaky:
        return Flaky()

    monkeypatch.setattr(me, "make_judge", judge)
    row = me.score_memo(memo_cfg, "claude-haiku-4.5", "dev", judge_model="judge-x")
    assert row["n"] == 2 and row["judge_errors"] == 1
    assert all(row[f"judge.{c}"] == 1.0 for c in me.MEMO_RUBRIC.names())  # not 0.5


def test_compare_memos_tallies_and_refuses_a_judging_writer(memo_cfg: AppConfig) -> None:
    res = me.compare_memos(memo_cfg, "a", "b", "dev")
    assert (res["a_wins"], res["position_bias_rate"]) == (1, 0.0)
    with pytest.raises(ValueError, match="own memos"):
        me.compare_memos(memo_cfg, "judge-x", "b", "dev")


def test_memo_datasets_upload_idempotently(memo_cfg: AppConfig) -> None:
    client = FakeClient()
    counts = me.upload_memo_datasets(memo_cfg, client)
    assert counts == {"filings-agent/memo-dev": 1, "filings-agent/memo-test": 0}
    me.upload_memo_datasets(memo_cfg, client)
    assert len(client.created) == 1 and len(client.updated) == 1


def test_sweep_target_and_evaluators(memo_cfg: AppConfig) -> None:
    ex = {"inputs": {"group": ["GE"], "rows": [], "changes": []}, "outputs": {}}
    client = EvaluatingClient({"filings-agent/memo-dev": [ex]})
    (row,) = me.run_memo_sweep(memo_cfg, client, ["claude-haiku-4.5"], "dev")
    assert row["experiment"] == "exp-123"
    with pytest.raises(ValueError, match="is the judge"):
        me.run_memo_sweep(memo_cfg, client, ["judge-x"], "dev")

    out = me.make_memo_target(cast(Any, FakeWriter("w")))(ex["inputs"])
    numbers, judged = me.make_memo_evaluators(cast(Any, FakeJudge()))
    run = SimpleNamespace(outputs=out)
    assert numbers(run, None).score == 1
    assert all(r.score == 1 for r in judged(run, None)["results"])


def test_pairwise_experiment_scores_both_runs(
    memo_cfg: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    import langsmith.evaluation

    seen: dict[str, Any] = {}

    def fake_comparative(experiments: Any, *, evaluators: list[Any], **kw: Any) -> Any:
        a = SimpleNamespace(id="ra", outputs={"table": "t", "memo": "A"})
        b = SimpleNamespace(id="rb", outputs={"table": "t", "memo": "B"})
        seen.update(evaluators[0]([a, b], None))
        return SimpleNamespace(experiment_name="memo-pairwise-1")

    monkeypatch.setattr(langsmith.evaluation, "evaluate_comparative", fake_comparative)
    assert me.run_pairwise_experiment(memo_cfg, None, "e1", "e2") == "memo-pairwise-1"
    assert seen["scores"] == {"ra": 1, "rb": 0}
