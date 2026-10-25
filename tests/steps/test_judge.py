"""The LLM judge: complete verdicts, retries, pairwise order, binary rubrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from filings_agent.config import ModelSpec
from filings_agent.llm.cache import StepCache
from filings_agent.steps.judge import (
    MEMO_RUBRIC,
    RECONCILE_RUBRIC,
    CriterionVerdict,
    JudgeOutput,
    LLMJudge,
    PairwiseOutput,
)
from tests.support.fakes import ScriptedChat

SPEC = ModelSpec(provider="bedrock", id="judge-x", price=(1.0, 5.0))


def _judge(script: list[Any], tmp: Path) -> tuple[LLMJudge, ScriptedChat]:
    chat = ScriptedChat(script=script)
    return LLMJudge("judge-x", SPEC, chat, StepCache(tmp)), chat


def _verdicts(**passed: bool) -> JudgeOutput:
    return JudgeOutput(
        verdicts=[
            CriterionVerdict(criterion=c, reason=f"because {c}", passed=p)
            for c, p in passed.items()
        ]
    )


def test_grade_returns_every_criterion_and_caches(tmp_path: Path) -> None:
    judge, chat = _judge([_verdicts(supported=True, specific=False, consistent=True)], tmp_path)
    res = judge.grade(RECONCILE_RUBRIC, "material", "output")
    assert res.passed == {"supported": True, "specific": False, "consistent": True}
    assert res.reasons["specific"] == "because specific" and res.pass_rate == pytest.approx(2 / 3)
    again = judge.grade(RECONCILE_RUBRIC, "material", "output")
    assert chat.calls == 1 and again.usage.cached and again.passed == res.passed


def test_incomplete_verdict_is_a_failed_grading_not_a_partial_pass(tmp_path: Path) -> None:
    judge, _ = _judge([_verdicts(supported=True)], tmp_path)  # 2 criteria missing
    res = judge.grade(RECONCILE_RUBRIC, "m", "o")
    assert res.passed == {} and "missing criteria" in (res.error or "")
    judge2, chat2 = _judge([None], tmp_path / "b")
    assert judge2.grade(MEMO_RUBRIC, "m", "o").error == "no structured output"
    assert chat2.calls == 2  # retried once before giving up


def test_one_schema_failure_is_retried(tmp_path: Path) -> None:
    ok = _verdicts(faithful=True, caveats_correct=True, comparable=True, actionable=False)
    judge, chat = _judge([None, ok], tmp_path)
    res = judge.grade(MEMO_RUBRIC, "m", "o")
    assert res.error is None and res.passed["actionable"] is False and chat.calls == 2


def test_pairwise_counts_a_winner_only_if_it_wins_both_orders(tmp_path: Path) -> None:
    # Order 1 (A=memo1): A wins. Order 2 (A=memo2): B wins -> memo1 wins both ways.
    judge, _ = _judge(
        [PairwiseOutput(reason="r1", winner="A"), PairwiseOutput(reason="r2", winner="B")], tmp_path
    )
    res = judge.compare("table", "memo1", "memo2")
    assert res.winner == "A" and res.consistent
    # Position bias: "A" both times means it just prefers whichever comes first.
    biased, _ = _judge([PairwiseOutput(reason="r", winner="A")], tmp_path / "b")
    res2 = biased.compare("table", "memo1", "memo2")
    assert res2.winner == "tie" and not res2.consistent


def test_rubrics_are_binary_questions_with_unique_names() -> None:
    for rubric in (RECONCILE_RUBRIC, MEMO_RUBRIC):
        names = rubric.names()
        assert len(names) == len(set(names)) and all(
            c.question.endswith("?") or "fail" in c.question for c in rubric.criteria
        )
