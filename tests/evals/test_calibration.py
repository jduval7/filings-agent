"""Judge calibration against human labels: kappa, false passes, label preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from filings_agent.config import AppConfig, ModelSpec
from filings_agent.evals import calibration as cal
from filings_agent.llm.cache import StepCache
from filings_agent.steps.judge import (
    CriterionVerdict,
    JudgeOutput,
    LLMJudge,
)
from tests.support.fakes import ScriptedChat

SPEC = ModelSpec(provider="bedrock", id="judge-x", price=(1.0, 5.0))


def _returns(value: Any) -> Any:
    def fn(*_: Any, **__: Any) -> Any:
        return value

    return fn


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


def test_cohen_kappa_known_values() -> None:
    assert cal.cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cal.cohen_kappa([True, True, False, False], [True, False, True, False]) == 0.0
    # A judge that always says "pass" agrees 80% of the time here but has no skill: kappa 0.
    human = [True] * 8 + [False] * 2
    assert cal.cohen_kappa(human, [True] * 10) == 0.0
    assert cal.cohen_kappa([], []) == 0.0


def test_agreement_flags_false_passes_as_untrustworthy() -> None:
    pairs = [(True, True)] * 6 + [(False, True)] * 2  # judge passed 2 that the human failed
    a = cal.agreement_for("memo.faithful", pairs)
    assert a.false_pass == 2 and a.false_fail == 0 and not a.trusted
    good = cal.agreement_for("x", [(True, True)] * 4 + [(False, False)] * 4)
    assert good.kappa == 1.0 and good.trusted
    too_few = cal.agreement_for("x", [(True, True), (False, False)])
    assert not too_few.trusted  # under MIN_LABELS: no verdict either way


def test_false_pass_rate_is_per_human_fail() -> None:
    """Review finding: false passes were divided by ALL labels, so 1 miss in 2 human fails on
    20 items read as 5% (trusted). It's 50%: the judge passed half of what you failed."""
    pairs = [(True, True)] * 18 + [(False, True), (False, False)]
    a = cal.agreement_for("x", pairs)
    assert a.human_fails == 2 and a.false_pass_rate == 0.5 and not a.trusted


def test_all_pass_labels_never_make_a_judge_trusted() -> None:
    """With no human fails the judge was never tested on catching one (kappa reads 1.0)."""
    a = cal.agreement_for("x", [(True, True)] * 10)
    assert a.kappa == 1.0 and a.human_fails == 0 and not a.trusted


def _items(cfg: AppConfig) -> list[cal.CalibrationItem]:
    items = [
        cal.CalibrationItem(
            item_id=f"rec-{i}",
            rubric="reconcile_explanation",
            source_model="m",
            material=f"mat{i}",
            output=f"out{i}",
            context={},
        )
        for i in range(8)
    ] + [
        cal.CalibrationItem(
            item_id="own",
            rubric="reconcile_explanation",
            source_model="judge-x",
            material="m",
            output="o",
            context={},
        )
    ]
    (cfg.data_dir / "calibration").mkdir(parents=True)
    (cfg.data_dir / "calibration" / "items.jsonl").write_text(
        "".join(i.model_dump_json() + "\n" for i in items)
    )
    return items


def test_score_compares_judge_to_human_and_skips_the_judges_own_outputs(
    cfg: AppConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _items(cfg)
    labels = {
        f"rec-{i}": {"supported": i % 2 == 0, "specific": True, "consistent": None}
        for i in range(8)
    } | {"own": {"supported": True}}
    path = tmp_path / "labels.yaml"
    path.write_text(yaml.safe_dump(labels))
    judge, _ = _judge([_verdicts(supported=True, specific=True, consistent=True)], tmp_path / "c")
    monkeypatch.setattr(cal, "make_judge", _returns(judge))
    result = cal.score(cfg, labels_path=path)
    by = {c["criterion"]: c for c in result["criteria"]}
    assert result["items_graded"] == 8  # "own" skipped: the judge never grades itself
    assert by["reconcile_explanation.supported"]["false_pass"] == 4  # judge always passes
    assert not by["reconcile_explanation.supported"]["trusted"]
    assert "reconcile_explanation.consistent" not in by  # null labels are skipped
    assert not result["trusted"] and len(result["disagreements"]) == 4
    md = cal.report_markdown(result)
    assert "NOT TRUSTED" in md and "Disagreements" in md
    assert cal.save(cfg, result).exists()


def test_prepare_keeps_existing_labels(
    cfg: AppConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    item = cal.CalibrationItem(
        item_id="rec-a",
        rubric="reconcile_explanation",
        source_model="m",
        material="m",
        output="o",
        context={"case": "X"},
    )
    monkeypatch.setattr(cal, "_reconcile_items", _returns([item]))
    monkeypatch.setattr(cal, "_memo_items", _returns([]))
    path = tmp_path / "labels.yaml"
    path.write_text(yaml.safe_dump({"rec-a": {"supported": False}}))
    cal.prepare(cfg, labels_path=path)
    labels = cal.load_labels(path)
    assert labels["rec-a"] == {"supported": False, "specific": None, "consistent": None}
    assert "rec-a" in (cfg.data_dir / "calibration" / "LABEL_ME.md").read_text()
