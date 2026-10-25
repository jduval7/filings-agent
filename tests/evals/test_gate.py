"""The CI eval gate: absolute bars and regressions against the baseline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from filings_agent.config import AppConfig
from filings_agent.evals import gate as gate_mod
from filings_agent.evals import local
from tests.support.eval_data import returns


def _gate_files(tmp: Path, baseline: dict[str, float] | None) -> tuple[Path, Path]:
    gate_file = tmp / "gate.yaml"
    gate_file.write_text(yaml.safe_dump({
        "tolerance": 0.02,
        "extract": {"f1_min": 0.97, "not_reported_min": 0.9, "schema_error_max": 0.05},
        "reconcile": {"cause_accuracy_min": 0.85, "quote_verified_min": 0.9},
    }))  # fmt: skip
    base = tmp / "baseline.json"
    if baseline is not None:
        base.write_text(json.dumps(baseline))
    return gate_file, base


def _fake_scores(monkeypatch: pytest.MonkeyPatch, f1: float, cause: float) -> None:
    rep = local.RepScore(0, {"f1": f1, "precision": f1, "recall": f1, "abstain_accuracy": 1.0,
                             "schema_error_rate": 0.0}, 0.0, 1.0, 2.0, 0, {})  # fmt: skip
    monkeypatch.setattr(gate_mod, "score_model", returns([rep]))
    monkeypatch.setattr(
        gate_mod, "score_reconciler", returns({"cause_accuracy": cause, "verified_rate": 1.0})
    )


def test_gate_passes_and_writes_baseline(cfg: AppConfig, monkeypatch: pytest.MonkeyPatch,
                                         tmp_path: Path) -> None:  # fmt: skip
    _fake_scores(monkeypatch, f1=0.99, cause=0.95)
    gate_file, base = _gate_files(tmp_path, None)
    ok, checks, values = gate_mod.run_gate(cfg, gate_file, base, update_baseline=True)
    assert ok and all(c.passed for c in checks)
    assert json.loads(base.read_text())["extract.f1"] == 0.99
    assert values["reconcile.cause_accuracy"] == 0.95
    assert checks[0].line().startswith("PASS")


def test_gate_fails_below_minimum(cfg: AppConfig, monkeypatch: pytest.MonkeyPatch,
                                  tmp_path: Path) -> None:  # fmt: skip
    _fake_scores(monkeypatch, f1=0.93, cause=0.95)
    ok, checks, _ = gate_mod.run_gate(cfg, *_gate_files(tmp_path, None))
    assert not ok
    (failed,) = [c for c in checks if not c.passed]
    assert failed.name == "extract.f1" and failed.line().startswith("FAIL")


def test_gate_fails_on_regression_even_above_minimum(
    cfg: AppConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_scores(monkeypatch, f1=0.99, cause=0.90)  # above the 0.85 minimum...
    gate_file, base = _gate_files(tmp_path, {"reconcile.cause_accuracy": 1.0})
    ok, checks, _ = gate_mod.run_gate(cfg, gate_file, base)
    assert not ok  # ...but 0.10 below baseline with 0.02 tolerance
    assert [c.name for c in checks if not c.passed] == ["reconcile.cause_accuracy"]


def test_one_wrong_auto_approval_fails_the_gate(
    cfg: AppConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Review finding: the easy lane (no human looks) wasn't gated. 19/20 right overall still
    blocks if the wrong one was auto-approved."""
    _fake_scores(monkeypatch, f1=0.99, cause=0.95)
    monkeypatch.setattr(gate_mod, "score_reconciler", returns({
        "cause_accuracy": 0.95, "verified_rate": 1.0, "auto_wrong": ["X:diluted_shares:FY1"],
        "auto_accuracy": 0.9,
    }))  # fmt: skip
    ok, checks, _ = gate_mod.run_gate(cfg, *_gate_files(tmp_path, None))
    assert not ok and [c.name for c in checks if not c.passed] == ["reconcile.easy_lane_accuracy"]


def test_the_gate_scores_the_configured_reconciler(
    cfg: AppConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_scores(monkeypatch, f1=0.99, cause=0.95)
    seen: list[tuple[str, str]] = []

    def spy(cfg_: AppConfig, model: str, prompt: str, split: str, reps: int,
            strategy: str = "pipeline") -> dict[str, float]:  # fmt: skip
        seen.append((strategy, prompt))
        return {"cause_accuracy": 0.95, "verified_rate": 1.0}

    monkeypatch.setattr(gate_mod, "score_reconciler", spy)
    agent_cfg = cfg.model_copy(
        update={"reconcile": cfg.reconcile.model_copy(update={"strategy": "agent"})}
    )
    gate_mod.run_gate(agent_cfg, *_gate_files(tmp_path, None))
    assert seen == [("agent", agent_cfg.reconcile.agent_version)]
